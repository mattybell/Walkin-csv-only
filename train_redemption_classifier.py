#!/usr/bin/env python3
"""
Research pipeline: hybrid redemption-probability classifier.

Implements the architecture research_proposal.md actually describes (XGBoost
tabular classifier + Fourier time-series classifier, combined via a
logistic-regression stacking meta-learner, evaluated with a time-based split
via AUC-ROC / ECE / Precision@3) which train_models.py / retrain_hybrid_models.py
do not: those only train count regressors evaluated with MAE/RMSE/R^2.

This script is READ-ONLY against Business/Deal/LocationPopularity - it never
writes to those tables. The "redeemed vs not-redeemed" labels the Redemption
table has no way to represent are produced by a seeded in-memory simulation
(simulate_exposures) instead. Output artifacts (models/*.joblib, reports/*)
are written under distinct paths from the production models, so
app/services/ml_inference.py and the live /forecast endpoints are unaffected.

Usage:
    python train_redemption_classifier.py

Note: this copy lives in a flattened, standalone package (see README.md in this
directory) extracted from the main WalkIn repo so it can run independently —
e.g. cloned into Google Colab — without the rest of the codebase. The only
difference from the original `backend/scripts/train_redemption_classifier.py`
is this path setup block and MODEL_DIR/REPORT_DIR below, adjusted for the
flatter directory layout; the pipeline logic itself is untouched.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import platform
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from app.db.session import SessionLocal
from app.models.models import Business, Deal, LocationPopularity

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
N_EXPOSURES_PER_DEAL = 40
SIMULATION_DAYS = 60
TEST_FRAC = 0.2

MODEL_DIR = Path(__file__).parent / "models" / "research"
REPORT_DIR = Path(__file__).parent / "reports" / "redemption_classifier"

DEAL_TYPES = [
    "STANDARD", "BUY_X_GET_Y", "PERCENT_OFF", "FIXED_PRICE_ITEM",
    "SPEND_X_GET_PERCENT", "SPEND_X_GET_AMOUNT", "CUSTOM_TEXT",
]
CATEGORY_VOCAB = [
    "restaurant", "cafe", "bar", "hair_care", "beauty_salon", "gym", "spa",
    "bakery", "pizza_restaurant", "chinese_restaurant", "japanese_restaurant",
    "thai_restaurant", "indian_restaurant", "mexican_restaurant",
]
PEAK_HOURS = {12, 13, 14, 17, 18, 19}
DISCOUNT_BEARING_TYPES = {"PERCENT_OFF", "SPEND_X_GET_PERCENT", "SPEND_X_GET_AMOUNT", "BUY_X_GET_Y"}

# Coefficients for the seeded exposure-labeling simulation (see simulate_exposures).
# Starting points calibrated toward a 5-25% blended redemption rate; not fit to
# real data (there is none) - sanity-checked against the printed per-deal-type
# empirical rate at runtime.
BETA0 = -2.6
DEAL_TYPE_OFFSET = {
    "STANDARD": -0.10, "BUY_X_GET_Y": 0.15, "PERCENT_OFF": 0.0,
    "FIXED_PRICE_ITEM": 0.10, "SPEND_X_GET_PERCENT": 0.05,
    "SPEND_X_GET_AMOUNT": -0.05, "CUSTOM_TEXT": -0.20,
}
BETA_DISCOUNT = 1.2
BETA_FEATURED = 0.35
BETA_TRAFFIC = 0.6
BETA_PEAK_HOUR = 0.25
BETA_WEEKEND = 0.15
BETA_RATING = 0.2
BETA_REVIEWS = 0.1
NOISE_STD = 0.5

TABULAR_XGB_PARAMS = dict(
    objective="binary:logistic", eval_metric="auc", n_estimators=200,
    max_depth=4, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
    random_state=RANDOM_SEED, n_jobs=-1,
)
BASELINE_XGB_PARAMS = dict(
    objective="binary:logistic", eval_metric="auc", n_estimators=150,
    max_depth=3, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5, random_state=RANDOM_SEED, n_jobs=-1,
)

CATEGORY_FEATURE_COLS = [f"category_{c}" for c in CATEGORY_VOCAB] + ["category_unknown"]
DEAL_TYPE_FEATURE_COLS = [f"deal_type_{t}" for t in DEAL_TYPES]
TABULAR_FEATURES = (
    DEAL_TYPE_FEATURE_COLS
    + ["discount_depth", "has_explicit_discount_depth", "is_featured", "is_progressive",
       "has_loyalty_program", "cooldown_hours"]
    + CATEGORY_FEATURE_COLS
    + ["business_rating", "business_rating_count_log", "deal_prior_redemptions_log"]
)
TEMPORAL_FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "week_sin", "week_cos",
    "is_weekend", "is_peak_hour", "traffic_score_norm", "lagged_traffic_avg_norm",
]
BASELINE_FEATURES = DEAL_TYPE_FEATURE_COLS + CATEGORY_FEATURE_COLS


# ---------------------------------------------------------------------------
# 1. Load data (read-only)
# ---------------------------------------------------------------------------
def load_source_data(db):
    businesses = db.query(Business).all()
    deals = db.query(Deal).filter(Deal.active.is_(True)).all()
    popularity = db.query(LocationPopularity).all()
    if not businesses:
        raise RuntimeError("No businesses found. Run scripts/import_real_businesses.py first.")
    if not deals:
        raise RuntimeError("No deals found. Run scripts/seed_walnut_creek_deals.py first.")
    return businesses, deals, popularity


# ---------------------------------------------------------------------------
# 2. Foot-traffic lookup with graceful 3-tier fallback
# ---------------------------------------------------------------------------
class PopularityLookup:
    """exact (business, hour, dow) -> (business, hour) -> per-business mean -> 50.0

    The mock BrightData response varies by hour only (not day-of-week), and a
    sync run stamps every row with that run's single day-of-week - so an exact
    (business, hour, dow) join mostly misses. This fallback chain guarantees
    the traffic feature is never trivially null.
    """

    def __init__(self, popularity_rows):
        exact = defaultdict(list)
        by_hour = defaultdict(list)
        by_business = defaultdict(list)
        for row in popularity_rows:
            exact[(row.business_id, row.hour_of_day, row.day_of_week)].append(row.popularity_score)
            by_hour[(row.business_id, row.hour_of_day)].append(row.popularity_score)
            by_business[row.business_id].append(row.popularity_score)

        self._exact = {k: float(np.mean(v)) for k, v in exact.items()}
        self._by_hour = {k: float(np.mean(v)) for k, v in by_hour.items()}
        self._by_business = {k: float(np.mean(v)) for k, v in by_business.items()}

    def get(self, business_id, hour, dow):
        if (business_id, hour, dow) in self._exact:
            return self._exact[(business_id, hour, dow)]
        if (business_id, hour) in self._by_hour:
            return self._by_hour[(business_id, hour)]
        if business_id in self._by_business:
            return self._by_business[business_id]
        return 50.0


# ---------------------------------------------------------------------------
# 3. Seeded exposure simulation - the labeled training signal
# ---------------------------------------------------------------------------
def discount_depth(deal):
    if deal.deal_type in ("PERCENT_OFF", "SPEND_X_GET_PERCENT"):
        return (deal.percent_off or 0) / 100.0
    if deal.deal_type == "SPEND_X_GET_AMOUNT":
        min_spend = deal.min_spend_cents or 1
        return float(np.clip((deal.amount_off_cents or 0) / min_spend, 0, 1))
    if deal.deal_type == "BUY_X_GET_Y":
        buy_qty = deal.buy_qty or 1
        get_qty = deal.get_qty or 0
        total = buy_qty + get_qty
        return get_qty / total if total else 0.0
    return 0.0


def simulate_exposures(businesses, deals, popularity_lookup, rng):
    """The "seeded simulation reflecting realistic redemption rate
    distributions across deal types" research_proposal.md line 10 describes.
    Lives entirely in memory/in the returned DataFrame - never written to the
    real `redemptions` table, which has no negative-class concept at all.
    """
    businesses_by_id = {b.id: b for b in businesses}
    hour_weights = np.array([3.0 if h in PEAK_HOURS else (1.0 if 6 <= h <= 21 else 0.3) for h in range(24)])
    hour_weights = hour_weights / hour_weights.sum()

    rows = []
    exposure_id = 0
    today = datetime.utcnow().date()

    for deal in deals:
        business = businesses_by_id.get(deal.business_id)
        if business is None:
            continue
        depth = discount_depth(deal)
        is_featured = deal.featured_at is not None
        rating = business.rating if business.rating is not None else 4.0
        rating_count = business.rating_count or 0

        day_offsets = rng.integers(0, SIMULATION_DAYS, size=N_EXPOSURES_PER_DEAL)
        hours = rng.choice(24, size=N_EXPOSURES_PER_DEAL, p=hour_weights)
        minutes = rng.integers(0, 60, size=N_EXPOSURES_PER_DEAL)
        noise = rng.normal(0, NOISE_STD, size=N_EXPOSURES_PER_DEAL)

        for i in range(N_EXPOSURES_PER_DEAL):
            exposure_date = today - timedelta(days=int(day_offsets[i]))
            hour = int(hours[i])
            exposure_ts = datetime(exposure_date.year, exposure_date.month, exposure_date.day,
                                    hour, int(minutes[i]))
            dow = exposure_ts.weekday()
            week_of_year = exposure_ts.isocalendar()[1]
            is_weekend = dow >= 5
            is_peak = hour in PEAK_HOURS
            traffic_score = popularity_lookup.get(business.id, hour, dow)

            logit = (
                BETA0
                + DEAL_TYPE_OFFSET.get(deal.deal_type, 0.0)
                + BETA_DISCOUNT * depth
                + BETA_FEATURED * (1.0 if is_featured else 0.0)
                + BETA_TRAFFIC * (traffic_score / 150.0)
                + BETA_PEAK_HOUR * (1.0 if is_peak else 0.0)
                + BETA_WEEKEND * (1.0 if is_weekend else 0.0)
                + BETA_RATING * (rating - 4.0)
                + BETA_REVIEWS * (np.log1p(rating_count) / 10.0)
                + noise[i]
            )
            p = 1.0 / (1.0 + np.exp(-logit))
            redeemed = int(rng.binomial(1, p))

            rows.append({
                "exposure_id": exposure_id,
                "deal_id": deal.id,
                "business_id": business.id,
                "deal_type": deal.deal_type,
                "exposure_timestamp": exposure_ts,
                "hour_of_day": hour,
                "day_of_week": dow,
                "week_of_year": week_of_year,
                "redeemed": redeemed,
                "true_probability": p,
            })
            exposure_id += 1

    df = pd.DataFrame(rows)
    print(f"\nSimulated {len(df)} exposures across {len(deals)} deals")
    print("Empirical redemption rate by deal type:")
    print(df.groupby("deal_type")["redeemed"].mean().round(3).to_string())
    print(f"Overall redemption rate: {df['redeemed'].mean():.3f}")
    return df


# ---------------------------------------------------------------------------
# 4. Feature engineering
# ---------------------------------------------------------------------------
def build_deal_feature_frame(deals):
    rows = []
    for deal in deals:
        rows.append({
            "deal_id": deal.id,
            "discount_depth": discount_depth(deal),
            "has_explicit_discount_depth": 1 if deal.deal_type in DISCOUNT_BEARING_TYPES else 0,
            "is_featured": 1 if deal.featured_at is not None else 0,
            "is_progressive": 1 if deal.is_progressive else 0,
            "has_loyalty_program": 1 if deal.loyalty_redemptions_required is not None else 0,
            "cooldown_hours": deal.cooldown_hours or 1,
        })
    return pd.DataFrame(rows).set_index("deal_id")


def build_business_feature_frame(businesses):
    rows = []
    for b in businesses:
        rows.append({
            "business_id": b.id,
            "business_rating": b.rating if b.rating is not None else 4.0,
            "business_rating_count_log": float(np.log1p(b.rating_count or 0)),
            "category": b.category if b.category in CATEGORY_VOCAB else None,
        })
    return pd.DataFrame(rows).set_index("business_id")


def engineer_features(exposures_df, businesses, deals, popularity_lookup):
    df = exposures_df.sort_values("exposure_timestamp").reset_index(drop=True)

    for t in DEAL_TYPES:
        df[f"deal_type_{t}"] = (df["deal_type"] == t).astype(int)

    deal_features = build_deal_feature_frame(deals)
    business_features = build_business_feature_frame(businesses)
    df = df.join(deal_features, on="deal_id").join(business_features, on="business_id")

    for c in CATEGORY_VOCAB:
        df[f"category_{c}"] = (df["category"] == c).astype(int)
    df["category_unknown"] = df["category"].isna().astype(int)

    # Causally-lagged prior-redemption count per deal (past exposures of the
    # SAME deal only - df is sorted by exposure_timestamp above, so a
    # groupby-cumsum respects chronological order within each deal).
    cum_redeemed = df.groupby("deal_id")["redeemed"].cumsum() - df["redeemed"]
    df["deal_prior_redemptions_log"] = np.log1p(cum_redeemed)

    # Fourier / cyclical temporal encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_peak_hour"] = df["hour_of_day"].isin(PEAK_HOURS).astype(int)

    # Foot-traffic features via the fallback lookup (this is where BrightData
    # signal actually reaches the model - the bug in retrain_hybrid_models.py
    # computes this and then discards it before training).
    traffic_norm = [
        popularity_lookup.get(bid, hour, dow) / 150.0
        for bid, hour, dow in zip(df["business_id"], df["hour_of_day"], df["day_of_week"])
    ]
    lagged_traffic = [
        float(np.mean([popularity_lookup.get(bid, (hour + d) % 24, dow) for d in (-1, 0, 1)])) / 150.0
        for bid, hour, dow in zip(df["business_id"], df["hour_of_day"], df["day_of_week"])
    ]
    df["traffic_score_norm"] = np.clip(traffic_norm, 0, None)
    df["lagged_traffic_avg_norm"] = np.clip(lagged_traffic, 0, None)

    feature_cols = TABULAR_FEATURES + TEMPORAL_FEATURES
    X = df[feature_cols].astype(float)
    y = df["redeemed"].astype(int)
    meta = df[["exposure_timestamp", "deal_id", "business_id", "true_probability"]].copy()
    return X, y, meta


# ---------------------------------------------------------------------------
# 5. Time-based split
# ---------------------------------------------------------------------------
def time_based_split(X, y, meta, test_frac=TEST_FRAC):
    order = meta["exposure_timestamp"].argsort().to_numpy()
    X, y, meta = X.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True), meta.iloc[order].reset_index(drop=True)
    split_idx = int(len(X) * (1 - test_frac))
    return (
        X.iloc[:split_idx].reset_index(drop=True), X.iloc[split_idx:].reset_index(drop=True),
        y.iloc[:split_idx].reset_index(drop=True), y.iloc[split_idx:].reset_index(drop=True),
        meta.iloc[:split_idx].reset_index(drop=True), meta.iloc[split_idx:].reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# 6. Base learners + logistic-regression stacking meta-learner
# ---------------------------------------------------------------------------
def make_tabular_model(scale_pos_weight):
    return XGBClassifier(scale_pos_weight=scale_pos_weight, **TABULAR_XGB_PARAMS)


def make_temporal_model():
    # A linear model over a Fourier/sin-cos basis is the Prophet approach
    # (Taylor & Letham 2018) - pairing it with the cyclical features gives the
    # "hybrid" architecture real substance rather than two identical tree models.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=RANDOM_SEED)),
    ])


def make_baseline_model(scale_pos_weight):
    return XGBClassifier(scale_pos_weight=scale_pos_weight, **BASELINE_XGB_PARAMS)


def oof_predictions(model_factory, X_train, y_train, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    return cross_val_predict(model_factory(), X_train, y_train, cv=cv, method="predict_proba")[:, 1]


def train_hybrid_pipeline(X_train, y_train):
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    # Out-of-fold predictions (train split only) feed the meta-learner so it
    # never sees a base learner's prediction on rows that trained it.
    oof_tab = oof_predictions(lambda: make_tabular_model(scale_pos_weight), X_train[TABULAR_FEATURES], y_train)
    oof_temp = oof_predictions(make_temporal_model, X_train[TEMPORAL_FEATURES], y_train)

    tabular_model = make_tabular_model(scale_pos_weight).fit(X_train[TABULAR_FEATURES], y_train)
    temporal_model = make_temporal_model().fit(X_train[TEMPORAL_FEATURES], y_train)

    # LogisticRegression's MLE objective IS binary cross-entropy minimization.
    meta_model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    meta_model.fit(np.column_stack([oof_tab, oof_temp]), y_train)

    return {
        "tabular": tabular_model,
        "temporal": temporal_model,
        "meta": meta_model,
        "oof_tabular_auc": roc_auc_score(y_train, oof_tab),
        "oof_temporal_auc": roc_auc_score(y_train, oof_temp),
        "scale_pos_weight": scale_pos_weight,
    }


def predict_stacked(models, X):
    p_tab = models["tabular"].predict_proba(X[TABULAR_FEATURES])[:, 1]
    p_temp = models["temporal"].predict_proba(X[TEMPORAL_FEATURES])[:, 1]
    p_meta = models["meta"].predict_proba(np.column_stack([p_tab, p_temp]))[:, 1]
    return p_meta, p_tab, p_temp


# ---------------------------------------------------------------------------
# 7. Metrics: AUC-ROC (sklearn), ECE (custom - sklearn has no scalar ECE),
#    Precision@3 (custom, ranking real per-business deals)
# ---------------------------------------------------------------------------
def compute_ece(y_true, y_prob, n_bins=10):
    """Equal-width binned Expected Calibration Error (Guo et al., "On
    Calibration of Modern Neural Networks," ICML 2017)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    bins = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & ((y_prob <= hi) if i == n_bins - 1 else (y_prob < hi))
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(y_prob[mask].mean())
        accuracy = float(y_true[mask].mean())
        ece += (count / n) * abs(accuracy - confidence)
        bins.append({"bin_lo": float(lo), "bin_hi": float(hi), "count": count,
                      "confidence": confidence, "accuracy": accuracy})
    return float(ece), bins


def precision_at_3(meta_test, y_test, y_prob):
    """Rank each business's real distinct deals (>=3 guaranteed by
    seed_walnut_creek_deals.py) by predicted vs. actual test-set redemption
    rate; measure top-3 overlap; average across qualifying businesses."""
    df = meta_test.copy()
    df["redeemed"] = np.asarray(y_test)
    df["predicted_proba"] = np.asarray(y_prob)

    scores = []
    for business_id, group in df.groupby("business_id"):
        deal_ids = group["deal_id"].unique()
        if len(deal_ids) < 3:
            continue
        true_rate = group.groupby("deal_id")["redeemed"].mean()
        pred_rate = group.groupby("deal_id")["predicted_proba"].mean()
        true_top3 = set(true_rate.sort_values(ascending=False).head(3).index)
        pred_top3 = set(pred_rate.sort_values(ascending=False).head(3).index)
        k = min(3, len(deal_ids))
        scores.append(len(true_top3 & pred_top3) / k)

    if not scores:
        return 0.0, 0, []
    return float(np.mean(scores)), len(scores), scores


# ---------------------------------------------------------------------------
# 8. Plots
# ---------------------------------------------------------------------------
def plot_roc(y_test, p_hybrid, auc_hybrid, p_baseline, auc_baseline, save_path=None):
    fpr_h, tpr_h, _ = roc_curve(y_test, p_hybrid)
    fpr_b, tpr_b, _ = roc_curve(y_test, p_baseline)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr_h, tpr_h, label=f"Hybrid (AUC={auc_hybrid:.3f})")
    plt.plot(fpr_b, tpr_b, label=f"Baseline (AUC={auc_baseline:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: Hybrid vs. Baseline Redemption Classifier")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_calibration(y_test, p_hybrid, n_bins, save_path=None):
    frac_pos, mean_pred = calibration_curve(y_test, p_hybrid, n_bins=n_bins, strategy="uniform")
    plt.figure(figsize=(6, 6))
    plt.plot(mean_pred, frac_pos, marker="o", label="Hybrid model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Redemption Rate")
    plt.title("Calibration (Reliability) Diagram")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_feature_importance(model, feature_names, save_path=None, top_n=15):
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in order][::-1]
    vals = [importances[i] for i in order][::-1]
    plt.figure(figsize=(7, 6))
    plt.barh(names, vals, color="#6C5CE7")
    plt.xlabel("XGBoost Feature Importance (gain-weighted)")
    plt.title("Tabular Base Learner - Top Feature Importances")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_auc_comparison(auc_dict, save_path=None):
    labels = list(auc_dict.keys())
    vals = list(auc_dict.values())
    colors = ["#B0AEDB" if l != "Hybrid (stacked)" else "#6C5CE7" for l in labels]
    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, vals, color=colors)
    plt.axhline(0.5, color="gray", linestyle="--", label="Chance (0.5)")
    plt.ylabel("Test AUC-ROC")
    plt.title("AUC-ROC by Model Variant")
    plt.ylim(0.45, max(vals) + 0.08)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)
    plt.legend()
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_redemption_rate_by_type(exposures_df, save_path=None):
    rates = exposures_df.groupby("deal_type")["redeemed"].mean().sort_values(ascending=False)
    overall = exposures_df["redeemed"].mean()
    plt.figure(figsize=(8, 5))
    bars = plt.bar(rates.index, rates.values, color="#6C5CE7")
    plt.axhline(overall, color="gray", linestyle="--", label=f"Overall ({overall:.1%})")
    plt.ylabel("Simulated Redemption Rate")
    plt.title("Redemption Rate by Deal Type (n=46,880 exposures)")
    plt.xticks(rotation=30, ha="right")
    for b, v in zip(bars, rates.values):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.1%}", ha="center", fontsize=9)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_prediction_distribution(y_test, p_hybrid_test, save_path=None):
    y_test = np.asarray(y_test)
    plt.figure(figsize=(7, 5))
    plt.hist(p_hybrid_test[y_test == 0], bins=30, alpha=0.6, label="Not redeemed", color="#D14B4B", density=True)
    plt.hist(p_hybrid_test[y_test == 1], bins=30, alpha=0.6, label="Redeemed", color="#1EA86B", density=True)
    plt.xlabel("Predicted Probability (hybrid model)")
    plt.ylabel("Density")
    plt.title("Predicted Probability Distribution by True Outcome")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_traffic_heatmap(popularity, save_path=None):
    rows = [{"hour": p.hour_of_day, "dow": p.day_of_week, "score": p.popularity_score} for p in popularity]
    if not rows:
        return
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="dow", columns="hour", values="score", aggfunc="mean")
    plt.figure(figsize=(9, 4))
    im = plt.imshow(pivot.values, aspect="auto", cmap="viridis")
    plt.colorbar(im, label="Mean popularity score")
    plt.yticks(range(len(pivot.index)), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][:len(pivot.index)])
    plt.xticks(range(0, 24, 2), range(0, 24, 2))
    plt.xlabel("Hour of Day")
    plt.title("BrightData-Sourced Foot Traffic: Hour x Day-of-Week")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_category_distribution(businesses, save_path=None):
    cats = [b.category if b.category in CATEGORY_VOCAB else "unknown" for b in businesses]
    counts = pd.Series(cats).value_counts()
    plt.figure(figsize=(8, 5))
    plt.barh(counts.index[::-1], counts.values[::-1], color="#6C5CE7")
    plt.xlabel("Number of Businesses")
    plt.title(f"Business Category Distribution (n={len(businesses)})")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_precision_at_3_distribution(scores, save_path=None):
    plt.figure(figsize=(7, 5))
    plt.hist(scores, bins=[0, 0.34, 0.67, 1.01], rwidth=0.85, color="#6C5CE7")
    plt.xticks([1/6, 0.5, 5/6], ["0/3 correct", "1-2/3 correct", "3/3 correct"])
    plt.ylabel("Number of Businesses")
    plt.title(f"Precision@3 Distribution Across {len(scores)} Businesses")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 9. Reproducibility snapshot
# ---------------------------------------------------------------------------
def save_reproducibility_snapshot():
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                 capture_output=True, text=True, check=True).stdout
    except Exception as e:
        freeze = f"pip freeze failed: {e}"
    (REPORT_DIR / "pip_freeze.txt").write_text(freeze)


def main():
    print("=" * 70)
    print("Redemption Classifier Research Pipeline")
    print("=" * 70)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    db = SessionLocal()
    try:
        businesses, deals, popularity = load_source_data(db)
        print(f"Loaded {len(businesses)} businesses, {len(deals)} active deals, "
              f"{len(popularity)} location_popularity rows")

        # Derived from LocationPopularity.source rather than hardcoded: "real" once
        # any row was sourced from BrightData's actual Datasets API (as opposed to
        # the "brightdata_mock" fallback used when no API key is configured).
        sources_present = {p.source for p in popularity}
        if sources_present == {"brightdata"}:
            brightdata_mode = "real"
        elif "brightdata" in sources_present:
            brightdata_mode = "mixed"
        elif sources_present:
            brightdata_mode = "mock"
        else:
            brightdata_mode = "none"
        businesses_with_popularity = len({p.business_id for p in popularity})
        print(f"BrightData coverage: {businesses_with_popularity}/{len(businesses)} businesses "
              f"have popularity data (mode: {brightdata_mode})")
        popularity_lookup = PopularityLookup(popularity)

        exposures_df = simulate_exposures(businesses, deals, popularity_lookup, rng)
        X, y, meta = engineer_features(exposures_df, businesses, deals, popularity_lookup)

        X_train, X_test, y_train, y_test, meta_train, meta_test = time_based_split(X, y, meta)
        print(f"\nTrain: {len(X_train)} exposures ({meta_train['exposure_timestamp'].min()} to "
              f"{meta_train['exposure_timestamp'].max()})")
        print(f"Test:  {len(X_test)} exposures ({meta_test['exposure_timestamp'].min()} to "
              f"{meta_test['exposure_timestamp'].max()})")

        print("\nTraining hybrid stacked classifier (tabular XGB + temporal LogReg + meta LogReg)...")
        hybrid_models = train_hybrid_pipeline(X_train, y_train)
        p_hybrid_test, p_tab_test, p_temp_test = predict_stacked(hybrid_models, X_test)

        print("Training baseline classifier (deal-type + category only, no foot traffic)...")
        baseline_model = make_baseline_model(hybrid_models["scale_pos_weight"]).fit(
            X_train[BASELINE_FEATURES], y_train)
        p_baseline_test = baseline_model.predict_proba(X_test[BASELINE_FEATURES])[:, 1]

        auc_hybrid = roc_auc_score(y_test, p_hybrid_test)
        auc_baseline = roc_auc_score(y_test, p_baseline_test)
        ece_hybrid, ece_bins = compute_ece(y_test.to_numpy(), p_hybrid_test)
        ece_baseline, _ = compute_ece(y_test.to_numpy(), p_baseline_test)
        prec3, n_biz_eval, prec3_scores = precision_at_3(meta_test, y_test, p_hybrid_test)
        logloss_hybrid = log_loss(y_test, p_hybrid_test)
        logloss_baseline = log_loss(y_test, p_baseline_test)

        print(f"\n{'=' * 70}\nRESULTS\n{'=' * 70}")
        print(f"Hybrid   AUC-ROC: {auc_hybrid:.4f}   ECE: {ece_hybrid:.4f}   LogLoss: {logloss_hybrid:.4f}")
        print(f"Baseline AUC-ROC: {auc_baseline:.4f}   ECE: {ece_baseline:.4f}   LogLoss: {logloss_baseline:.4f}")
        print(f"AUC delta (hybrid - baseline): {auc_hybrid - auc_baseline:+.4f}")
        print(f"Precision@3: {prec3:.4f} (evaluated over {n_biz_eval} businesses)")
        print(f"Tabular base learner:  OOF AUC {hybrid_models['oof_tabular_auc']:.4f}, "
              f"test AUC {roc_auc_score(y_test, p_tab_test):.4f}")
        print(f"Temporal base learner: OOF AUC {hybrid_models['oof_temporal_auc']:.4f}, "
              f"test AUC {roc_auc_score(y_test, p_temp_test):.4f}")

        plot_roc(y_test, p_hybrid_test, auc_hybrid, p_baseline_test, auc_baseline, REPORT_DIR / "roc_curve.png")
        plot_calibration(y_test, p_hybrid_test, 10, REPORT_DIR / "calibration_diagram.png")
        plot_feature_importance(hybrid_models["tabular"], TABULAR_FEATURES, REPORT_DIR / "feature_importance.png")
        plot_auc_comparison({
            "Baseline\n(type+category)": auc_baseline,
            "Tabular\nlearner alone": roc_auc_score(y_test, p_tab_test),
            "Temporal\nlearner alone": roc_auc_score(y_test, p_temp_test),
            "Hybrid (stacked)": auc_hybrid,
        }, REPORT_DIR / "auc_comparison.png")
        plot_redemption_rate_by_type(exposures_df, REPORT_DIR / "redemption_rate_by_type.png")
        plot_prediction_distribution(y_test, p_hybrid_test, REPORT_DIR / "prediction_distribution.png")
        plot_traffic_heatmap(popularity, REPORT_DIR / "traffic_heatmap.png")
        plot_category_distribution(businesses, REPORT_DIR / "category_distribution.png")
        plot_precision_at_3_distribution(prec3_scores, REPORT_DIR / "precision_at_3_distribution.png")
        print(f"\nSaved 9 plots to {REPORT_DIR}")

        joblib.dump(hybrid_models["tabular"], MODEL_DIR / "redemption_classifier_tabular_xgb_v1.joblib")
        joblib.dump(hybrid_models["temporal"], MODEL_DIR / "redemption_classifier_temporal_logreg_v1.joblib")
        joblib.dump(hybrid_models["meta"], MODEL_DIR / "redemption_classifier_meta_v1.joblib")
        joblib.dump(baseline_model, MODEL_DIR / "redemption_classifier_baseline_v1.joblib")
        print(f"Saved model artifacts to {MODEL_DIR}")

        metrics = {
            "run_metadata": {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "random_seed": RANDOM_SEED,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "n_businesses": len(businesses),
                "n_deals": len(deals),
                "n_exposures_total": len(exposures_df),
                "n_exposures_train": len(X_train),
                "n_exposures_test": len(X_test),
                "overall_redemption_rate": round(float(exposures_df["redeemed"].mean()), 4),
                "redemption_rate_by_deal_type": exposures_df.groupby("deal_type")["redeemed"].mean().round(4).to_dict(),
                "brightdata_mode": brightdata_mode,
                "n_location_popularity_rows": len(popularity),
                "businesses_with_popularity_data": businesses_with_popularity,
            },
            "hybrid_model": {
                "auc_roc": float(auc_hybrid),
                "log_loss": float(logloss_hybrid),
                "ece": float(ece_hybrid),
                "ece_bins": ece_bins,
                "precision_at_3": float(prec3),
                "precision_at_3_n_businesses_evaluated": n_biz_eval,
            },
            "baseline_model": {
                "auc_roc": float(auc_baseline),
                "log_loss": float(logloss_baseline),
                "ece": float(ece_baseline),
            },
            "comparison": {
                "auc_delta_hybrid_minus_baseline": float(auc_hybrid - auc_baseline),
            },
            "base_learners": {
                "tabular_xgb": {
                    "auc_roc_oof_train": float(hybrid_models["oof_tabular_auc"]),
                    "auc_roc_test": float(roc_auc_score(y_test, p_tab_test)),
                },
                "temporal_logreg": {
                    "auc_roc_oof_train": float(hybrid_models["oof_temporal_auc"]),
                    "auc_roc_test": float(roc_auc_score(y_test, p_temp_test)),
                },
            },
            "hyperparameters": {
                "tabular_xgb": {**TABULAR_XGB_PARAMS, "scale_pos_weight": float(hybrid_models["scale_pos_weight"])},
                "baseline_xgb": {**BASELINE_XGB_PARAMS, "scale_pos_weight": float(hybrid_models["scale_pos_weight"])},
                "temporal_logreg": {"C": 1.0, "class_weight": "balanced", "max_iter": 2000},
                "meta_logreg": {"max_iter": 1000},
                "simulation": {
                    "n_exposures_per_deal": N_EXPOSURES_PER_DEAL,
                    "simulation_days": SIMULATION_DAYS,
                    "beta0": BETA0,
                    "deal_type_offset": DEAL_TYPE_OFFSET,
                    "beta_discount": BETA_DISCOUNT,
                    "beta_featured": BETA_FEATURED,
                    "beta_traffic": BETA_TRAFFIC,
                    "beta_peak_hour": BETA_PEAK_HOUR,
                    "beta_weekend": BETA_WEEKEND,
                    "beta_rating": BETA_RATING,
                    "beta_reviews": BETA_REVIEWS,
                    "noise_std": NOISE_STD,
                },
            },
            "feature_columns": {
                "tabular_features": TABULAR_FEATURES,
                "temporal_features": TEMPORAL_FEATURES,
                "baseline_features": BASELINE_FEATURES,
            },
        }
        (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
        print(f"Saved metrics to {REPORT_DIR / 'metrics.json'}")

        save_reproducibility_snapshot()
        print(f"Saved pip freeze snapshot to {REPORT_DIR / 'pip_freeze.txt'}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
