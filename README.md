# WalkIn Redemption Classifier — Standalone Research Package

This folder is a self-contained extract of the WalkIn project's research pipeline —
enough to run `research_notebook.ipynb` end-to-end in **Google Colab** (or any machine)
without access to the rest of the WalkIn codebase, without a Postgres database, and
without any API keys.

## What's in here

| Path | What it is |
|---|---|
| `research_notebook.ipynb` | The full research notebook — same analysis, figures, and results as the original, plus one new setup cell at the top |
| `train_redemption_classifier.py` | The training/evaluation pipeline itself, unmodified except for two path lines (see below) |
| `build_local_db.py` | Rebuilds a local SQLite database from the CSV snapshot in `data/` |
| `data/*.csv` | A frozen snapshot of the real research dataset: 583 Walnut Creek, CA businesses, 4,091 seeded deals, 25,172 `location_popularity` rows |
| `app/` | The minimal slice of the WalkIn backend needed for the ORM models to import — just `config.py`, `session.py`, and `models.py`. No routers, no auth, no billing, nothing else from the app |
| `requirements.txt` | Pinned (minimum-version) dependencies |

## How this was produced

The original notebook queries a live Postgres database via SQLAlchemy. Nothing in the
training pipeline touches any other part of the app, any external API (Google Places,
BrightData, Stripe, Twilio), or the filesystem beyond three read-only queries against the
`businesses`, `deals`, and `location_popularity` tables. So instead of requiring Postgres,
this package ships a frozen CSV snapshot of those three tables (exported once from the
real dataset, ordered by `id` for determinism) and a small script (`build_local_db.py`)
that recreates them in a local SQLite file using the *exact same* SQLAlchemy models. The
pipeline code then runs completely unaware it's talking to SQLite instead of Postgres.

The only lines changed from the original files:
- `train_redemption_classifier.py`: the `sys.path` setup and `MODEL_DIR`/`REPORT_DIR`
  computation were adjusted for this package's flatter layout (script + `app/` at the same
  level, rather than nested under `backend/`). The actual pipeline logic — feature
  engineering, the exposure simulation, model training, evaluation, every plotting
  function — is byte-for-byte identical to the original.
- `research_notebook.ipynb`: one new setup cell at the top (installs dependencies, builds
  the local database), and the first code cell's `sys.path` lines adjusted the same way.

## Usage in Google Colab

1. Push this folder to a GitHub repo (just this folder's contents, not the whole WalkIn monorepo).
2. In Colab:
   ```
   !git clone https://github.com/<your-username>/<your-repo>.git
   %cd <your-repo>
   ```
3. Open `research_notebook.ipynb` in Colab (or run `jupyter nbconvert --execute` on it),
   and run the first code cell — it installs dependencies and builds
   `walkin_research.db` automatically. Every cell after that runs exactly like the
   original.

## Usage locally

```
cd research_colab_package
pip install -r requirements.txt
python build_local_db.py
jupyter notebook research_notebook.ipynb
```

## A note on exact reproducibility

The main `research_report.md`/`.pdf` reports a test AUC-ROC of **0.5974** for the hybrid
model. Running this standalone package reproduces the same dataset, the same seeded
simulation (`RANDOM_SEED = 42`), and the same qualitative results — the hybrid model
beating the baseline and both individual base learners, calibration roughly 40x tighter
than the baseline — but the *exact* AUC figure can differ by a few thousandths (observed:
~0.593–0.597 depending on row order) from the report. This is because the exposure
simulation draws from a seeded RNG *in the order businesses and deals are iterated*, and
that iteration order isn't perfectly preserved by a CSV export/reload round-trip the way
it would be by querying the original live database directly. The architecture comparison,
calibration advantage, and every qualitative conclusion in the report are robust to this;
only the last one or two decimal places of the headline AUC number move.
