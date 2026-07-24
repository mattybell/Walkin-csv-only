#!/usr/bin/env python3
"""
Rebuilds a local, standalone SQLite database from the CSV snapshots in data/,
using the exact same SQLAlchemy models the training pipeline expects
(app/models/models.py). Run this once before using train_redemption_classifier.py
or research_notebook.ipynb outside the original WalkIn repo (e.g. in Google Colab).

This does not require Postgres, Google Places, BrightData, or any other live
service — data/*.csv is a frozen snapshot of the real research dataset
(583 Walnut Creek, CA businesses / 4,091 deals / 25,172 location_popularity
rows) exported once from the original project's database.

Usage:
    python build_local_db.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DATABASE_URL", "sqlite:///./walkin_research.db")

import pandas as pd

from app.db.session import engine, Base
from app.models.models import Business, Deal, LocationPopularity  # noqa: F401 (registers ORM tables)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TABLES = ["businesses", "deals", "location_popularity"]


def main():
    db_path = str(engine.url).replace("sqlite:///", "")
    if engine.url.get_backend_name() == "sqlite" and db_path not in ("", ":memory:") and os.path.exists(db_path):
        os.remove(db_path)  # rebuild fresh every run rather than risk double-inserting

    Base.metadata.create_all(engine)

    for table in TABLES:
        csv_path = os.path.join(DATA_DIR, f"{table}.csv")
        df = pd.read_csv(csv_path)

        # Defends against schema drift between this snapshot and app/models/models.py
        # (the original project has one such drift already: a stale `cooldown_minutes`
        # column present in the live deals table but not in the current ORM model).
        target_cols = list(Base.metadata.tables[table].columns.keys())
        extra = set(df.columns) - set(target_cols)
        if extra:
            print(f"  ({table}: ignoring columns not in the current model: {sorted(extra)})")
        df = df[[c for c in df.columns if c in target_cols]]

        df.to_sql(table, engine, if_exists="append", index=False)
        print(f"Loaded {len(df):,} rows into {table}")

    print(f"\nLocal SQLite database ready at: {engine.url}")
    print("You can now `import train_redemption_classifier as trc` or run the notebook.")


if __name__ == "__main__":
    main()
