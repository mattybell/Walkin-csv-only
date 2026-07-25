# WalkIn Redemption Classifier — Standalone Research Package

Self-contained copy of the WalkIn research pipeline for `research_notebook.ipynb` — no
Postgres, no API keys, no access to the rest of the WalkIn codebase. Uses a frozen CSV
snapshot of the real dataset (583 businesses, 4,091 deals, 25,172 popularity rows), loaded
into a local SQLite file.

## Run in Google Colab

Open directly from GitHub, then run the first cell — it clones this repo, installs
dependencies, and builds the local database automatically:

`https://colab.research.google.com/github/mattybell/Walkin-csv-only/blob/main/research_notebook.ipynb`

## Run locally

```
pip install -r requirements.txt
python build_local_db.py
jupyter notebook research_notebook.ipynb
```

## What's in here

| Path | What it is |
|---|---|
| `research_notebook.ipynb` | The research notebook |
| `train_redemption_classifier.py` | Training/evaluation pipeline |
| `build_local_db.py` | Builds the local SQLite DB from `data/*.csv` |
| `data/*.csv` | Frozen dataset snapshot |
| `app/` | Minimal ORM models needed by the pipeline |
| `requirements.txt` | Pinned dependencies |

## Note on reproducibility

Exact AUC-ROC can differ by a few thousandths from `research_report.md` (observed
~0.593–0.597 vs. reported 0.5974) due to CSV row-order effects on the seeded exposure
simulation. All qualitative conclusions hold.
