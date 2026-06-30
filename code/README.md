# Code

Pipeline for the report (human TL;DR only, no LLM).

- `scripts/01_inventory.py` → `02_sample.py` → `03_features.py` build the data.
- `src/tldr_audit/` holds the analysis logic (`features.py` incl. the `tldr_type` classifier).
- `notebooks/` turn `features.parquet` into the figures in `../figures/`.

See `PIPELINE.md` (how files connect) and `RUNNING.md` (how to run).
