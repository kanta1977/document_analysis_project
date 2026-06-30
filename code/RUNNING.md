# RUNNING — how to run the pipeline

All commands are run from the project root. Steps 1–3 build the data; the
notebooks build the figures; the website just opens in a browser.

## 0. Set up the environment (once)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

That installs pandas, pyarrow, vaderSentiment, matplotlib, pyyaml, pytest, etc.
NER and GPU embeddings are optional extras (see the end).

## 1. Get the corpus

Download the Webis-TLDR-17 corpus zip and put it here (it is large and is **not**
in git):

```
data/raw/corpus-webis-tldr-17.zip
```

The path is set in `configs/project.yaml` under `corpus.zip_path`.

## 2. Build the data (scripts, in order)

```bash
# 2a. Inventory: count posts per subreddit (run once; slow on the full zip).
python scripts/01_inventory.py
#     quick smoke test instead:
python scripts/01_inventory.py --limit 100000

# 2b. Sample: filter + take ~5k posts per subreddit -> data/interim/sample.jsonl
python scripts/02_sample.py

# 2c. Features: one row per post -> data/processed/features.parquet
python scripts/03_features.py
#     test on the first 500 posts:
python scripts/03_features.py --limit 500
```

After step 2c you will see, printed to the screen: a column check, bucket-level
means, and the **TL;DR type mix by bucket** (the `tldr_type` breakdown).

> Everything below only needs `features.parquet`, so once you have it you can
> re-run the analysis without touching the big corpus again.

## 3. Build the figures

```bash
python notebooks/03_explore_human_tldr.py        # figures 01–03 + summary table
python notebooks/04b_summariness_decomposed.py   # figure 04b
```

Figures are written to `results/figures/` as `.svg` and `.png`.

## 4. Build / view the website

The website embeds the figures into a single self-contained file. To rebuild it
after changing figures, re-run the small inliner that produced
`docs/site/index.html` (it reads the four SVGs and writes the page). Then just
open it:

```bash
open docs/site/index.html          # macOS  (Linux: xdg-open, Windows: start)
```

For GitHub Pages, serve the `docs/` folder (Settings → Pages → Source: `/docs`),
or move `docs/site/index.html` to `docs/index.html`.

## 5. Run the tests

```bash
pytest -q
```

You should see the feature, corpus, and sampling tests pass. (One NER test is
skipped unless the spaCy model is installed.)

## Optional extras

```bash
# Named-entity survival columns (needs the spaCy model):
python -m spacy download en_core_web_sm
python scripts/03_features.py --with-ner

# Semantic / topic-drift analysis (needs a GPU): open notebooks/05_embeddings_gpu
# on a GPU server and Run All; it reads sample.jsonl and writes its own outputs.
```

## Common issues

- **`Input sample file not found`** — run steps 2a/2b first; `03_features.py`
  needs `data/interim/sample.jsonl`.
- **parquet engine error** — `pip install pyarrow`.
- **`en_core_web_sm` not found** — only needed for `--with-ner`; otherwise ignore.
