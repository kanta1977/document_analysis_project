# PIPELINE — how the files connect

This project turns the raw Reddit corpus into figures about **what a human
TL;DR actually is**. There is **no LLM step**: every number comes from the
authors' own TL;DRs. Data flows in one direction, left to right.

```
                 configs/project.yaml   ← one config, every script reads it
                          │
   data/raw/corpus-…zip   │
          │               │
          ▼               ▼
  scripts/01_inventory.py ──▶ results/tables/subreddit_inventory.csv
          │  (count posts per subreddit; decide the 9 subreddits)
          ▼
  scripts/02_sample.py ─────▶ data/interim/sample.jsonl
          │  (filter + take ~5k posts per subreddit, deterministic)
          ▼
  scripts/03_features.py ───▶ data/processed/features.parquet
          │  (one row per post; calls features_for_post())
          ▼
  notebooks/  ─────────────▶ results/figures/*.svg + results/tables/*.csv
          │
          ▼
  docs/site/index.html  ← embeds the figures = the website
```

## The library (`src/tldr_audit/`)

The scripts are thin; the real logic lives here so it can be unit-tested.

| File | What it does |
|------|--------------|
| `corpus.py` | `load_config()` reads the YAML; `iter_posts()` streams the zip; `passes_filters()` / `word_count()` apply the sampling rules. |
| `features.py` | All per-post measurements **and** the heuristic classifier. One pure function per feature, plus `features_for_post()` that bundles them into a row, plus `classify_tldr()` that returns the `tldr_type` label. No I/O. |

## What `features.py` produces (per post)

`features_for_post(post)` returns one flat dict. The columns, by group:

- **length / compression** — `content_words`, `summary_words`, `compression_ratio`, `word_drop_rate`
- **summariness (is it even a summary?)** — `summary_novelty`, `novel_bigram_rate`
- **sentiment** — `sentiment_content`, `sentiment_summary`, `sentiment_shift`, `sentiment_flip`
- **first-person voice** — `first_person_content`, `first_person_summary`, `first_person_drop`, `i_disappears`
- **surface flags** (ingredients for the classifier) — `has_question_mark`, `has_second_person`, `has_advice_marker`, `has_joke_marker`
- **hedges** (kept, cheap) — `hedge_rate_content`, `hedge_rate_summary`
- **the classifier** — `tldr_type` ∈ {`summary`, `question`, `advice`, `reaction`}

`tldr_type` is decided by a fixed priority in `classify_tldr()`:
`?` → question; joke marker → reaction; advice language → advice;
novelty ≥ 0.8 → reaction; otherwise → summary. It is a transparent heuristic,
not a trained/validated model — good for *describing* how each community uses
the slot, and clearly labelled as a proxy.

## The notebooks (read `features.parquet`, write figures)

| Notebook | Reads | Writes |
|----------|-------|--------|
| `03_explore_human_tldr.py` | `features.parquet` | figures 01–03, `rq1_bucket_summary.csv` |
| `04b_summariness_decomposed.py` | `features.parquet` | figure 04b (summariness split by post type) |
| `04_compare_communities.py` | `features.parquet` | community-comparison views |
| `05_embeddings_gpu` *(optional, GPU)* | `sample.jsonl` | semantic / topic-drift outputs |

## Folders

- `data/raw/` — the downloaded corpus zip (not in git).
- `data/interim/sample.jsonl` — the sampled posts **with text** (not in git).
- `data/processed/features.parquet` — numbers only, no post text (safe to share).
- `results/figures/`, `results/tables/` — outputs used by the website.
- `tests/` — unit tests for `corpus.py`, `features.py`, and sampling.

## What was removed

The earlier LLM-summary path (`src/tldr_audit/summarize.py`,
`scripts/04_summarize.py`, the `models:` / `summarization:` config blocks, and
their test) has been deleted. Comparing against machine summaries is now
"future work", not part of this pipeline.
