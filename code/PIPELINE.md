# PIPELINE — how the files connect

The project turns raw Reddit posts into a comparison of **human TL;DRs vs. an
AI summary baseline**, per community. Data flows left to right.

```
                configs/project.yaml   ← one config, every script reads it
                          │
  data/raw/…zip           │
        │                 ▼
  01_inventory.py ──▶ results/tables/subreddit_inventory.csv
        │  (count posts per subreddit → fix the 9 subreddits)
        ▼
  02_sample.py ─────▶ data/interim/sample.jsonl        (posts WITH text)
        │  (filter + stratified sample)
        ├───────────────────────────────┐
        ▼                                ▼
  03_features.py                   04_ai_baseline.py  ── OpenAI mini, temp 0
        │  human features               │  (subsample → 1 summary per post)
        ▼                                ▼
  data/processed/features.parquet   data/interim/ai_summaries.jsonl
        │                                │
        └──────────────┬─────────────────┘
                       ▼
     06_semantic_distance.py ──▶ data/processed/cosine.parquet + fig 05
                       ▼
     07_human_vs_ai.py ───────▶ results/tables/human_vs_ai_by_subreddit.csv + fig 07
                       │
     03_explore_human_tldr.py / 04b_summariness_decomposed.py ──▶ figs 01–04b
                       ▼
                 ../figures/  →  ../README.md (the website)
```

## The library (`src/tldr_audit/`)

| File | Role |
|------|------|
| `corpus.py` | `load_config()`, `iter_posts()`, sampling filters. |
| `features.py` | All per-post measures + surface flags + `keyword_containment` + the `tldr_type` heuristic. Pure functions, no I/O. |
| `semantic.py` | `pairwise_cosine()` — Sentence-BERT if available, else TF-IDF cosine (no download). Returns the backend used. |

## Items 1–4 and where they live

| Item | What | Column / output |
|------|------|-----------------|
| ① first person | voice survives? | `first_person_summary` (human) vs AI, in nb 07 |
| ② surface signals | rates of `?` / advice words (no labels) | `has_question_mark`, `has_advice_marker` → rates in nb 07 |
| ③ semantic distance | cosine(post, TL;DR) | `semantic.py` → `cosine.parquet` (nb 06) |
| ④ keyword containment | post keywords reused? | `keyword_containment` (features) |

Items ③ + ④ combine into one map (nb 06): low containment + high cosine =
paraphrase; low + low = diverged. Cosine is a **distance**, not a verdict.

## The AI baseline

`04_ai_baseline.py` (script) and `notebooks/04_ai_baseline.py` (JupyterHub
version) are a fixed reference point: Gemma 3 27B at temperature 0, one neutral
prompt for all subreddits, ~200 posts/subreddit. Gemma is served via vLLM on a
Kubernetes pod; start a kubectl port-forward to localhost:8001 before running
(see RUNNING.md). No API key needed. Same feature functions are applied to the
AI summary in nb 07, so human and AI are measured identically.

## Folders

- `data/raw/` corpus zip (gitignored) · `data/interim/` sample.jsonl + ai_summaries.jsonl (gitignored)
- `data/processed/` features.parquet, cosine.parquet (numbers only)
- `results/figures/`, `results/tables/` outputs · `../figures/` figures used by the report

## Removed

The earlier stand-alone LLM-generation path (`summarize.py`, the old
`04_summarize.py`, config `models:`/`summarization:`) is gone. The AI model now
appears only as the neutral baseline in `04_ai_baseline.py`.
