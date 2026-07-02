# GUIDE — the whole project, end to end

A single walkthrough of *what this project is, how the pieces fit, how to run it,
and how to read what comes out*. For the one-line data flow see `PIPELINE.md`;
for exact commands see `RUNNING.md`. This file is the map that connects them.

---

## 1. The idea in one paragraph

A Reddit "TL;DR" is usually assumed to be a summary. It often isn't — it can be a
joke, a question, or a reply. Instead of judging each TL;DR "summary or not"
(which needs hand-labelling we can't validate in a week), we treat the TL;DR as a
**text of unknown type** and measure **how far it sits from a plain summary**,
using an **AI summary of the same post as a fixed reference point**. Then we ask
how that distance differs **by community**. The AI is a ruler, not a judge.

## 2. What we measure (items 1–4)

For the human TL;DR *and* the AI summary of the same post:

1. **First person** — does "I" survive? (voice). Summaries usually thin out "I";
   Reddit's self-narration culture keeps it. Measured as first-person density on
   each side, placed side by side (not as a human-minus-AI subtraction).
2. **Surface signals** — how often a TL;DR contains a `?` or advice words
   (should/need to/avoid). Reported as **rates only**; we say a `?` *may* mean a
   question, we do **not** stamp the label "question" on it.
3. **Semantic distance** — cosine(post, TL;DR). Are they about the same thing,
   even in different words?
4. **Keyword containment** — do the post's key words appear in the TL;DR?

Items 3 + 4 work as a pair: containment (lexical) alone can't tell a paraphrase
from a real divergence; cosine (semantic) can. Low containment + high cosine =
paraphrase; low + low = genuinely diverged. **Cosine is a distance we describe,
never a verdict** that something "is not a summary".

## 3. File map

```
report_site/
├── README.md          the report = the website (GitHub Pages, minimal theme)
├── _config.yml        Jekyll config (title, theme)
├── figures/           images the report embeds
└── code/
    ├── configs/project.yaml      one config all scripts read
    ├── src/tldr_audit/
    │   ├── corpus.py             load config, stream corpus, sampling filters
    │   ├── features.py           per-post measures, surface flags,
    │   │                         keyword_containment, tldr_type heuristic
    │   └── semantic.py           cosine (Sentence-BERT or TF-IDF fallback)
    ├── scripts/
    │   ├── 01_inventory.py       count posts per subreddit
    │   ├── 02_sample.py          filter + stratified sample -> sample.jsonl
    │   ├── 03_features.py        human features -> features.parquet
    │   └── 04_ai_baseline.py     AI summaries (OpenAI mini) -> ai_summaries.jsonl
    ├── notebooks/
    │   ├── 03_explore_human_tldr.py     figs 01–03 + bucket table
    │   ├── 04b_summariness_decomposed.py fig 04b (summariness by post type)
    │   ├── 04_compare_communities.py     extra community views
    │   ├── 05_embeddings_gpu.py          optional GPU embeddings
    │   ├── 06_semantic_distance.py       cosine + map 05 -> cosine.parquet
    │   └── 07_human_vs_ai.py             human vs AI table + fig 07
    ├── tests/                    unit tests (features, corpus, sampling)
    ├── requirements.txt          CPU deps   · requirements-gpu.txt  GPU extras
    ├── PIPELINE.md · RUNNING.md · GUIDE.md
```

## 4. Run it end to end

```bash
# 0. env
conda create --name tldr python=3.11 && conda activate tldr
pip install -r code/requirements.txt
pytest code/tests -q

# 1. corpus zip -> data/raw/corpus-webis-tldr-17.zip

# 2. build data
python code/scripts/01_inventory.py
python code/scripts/02_sample.py
python code/scripts/03_features.py

# 3. AI baseline (needs a key; it is a reference point, not a gold standard)
export OPENAI_API_KEY=sk-...        # PowerShell: $env:OPENAI_API_KEY="sk-..."
python code/scripts/04_ai_baseline.py

# 4. distance, comparison, figures
python code/notebooks/06_semantic_distance.py
python code/notebooks/07_human_vs_ai.py
python code/notebooks/03_explore_human_tldr.py
python code/notebooks/04b_summariness_decomposed.py
```

Steps 3–4 need step 2. Once `features.parquet` and `ai_summaries.jsonl` exist,
re-run only the notebooks to redo analysis without touching the corpus.

## 5. How to read the outputs

- **`features.parquet`** — one row per post; the human-side numbers for items 1–4
  plus sentiment/compression. `03_features.py` prints bucket means and the
  surface-flag rates so you can sanity-check.
- **`ai_summaries.jsonl`** — one AI summary per subsampled post (id, subreddit,
  content, `summary` = AI, `human_tldr`). Resumable.
- **`cosine.parquet`** — per-post cosine for human (and AI) plus containment.
- **`results/tables/human_vs_ai_by_subreddit.csv`** — the headline table: for each
  metric, the human value and the AI value, per subreddit, with an `ALL` row =
  the overall baseline. **Read a column pair (human vs AI) as "how far the human
  TL;DR is from a plain summary in this community."**
- **`figures/`**: `01` compression, `02` sentiment, `03` first person, `04b`
  summariness by post type, `05` containment×cosine map, `07` human vs AI bars.
  To publish, copy the ones the report uses into top-level `figures/`.

## 6. What we are careful NOT to claim

- Not "the AI is correct" — it is one model at temperature 0, a ruler.
- Not "this TL;DR is not a summary" — low cosine = far in meaning = evidence,
  not proof. We describe distance.
- Not category labels from surface flags — only rates ("may read as a question").
- Not human-minus-AI subtraction as if the human were ground truth — we place the
  two side by side.
- Numbers are directional: lexical/abstractive proxies, VADER on short text,
  2017 English Reddit.

## 7. Publish

`README.md` + `_config.yml` are the site. Commit and push, then on GitHub:
Settings → Pages → Source "Deploy from a branch" → Branch `main` / root → Save.
Live at `https://<user>.github.io/<repo>/` after a minute or two. Do not commit
`data/` or API keys (both are covered by `.gitignore` / read from the
environment).
