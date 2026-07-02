# RUNNING — how to run the pipeline

All commands run from the project root.

## 0. Environment (once)

```bash
conda create --name tldr python=3.11
conda activate tldr
pip install -r code/requirements.txt
pytest code/tests -q            # 28 pass, 1 skipped (NER)
```

## 1. Corpus

Put the Webis-TLDR-17 zip at the path in `code/configs/project.yaml`
(`corpus.zip_path`, not in git):

```
data/raw/corpus-webis-tldr-17.zip
```

## 2. Build the data

```bash
python code/scripts/01_inventory.py                 # counts per subreddit (slow once)
python code/scripts/02_sample.py                    # -> data/interim/sample.jsonl
python code/scripts/03_features.py                  # -> data/processed/features.parquet
#   test on 500 first:  python code/scripts/03_features.py --limit 500
```

`03_features.py` prints bucket means and the surface-flag rates.

## 3. AI baseline (items measured against it)

```bash
# set your key (never commit it)
export OPENAI_API_KEY=sk-...           # PowerShell: $env:OPENAI_API_KEY="sk-..."

python code/scripts/04_ai_baseline.py --dry-run     # check the subsample, no API calls
python code/scripts/04_ai_baseline.py               # ~200/subreddit, temp 0
#   -> data/interim/ai_summaries.jsonl  (resumable; re-run to continue)
```

Model defaults to `gpt-4o-mini`; change with `--model`. It is a fixed reference
point, not a gold standard.

## 4. Distance + comparison + figures

```bash
python code/notebooks/06_semantic_distance.py       # cosine + map 05  -> cosine.parquet
python code/notebooks/07_human_vs_ai.py             # human vs AI table + fig 07
python code/notebooks/03_explore_human_tldr.py      # figs 01–03 + summary table
python code/notebooks/04b_summariness_decomposed.py # fig 04b
```

Figures land in `results/figures/`. To show them on the website, copy the ones
the report uses into the top-level `figures/` folder, e.g.:

```bash
cp results/figures/{01_compression_by_bucket,02_sentiment_shift_by_subreddit,03_first_person_survival,04b_summariness_decomposed,05_containment_vs_cosine_human,07_human_vs_ai_by_subreddit}.png figures/
```

then reference them in `README.md`.

## 5. View the website

`README.md` + `_config.yml` are the GitHub Pages site (theme: minimal). Push,
then Settings → Pages → Branch `main` → your site is at
`https://<user>.github.io/<repo>/`.

## Notes

- `cosine` uses Sentence-BERT if installed, else TF-IDF automatically (no
  download). For SBERT: `pip install sentence-transformers`.
- NER is optional: `python -m spacy download en_core_web_sm` then
  `python code/scripts/03_features.py --with-ner`.
- Common errors: `Input sample not found` → run 01/02 first; `OPENAI_API_KEY is
  not set` → export it; parquet engine error → `pip install pyarrow`.
