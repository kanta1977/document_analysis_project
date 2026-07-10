# %% [markdown]
# # 07 — Human TL;DR vs. AI baseline, by community
#
# The AI summary is a fixed REFERENCE POINT ("what a plain summary looks like").
# For every post we measure the same five things on the human TL;DR and on the
# AI summary, then compare per subreddit and overall:
#
#   ① first-person density        (voice — does "I" survive?)
#   ② share with question mark    (surface signal — rates only, no label)
#   ③ share with advice words     (surface signal — rates only, no label)
#   ④ cosine to the post          (semantic distance)
#   ⑤ keyword containment         (lexical overlap with the post)
#
# Each metric is saved as its own figure (07a–07e).
# We do NOT subtract "human minus AI"; we place the two side by side.
#
# Needs: data/interim/sample.jsonl  and  data/interim/ai_summaries.jsonl
# Optional: data/processed/cosine.parquet from nb 06 (TF-IDF fallback otherwise)

# %% [markdown]
# ## GPU cluster setup (run this cell first on JupyterHub / H100)

# %%
# Uncomment if needed on the cluster:
# !pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --quiet
# !pip install sentence-transformers --quiet

import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))

# %%
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Robust ROOT: walk up from wherever we are until we find src/ or data/.
# Works when CWD is code/, code/notebooks/, or the project root.
def _find_root() -> Path:
    if "__file__" in dir():
        return Path(__file__).resolve().parents[1]
    for p in [Path.cwd()] + list(Path.cwd().parents):
        if (p / "src" / "tldr_audit").exists() or (p / "data" / "interim").exists():
            return p
    raise RuntimeError(
        "Cannot locate project root. Open JupyterHub from the code/ folder."
    )

ROOT = _find_root()
print("ROOT:", ROOT)
sys.path.insert(0, str(ROOT / "src"))
from tldr_audit.features import features_for_post  # noqa: E402
from tldr_audit.semantic import pairwise_cosine    # noqa: E402

FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

COLORS = {"human": "#333333", "ai": "#D55E00"}
LEXICON = str(ROOT / "configs" / "lexicons" / "hedges.txt")

# %% load data
sample_path = ROOT / "data" / "interim" / "sample.jsonl"
ai_path     = ROOT / "data" / "interim" / "ai_summaries.jsonl"
if not ai_path.exists():
    raise FileNotFoundError(
        "data/interim/ai_summaries.jsonl not found — run scripts/04_ai_baseline.py first."
    )

posts = {json.loads(l)["id"]: json.loads(l) for l in open(sample_path, encoding="utf-8")}
ai    = [json.loads(l) for l in open(ai_path, encoding="utf-8")]

# %% per-post metrics for human and AI
rows = []
for a in ai:
    pid        = a["id"]
    src        = posts.get(pid, {})
    content    = a.get("content") or src.get("content")
    human_tldr = a.get("human_tldr") or src.get("summary")
    ai_summary = a.get("summary")

    fh = features_for_post({"content": content, "summary": human_tldr}, lexicon_path=LEXICON, with_ner=False)
    fa = features_for_post({"content": content, "summary": ai_summary},  lexicon_path=LEXICON, with_ner=False)
    rows.append({
        "id":            pid,
        "subreddit":     a.get("subreddit"),
        "bucket":        a.get("_bucket"),
        "content":       content,
        "human":         human_tldr,
        "ai":            ai_summary,
        "fp_human":      fh["first_person_summary"],
        "fp_ai":         fa["first_person_summary"],
        "q_human":       fh["has_question_mark"],
        "q_ai":          fa["has_question_mark"],
        "adv_human":     fh["has_advice_marker"],
        "adv_ai":        fa["has_advice_marker"],
        "contain_human": fh["keyword_containment"],
        "contain_ai":    fa["keyword_containment"],
    })
d = pd.DataFrame(rows)

# %% cosine — reuse nb 06 output if available, else compute here
cos_pq = ROOT / "data" / "processed" / "cosine.parquet"
if cos_pq.exists():
    cos = pd.read_parquet(cos_pq)[["id", "cosine_human", "cosine_ai"]]
    d = d.merge(cos, on="id", how="left")
else:
    print("cosine.parquet not found — computing with TF-IDF fallback (run nb 06 for SBERT)")
    d["cosine_human"], _ = pairwise_cosine(d["content"], d["human"])
    d["cosine_ai"],   _ = pairwise_cosine(d["content"], d["ai"])

# %% aggregate: per subreddit + overall baseline
METRICS = {
    "first_person_density":    ("fp_human",      "fp_ai"),
    "share_question_mark":     ("q_human",        "q_ai"),
    "share_advice_words":      ("adv_human",      "adv_ai"),
    "cosine_to_post":          ("cosine_human",   "cosine_ai"),
    "keyword_containment":     ("contain_human",  "contain_ai"),
}

def agg(frame):
    out = {}
    for name, (hc, ac) in METRICS.items():
        out[(name, "human")] = frame[hc].astype(float).mean()
        out[(name, "ai")]    = frame[ac].astype(float).mean()
    return pd.Series(out)

per_sub = d.groupby("subreddit").apply(agg)
overall  = agg(d).to_frame("ALL").T
table    = pd.concat([per_sub, overall])
table.columns = pd.MultiIndex.from_tuples(table.columns, names=["metric", "source"])
table = table.round(3)
table.to_csv(TAB / "human_vs_ai_by_subreddit.csv")
print("=== Human TL;DR vs AI baseline (means; last row = overall baseline) ===")
with pd.option_context("display.width", 200, "display.max_columns", None):
    print(table)

# %% Figure 07a–07e — one figure per metric
#
# File names and titles for each metric figure.
FIGURE_DEFS = [
    ("first_person_density",  "07a_first_person_density",  "first person density"),
    ("share_question_mark",   "07b_share_question_mark",   "share question mark"),
    ("share_advice_words",    "07c_share_advice_words",    "share advice words"),
    ("cosine_to_post",        "07d_cosine_to_post",        "cosine to post"),
    ("keyword_containment",   "07e_keyword_containment",   "keyword containment"),
]

subs = sorted(per_sub.index)   # alphabetical — consistent across all figures
x    = np.arange(len(subs))
w    = 0.38

for metric_key, fname, title in FIGURE_DEFS:
    hc, ac = METRICS[metric_key]

    h = [per_sub.loc[s, (metric_key, "human")] for s in subs]
    a = [per_sub.loc[s, (metric_key, "ai")]    for s in subs]
    ai_overall = overall[(metric_key, "ai")].iloc[0]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w / 2, h, w, label="human TL;DR", color=COLORS["human"])
    ax.bar(x + w / 2, a, w, label="AI baseline",  color=COLORS["ai"], alpha=0.85)
    ax.axhline(ai_overall, color=COLORS["ai"], lw=0.9, ls=":",
               label=f"AI overall ({ai_overall:.3f})")

    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(subs, rotation=45, ha="right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    fig.savefig(FIG / f"{fname}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / f"{fname}.svg",           bbox_inches="tight")
    plt.show()
    print(f"saved {fname}")
