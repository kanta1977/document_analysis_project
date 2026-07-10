# %% [markdown]
# # 07b — Human vs. AI, by community — but each metric as its OWN image
#
# Same numbers and same logic as `07_human_vs_ai.py`: for every post we measure
# the same five things on the human TL;DR and on the AI baseline, aggregate per
# subreddit, and mark the overall AI baseline mean with a dotted line. The ONLY
# difference is the output: instead of one 2×3 grid, each metric is saved as its
# own figure, so you can drop them into the report/site individually.
#
# Reuses `features_for_post` (identical measurements) and, for cosine, the same
# `cosine.parquet` from nb 06 when present (so the cosine panel matches your
# existing figure exactly); otherwise it computes cosine with the same
# `pairwise_cosine` fallback as nb 07.
#
# **Output (NEW names — the combined `07_human_vs_ai_by_subreddit.*` is untouched):**
#   results/figures/07_first_person_density.png / .svg
#   results/figures/07_share_question_mark.png / .svg
#   results/figures/07_share_advice_words.png / .svg
#   results/figures/07_cosine_to_post.png / .svg
#   results/figures/07_keyword_containment.png / .svg
#
# Run: `python notebooks/07b_human_vs_ai_split.py`

# %%
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _find_root() -> Path:
    if "__file__" in dir():
        return Path(__file__).resolve().parents[1]
    for p in [Path.cwd()] + list(Path.cwd().parents):
        if (p / "src" / "tldr_audit").exists() or (p / "data" / "interim").exists():
            return p
    raise RuntimeError("Cannot locate project root. Open VS Code from the code/ folder.")
ROOT = _find_root()
print("ROOT:", ROOT)
sys.path.insert(0, str(ROOT / "src"))
from tldr_audit.features import features_for_post  # noqa: E402
from tldr_audit.semantic import pairwise_cosine  # noqa: E402

FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
COLORS = {"human": "#333333", "ai": "#D55E00"}

sample_path = ROOT / "data" / "interim" / "sample.jsonl"
ai_path = ROOT / "data" / "interim" / "ai_summaries.jsonl"
if not ai_path.exists():
    raise FileNotFoundError(
        "data/interim/ai_summaries.jsonl not found — run scripts/04_ai_baseline.py first."
    )

# sample.jsonl is only used to fill in content/human_tldr if missing from the AI
# file. ai_summaries.jsonl already carries both, so this is optional.
posts = ({json.loads(l)["id"]: json.loads(l) for l in open(sample_path, encoding="utf-8")}
         if sample_path.exists() else {})
ai = [json.loads(l) for l in open(ai_path, encoding="utf-8")]

# %% per-post metrics for human and AI (identical to nb 07)
rows = []
for a in ai:
    pid = a["id"]
    src = posts.get(pid, {})
    content = a.get("content") or src.get("content")
    human_tldr = a.get("human_tldr") or src.get("summary")
    ai_summary = a.get("summary")
    fh = features_for_post({"content": content, "summary": human_tldr}, with_ner=False)
    fa = features_for_post({"content": content, "summary": ai_summary}, with_ner=False)
    rows.append({
        "id": pid, "subreddit": a.get("subreddit"), "bucket": a.get("_bucket"),
        "content": content, "human": human_tldr, "ai": ai_summary,
        "fp_human": fh["first_person_summary"], "fp_ai": fa["first_person_summary"],
        "q_human": fh["has_question_mark"], "q_ai": fa["has_question_mark"],
        "adv_human": fh["has_advice_marker"], "adv_ai": fa["has_advice_marker"],
        "contain_human": fh["keyword_containment"], "contain_ai": fa["keyword_containment"],
    })
d = pd.DataFrame(rows)

# %% cosine — reuse nb 06 output if available, else compute (same as nb 07)
cos_pq = ROOT / "data" / "processed" / "cosine.parquet"
if cos_pq.exists():
    cos = pd.read_parquet(cos_pq)[["id", "cosine_human", "cosine_ai"]]
    d = d.merge(cos, on="id", how="left")
else:
    d["cosine_human"], _ = pairwise_cosine(d["content"], d["human"])
    d["cosine_ai"], _ = pairwise_cosine(d["content"], d["ai"])

# %% aggregate per subreddit + overall (identical to nb 07)
METRICS = {
    "first_person_density": ("fp_human", "fp_ai"),
    "share_question_mark": ("q_human", "q_ai"),
    "share_advice_words": ("adv_human", "adv_ai"),
    "cosine_to_post": ("cosine_human", "cosine_ai"),
    "keyword_containment": ("contain_human", "contain_ai"),
}


def agg(frame):
    out = {}
    for name, (hc, ac) in METRICS.items():
        out[(name, "human")] = frame[hc].astype(float).mean()
        out[(name, "ai")] = frame[ac].astype(float).mean()
    return pd.Series(out)


per_sub = d.groupby("subreddit").apply(agg)
overall = agg(d).to_frame("ALL").T
per_sub.columns = pd.MultiIndex.from_tuples(per_sub.columns)
overall.columns = pd.MultiIndex.from_tuples(overall.columns)

# %% one FIGURE PER METRIC (this is the only real change vs nb 07)
subs = list(per_sub.index)
x = np.arange(len(subs)); w = 0.38
for name, (hc, ac) in METRICS.items():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    h = [per_sub.loc[s, (name, "human")] for s in subs]
    a = [per_sub.loc[s, (name, "ai")] for s in subs]
    ax.bar(x - w / 2, h, w, label="human TL;DR", color=COLORS["human"])
    ax.bar(x + w / 2, a, w, label="AI baseline", color=COLORS["ai"], alpha=0.85)
    ax.axhline(overall[(name, "ai")].iloc[0], color=COLORS["ai"], lw=.8, ls=":")
    ax.set_title(name.replace("_", " "))
    ax.set_xticks(x); ax.set_xticklabels("r/" + pd.Index(subs), rotation=45, ha="right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / f"07_{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / f"07_{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"saved 07_{name}.png/.svg")