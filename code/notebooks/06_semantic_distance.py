# %% [markdown]
# # 06 — Semantic distance (item 3) and the two-axis map (items 3 + 4)
#
# For every post we measure cosine(post, TL;DR): does the summary *mean* the
# same thing, even in different words? We then place each community on two
# axes — keyword containment (lexical, item 4) vs. cosine (semantic, item 3) —
# which separates paraphrase from genuine divergence.
#
# Needs the post text, so it reads data/interim/sample.jsonl (the human TL;DRs)
# and, if present, data/interim/ai_summaries.jsonl (the AI baseline).
# Writes data/processed/cosine.parquet and two figures.

# %%
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Robust ROOT: walk up from wherever we are until we find src/tldr_audit/.
# Works when CWD is code/, code/notebooks/, or anywhere else.
def _find_root() -> Path:
    if "__file__" in dir():
        return Path(__file__).resolve().parents[1]
    for p in [Path.cwd()] + list(Path.cwd().parents):
        if (p / "src" / "tldr_audit").exists():
            return p
    raise RuntimeError(
        "Cannot locate project root. Open VS Code from the code/ folder "
        "or run: cd .../document_analysis_project/code"
    )
ROOT = _find_root()
print("ROOT:", ROOT)
sys.path.insert(0, str(ROOT / "src"))
from tldr_audit.semantic import pairwise_cosine  # noqa: E402
from tldr_audit.features import keyword_containment, summary_novelty  # noqa: E402

FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
COLORS = {"political": "#D55E00", "mental_health": "#0072B2", "advice": "#009E73"}

sample_path = ROOT / "data" / "interim" / "sample.jsonl"
ai_path = ROOT / "data" / "interim" / "ai_summaries.jsonl"
if not sample_path.exists():
    raise FileNotFoundError("Run scripts/01 and 02 first to create sample.jsonl")

posts = [json.loads(l) for l in open(sample_path, encoding="utf-8")]
df = pd.DataFrame({
    "id": [p.get("id") for p in posts],
    "subreddit": [p.get("subreddit") for p in posts],
    "bucket": [p.get("_bucket") for p in posts],
    "content": [p.get("content") for p in posts],
    "tldr": [p.get("summary") for p in posts],
})

# %% cosine(post, human TL;DR)
cos_h, backend = pairwise_cosine(df["content"], df["tldr"])
df["cosine_human"] = cos_h
print("cosine backend:", backend)

# lexical companions (reuse the same feature functions)
df["containment_human"] = [keyword_containment(c, s) for c, s in zip(df.content, df.tldr)]
df["novelty_human"] = [summary_novelty(c, s) for c, s in zip(df.content, df.tldr)]

# %% AI baseline, if generated
if ai_path.exists():
    ai = {json.loads(l)["id"]: json.loads(l) for l in open(ai_path, encoding="utf-8")}
    df["ai_summary"] = df["id"].map(lambda i: ai.get(i, {}).get("summary"))
    have_ai = df["ai_summary"].notna()
    if have_ai.any():
        cos_a, _ = pairwise_cosine(df.loc[have_ai, "content"], df.loc[have_ai, "ai_summary"])
        df.loc[have_ai, "cosine_ai"] = cos_a
        df.loc[have_ai, "containment_ai"] = [
            keyword_containment(c, s) for c, s in
            zip(df.loc[have_ai, "content"], df.loc[have_ai, "ai_summary"])
        ]
    print(f"AI summaries matched: {int(have_ai.sum()):,}")
else:
    print("No ai_summaries.jsonl yet — human-only maps (run scripts/04 for the AI baseline).")

df.drop(columns=["content", "tldr"]).to_parquet(
    ROOT / "data" / "processed" / "cosine.parquet", index=False
)


# %% Figure 05 — two-axis map: containment (lexical) vs cosine (semantic), human
def two_axis(ax, x, y, buckets, title):
    for b in ["political", "mental_health", "advice"]:
        m = buckets == b
        ax.scatter(x[m], y[m], s=6, alpha=0.25, color=COLORS[b], label=b)
    ax.axvline(0.5, color="#999", lw=.8, ls="--")
    ax.axhline(0.5, color="#999", lw=.8, ls="--")
    ax.set_xlabel("keyword containment  (lexical: are the post's key words reused?)")
    ax.set_ylabel("cosine to post  (semantic: same meaning?)")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    # quadrant labels
    ax.text(0.97, 0.05, "paraphrase\n(low words, same meaning)", fontsize=8, ha="right", color="#555")
    ax.text(0.03, 0.05, "diverged\n(few words, different meaning)", fontsize=8, color="#555")
    ax.text(0.97, 0.97, "extractive\nsummary", fontsize=8, ha="right", va="top", color="#555")


fig, ax = plt.subplots(figsize=(6.2, 5.4))
two_axis(ax, df["containment_human"].fillna(0).values,
         df["cosine_human"].values, df["bucket"].values,
         "Where human TL;DRs sit: words vs. meaning")
ax.legend(frameon=False, fontsize=8, markerscale=2, loc="lower center", ncol=3)
fig.tight_layout()
fig.savefig(FIG / "05_containment_vs_cosine_human.png", dpi=200, bbox_inches="tight")
fig.savefig(FIG / "05_containment_vs_cosine_human.svg", bbox_inches="tight")
print("saved 05_containment_vs_cosine_human")

# %% per-subreddit mean cosine (quick view)
print("\nMean cosine(post, TL;DR) by subreddit (self-report):")
print(df.groupby("subreddit")["cosine_human"].mean().round(3).sort_values())
