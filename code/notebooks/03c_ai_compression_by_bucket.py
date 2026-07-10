# %% [markdown]
# # 03c — How hard the AI baseline compresses (companion to Figure 1)
#
# **What this notebook does (ELI5):** Figure 1 (`01_compression_by_bucket`,
# built in `03_explore_human_tldr.py`) shows how hard *people* compress their
# own posts. This notebook draws the exact same figure for the **AI baseline**
# (Gemma 3 27B, from `04_ai_baseline.py`) — on its own, not paired with the
# human boxes — so the two images can sit side by side and be read as a matched
# pair (same axis, same colours, same style).
#
# Same definition as Figure 1:
#   compression_ratio = (words in the summary) / (words in the post)
# via `tldr_audit.features.compression_ratio`. Lower = more thrown away. Like
# Figures 1–3, we use **self-posts only** (`is_comment` recovered from the id:
# a self-post id starts with `t3_`; a comment id does not — matches the rest of
# the project's `title.isna()` definition).
#
# **Input:** `data/interim/ai_summaries.jsonl` (already carries `content` and
# the AI `summary` per post — no features.parquet needed).
#
# **Output (NEW file names — nothing existing is overwritten):**
#   results/figures/01c_ai_compression_by_bucket.png / .svg
#
# Run: `python notebooks/03c_ai_compression_by_bucket.py`

# %%
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


# Robust ROOT: walk up until we find src/ or data/ (same helper as the other nbs).
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
from tldr_audit.features import compression_ratio  # noqa: E402

FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Same per-bucket Okabe-Ito palette as Figure 1, so this reads as its twin.
COLORS = {"political": "#D55E00", "mental_health": "#0072B2", "advice": "#009E73"}
BUCKET_LABEL = {"political": "Political debate", "mental_health": "Mental-health support",
                "advice": "Practical advice"}
ORDER = ["political", "mental_health", "advice"]

# %% [markdown]
# ## Measure the AI summary on self-posts only

# %%
ai_path = ROOT / "data" / "interim" / "ai_summaries.jsonl"
if not ai_path.exists():
    raise FileNotFoundError(
        f"{ai_path} not found — run scripts/04_ai_baseline.py first."
    )

recs = []
for line in open(ai_path, encoding="utf-8"):
    r = json.loads(line)
    recs.append({
        "id": r["id"],
        "bucket": r.get("_bucket"),
        "is_comment": not str(r["id"]).startswith("t3_"),
        "compression_ai": compression_ratio(r.get("content"), r.get("summary")),
    })
df = pd.DataFrame(recs)
posts = df[~df["is_comment"]].copy()   # self-posts only, like Figure 1
print(f"{len(df):,} rows total; {len(posts):,} self-posts used.")
print(posts.groupby("bucket").size())

# %% [markdown]
# ## Figure 01c — AI baseline compression by community
#
# Same y-axis and layout as Figure 1 (share of post length), so the two figures
# line up when placed next to each other.

# %%
fig, ax = plt.subplots(figsize=(7, 4.2))
data = [posts.loc[posts.bucket == b, "compression_ai"].dropna() for b in ORDER]
bp = ax.boxplot(data, labels=[BUCKET_LABEL[b] for b in ORDER], showfliers=False,
                patch_artist=True, medianprops=dict(color="black", linewidth=1.5))
for patch, b in zip(bp["boxes"], ORDER):
    patch.set_facecolor(COLORS[b]); patch.set_alpha(0.75)
ax.set_ylabel("AI summary length as a share of post length")
ax.set_title("A plain AI summary keeps ~17–21% of the post — everywhere")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
for i, d in enumerate(data, start=1):
    ax.annotate(f"median {d.median():.0%}", (i, d.median()),
                textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)

# NEW file names on purpose: does not touch 01_compression_by_bucket.*
fig.savefig(FIG / "01c_ai_compression_by_bucket.png", dpi=200, bbox_inches="tight")
fig.savefig(FIG / "01c_ai_compression_by_bucket.svg", bbox_inches="tight")
print("saved 01c_ai_compression_by_bucket.png/.svg")
plt.show()

# %% [markdown]
# ## Numbers behind the figure (self-posts only)

# %%
tbl = (posts.groupby("bucket")
       .agg(n=("id", "size"),
            ai_median=("compression_ai", "median"),
            ai_mean=("compression_ai", "mean"))
       .reindex(ORDER)
       .round(3))
print(tbl)