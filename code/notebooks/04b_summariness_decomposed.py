# %% [markdown]
# # 04b — Is the TL;DR even a summary? (decomposed by post type)
#
# Replaces the pooled figure 04, which conflated two things: a *comment*
# TL;DR (a reply in a thread) and a *self-post* TL;DR (the author summarizing
# their own story). Pooling them made political communities look far more
# "non-summary" than they are — the effect was mostly that political buckets
# are ~92% comments. Here we split the two and keep the community axis visible.
#
# Reads only data/processed/features.parquet. No heavy computation.

# %%
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Robust ROOT: walk up from wherever we are until we find src/ or data/.
# Works when CWD is code/, code/notebooks/, or anywhere else.
def _find_root() -> Path:
    if "__file__" in dir():
        return Path(__file__).resolve().parents[1]
    for p in [Path.cwd()] + list(Path.cwd().parents):
        if (p / "src" / "tldr_audit").exists() or (p / "data" / "interim").exists():
            return p
    raise RuntimeError(
        "Cannot locate project root. Open VS Code from the code/ folder."
    )
ROOT = _find_root()
print("ROOT:", ROOT)
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

COLORS = {"political": "#D55E00", "mental_health": "#0072B2", "advice": "#009E73"}
BUCKET_LABEL = {"political": "Political", "mental_health": "Mental-health", "advice": "Advice"}
ORDER = ["political", "mental_health", "advice"]

df = pd.read_parquet(ROOT / "data" / "processed" / "features.parquet")

# Two independent "this is not a faithful summary" signals.
df["hi_nov"] = df["summary_novelty"] >= 0.8                     # vocab barely from the post
TH = 0.5
df["sent_flip"] = (((df.sentiment_content > TH) & (df.sentiment_summary < -TH))
                   | ((df.sentiment_content < -TH) & (df.sentiment_summary > TH)))


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    print("saved", name)


# %% Panel figure: (left) high-novelty share by bucket x post type,
#                  (right) novelty distribution self-post vs comment.
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

# --- left: grouped bars, the decomposition that fixes the confound ----------
share = (df.groupby(["bucket", "is_comment"])["hi_nov"].mean()
         .unstack("is_comment"))            # columns: False=self-post, True=comment
share = share.loc[ORDER]
x = np.arange(len(ORDER))
w = 0.38
axL.bar(x - w / 2, share[False] * 100, w, label="Self-post (own story)",
        color=[COLORS[b] for b in ORDER])
axL.bar(x + w / 2, share[True] * 100, w, label="Comment (reply in a thread)",
        color=[COLORS[b] for b in ORDER], alpha=0.45, hatch="//")
for xi, b in zip(x, ORDER):
    axL.text(xi - w / 2, share[False][b] * 100 + 0.6, f"{share[False][b]*100:.0f}%",
             ha="center", va="bottom", fontsize=9)
    axL.text(xi + w / 2, share[True][b] * 100 + 0.6, f"{share[True][b]*100:.0f}%",
             ha="center", va="bottom", fontsize=9)
axL.set_xticks(x)
axL.set_xticklabels([BUCKET_LABEL[b] for b in ORDER])
axL.set_ylabel("Share of TL;DRs that barely reuse the post (%)")
axL.set_title("Once you split post from comment, the\ncommunity gap shrinks — but doesn't vanish", fontsize=11)
axL.legend(frameon=False, fontsize=8.5, loc="upper left")
axL.spines[["top", "right"]].set_visible(False)
axL.set_ylim(0, 30)

# --- right: novelty distribution, self-post vs comment ----------------------
bins = np.linspace(0, 1, 21)
for is_c, lab, sty in [(False, "Self-post", dict(color="#444444")),
                       (True, "Comment", dict(color="#444444", alpha=0.4, hatch="//"))]:
    vals = df.loc[df.is_comment == is_c, "summary_novelty"]
    axR.hist(vals, bins=bins, density=True, histtype="stepfilled",
             linewidth=1.4, label=f"{lab} (median {vals.median():.2f})", **sty)
axR.axvline(0.8, color="#D55E00", lw=1, ls="--")
axR.text(0.805, axR.get_ylim()[1] * 0.92, "≥0.8: almost\nnever a summary",
         fontsize=8, color="#D55E00", va="top")
axR.set_xlabel("Summariness: share of TL;DR words NOT in the post  (0 = pure summary, 1 = all new)")
axR.set_ylabel("Density")
axR.set_title("Comments lean toward 'new text';\nself-posts toward reusing the post", fontsize=11)
axR.legend(frameon=False, fontsize=8.5)
axR.spines[["top", "right"]].set_visible(False)

fig.suptitle("Not every TL;DR is a summary — and the reason is mostly post type, not community",
             fontsize=12.5, fontweight="bold", y=1.02)
fig.tight_layout()
save(fig, "04b_summariness_decomposed")

# %% Print the underlying numbers (for captions / slides)
tbl = (df.groupby(["bucket", "is_comment"])
       .agg(n=("id", "size"),
            median_novelty=("summary_novelty", "median"),
            hi_novelty_pct=("hi_nov", lambda s: round(s.mean() * 100, 1)),
            sent_flip_pct=("sent_flip", lambda s: round(s.mean() * 100, 1)))
       .round(3))
print(tbl)
extractive = (df.summary_novelty <= 0.2).mean() * 100
print(f"\nClearly extractive (novelty<=0.2), all posts: {extractive:.1f}%")
