# %% [markdown]
# # 08 — Sentiment: body → human TL;DR → AI summary, by community
#
# **What this notebook does:**
# Extends figure 02 with a third data point — the AI summary's sentiment.
# For each subreddit, one row now shows:
#
#   ●  dot         → average post body sentiment
#   →  arrowhead   → average human TL;DR sentiment  (same as fig 02)
#   ◆  diamond     → average AI summary sentiment    (new)
#
# Key question: when humans write a TL;DR they sometimes shift in mood vs.
# the post. Does the AI also shift, or does it stay near the body?
# The AI is our neutral reference — if it stays close to the body and the
# human drifts away, that drift belongs to the *human*, not the topic.
#
# Also includes: qualitative examples per genre.
# For each bucket (political, mental_health, advice) we surface the 3 posts
# where the human TL;DR diverges most from the AI baseline in voice + emotion.
#
# Privacy rules (CLAUDE.md):
#   • No full post texts or usernames on the website or in slides.
#   • Mental-health subreddits (depression, offmychest, Anxiety):
#     NEVER quote, even anonymized. Notebook marks these examples clearly.
#
# Reads:  data/interim/ai_summaries.jsonl
# Writes: results/figures/08_sentiment_body_human_ai.{png,svg}
#         results/tables/sentiment_body_human_ai.csv
#         ../figures/ (website copy)

# %% Cell 1 — deps
import subprocess
import sys


def _pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])


for pkg, imp in [
    ("vaderSentiment", "vaderSentiment"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("numpy", "numpy"),
]:
    try:
        __import__(imp)
    except ImportError:
        print(f"Installing {pkg} …")
        _pip(pkg)

print("deps OK")

# %% Cell 2 — ROOT + imports
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def _find_root() -> Path:
    if "__file__" in dir():
        return Path(__file__).resolve().parents[1]
    for p in [Path.cwd()] + list(Path.cwd().parents):
        if (p / "src" / "tldr_audit").exists() or (p / "data" / "interim").exists():
            return p
    raise RuntimeError(
        "Cannot locate project root. Open VS Code / JupyterHub from the code/ folder."
    )


ROOT = _find_root()
print("ROOT:", ROOT)

FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

COLORS = {
    "political": "#D55E00",
    "mental_health": "#0072B2",
    "advice": "#009E73",
}
BUCKET_LABEL = {
    "political": "Political debate",
    "mental_health": "Mental-health support",
    "advice": "Practical advice",
}
ORDER = ["political", "mental_health", "advice"]

MENTAL_HEALTH_SUBS = {"depression", "offmychest", "Anxiety"}

_sia = SentimentIntensityAnalyzer()


def vader(text: str | None) -> float:
    if not text or not text.strip():
        return 0.0
    return _sia.polarity_scores(str(text))["compound"]


# %% Cell 3 — load ai_summaries.jsonl
ai_path = ROOT / "data" / "interim" / "ai_summaries.jsonl"
if not ai_path.exists():
    raise FileNotFoundError(
        f"ai_summaries.jsonl not found at {ai_path}\n"
        "Run notebooks/04_ai_baseline.py first."
    )

rows = [json.loads(line) for line in open(ai_path, encoding="utf-8")]
print(f"Loaded {len(rows):,} rows from ai_summaries.jsonl")

df = pd.DataFrame({
    "id":         [r["id"] for r in rows],
    "subreddit":  [r.get("subreddit") for r in rows],
    "bucket":     [r.get("_bucket") for r in rows],
    "content":    [r.get("content") for r in rows],
    "human_tldr": [r.get("human_tldr") for r in rows],
    "ai_summary": [r.get("summary") for r in rows],
})

print("\nPosts per subreddit:")
print(df.groupby(["bucket", "subreddit"]).size().to_string())

# %% Cell 4 — compute VADER sentiment for all three texts
print("\nComputing VADER sentiment (body / human TL;DR / AI summary) …")
df["sent_body"]  = df["content"].apply(vader)
df["sent_human"] = df["human_tldr"].apply(vader)
df["sent_ai"]    = df["ai_summary"].apply(vader)

# shift = how much each summary moves from the body
df["shift_human"] = df["sent_human"] - df["sent_body"]
df["shift_ai"]    = df["sent_ai"]    - df["sent_body"]

print("\nOverall mean sentiment:")
print(df[["sent_body", "sent_human", "sent_ai", "shift_human", "shift_ai"]].mean().round(3))

# %% Cell 5 — aggregate per subreddit + save table
agg = (
    df.groupby(["bucket", "subreddit"])
    [["sent_body", "sent_human", "sent_ai", "shift_human", "shift_ai"]]
    .mean()
    .reset_index()
    .sort_values(["bucket", "sent_body"])
)

print("\nPer-subreddit averages:")
print(agg.round(3).to_string(index=False))

agg.round(3).to_csv(TAB / "sentiment_body_human_ai.csv", index=False)
print(f"\nSaved → results/tables/sentiment_body_human_ai.csv")

# %% Cell 6 — Figure 08: extended arrow plot
# Layout: one row per subreddit.
#   ● (circle)  = post body
#   → (arrow)   = direction of human TL;DR
#   ◆ (diamond) = AI summary position
# This lets you see at a glance: where does the human drift relative to body,
# and where does the AI land?

fig, ax = plt.subplots(figsize=(9, 5.5))

for y, row in enumerate(agg.itertuples()):
    c = COLORS[row.bucket]
    # body dot
    ax.plot(row.sent_body, y, "o", color=c, ms=8, zorder=4)
    # arrow: body → human TL;DR
    ax.annotate(
        "",
        xy=(row.sent_human, y),
        xytext=(row.sent_body, y),
        arrowprops=dict(arrowstyle="->", color=c, lw=2),
    )
    # AI summary diamond (slightly offset upward so it doesn't overlap the arrow)
    ax.plot(row.sent_ai, y + 0.18, "D", color=c, ms=7, alpha=0.65,
            markeredgewidth=0.8, markeredgecolor="white", zorder=4)

ax.set_yticks(range(len(agg)))
ax.set_yticklabels("r/" + agg["subreddit"], fontsize=9)
ax.axvline(0, color="#888", lw=0.8, ls="--")
ax.set_xlabel("average VADER compound score  (← more negative | more positive →)", fontsize=9)
ax.set_title(
    "Mood: post body (●) → human TL;DR (→) and AI summary (◆)",
    fontweight="bold", fontsize=11,
)

# custom legend
shape_handles = [
    plt.Line2D([0], [0], marker="o", color="#555", ms=8, ls="",
               label="post body (●)"),
    plt.Line2D([0], [0], color="#555", lw=2, marker=">", ms=7,
               label="human TL;DR (arrowhead)"),
    plt.Line2D([0], [0], marker="D", color="#555", ms=7, ls="",
               alpha=0.65, label="AI summary (◆)"),
]
color_handles = [
    plt.Line2D([], [], color=COLORS[b], lw=3, label=BUCKET_LABEL[b])
    for b in ORDER
]
ax.legend(
    handles=shape_handles + color_handles,
    frameon=False, fontsize=8, loc="lower right", ncol=1,
)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()

for ext in ["png", "svg"]:
    path = FIG / f"08_sentiment_body_human_ai.{ext}"
    kw = dict(bbox_inches="tight")
    if ext == "png":
        kw["dpi"] = 200
    fig.savefig(path, **kw)
    print(f"saved {path.name}")

plt.show()

# %% Cell 7 — copy to website figures folder
import shutil

website_fig = ROOT.parent / "figures"
if website_fig.exists():
    for fname in ["08_sentiment_body_human_ai.png", "08_sentiment_body_human_ai.svg"]:
        shutil.copy(FIG / fname, website_fig / fname)
    print(f"Copied to {website_fig}")
else:
    print(f"Website figures folder not found at {website_fig}; skipping.")

# %% Cell 8 — qualitative examples per genre
# -------------------------------------------------------------------------
# Scoring: we want posts where the human TL;DR diverges most from the AI in
#   (a) voice  — human kept "I/me/my", AI didn't
#   (b) emotion — human and AI landed at very different sentiment scores
#   (c) length  — TL;DR and AI summary are roughly comparable (not 10 words vs 200)
#
# Divergence score (higher = more interesting contrast):
#   fp_human * 8          — first person preserved in human TL;DR
#   (fp_human - fp_ai)    — human > AI on first person (bonus for the gap)
#   |sent_human - sent_ai|— human and AI landed emotionally differently
#   |shift_human|         — human itself moved a lot from the body
#   len_ratio             — bonus for similar lengths (more comparable)
# -------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z']+")
_FP = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}


def _fp_rate(text: str | None) -> float:
    toks = _WORD_RE.findall((text or "").lower())
    return sum(t in _FP for t in toks) / len(toks) if toks else 0.0


df["fp_human"] = df["human_tldr"].apply(_fp_rate)
df["fp_ai"]    = df["ai_summary"].apply(_fp_rate)

df["tldr_len"] = df["human_tldr"].str.len().fillna(0)
df["ai_len"]   = df["ai_summary"].str.len().fillna(0)
df["len_ratio"] = df.apply(
    lambda r: (
        min(r.tldr_len, r.ai_len) / max(r.tldr_len, r.ai_len)
        if max(r.tldr_len, r.ai_len) > 0 else 0.0
    ),
    axis=1,
)

df["divergence"] = (
    df["fp_human"] * 8
    + (df["fp_human"] - df["fp_ai"]).clip(0) * 5
    + (df["sent_human"] - df["sent_ai"]).abs() * 3
    + df["shift_human"].abs() * 2
    + df["len_ratio"]
)

# filter: readable TL;DR, non-empty, some first person in human
mask = (
    (df["tldr_len"] >= 30) & (df["tldr_len"] <= 400)
    & (df["ai_len"]  >= 20)
    & (df["fp_human"] > 0.02)
)
filtered = df[mask].copy()

print("\n" + "=" * 70)
print("QUALITATIVE EXAMPLES — top 3 per genre by human-vs-AI divergence")
print("=" * 70)
print()

for bucket in ORDER:
    bucket_df = filtered[filtered["bucket"] == bucket].nlargest(3, "divergence")
    print(f"{'─' * 70}")
    print(f"GENRE: {BUCKET_LABEL[bucket]}")
    print(f"{'─' * 70}")
    for i, (_, row) in enumerate(bucket_df.iterrows(), 1):
        is_mh = row["subreddit"] in MENTAL_HEALTH_SUBS
        if is_mh:
            label = "⚠  ANALYSIS ONLY — do NOT cite on website or slides (mental health)"
        else:
            label = "✓  website-safe  (anonymize before citing)"
        print(f"\nExample {i}  |  r/{row['subreddit']}  |  {label}")
        print(
            f"  Sentiment  body={row['sent_body']:.2f}  "
            f"human={row['sent_human']:.2f}  AI={row['sent_ai']:.2f}"
        )
        print(
            f"  First-person  human={row['fp_human']:.1%}  "
            f"AI={row['fp_ai']:.1%}"
        )
        if not is_mh:
            body_excerpt = (str(row["content"]) or "")[:250].strip()
            print(f"\n  BODY (excerpt):  {body_excerpt} …")
            print(f"  HUMAN TL;DR:     {row['human_tldr']}")
            print(f"  AI SUMMARY:      {row['ai_summary']}")
        else:
            print("  [Post text withheld — mental-health subreddit]")
    print()

print("=" * 70)
print("Reminder for slides / website:")
print("  • ✓ examples: use short, anonymized excerpts only (no usernames).")
print("  • ⚠  examples: show metrics only — never quote text publicly.")
print("=" * 70)
