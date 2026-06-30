# %% [markdown]
# # 04 — Comparing how communities summarize themselves (the pivot)
#
# **What this notebook is (ELI5):** instead of "human vs machine", we now ask
# **"human vs human across communities"** — do r/depression, r/politics and
# r/legaladvice compress themselves in *measurably different ways*? This is the
# new headline of the project (decision logged 2026-06-12). It uses the
# numbers already in `data/processed/features.parquet` plus, later, sentence
# embeddings computed on the H100 (notebook 05).
#
# Run cell-by-cell in VS Code, or `python notebooks/04_compare_communities.py`.

# %%
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path.cwd()
FIG = ROOT / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
TAB = ROOT / "results" / "tables"; TAB.mkdir(parents=True, exist_ok=True)

COLORS = {"political": "#D55E00", "mental_health": "#0072B2", "advice": "#009E73"}
BUCKET_LABEL = {"political": "Political", "mental_health": "Mental-health", "advice": "Advice"}

df = pd.read_parquet(ROOT / "data" / "processed" / "features.parquet")
# Fair comparison: self-posts only, and subreddits with enough of them.
posts = df[~df["is_comment"]].copy()
posts = posts[posts["subreddit"].map(posts["subreddit"].value_counts()) >= 100]
print(f"{len(posts):,} self-posts across {posts['subreddit'].nunique()} subreddits")


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    print(f"saved {name}")

# %% [markdown]
# ## The community fingerprint
#
# ELI5: we describe each community with six numbers, each scaled 0–1 across the
# subreddits we study (0 = the lowest community on that trait, 1 = the highest),
# so the *shape* shows what makes a community distinctive:
#
# - **Keeps length**: how long the TL;DR is relative to the post (less compression).
# - **Keeps "I"**: first-person pronoun rate in the TL;DR.
# - **Emotional post**: how strongly positive/negative the original post is.
# - **Keeps emotion**: how much of that emotional charge the TL;DR retains.
# - **Hedges**: rate of "maybe / I guess / sort of" in the TL;DR.
# - **Abstractive**: how much the TL;DR rewrites instead of copying phrases.
#
# A radar/spider chart makes the *profile* of each community pop out.

# %%
def fingerprint_table(posts):
    # Pre-compute absolute sentiment so we never need groupby.apply (which has
    # changed signature across pandas versions). Works on any pandas.
    p = posts.assign(abs_c=posts["sentiment_content"].abs(),
                     abs_s=posts["sentiment_summary"].abs())
    g = p.groupby("subreddit")
    raw = pd.DataFrame({
        "Keeps length":   g["compression_ratio"].median(),
        'Keeps "I"':      g["first_person_summary"].mean(),
        "Emotional post": g["abs_c"].mean(),
        "Keeps emotion":  g["abs_s"].mean() / g["abs_c"].mean().clip(lower=1e-9),
        "Hedges":         g["hedge_rate_summary"].mean(),
        "Abstractive":    g["novel_bigram_rate"].mean(),
    })
    # min-max scale each trait to 0..1 so the radar compares shapes, not units
    scaled = (raw - raw.min()) / (raw.max() - raw.min())
    bucket = g["bucket"].first()
    return raw, scaled, bucket


raw, scaled, bucket = fingerprint_table(posts)
raw.round(3).to_csv(TAB / "community_fingerprint_raw.csv")
print(raw.round(3).to_string())

# %%
traits = list(scaled.columns)
angles = np.linspace(0, 2 * np.pi, len(traits), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
for sub in scaled.index:
    vals = scaled.loc[sub].tolist(); vals += vals[:1]
    c = COLORS[bucket[sub]]
    ax.plot(angles, vals, color=c, lw=1.8, label=f"r/{sub}")
    ax.fill(angles, vals, color=c, alpha=0.05)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(traits, fontsize=10)
ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["low", "", "high"], fontsize=8)
ax.set_title("Each community has a distinct self-summarizing 'fingerprint'", pad=24)
# de-duplicate legend by bucket colour
handles = [plt.Line2D([], [], color=COLORS[b], lw=3, label=BUCKET_LABEL[b]) for b in COLORS]
leg1 = ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.32, 1.10), fontsize=9, title="Bucket")
ax.add_artist(leg1)
ax.legend(loc="lower right", bbox_to_anchor=(1.38, -0.05), fontsize=7.5, title="Subreddit")
save(fig, "05_community_fingerprint")
plt.show()

# %% [markdown]
# ## Reading the fingerprint
# Look for *shapes*, not single points: political communities stretch toward
# "Abstractive" and away from 'Keeps "I"'; mental-health communities stretch
# toward 'Keeps "I"' and "Emotional post"; advice sits in between but leans on
# "Keeps length". The point of the pivot is exactly this: the same act —
# writing your own TL;DR — produces systematically different shapes depending
# on what community you are in.

# %% [markdown]
# ## Next: do communities also differ in *meaning*? (notebook 05, H100)
# The fingerprint is built from surface features. The embeddings notebook adds
# the semantic layer: which communities' TL;DRs are close in meaning-space
# (similarity & clustering), and how much meaning each preserves from post to
# TL;DR (semantic compression).


# %% [markdown]
# ## Do communities use *different words* in their TL;DRs? (lexical map, CPU)
#
# ELI5: before the fancy meaning-based embeddings (notebook 05, GPU), here is
# the cheap "bag of words" version. We pool all TL;DRs of each subreddit into
# one big document, turn it into a **TF-IDF vector** (a number per word that is
# high if the word is common in this community but rare across communities —
# i.e. its "signature" words), and then measure how similar two communities'
# signatures are (cosine similarity). This is the classic vector-space model
# from the lectures; it captures *vocabulary*, not meaning, which is exactly
# why pairing it with embeddings later is interesting.

# %%
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage

# Pool TL;DRs per subreddit from the sample (self-posts with enough data only).
keep_subs = set(posts["subreddit"].unique())
keep_ids = set(posts["id"])
pooled = {s: [] for s in keep_subs}
with open(ROOT / "data" / "interim" / "sample.jsonl") as f:
    for line in f:
        p = json.loads(line)
        if p["id"] in keep_ids:
            pooled[p["subreddit"]].append(p["summary"] or "")
subs = sorted(pooled)
docs = [" ".join(pooled[s]) for s in subs]

# TF-IDF over words appearing in >=2 communities, dropping ultra-rare noise.
vec = TfidfVectorizer(stop_words="english", min_df=2, max_features=20000, sublinear_tf=True)
X = vec.fit_transform(docs)
sim = cosine_similarity(X)
sim_df = pd.DataFrame(sim, index=["r/" + s for s in subs], columns=["r/" + s for s in subs])
sim_df.round(3).to_csv(TAB / "subreddit_tfidf_similarity.csv")
print("TF-IDF vocabulary similarity between communities:")
print(sim_df.round(2).to_string())

# %% [markdown]
# ### Heatmap + clustering: which communities "talk alike"?

# %%
fig, ax = plt.subplots(figsize=(7.5, 6.2))
im = ax.imshow(sim, cmap="viridis", vmin=sim[~np.eye(len(subs), dtype=bool)].min(), vmax=1)
ax.set_xticks(range(len(subs))); ax.set_xticklabels(["r/" + s for s in subs], rotation=45, ha="right")
ax.set_yticks(range(len(subs))); ax.set_yticklabels(["r/" + s for s in subs])
for i in range(len(subs)):
    for j in range(len(subs)):
        ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center",
                color="white" if sim[i, j] < 0.6 else "black", fontsize=8)
ax.set_title("Vocabulary overlap of community TL;DRs (TF-IDF cosine)")
fig.colorbar(im, ax=ax, shrink=0.8, label="cosine similarity")
save(fig, "06_tfidf_similarity_heatmap")
plt.show()

# %%
# Hierarchical clustering: turn similarity into distance and draw the tree.
dist = 1 - sim
np.fill_diagonal(dist, 0)
Z = linkage(dist[np.triu_indices(len(subs), 1)], method="average")
fig, ax = plt.subplots(figsize=(8, 4.5))
dendrogram(Z, labels=["r/" + s for s in subs], ax=ax, color_threshold=0.0,
           above_threshold_color="grey", leaf_rotation=30)
ax.set_title("Communities that summarize with similar vocabulary cluster together")
ax.set_ylabel("distance (1 − cosine)")
save(fig, "07_tfidf_dendrogram")
plt.show()

# %% [markdown]
# ### What to look for
# Buckets that share jargon cluster tightly (the two political subs; the
# personal-story subs). If a community lands somewhere surprising, that is a
# finding — note it rather than smoothing it. The embeddings map in notebook 05
# answers the deeper question: do communities that use *different words* still
# mean *similar things* when they summarize?
