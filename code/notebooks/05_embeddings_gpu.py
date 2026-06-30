# %% [markdown]
# # 05 — Semantic comparison of communities (RUN THIS ON THE H100)
#
# **How to run:** open this notebook on your JupyterHub **Full GPU (80 GB)**
# server and use **Run ▸ Run All Cells**. It is self-contained — it finds
# `sample.jsonl` wherever you uploaded it and writes everything into a new
# folder `embeddings_outputs/` next to it (figures + tables + cached vectors),
# which you can then download and drop into the repo's `results/`.
#
# **What it does, in three acts:**
#  1. **Benchmark** 2–3 sentence-embedding models and let the numbers pick the
#     winner (speed + how well TL;DRs separate by community).
#  2. **Embed** every post and every TL;DR once, cache the vectors to disk.
#  3. **Analyse:** semantic compression (how much meaning each community keeps)
#     and a semantic similarity map/dendrogram (who summarizes alike in meaning).

# %% [markdown]
# ### Act 0 — dependencies + matching the GPU
# Run this cell first. The default `torch` wheel is built for CUDA 13, but this
# node's driver is CUDA 12.4 — a mismatch that silently drops you onto the CPU.
# So this cell installs the **CUDA-12.4 build of torch**. The FIRST time it does
# that, it will ask you to **restart the kernel once** (Kernel ▸ Restart Kernel),
# then run everything again — after that it detects the H100 and flies.

# %%
import subprocess, sys


def _pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=False)


_pip("sentence-transformers", "scikit-learn", "pandas", "pyarrow", "matplotlib")

# Is torch present AND able to see the GPU?
try:
    import torch
    _gpu_ok = torch.cuda.is_available()
except Exception:
    _gpu_ok = False

if not _gpu_ok:
    # Install the torch build that matches this node's CUDA 12.4 driver.
    print("Installing the CUDA-12.4 build of torch to match the driver ...")
    _pip("--force-reinstall", "torch", "--index-url",
         "https://download.pytorch.org/whl/cu124")
    print("\n*** DONE. Now RESTART THE KERNEL (Kernel > Restart Kernel) and "
          "Run All Cells again. After the restart this message won't reappear "
          "and the notebook will use the H100. ***")
else:
    print("dependencies ready; GPU visible to torch:", _gpu_ok)

# %%
import glob
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Find sample.jsonl no matter where it was uploaded (cwd, data/interim/, anywhere).
_cands = ["sample.jsonl", "data/interim/sample.jsonl"] + glob.glob("**/sample.jsonl", recursive=True)
SAMPLE = next((Path(c) for c in _cands if Path(c).exists()), None)
assert SAMPLE is not None, "Upload sample.jsonl first (see the file browser on the left)."
print("using sample:", SAMPLE.resolve())

# All outputs go here — one tidy folder you can download afterward.
OUT = Path("embeddings_outputs")
FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
TAB = OUT / "tables"; TAB.mkdir(parents=True, exist_ok=True)
EMB = OUT / "vectors"; EMB.mkdir(parents=True, exist_ok=True)

rows = [json.loads(line) for line in open(SAMPLE)]
df = pd.DataFrame(rows)
# 'title' is null for comments; the bucket label rode along as '_bucket' in the sample.
df["is_comment"] = df["title"].isna()
if "_bucket" in df.columns:
    df = df.rename(columns={"_bucket": "bucket"})
# Same fair-comparison filter as the laptop notebook: self-posts, subs with >=100.
df = df[~df["is_comment"]]
df = df[df["subreddit"].map(df["subreddit"].value_counts()) >= 100].reset_index(drop=True)
print(f"{len(df):,} self-posts across {df['subreddit'].nunique()} subreddits")
print(df.groupby(["bucket", "subreddit"]).size().to_string())

# %%
import torch
from sentence_transformers import SentenceTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "|", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU only")

# %% [markdown]
# ## Act 1 — Benchmark candidate models
#
# We embed the TL;DRs of a balanced subsample with each model and ask: how fast
# is it, and do the vectors *separate the communities*? Separation is measured
# with the **silhouette score** (−1…1; higher = same-subreddit posts sit closer
# together). Higher silhouette + acceptable speed = the better model for us.

# %%
from sklearn.metrics import silhouette_score

if DEVICE == "cuda":
    CANDIDATES = [
        "sentence-transformers/all-MiniLM-L6-v2",   # fast, small (384-dim)
        "sentence-transformers/all-mpnet-base-v2",  # stronger general (768-dim)
        "BAAI/bge-base-en-v1.5",                     # strong retrieval (768-dim)
    ]
else:
    # No GPU detected -> embedding is CPU-bound. Benchmark only the fast model
    # so the notebook still finishes in minutes. (Re-add the others if you get
    # the GPU working.)
    print("No GPU -> using only all-MiniLM-L6-v2 to keep CPU runtime short.")
    CANDIDATES = ["sentence-transformers/all-MiniLM-L6-v2"]

per_sub = max(50, 1500 // df["subreddit"].nunique())
# Balanced subsample, written to work on any pandas version (no .apply tricks).
_parts = [d.sample(min(len(d), per_sub), random_state=2026)
          for _, d in df.groupby("subreddit")]
sub = pd.concat(_parts)
sub_labels = sub["subreddit"].tolist()
bench_texts = sub["summary"].fillna("").tolist()

bench = []
for name in CANDIDATES:
    try:
        m = SentenceTransformer(name, device=DEVICE)
        t0 = time.time()
        emb = m.encode(bench_texts, batch_size=256, normalize_embeddings=True,
                       show_progress_bar=False)
        dt = time.time() - t0
        sil = float(silhouette_score(emb, sub_labels, metric="cosine"))
        bench.append({"model": name, "dim": int(emb.shape[1]),
                      "texts_per_sec": round(len(bench_texts) / dt),
                      "silhouette": round(sil, 4)})
        print(bench[-1])
    except Exception as e:
        print(f"!! {name} failed: {e}")
    finally:
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

bench_df = pd.DataFrame(bench)
bench_df.to_csv(TAB / "embedding_model_benchmark.csv", index=False)
print("\n", bench_df.to_string(index=False))

# Auto-pick best silhouette (tie-break: faster). To override, set CHOSEN by hand.
CHOSEN = bench_df.sort_values(["silhouette", "texts_per_sec"],
                              ascending=[False, False]).iloc[0]["model"]
print("\nCHOSEN MODEL:", CHOSEN)

# %% [markdown]
# ## Act 2 — Embed everything once, cache to disk
# Vectors are saved as .npy in `embeddings_outputs/vectors/`. Re-running this
# cell reloads them instead of recomputing (so you never pay GPU time twice).

# %%
tag = CHOSEN.split("/")[-1]
post_path = EMB / f"emb_post_{tag}.npy"
summ_path = EMB / f"emb_summary_{tag}.npy"

if post_path.exists() and summ_path.exists():
    post_emb = np.load(post_path); summ_emb = np.load(summ_path)
    print("loaded cached embeddings:", post_emb.shape, summ_emb.shape)
else:
    model = SentenceTransformer(CHOSEN, device=DEVICE)
    t0 = time.time()
    post_emb = model.encode(df["content"].fillna("").tolist(), batch_size=256,
                            normalize_embeddings=True, show_progress_bar=True)
    summ_emb = model.encode(df["summary"].fillna("").tolist(), batch_size=256,
                            normalize_embeddings=True, show_progress_bar=True)
    post_emb = np.asarray(post_emb, dtype=np.float32)
    summ_emb = np.asarray(summ_emb, dtype=np.float32)
    np.save(post_path, post_emb); np.save(summ_path, summ_emb)
    print(f"embedded {2*len(df):,} texts in {time.time()-t0:.0f}s -> cached")

# %% [markdown]
# ## Act 3a — Semantic compression: how much meaning survives?
# Each post and its TL;DR are points in meaning-space; their cosine similarity
# says how close the summary's meaning is to the full post (1.0 = keeps
# everything). Averaged per community = who writes the most "meaning-complete"
# TL;DRs. This is the semantic twin of the simple length-based compression.

# %%
import matplotlib.pyplot as plt

df["semantic_similarity"] = (post_emb * summ_emb).sum(axis=1)  # both L2-normalized
sem = (df.groupby(["bucket", "subreddit"])["semantic_similarity"]
         .agg(["mean", "median", "count"]).round(3))
sem.to_csv(TAB / "semantic_compression_by_subreddit.csv")
print(sem.to_string())

COLORS = {"political": "#D55E00", "mental_health": "#0072B2", "advice": "#009E73"}
order = df.groupby("subreddit")["semantic_similarity"].mean().sort_values()
bckt = df.groupby("subreddit")["bucket"].first()
fig, ax = plt.subplots(figsize=(7.5, 4.4))
ax.barh(["r/" + s for s in order.index], order.values,
        color=[COLORS.get(bckt[s], "grey") for s in order.index], alpha=0.85)
ax.set_xlabel("avg meaning kept (cosine between post and its TL;DR)")
ax.set_title("Semantic compression: whose TL;DR stays closest to the post?")
for y, s in enumerate(order.index):
    ax.annotate(f"{order[s]:.2f}", (order[s], y), xytext=(4, 0),
                textcoords="offset points", va="center", fontsize=8)
fig.savefig(FIG / "08_semantic_compression.png", dpi=200, bbox_inches="tight")
fig.savefig(FIG / "08_semantic_compression.svg", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Act 3b — Semantic community map: who summarizes *alike* in meaning?
# Each community's average TL;DR vector (its "semantic centroid"), compared
# pairwise. Read this against the TF-IDF *vocabulary* map (laptop notebook 04):
# different words but close here = they mean similar things.

# %%
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import dendrogram, linkage

subs = sorted(df["subreddit"].unique())
cent = np.vstack([normalize(summ_emb[(df["subreddit"] == s).values].mean(axis=0, keepdims=True))
                  for s in subs])
sim = cosine_similarity(cent)
sim_df = pd.DataFrame(sim, index=["r/" + s for s in subs], columns=["r/" + s for s in subs])
sim_df.round(3).to_csv(TAB / "subreddit_semantic_similarity.csv")
print(sim_df.round(2).to_string())

fig, ax = plt.subplots(figsize=(7.5, 6.2))
off = ~np.eye(len(subs), dtype=bool)
im = ax.imshow(sim, cmap="magma", vmin=sim[off].min(), vmax=1)
ax.set_xticks(range(len(subs))); ax.set_xticklabels(["r/" + s for s in subs], rotation=45, ha="right")
ax.set_yticks(range(len(subs))); ax.set_yticklabels(["r/" + s for s in subs])
for i in range(len(subs)):
    for j in range(len(subs)):
        ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center",
                color="white" if sim[i, j] < 0.7 else "black", fontsize=8)
ax.set_title("Semantic similarity of community TL;DRs (embedding centroids)")
fig.colorbar(im, ax=ax, shrink=0.8, label="cosine similarity")
fig.savefig(FIG / "09_semantic_similarity_heatmap.png", dpi=200, bbox_inches="tight")
fig.savefig(FIG / "09_semantic_similarity_heatmap.svg", bbox_inches="tight")
plt.show()

dist = 1 - sim; np.fill_diagonal(dist, 0)
Z = linkage(dist[np.triu_indices(len(subs), 1)], method="average")
fig, ax = plt.subplots(figsize=(8, 4.5))
dendrogram(Z, labels=["r/" + s for s in subs], ax=ax, color_threshold=0.0,
           above_threshold_color="grey", leaf_rotation=30)
ax.set_title("Communities clustered by MEANING of their TL;DRs")
ax.set_ylabel("distance (1 − cosine)")
fig.savefig(FIG / "10_semantic_dendrogram.png", dpi=200, bbox_inches="tight")
fig.savefig(FIG / "10_semantic_dendrogram.svg", bbox_inches="tight")
plt.show()

print("\nDONE. Everything is in the 'embeddings_outputs/' folder — download it "
      "and send Claude: tables/embedding_model_benchmark.csv, "
      "figures 08/09/10, tables/semantic_compression_by_subreddit.csv")
