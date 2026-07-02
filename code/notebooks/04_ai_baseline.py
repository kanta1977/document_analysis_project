# %% [markdown]
# # 04 — AI baseline summaries (Gemma 3 27B via vLLM)
#
# **What this notebook does (ELI5):**
# We take ~200 posts per subreddit from our sample and ask Gemma 3 27B to write
# a plain, neutral summary of each post. This is our **reference point** —
# "what does a straight summary of this post look like?" We never claim the AI
# is correct or better; we use it as a measuring stick so that "the human TL;DR
# uses a lot more first-person than a plain summary would" becomes a statement
# about the *human*, not just a raw number.
#
# **How Gemma is reached (no kubectl needed here):**
# JupyterHub and the Gemma pod are both on the same Kubernetes cluster, so we
# can reach Gemma directly by its pod IP — no port-forward needed.
# You only need to get the pod IP once, from your local machine (Cell 2 explains
# the exact command).
#
# **Output:** `data/interim/ai_summaries.jsonl` (one JSON line per post,
# resumable — re-run the notebook to continue from where it left off).

# %% [markdown]
# ## Cell 1 — Install / verify deps

# %%
import subprocess, sys

def _pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

try:
    import openai
    print(f"openai {openai.__version__} already installed")
except ImportError:
    print("installing openai …")
    _pip("openai>=1.0")
    import openai

try:
    import yaml
    print("pyyaml already installed")
except ImportError:
    _pip("pyyaml")
    import yaml

import json, time
from pathlib import Path

print("deps OK")

# %% [markdown]
# ## Cell 2 — Set the Gemma pod IP
#
# **Do this once on your local machine** (the one that has kubectl):
#
# ```bash
# kubectl get pod jorge-vllm-gemma3-27b \
#     -n user-jorge-lastra-cerda \
#     -o jsonpath='{.status.podIP}'
# ```
#
# It prints something like `10.42.1.234`. Paste it below.
# The port is always 8000 (that is the vLLM default inside the pod).
#
# Why does this work without port-forward? JupyterHub and the Gemma pod are both
# inside the same cluster network, so pods can talk to each other by IP directly.

# %%
POD_IP = "PASTE_POD_IP_HERE"   # ← replace with output of the kubectl command above

VLLM_BASE_URL = f"http://{POD_IP}:8000/v1"
VLLM_API_KEY  = "token"   # vLLM ignores this; any non-empty string works

print(f"Will call: {VLLM_BASE_URL}")

# %% [markdown]
# ## Cell 3 — Test the connection and detect the model name

# %%
client = openai.OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)

try:
    models = client.models.list()
    MODEL = models.data[0].id
    print(f"✓ Endpoint reachable. Model: {MODEL}")
except Exception as e:
    raise RuntimeError(
        f"Cannot reach vLLM endpoint at {VLLM_BASE_URL}.\n"
        f"Error: {e}\n\n"
        "Check that:\n"
        "  1. The pod is running:  kubectl get pod jorge-vllm-gemma3-27b -n user-jorge-lastra-cerda\n"
        "  2. The POD_IP in Cell 2 is correct (run the kubectl command above)\n"
        "  3. Port 8000 is not blocked inside the cluster"
    ) from e

# Quick smoke-test
_test = client.chat.completions.create(
    model=MODEL,
    temperature=0,
    max_tokens=30,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Summarize in one sentence: The cat sat on the mat."},
    ],
)
print("Smoke-test:", _test.choices[0].message.content.strip())

# %% [markdown]
# ## Cell 4 — Load config + sample
#
# ROOT is the `code/` directory. Run this notebook from inside
# `document_analysis_project/code/` (i.e. open JupyterHub from that folder,
# or use `%cd` to navigate there first).

# %%
# If __file__ is defined (running as a script), go up one level from notebooks/.
# In JupyterHub, Path.cwd() should be the code/ directory.
ROOT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path.cwd()
print("ROOT:", ROOT)

CONFIG_PATH = ROOT / "configs" / "project.yaml"
SAMPLE_PATH = ROOT / "data" / "interim" / "sample.jsonl"
OUT_PATH    = ROOT / "data" / "interim" / "ai_summaries.jsonl"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"Config not found at {CONFIG_PATH}\n"
        "Make sure you are running this notebook from the code/ directory.\n"
        "In JupyterHub terminal:  cd ~/document_analysis_project/code\n"
        "Then reopen the notebook."
    )

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

SEED = config.get("seed", 2026)
PER_SUBREDDIT = 200   # ~200 posts per subreddit → ~1,800 total

if not SAMPLE_PATH.exists():
    raise FileNotFoundError(
        f"sample.jsonl not found at {SAMPLE_PATH}\n"
        "Run:  python scripts/02_sample.py"
    )

posts = [json.loads(l) for l in open(SAMPLE_PATH, encoding="utf-8")]
print(f"Loaded {len(posts):,} posts from sample.jsonl")

# %% [markdown]
# ## Cell 5 — Stratified subsample
#
# At most PER_SUBREDDIT posts per subreddit, drawn deterministically.

# %%
import random

rng = random.Random(SEED)

by_sub: dict[str, list] = {}
for p in posts:
    by_sub.setdefault(p.get("subreddit"), []).append(p)

subsample = []
for sub, group in sorted(by_sub.items()):
    shuffled = list(group)
    rng.shuffle(shuffled)
    subsample.extend(shuffled[:PER_SUBREDDIT])

print(f"Subsample: {len(subsample):,} posts across {len(by_sub)} subreddits")
for sub in sorted(by_sub):
    n = sum(1 for p in subsample if p.get("subreddit") == sub)
    print(f"  r/{sub:<22s} {n}")

# %% [markdown]
# ## Cell 6 — Resume support
#
# If `ai_summaries.jsonl` already exists, we skip IDs already done.
# Safe to interrupt and re-run at any time.

# %%
done_ids: set[str] = set()
if OUT_PATH.exists():
    for line in open(OUT_PATH, encoding="utf-8"):
        try:
            done_ids.add(json.loads(line)["id"])
        except (json.JSONDecodeError, KeyError):
            pass
    print(f"Resuming: {len(done_ids)} posts already done, skipping them.")
else:
    print("Starting fresh.")

todo = [p for p in subsample if p.get("id") not in done_ids]
print(f"Posts to summarize: {len(todo):,}")

# %% [markdown]
# ## Cell 7 — Prompts
#
# One neutral system prompt for every subreddit. We deliberately do NOT tell the
# model to keep the first person — that is what we want to *measure*.

# %%
SYSTEM_PROMPT = (
    "You are a neutral summarization baseline. Write a faithful, concise "
    "summary of a single Reddit post. Add no opinions, advice, questions, or "
    "jokes, and no information not in the post. Output only the summary text — "
    "no preamble, no quotation marks, no labels."
)

USER_TEMPLATE = (
    "Summarize the following Reddit post in one or two sentences. "
    "Do not add anything that is not in the post.\n\nPOST:\n{post}"
)


def summarize_one(post_text: str, max_tokens: int = 150) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_TEMPLATE.format(post=post_text[:3000])},
        ],
    )
    return resp.choices[0].message.content.strip()

print("Prompts defined. Ready to run.")

# %% [markdown]
# ## Cell 8 — Run the baseline
#
# Progress printed every 50 posts. Output written after each post (resumable).
# Expected speed: ~3–6 posts/second → 5–10 min for ~1,800 posts total.

# %%
errors = 0
t0 = time.time()

with open(OUT_PATH, "a", encoding="utf-8") as out_f:
    for i, post in enumerate(todo):
        pid = post.get("id")
        content = post.get("content") or ""

        try:
            ai_summary = summarize_one(content)
        except Exception as e:
            print(f"  [!] {pid} failed: {e}; retrying in 5s …")
            time.sleep(5)
            try:
                ai_summary = summarize_one(content)
            except Exception as e2:
                print(f"  [!!] skipping {pid}: {e2}")
                errors += 1
                if errors > 20:
                    raise RuntimeError("Too many consecutive errors — check the endpoint.")
                continue

        out_f.write(json.dumps({
            "id":        pid,
            "subreddit": post.get("subreddit"),
            "_bucket":   post.get("_bucket"),
            "content":   content,
            "human_tldr": post.get("summary"),
            "summary":   ai_summary,   # ← AI summary, compared against human_tldr in nb 07
        }, ensure_ascii=False) + "\n")
        out_f.flush()

        if (i + 1) % 50 == 0 or (i + 1) == len(todo):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(todo) - i - 1) / rate if rate > 0 else 0
            print(f"  {i+1:>4}/{len(todo)}  {rate:.1f} posts/s  ETA {eta/60:.1f} min")

print(f"\nDone. {len(todo)-errors:,} summaries written to {OUT_PATH.name}")

# %% [markdown]
# ## Cell 9 — Sanity check
#
# Read back 3 random examples and compare human vs AI.
# Look for: AI drops "I/me", stays factual, no jokes or questions.

# %%
import random as _rand
written = [json.loads(l) for l in open(OUT_PATH, encoding="utf-8")]
print(f"Total in {OUT_PATH.name}: {len(written):,}\n")

_rand.seed(42)
for ex in _rand.sample(written, min(3, len(written))):
    print(f"r/{ex['subreddit']}  [{ex.get('_bucket','')}]")
    print(f"  HUMAN : {(ex.get('human_tldr') or '')[:200]}")
    print(f"  GEMMA : {ex['summary'][:200]}")
    print()

# %% [markdown]
# ## Next steps
#
# Once this cell finishes and `ai_summaries.jsonl` is ready:
#
# 1. **`python notebooks/06_semantic_distance.py`** — cosine(post, human TL;DR)
#    and cosine(post, AI summary) → figure 05 (containment vs cosine map)
# 2. **`python notebooks/07_human_vs_ai.py`** — items 1–4 per community →
#    figure 07 and `results/tables/human_vs_ai_by_subreddit.csv`
# 3. Copy the new figures to `../figures/` and push to GitHub Pages.
