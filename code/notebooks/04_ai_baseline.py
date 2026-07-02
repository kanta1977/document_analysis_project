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
# **How Gemma is served:**
# Gemma 3 27B runs as a Kubernetes pod (`jorge-vllm-gemma3-27b`) with a vLLM
# backend. vLLM exposes an OpenAI-compatible REST API at port 8000 in the pod.
# We port-forward that port to localhost:8001 in this notebook (or you can open
# a terminal and run the command manually — see cell 2).
#
# **Output:** `data/interim/ai_summaries.jsonl` (one JSON line per post,
# resumable — re-run the notebook to continue from where it left off).
#
# Format: py:percent — open in JupyterHub as a notebook, or run as a plain script.

# %% [markdown]
# ## Cell 1 — Install / verify deps
#
# The `openai` Python package works with any OpenAI-compatible endpoint,
# including vLLM. We use it here so the code is identical to what you'd write
# for the real OpenAI API — only the `base_url` and `api_key` change.

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
    print(f"pyyaml already installed")
except ImportError:
    _pip("pyyaml")
    import yaml

import json, os, time, atexit
from pathlib import Path

# %% [markdown]
# ## Cell 2 — Port-forward: connect to the Gemma pod
#
# **Option A (automatic, this cell):** The cell below starts the kubectl
# port-forward as a background process. It will stay alive as long as the
# notebook kernel is running.
#
# **Option B (manual):** Open a JupyterHub terminal and run:
# ```
# kubectl port-forward pod/jorge-vllm-gemma3-27b 8001:8000 \
#     -n user-jorge-lastra-cerda --address 0.0.0.0
# ```
# Then skip this cell (set `MANUAL_PORTFORWARD = True` below).

# %%
MANUAL_PORTFORWARD = False   # set True if you already ran the command in a terminal

VLLM_BASE_URL = "http://localhost:8001/v1"
VLLM_API_KEY  = "token"          # vLLM ignores this; any non-empty string works

_pf_proc = None

if not MANUAL_PORTFORWARD:
    print("Starting port-forward: pod/jorge-vllm-gemma3-27b 8001:8000 …")
    _pf_proc = subprocess.Popen(
        [
            "kubectl", "port-forward",
            "pod/jorge-vllm-gemma3-27b",
            "8001:8000",
            "-n", "user-jorge-lastra-cerda",
            "--address", "0.0.0.0",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(_pf_proc.terminate)   # clean up when kernel shuts down
    time.sleep(3)                          # give kubectl a moment to establish

    if _pf_proc.poll() is not None:
        raise RuntimeError(
            "Port-forward exited immediately — is the pod running?\n"
            "Check with:  kubectl get pod jorge-vllm-gemma3 -n user-jorge-lastra-cerda"
        )
    print("Port-forward running (PID", _pf_proc.pid, ")")
else:
    print("Manual port-forward assumed; skipping subprocess.")

# %% [markdown]
# ## Cell 3 — Detect the model name and test the endpoint

# %%
client = openai.OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)

# Ask vLLM which models are loaded — pick the first one.
try:
    models = client.models.list()
    MODEL = models.data[0].id
    print(f"Endpoint reachable. Model: {MODEL}")
except Exception as e:
    raise RuntimeError(
        f"Cannot reach vLLM endpoint at {VLLM_BASE_URL}.\n"
        f"Error: {e}\n"
        "Make sure the port-forward is running (see Cell 2)."
    ) from e

# Quick smoke-test: summarize one sentence.
_test = client.chat.completions.create(
    model=MODEL,
    temperature=0,
    max_tokens=50,
    messages=[
        {"role": "system",  "content": "You are a helpful assistant."},
        {"role": "user",    "content": "Summarize: The cat sat on the mat."},
    ],
)
print("Smoke-test response:", _test.choices[0].message.content.strip())

# %% [markdown]
# ## Cell 4 — Load config + sample

# %%
# ROOT: when running as a notebook (__file__ not defined), Path.cwd() should be
# the `code/` directory (where you launched JupyterHub from).
# When run as a script it resolves to code/ via parents[1].
ROOT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path.cwd()
print("ROOT:", ROOT)

CONFIG_PATH  = ROOT / "configs" / "project.yaml"
SAMPLE_PATH  = ROOT / "data" / "interim" / "sample.jsonl"
OUT_PATH     = ROOT / "data" / "interim" / "ai_summaries.jsonl"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

SEED = config.get("seed", 2026)
PER_SUBREDDIT = 200   # ~200 posts per subreddit → ~1,800 total; change if needed

if not SAMPLE_PATH.exists():
    raise FileNotFoundError(
        f"sample.jsonl not found at {SAMPLE_PATH}\n"
        "Run scripts/02_sample.py first."
    )

posts = [json.loads(l) for l in open(SAMPLE_PATH, encoding="utf-8")]
print(f"Loaded {len(posts):,} posts from sample.jsonl")

# %% [markdown]
# ## Cell 5 — Stratified subsample
#
# We draw at most PER_SUBREDDIT posts per subreddit, deterministically.
# This keeps the AI call budget manageable (~1,800 posts total).

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
for sub, group in sorted(by_sub.items()):
    n = sum(1 for p in subsample if p.get("subreddit") == sub)
    print(f"  r/{sub}: {n}")

# %% [markdown]
# ## Cell 6 — Load already-done IDs (resume support)
#
# If `ai_summaries.jsonl` already exists (from a previous run), we skip any
# post IDs it already contains. Re-run this notebook at any time to continue.

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
    print("Starting fresh (no existing ai_summaries.jsonl).")

todo = [p for p in subsample if p.get("id") not in done_ids]
print(f"Posts to summarize: {len(todo):,}")

# %% [markdown]
# ## Cell 7 — Prompts
#
# One neutral system prompt for every subreddit. We deliberately do NOT tell the
# model to keep the first person — the whole point is to measure whether the
# *human* TL;DR retains "I/me/my" more than a plain summary would.

# %%
SYSTEM_PROMPT = (
    "You are a neutral summarization baseline. Write a faithful, concise "
    "summary of a single Reddit post. Add no opinions, advice, questions, or "
    "jokes, and no information not in the post. Output only the summary text — "
    "no preamble, no quotation marks, no labels."
)

USER_TEMPLATE = (
    "Summarize the following Reddit post. "
    "Do not add anything that is not in the post.\n\nPOST:\n{post}"
)


def summarize_one(post_text: str, max_tokens: int = 150) -> str:
    """Call Gemma once; return the summary string."""
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

# %% [markdown]
# ## Cell 8 — Run the baseline
#
# Progress is printed every 50 posts. The output file is written incrementally,
# so if the kernel dies or the port-forward drops, you can re-run from Cell 6.
#
# Expected speed on Gemma 3 27B: ~3–6 posts/second → ~300–600 s for 1,800 posts.

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
            print(f"  [!] post {pid} failed: {e}")
            errors += 1
            if errors > 20:
                raise RuntimeError("Too many errors — check the vLLM endpoint.")
            continue

        record = {
            "id":           pid,
            "subreddit":    post.get("subreddit"),
            "_bucket":      post.get("_bucket"),
            "content":      content,
            "human_tldr":   post.get("summary"),
            "summary":      ai_summary,          # AI summary
        }
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_f.flush()

        if (i + 1) % 50 == 0 or (i + 1) == len(todo):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(todo) - i - 1) / rate if rate > 0 else 0
            print(
                f"  {i+1:>4}/{len(todo)}  "
                f"{rate:.1f} posts/s  "
                f"ETA {remaining/60:.1f} min"
            )

elapsed_total = time.time() - t0
print(f"\nDone. {len(todo) - errors:,} summaries written in {elapsed_total/60:.1f} min.")
print(f"Output: {OUT_PATH}")

# %% [markdown]
# ## Cell 9 — Quick sanity check
#
# Read back the file and print a few examples so you can eyeball the quality.
# Look for: is the summary faithful? Does it avoid adding opinions or questions?
# Does it drop the first person ("I" → "the poster")?

# %%
written = [json.loads(l) for l in open(OUT_PATH, encoding="utf-8")]
print(f"Total records in {OUT_PATH.name}: {len(written):,}\n")

import random as _rand
_rand.seed(99)
examples = _rand.sample(written, min(3, len(written)))

for ex in examples:
    print(f"r/{ex['subreddit']} [{ex['_bucket']}]")
    print(f"  HUMAN TL;DR : {ex['human_tldr'][:200]}")
    print(f"  AI SUMMARY  : {ex['summary'][:200]}")
    print()

# %% [markdown]
# ## Next steps
#
# Once this notebook finishes:
# 1. Run `notebooks/06_semantic_distance.py` — cosine(post, human TL;DR) and
#    cosine(post, AI summary) per post → figure 05 (containment vs cosine map).
# 2. Run `notebooks/07_human_vs_ai.py` — items 1–4 per community → figure 07
#    and `results/tables/human_vs_ai_by_subreddit.csv`.
# 3. Copy the new figures to `../figures/` for the website.
