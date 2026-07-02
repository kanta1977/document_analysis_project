#!/usr/bin/env python
"""Step 4 — AI baseline summaries (OpenAI mini).

For a stratified subsample of posts, ask an OpenAI *mini* model for a neutral
one-to-two sentence summary of each post body. These summaries are a fixed
REFERENCE POINT ("what a plain summary looks like"), NOT a gold standard and
NOT an object of study. We compare the human TL;DR against this reference.

Design choices (do not change casually — they keep the baseline reproducible):
  * one fixed mini model, temperature 0
  * one neutral prompt for every subreddit (no per-community wording)
  * NO instruction to keep the first person — we want the model's own default,
    so that "first person survives in human TL;DRs" is measured, not imposed
  * stratified subsample: up to N posts per subreddit (default 200)

The API key is read from the OPENAI_API_KEY environment variable. It is never
written to disk or hard-coded.

Run:
    export OPENAI_API_KEY=sk-...          # (Windows PowerShell: $env:OPENAI_API_KEY="sk-...")
    python scripts/04_ai_baseline.py
    python scripts/04_ai_baseline.py --per-subreddit 200 --model gpt-4o-mini
    python scripts/04_ai_baseline.py --dry-run          # no API calls; checks the sample
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tldr_audit.corpus import load_config  # noqa: E402

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


def stratified_subsample(posts, per_subreddit, seed=13):
    """Keep up to `per_subreddit` posts for each subreddit, deterministically."""
    import random

    rng = random.Random(seed)
    by_sub: dict[str, list] = {}
    for p in posts:
        by_sub.setdefault(p.get("subreddit"), []).append(p)
    picked = []
    for sub, group in sorted(by_sub.items()):
        rng.shuffle(group)
        picked.extend(group[:per_subreddit])
    return picked


def summarize_one(client, model, post_text):
    """One post -> one summary string (temperature 0)."""
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(post=post_text)},
        ],
    )
    return resp.choices[0].message.content.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/project.yaml")
    ap.add_argument("--input", default=None, help="default: sample.output from config")
    ap.add_argument("--output", default="data/interim/ai_summaries.jsonl")
    ap.add_argument("--model", default="gpt-4o-mini", help="an OpenAI mini model")
    ap.add_argument("--per-subreddit", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true",
                    help="build the subsample and print counts, but call no API")
    args = ap.parse_args()

    cfg = load_config(args.config)
    src = Path(args.input or cfg["sample"]["output"])
    if not src.exists():
        raise FileNotFoundError(
            f"Input sample not found: {src}\nRun scripts/01 and 02 first."
        )

    posts = [json.loads(line) for line in open(src, encoding="utf-8")]
    picked = stratified_subsample(posts, args.per_subreddit)

    # report the plan
    from collections import Counter
    counts = Counter(p.get("subreddit") for p in picked)
    print(f"Subsample: {len(picked):,} posts across {len(counts)} subreddits "
          f"(up to {args.per_subreddit} each)")
    for sub, n in sorted(counts.items()):
        print(f"  {sub:20s} {n}")

    if args.dry_run:
        print("\n--dry-run: no API calls made.")
        return

    # resume support: skip ids already written
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out.exists():
        for line in open(out, encoding="utf-8"):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
        if done:
            print(f"Resuming: {len(done):,} summaries already present, will skip them.")

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit(
            "OPENAI_API_KEY is not set.\n"
            "  PowerShell:  $env:OPENAI_API_KEY=\"sk-...\"\n"
            "  bash:        export OPENAI_API_KEY=sk-..."
        )
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("openai package missing. Run: pip install openai")

    client = OpenAI()
    written = 0
    with open(out, "a", encoding="utf-8") as f:
        for i, p in enumerate(picked):
            pid = p.get("id")
            if pid in done:
                continue
            try:
                summary = summarize_one(client, args.model, p.get("content", ""))
            except Exception as e:  # keep going; log and continue
                print(f"  ! error on {pid}: {e}; retrying once in 5s", flush=True)
                time.sleep(5)
                try:
                    summary = summarize_one(client, args.model, p.get("content", ""))
                except Exception as e2:
                    print(f"  !! skipping {pid}: {e2}", flush=True)
                    continue
            f.write(json.dumps({
                "id": pid,
                "subreddit": p.get("subreddit"),
                "_bucket": p.get("_bucket"),
                "content": p.get("content"),
                "summary": summary,          # the AI summary goes in "summary"
                "human_tldr": p.get("summary"),
            }, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
            if written % 100 == 0:
                print(f"  ... {written:,} summaries", flush=True)

    print(f"\nWrote {written:,} new AI summaries -> {out}")
    print(f"Model={args.model}, temperature=0. This is a fixed reference point.")


if __name__ == "__main__":
    main()
