#!/usr/bin/env python
"""Step 2 — Stratified sample.

Reservoir-samples `per_subreddit` posts for each configured subreddit in a
single streaming pass (memory stays bounded), applying quality filters.
Deterministic given the seed in configs/project.yaml.

Run:  python scripts/02_sample.py
Output: data/interim/sample.jsonl  (+ a small summary printed)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tldr_audit.corpus import iter_posts, load_config, passes_filters  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/project.yaml")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    rng = random.Random(cfg["seed"])
    k = cfg["sample"]["per_subreddit"]
    filters = cfg["sample"]["filters"]

    # subreddit -> bucket, case-insensitive match
    target = {
        sub.lower(): bucket
        for bucket, subs in cfg["subreddits"].items()
        for sub in subs
    }
    reservoirs: dict[str, list[dict]] = {s: [] for s in target}
    seen: dict[str, int] = {s: 0 for s in target}

    for post in iter_posts(cfg["corpus"]["zip_path"], cfg["corpus"].get("member"), args.limit):
        sub = (post.get("subreddit") or "").lower()
        if sub not in target or not passes_filters(post, filters):
            continue
        seen[sub] += 1
        res = reservoirs[sub]
        if len(res) < k:
            res.append(post)
        else:
            j = rng.randrange(seen[sub])
            if j < k:
                res[j] = post

    out = Path(cfg["sample"]["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(out, "w") as f:
        for sub, posts in reservoirs.items():
            for p in posts:
                p["_bucket"] = target[sub]
                f.write(json.dumps(p) + "\n")
                n_written += 1

    print(f"Wrote {n_written:,} posts -> {out}")
    for sub in sorted(target):
        print(f"  r/{sub:<22} eligible: {seen[sub]:>8,}   sampled: {len(reservoirs[sub]):>6,}")
    short = [s for s in target if len(reservoirs[s]) < k]
    if short:
        print(f"\nWARNING: under-filled subreddits: {short} — revisit config or filters.")


if __name__ == "__main__":
    main()
