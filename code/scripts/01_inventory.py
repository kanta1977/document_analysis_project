#!/usr/bin/env python
"""Step 1 — Corpus inventory.

One streaming pass over the full corpus. Counts posts per subreddit and
basic length stats. Run this FIRST after downloading the zip; its output
decides the final subreddit selection in configs/project.yaml.

Run:  python scripts/01_inventory.py            (full pass, ~30-60 min)
      python scripts/01_inventory.py --limit 100000   (smoke test, seconds)

Output: results/tables/subreddit_inventory.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tldr_audit.corpus import iter_posts, load_config, word_count  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap lines for a smoke test")
    ap.add_argument("--config", default="configs/project.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    stats: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "content_words": 0, "summary_words": 0, "has_title": 0}
    )

    n_total = 0
    for post in iter_posts(cfg["corpus"]["zip_path"], cfg["corpus"].get("member"), args.limit):
        sub = post.get("subreddit") or "_unknown"
        s = stats[sub]
        s["n"] += 1
        s["content_words"] += word_count(post.get("content"))
        s["summary_words"] += word_count(post.get("summary"))
        s["has_title"] += bool(post.get("title"))
        n_total += 1
        if n_total % 500_000 == 0:
            print(f"  ... {n_total:,} posts, {len(stats):,} subreddits", flush=True)

    out = Path("results/tables/subreddit_inventory.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("subreddit,n_posts,avg_content_words,avg_summary_words,share_with_title\n")
        for sub, s in sorted(stats.items(), key=lambda kv: -kv[1]["n"]):
            n = s["n"]
            f.write(
                f"{sub},{n},{s['content_words']/n:.1f},"
                f"{s['summary_words']/n:.1f},{s['has_title']/n:.3f}\n"
            )

    print(f"\nDone. {n_total:,} posts across {len(stats):,} subreddits -> {out}")
    print("Next: check the proposed subreddits in configs/project.yaml against this table.")


if __name__ == "__main__":
    main()
