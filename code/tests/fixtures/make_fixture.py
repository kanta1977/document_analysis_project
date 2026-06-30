#!/usr/bin/env python
"""Generate tests/fixtures/mini_corpus.zip — a tiny synthetic corpus.

Deterministic (seed below). ~60 fake posts matching the real Webis-TLDR-17
schema, spread across the subreddits in configs/project.yaml, plus edge cases:
missing title, empty summary, missing summary, non-ASCII text, and one
malformed (non-JSON) line.

Run from repo root:  python tests/fixtures/make_fixture.py
"""

from __future__ import annotations

import json
import random
import zipfile
from pathlib import Path

SEED = 1234
SUBREDDITS = [
    # mirror configs/project.yaml buckets
    "politics", "PoliticalDiscussion", "worldnews",
    "depression", "offmychest", "Anxiety",
    "legaladvice", "personalfinance", "relationships",
]

LOREM = (
    "I have been thinking about this for a while and I guess maybe it could be "
    "that things will perhaps change soon. My friend John from Berlin said the "
    "government might act, but honestly I am not sure what to do about my rent "
    "and my landlord. I feel like I should ask for advice here because "
).split()


def make_post(rng: random.Random, i: int, sub: str) -> dict:
    n_content = rng.randint(80, 250)
    n_summary = rng.randint(8, 30)
    content = " ".join(rng.choices(LOREM, k=n_content))
    summary = " ".join(rng.choices(LOREM, k=n_summary))
    return {
        "author": f"user_{i}",
        "body": content + " tl;dr " + summary,
        "normalizedBody": content + " tl;dr " + summary,
        "content": content,
        "content_len": n_content,
        "summary": summary,
        "summary_len": n_summary,
        "id": f"t3_fake{i:04d}",
        "subreddit": sub,
        "subreddit_id": f"t5_{abs(hash(sub)) % 99999:05d}",
        "title": f"Fake post {i} in r/{sub}",
    }


def main() -> None:
    rng = random.Random(SEED)
    lines: list[str] = []
    i = 0
    for sub in SUBREDDITS:
        for _ in range(6):  # 54 regular posts
            lines.append(json.dumps(make_post(rng, i, sub)))
            i += 1

    # Edge cases
    p = make_post(rng, i := i + 1, "politics"); p["title"] = None
    lines.append(json.dumps(p))                                   # missing title
    p = make_post(rng, i := i + 1, "depression"); p["summary"] = ""
    lines.append(json.dumps(p))                                   # empty summary
    p = make_post(rng, i := i + 1, "Anxiety"); del p["summary"]
    lines.append(json.dumps(p))                                   # missing summary
    p = make_post(rng, i := i + 1, "worldnews")
    p["content"] = "Üñïcødé tëst — naïve café résumé 日本語 " + p["content"]
    lines.append(json.dumps(p, ensure_ascii=False))               # non-ASCII
    p = make_post(rng, i := i + 1, "relationships")
    p["content"] = " ".join(["word"] * 20)                        # too short for filters
    lines.append(json.dumps(p))
    lines.append('{"author": "broken", THIS IS NOT JSON')         # malformed line
    lines.append("")                                              # blank line

    out = Path(__file__).parent / "mini_corpus.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("corpus-webis-tldr-17.json", "\n".join(lines) + "\n")
    print(f"Wrote {out} with {len(lines)} lines")


if __name__ == "__main__":
    main()
