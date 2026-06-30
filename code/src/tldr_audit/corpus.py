"""Streaming access to the Webis-TLDR-17 corpus.

The corpus ships as a 3.1 GB zip containing one ~19 GB JSON-lines file.
We never extract it: ZipFile.open() gives a file object we can wrap and
iterate line by line, so a full pass costs ~zero disk and modest time.

Schema per line (all nullable):
  author, body, normalizedBody, content, content_len,
  summary, summary_len, id, subreddit, subreddit_id, title
`content` is the post without the TL;DR; `summary` is the author's TL;DR.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Iterator
from pathlib import Path

import yaml


def load_config(path: str | Path = "configs/project.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_member(zf: zipfile.ZipFile, member: str | None) -> str:
    """Find the jsonl member inside the zip (name in config may be stale)."""
    names = zf.namelist()
    if member and member in names:
        return member
    candidates = [n for n in names if n.endswith((".json", ".jsonl"))]
    if not candidates:
        raise FileNotFoundError(f"No json member found in zip; contents: {names[:10]}")
    return max(candidates, key=lambda n: zf.getinfo(n).file_size)


def iter_posts(
    zip_path: str | Path,
    member: str | None = None,
    limit: int | None = None,
) -> Iterator[dict]:
    """Yield posts one by one, streaming from inside the zip.

    Usage:
        for post in iter_posts("data/raw/corpus-webis-tldr-17.zip", limit=1000):
            ...
    """
    with zipfile.ZipFile(zip_path) as zf:
        name = _resolve_member(zf, member)
        with zf.open(name) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
            for i, line in enumerate(text):
                if limit is not None and i >= limit:
                    return
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # rare malformed lines; count separately if it matters


def word_count(text: str | None) -> int:
    return len(text.split()) if text else 0


def passes_filters(post: dict, filters: dict) -> bool:
    cw = word_count(post.get("content"))
    sw = word_count(post.get("summary"))
    if not (filters["content_words_min"] <= cw <= filters["content_words_max"]):
        return False
    if not (filters["summary_words_min"] <= sw <= filters["summary_words_max"]):
        return False
    if filters.get("require_title") and not post.get("title"):
        return False
    return True
