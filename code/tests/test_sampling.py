"""Run scripts/02_sample.py end-to-end against the fixture corpus.

Checks: determinism (same seed -> same sample), per-subreddit cap respected,
filters applied, bucket labels attached.
"""

import json
import subprocess
import sys

import yaml


def run_sample(repo_root, fixture_zip, tmp_path, seed=2026, k=3):
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((repo_root / "configs" / "project.yaml").read_text())
    cfg["seed"] = seed
    cfg["corpus"]["zip_path"] = str(fixture_zip)
    cfg["sample"]["per_subreddit"] = k
    out = tmp_path / "sample.jsonl"
    cfg["sample"]["output"] = str(out)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.dump(cfg))

    res = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "02_sample.py"), "--config", str(cfg_path)],
        capture_output=True, text=True, cwd=repo_root,
    )
    assert res.returncode == 0, res.stderr
    return [json.loads(line) for line in out.read_text().splitlines()], cfg


def test_sample_respects_cap_and_filters(repo_root, fixture_zip, tmp_path):
    posts, cfg = run_sample(repo_root, fixture_zip, tmp_path, k=3)
    per_sub = {}
    for p in posts:
        per_sub[p["subreddit"].lower()] = per_sub.get(p["subreddit"].lower(), 0) + 1
    assert all(n <= 3 for n in per_sub.values())
    f = cfg["sample"]["filters"]
    for p in posts:
        cw = len(p["content"].split())
        sw = len(p["summary"].split())
        assert f["content_words_min"] <= cw <= f["content_words_max"]
        assert f["summary_words_min"] <= sw <= f["summary_words_max"]


def test_sample_attaches_bucket(repo_root, fixture_zip, tmp_path):
    posts, cfg = run_sample(repo_root, fixture_zip, tmp_path, k=3)
    buckets = set(cfg["subreddits"])
    assert posts, "fixture should yield a non-empty sample"
    assert all(p["_bucket"] in buckets for p in posts)


def test_sample_is_deterministic(repo_root, fixture_zip, tmp_path):
    a, _ = run_sample(repo_root, fixture_zip, tmp_path / "a", seed=2026)
    b, _ = run_sample(repo_root, fixture_zip, tmp_path / "b", seed=2026)
    assert [p["id"] for p in a] == [p["id"] for p in b]
