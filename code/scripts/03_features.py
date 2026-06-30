#!/usr/bin/env python
"""Step 3 — Feature extraction (human TL;DR only; no LLM).

data/interim/sample.jsonl を読み込み、各投稿ごとに特徴量を計算して、
data/processed/features.parquet に保存するスクリプト。

このスクリプト自体は特徴量の中身を計算するのではなく、
src/tldr_audit/features.py の features_for_post() を呼び出す役割。

features_for_post() が返す主な列:
  長さ/圧縮 : content_words, summary_words, compression_ratio, word_drop_rate
  要約らしさ: summary_novelty, novel_bigram_rate
  感情      : sentiment_content/summary, sentiment_shift, sentiment_flip
  一人称    : first_person_content/summary, first_person_drop, i_disappears
  表層フラグ: has_question_mark, has_second_person, has_advice_marker, has_joke_marker
  分類器    : tldr_type  (summary / question / advice / reaction)

Run:
    python scripts/03_features.py
    python scripts/03_features.py --with-ner      # spaCy が入っているときだけ
    python scripts/03_features.py --limit 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# src/ 以下の自作モジュールを import できるようにする
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tldr_audit.corpus import load_config  # noqa: E402
from tldr_audit.features import features_for_post  # noqa: E402


# このスクリプトで最終的に存在していてほしい列
# 実際にこれらを作るのは features.py の features_for_post()
EXPECTED_NEW_COLUMNS = [
    # 長さ・圧縮関連
    "content_words",
    "summary_words",
    "compression_ratio",
    "word_drop_rate",

    # TL;DR側の要約らしさ
    "summary_novelty",

    # sentiment 関連
    "sentiment_content",
    "sentiment_summary",
    "sentiment_shift",
    "sentiment_flip",

    # 一人称 "I" 関連
    "first_person_content",
    "first_person_summary",
    "first_person_drop",
    "i_disappears",

    # TL;DR usage indicators (分類器の材料になるフラグ)
    "has_question_mark",
    "has_second_person",
    "has_advice_marker",
    "has_joke_marker",

    # 発話行為ラベル(ヒューリスティック分類器の出力)
    "tldr_type",
]


def print_column_check(df: pd.DataFrame) -> None:
    """期待している特徴量の列がちゃんと作られているか確認する。"""
    missing = [c for c in EXPECTED_NEW_COLUMNS if c not in df.columns]
    if missing:
        print("\nWARNING: Some expected feature columns are missing:")
        for c in missing:
            print(f"  - {c}")
        print(
            "\nThese columns need to be returned by "
            "src/tldr_audit/features.py::features_for_post()."
        )
    else:
        print("\nAll expected new feature columns are present.")


def print_bucket_summary(df: pd.DataFrame) -> None:
    """bucket ごとの平均値を表示する(妥当性のざっくり確認)。"""
    cols = [
        "compression_ratio",
        "word_drop_rate",
        "summary_novelty",
        "sentiment_shift",
        "sentiment_flip",
        "i_disappears",
        "has_question_mark",
        "has_second_person",
        "has_advice_marker",
        "has_joke_marker",
    ]
    available = [c for c in cols if c in df.columns]
    if "bucket" not in df.columns or not available:
        return
    print("\nBucket-level feature means:")
    print(df.groupby("bucket")[available].mean(numeric_only=True).round(3))


def print_type_distribution(df: pd.DataFrame) -> None:
    """tldr_type の構成比を bucket ごとに表示する。

    「各スレッドが TL;DR枠を何に使っているか」を一番直接に見せる表。
    """
    if "tldr_type" not in df.columns or "bucket" not in df.columns:
        return
    print("\nTL;DR type mix by bucket (share of posts):")
    mix = (
        df.groupby("bucket")["tldr_type"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .round(3)
    )
    print(mix)


def print_subreddit_summary(df: pd.DataFrame, top_n: int = 20) -> None:
    """subreddit ごとの平均値を表示する(サンプル数上位だけ)。"""
    cols = [
        "compression_ratio",
        "summary_novelty",
        "sentiment_shift",
        "i_disappears",
        "has_question_mark",
        "has_advice_marker",
    ]
    available = [c for c in cols if c in df.columns]
    if "subreddit" not in df.columns or not available:
        return
    summary = (
        df.groupby("subreddit")
        .agg(n=("subreddit", "size"), **{c: (c, "mean") for c in available})
        .sort_values("n", ascending=False)
        .head(top_n)
    )
    print(f"\nTop {top_n} subreddits by sample size:")
    print(summary.round(3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/project.yaml")
    ap.add_argument("--input", default=None, help="default: sample.output from config")
    ap.add_argument("--output", default="data/processed/features.parquet")
    ap.add_argument("--limit", type=int, default=None)
    # NER は重い & spaCy モデルが要るので、既定では計算しない。
    # 使いたいときだけ --with-ner を付ける。
    ap.add_argument("--with-ner", action="store_true",
                    help="add NER survival columns (needs en_core_web_sm)")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # 入力ファイルの場所(指定が無ければ config の sample.output)
    src = Path(args.input or cfg["sample"]["output"])

    # hedge words の辞書ファイル("maybe" などの曖昧表現用)
    lexicon = cfg["features"]["hedge_lexicon"]

    if not src.exists():
        raise FileNotFoundError(
            f"Input sample file not found: {src}\n"
            "Run scripts/01_inventory.py and scripts/02_sample.py first."
        )

    rows = []
    with open(src, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit is not None and i >= args.limit:
                break
            post = json.loads(line)
            rows.append(
                features_for_post(post, lexicon, with_ner=args.with_ner)
            )
            if (i + 1) % 2000 == 0:
                print(f"  ... {i + 1:,} posts", flush=True)

    df = pd.DataFrame(rows)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    print(f"\nWrote {len(df):,} rows x {len(df.columns)} cols -> {out}")

    print_column_check(df)
    print_bucket_summary(df)
    print_type_distribution(df)
    print_subreddit_summary(df)


if __name__ == "__main__":
    main()
