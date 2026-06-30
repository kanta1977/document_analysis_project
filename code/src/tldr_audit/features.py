"""Linguistic features for RQ1/RQ2 (one pure function per feature).

All functions take plain strings (or a post dict for the convenience wrapper)
and return numbers/dicts; no I/O except the lazy lexicon/model loaders.

Sentiment uses the vaderSentiment package (lexicon ships with the wheel, no
nltk download step needed -> reproducible installs).
NER uses spaCy en_core_web_sm, loaded lazily; install with
`python -m spacy download en_core_web_sm`.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

_WORD_RE = re.compile(r"[a-z']+")

FIRST_PERSON = {
    "i", "me", "my", "mine", "myself",
    "we", "us", "our", "ours", "ourselves",
}


def tokenize(text: str | None) -> list[str]:
    """Lowercase word tokens (letters + apostrophes only)."""
    return _WORD_RE.findall(text.lower()) if text else []


# ---------------------------------------------------------------- compression

def compression_ratio(content: str | None, summary: str | None) -> float | None:
    """len(summary) / len(content) in word tokens; None if content is empty."""
    c, s = tokenize(content), tokenize(summary)
    return len(s) / len(c) if c else None


# ------------------------------------------------------------------ sentiment

@lru_cache(maxsize=1)
def _vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


def sentiment(text: str | None) -> float:
    """VADER compound score in [-1, 1]; 0.0 for empty text."""
    if not text or not text.strip():
        return 0.0
    return _vader().polarity_scores(text)["compound"]


# ------------------------------------------------------------- pronouns/hedges

def first_person_rate(text: str | None) -> float | None:
    """Share of word tokens that are first-person pronouns; None if no tokens."""
    toks = tokenize(text)
    if not toks:
        return None
    return sum(t in FIRST_PERSON for t in toks) / len(toks)


@lru_cache(maxsize=4)
def load_hedges(path: str = "configs/lexicons/hedges.txt") -> tuple[frozenset, tuple]:
    """Return (single-word hedges, multi-word hedge phrases)."""
    singles, phrases = set(), []
    for line in Path(path).read_text().splitlines():
        line = line.strip().lower()
        if not line or line.startswith("#"):
            continue
        if " " in line:
            phrases.append(tuple(line.split()))
        else:
            singles.add(line)
    return frozenset(singles), tuple(phrases)


def hedge_rate(text: str | None, lexicon_path: str = "configs/lexicons/hedges.txt") -> float | None:
    """Hedge occurrences per word token (phrases counted once per match)."""
    toks = tokenize(text)
    if not toks:
        return None
    singles, phrases = load_hedges(lexicon_path)
    n = sum(t in singles for t in toks)
    for ph in phrases:
        L = len(ph)
        n += sum(tuple(toks[i:i + L]) == ph for i in range(len(toks) - L + 1))
    return n / len(toks)


# ------------------------------------------------------------------------ NER

@lru_cache(maxsize=1)
def _nlp():
    import spacy
    return spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "tagger"])


def ner_survival(content: str | None, summary: str | None) -> dict:
    """Which named entities in the content reappear (verbatim, case-insensitive)
    in the summary.

    Returns {"n_entities": int, "n_survived": int, "survival_rate": float|None,
             "by_label": {label: [n, n_survived]}}.
    """
    out: dict = {"n_entities": 0, "n_survived": 0, "survival_rate": None, "by_label": {}}
    if not content:
        return out
    doc = _nlp()(content)
    summary_low = (summary or "").lower()
    by_label: dict[str, list[int]] = {}
    seen = set()
    for ent in doc.ents:
        key = (ent.text.lower(), ent.label_)
        if key in seen:        # count each distinct entity once
            continue
        seen.add(key)
        survived = ent.text.lower() in summary_low
        rec = by_label.setdefault(ent.label_, [0, 0])
        rec[0] += 1
        rec[1] += survived
        out["n_entities"] += 1
        out["n_survived"] += survived
    out["by_label"] = by_label
    if out["n_entities"]:
        out["survival_rate"] = out["n_survived"] / out["n_entities"]
    return out


# ----------------------------------------------------------------- aggregator

def features_for_post(
    post: dict,
    lexicon_path: str = "configs/lexicons/hedges.txt",
    with_ner: bool = False,
) -> dict:
    """Flat feature dict for one post (content vs. author summary).

    Groups of columns:
      * length / compression   - how much was dropped
      * summariness            - is it even a summary? (vocabulary reuse)
      * sentiment              - how the mood moves
      * first-person voice     - does "I" survive?
      * surface-form flags     - ingredients for the classifier
      * tldr_type              - the heuristic speech-act label
    NER is OFF by default (needs the spaCy model); pass with_ner=True to add it.
    """
    content, summary = post.get("content"), post.get("summary")

    # compute the building blocks once
    c_words = len(tokenize(content))
    s_words = len(tokenize(summary))
    sent_c = sentiment(content)
    sent_s = sentiment(summary)
    fp_c = first_person_rate(content)
    fp_s = first_person_rate(summary)
    novelty = summary_novelty(content, summary)

    # how much length is dropped (0 = nothing dropped, 1 = everything)
    word_drop = 1 - (s_words / c_words) if c_words else None
    if word_drop is not None:
        word_drop = max(0.0, min(1.0, word_drop))

    # first-person voice: drop and a simple "did I disappear?" flag
    fp_drop = (fp_c - fp_s) if (fp_c is not None and fp_s is not None) else None

    row = {
        "id": post.get("id"),
        "subreddit": post.get("subreddit"),
        "bucket": post.get("_bucket"),

        # --- length / compression ---
        "content_words": c_words,
        "summary_words": s_words,
        "compression_ratio": (s_words / c_words) if c_words else None,
        "word_drop_rate": word_drop,

        # --- is it even a summary? (vocabulary reuse) ---
        "summary_novelty": novelty,
        "novel_bigram_rate": novel_ngram_rate(content, summary, n=2),

        # --- sentiment ---
        "sentiment_content": sent_c,
        "sentiment_summary": sent_s,
        "sentiment_shift": sent_s - sent_c,
        "sentiment_flip": (sent_c > 0.5 and sent_s < -0.5)
        or (sent_c < -0.5 and sent_s > 0.5),

        # --- first-person voice ---
        "first_person_content": fp_c,
        "first_person_summary": fp_s,
        "first_person_drop": fp_drop,
        "i_disappears": bool(fp_c and not fp_s),

        # --- surface-form flags (ingredients for the classifier) ---
        "has_question_mark": has_question_mark(summary),
        "has_second_person": has_second_person(summary),
        "has_advice_marker": has_advice_marker(summary),
        "has_joke_marker": has_joke_marker(summary),

        # --- hedges (kept for backward-compat; cheap) ---
        "hedge_rate_content": hedge_rate(content, lexicon_path),
        "hedge_rate_summary": hedge_rate(summary, lexicon_path),

        # --- the heuristic speech-act label ---
        "tldr_type": classify_tldr(summary, novelty),
    }
    if with_ner:
        ner = ner_survival(content, summary)
        row["n_entities"] = ner["n_entities"]
        row["ner_survival_rate"] = ner["survival_rate"]
    return row


# ------------------------------------------------------- summariness (proxy)

# A small stopword list: words so common they carry no topical meaning.
# We ignore them when comparing TL;DR to post, otherwise every TL;DR would
# look like it "reuses" the post's words just by containing "the" and "I".
_STOPWORDS = frozenset(
    "the a an and or but if of to in on for with at by from as is are was were "
    "be been being am i you he she it we they me him her us them my your his "
    "its our their this that these those not no do does did done have has had "
    "will would can could should may might must just so very really there "
    "what when where which who how all any some out up down about into over "
    "after before again then than too only own same s t don now get got".split()
)


def summary_novelty(content: str | None, summary: str | None) -> float | None:
    """Share of the TL;DR's *meaningful* words that do NOT appear in the post.

    ELI5: imagine highlighting every word of the TL;DR that you can also find
    somewhere in the post. A faithful summary is mostly highlighted (novelty
    near 0). A joke, a comeback, or a brand-new question is mostly
    un-highlighted (novelty near 1). This is our cheap, no-API "is this even
    a summary?" detector — used as a continuous variable, not a hard filter.

    Returns None when the TL;DR has no meaningful words at all (so we can't say).
    """
    s_toks = set(tokenize(summary)) - _STOPWORDS
    if not s_toks:
        return None
    c_toks = set(tokenize(content)) - _STOPWORDS
    return len(s_toks - c_toks) / len(s_toks)

# ------------------------------------------------- abstractiveness (n-grams)

def _ngrams(tokens, n):
    """All overlapping n-grams (n consecutive words) of a token list."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def novel_ngram_rate(content, summary, n=2):
    """Share of the TL;DR's n-grams (default bigrams) that never appear in the post.

    ELI5: `summary_novelty` asked "are the *words* new?". This asks a stricter
    question about *phrases*. An EXTRACTIVE summary copies chunks of the post
    ("...starting a petition to get us evicted") -> most of its 2-word phrases
    are found in the post -> low novel-ngram rate. An ABSTRACTIVE summary
    rewrites the idea in fresh wording even if it reuses individual words ->
    high novel-ngram rate. This is the standard way summarization research
    separates "copy-paste" from "rephrase" (e.g. Grusky et al. 2018, "Newsroom").

    n=1 ~ word novelty; n=2 (bigrams) is the usual abstractiveness measure.
    Returns None if the TL;DR is too short to form a single n-gram.
    """
    s = tokenize(summary)
    if len(s) < n:
        return None
    c_set = set(_ngrams(tokenize(content), n))
    s_grams = _ngrams(s, n)
    novel = sum(g not in c_set for g in s_grams)
    return novel / len(s_grams)


# ----------------------------------------- TL;DR type (heuristic classifier)
#
# This is the cheap, no-training "what KIND of TL;DR is this?" labeller — the
# (a) heuristic typology. It looks only at simple surface signals of the TL;DR
# plus the novelty score, and returns ONE label. It is good enough to DESCRIBE
# how each community uses the slot; it is NOT a validated classifier (no
# inter-annotator agreement, no learned model). Overlaps are resolved by a
# fixed priority order and that is a documented limitation.

SECOND_PERSON = {"you", "your", "yours", "yourself", "u", "ur"}

# words that signal the author is telling someone what to do
_ADVICE_RE = re.compile(
    r"\b(should|shouldn't|ought to|need to|needs to|have to|has to|must|"
    r"make sure|don't|do not|avoid|try to|recommend|suggest|advice|consider)\b"
)
# light markers of a joke / sarcasm / aside (weak signal on purpose)
_JOKE_RE = re.compile(r"(/s\b|\blol\b|\blmao\b|\bhaha+\b|\bjk\b|\brofl\b|ironic)")


def has_question_mark(summary: str | None) -> bool:
    """TL;DR contains a question mark."""
    return "?" in (summary or "")


def has_second_person(summary: str | None) -> bool:
    """TL;DR addresses 'you' (talking TO someone, not summarizing oneself)."""
    return any(t in SECOND_PERSON for t in tokenize(summary))


def has_advice_marker(summary: str | None) -> bool:
    """TL;DR uses advice-giving language (should / need to / avoid ...)."""
    return bool(_ADVICE_RE.search((summary or "").lower()))


def has_joke_marker(summary: str | None) -> bool:
    """TL;DR has a light joke / sarcasm marker (lol, /s, haha ...)."""
    return bool(_JOKE_RE.search((summary or "").lower()))


def classify_tldr(summary: str | None, novelty: float | None = None) -> str:
    """Rule-based speech-act label for one TL;DR.

    Priority order (first match wins):
        1. has '?'                  -> "question"
        2. joke / sarcasm marker    -> "reaction"
        3. advice-giving language   -> "advice"
        4. novelty >= 0.8           -> "reaction"   (brand-new text, not a summary)
        5. otherwise                -> "summary"

    ELI5: most TL;DRs are summaries. But some are really a question, a joke, or
    a piece of advice wearing the TL;DR label. This function spots those by a
    few obvious tells. A joke phrased as a question is labelled "question"
    because '?' is checked first — that ordering is a deliberate, documented
    simplification, not a mistake.
    """
    s = summary or ""
    if has_question_mark(s):
        return "question"
    if has_joke_marker(s):
        return "reaction"
    if has_advice_marker(s):
        return "advice"
    if novelty is not None and novelty >= 0.8:
        return "reaction"
    return "summary"
