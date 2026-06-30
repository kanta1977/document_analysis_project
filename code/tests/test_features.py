"""Unit tests for tldr_audit.features (pure functions, fixture-driven)."""

import pytest

from tldr_audit import features as F
from tldr_audit.corpus import iter_posts

LEX = "configs/lexicons/hedges.txt"


def test_tokenize():
    assert F.tokenize("Hello, World! don't") == ["hello", "world", "don't"]
    assert F.tokenize(None) == []
    assert F.tokenize("") == []


def test_compression_ratio():
    assert F.compression_ratio("a b c d", "a b") == 0.5
    assert F.compression_ratio("", "a") is None
    assert F.compression_ratio(None, None) is None


def test_sentiment_signs():
    assert F.sentiment("I love this, it is wonderful and great") > 0.5
    assert F.sentiment("This is horrible, I hate everything") < -0.5
    assert F.sentiment("") == 0.0
    assert F.sentiment(None) == 0.0


def test_first_person_rate():
    assert F.first_person_rate("I like my dog") == 0.5
    assert F.first_person_rate("the dog barks") == 0.0
    assert F.first_person_rate("") is None


def test_hedge_rate_singles_and_phrases():
    assert F.hedge_rate("maybe yes", LEX) == 0.5
    # "kind of" is a phrase: counted once over 4 tokens
    assert F.hedge_rate("it is kind of", LEX) == pytest.approx(0.25)
    assert F.hedge_rate("dog cat tree", LEX) == 0.0
    assert F.hedge_rate("", LEX) is None


def test_load_hedges_has_reasonable_size():
    singles, phrases = F.load_hedges(LEX)
    assert len(singles) >= 40
    assert ("kind", "of") in phrases


def _has_spacy_model():
    try:
        F._nlp()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_spacy_model(), reason="en_core_web_sm not installed")
def test_ner_survival():
    r = F.ner_survival("John lives in Berlin and works for Google.", "John moved to Berlin.")
    assert r["n_entities"] >= 3
    assert 0 < r["survival_rate"] < 1
    assert r["n_survived"] >= 2


def test_ner_survival_empty():
    r = F.ner_survival("", "x")
    assert r["n_entities"] == 0 and r["survival_rate"] is None


def test_features_for_post_on_fixture(fixture_zip):
    post = next(iter_posts(fixture_zip, limit=1))
    row = F.features_for_post(post, LEX, with_ner=False)
    assert row["id"] == post["id"]
    assert row["content_words"] > 0
    assert 0 < row["compression_ratio"] < 1
    assert 0 <= row["hedge_rate_content"] <= 1
    assert -1 <= row["sentiment_content"] <= 1


def test_summary_novelty():
    # TL;DR reuses the post's words -> novelty 0
    assert F.summary_novelty("my landlord raised the rent", "landlord raised rent") == 0.0
    # TL;DR is a completely different joke -> novelty 1
    assert F.summary_novelty("my landlord raised the rent", "monkey knife fights") == 1.0
    # half the meaningful words are new -> 0.5
    assert F.summary_novelty("cat dog", "cat banana") == 0.5
    # empty / stopword-only TL;DR -> None (we cannot judge)
    assert F.summary_novelty("cat dog", "the and of") is None
    assert F.summary_novelty("cat dog", "") is None


def test_novel_ngram_rate():
    # TL;DR copies a phrase verbatim -> bigram novelty 0
    assert F.novel_ngram_rate("the cat sat on the mat", "the cat sat") == 0.0
    # same words, reordered so no shared bigram -> novelty 1
    assert F.novel_ngram_rate("cat dog bird", "bird dog cat") == 1.0
    # too short to form a bigram -> None
    assert F.novel_ngram_rate("cat dog bird", "cat") is None
    # unigram mode behaves like word membership
    assert F.novel_ngram_rate("cat dog", "cat fish", n=1) == 0.5


# --- heuristic TL;DR classifier (added with the no-LLM pipeline) ------------

def test_surface_flags():
    assert F.has_question_mark("is this ok?") is True
    assert F.has_question_mark("this is fine") is False
    assert F.has_second_person("you should leave") is True
    assert F.has_second_person("i should leave") is False
    assert F.has_advice_marker("you need to call a lawyer") is True
    assert F.has_joke_marker("sure buddy /s") is True
    assert F.has_joke_marker("my rent went up") is False


def test_classify_tldr_priority():
    # 1. question mark wins first
    assert F.classify_tldr("wait, is that legal?", novelty=1.0) == "question"
    # 2. joke marker
    assert F.classify_tldr("totally not mad lol", novelty=0.1) == "reaction"
    # 3. advice language
    assert F.classify_tldr("you should document everything", novelty=0.1) == "advice"
    # 4. high novelty with none of the above -> reaction (not a summary)
    assert F.classify_tldr("monkey knife fights forever", novelty=0.95) == "reaction"
    # 5. plain faithful summary
    assert F.classify_tldr("landlord raised my rent without notice", novelty=0.1) == "summary"


def test_features_for_post_has_new_columns(fixture_zip):
    post = next(iter_posts(fixture_zip, limit=1))
    row = F.features_for_post(post, LEX, with_ner=False)
    for col in ["word_drop_rate", "sentiment_shift", "sentiment_flip",
                "first_person_drop", "i_disappears", "has_question_mark",
                "has_advice_marker", "tldr_type"]:
        assert col in row
    assert row["tldr_type"] in {"summary", "question", "advice", "reaction"}
