"""Tests for tldr_audit.corpus: streaming reader and filters."""

from tldr_audit.corpus import iter_posts, passes_filters, word_count

FILTERS = {
    "content_words_min": 100,
    "content_words_max": 600,
    "summary_words_min": 10,
    "summary_words_max": 100,
    "require_title": False,
}


def test_iter_posts_streams_all_valid_lines(fixture_zip):
    posts = list(iter_posts(fixture_zip))
    # 54 regular + 5 edge-case posts; malformed + blank lines skipped silently
    assert len(posts) == 59
    assert all(isinstance(p, dict) for p in posts)


def test_iter_posts_respects_limit(fixture_zip):
    assert len(list(iter_posts(fixture_zip, limit=10))) == 10
    assert list(iter_posts(fixture_zip, limit=0)) == []


def test_iter_posts_schema(fixture_zip):
    post = next(iter_posts(fixture_zip, limit=1))
    for key in ("author", "content", "summary", "subreddit", "id", "title"):
        assert key in post


def test_iter_posts_handles_non_ascii(fixture_zip):
    posts = list(iter_posts(fixture_zip))
    assert any("Üñïcødé" in (p.get("content") or "") for p in posts)


def test_iter_posts_resolves_member_name(fixture_zip):
    # config may carry a stale member name; reader must fall back gracefully
    posts = list(iter_posts(fixture_zip, member="does-not-exist.json", limit=5))
    assert len(posts) == 5


def test_word_count():
    assert word_count("one two three") == 3
    assert word_count("") == 0
    assert word_count(None) == 0


def test_passes_filters_accepts_good_post():
    post = {"content": " ".join(["w"] * 200), "summary": " ".join(["s"] * 20), "title": "t"}
    assert passes_filters(post, FILTERS)


def test_passes_filters_rejects_short_content():
    post = {"content": " ".join(["w"] * 20), "summary": " ".join(["s"] * 20)}
    assert not passes_filters(post, FILTERS)


def test_passes_filters_rejects_empty_or_missing_summary():
    long_content = " ".join(["w"] * 200)
    assert not passes_filters({"content": long_content, "summary": ""}, FILTERS)
    assert not passes_filters({"content": long_content}, FILTERS)


def test_passes_filters_require_title():
    f = dict(FILTERS, require_title=True)
    post = {"content": " ".join(["w"] * 200), "summary": " ".join(["s"] * 20), "title": None}
    assert not passes_filters(post, f)
    post["title"] = "hello"
    assert passes_filters(post, f)


def test_fixture_filters_pass_rate(fixture_zip):
    """Most regular fixture posts pass; the deliberately-broken ones don't."""
    posts = list(iter_posts(fixture_zip))
    n_pass = sum(passes_filters(p, FILTERS) for p in posts)
    assert 0 < n_pass < len(posts)
