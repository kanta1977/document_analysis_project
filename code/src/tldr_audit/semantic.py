"""Semantic distance between a post and its summary (item 3).

cosine_similarity(post, summary) tells us whether the summary *means* the same
thing as the post, even when it uses different words. Paired with keyword
containment (item 4, lexical), it separates:

    high containment, high cosine  -> extractive summary
    low  containment, high cosine  -> paraphrase (still a summary)
    low  containment, low  cosine  -> genuinely diverged (not a summary)

cosine is a DISTANCE we describe, not a yes/no verdict on "is it a summary".

Backends, in order of preference:
  1. sentence-transformers (semantic embeddings) if installed AND the model is
     available locally / downloadable.
  2. TF-IDF cosine (scikit-learn) as a no-download fallback that runs anywhere.
The chosen backend is returned so it can be reported honestly in the write-up.
"""

from __future__ import annotations

import numpy as np


def _sbert_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer  # may raise ImportError

    model = SentenceTransformer(model_name)
    return np.asarray(model.encode(list(texts), normalize_embeddings=True))


def pairwise_cosine(a_texts, b_texts, prefer="auto"):
    """Row-wise cosine similarity between paired texts a_texts[i], b_texts[i].

    Returns (similarities: np.ndarray, backend: str).
    `prefer` in {"auto", "sbert", "tfidf"}.
    """
    a_texts = ["" if t is None else str(t) for t in a_texts]
    b_texts = ["" if t is None else str(t) for t in b_texts]

    # try SBERT unless tfidf was explicitly requested
    if prefer in ("auto", "sbert"):
        try:
            emb_a = _sbert_embeddings(a_texts)
            emb_b = _sbert_embeddings(b_texts)
            sims = np.sum(emb_a * emb_b, axis=1)  # already normalized
            return sims, "sbert:all-MiniLM-L6-v2"
        except Exception as e:
            if prefer == "sbert":
                raise
            print(f"[semantic] sentence-transformers unavailable ({e}); "
                  f"falling back to TF-IDF cosine.")

    # TF-IDF fallback (no model download needed)
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(stop_words="english")
    all_text = a_texts + b_texts
    m = vec.fit_transform(all_text)
    n = len(a_texts)
    ma, mb = m[:n], m[n:]
    # row-wise cosine on sparse tf-idf (vectors are L2-normalized by default)
    sims = np.asarray(ma.multiply(mb).sum(axis=1)).ravel()
    return sims, "tfidf"
