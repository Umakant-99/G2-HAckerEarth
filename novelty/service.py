"""Score one submission against the cached reference and corpus.

The returned numbers are computed from embeddings on each call. They are not
read from a stored score.
"""

from __future__ import annotations

import numpy as np

from novelty.gemini_client import embed_texts
from novelty.models import Submission
from novelty.scorer import Score, score_vectors
from novelty.synth_data import load_fixture, submissions_from_fixture, vectors_from_fixture


def score_text(text: str, topic: str = "adhoc", submission_id: str = "submission", reduction: str = "max") -> dict:
    """Embed ``text`` if needed and return relevance, novelty, and final_score."""
    submission = Submission(id=submission_id, text=text, topic=topic)
    payload = load_fixture()
    _reference, corpus, _cases = submissions_from_fixture(payload)
    vectors = vectors_from_fixture(payload)
    submission_vector = vector_for_text(submission.text, vectors, payload)
    corpus_matrix = np.vstack([vectors[item.id] for item in corpus])
    result = score_vectors(
        submission_vector,
        vectors["reference"],
        corpus_matrix,
        reduction=reduction,
    )
    return _public_score(submission, payload["embeddings"]["model"], result)


def context() -> dict:
    """Reference passage and corpus size for the local page. No scores."""
    payload = load_fixture()
    reference, corpus, _cases = submissions_from_fixture(payload)
    return {
        "reference": reference.to_dict(),
        "corpus_size": len(corpus),
        "embedding_model": payload["embeddings"]["model"],
    }


def vector_for_text(text: str, vectors: dict[str, np.ndarray], payload: dict) -> np.ndarray:
    """Reuse a fixture vector when the text matches, otherwise embed it."""
    lookup = {payload["reference"]["text"]: vectors["reference"]}
    for item in payload["corpus"]:
        lookup[item["text"]] = vectors[item["id"]]
    for item in payload["cases"].values():
        lookup[item["text"]] = vectors[item["id"]]
    cached = lookup.get(text.strip())
    if cached is not None:
        return cached
    return embed_texts([text])[0]


def _public_score(submission: Submission, embedding_model: str, result: Score) -> dict:
    return {
        "id": submission.id,
        "topic": submission.topic,
        "embedding_model": embedding_model,
        "relevance": round(result.relevance, 3),
        "novelty": round(result.novelty, 3),
        "final_score": round(result.final_score, 3),
    }
