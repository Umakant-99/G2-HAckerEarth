"""Relevance, novelty, and the multiplicative final score.

The arithmetic is pure numpy. Pass vectors in; this module does not call Gemini.

    relevance = cosine(submission, reference)
    novelty   = 1 - max cosine to any corpus submission
    final     = max(relevance, 0) * novelty

Max is the default neighbor reduction. ``top3_mean`` averages the three
closest corpus items instead, which still punishes a near duplicate more than
a mean over the whole pool would.

A weighted sum would let an off-topic but original submission keep about half
credit. The product does not: if either factor is near zero, the score is near
zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


NOVELTY_REDUCTION = "max"
TOP_K = 3


@dataclass(frozen=True)
class Score:
    """One live scoring result. ``final_score`` is in ``[0, 1]``."""

    relevance: float
    novelty: float
    final_score: float


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine similarity in ``[-1, 1]``. A zero vector yields ``0``."""
    left_vector = np.asarray(left, dtype=np.float64).reshape(-1)
    right_vector = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
    if denominator == 0.0:
        return 0.0
    similarity = float(np.dot(left_vector, right_vector) / denominator)
    return float(np.clip(similarity, -1.0, 1.0))


def relevance(submission_vector: np.ndarray, reference_vector: np.ndarray) -> float:
    """Cosine similarity between a submission and the reference."""
    return cosine(submission_vector, reference_vector)


def novelty(
    submission_vector: np.ndarray,
    corpus_vectors: np.ndarray,
    reduction: str = NOVELTY_REDUCTION,
) -> float:
    """``1 - nearest-neighbor similarity``, clipped to ``[0, 1]``.

    ``reduction="max"`` uses the single closest corpus item.
    ``reduction="top3_mean"`` uses the mean of the closest three.
    An empty corpus has nothing to collide with, so novelty is ``1``.
    """
    matrix = np.asarray(corpus_vectors, dtype=np.float64)
    if matrix.size == 0:
        return 1.0
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    similarities = np.array(
        [cosine(submission_vector, row) for row in matrix],
        dtype=np.float64,
    )
    nearest = _nearest(similarities, reduction)
    return float(np.clip(1.0 - nearest, 0.0, 1.0))


def score_vectors(
    submission_vector: np.ndarray,
    reference_vector: np.ndarray,
    corpus_vectors: np.ndarray,
    reduction: str = NOVELTY_REDUCTION,
) -> Score:
    """Combine relevance and novelty multiplicatively.

    Negative relevance is treated as zero in the product so an opposing
    embedding cannot produce a negative score. The raw relevance value is
    still returned on the result.
    """
    relevance_score = relevance(submission_vector, reference_vector)
    novelty_score = novelty(submission_vector, corpus_vectors, reduction=reduction)
    final = max(relevance_score, 0.0) * novelty_score
    final = float(np.clip(final, 0.0, 1.0))
    return Score(
        relevance=float(relevance_score),
        novelty=float(novelty_score),
        final_score=final,
    )


def _nearest(similarities: np.ndarray, reduction: str) -> float:
    if reduction == "max":
        return float(np.max(similarities))
    if reduction == "top3_mean":
        k = max(1, min(TOP_K, similarities.size))
        return float(np.mean(np.sort(similarities)[-k:]))
    raise ValueError(f"unknown novelty reduction: {reduction!r}")
