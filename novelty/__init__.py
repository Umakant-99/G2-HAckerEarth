"""Novelty scoring against a reference submission and a prior corpus."""

from novelty.models import Submission
from novelty.scorer import Score, novelty, relevance, score_vectors

__all__ = ["Submission", "Score", "novelty", "relevance", "score_vectors"]
