"""Scores are computed from the cached embeddings when the test runs.

The fixture stores submissions and vectors only. It does not store scores.
"""

from __future__ import annotations

import numpy as np
import pytest

from novelty.scorer import Score, cosine, score_vectors
from novelty.synth_data import load_fixture, submissions_from_fixture, vectors_from_fixture


@pytest.fixture(scope="module")
def scored() -> dict[str, Score]:
    payload = load_fixture()
    _reference, corpus, cases = submissions_from_fixture(payload)
    vectors = vectors_from_fixture(payload)
    corpus_matrix = np.vstack([vectors[item.id] for item in corpus])
    reference_vector = vectors["reference"]
    return {
        name: score_vectors(vectors[item.id], reference_vector, corpus_matrix)
        for name, item in cases.items()
    }


def test_final_score_is_the_product_of_the_two_signals(scored: dict[str, Score]) -> None:
    for name, result in scored.items():
        product = max(result.relevance, 0.0) * result.novelty
        assert result.final_score == pytest.approx(product), name


def test_novel_and_relevant_outranks_the_other_cases(scored: dict[str, Score]) -> None:
    novel = scored["novel_relevant"]
    assert novel.final_score > scored["non_novel"].final_score
    assert novel.final_score > scored["novel_irrelevant"].final_score


def test_non_novel_is_closer_to_the_corpus(scored: dict[str, Score]) -> None:
    assert scored["non_novel"].novelty < scored["novel_relevant"].novelty


def test_irrelevant_case_is_less_relevant(scored: dict[str, Score]) -> None:
    assert scored["novel_irrelevant"].relevance < scored["novel_relevant"].relevance


def test_paraphrase_matches_its_source_more_than_the_original_take() -> None:
    payload = load_fixture()
    _reference, _corpus, cases = submissions_from_fixture(payload)
    vectors = vectors_from_fixture(payload)
    source_id = payload["paraphrase_of"]
    source = vectors[source_id]
    paraphrase = cosine(vectors[cases["non_novel"].id], source)
    original = cosine(vectors[cases["novel_relevant"].id], source)
    assert paraphrase > original
