import math

import pytest

from kc.indexing.vector_math import cosine_similarity, norm, normalize, top_k


def test_normalize_gives_unit_length() -> None:
    assert math.isclose(norm(normalize([3.0, 4.0])), 1.0)
    assert normalize([3.0, 4.0]) == pytest.approx([0.6, 0.8])


def test_normalize_zero_vector_stays_zero() -> None:
    assert normalize([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


def test_cosine_identical_orthogonal_opposite() -> None:
    assert cosine_similarity([1.0, 2.0], [2.0, 4.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 1.0], [-1.0, -1.0]) == pytest.approx(-1.0)


def test_cosine_with_zero_vector_is_zero() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_cosine_dimension_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine_similarity([1.0], [1.0, 2.0])


def test_top_k_orders_by_score_desc() -> None:
    query = [1.0, 0.0]
    vectors = [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    result = top_k(query, vectors, k=2)
    assert [i for i, _ in result] == [1, 2]
    assert result[0][1] == pytest.approx(1.0)


def test_top_k_ties_keep_original_order() -> None:
    vectors = [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    assert [i for i, _ in top_k([1.0, 0.0], vectors, k=3)] == [0, 1, 2]


def test_top_k_edge_cases() -> None:
    assert top_k([1.0], [], k=3) == []
    assert top_k([1.0], [[1.0]], k=0) == []
    assert len(top_k([1.0], [[1.0], [2.0]], k=10)) == 2
    with pytest.raises(ValueError):
        top_k([1.0], [[1.0]], k=-1)
