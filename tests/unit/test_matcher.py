"""
Unit tests for the identity matcher.
"""

import numpy as np
import pytest

from src.cv.recognition.matcher import IdentityMatcher, cosine_similarity


@pytest.fixture
def matcher():
    return IdentityMatcher(threshold=0.45, top_k=3)


def make_embedding(dim=512, seed=None):
    """Create a random L2-normalized embedding."""
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestCosignSimilarity:
    """Tests for the standalone cosine similarity function."""

    def test_identical_vectors(self):
        v = make_embedding(seed=42)
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-5

    def test_orthogonal_vectors(self):
        a = np.zeros(512, dtype=np.float32)
        b = np.zeros(512, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0
        assert abs(cosine_similarity(a, b)) < 1e-5

    def test_opposite_vectors(self):
        v = make_embedding(seed=42)
        assert abs(cosine_similarity(v, -v) + 1.0) < 1e-5

    def test_zero_vector(self):
        v = make_embedding(seed=42)
        zero = np.zeros(512, dtype=np.float32)
        assert cosine_similarity(v, zero) == 0.0


class TestIdentityMatcher:
    """Tests for gallery management and matching."""

    def test_empty_gallery_returns_not_recognized(self, matcher):
        query = make_embedding(seed=1)
        result = matcher.match(query)
        assert result.recognized is False
        assert result.identity is None
        assert result.candidates == []

    def test_register_and_match_same_embedding(self, matcher):
        emb = make_embedding(seed=42)
        matcher.register(1, "EMP001", [emb])
        result = matcher.match(emb)
        assert result.recognized is True
        assert result.identity == "EMP001"
        assert result.employee_id == 1
        assert result.similarity > 0.99

    def test_match_returns_best_candidate(self, matcher):
        emb1 = make_embedding(seed=10)
        emb2 = make_embedding(seed=20)
        emb3 = make_embedding(seed=30)
        matcher.register(1, "EMP001", [emb1])
        matcher.register(2, "EMP002", [emb2])
        matcher.register(3, "EMP003", [emb3])

        # Query with emb1 should return EMP001
        result = matcher.match(emb1)
        assert result.recognized is True
        assert result.identity == "EMP001"

    def test_unknown_person_below_threshold(self, matcher):
        emb1 = make_embedding(seed=10)
        matcher.register(1, "EMP001", [emb1])

        # A very different embedding should be below threshold
        query = make_embedding(seed=999)
        result = matcher.match(query)
        # With random 512-D vectors, similarity will typically be ~0
        assert result.similarity < matcher.threshold
        assert result.recognized is False
        assert result.identity is None

    def test_multiple_embeddings_per_employee(self, matcher):
        emb_a = make_embedding(seed=10)
        emb_b = make_embedding(seed=11)
        matcher.register(1, "EMP001", [emb_a, emb_b])

        # Query with emb_a should match
        result = matcher.match(emb_a)
        assert result.recognized is True
        assert result.identity == "EMP001"
        assert result.similarity > 0.99

    def test_gallery_size(self, matcher):
        assert matcher.gallery_size == 0
        matcher.register(1, "EMP001", [make_embedding(seed=1)])
        matcher.register(2, "EMP002", [make_embedding(seed=2)])
        assert matcher.gallery_size == 2

    def test_unregister(self, matcher):
        matcher.register(1, "EMP001", [make_embedding(seed=1)])
        assert matcher.gallery_size == 1
        matcher.unregister(1)
        assert matcher.gallery_size == 0

    def test_update_threshold(self, matcher):
        matcher.update_threshold(0.60)
        assert matcher.threshold == 0.60

    def test_top_k_candidates(self, matcher):
        for i in range(10):
            matcher.register(i, f"EMP{i:03d}", [make_embedding(seed=i)])

        query = make_embedding(seed=0)
        result = matcher.match(query)
        # Should return at most top_k=3 candidates
        assert len(result.candidates) <= 3
        # First candidate should be the best match
        assert result.candidates[0].similarity >= result.candidates[-1].similarity
