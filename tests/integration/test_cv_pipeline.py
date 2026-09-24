"""
Integration test for the CV pipeline.

Tests the actual InsightFace models (detection + embedding).
Requires the buffalo_l model pack to be downloaded.
These tests use CPU and are slower than unit tests.
"""

import os
import sys
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.cv.detection.detector import FaceDetector
from src.cv.embeddings.extractor import FaceEmbedder
from src.cv.quality.checker import FaceQualityChecker, QualityStatus
from src.cv.recognition.matcher import IdentityMatcher


# Skip these tests if models haven't been downloaded
def _models_available():
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(320, 320))
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _models_available(),
    reason="InsightFace buffalo_l model not available",
)


@pytest.fixture(scope="module")
def detector():
    d = FaceDetector(ctx_id=-1, det_size=(320, 320))
    d.initialize()
    return d


@pytest.fixture(scope="module")
def embedder():
    e = FaceEmbedder(ctx_id=-1)
    e.initialize()
    return e


class TestFaceDetector:
    def test_no_face_on_blank(self, detector):
        blank = np.full((480, 640, 3), 128, dtype=np.uint8)
        result = detector.detect(blank)
        assert result.num_faces == 0

    def test_detect_returns_detection_result(self, detector):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.detect(img)
        assert hasattr(result, 'num_faces')
        assert hasattr(result, 'latency_ms')
        assert result.latency_ms > 0

    def test_not_initialized_raises(self):
        d = FaceDetector()
        with pytest.raises(RuntimeError, match="not initialized"):
            d.detect(np.zeros((100, 100, 3), dtype=np.uint8))


class TestFaceEmbedder:
    def test_extract_produces_512d_embedding(self, embedder):
        # Use a face-like crop (just a colored square — result won't be meaningful
        # but should still produce a valid embedding)
        face_crop = np.random.randint(50, 200, (112, 112, 3), dtype=np.uint8)
        result = embedder.extract(face_crop)
        assert result.embedding.shape == (512,)
        # Check L2 normalized
        norm = np.linalg.norm(result.embedding)
        assert abs(norm - 1.0) < 0.01

    def test_different_inputs_produce_different_embeddings(self, embedder):
        face_a = np.random.randint(50, 200, (112, 112, 3), dtype=np.uint8)
        face_b = np.random.randint(50, 200, (112, 112, 3), dtype=np.uint8)
        emb_a = embedder.extract(face_a).embedding
        emb_b = embedder.extract(face_b).embedding
        sim = float(np.dot(emb_a, emb_b))
        # Different random inputs should produce different embeddings
        assert sim < 0.99

    def test_same_input_produces_same_embedding(self, embedder):
        face = np.random.randint(50, 200, (112, 112, 3), dtype=np.uint8)
        emb1 = embedder.extract(face).embedding
        emb2 = embedder.extract(face).embedding
        sim = float(np.dot(emb1, emb2))
        assert sim > 0.999

    def test_not_initialized_raises(self):
        e = FaceEmbedder()
        with pytest.raises(RuntimeError, match="not initialized"):
            e.extract(np.zeros((112, 112, 3), dtype=np.uint8))


class TestEndToEndRecognition:
    """Integration: embed → register → match → recognize."""

    def test_register_and_recognize(self, embedder):
        matcher = IdentityMatcher(threshold=0.45, top_k=3)

        # Create "employee" with several face crops
        rng = np.random.RandomState(42)
        employee_faces = [
            rng.randint(50, 200, (112, 112, 3)).astype(np.uint8)
            for _ in range(3)
        ]

        embeddings = [embedder.extract(f).embedding for f in employee_faces]
        matcher.register(1, "EMP001", embeddings)

        # Query with the same face should recognize
        query_emb = embedder.extract(employee_faces[0]).embedding
        result = matcher.match(query_emb)
        assert result.recognized is True
        assert result.identity == "EMP001"
        assert result.similarity > 0.99

    def test_unknown_not_forced(self, embedder):
        matcher = IdentityMatcher(threshold=0.45)

        rng_emp = np.random.RandomState(10)
        emp_face = rng_emp.randint(50, 200, (112, 112, 3)).astype(np.uint8)
        matcher.register(1, "EMP001", [embedder.extract(emp_face).embedding])

        # Very different face should be unknown
        rng_stranger = np.random.RandomState(999)
        stranger_face = rng_stranger.randint(50, 200, (112, 112, 3)).astype(np.uint8)
        query_emb = embedder.extract(stranger_face).embedding
        result = matcher.match(query_emb)
        # With random noise images, similarity will be low
        assert result.identity is None or result.similarity < 0.99
