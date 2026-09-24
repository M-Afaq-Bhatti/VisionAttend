"""
Unit tests for the anti-spoofing/liveness checker.

Tests the movement-based liveness detection system:
  - Empty/blurry image handling
  - Warmup period (UNCERTAIN while collecting frames)
  - Live face with simulated blinks (→ LIVE)
  - Static photo with no movement (→ SPOOF)
  - Fallback when no landmarks available
"""

import numpy as np
import pytest

from src.antispoof.checker import LivenessChecker, SpoofStatus


@pytest.fixture
def checker():
    return LivenessChecker(
        threshold=0.5,
        warmup_frames=10,
        movement_diff_threshold=0.03,
    )


def _make_face(rng, base_seed=42):
    """Create a realistic-looking synthetic face crop (100x100 BGR)."""
    face = np.zeros((100, 100, 3), dtype=np.uint8)
    face[:, :, 0] = rng.randint(60, 100, (100, 100))   # B
    face[:, :, 1] = rng.randint(100, 140, (100, 100))   # G
    face[:, :, 2] = rng.randint(140, 190, (100, 100))   # R
    return face


def _make_landmarks():
    """Standard 5-point landmarks for a 100x100 face crop."""
    return np.array([
        [30.0, 35.0],   # left eye
        [70.0, 35.0],   # right eye
        [50.0, 55.0],   # nose
        [35.0, 75.0],   # left mouth
        [65.0, 75.0],   # right mouth
    ], dtype=np.float32)


def test_empty_image(checker):
    result = checker.analyze(np.array([]))
    assert result.status == SpoofStatus.UNCERTAIN


def test_blurry_image(checker):
    """A completely uniform image is too blurry to analyze."""
    blurry = np.full((100, 100, 3), 128, dtype=np.uint8)
    # No landmarks → falls back to basic check
    result = checker.analyze(blurry)
    assert result.status == SpoofStatus.UNCERTAIN
    assert "blurry" in result.details.lower()


def test_warmup_returns_uncertain(checker):
    """During the warmup period, status should be UNCERTAIN."""
    rng = np.random.RandomState(42)
    face = _make_face(rng)
    landmarks = _make_landmarks()

    for i in range(checker.warmup_frames - 1):
        result = checker.analyze(face.copy(), landmarks)
        assert result.status == SpoofStatus.UNCERTAIN
        assert "Verifying" in result.details


def test_live_face_with_blinks(checker):
    """
    Simulate a live person: mostly stable frames with occasional 
    blink-like changes in the eye region.
    """
    rng = np.random.RandomState(42)
    base_face = _make_face(rng)
    landmarks = _make_landmarks()

    results = []
    for i in range(20):
        face = base_face.copy()

        # Simulate a blink at frames 12 and 13 (darken the eye region)
        if i in (12, 13):
            face[25:45, 20:80, :] = np.clip(
                face[25:45, 20:80, :].astype(int) - 60, 0, 255
            ).astype(np.uint8)

        # Add natural micro-movement noise every frame
        noise = rng.randint(-3, 4, face.shape, dtype=np.int16)
        face = np.clip(face.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        result = checker.analyze(face, landmarks)
        results.append(result)

    # After warmup + blink, the final result should be LIVE
    assert results[-1].status == SpoofStatus.LIVE
    assert results[-1].liveness_score >= checker.threshold


def test_static_photo_detected(checker):
    """
    Simulate a static photo on a phone screen: the exact same image 
    every frame with only tiny sensor noise. No blinks, no movement.
    """
    rng = np.random.RandomState(99)
    static_face = _make_face(rng)
    landmarks = _make_landmarks()

    results = []
    for i in range(20):
        face = static_face.copy()
        # Only add minimal sensor noise (much smaller than a real blink)
        noise = rng.randint(-1, 2, face.shape, dtype=np.int16)
        face = np.clip(face.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        result = checker.analyze(face, landmarks)
        results.append(result)

    # After warmup, should detect no movement → SPOOF
    assert results[-1].status == SpoofStatus.SPOOF
    assert results[-1].liveness_score < checker.threshold


def test_no_landmarks_fallback(checker):
    """Without landmarks, falls back to basic check (permissive)."""
    rng = np.random.RandomState(42)
    face = _make_face(rng)

    result = checker.analyze(face, landmarks=None)
    # Should pass with basic check (no landmarks to do movement detection)
    assert result.status == SpoofStatus.LIVE
    assert "no landmarks" in result.details.lower()


def test_reset_clears_state(checker):
    """After reset(), warmup period starts again."""
    rng = np.random.RandomState(42)
    face = _make_face(rng)
    landmarks = _make_landmarks()

    # Feed some frames
    for _ in range(15):
        checker.analyze(face.copy(), landmarks)

    # Reset
    checker.reset()

    # Next frame should be in warmup again
    result = checker.analyze(face.copy(), landmarks)
    assert result.status == SpoofStatus.UNCERTAIN
    assert "Verifying" in result.details
