"""
Unit tests for the anti-spoofing/liveness checker.

Tests the multi-signal liveness detection system:
  - Color temperature analysis
  - LBP texture analysis
  - Moiré pattern detection
  - Reflection uniformity
"""

import numpy as np
import pytest

from src.antispoof.checker import LivenessChecker, SpoofStatus


@pytest.fixture
def checker():
    return LivenessChecker(
        threshold=0.5,
        min_blur_var=15.0,
        max_glare_ratio=0.08
    )


def test_empty_image(checker):
    result = checker.analyze(np.array([]))
    assert result.status == SpoofStatus.UNCERTAIN


def test_blurry_image(checker):
    # A completely uniform image has 0 Laplacian variance → too blurry
    blurry = np.full((100, 100, 3), 128, dtype=np.uint8)
    result = checker.analyze(blurry)
    assert result.status == SpoofStatus.UNCERTAIN
    assert "blurry" in result.details.lower()


def test_live_face(checker):
    """
    Simulate a live face: warm-toned skin colors with natural texture 
    variation and uneven lighting (shadows on one side).
    """
    rng = np.random.RandomState(42)
    face = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Warm skin tones (R > G > B — natural lighting)
    face[:, :, 0] = rng.randint(60, 100, (100, 100))    # Blue (low)
    face[:, :, 1] = rng.randint(100, 140, (100, 100))    # Green (mid)
    face[:, :, 2] = rng.randint(140, 190, (100, 100))    # Red (high)
    
    # Add natural shadows on the left side (uneven illumination)
    face[:, :30, :] = (face[:, :30, :] * 0.5).astype(np.uint8)
    
    result = checker.analyze(face)
    assert result.status == SpoofStatus.LIVE
    assert result.liveness_score >= checker.threshold


def test_screen_spoof(checker):
    """
    Simulate a phone screen spoof: blue-tinted, very uniform lighting,
    with periodic patterns (simulating Moiré).
    """
    rng = np.random.RandomState(99)
    face = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Blue-heavy color (screen backlight)
    face[:, :, 0] = rng.randint(130, 170, (100, 100))    # Blue (high)
    face[:, :, 1] = rng.randint(100, 130, (100, 100))    # Green (mid)
    face[:, :, 2] = rng.randint(90, 120, (100, 100))     # Red (low)
    
    # Add periodic stripe pattern (Moiré-like artifact)
    for y in range(0, 100, 3):
        face[y, :, :] = np.clip(face[y, :, :].astype(int) + 40, 0, 255).astype(np.uint8)
    
    result = checker.analyze(face)
    assert result.status == SpoofStatus.SPOOF
    assert result.liveness_score < checker.threshold


def test_glare_image(checker):
    """Heavy glare (all bright white pixels) should be rejected."""
    glare = np.full((100, 100, 3), 250, dtype=np.uint8)
    
    # Add some noise to pass the blur pre-check
    glare[10:90, 10:90] = np.random.randint(100, 255, (80, 80, 3), dtype=np.uint8)
    
    result = checker.analyze(glare)
    # Should be either SPOOF or UNCERTAIN due to extreme glare
    assert result.status in (SpoofStatus.SPOOF, SpoofStatus.UNCERTAIN)
