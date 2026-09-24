"""
Unit tests for the anti-spoofing/liveness checker.
"""

import numpy as np
import pytest

from src.antispoof.checker import LivenessChecker, SpoofStatus


@pytest.fixture
def checker():
    return LivenessChecker(
        threshold=0.5,
        min_blur_var=20.0,
        max_glare_ratio=0.1
    )


def test_empty_image(checker):
    result = checker.analyze(np.array([]))
    assert result.status == SpoofStatus.UNCERTAIN


def test_blurry_image(checker):
    # A completely uniform image has 0 variance
    blurry = np.full((100, 100, 3), 128, dtype=np.uint8)
    result = checker.analyze(blurry)
    assert result.status == SpoofStatus.SPOOF
    assert "Texture too smooth" in result.details


def test_live_face(checker):
    # Random noise usually has high variance
    # Generate random texture with some structure but not pure noise
    live = np.random.randint(50, 150, (100, 100, 3), dtype=np.uint8)
    
    # Ensure it's not all glare
    result = checker.analyze(live)
    
    # Random noise has extremely high laplacian variance, so it should pass the blur check.
    # It also has low glare since pixels are below 150 (glare threshold is 240).
    assert result.status == SpoofStatus.LIVE
    assert result.liveness_score >= checker.threshold


def test_glare_image(checker):
    # Image with mostly white pixels
    glare = np.full((100, 100, 3), 250, dtype=np.uint8)
    
    # Add some noise to pass the blur check
    glare[10:90, 10:90] = np.random.randint(100, 255, (80, 80, 3), dtype=np.uint8)
    
    result = checker.analyze(glare)
    assert result.status == SpoofStatus.SPOOF
    assert "Screen glare" in result.details or result.liveness_score < checker.threshold
