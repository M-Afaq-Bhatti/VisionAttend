"""
Unit tests for the face quality checker.
"""

import numpy as np
import pytest
import cv2

from src.cv.quality.checker import FaceQualityChecker, QualityStatus


@pytest.fixture
def checker():
    return FaceQualityChecker(
        min_face_size=80,
        min_blur_score=30.0,
        min_brightness=40.0,
        max_brightness=220.0,
    )


class TestFaceQualityChecker:
    """Tests for quality assessment logic."""

    def test_empty_image_returns_no_face(self, checker):
        result = checker.check(np.array([]))
        assert result.status == QualityStatus.NO_FACE

    def test_none_image_returns_no_face(self, checker):
        result = checker.check(None)
        assert result.status == QualityStatus.NO_FACE

    def test_small_face_rejected(self, checker):
        # 50x50 is below the 80px minimum
        small = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        result = checker.check(small)
        assert result.status == QualityStatus.FACE_TOO_SMALL
        assert result.face_size == 50

    def test_valid_face_accepted(self, checker):
        # Create a 200x200 image with enough texture (not blurry) and good brightness
        img = np.random.randint(80, 180, (200, 200, 3), dtype=np.uint8)
        result = checker.check(img)
        assert result.status == QualityStatus.VALID
        assert result.face_size == 200

    def test_dark_image_rejected(self, checker):
        # Very dark image
        dark = np.full((200, 200, 3), 10, dtype=np.uint8)
        # Add some texture to avoid blur rejection first
        dark[50:150, 50:150] = np.random.randint(5, 25, (100, 100, 3), dtype=np.uint8)
        result = checker.check(dark)
        # Could be BLURRY or LOW_LIGHT depending on which check fires first
        assert result.status in (QualityStatus.LOW_LIGHT, QualityStatus.BLURRY)

    def test_overexposed_image_rejected(self, checker):
        # Very bright image with enough texture
        bright = np.random.randint(230, 255, (200, 200, 3), dtype=np.uint8)
        result = checker.check(bright)
        assert result.status in (QualityStatus.LOW_QUALITY, QualityStatus.BLURRY)

    def test_blurry_image_rejected(self, checker):
        # Uniform color = no edges = blur score ~0
        blurry = np.full((200, 200, 3), 128, dtype=np.uint8)
        result = checker.check(blurry)
        assert result.status == QualityStatus.BLURRY
        assert result.blur_score is not None
        assert result.blur_score < checker.min_blur_score

    def test_pose_estimation_with_landmarks(self, checker):
        img = np.random.randint(80, 180, (200, 200, 3), dtype=np.uint8)
        # Frontal-ish landmarks
        landmarks = np.array([
            [70, 80],   # left eye
            [130, 80],  # right eye
            [100, 110], # nose
            [80, 140],  # mouth left
            [120, 140], # mouth right
        ], dtype=np.float32)
        result = checker.check(img, landmarks)
        assert result.status == QualityStatus.VALID
        assert result.yaw_estimate is not None
        assert result.pitch_estimate is not None

    def test_extreme_yaw_rejected(self, checker):
        img = np.random.randint(80, 180, (200, 200, 3), dtype=np.uint8)
        # Very offset nose = extreme yaw
        landmarks = np.array([
            [70, 80],
            [130, 80],
            [170, 110],  # nose way to the right
            [80, 140],
            [120, 140],
        ], dtype=np.float32)
        result = checker.check(img, landmarks)
        assert result.status == QualityStatus.EXTREME_POSE
