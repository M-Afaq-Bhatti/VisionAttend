"""
Face quality assessment module.

Evaluates detected faces for suitability before recognition.
Checks face size, blur, pose, and brightness to filter out
low-quality detections that would produce unreliable embeddings.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class QualityStatus(str, Enum):
    """Enumeration of possible quality assessment outcomes."""
    VALID = "VALID"
    NO_FACE = "NO_FACE"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    FACE_TOO_SMALL = "FACE_TOO_SMALL"
    LOW_QUALITY = "LOW_QUALITY"
    EXTREME_POSE = "EXTREME_POSE"
    LOW_LIGHT = "LOW_LIGHT"
    BLURRY = "BLURRY"


@dataclass
class QualityResult:
    """Result of a face quality assessment."""
    status: QualityStatus
    face_size: Optional[int] = None
    blur_score: Optional[float] = None
    brightness: Optional[float] = None
    yaw_estimate: Optional[float] = None
    pitch_estimate: Optional[float] = None
    details: str = ""


class FaceQualityChecker:
    """
    Evaluates whether a detected face crop meets minimum quality
    requirements for reliable face recognition.

    All thresholds are configurable at init time.
    """

    def __init__(
        self,
        min_face_size: int = 80,
        min_blur_score: float = 30.0,
        min_brightness: float = 40.0,
        max_brightness: float = 220.0,
        max_yaw: float = 45.0,
        max_pitch: float = 35.0,
    ):
        self.min_face_size = min_face_size
        self.min_blur_score = min_blur_score
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.max_yaw = max_yaw
        self.max_pitch = max_pitch

    def check(
        self,
        face_img: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
    ) -> QualityResult:
        """
        Run all quality checks on a face crop.

        Args:
            face_img: BGR face crop as numpy array (H, W, 3).
            landmarks: Optional 5-point facial landmarks (5, 2).

        Returns:
            QualityResult with status and diagnostic values.
        """
        if face_img is None or face_img.size == 0:
            return QualityResult(status=QualityStatus.NO_FACE, details="Empty image")

        h, w = face_img.shape[:2]
        face_size = min(h, w)

        # --- Size check ---
        if face_size < self.min_face_size:
            return QualityResult(
                status=QualityStatus.FACE_TOO_SMALL,
                face_size=face_size,
                details=f"Face size {face_size}px < minimum {self.min_face_size}px",
            )

        # --- Blur check (Laplacian variance) ---
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        if blur_score < self.min_blur_score:
            return QualityResult(
                status=QualityStatus.BLURRY,
                face_size=face_size,
                blur_score=blur_score,
                details=f"Blur score {blur_score:.1f} < threshold {self.min_blur_score}",
            )

        # --- Brightness check ---
        brightness = float(np.mean(gray))

        if brightness < self.min_brightness:
            return QualityResult(
                status=QualityStatus.LOW_LIGHT,
                face_size=face_size,
                blur_score=blur_score,
                brightness=brightness,
                details=f"Brightness {brightness:.1f} < minimum {self.min_brightness}",
            )

        if brightness > self.max_brightness:
            return QualityResult(
                status=QualityStatus.LOW_QUALITY,
                face_size=face_size,
                blur_score=blur_score,
                brightness=brightness,
                details=f"Over-exposed: brightness {brightness:.1f} > {self.max_brightness}",
            )

        # --- Pose estimation from landmarks ---
        yaw_est = None
        pitch_est = None
        if landmarks is not None and len(landmarks) >= 5:
            yaw_est, pitch_est = self._estimate_pose(landmarks, w, h)
            if abs(yaw_est) > self.max_yaw:
                return QualityResult(
                    status=QualityStatus.EXTREME_POSE,
                    face_size=face_size,
                    blur_score=blur_score,
                    brightness=brightness,
                    yaw_estimate=yaw_est,
                    pitch_estimate=pitch_est,
                    details=f"Yaw {yaw_est:.1f}° exceeds ±{self.max_yaw}°",
                )
            if abs(pitch_est) > self.max_pitch:
                return QualityResult(
                    status=QualityStatus.EXTREME_POSE,
                    face_size=face_size,
                    blur_score=blur_score,
                    brightness=brightness,
                    yaw_estimate=yaw_est,
                    pitch_estimate=pitch_est,
                    details=f"Pitch {pitch_est:.1f}° exceeds ±{self.max_pitch}°",
                )

        return QualityResult(
            status=QualityStatus.VALID,
            face_size=face_size,
            blur_score=blur_score,
            brightness=brightness,
            yaw_estimate=yaw_est,
            pitch_estimate=pitch_est,
            details="Quality checks passed",
        )

    @staticmethod
    def _estimate_pose(
        landmarks: np.ndarray, img_w: int, img_h: int
    ) -> tuple[float, float]:
        """
        Rough yaw/pitch estimation from 5-point landmarks.

        Landmarks assumed order: left_eye, right_eye, nose, mouth_left, mouth_right.
        Uses inter-eye distance and nose position relative to eye midpoint.

        Returns:
            (yaw_degrees, pitch_degrees) — approximate values.
        """
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]

        eye_center = (left_eye + right_eye) / 2.0
        inter_eye = np.linalg.norm(right_eye - left_eye)

        if inter_eye < 1e-6:
            return 0.0, 0.0

        # Yaw: nose horizontal offset from eye center, normalized
        nose_offset_x = (nose[0] - eye_center[0]) / inter_eye
        yaw = float(np.degrees(np.arctan2(nose_offset_x, 1.0)))

        # Pitch: nose vertical offset from eye center, normalized
        nose_offset_y = (nose[1] - eye_center[1]) / inter_eye
        # Typical ratio is ~0.6 for frontal; deviation indicates pitch
        pitch_ratio = nose_offset_y - 0.6
        pitch = float(np.degrees(np.arctan2(pitch_ratio, 1.0)))

        return yaw, pitch
