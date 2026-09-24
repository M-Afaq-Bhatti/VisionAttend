"""
Anti-Spoofing / Liveness Detection Module.

Uses temporal movement detection (blink/micro-expression analysis) across
multiple frames to distinguish live humans from static photos/screens.

Why this approach works:
  - A real person naturally blinks (~15-20 times per minute) and has
    micro-movements even when "standing still".
  - A photo on a phone screen is STATIC — the eye region never changes.
  - Previous heuristic approaches (texture, color, Moiré) failed because
    the face crop of a high-quality photo looks identical to a real face.
  - Movement detection relies on a physical truth: living humans move.
"""

import collections
import logging
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SpoofStatus(Enum):
    LIVE = "live"
    SPOOF = "spoof"
    UNCERTAIN = "uncertain"


@dataclass
class LivenessResult:
    """Result of liveness analysis."""
    status: SpoofStatus
    liveness_score: float  # 0.0 to 1.0 (higher is more likely live)
    details: str = ""


class LivenessChecker:
    """
    Determines liveness by detecting eye blinks and facial micro-movements
    across consecutive video frames.

    How it works:
      1. For each frame, extract and normalize the eye region using
         facial landmarks (left eye + right eye).
      2. Store eye regions in a rolling buffer.
      3. After a warmup period, compare consecutive eye regions.
         A blink or micro-expression causes a spike in pixel difference;
         a static photo stays constant.
      4. If sufficient movement is detected → LIVE.
         If no movement after enough frames → SPOOF (likely a photo).

    Args:
        threshold: Movement score threshold (0-1) to classify as LIVE.
        warmup_frames: Number of frames to collect before making a decision.
        movement_diff_threshold: Minimum pixel difference to count as movement.
    """

    def __init__(
        self,
        threshold: float = 0.50,
        warmup_frames: int = 12,
        movement_diff_threshold: float = 0.03,
    ):
        self.threshold = threshold
        self.warmup_frames = warmup_frames
        self.movement_diff_threshold = movement_diff_threshold
        self._eye_buffer: collections.deque = collections.deque(maxlen=30)
        self._movement_detected = False
        logger.info(
            "LivenessChecker initialized: threshold=%.2f, warmup=%d frames",
            threshold, warmup_frames,
        )

    def reset(self) -> None:
        """Reset internal state. Call when a new person appears."""
        self._eye_buffer.clear()
        self._movement_detected = False
        logger.debug("LivenessChecker state reset")

    # ------------------------------------------------------------------
    # Eye region extraction
    # ------------------------------------------------------------------
    def _extract_eye_region(
        self, face_crop: np.ndarray, landmarks: np.ndarray
    ) -> np.ndarray | None:
        """
        Crop and normalize the eye region from a face image.

        Uses the first two landmarks (left eye center, right eye center)
        from InsightFace's 5-point landmark set to define the region.

        Returns a fixed-size grayscale float32 array (20×60), or None.
        """
        if landmarks is None or len(landmarks) < 2:
            return None

        h, w = face_crop.shape[:2]
        left_eye = landmarks[0]   # (x, y)
        right_eye = landmarks[1]  # (x, y)

        # Compute eye distance for proportional cropping
        eye_dist = abs(right_eye[0] - left_eye[0])
        if eye_dist < 5:
            return None

        pad_x = int(eye_dist * 0.35)
        pad_y = int(eye_dist * 0.25)

        min_x = max(0, int(min(left_eye[0], right_eye[0])) - pad_x)
        max_x = min(w, int(max(left_eye[0], right_eye[0])) + pad_x)
        min_y = max(0, int(min(left_eye[1], right_eye[1])) - pad_y)
        max_y = min(h, int(max(left_eye[1], right_eye[1])) + pad_y)

        if max_x <= min_x or max_y <= min_y:
            return None

        eye_crop = face_crop[min_y:max_y, min_x:max_x]
        if eye_crop.size == 0:
            return None

        # Normalize to fixed size and convert to float grayscale
        eye_gray = cv2.cvtColor(
            cv2.resize(eye_crop, (60, 20)), cv2.COLOR_BGR2GRAY
        ).astype(np.float32) / 255.0

        return eye_gray

    # ------------------------------------------------------------------
    # Movement scoring
    # ------------------------------------------------------------------
    def _compute_movement_score(self) -> float:
        """
        Analyze the eye region buffer for blinks / micro-movements.

        Compares consecutive frames and looks for at least one significant
        spike in pixel-level difference (indicating a blink or expression
        change). Static photos produce near-zero differences.

        Returns a score from 0.0 (static) to 1.0 (clear movement).
        """
        frames = list(self._eye_buffer)
        if len(frames) < 2:
            return 0.0

        # Compute frame-to-frame differences
        diffs = []
        for i in range(1, len(frames)):
            diff = np.mean(np.abs(frames[i] - frames[i - 1]))
            diffs.append(diff)

        max_diff = max(diffs)
        std_diff = np.std(diffs)

        # Scoring logic:
        # - Real person: diffs are mostly small (micro-movements ~0.01-0.02)
        #   but with occasional spikes (blinks ~0.04-0.15)
        # - Static photo: all diffs are near-zero (~0.00-0.01), just noise
        #
        # We look at max_diff (did ANY blink happen?) and std_diff
        # (is there variation in the diffs, indicating natural movement?).

        score = 0.0

        # Primary signal: at least one significant change (blink)
        if max_diff > 0.06:
            score += 0.50  # Strong blink detected
        elif max_diff > self.movement_diff_threshold:
            score += 0.30  # Moderate movement

        # Secondary signal: variation in diffs (natural micro-movement)
        if std_diff > 0.015:
            score += 0.35  # Natural movement variation
        elif std_diff > 0.008:
            score += 0.15  # Some variation

        # Small bonus if overall mean diff is above noise floor
        mean_diff = np.mean(diffs)
        if mean_diff > 0.012:
            score += 0.15

        return min(1.0, score)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(
        self,
        face_crop: np.ndarray,
        landmarks: np.ndarray | None = None,
    ) -> LivenessResult:
        """
        Analyze a face crop for liveness.

        When landmarks are provided (preferred), uses temporal movement
        detection across frames. Falls back to a permissive basic check
        when no landmarks are available.

        Args:
            face_crop: BGR numpy array of the detected face.
            landmarks: Optional (N, 2) array of facial landmarks.

        Returns:
            LivenessResult with status, score, and explanation.
        """
        if face_crop is None or face_crop.size == 0:
            return LivenessResult(SpoofStatus.UNCERTAIN, 0.0, "Empty image")

        # --- Movement-based detection (when landmarks available) ---
        if landmarks is not None:
            eye_region = self._extract_eye_region(face_crop, landmarks)

            if eye_region is not None:
                self._eye_buffer.append(eye_region)

                # Warmup: collecting frames
                n = len(self._eye_buffer)
                if n < self.warmup_frames:
                    return LivenessResult(
                        SpoofStatus.UNCERTAIN, 0.5,
                        f"Verifying liveness... look at the camera "
                        f"({n}/{self.warmup_frames})"
                    )

                # Compute movement score
                movement_score = self._compute_movement_score()

                if movement_score >= self.threshold:
                    self._movement_detected = True
                    return LivenessResult(
                        SpoofStatus.LIVE, movement_score,
                        f"Liveness confirmed (movement={movement_score:.2f})"
                    )
                else:
                    return LivenessResult(
                        SpoofStatus.SPOOF, movement_score,
                        f"No facial movement detected — possible photo "
                        f"(score={movement_score:.2f})"
                    )

        # --- Fallback: no landmarks, use basic check ---
        return self._basic_check(face_crop)

    def _basic_check(self, face_crop: np.ndarray) -> LivenessResult:
        """
        Permissive fallback when landmarks are not available.
        Only rejects extremely blurry or blank images.
        """
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        if laplacian_var < 10.0:
            return LivenessResult(
                SpoofStatus.UNCERTAIN, 0.2,
                f"Image too blurry (var={laplacian_var:.1f})"
            )

        # Without landmarks we can't do movement detection,
        # so we pass with a moderate score
        return LivenessResult(
            SpoofStatus.LIVE, 0.65,
            "Basic check passed (no landmarks for movement detection)"
        )
