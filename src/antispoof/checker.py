"""
Anti-Spoofing / Liveness Detection Module.

Analyzes face crops to determine if they are live humans or spoof
attempts (printed photos, digital screens, masks).
"""

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
    Evaluates the liveness of a detected face.
    
    For this MVP, it uses texture analysis (Laplacian variance) and 
    specular reflection analysis to detect common screen/photo spoofs.
    In a full production environment, this is designed to be replaced 
    or augmented with a Deep Learning model (e.g., MiniFASNet ONNX).
    """

    def __init__(
        self,
        threshold: float = 0.60,
        min_blur_var: float = 30.0,
        max_glare_ratio: float = 0.05,
    ):
        """
        Args:
            threshold: Minimum score (0-1) to be considered LIVE.
            min_blur_var: Variance threshold to detect printed/screen blur.
            max_glare_ratio: Max percentage of overexposed pixels (glare from screens).
        """
        self.threshold = threshold
        self.min_blur_var = min_blur_var
        self.max_glare_ratio = max_glare_ratio
        logger.info("LivenessChecker initialized with threshold=%.2f", threshold)

    def analyze(self, face_crop: np.ndarray) -> LivenessResult:
        """
        Analyze a face crop for liveness.
        
        Args:
            face_crop: BGR numpy array of the detected face.
            
        Returns:
            LivenessResult containing the decision and score.
        """
        if face_crop is None or face_crop.size == 0:
            return LivenessResult(SpoofStatus.UNCERTAIN, 0.0, "Empty image")

        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        
        # 1. Texture/Sharpness analysis (Screens/Photos often lack fine high-freq details)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Glare/Specular highlight analysis (Screens reflect light heavily)
        # Count pixels that are almost pure white
        _, glare_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        glare_ratio = cv2.countNonZero(glare_mask) / (gray.shape[0] * gray.shape[1])
        
        # Calculate a pseudo-score (0.0 to 1.0)
        # A good live face has decent sharpness (laplacian) and low glare.
        
        # Normalize blur score (assume max expected is around 1000)
        norm_blur = min(1.0, laplacian_var / 500.0)
        
        # Penalty for excessive glare
        glare_penalty = 0.0
        if glare_ratio > self.max_glare_ratio:
            # Drop score significantly if there is too much glare
            glare_penalty = min(1.0, (glare_ratio - self.max_glare_ratio) * 10)
            
        final_score = max(0.0, norm_blur - glare_penalty)
        
        # Decision
        if laplacian_var < self.min_blur_var:
            return LivenessResult(
                status=SpoofStatus.SPOOF,
                liveness_score=final_score,
                details=f"Texture too smooth (var={laplacian_var:.1f})"
            )
            
        if glare_ratio > self.max_glare_ratio * 2:
            return LivenessResult(
                status=SpoofStatus.SPOOF,
                liveness_score=final_score,
                details=f"Screen glare detected (ratio={glare_ratio:.3f})"
            )

        if final_score >= self.threshold:
            return LivenessResult(
                status=SpoofStatus.LIVE, 
                liveness_score=final_score,
                details="Passed texture and glare checks"
            )
        else:
            return LivenessResult(
                status=SpoofStatus.SPOOF,
                liveness_score=final_score,
                details="Liveness score below threshold"
            )

