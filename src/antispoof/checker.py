"""
Anti-Spoofing / Liveness Detection Module.

Analyzes face crops to determine if they are live humans or spoof
attempts (printed photos, digital screens, masks).

Uses a multi-signal approach:
  1. Texture analysis (LBP variance) — real skin has unique micro-texture
  2. Color distribution — screens have a blue/cool tint
  3. Moiré pattern detection — phone/monitor pixels create interference
  4. Specular reflection — screens produce uniform highlights
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
    Evaluates the liveness of a detected face using multiple visual cues.
    
    Uses a weighted combination of:
      - Color temperature (screens are bluer than natural light)
      - Texture micro-patterns via Local Binary Patterns (LBP)
      - Moiré/frequency analysis (screens produce periodic artifacts)
      - Reflection uniformity (screens light faces too evenly)
    
    In a full production environment, this is designed to be replaced 
    or augmented with a Deep Learning model (e.g., MiniFASNet ONNX).
    """

    def __init__(
        self,
        threshold: float = 0.50,
        min_blur_var: float = 15.0,
        max_glare_ratio: float = 0.08,
    ):
        """
        Args:
            threshold: Minimum score (0-1) to be considered LIVE.
            min_blur_var: Variance threshold to detect extremely blurry images.
            max_glare_ratio: Max percentage of overexposed pixels.
        """
        self.threshold = threshold
        self.min_blur_var = min_blur_var
        self.max_glare_ratio = max_glare_ratio
        logger.info("LivenessChecker initialized with threshold=%.2f", threshold)

    def _compute_color_score(self, face_bgr: np.ndarray) -> float:
        """
        Detect screen color temperature bias.
        
        Screens (especially phones) emit more blue light than natural 
        ambient lighting. A real face under normal light has warmer tones.
        
        Returns: 0.0 (screen-like) to 1.0 (natural skin tones)
        """
        b, g, r = cv2.split(face_bgr.astype(np.float32))
        
        b_mean, g_mean, r_mean = b.mean(), g.mean(), r.mean()
        total = b_mean + g_mean + r_mean + 1e-6
        
        # Ratio of blue channel — screens have higher blue ratio
        blue_ratio = b_mean / total
        
        # Natural skin typically has blue_ratio < 0.30
        # Screen-lit faces often have blue_ratio > 0.34
        if blue_ratio > 0.36:
            return 0.1  # Very screen-like
        elif blue_ratio > 0.33:
            return 0.4  # Suspicious
        else:
            return 0.9  # Natural lighting
    
    def _compute_lbp_score(self, gray: np.ndarray) -> float:
        """
        Local Binary Pattern (LBP) texture analysis.
        
        Real skin has rich, varied micro-texture (pores, fine lines).
        Phone screens showing a photo have smoother, more uniform texture
        because the webcam can't resolve the phone's sub-pixels into 
        natural-looking skin detail.
        
        Returns: 0.0 (screen-like uniform) to 1.0 (real skin texture)
        """
        # Resize to standard size for consistent analysis
        resized = cv2.resize(gray, (128, 128))
        
        # Compute simple LBP
        lbp = np.zeros_like(resized, dtype=np.uint8)
        for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]:
            shifted = np.roll(np.roll(resized, dy, axis=0), dx, axis=1)
            lbp = (lbp << 1) | (shifted >= resized).astype(np.uint8)
        
        # Calculate histogram of LBP values
        hist, _ = np.histogram(lbp.ravel(), bins=64, range=(0, 256))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-6)
        
        # Entropy of LBP histogram — real skin has higher entropy (more varied texture)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Real faces: entropy typically 4.5-6.0
        # Screen photos: entropy typically 3.0-4.5
        if entropy > 4.8:
            return 0.9  # Rich real skin texture
        elif entropy > 4.0:
            return 0.6  # Could be either
        else:
            return 0.2  # Too uniform — likely screen/print

    def _compute_moire_score(self, gray: np.ndarray) -> float:
        """
        Detect Moiré patterns from digital screens.
        
        When a webcam captures a phone/monitor screen, the interaction 
        between the two pixel grids creates periodic interference patterns 
        (Moiré) visible in the frequency domain as strong peaks.
        
        Returns: 0.0 (strong Moiré = screen) to 1.0 (no Moiré = real)
        """
        resized = cv2.resize(gray, (128, 128)).astype(np.float32)
        
        # Apply FFT
        f_transform = np.fft.fft2(resized)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log1p(np.abs(f_shift))
        
        # Mask out the DC component (center)
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        mask_size = 5
        magnitude[cy-mask_size:cy+mask_size, cx-mask_size:cx+mask_size] = 0
        
        # Moiré creates strong isolated peaks in the high-frequency region
        # Real faces have a smoother, more gradual frequency falloff
        
        # Look at the ratio of max peak to mean — high ratio = periodic pattern
        mean_mag = magnitude.mean()
        max_mag = magnitude.max()
        
        if mean_mag < 1e-6:
            return 0.5  # Can't determine
        
        peak_ratio = max_mag / (mean_mag + 1e-6)
        
        # Real faces: peak_ratio typically < 8
        # Screen captures: peak_ratio typically > 10 due to Moiré
        if peak_ratio > 12:
            return 0.1  # Strong Moiré pattern
        elif peak_ratio > 8:
            return 0.4  # Some periodic artifacts
        else:
            return 0.9  # Natural frequency distribution

    def _compute_reflection_score(self, gray: np.ndarray) -> float:
        """
        Analyze reflection uniformity.
        
        Screens illuminate faces very evenly (uniform flat light source).
        Natural lighting creates shadows and gradients across the face.
        
        Returns: 0.0 (too uniform = screen-lit) to 1.0 (natural shadows)
        """
        resized = cv2.resize(gray, (64, 64))
        
        # Divide face into quadrants and compare brightness
        h, w = resized.shape
        mid_h, mid_w = h // 2, w // 2
        
        quadrants = [
            resized[:mid_h, :mid_w],   # top-left
            resized[:mid_h, mid_w:],    # top-right
            resized[mid_h:, :mid_w],    # bottom-left
            resized[mid_h:, mid_w:],    # bottom-right
        ]
        
        means = [q.mean() for q in quadrants]
        overall_mean = np.mean(means)
        
        if overall_mean < 1e-6:
            return 0.5
        
        # Coefficient of variation across quadrants
        variation = np.std(means) / (overall_mean + 1e-6)
        
        # Real faces under natural light: variation > 0.08 (shadows on one side)
        # Screen-lit faces: variation < 0.05 (very even illumination)
        if variation > 0.10:
            return 0.9  # Natural lighting with shadows
        elif variation > 0.05:
            return 0.6  # Moderate variation
        else:
            return 0.2  # Too uniform — possibly screen-lit

    def analyze(self, face_crop: np.ndarray) -> LivenessResult:
        """
        Analyze a face crop for liveness using multiple signals.
        
        Args:
            face_crop: BGR numpy array of the detected face.
            
        Returns:
            LivenessResult containing the decision and score.
        """
        if face_crop is None or face_crop.size == 0:
            return LivenessResult(SpoofStatus.UNCERTAIN, 0.0, "Empty image")

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        
        # Quick reject: extremely blurry images (not even a screen, just garbage)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < self.min_blur_var:
            return LivenessResult(
                SpoofStatus.UNCERTAIN, 0.1,
                f"Image too blurry to analyze (var={laplacian_var:.1f})"
            )
        
        # Glare check (very heavy glare = useless frame)
        _, glare_mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
        glare_ratio = cv2.countNonZero(glare_mask) / (gray.shape[0] * gray.shape[1])
        if glare_ratio > self.max_glare_ratio * 3:
            return LivenessResult(
                SpoofStatus.UNCERTAIN, 0.1,
                f"Excessive glare (ratio={glare_ratio:.3f})"
            )

        # --- Compute all signals ---
        color_score = self._compute_color_score(face_crop)
        lbp_score = self._compute_lbp_score(gray)
        moire_score = self._compute_moire_score(gray)
        reflection_score = self._compute_reflection_score(gray)
        
        # Weighted combination
        # Moiré and color are the strongest indicators for phone screen spoofs
        final_score = (
            0.30 * color_score +
            0.25 * lbp_score +
            0.25 * moire_score +
            0.20 * reflection_score
        )
        
        details = (
            f"color={color_score:.2f} lbp={lbp_score:.2f} "
            f"moire={moire_score:.2f} reflect={reflection_score:.2f} "
            f"→ final={final_score:.2f}"
        )
        
        logger.debug("Liveness signals: %s", details)

        if final_score >= self.threshold:
            return LivenessResult(
                status=SpoofStatus.LIVE, 
                liveness_score=final_score,
                details=f"LIVE: {details}"
            )
        else:
            return LivenessResult(
                status=SpoofStatus.SPOOF,
                liveness_score=final_score,
                details=f"SPOOF: {details}"
            )
