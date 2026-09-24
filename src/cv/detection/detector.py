"""
Face detection module using InsightFace.

Wraps the InsightFace FaceAnalysis detector to provide a clean
interface for face detection and landmark extraction.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectedFace:
    """Represents a single detected face in a frame."""
    bbox: np.ndarray            # [x1, y1, x2, y2]
    confidence: float
    landmarks: Optional[np.ndarray] = None  # (5, 2) for 5-point
    face_crop: Optional[np.ndarray] = None  # BGR crop from original image
    embedding: Optional[np.ndarray] = None  # Filled later by embedding module


@dataclass
class DetectionResult:
    """Result from face detection on a single frame."""
    faces: list[DetectedFace] = field(default_factory=list)
    num_faces: int = 0
    latency_ms: float = 0.0
    frame_shape: tuple = ()


class FaceDetector:
    """
    Face detector powered by InsightFace's SCRFD model.

    Loads once and reuses across frames. Supports both CPU and CUDA.

    Args:
        model_name: InsightFace model pack name (default 'buffalo_l').
        ctx_id: Execution provider context. 0 = GPU, -1 = CPU.
        det_size: Detection input size (width, height).
        det_thresh: Minimum detection confidence threshold.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        ctx_id: int = -1,
        det_size: tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
    ):
        self._model_name = model_name
        self._ctx_id = ctx_id
        self._det_size = det_size
        self._det_thresh = det_thresh
        self._app = None
        self._initialized = False

    def initialize(self) -> None:
        """Load the InsightFace model. Call once before detect()."""
        if self._initialized:
            return

        try:
            from insightface.app import FaceAnalysis

            logger.info(
                "Loading InsightFace model '%s' (ctx_id=%d, det_size=%s)",
                self._model_name, self._ctx_id, self._det_size,
            )
            t0 = time.perf_counter()

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self._ctx_id >= 0 \
                else ["CPUExecutionProvider"]

            self._app = FaceAnalysis(
                name=self._model_name,
                providers=providers,
            )
            self._app.prepare(
                ctx_id=self._ctx_id,
                det_size=self._det_size,
                det_thresh=self._det_thresh,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info("InsightFace model loaded in %.0f ms", elapsed)
            self._initialized = True
        except Exception:
            logger.exception("Failed to initialize InsightFace model")
            raise

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def detect(self, frame: np.ndarray, max_faces: int = 10) -> DetectionResult:
        """
        Detect faces in a BGR frame.

        Args:
            frame: BGR image as numpy array.
            max_faces: Maximum number of faces to return.

        Returns:
            DetectionResult containing detected faces with bboxes and landmarks.
        """
        if not self._initialized:
            raise RuntimeError("FaceDetector not initialized. Call initialize() first.")

        t0 = time.perf_counter()
        # insightface 2.0: app.get() returns list of Face dicts
        raw_faces = self._app.get(frame)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Face is a dict subclass with keys: bbox, kps, det_score, embedding, etc.
        # Sort by detection score descending and limit
        raw_faces = sorted(
            raw_faces,
            key=lambda f: f.get("det_score", 0.0),
            reverse=True,
        )[:max_faces]

        detected = []
        for face_dict in raw_faces:
            bbox_raw = face_dict.get("bbox")
            if bbox_raw is None:
                continue
            bbox = np.array(bbox_raw).astype(int)

            # Clamp to frame bounds
            h, w = frame.shape[:2]
            x1 = max(0, bbox[0])
            y1 = max(0, bbox[1])
            x2 = min(w, bbox[2])
            y2 = min(h, bbox[3])

            if x2 <= x1 or y2 <= y1:
                continue

            face_crop = frame[y1:y2, x1:x2].copy()

            landmarks = None
            kps = face_dict.get("kps")
            if kps is not None:
                landmarks = np.array(kps).astype(np.float32)

            det_score = float(face_dict.get("det_score", 0.0))

            detected.append(DetectedFace(
                bbox=np.array([x1, y1, x2, y2]),
                confidence=det_score,
                landmarks=landmarks,
                face_crop=face_crop,
            ))

        return DetectionResult(
            faces=detected,
            num_faces=len(detected),
            latency_ms=latency_ms,
            frame_shape=frame.shape[:2],
        )

    def detect_single(self, frame: np.ndarray) -> Optional[DetectedFace]:
        """
        Convenience: detect and return the highest-confidence face, or None.
        """
        result = self.detect(frame, max_faces=1)
        if result.faces:
            return result.faces[0]
        return None
