"""
Recognition pipeline.

End-to-end pipeline for recognizing a person in a single frame:
  Frame → Detection → Quality → Embedding → Matching → Result
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.cv.detection.detector import FaceDetector, DetectedFace, DetectionResult
from src.cv.quality.checker import FaceQualityChecker, QualityStatus, QualityResult
from src.cv.embeddings.extractor import FaceEmbedder, EmbeddingResult
from src.cv.recognition.matcher import IdentityMatcher, MatchResult
from src.antispoof.checker import LivenessChecker, SpoofStatus, LivenessResult

logger = logging.getLogger(__name__)


@dataclass
class RecognitionResult:
    """Full result of processing a single frame through the recognition pipeline."""
    # Detection
    num_faces: int = 0
    detection_latency_ms: float = 0.0

    # Per-face results (for the primary/best face)
    face_detected: bool = False
    face_bbox: Optional[np.ndarray] = None
    face_crop: Optional[np.ndarray] = None
    face_confidence: float = 0.0
    face_landmarks: Optional[np.ndarray] = None

    # Quality
    quality_status: QualityStatus = QualityStatus.NO_FACE
    quality_details: str = ""

    # Liveness
    liveness_status: SpoofStatus = SpoofStatus.UNCERTAIN
    liveness_score: float = 0.0
    liveness_details: str = ""

    # Embedding
    embedding: Optional[np.ndarray] = None
    embedding_latency_ms: float = 0.0

    # Matching
    match_result: Optional[MatchResult] = None
    recognized: bool = False
    identity: Optional[str] = None
    employee_id: Optional[int] = None
    similarity: float = 0.0

    # Timing
    total_latency_ms: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict (no numpy arrays)."""
        d = {
            "num_faces": self.num_faces,
            "face_detected": self.face_detected,
            "face_confidence": round(self.face_confidence, 3),
            "quality_status": self.quality_status.value if self.quality_status else None,
            "quality_details": self.quality_details,
            "liveness_status": self.liveness_status.value if self.liveness_status else None,
            "liveness_score": round(self.liveness_score, 3),
            "liveness_details": self.liveness_details,
            "recognized": self.recognized,
            "identity": self.identity,
            "employee_id": self.employee_id,
            "similarity": round(self.similarity, 4),
            "detection_latency_ms": round(self.detection_latency_ms, 1),
            "embedding_latency_ms": round(self.embedding_latency_ms, 1),
            "total_latency_ms": round(self.total_latency_ms, 1),
        }
        if self.match_result:
            d["threshold"] = self.match_result.threshold
        return d


class RecognitionPipeline:
    """
    End-to-end face recognition pipeline.

    Chains detection -> quality check -> liveness -> embedding -> identity matching.

    Args:
        detector: Initialized FaceDetector.
        embedder: Initialized FaceEmbedder.
        matcher: IdentityMatcher with loaded gallery.
        quality_checker: FaceQualityChecker instance.
        liveness_checker: Optional LivenessChecker instance.
    """

    def __init__(
        self,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        matcher: IdentityMatcher,
        quality_checker: Optional[FaceQualityChecker] = None,
        liveness_checker: Optional[LivenessChecker] = None,
    ):
        self.detector = detector
        self.embedder = embedder
        self.matcher = matcher
        self.quality_checker = quality_checker or FaceQualityChecker()
        self.liveness_checker = liveness_checker or LivenessChecker()

    def process_frame(self, frame: np.ndarray) -> RecognitionResult:
        """
        Process a single BGR frame through the full recognition pipeline.

        Returns:
            RecognitionResult with detection, quality, liveness, embedding, and matching info.
        """
        t_start = time.perf_counter()
        result = RecognitionResult()

        # --- Step 1: Face Detection ---
        det = self.detector.detect(frame, max_faces=5)
        result.num_faces = det.num_faces
        result.detection_latency_ms = det.latency_ms

        if det.num_faces == 0:
            result.quality_status = QualityStatus.NO_FACE
            result.quality_details = "No face detected"
            result.total_latency_ms = (time.perf_counter() - t_start) * 1000
            return result

        # Use the highest-confidence face
        best_face = det.faces[0]
        result.face_detected = True
        result.face_bbox = best_face.bbox
        result.face_crop = best_face.face_crop
        result.face_confidence = best_face.confidence
        result.face_landmarks = best_face.landmarks

        # --- Step 2: Quality Check ---
        quality = self.quality_checker.check(
            best_face.face_crop, best_face.landmarks
        )
        result.quality_status = quality.status
        result.quality_details = quality.details

        if quality.status != QualityStatus.VALID:
            result.total_latency_ms = (time.perf_counter() - t_start) * 1000
            return result
            
        # --- Step 3: Liveness Check ---
        liveness = self.liveness_checker.analyze(best_face.face_crop, best_face.landmarks)
        result.liveness_status = liveness.status
        result.liveness_score = liveness.liveness_score
        result.liveness_details = liveness.details
        
        if liveness.status != SpoofStatus.LIVE:
            logger.warning("Spoof detected! Score: %.3f - %s", liveness.liveness_score, liveness.details)
            result.total_latency_ms = (time.perf_counter() - t_start) * 1000
            return result

        # --- Step 4: Embedding ---
        t_emb = time.perf_counter()
        try:
            emb_result = self.embedder.extract(
                face_img=best_face.face_crop,
                landmarks=best_face.landmarks,
                frame=frame,
            )
            result.embedding = emb_result.embedding
            result.embedding_latency_ms = emb_result.latency_ms
        except Exception as e:
            logger.error("Embedding extraction failed: %s", e)
            result.quality_status = QualityStatus.LOW_QUALITY
            result.quality_details = f"Embedding extraction error: {e}"
            result.total_latency_ms = (time.perf_counter() - t_start) * 1000
            return result

        # --- Step 4: Identity Matching ---
        match = self.matcher.match(result.embedding)
        result.match_result = match
        result.recognized = match.recognized
        result.identity = match.identity
        result.employee_id = match.employee_id
        result.similarity = match.similarity

        result.total_latency_ms = (time.perf_counter() - t_start) * 1000

        if result.recognized:
            logger.info(
                "Recognized %s (sim=%.3f) in %.0f ms",
                result.identity, result.similarity, result.total_latency_ms,
            )
        else:
            logger.debug(
                "Unknown face (best_sim=%.3f, threshold=%.3f) in %.0f ms",
                result.similarity, match.threshold, result.total_latency_ms,
            )

        return result

    def process_frame_multi(self, frame: np.ndarray) -> list[RecognitionResult]:
        """
        Process all faces in a frame. Returns a list of RecognitionResults.
        """
        t_start = time.perf_counter()
        det = self.detector.detect(frame, max_faces=10)

        if det.num_faces == 0:
            return [RecognitionResult(
                quality_status=QualityStatus.NO_FACE,
                quality_details="No face detected",
                total_latency_ms=(time.perf_counter() - t_start) * 1000,
            )]

        results = []
        for face in det.faces:
            result = RecognitionResult()
            result.num_faces = det.num_faces
            result.detection_latency_ms = det.latency_ms
            result.face_detected = True
            result.face_bbox = face.bbox
            result.face_crop = face.face_crop
            result.face_confidence = face.confidence
            result.face_landmarks = face.landmarks

            # Quality
            quality = self.quality_checker.check(face.face_crop, face.landmarks)
            result.quality_status = quality.status
            result.quality_details = quality.details

            if quality.status != QualityStatus.VALID:
                result.total_latency_ms = (time.perf_counter() - t_start) * 1000
                results.append(result)
                continue
                
            # Liveness
            liveness = self.liveness_checker.analyze(face.face_crop, face.landmarks)
            result.liveness_status = liveness.status
            result.liveness_score = liveness.liveness_score
            result.liveness_details = liveness.details
            
            if liveness.status != SpoofStatus.LIVE:
                result.total_latency_ms = (time.perf_counter() - t_start) * 1000
                results.append(result)
                continue

            # Embedding
            try:
                emb_result = self.embedder.extract(
                    face_img=face.face_crop,
                    landmarks=face.landmarks,
                    frame=frame,
                )
                result.embedding = emb_result.embedding
                result.embedding_latency_ms = emb_result.latency_ms
            except Exception as e:
                logger.error("Embedding failed: %s", e)
                result.quality_status = QualityStatus.LOW_QUALITY
                result.quality_details = str(e)
                result.total_latency_ms = (time.perf_counter() - t_start) * 1000
                results.append(result)
                continue

            # Matching
            match = self.matcher.match(result.embedding)
            result.match_result = match
            result.recognized = match.recognized
            result.identity = match.identity
            result.employee_id = match.employee_id
            result.similarity = match.similarity

            result.total_latency_ms = (time.perf_counter() - t_start) * 1000
            results.append(result)

        return results
