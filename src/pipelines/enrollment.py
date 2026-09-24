"""
Enrollment pipeline.

Handles the full workflow of registering an employee's face:
  1. Accept multiple images (files or webcam frames).
  2. Detect face in each image.
  3. Run quality checks.
  4. Align face.
  5. Generate embedding.
  6. Store embeddings on disk and in the database.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.cv.detection.detector import FaceDetector, DetectedFace
from src.cv.quality.checker import FaceQualityChecker, QualityStatus, QualityResult
from src.cv.embeddings.extractor import FaceEmbedder, EmbeddingResult

logger = logging.getLogger(__name__)


@dataclass
class EnrollmentImageResult:
    """Result of processing a single enrollment image."""
    index: int
    accepted: bool
    quality_status: QualityStatus
    quality_details: str = ""
    embedding: Optional[np.ndarray] = None
    face_crop: Optional[np.ndarray] = None
    confidence: float = 0.0


@dataclass
class EnrollmentResult:
    """Result of the full enrollment pipeline for one employee."""
    employee_code: str
    success: bool
    total_images: int = 0
    accepted_images: int = 0
    rejected_images: int = 0
    image_results: list[EnrollmentImageResult] = field(default_factory=list)
    embeddings: list[np.ndarray] = field(default_factory=list)
    representative_embedding: Optional[np.ndarray] = None
    total_latency_ms: float = 0.0
    error: str = ""


class EnrollmentPipeline:
    """
    Orchestrates face enrollment for a single employee.

    Requires an initialized FaceDetector and FaceEmbedder.

    Args:
        detector: Initialized FaceDetector.
        embedder: Initialized FaceEmbedder.
        quality_checker: FaceQualityChecker instance.
        enrollment_dir: Base directory for storing enrollment data.
        min_accepted_images: Minimum number of valid images required.
        max_images: Maximum images to process.
    """

    def __init__(
        self,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        quality_checker: Optional[FaceQualityChecker] = None,
        enrollment_dir: str = "data/enrollment",
        min_accepted_images: int = 3,
        max_images: int = 10,
    ):
        self.detector = detector
        self.embedder = embedder
        self.quality_checker = quality_checker or FaceQualityChecker()
        self.enrollment_dir = Path(enrollment_dir)
        self.min_accepted_images = min_accepted_images
        self.max_images = max_images

    def enroll(
        self,
        employee_code: str,
        images: list[np.ndarray],
    ) -> EnrollmentResult:
        """
        Process multiple face images and generate enrollment embeddings.

        Args:
            employee_code: Unique employee identifier.
            images: List of BGR images (numpy arrays).

        Returns:
            EnrollmentResult with accepted embeddings and diagnostics.
        """
        t0 = time.perf_counter()
        result = EnrollmentResult(
            employee_code=employee_code,
            success=False,
            total_images=len(images),
        )

        if not images:
            result.error = "No images provided"
            return result

        images = images[:self.max_images]
        result.total_images = len(images)

        # Create employee enrollment directory
        emp_dir = self.enrollment_dir / employee_code
        emp_dir.mkdir(parents=True, exist_ok=True)

        for idx, img in enumerate(images):
            img_result = self._process_single_image(idx, img, employee_code, emp_dir)
            result.image_results.append(img_result)

            if img_result.accepted and img_result.embedding is not None:
                result.accepted_images += 1
                result.embeddings.append(img_result.embedding)
            else:
                result.rejected_images += 1

        # Check minimum accepted
        if result.accepted_images < self.min_accepted_images:
            result.error = (
                f"Only {result.accepted_images} images accepted, "
                f"minimum {self.min_accepted_images} required"
            )
            result.total_latency_ms = (time.perf_counter() - t0) * 1000
            return result

        # Compute representative embedding (mean of all accepted, re-normalized)
        if result.embeddings:
            stacked = np.stack(result.embeddings)
            mean_emb = stacked.mean(axis=0)
            norm = np.linalg.norm(mean_emb)
            if norm > 0:
                mean_emb = mean_emb / norm
            result.representative_embedding = mean_emb

        # Save embeddings to disk
        self._save_embeddings(employee_code, result.embeddings, emp_dir)

        result.success = True
        result.total_latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Enrollment complete for %s: %d/%d accepted in %.0f ms",
            employee_code,
            result.accepted_images,
            result.total_images,
            result.total_latency_ms,
        )
        return result

    def _process_single_image(
        self,
        index: int,
        image: np.ndarray,
        employee_code: str,
        emp_dir: Path,
    ) -> EnrollmentImageResult:
        """Process a single enrollment image through detection, quality, and embedding."""
        # Detect face
        det_result = self.detector.detect(image, max_faces=5)

        if det_result.num_faces == 0:
            return EnrollmentImageResult(
                index=index,
                accepted=False,
                quality_status=QualityStatus.NO_FACE,
                quality_details="No face detected in image",
            )

        if det_result.num_faces > 1:
            # For enrollment, we require exactly one face
            return EnrollmentImageResult(
                index=index,
                accepted=False,
                quality_status=QualityStatus.MULTIPLE_FACES,
                quality_details=f"{det_result.num_faces} faces detected, expected 1",
            )

        face = det_result.faces[0]

        # Quality check
        quality = self.quality_checker.check(face.face_crop, face.landmarks)
        if quality.status != QualityStatus.VALID:
            return EnrollmentImageResult(
                index=index,
                accepted=False,
                quality_status=quality.status,
                quality_details=quality.details,
                face_crop=face.face_crop,
                confidence=face.confidence,
            )

        # Extract embedding
        try:
            emb_result = self.embedder.extract(
                face_img=face.face_crop,
                landmarks=face.landmarks,
                frame=image,
            )
        except Exception as e:
            logger.error("Embedding extraction failed for image %d: %s", index, e)
            return EnrollmentImageResult(
                index=index,
                accepted=False,
                quality_status=QualityStatus.LOW_QUALITY,
                quality_details=f"Embedding extraction failed: {e}",
                face_crop=face.face_crop,
                confidence=face.confidence,
            )

        # Save face crop
        crop_path = emp_dir / f"face_{index:03d}.jpg"
        cv2.imwrite(str(crop_path), face.face_crop)

        return EnrollmentImageResult(
            index=index,
            accepted=True,
            quality_status=QualityStatus.VALID,
            quality_details="Passed all checks",
            embedding=emb_result.embedding,
            face_crop=face.face_crop,
            confidence=face.confidence,
        )

    @staticmethod
    def _save_embeddings(
        employee_code: str,
        embeddings: list[np.ndarray],
        emp_dir: Path,
    ) -> str:
        """
        Save all enrollment embeddings as a single .npz file.

        Returns:
            Path to the saved file.
        """
        save_path = emp_dir / "embeddings.npz"
        np.savez(
            str(save_path),
            embeddings=np.stack(embeddings),
            employee_code=employee_code,
        )
        logger.info("Saved %d embeddings to %s", len(embeddings), save_path)
        return str(save_path)

    @staticmethod
    def load_embeddings(employee_code: str, enrollment_dir: str = "data/enrollment") -> Optional[np.ndarray]:
        """
        Load saved embeddings for an employee.

        Returns:
            (N, 512) array of embeddings, or None if not found.
        """
        path = Path(enrollment_dir) / employee_code / "embeddings.npz"
        if not path.exists():
            return None
        data = np.load(str(path))
        return data["embeddings"]
