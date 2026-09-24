"""
Face embedding extraction module.

Uses InsightFace's ArcFace model to produce 512-dimensional face
embeddings suitable for identity matching via cosine similarity.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Standard ArcFace alignment reference landmarks (for 112x112 input)
ARCFACE_REF_LANDMARKS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


@dataclass
class EmbeddingResult:
    """Result from embedding extraction."""
    embedding: np.ndarray       # (512,) normalized float32
    latency_ms: float = 0.0
    model_name: str = ""
    dimension: int = 512


class FaceEmbedder:
    """
    Extracts face embeddings using InsightFace's ArcFace recognition model.

    The model is loaded from the InsightFace model pack (e.g. 'buffalo_l')
    and produces L2-normalized 512-D embeddings.

    Args:
        model_name: InsightFace model pack name.
        ctx_id: 0 for GPU, -1 for CPU.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        ctx_id: int = -1,
    ):
        self._model_name = model_name
        self._ctx_id = ctx_id
        self._rec_model = None
        self._initialized = False
        self._embedding_dim = 512

    def initialize(self) -> None:
        """Load the recognition model from InsightFace model pack."""
        if self._initialized:
            return

        try:
            from insightface.app import FaceAnalysis

            logger.info("Loading ArcFace embedding model '%s'", self._model_name)
            t0 = time.perf_counter()

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self._ctx_id >= 0 \
                else ["CPUExecutionProvider"]

            app = FaceAnalysis(
                name=self._model_name,
                providers=providers,
            )
            app.prepare(ctx_id=self._ctx_id, det_size=(640, 640))

            # Extract the recognition model (ArcFaceONNX)
            self._rec_model = app.models.get("recognition")

            if self._rec_model is None:
                raise RuntimeError(
                    f"Could not find recognition model in pack '{self._model_name}'"
                )

            elapsed = (time.perf_counter() - t0) * 1000
            logger.info("ArcFace model loaded in %.0f ms", elapsed)
            self._initialized = True
        except Exception:
            logger.exception("Failed to initialize ArcFace model")
            raise

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def align_face(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        output_size: tuple[int, int] = (112, 112),
    ) -> np.ndarray:
        """
        Align a face using 5-point landmarks to the ArcFace reference.

        Args:
            frame: Full BGR frame.
            landmarks: (5, 2) facial landmarks in frame coordinates.
            output_size: Target crop size.

        Returns:
            Aligned face crop (112x112 BGR).
        """
        src_pts = landmarks.astype(np.float32)
        dst_pts = ARCFACE_REF_LANDMARKS.copy()

        # Compute similarity transform
        tform = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
        if tform is None:
            logger.warning("Affine transform failed, using simple resize")
            return cv2.resize(frame, output_size)

        aligned = cv2.warpAffine(frame, tform, output_size, borderValue=0)
        return aligned

    def extract(
        self,
        face_img: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        frame: Optional[np.ndarray] = None,
    ) -> EmbeddingResult:
        """
        Extract a face embedding.

        If landmarks and the full frame are provided, alignment is performed
        first. Otherwise, the face_img is used directly (resized).

        Args:
            face_img: Face crop (BGR).
            landmarks: Optional 5-point landmarks in frame coordinates.
            frame: Optional full frame (used with landmarks for alignment).

        Returns:
            EmbeddingResult with L2-normalized embedding vector.
        """
        if not self._initialized:
            raise RuntimeError("FaceEmbedder not initialized. Call initialize() first.")

        t0 = time.perf_counter()

        # Align if we have landmarks and a full frame
        if landmarks is not None and frame is not None and len(landmarks) >= 5:
            aligned = self.align_face(frame, landmarks[:5])
        else:
            aligned = cv2.resize(face_img, (112, 112))

        embedding = self._get_embedding(aligned)
        latency_ms = (time.perf_counter() - t0) * 1000

        return EmbeddingResult(
            embedding=embedding,
            latency_ms=latency_ms,
            model_name=self._model_name,
            dimension=len(embedding),
        )

    def _get_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Run the ArcFace ONNX model on an aligned 112x112 BGR face.

        Returns:
            L2-normalized embedding vector (512,).
        """
        # ArcFaceONNX.get_feat expects a blob: (1, 3, 112, 112)
        # Preprocessing: subtract mean, divide by std, BGR->RGB, HWC->CHW
        blob = cv2.dnn.blobFromImage(
            aligned_face, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True
        )

        # Run through ONNX session
        session = self._rec_model.session
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: blob})[0]

        embedding = output.flatten().astype(np.float32)

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def extract_batch(
        self,
        face_images: list[np.ndarray],
    ) -> list[EmbeddingResult]:
        """Extract embeddings for multiple face crops."""
        return [self.extract(img) for img in face_images]
