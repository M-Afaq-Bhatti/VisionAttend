"""
Identity matching module.

Compares a query face embedding against a gallery of registered
employee embeddings using cosine similarity, with configurable
thresholds and top-k retrieval.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MatchCandidate:
    """A single match candidate with score."""
    employee_id: int
    employee_code: str
    similarity: float


@dataclass
class MatchResult:
    """Result of an identity matching attempt."""
    recognized: bool
    identity: Optional[str] = None          # employee_code
    employee_id: Optional[int] = None
    similarity: float = 0.0
    threshold: float = 0.0
    candidates: list[MatchCandidate] = None

    def __post_init__(self):
        if self.candidates is None:
            self.candidates = []

    def to_dict(self) -> dict:
        return {
            "identity": self.identity,
            "employee_id": self.employee_id,
            "similarity": round(self.similarity, 4),
            "threshold": self.threshold,
            "recognized": self.recognized,
            "num_candidates": len(self.candidates),
        }


class IdentityMatcher:
    """
    Matches a query embedding against a gallery of registered embeddings
    using cosine similarity.

    The gallery is a dictionary mapping (employee_id, employee_code) to
    a list of embeddings (one employee can have multiple enrollment images).

    Args:
        threshold: Minimum cosine similarity to consider a match.
        top_k: Number of top candidates to return.
    """

    def __init__(self, threshold: float = 0.45, top_k: int = 5):
        self.threshold = threshold
        self.top_k = top_k
        # Gallery: { (employee_id, employee_code): [embedding1, embedding2, ...] }
        self._gallery: dict[tuple[int, str], list[np.ndarray]] = {}

    @property
    def gallery_size(self) -> int:
        """Number of registered employees in the gallery."""
        return len(self._gallery)

    @property
    def total_embeddings(self) -> int:
        """Total number of embeddings across all employees."""
        return sum(len(embs) for embs in self._gallery.values())

    def clear_gallery(self) -> None:
        """Remove all registered embeddings."""
        self._gallery.clear()
        logger.info("Gallery cleared")

    def register(
        self,
        employee_id: int,
        employee_code: str,
        embeddings: list[np.ndarray],
    ) -> None:
        """
        Register an employee's embeddings in the gallery.

        Replaces any existing embeddings for the same employee.

        Args:
            employee_id: Database employee ID.
            employee_code: Employee code string.
            embeddings: List of L2-normalized embedding vectors.
        """
        key = (employee_id, employee_code)
        self._gallery[key] = [e.astype(np.float32) for e in embeddings]
        logger.info(
            "Registered %d embeddings for %s (id=%d)",
            len(embeddings), employee_code, employee_id,
        )

    def unregister(self, employee_id: int) -> bool:
        """Remove an employee from the gallery by ID."""
        to_remove = [k for k in self._gallery if k[0] == employee_id]
        for k in to_remove:
            del self._gallery[k]
        return len(to_remove) > 0

    def load_gallery(self, db) -> int:
        """
        Load all employee embeddings from disk into the gallery.

        Reads the FaceProfile records from the database, then loads
        the referenced .npz embedding files from disk.

        Args:
            db: SQLAlchemy Session.

        Returns:
            Number of employees loaded.
        """
        import os
        from src.database.models import Employee, FaceProfile

        self.clear_gallery()
        profiles = db.query(FaceProfile).join(Employee).filter(
            Employee.status == "active"
        ).all()

        loaded = 0
        for profile in profiles:
            emp = db.query(Employee).filter(Employee.id == profile.employee_id).first()
            if emp is None:
                continue

            emb_path = profile.embedding_reference
            if not os.path.exists(emb_path):
                logger.warning("Embedding file not found: %s", emb_path)
                continue

            try:
                data = np.load(emb_path)
                embeddings = list(data["embeddings"])
                self.register(emp.id, emp.employee_code, embeddings)
                loaded += 1
            except Exception as e:
                logger.error("Failed to load embeddings for %s: %s", emp.employee_code, e)

        logger.info("Gallery loaded: %d employees, %d total embeddings",
                     loaded, self.total_embeddings)
        return loaded

    def match(self, query_embedding: np.ndarray) -> MatchResult:
        """
        Match a query embedding against the gallery.

        For each registered employee, the maximum cosine similarity
        across all their enrolled embeddings is used.

        Args:
            query_embedding: L2-normalized (512,) vector.

        Returns:
            MatchResult with the best candidate and recognition decision.
        """
        if not self._gallery:
            return MatchResult(
                recognized=False,
                threshold=self.threshold,
            )

        query = query_embedding.astype(np.float32).flatten()
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        candidates = []

        for (emp_id, emp_code), embeddings in self._gallery.items():
            # Compute cosine similarity with each enrolled embedding
            # and take the maximum
            best_sim = -1.0
            for emb in embeddings:
                sim = float(np.dot(query, emb))
                if sim > best_sim:
                    best_sim = sim

            candidates.append(MatchCandidate(
                employee_id=emp_id,
                employee_code=emp_code,
                similarity=best_sim,
            ))

        # Sort by similarity descending
        candidates.sort(key=lambda c: c.similarity, reverse=True)
        top_candidates = candidates[:self.top_k]

        best = top_candidates[0] if top_candidates else None
        recognized = best is not None and best.similarity >= self.threshold

        return MatchResult(
            recognized=recognized,
            identity=best.employee_code if recognized else None,
            employee_id=best.employee_id if recognized else None,
            similarity=best.similarity if best else 0.0,
            threshold=self.threshold,
            candidates=top_candidates,
        )

    def update_threshold(self, new_threshold: float) -> None:
        """Update the recognition threshold at runtime."""
        old = self.threshold
        self.threshold = new_threshold
        logger.info("Recognition threshold updated: %.3f -> %.3f", old, new_threshold)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Both vectors should be L2-normalized for best performance,
    but this function handles unnormalized inputs as well.
    """
    a = a.flatten().astype(np.float32)
    b = b.flatten().astype(np.float32)

    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0

    return float(dot / (norm_a * norm_b))
