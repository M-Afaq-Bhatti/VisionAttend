"""
Rebuild embeddings script.

Reloads face images from the enrollment directory and regenerates
all embeddings. Useful after model updates or threshold changes.

Usage:
    python scripts/rebuild_embeddings.py
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
import cv2
import numpy as np

from src.config.settings import settings
from src.utils.logging_config import setup_logging
from src.database.session import SessionLocal, init_db
from src.repositories.employee_repo import EmployeeRepository, FaceProfileRepository
from src.cv.detection.detector import FaceDetector
from src.cv.embeddings.extractor import FaceEmbedder
from src.cv.quality.checker import FaceQualityChecker
from src.pipelines.enrollment import EnrollmentPipeline


def main():
    setup_logging()
    init_db()
    db = SessionLocal()

    try:
        emp_repo = EmployeeRepository(db)
        profile_repo = FaceProfileRepository(db)

        employees = emp_repo.get_all(status="active")
        if not employees:
            print("No active employees found.")
            return

        print(f"Found {len(employees)} active employees")

        # Initialize models
        print("Loading face detection model...")
        detector = FaceDetector(ctx_id=-1)
        detector.initialize()

        print("Loading face embedding model...")
        embedder = FaceEmbedder(ctx_id=-1)
        embedder.initialize()

        quality_checker = FaceQualityChecker()
        pipeline = EnrollmentPipeline(
            detector=detector,
            embedder=embedder,
            quality_checker=quality_checker,
            min_accepted_images=1,  # More lenient for rebuild
        )

        for emp in employees:
            emp_dir = Path("data/enrollment") / emp.employee_code
            if not emp_dir.exists():
                print(f"  {emp.employee_code}: No enrollment directory found, skipping")
                continue

            # Find face images
            image_files = sorted(emp_dir.glob("face_*.jpg"))
            if not image_files:
                # Try original source images
                image_files = sorted(
                    f for f in emp_dir.iterdir()
                    if f.suffix.lower() in (".jpg", ".jpeg", ".png") and f.stem != "embeddings"
                )

            if not image_files:
                print(f"  {emp.employee_code}: No images found, skipping")
                continue

            images = []
            for f in image_files:
                img = cv2.imread(str(f))
                if img is not None:
                    images.append(img)

            if not images:
                print(f"  {emp.employee_code}: No valid images, skipping")
                continue

            print(f"  {emp.employee_code}: Rebuilding from {len(images)} images...")
            result = pipeline.enroll(emp.employee_code, images)

            if result.success:
                # Update profile reference
                profile_repo.delete_by_employee(emp.id)
                emb_path = f"data/enrollment/{emp.employee_code}/embeddings.npz"
                profile_repo.create(
                    employee_id=emp.id,
                    embedding_reference=emb_path,
                    model_name=embedder.model_name,
                    embedding_dimension=embedder.embedding_dim,
                )
                print(f"    ✓ {result.accepted_images}/{result.total_images} accepted")
            else:
                print(f"    ✗ Failed: {result.error}")

        print("\nRebuild complete.")

    finally:
        db.close()


if __name__ == "__main__":
    main()
