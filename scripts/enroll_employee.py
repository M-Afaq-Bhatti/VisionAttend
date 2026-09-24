"""
Employee enrollment CLI script.

Registers an employee and processes their face images for enrollment.

Usage:
    python scripts/enroll_employee.py --code EMP001 --name "John Doe" \\
        --department "Engineering" --images path/to/img1.jpg path/to/img2.jpg
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    parser = argparse.ArgumentParser(description="Enroll an employee into VisionAttend")
    parser.add_argument("--code", required=True, help="Employee code (e.g., EMP001)")
    parser.add_argument("--name", required=True, help="Full name")
    parser.add_argument("--department", default=None, help="Department")
    parser.add_argument("--position", default=None, help="Position/title")
    parser.add_argument("--email", default=None, help="Email address")
    parser.add_argument("--images", nargs="+", required=True, help="Paths to face images")
    args = parser.parse_args()

    setup_logging()

    # Initialize database
    init_db()
    db = SessionLocal()

    try:
        # Check for duplicate
        emp_repo = EmployeeRepository(db)
        existing = emp_repo.get_by_code(args.code)
        if existing:
            print(f"Employee {args.code} already exists (id={existing.id})")
            print("To re-enroll, first remove existing face profiles.")
            return

        # Load images
        images = []
        for path in args.images:
            if not os.path.exists(path):
                print(f"Warning: Image not found: {path}")
                continue
            img = cv2.imread(path)
            if img is None:
                print(f"Warning: Could not read image: {path}")
                continue
            images.append(img)

        if len(images) < 3:
            print(f"Error: Need at least 3 valid images, got {len(images)}")
            return

        print(f"Loaded {len(images)} images for enrollment")

        # Initialize CV models
        print("Loading face detection model...")
        detector = FaceDetector(ctx_id=-1)
        detector.initialize()

        print("Loading face embedding model...")
        embedder = FaceEmbedder(ctx_id=-1)
        embedder.initialize()

        # Run enrollment pipeline
        pipeline = EnrollmentPipeline(
            detector=detector,
            embedder=embedder,
            quality_checker=FaceQualityChecker(),
        )

        print(f"Processing {len(images)} images...")
        result = pipeline.enroll(args.code, images)

        # Print results
        print(f"\nEnrollment Results for {args.code}:")
        print(f"  Total images:    {result.total_images}")
        print(f"  Accepted:        {result.accepted_images}")
        print(f"  Rejected:        {result.rejected_images}")
        print(f"  Processing time: {result.total_latency_ms:.0f} ms")

        for img_res in result.image_results:
            status = "✓" if img_res.accepted else "✗"
            print(f"  Image {img_res.index}: {status} {img_res.quality_status.value} — {img_res.quality_details}")

        if not result.success:
            print(f"\nEnrollment FAILED: {result.error}")
            return

        # Create employee record
        employee = emp_repo.create(
            employee_code=args.code,
            full_name=args.name,
            department=args.department,
            position=args.position,
            email=args.email,
        )
        print(f"\nEmployee created: {employee.full_name} (id={employee.id})")

        # Store face profile references
        profile_repo = FaceProfileRepository(db)
        emb_path = f"data/enrollment/{args.code}/embeddings.npz"
        profile = profile_repo.create(
            employee_id=employee.id,
            embedding_reference=emb_path,
            model_name=embedder.model_name,
            embedding_dimension=embedder.embedding_dim,
        )
        print(f"Face profile stored: {profile.embedding_reference}")
        print(f"\nEnrollment SUCCESSFUL for {args.code}")

    finally:
        db.close()


if __name__ == "__main__":
    main()
