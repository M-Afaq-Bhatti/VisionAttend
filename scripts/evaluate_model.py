"""
Evaluation Script for VisionAttend.

This script tests the recognition accuracy of the system.
It takes a directory of test images, runs them through the pipeline,
and compares the predicted identity against the ground truth (based on folder names).

Directory structure expected:
test_images/
  ├── EMP001/
  │   ├── test1.jpg
  │   └── test2.jpg
  ├── EMP002/
  │   └── test1.jpg
  ├── unknown/
  │   ├── stranger1.jpg
  │   └── stranger2.jpg
  └── spoof/
      ├── phone_screen.jpg
      └── printed_photo.jpg
"""

import argparse
import os
import glob
import cv2
from collections import defaultdict
from rich.console import Console
from rich.table import Table

from src.database.session import SessionLocal, init_db
from src.cv.detection.detector import FaceDetector
from src.cv.embeddings.extractor import FaceEmbedder
from src.cv.quality.checker import FaceQualityChecker
from src.cv.recognition.matcher import IdentityMatcher
from src.antispoof.checker import LivenessChecker, SpoofStatus
from src.pipelines.recognition_pipeline import RecognitionPipeline

console = Console()

def main():
    parser = argparse.ArgumentParser(description="Evaluate VisionAttend Accuracy")
    parser.add_argument("--test-dir", type=str, required=True, help="Path to folder containing test images")
    parser.add_argument("--threshold", type=float, default=0.45, help="Recognition threshold")
    args = parser.parse_args()

    if not os.path.exists(args.test_dir):
        console.print(f"[red]Error: Test directory '{args.test_dir}' not found.[/red]")
        return

    console.print("[bold blue]Initializing Models and Database...[/bold blue]")
    init_db()
    db = SessionLocal()

    # Load Models
    detector = FaceDetector(ctx_id=-1)
    detector.initialize()
    embedder = FaceEmbedder(ctx_id=-1)
    embedder.initialize()
    quality = FaceQualityChecker()
    liveness = LivenessChecker()
    matcher = IdentityMatcher(threshold=args.threshold)
    
    loaded_emps = matcher.load_gallery(db)
    console.print(f"[green]Loaded {loaded_emps} employees into gallery.[/green]")

    pipeline = RecognitionPipeline(
        detector=detector,
        embedder=embedder,
        matcher=matcher,
        quality_checker=quality,
        liveness_checker=liveness
    )

    # Metrics
    metrics = {
        "total_images": 0,
        "true_positives": 0,    # Known person recognized correctly
        "false_positives": 0,   # Unknown person recognized as an employee (CRITICAL)
        "false_negatives": 0,   # Known person NOT recognized
        "true_negatives": 0,    # Unknown person correctly rejected
        "spoofs_detected": 0,   # Spoof image correctly rejected
        "spoofs_missed": 0,     # Spoof image slipped through
        "no_face_detected": 0
    }

    results_table = Table(title="Detailed Predictions")
    results_table.add_column("Image", style="cyan")
    results_table.add_column("True Label", style="magenta")
    results_table.add_column("Predicted", style="green")
    results_table.add_column("Status", style="yellow")
    results_table.add_column("Similarity", justify="right")

    # Read all images
    for root, _, files in os.walk(args.test_dir):
        label = os.path.basename(root).strip().upper()
        if label == os.path.basename(args.test_dir).upper():
            continue # Skip root dir

        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(root, file)
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            metrics["total_images"] += 1
            result = pipeline.process_frame(frame)

            predicted = result.identity if result.recognized else "UNKNOWN"
            sim = f"{result.similarity:.3f}" if result.match_result else "N/A"
            status = "Pass"

            if not result.face_detected:
                metrics["no_face_detected"] += 1
                status = "[red]No Face[/red]"
            elif label == "SPOOF":
                if result.liveness_status == SpoofStatus.SPOOF:
                    metrics["spoofs_detected"] += 1
                    status = "[green]Spoof Caught[/green]"
                else:
                    metrics["spoofs_missed"] += 1
                    status = "[red]Spoof Missed[/red]"
            elif label == "UNKNOWN":
                if result.recognized:
                    metrics["false_positives"] += 1
                    status = "[red]False Accept[/red]"
                else:
                    metrics["true_negatives"] += 1
                    status = "[green]Correct Reject[/green]"
            else: # Known Employee
                if result.recognized and result.identity.upper() == label:
                    metrics["true_positives"] += 1
                    status = "[green]Correct[/green]"
                elif result.recognized and result.identity.upper() != label:
                    metrics["false_positives"] += 1 # Recognized as WRONG person
                    status = "[red]Wrong Person[/red]"
                else:
                    metrics["false_negatives"] += 1
                    status = "[yellow]Not Recognized[/yellow]"

            results_table.add_row(file, label, predicted, status, sim)

    # Print Table
    console.print(results_table)

    # Print Summary
    console.print("\n[bold blue]=== Evaluation Summary ===[/bold blue]")
    
    total_known = metrics["true_positives"] + metrics["false_negatives"]
    total_unknown = metrics["true_negatives"] + metrics["false_positives"]
    total_spoof = metrics["spoofs_detected"] + metrics["spoofs_missed"]

    tpr = (metrics["true_positives"] / total_known * 100) if total_known > 0 else 0.0
    far = (metrics["false_positives"] / total_unknown * 100) if total_unknown > 0 else 0.0
    srr = (metrics["spoofs_detected"] / total_spoof * 100) if total_spoof > 0 else 0.0

    console.print(f"Total Images Processed: {metrics['total_images']}")
    console.print(f"No Face Detected: {metrics['no_face_detected']}")
    console.print(f"[green]True Positives (Correctly Recognized):[/green] {metrics['true_positives']}/{total_known} ({tpr:.1f}%)")
    console.print(f"[yellow]False Negatives (Failed to Recognize):[/yellow] {metrics['false_negatives']}/{total_known}")
    console.print(f"[red]False Positives (Strangers Accepted):[/red] {metrics['false_positives']}/{total_unknown} ({far:.1f}%)")
    if total_spoof > 0:
        console.print(f"[magenta]Spoof Rejection Rate:[/magenta] {metrics['spoofs_detected']}/{total_spoof} ({srr:.1f}%)")

    console.print("\n[bold]Note:[/bold] Adjust the --threshold parameter to balance False Positives and False Negatives.")

if __name__ == "__main__":
    main()
