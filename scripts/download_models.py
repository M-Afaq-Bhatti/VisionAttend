"""
Model download and setup script.

Downloads the InsightFace model pack required for face detection
and recognition. Should be run once during initial project setup.

Usage:
    python scripts/download_models.py

Models:
    - buffalo_l (InsightFace): RetinaFace detection + ArcFace recognition
      Source: https://github.com/deepinsight/insightface
      License: MIT (InsightFace library), model weights may have separate terms
      Size: ~300 MB
      Input: BGR image
      Output: Face bboxes, landmarks, 512-D embeddings
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def download_insightface_model(model_name: str = "buffalo_l"):
    """
    Download the InsightFace model pack.

    InsightFace auto-downloads models on first use of FaceAnalysis.
    This script triggers that download explicitly so users know
    when the download happens.
    """
    print(f"Downloading InsightFace model: {model_name}")
    print("This may take a few minutes depending on your connection...")
    print()

    try:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(
            name=model_name,
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=-1, det_size=(640, 640))
        print(f"Model '{model_name}' downloaded and verified successfully.")
        print()

        # Print model info
        if hasattr(app, "models"):
            print("Available sub-models:")
            for name, model in app.models.items():
                print(f"  - {name}: {type(model).__name__}")
        print()
        print("Model location: ~/.insightface/models/")

    except Exception as e:
        print(f"Error downloading model: {e}")
        print()
        print("If this is a network error, ensure you have internet access.")
        print("If the error persists, you can manually download from:")
        print("  https://github.com/deepinsight/insightface/tree/master/model_zoo")
        sys.exit(1)


if __name__ == "__main__":
    download_insightface_model()
