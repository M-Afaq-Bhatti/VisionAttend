"""
UI Helper components for Streamlit.
"""

import cv2
import numpy as np
from src.antispoof.checker import SpoofStatus
from src.pipelines.recognition_pipeline import RecognitionResult


def draw_bounding_boxes(frame: np.ndarray, results: list[RecognitionResult]) -> np.ndarray:
    """
    Draw bounding boxes, names, and spoof alerts on a frame.
    
    Args:
        frame: Original BGR frame.
        results: List of RecognitionResults from the pipeline.
        
    Returns:
        Annotated BGR frame.
    """
    annotated = frame.copy()
    
    for res in results:
        if not res.face_detected or res.face_bbox is None:
            continue
            
        x1, y1, x2, y2 = map(int, res.face_bbox)
        
        # Determine color and text based on liveness and recognition
        color = (0, 255, 255)  # Yellow for unknown/default
        text = "Unknown"
        sub_text = ""
        
        if res.liveness_status == SpoofStatus.SPOOF:
            color = (0, 0, 255)  # Red
            text = "SPOOF DETECTED"
            sub_text = res.liveness_details
        elif res.recognized:
            color = (0, 255, 0)  # Green
            text = str(res.identity)
            sub_text = f"Sim: {res.similarity:.2f}"
            
        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Draw background for text
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
        
        # Draw text
        cv2.putText(annotated, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        if sub_text:
            cv2.putText(annotated, sub_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
    return annotated
