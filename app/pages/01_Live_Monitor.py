"""
Live Monitor Page for Streamlit.
Connects to the webcam, runs the recognition pipeline, and logs attendance.
"""

import os
import sys
import time

# Ensure project root is on the Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import streamlit as st
import numpy as np

from src.database.session import SessionLocal, init_db
from src.cv.detection.detector import FaceDetector
from src.cv.embeddings.extractor import FaceEmbedder
from src.cv.recognition.matcher import IdentityMatcher
from src.cv.quality.checker import FaceQualityChecker
from src.antispoof.checker import LivenessChecker, SpoofStatus
from src.pipelines.recognition_pipeline import RecognitionPipeline
from src.attendance.engine import AttendanceEngine
from app.components.ui_helpers import draw_bounding_boxes

st.set_page_config(page_title="Live Monitor - VisionAttend", page_icon="📷", layout="wide")
st.title("📷 Live Attendance Monitor")


@st.cache_resource
def load_pipeline():
    """Load AI models and create pipeline (cached across reruns)."""
    # Initialize database
    init_db()
    db = SessionLocal()

    # Load CV models
    detector = FaceDetector(ctx_id=-1)
    detector.initialize()

    embedder = FaceEmbedder(ctx_id=-1)
    embedder.initialize()

    quality = FaceQualityChecker()
    liveness = LivenessChecker()

    matcher = IdentityMatcher(threshold=0.45)
    matcher.load_gallery(db)

    pipeline = RecognitionPipeline(
        detector=detector,
        embedder=embedder,
        matcher=matcher,
        quality_checker=quality,
        liveness_checker=liveness,
    )

    engine = AttendanceEngine(db_session=db, cooldown_minutes=5)
    return pipeline, engine


# Session state
if "run_camera" not in st.session_state:
    st.session_state.run_camera = False
if "recent_logs" not in st.session_state:
    st.session_state.recent_logs = []

col1, col2 = st.columns([3, 1])

with col2:
    st.markdown("### Controls")
    if st.button("▶️ Start Camera", type="primary", use_container_width=True):
        st.session_state.run_camera = True
        st.rerun()

    if st.button("⏹️ Stop Camera", type="secondary", use_container_width=True):
        st.session_state.run_camera = False
        st.rerun()

    st.markdown("### Recent Activity")
    activity_placeholder = st.empty()

with col1:
    st.markdown("### Camera Feed")
    frame_placeholder = st.empty()

# Camera Loop
if st.session_state.run_camera:
    try:
        pipeline, engine = load_pipeline()
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Could not open webcam. Make sure no other app is using it.")
            st.session_state.run_camera = False
            st.rerun()

        while st.session_state.run_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read frame from webcam.")
                break

            # Process frame through the full pipeline
            results = pipeline.process_frame_multi(frame)

            # Process attendance for each detected face
            for res in results:
                if not res.face_detected:
                    continue

                if res.liveness_status == SpoofStatus.SPOOF:
                    event_info = engine.process_spoof_attempt(
                        liveness_score=res.liveness_score,
                        employee_id=res.employee_id,
                    )
                    st.session_state.recent_logs.insert(0, f"🔴 SPOOF: {res.liveness_details}")
                elif res.recognized and res.employee_id:
                    event_info = engine.process_recognition(
                        employee_id=res.employee_id,
                        similarity_score=res.similarity,
                        liveness_score=res.liveness_score,
                    )
                    status_icon = "✅" if event_info["status"] == "recorded" else "⏳"
                    st.session_state.recent_logs.insert(0, f"{status_icon} {res.identity}: {event_info['message']}")
                elif res.face_detected and not res.recognized:
                    engine.process_unknown()
                    st.session_state.recent_logs.insert(0, "🟡 Unknown person detected")

                # Keep only last 10 logs
                st.session_state.recent_logs = st.session_state.recent_logs[:10]

            # Draw bounding boxes on frame
            annotated_frame = draw_bounding_boxes(frame, results)

            # Convert BGR to RGB for Streamlit
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            # Update activity log
            if st.session_state.recent_logs:
                activity_placeholder.markdown(
                    "\n".join([f"- {log}" for log in st.session_state.recent_logs])
                )

            time.sleep(0.03)

    except Exception as e:
        st.error(f"Camera feed error: {e}")
        import traceback
        st.code(traceback.format_exc())
    finally:
        if "cap" in locals():
            cap.release()
else:
    frame_placeholder.info("📹 Camera is stopped. Click **Start Camera** to begin monitoring.")
