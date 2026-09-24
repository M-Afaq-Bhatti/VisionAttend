"""
Live Monitor Page for Streamlit.
Connects to the webcam, runs the recognition pipeline, and logs attendance.
"""

import time
import cv2
import streamlit as st
import numpy as np

from src.database.session import SessionLocal
from src.cv.detection.detector import FaceDetector
from src.cv.embeddings.extractor import FaceEmbedder
from src.cv.recognition.matcher import IdentityMatcher
from src.cv.quality.checker import FaceQualityChecker
from src.antispoof.checker import LivenessChecker
from src.pipelines.recognition_pipeline import RecognitionPipeline
from src.attendance.engine import AttendanceEngine
from app.components.ui_helpers import draw_bounding_boxes

st.set_page_config(page_title="Live Monitor - VisionAttend", page_icon="📷", layout="wide")
st.title("📷 Live Attendance Monitor")

# Initialize models in session state so we don't reload them on every UI click
@st.cache_resource
def load_models():
    with st.spinner("Loading AI Models (this may take a moment)..."):
        detector = FaceDetector(ctx_id=-1)
        detector.initialize()
        
        embedder = FaceEmbedder(ctx_id=-1)
        embedder.initialize()
        
        quality = FaceQualityChecker()
        liveness = LivenessChecker()
        return detector, embedder, quality, liveness

@st.cache_resource
def get_pipeline_and_engine():
    db = SessionLocal()
    matcher = IdentityMatcher(threshold=0.45)
    matcher.load_gallery(db)
    
    detector, embedder, quality, liveness = load_models()
    
    pipeline = RecognitionPipeline(
        detector=detector,
        embedder=embedder,
        matcher=matcher,
        quality_checker=quality,
        liveness_checker=liveness
    )
    
    engine = AttendanceEngine(db=db, cooldown_minutes=5)
    return pipeline, engine, db

# State flags
if "run_camera" not in st.session_state:
    st.session_state.run_camera = False

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
    recent_logs = []

with col1:
    st.markdown("### Camera Feed")
    frame_placeholder = st.empty()

# Camera Loop
if st.session_state.run_camera:
    try:
        pipeline, engine, db = get_pipeline_and_engine()
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Error: Could not open webcam.")
            st.session_state.run_camera = False
            st.rerun()
            
        while st.session_state.run_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read frame from webcam.")
                break
                
            # Process frame
            results = pipeline.process_frame_multi(frame)
            
            # Process attendance
            for res in results:
                if res.face_detected:
                    event = engine.process_recognition_result(res)
                    if event:
                        recent_logs.insert(0, f"✅ {event.event_type.upper()}: {res.identity or 'Unknown'}")
                        if len(recent_logs) > 5:
                            recent_logs.pop()
                            
            # Draw UI
            annotated_frame = draw_bounding_boxes(frame, results)
            
            # Convert BGR to RGB for Streamlit
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
            
            # Update activity log
            if recent_logs:
                activity_placeholder.markdown("\n".join([f"- {log}" for log in recent_logs]))
                
            # Small sleep to yield to browser
            time.sleep(0.03)
            
    except Exception as e:
        st.error(f"Camera feed error: {e}")
    finally:
        if 'cap' in locals():
            cap.release()
else:
    frame_placeholder.info("Camera is currently stopped. Click 'Start Camera' to begin monitoring.")
