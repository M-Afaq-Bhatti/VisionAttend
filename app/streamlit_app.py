import streamlit as st
import sys
import os

# Add project root to path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.settings import settings

st.set_page_config(
    page_title="VisionAttend",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛡️ VisionAttend Dashboard")
st.markdown("### Smart Workforce Attendance & Access Monitoring System")

st.info(f"System running in DEMO_MODE: {settings.demo_mode}")

st.sidebar.title("Navigation")
st.sidebar.info("Select a page from the sidebar (pages will be implemented in Phase 6).")

# Placeholder for metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Employees", "0")
with col2:
    st.metric("Today's Check-ins", "0")
with col3:
    st.metric("Unknown Events", "0")
with col4:
    st.metric("Spoof Attempts", "0")

st.write("---")
st.write("Welcome to the VisionAttend MVP. Currently in development (Phase 2).")
