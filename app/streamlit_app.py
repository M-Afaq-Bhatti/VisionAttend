"""
VisionAttend - Smart Workforce Attendance & Access Monitoring System
Main Streamlit Application Entry Point.
"""

import streamlit as st
import os

# Must be the first Streamlit command
st.set_page_config(
    page_title="VisionAttend",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    st.sidebar.title("👁️ VisionAttend")
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Welcome to the VisionAttend Admin Dashboard. "
        "Please select a page from the navigation menu above."
    )

    st.title("Welcome to VisionAttend")
    st.markdown("""
    ### Smart Workforce Attendance & Access Monitoring System

    This dashboard allows you to manage and monitor your workplace using advanced Computer Vision.

    **Navigation:**
    * 📷 **Live Monitor:** Start the webcam to actively recognize employees and log attendance.
    * 👥 **Employee Management:** View enrolled employees and register new ones.
    * 📊 **Attendance Logs:** View the history of check-ins and security alerts.
    
    👈 Select a page from the sidebar to get started.
    """)

    # Show some quick stats
    try:
        from src.database.session import SessionLocal, init_db
        from src.repositories.employee_repo import EmployeeRepository
        from src.repositories.event_repo import AttendanceRepository
        
        # Ensure DB is initialized
        init_db()
        db = SessionLocal()
        emp_repo = EmployeeRepository(db)
        att_repo = AttendanceRepository(db)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Enrolled Employees", emp_repo.count())
        with col2:
            st.metric("Total Attendance Logs", att_repo.count())
            
        db.close()
    except Exception as e:
        st.warning(f"Could not load database statistics: {e}")

if __name__ == "__main__":
    main()
