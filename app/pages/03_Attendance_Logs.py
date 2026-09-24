"""
Attendance Logs Page for Streamlit.
View check-ins and security events.
"""

import os
import sys

# Ensure project root is on the Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from src.database.session import SessionLocal, init_db
from src.repositories.event_repo import AttendanceRepository, SecurityEventRepository
from src.repositories.employee_repo import EmployeeRepository

st.set_page_config(page_title="Logs - VisionAttend", page_icon="📊", layout="wide")
st.title("📊 Attendance & Security Logs")

def get_attendance_logs():
    init_db()
    db = SessionLocal()
    try:
        att_repo = AttendanceRepository(db)
        emp_repo = EmployeeRepository(db)
        
        events = att_repo.get_recent(limit=100)
        data = []
        for ev in events:
            emp = emp_repo.get_by_id(ev.employee_id) if ev.employee_id else None
            emp_name = emp.full_name if emp else "Unknown"
            
            data.append({
                "Time": ev.timestamp.strftime("%Y-%m-%d %H:%M:%S") if ev.timestamp else "N/A",
                "Employee": emp_name,
                "Event": ev.event_type.upper() if ev.event_type else "N/A",
                "Similarity": f"{ev.similarity_score:.3f}" if ev.similarity_score else "N/A",
                "Liveness": f"{ev.liveness_score:.3f}" if ev.liveness_score else "N/A",
                "Status": ev.status or "N/A"
            })
        return data
    finally:
        db.close()

def get_security_logs():
    init_db()
    db = SessionLocal()
    try:
        sec_repo = SecurityEventRepository(db)
        events = sec_repo.get_recent(limit=100)
        data = []
        for ev in events:
            data.append({
                "Time": ev.timestamp.strftime("%Y-%m-%d %H:%M:%S") if ev.timestamp else "N/A",
                "Type": ev.event_type.replace('_', ' ').title() if ev.event_type else "N/A",
                "Source": ev.source or "N/A",
                "Resolved": "✅" if ev.resolved else "❌"
            })
        return data
    finally:
        db.close()

tab1, tab2 = st.tabs(["Attendance Records", "Security Alerts"])

with tab1:
    st.markdown("### Recent Check-Ins")
    att_data = get_attendance_logs()
    if not att_data:
        st.info("No attendance records found. Start the Live Monitor to begin logging.")
    else:
        # Header row
        cols = st.columns([3, 3, 2, 2, 2, 2])
        for col, header in zip(cols, ["Time", "Employee", "Event", "Similarity", "Liveness", "Status"]):
            col.markdown(f"**{header}**")
        st.markdown("---")
        
        for row in att_data:
            cols = st.columns([3, 3, 2, 2, 2, 2])
            cols[0].write(row["Time"])
            cols[1].write(row["Employee"])
            cols[2].write(row["Event"])
            cols[3].write(row["Similarity"])
            cols[4].write(row["Liveness"])
            cols[5].write(row["Status"])

with tab2:
    st.markdown("### Security Events")
    sec_data = get_security_logs()
    if not sec_data:
        st.success("No security alerts found! System is secure. ✅")
    else:
        cols = st.columns([3, 3, 2, 2])
        for col, header in zip(cols, ["Time", "Type", "Source", "Resolved"]):
            col.markdown(f"**{header}**")
        st.markdown("---")
        
        for row in sec_data:
            cols = st.columns([3, 3, 2, 2])
            cols[0].write(row["Time"])
            cols[1].write(row["Type"])
            cols[2].write(row["Source"])
            cols[3].write(row["Resolved"])
