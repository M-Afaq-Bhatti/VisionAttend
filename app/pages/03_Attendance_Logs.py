"""
Attendance Logs Page for Streamlit.
View check-ins and security events.
"""

import streamlit as st
import pandas as pd
from src.database.session import SessionLocal
from src.repositories.event_repo import AttendanceRepository, SecurityEventRepository
from src.repositories.employee_repo import EmployeeRepository

st.set_page_config(page_title="Logs - VisionAttend", page_icon="📊", layout="wide")
st.title("📊 Attendance & Security Logs")

@st.cache_data(ttl=5)
def get_attendance_logs():
    db = SessionLocal()
    try:
        att_repo = AttendanceRepository(db)
        emp_repo = EmployeeRepository(db)
        
        events = att_repo.get_recent(limit=100)
        data = []
        for ev in events:
            emp = emp_repo.get_by_id(ev.employee_id)
            emp_name = emp.full_name if emp else "Unknown"
            
            data.append({
                "Time": ev.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "Employee": emp_name,
                "Event": ev.event_type.upper(),
                "Similarity": f"{ev.similarity_score:.3f}" if ev.similarity_score else "N/A",
                "Liveness": f"{ev.liveness_score:.3f}" if ev.liveness_score else "N/A",
                "Status": ev.status
            })
        return pd.DataFrame(data)
    finally:
        db.close()

@st.cache_data(ttl=5)
def get_security_logs():
    db = SessionLocal()
    try:
        sec_repo = SecurityEventRepository(db)
        events = sec_repo.get_recent(limit=100)
        data = []
        for ev in events:
            data.append({
                "Time": ev.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "Type": ev.event_type.replace('_', ' ').title(),
                "Source": ev.source,
                "Resolved": ev.resolved
            })
        return pd.DataFrame(data)
    finally:
        db.close()

tab1, tab2 = st.tabs(["Attendance Records", "Security Alerts"])

with tab1:
    st.markdown("### Recent Check-Ins")
    df_att = get_attendance_logs()
    if df_att.empty:
        st.info("No attendance records found.")
    else:
        st.dataframe(df_att, use_container_width=True, hide_index=True)

with tab2:
    st.markdown("### Security Events")
    df_sec = get_security_logs()
    if df_sec.empty:
        st.success("No security alerts found! System is secure.")
    else:
        # Highlight unresolved events
        st.dataframe(df_sec, use_container_width=True, hide_index=True)
