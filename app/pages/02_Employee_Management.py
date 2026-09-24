"""
Employee Management Page for Streamlit.
View, add, and manage enrolled employees.
"""

import os
import sys

# Ensure project root is on the Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from src.database.session import SessionLocal, init_db
from src.repositories.employee_repo import EmployeeRepository

st.set_page_config(page_title="Employees - VisionAttend", page_icon="👥", layout="wide")
st.title("👥 Employee Management")

def get_employees():
    init_db()
    db = SessionLocal()
    try:
        repo = EmployeeRepository(db)
        emps = repo.get_all()
        data = []
        for e in emps:
            data.append({
                "ID": e.id,
                "Code": e.employee_code,
                "Name": e.full_name,
                "Department": e.department or "N/A",
                "Status": e.status,
                "Enrolled On": e.created_at.strftime("%Y-%m-%d") if e.created_at else "N/A"
            })
        return data
    finally:
        db.close()

tab1, tab2 = st.tabs(["Employee Directory", "Enroll New Employee"])

with tab1:
    st.markdown("### Enrolled Employees")
    data = get_employees()
    if not data:
        st.info("No employees enrolled yet. Use the 'Enroll New Employee' tab to get started.")
    else:
        # Use native Streamlit table (no pandas needed)
        # Create header
        cols = st.columns([1, 2, 3, 2, 2, 2])
        headers = ["ID", "Code", "Name", "Department", "Status", "Enrolled On"]
        for col, header in zip(cols, headers):
            col.markdown(f"**{header}**")
        
        st.markdown("---")
        
        for emp in data:
            cols = st.columns([1, 2, 3, 2, 2, 2])
            cols[0].write(emp["ID"])
            cols[1].write(emp["Code"])
            cols[2].write(emp["Name"])
            cols[3].write(emp["Department"])
            status_color = "🟢" if emp["Status"] == "active" else "🔴"
            cols[4].write(f"{status_color} {emp['Status']}")
            cols[5].write(emp["Enrolled On"])

with tab2:
    st.markdown("### Enroll New Employee")
    st.info("For this MVP, please use the CLI script to enroll users with high-quality photos.")
    
    st.markdown("""
    **To enroll an employee via terminal:**
    ```bash
    python scripts/enroll_employee.py --code EMP001 --name "John Doe" --department "Engineering" --images path/to/img1.jpg path/to/img2.jpg path/to/img3.jpg
    ```
    
    **Requirements:**
    - At least 3 clear face images per employee
    - Well-lit, frontal shots for best accuracy
    - Minimum face size: 80x80 pixels
    """)
