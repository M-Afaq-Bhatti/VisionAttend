"""
Employee Management Page for Streamlit.
View, add, and manage enrolled employees.
"""

import streamlit as st
import pandas as pd
from src.database.session import SessionLocal
from src.repositories.employee_repo import EmployeeRepository

st.set_page_config(page_title="Employees - VisionAttend", page_icon="👥", layout="wide")
st.title("👥 Employee Management")

@st.cache_data(ttl=5) # Refresh every 5 seconds
def get_employees():
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
                "Enrolled On": e.created_at.strftime("%Y-%m-%d")
            })
        return pd.DataFrame(data)
    finally:
        db.close()

tab1, tab2 = st.tabs(["Employee Directory", "Enroll New Employee"])

with tab1:
    st.markdown("### Enrolled Employees")
    df = get_employees()
    if df.empty:
        st.info("No employees enrolled yet.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

with tab2:
    st.markdown("### Enroll New Employee")
    st.info("For this MVP, please use the CLI script `scripts/enroll_employee.py` to enroll users with high-quality photos to ensure the best possible embeddings.")
    
    st.markdown("""
    **To enroll an employee via terminal:**
    ```bash
    python scripts/enroll_employee.py --code EMP001 --name "John Doe" --department "Engineering" --images path/to/img1.jpg path/to/img2.jpg path/to/img3.jpg
    ```
    """)
