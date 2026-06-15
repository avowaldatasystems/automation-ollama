import streamlit as st

from app import schemas
from app.api_client import ApiError, get_client


st.set_page_config(page_title="Employee Office RAG", layout="wide")


@st.cache_resource
def get_api_client():
    return get_client()


client = get_api_client()


def safe_run(label: str, fn):
    try:
        return fn()
    except ApiError as exc:
        st.error(f"{label} failed: {exc}")
        return None
    except Exception as exc:
        st.error(f"{label} failed: {exc}")
        return None


def require_apis() -> bool:
    statuses = safe_run("API health", client.health_all)
    if not statuses:
        return False
    offline = [name for name, info in statuses.items() if info.get("status") != "ok"]
    if offline:
        st.error(
            "These APIs are offline: "
            + ", ".join(offline)
            + ". Run `.\\scripts\\start_apis.cmd`."
        )
        return False
    return True


with st.sidebar:
    st.subheader("API services (local)")
    statuses = safe_run("API health", client.health_all)
    if statuses:
        all_ok = True
        for name, info in statuses.items():
            ok = info.get("status") == "ok"
            url = client.services[name]
            if ok:
                st.success(f"{name}: {url}")
            else:
                all_ok = False
                st.error(f"{name}: offline")
        if not all_ok:
            st.warning("Start APIs: `.\\scripts\\start_apis.cmd`")
    else:
        st.warning("Start APIs: `.\\scripts\\start_apis.cmd`")

    if st.button("Refresh API status"):
        st.cache_resource.clear()
        st.rerun()


st.title("Employee Office RAG")
st.caption("MySQL employee data + Qdrant company knowledge + Ollama office assistant (all via local APIs)")

tab_chat, tab_employees, tab_office, tab_salary, tab_attendance, tab_knowledge = st.tabs(
    ["Chat", "Employees", "Office", "Salary & Leave", "Attendance", "Knowledge"]
)

with tab_chat:
    st.subheader("Ask the office assistant")
    question = st.text_area(
        "Question",
        placeholder="Example: What is Varshith's salary and what does the leave policy say?",
        height=110,
    )

    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
        elif not require_apis():
            pass
        else:
            with st.spinner("Routing question and retrieving context..."):
                result = safe_run("Chat", lambda: client.chat(question.strip()))

            if result:
                st.info(f"Route used: {result.get('route', 'both')}")
                st.write(result.get("answer", ""))

                with st.expander("SQL context"):
                    st.json(result.get("sql_context", []))
                with st.expander("Qdrant context"):
                    st.json(result.get("vector_sources", []))

with tab_employees:
    left, right = st.columns([1, 1.4])

    with left:
        st.subheader("Add employee")
        with st.form("employee_form"):
            first_name = st.text_input("First name")
            last_name = st.text_input("Last name")
            gender = st.selectbox("Gender", ["", "Male", "Female", "Other"])
            date_of_birth = st.date_input("Date of birth", value=None)
            age = st.number_input("Age", min_value=0, max_value=100, value=None, step=1)
            personal_email = st.text_input("Personal email")
            office_email = st.text_input("Office email")
            personal_phone = st.text_input("Personal phone")
            emergency_contact = st.text_input("Emergency contact")
            city = st.text_input("City")
            state = st.text_input("State")
            country = st.text_input("Country", value="India")
            qualification = st.text_input("Qualification")
            skills = st.text_area("Skills")
            submitted = st.form_submit_button("Add employee", type="primary")

        if submitted:
            if not require_apis():
                pass
            elif not first_name:
                st.warning("First name is required.")
            else:
                payload = schemas.EmployeePersonalCreate(
                    first_name=first_name,
                    last_name=last_name or None,
                    gender=gender or None,
                    date_of_birth=date_of_birth,
                    age=age,
                    personal_email=personal_email or None,
                    office_email=office_email or None,
                    personal_phone=personal_phone or None,
                    emergency_contact=emergency_contact or None,
                    city=city or None,
                    state=state or None,
                    country=country or None,
                    qualification=qualification or None,
                    skills=skills or None,
                )
                employee = safe_run(
                    "Add employee",
                    lambda: client.create_employee(payload),
                )
                if employee:
                    st.success(f"Added employee ID {employee['employee_id']}")

        st.divider()
        st.subheader("Employee document (metadata)")
        with st.form("document_form"):
            doc_employee_id = st.number_input("Employee ID", min_value=1, step=1, key="doc_employee_id")
            document_type = st.text_input("Document type", placeholder="Aadhaar, PAN, Offer Letter")
            document_number = st.text_input("Document number")
            file_path = st.text_input("File path", placeholder="data/uploads/file.pdf")
            doc_submitted = st.form_submit_button("Add document record", type="primary")
        if doc_submitted and require_apis():
            payload = schemas.EmployeeDocumentCreate(
                employee_id=int(doc_employee_id),
                document_type=document_type or None,
                document_number=document_number or None,
                file_path=file_path or None,
            )
            row = safe_run("Add document", lambda: client.create_document(payload))
            if row:
                st.success(f"Added document ID {row.get('document_id', '')}")

    with right:
        st.subheader("Employee records")
        if require_apis():
            rows = safe_run("Load employees", client.list_employees)
            if rows is not None:
                st.dataframe(rows, use_container_width=True)

        st.subheader("Employee profile")
        lookup_id = st.number_input("Lookup employee ID", min_value=1, step=1, key="lookup_employee_id")
        if st.button("Load profile") and require_apis():
            profile = safe_run("Load employee", lambda: client.get_employee(int(lookup_id)))
            if profile:
                st.json(profile)

with tab_office:
    st.subheader("Add office details")
    with st.form("office_form"):
        employee_id = st.number_input("Employee ID", min_value=1, step=1)
        employee_code = st.text_input("Employee code")
        department = st.text_input("Department")
        designation = st.text_input("Designation")
        employment_type = st.selectbox(
            "Employment type", ["", "Full-Time", "Part-Time", "Intern", "Contract"]
        )
        work_location = st.text_input("Work location")
        manager_name = st.text_input("Manager name")
        joining_date = st.date_input("Joining date", value=None)
        salary_per_month = st.number_input("Salary per month", min_value=0.0, step=1000.0)
        bonus = st.number_input("Bonus", min_value=0.0, step=1000.0)
        shift_type = st.text_input("Shift type")
        work_mode = st.selectbox("Work mode", ["", "Remote", "Hybrid", "Office"])
        submitted = st.form_submit_button("Add office details", type="primary")

    if submitted and require_apis():
        payload = schemas.EmployeeOfficeCreate(
            employee_id=int(employee_id),
            employee_code=employee_code or None,
            department=department or None,
            designation=designation or None,
            employment_type=employment_type or None,
            work_location=work_location or None,
            manager_name=manager_name or None,
            joining_date=joining_date,
            salary_per_month=salary_per_month,
            bonus=bonus,
            shift_type=shift_type or None,
            work_mode=work_mode or None,
        )
        row = safe_run(
            "Add office details",
            lambda: client.create_office_details(payload),
        )
        if row:
            st.success(f"Added office record {row['office_record_id']}")

with tab_salary:
    st.subheader("Add salary and leave details")
    with st.form("salary_form"):
        employee_id = st.number_input("Employee ID", min_value=1, step=1, key="salary_employee_id")
        salary_month = st.text_input("Salary month", placeholder="May-2026")
        total_working_days = st.number_input("Total working days", min_value=0, step=1)
        present_days = st.number_input("Present days", min_value=0, step=1)
        leave_days = st.number_input("Leave days", min_value=0, step=1)
        unpaid_leave_days = st.number_input("Unpaid leave days", min_value=0, step=1)
        paid_leave_days = st.number_input("Paid leave days", min_value=0, step=1)
        overtime_hours = st.number_input("Overtime hours", min_value=0.0, step=1.0)
        monthly_salary = st.number_input("Monthly salary", min_value=0.0, step=1000.0)
        leave_deduction = st.number_input("Leave deduction", min_value=0.0, step=500.0)
        tax_deduction = st.number_input("Tax deduction", min_value=0.0, step=500.0)
        pf_deduction = st.number_input("PF deduction", min_value=0.0, step=500.0)
        bonus_amount = st.number_input("Bonus amount", min_value=0.0, step=500.0)
        final_salary = st.number_input("Final salary", min_value=0.0, step=1000.0)
        payment_status = st.selectbox("Payment status", ["Pending", "Paid"])
        payment_date = st.date_input("Payment date", value=None)
        submitted = st.form_submit_button("Add salary row", type="primary")

    if submitted and require_apis():
        payload = schemas.EmployeeSalaryLeaveCreate(
            employee_id=int(employee_id),
            salary_month=salary_month or None,
            total_working_days=int(total_working_days),
            present_days=int(present_days),
            leave_days=int(leave_days),
            unpaid_leave_days=int(unpaid_leave_days),
            paid_leave_days=int(paid_leave_days),
            overtime_hours=overtime_hours,
            monthly_salary=monthly_salary,
            leave_deduction=leave_deduction,
            tax_deduction=tax_deduction,
            pf_deduction=pf_deduction,
            bonus_amount=bonus_amount,
            final_salary=final_salary,
            payment_status=payment_status,
            payment_date=payment_date,
        )
        row = safe_run(
            "Add salary row",
            lambda: client.create_salary_leave(payload),
        )
        if row:
            st.success(f"Added salary record {row['salary_record_id']}")

with tab_attendance:
    st.subheader("Add attendance")
    with st.form("attendance_form"):
        employee_id = st.number_input(
            "Employee ID", min_value=1, step=1, key="attendance_employee_id"
        )
        attendance_date = st.date_input("Attendance date", value=None)
        check_in = st.time_input("Check in", value=None)
        check_out = st.time_input("Check out", value=None)
        total_hours = st.number_input("Total hours", min_value=0.0, step=0.5)
        attendance_status = st.selectbox(
            "Attendance status", ["", "Present", "Absent", "Leave", "Half-Day"]
        )
        submitted = st.form_submit_button("Add attendance", type="primary")

    if submitted and require_apis():
        payload = schemas.EmployeeAttendanceCreate(
            employee_id=int(employee_id),
            attendance_date=attendance_date,
            check_in=check_in,
            check_out=check_out,
            total_hours=total_hours,
            attendance_status=attendance_status or None,
        )
        row = safe_run(
            "Add attendance",
            lambda: client.create_attendance(payload),
        )
        if row:
            st.success(f"Added attendance record {row['attendance_id']}")

with tab_knowledge:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Upload company knowledge")
        uploaded_file = st.file_uploader(
            "PDF, DOCX, TXT, MD, PNG, JPG",
            type=["pdf", "docx", "txt", "md", "png", "jpg", "jpeg"],
        )
        if uploaded_file and st.button("Ingest uploaded file", type="primary"):
            if require_apis():
                result = safe_run(
                    "Ingest file",
                    lambda: client.upload_knowledge_file(
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                    ),
                )
                if result is not None:
                    st.success(f"Added {result.get('chunks_added', 0)} chunks to Qdrant")

        st.divider()
        st.subheader("Add text directly")
        title = st.text_input("Title")
        text = st.text_area("Text", height=160)
        if st.button("Ingest text"):
            if not title or not text:
                st.warning("Title and text are required.")
            elif require_apis():
                result = safe_run(
                    "Ingest text",
                    lambda: client.ingest_text(title, text, "manual"),
                )
                if result is not None:
                    st.success(f"Added {result.get('chunks_added', 0)} chunks to Qdrant")

    with right:
        st.subheader("Search Qdrant")
        search = st.text_input("Search query")
        if st.button("Search knowledge") and require_apis():
            results = safe_run(
                "Search knowledge",
                lambda: client.search_knowledge(search),
            )
            if results is not None:
                for result in results:
                    st.markdown(f"**{result.get('title', 'Untitled')}**")
                    st.caption(result.get("source_path", ""))
                    st.write(result.get("content", ""))
                    st.divider()
