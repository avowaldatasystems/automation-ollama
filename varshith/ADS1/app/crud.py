from typing import Any

from sqlalchemy import Select, or_, select
from sqlalchemy.orm import Session

from app import models, schemas


def create_employee(db: Session, payload: schemas.EmployeePersonalCreate) -> models.EmployeePersonalDetails:
    employee = models.EmployeePersonalDetails(**payload.model_dump(exclude_none=True))
    db.add(employee)
    db.commit()
    db.refresh(employee)
    return employee


def create_office_details(
    db: Session, payload: schemas.EmployeeOfficeCreate
) -> models.EmployeeOfficeDetails:
    row = models.EmployeeOfficeDetails(**payload.model_dump(exclude_none=True))
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def create_salary_leave_details(
    db: Session, payload: schemas.EmployeeSalaryLeaveCreate
) -> models.EmployeeSalaryLeaveDetails:
    row = models.EmployeeSalaryLeaveDetails(**payload.model_dump(exclude_none=True))
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def create_attendance(
    db: Session, payload: schemas.EmployeeAttendanceCreate
) -> models.EmployeeAttendance:
    row = models.EmployeeAttendance(**payload.model_dump(exclude_none=True))
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def create_document(db: Session, payload: schemas.EmployeeDocumentCreate) -> models.EmployeeDocument:
    row = models.EmployeeDocument(**payload.model_dump(exclude_none=True))
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def employee_summary_query() -> Select[Any]:
    return (
        select(
            models.EmployeePersonalDetails.employee_id,
            models.EmployeePersonalDetails.first_name,
            models.EmployeePersonalDetails.last_name,
            models.EmployeePersonalDetails.personal_email,
            models.EmployeePersonalDetails.office_email,
            models.EmployeePersonalDetails.personal_phone,
            models.EmployeeOfficeDetails.department,
            models.EmployeeOfficeDetails.designation,
            models.EmployeeOfficeDetails.employee_code,
            models.EmployeeOfficeDetails.salary_per_month,
            models.EmployeeSalaryLeaveDetails.final_salary,
        )
        .outerjoin(
            models.EmployeeOfficeDetails,
            models.EmployeePersonalDetails.employee_id == models.EmployeeOfficeDetails.employee_id,
        )
        .outerjoin(
            models.EmployeeSalaryLeaveDetails,
            models.EmployeePersonalDetails.employee_id == models.EmployeeSalaryLeaveDetails.employee_id,
        )
    )


def list_employee_summaries(db: Session) -> list[dict[str, Any]]:
    return [dict(row._mapping) for row in db.execute(employee_summary_query()).all()]


def search_employee_records(db: Session, question: str, limit: int = 20) -> list[dict[str, Any]]:
    tokens = [
        token.strip(" ,.?;:'\"()[]{}").lower()
        for token in question.split()
        if len(token.strip(" ,.?;:'\"()[]{}")) >= 3
    ]
    conditions = []
    for token in tokens:
        pattern = f"%{token}%"
        conditions.extend(
            [
                models.EmployeePersonalDetails.first_name.ilike(pattern),
                models.EmployeePersonalDetails.last_name.ilike(pattern),
                models.EmployeePersonalDetails.personal_email.ilike(pattern),
                models.EmployeePersonalDetails.office_email.ilike(pattern),
                models.EmployeePersonalDetails.personal_phone.ilike(pattern),
                models.EmployeePersonalDetails.skills.ilike(pattern),
                models.EmployeeOfficeDetails.employee_code.ilike(pattern),
                models.EmployeeOfficeDetails.department.ilike(pattern),
                models.EmployeeOfficeDetails.designation.ilike(pattern),
                models.EmployeeOfficeDetails.manager_name.ilike(pattern),
                models.EmployeeSalaryLeaveDetails.salary_month.ilike(pattern),
            ]
        )

    query = employee_summary_query()
    if conditions:
        query = query.where(or_(*conditions))

    rows = db.execute(query.limit(limit)).all()
    return [dict(row._mapping) for row in rows]


def get_employee_full_profile(db: Session, employee_id: int) -> dict[str, Any] | None:
    employee = db.get(models.EmployeePersonalDetails, employee_id)
    if not employee:
        return None

    return {
        "personal": model_to_dict(employee),
        "office": [model_to_dict(row) for row in employee.office_details],
        "salary_leave": [model_to_dict(row) for row in employee.salary_leave_details],
        "attendance": [model_to_dict(row) for row in employee.attendance],
        "documents": [model_to_dict(row) for row in employee.documents],
    }


def model_to_dict(row: Any) -> dict[str, Any]:
    data = {}
    for column in row.__table__.columns:
        value = getattr(row, column.name)
        data[column.name] = value
    return data
