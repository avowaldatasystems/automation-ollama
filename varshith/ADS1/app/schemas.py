from datetime import date, time
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EmployeePersonalCreate(BaseModel):
    first_name: str
    last_name: str | None = None
    gender: str | None = None
    date_of_birth: date | None = None
    age: int | None = None
    personal_email: str | None = None
    office_email: str | None = None
    personal_phone: str | None = None
    emergency_contact: str | None = None
    father_name: str | None = None
    mother_name: str | None = None
    blood_group: str | None = None
    marital_status: str | None = None
    aadhaar_number: str | None = None
    pan_number: str | None = None
    nationality: str | None = "Indian"
    current_address: str | None = None
    permanent_address: str | None = None
    city: str | None = None
    state: str | None = None
    country: str | None = None
    pincode: str | None = None
    qualification: str | None = None
    skills: str | None = None


class EmployeeOfficeCreate(BaseModel):
    employee_id: int
    employee_code: str | None = None
    department: str | None = None
    designation: str | None = None
    employment_type: str | None = None
    work_location: str | None = None
    manager_name: str | None = None
    joining_date: date | None = None
    probation_end_date: date | None = None
    total_years_experience: Decimal | None = None
    years_in_company: Decimal | None = None
    salary_per_month: Decimal | None = None
    bonus: Decimal | None = Decimal("0")
    shift_type: str | None = None
    work_mode: str | None = None
    official_status: str | None = "Active"
    last_promotion_date: date | None = None


class EmployeeSalaryLeaveCreate(BaseModel):
    employee_id: int
    salary_month: str | None = None
    total_working_days: int | None = None
    present_days: int | None = None
    leave_days: int | None = 0
    unpaid_leave_days: int | None = 0
    paid_leave_days: int | None = 0
    overtime_hours: Decimal | None = Decimal("0")
    monthly_salary: Decimal | None = None
    leave_deduction: Decimal | None = Decimal("0")
    tax_deduction: Decimal | None = Decimal("0")
    pf_deduction: Decimal | None = Decimal("0")
    bonus_amount: Decimal | None = Decimal("0")
    final_salary: Decimal | None = None
    payment_status: str | None = "Pending"
    payment_date: date | None = None
    remarks: str | None = None


class EmployeeAttendanceCreate(BaseModel):
    employee_id: int
    attendance_date: date | None = None
    check_in: time | None = None
    check_out: time | None = None
    total_hours: Decimal | None = None
    attendance_status: str | None = None


class EmployeeDocumentCreate(BaseModel):
    employee_id: int
    document_type: str | None = None
    document_number: str | None = None
    file_path: str | None = None


class EmployeeOut(EmployeePersonalCreate):
    model_config = ConfigDict(from_attributes=True)

    employee_id: int


class EmployeeSummary(BaseModel):
    employee_id: int
    first_name: str
    last_name: str | None
    personal_email: str | None
    office_email: str | None
    personal_phone: str | None
    department: str | None
    designation: str | None
    employee_code: str | None
    salary_per_month: Decimal | None
    final_salary: Decimal | None


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    route: str
    answer: str
    sql_context: list[dict[str, Any]] = []
    vector_sources: list[dict[str, Any]] = []


class IngestTextRequest(BaseModel):
    title: str
    text: str
    source_type: str = "manual"

