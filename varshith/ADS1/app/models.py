from sqlalchemy import Date, DateTime, DECIMAL, Enum, ForeignKey, Integer, String, Text, Time, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class EmployeePersonalDetails(Base):
    __tablename__ = "employee_personal_details"

    employee_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str | None] = mapped_column(String(100))
    gender: Mapped[str | None] = mapped_column(Enum("Male", "Female", "Other"))
    date_of_birth: Mapped[Date | None] = mapped_column(Date)
    age: Mapped[int | None] = mapped_column(Integer)
    personal_email: Mapped[str | None] = mapped_column(String(150), unique=True)
    office_email: Mapped[str | None] = mapped_column(String(150), unique=True)
    personal_phone: Mapped[str | None] = mapped_column(String(20))
    emergency_contact: Mapped[str | None] = mapped_column(String(20))
    father_name: Mapped[str | None] = mapped_column(String(150))
    mother_name: Mapped[str | None] = mapped_column(String(150))
    blood_group: Mapped[str | None] = mapped_column(String(10))
    marital_status: Mapped[str | None] = mapped_column(
        Enum("Single", "Married", "Divorced", "Widowed")
    )
    aadhaar_number: Mapped[str | None] = mapped_column(String(20), unique=True)
    pan_number: Mapped[str | None] = mapped_column(String(20), unique=True)
    nationality: Mapped[str | None] = mapped_column(String(100), default="Indian")
    current_address: Mapped[str | None] = mapped_column(Text)
    permanent_address: Mapped[str | None] = mapped_column(Text)
    city: Mapped[str | None] = mapped_column(String(100))
    state: Mapped[str | None] = mapped_column(String(100))
    country: Mapped[str | None] = mapped_column(String(100))
    pincode: Mapped[str | None] = mapped_column(String(20))
    qualification: Mapped[str | None] = mapped_column(String(200))
    skills: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())

    office_details: Mapped[list["EmployeeOfficeDetails"]] = relationship(
        back_populates="employee", cascade="all, delete-orphan"
    )
    salary_leave_details: Mapped[list["EmployeeSalaryLeaveDetails"]] = relationship(
        back_populates="employee", cascade="all, delete-orphan"
    )
    attendance: Mapped[list["EmployeeAttendance"]] = relationship(
        back_populates="employee", cascade="all, delete-orphan"
    )
    documents: Mapped[list["EmployeeDocument"]] = relationship(
        back_populates="employee", cascade="all, delete-orphan"
    )


class EmployeeOfficeDetails(Base):
    __tablename__ = "employee_office_details"

    office_record_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    employee_id: Mapped[int | None] = mapped_column(
        ForeignKey("employee_personal_details.employee_id", ondelete="CASCADE")
    )
    employee_code: Mapped[str | None] = mapped_column(String(50), unique=True)
    department: Mapped[str | None] = mapped_column(String(100))
    designation: Mapped[str | None] = mapped_column(String(100))
    employment_type: Mapped[str | None] = mapped_column(
        Enum("Full-Time", "Part-Time", "Intern", "Contract")
    )
    work_location: Mapped[str | None] = mapped_column(String(150))
    manager_name: Mapped[str | None] = mapped_column(String(150))
    joining_date: Mapped[Date | None] = mapped_column(Date)
    probation_end_date: Mapped[Date | None] = mapped_column(Date)
    total_years_experience: Mapped[float | None] = mapped_column(DECIMAL(4, 1))
    years_in_company: Mapped[float | None] = mapped_column(DECIMAL(4, 1))
    salary_per_month: Mapped[float | None] = mapped_column(DECIMAL(12, 2))
    bonus: Mapped[float | None] = mapped_column(DECIMAL(12, 2), default=0)
    shift_type: Mapped[str | None] = mapped_column(String(50))
    work_mode: Mapped[str | None] = mapped_column(Enum("Remote", "Hybrid", "Office"))
    official_status: Mapped[str | None] = mapped_column(
        Enum("Active", "On Leave", "Resigned", "Terminated"), default="Active"
    )
    last_promotion_date: Mapped[Date | None] = mapped_column(Date)
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())

    employee: Mapped[EmployeePersonalDetails] = relationship(back_populates="office_details")


class EmployeeSalaryLeaveDetails(Base):
    __tablename__ = "employee_salary_leave_details"

    salary_record_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    employee_id: Mapped[int | None] = mapped_column(
        ForeignKey("employee_personal_details.employee_id", ondelete="CASCADE")
    )
    salary_month: Mapped[str | None] = mapped_column(String(20))
    total_working_days: Mapped[int | None] = mapped_column(Integer)
    present_days: Mapped[int | None] = mapped_column(Integer)
    leave_days: Mapped[int | None] = mapped_column(Integer, default=0)
    unpaid_leave_days: Mapped[int | None] = mapped_column(Integer, default=0)
    paid_leave_days: Mapped[int | None] = mapped_column(Integer, default=0)
    overtime_hours: Mapped[float | None] = mapped_column(DECIMAL(5, 2), default=0)
    monthly_salary: Mapped[float | None] = mapped_column(DECIMAL(12, 2))
    leave_deduction: Mapped[float | None] = mapped_column(DECIMAL(12, 2), default=0)
    tax_deduction: Mapped[float | None] = mapped_column(DECIMAL(12, 2), default=0)
    pf_deduction: Mapped[float | None] = mapped_column(DECIMAL(12, 2), default=0)
    bonus_amount: Mapped[float | None] = mapped_column(DECIMAL(12, 2), default=0)
    final_salary: Mapped[float | None] = mapped_column(DECIMAL(12, 2))
    payment_status: Mapped[str | None] = mapped_column(Enum("Pending", "Paid"), default="Pending")
    payment_date: Mapped[Date | None] = mapped_column(Date)
    remarks: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())

    employee: Mapped[EmployeePersonalDetails] = relationship(back_populates="salary_leave_details")


class EmployeeAttendance(Base):
    __tablename__ = "employee_attendance"

    attendance_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    employee_id: Mapped[int | None] = mapped_column(
        ForeignKey("employee_personal_details.employee_id", ondelete="CASCADE")
    )
    attendance_date: Mapped[Date | None] = mapped_column(Date)
    check_in: Mapped[Time | None] = mapped_column(Time)
    check_out: Mapped[Time | None] = mapped_column(Time)
    total_hours: Mapped[float | None] = mapped_column(DECIMAL(4, 2))
    attendance_status: Mapped[str | None] = mapped_column(
        Enum("Present", "Absent", "Leave", "Half-Day")
    )
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())

    employee: Mapped[EmployeePersonalDetails] = relationship(back_populates="attendance")


class EmployeeDocument(Base):
    __tablename__ = "employee_documents"

    document_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    employee_id: Mapped[int | None] = mapped_column(
        ForeignKey("employee_personal_details.employee_id", ondelete="CASCADE")
    )
    document_type: Mapped[str | None] = mapped_column(String(100))
    document_number: Mapped[str | None] = mapped_column(String(100))
    file_path: Mapped[str | None] = mapped_column(Text)
    uploaded_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())

    employee: Mapped[EmployeePersonalDetails] = relationship(back_populates="documents")
