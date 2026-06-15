from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import crud, schemas
from app.database import get_db

router = APIRouter(prefix="/employees", tags=["employees"])


@router.post("", response_model=schemas.EmployeeOut)
def add_employee(payload: schemas.EmployeePersonalCreate, db: Session = Depends(get_db)):
    return crud.create_employee(db, payload)


@router.get("")
def list_employees(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    return crud.list_employee_summaries(db)


@router.get("/{employee_id}")
def get_employee(employee_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    profile = crud.get_employee_full_profile(db, employee_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Employee not found")
    return profile


@router.post("/office")
def add_office_details(payload: schemas.EmployeeOfficeCreate, db: Session = Depends(get_db)):
    return crud.model_to_dict(crud.create_office_details(db, payload))


@router.post("/salary-leave")
def add_salary_leave(payload: schemas.EmployeeSalaryLeaveCreate, db: Session = Depends(get_db)):
    return crud.model_to_dict(crud.create_salary_leave_details(db, payload))


@router.post("/attendance")
def add_attendance(payload: schemas.EmployeeAttendanceCreate, db: Session = Depends(get_db)):
    return crud.model_to_dict(crud.create_attendance(db, payload))


@router.post("/documents")
def add_employee_document(payload: schemas.EmployeeDocumentCreate, db: Session = Depends(get_db)):
    return crud.model_to_dict(crud.create_document(db, payload))
