from fastapi import FastAPI, Query
from sqlalchemy import text
from pydantic import BaseModel
from typing import Optional

from db import engine

app = FastAPI()


# -----------------------------------
# HOME
# -----------------------------------
@app.get("/")
def home():

    return {
        "message":
        "Employee API Running"
    }


# -----------------------------------
# AGE API
# -----------------------------------
@app.get("/employees/age")
def get_by_age(
    age: int
):

    with engine.connect() as conn:

        count_result = conn.execute(
            text("""
                SELECT COUNT(*)
                FROM employees
                WHERE age > :age
            """),
            {"age": age}
        )

        count = count_result.scalar()

        result = conn.execute(
            text("""
                SELECT *
                FROM employees
                WHERE age > :age
                LIMIT 100
            """),
            {"age": age}
        )

        rows = result.fetchall()

    return {
        "count": count,
        "rows": [
            dict(row._mapping)
            for row in rows
        ]
    }


# -----------------------------------
# SALARY API
# -----------------------------------
@app.get("/employees/salary")
def get_by_salary(
    salary: int
):

    with engine.connect() as conn:

        count_result = conn.execute(
            text("""
                SELECT COUNT(*)
                FROM employees
                WHERE salary > :salary
            """),
            {"salary": salary}
        )

        count = count_result.scalar()

        result = conn.execute(
            text("""
                SELECT *
                FROM employees
                WHERE salary > :salary
                LIMIT 100
            """),
            {"salary": salary}
        )

        rows = result.fetchall()

    return {
        "count": count,
        "rows": [
            dict(row._mapping)
            for row in rows
        ]
    }


# -----------------------------------
# CITY API
# -----------------------------------
@app.get("/employees/city")
def get_by_city(
    city: str
):

    with engine.connect() as conn:

        count_result = conn.execute(
            text("""
                SELECT COUNT(*)
                FROM employees
                WHERE LOWER(city)
                = LOWER(:city)
            """),
            {"city": city}
        )

        count = count_result.scalar()

        result = conn.execute(
            text("""
                SELECT *
                FROM employees
                WHERE LOWER(city)
                = LOWER(:city)
                LIMIT 100
            """),
            {"city": city}
        )

        rows = result.fetchall()

    return {
        "count": count,
        "rows": [
            dict(row._mapping)
            for row in rows
        ]
    }


# -----------------------------------
# DEPARTMENT API
# -----------------------------------
@app.get("/employees/department")
def get_by_department(
    department: str
):

    with engine.connect() as conn:

        count_result = conn.execute(
            text("""
                SELECT COUNT(*)
                FROM employees
                WHERE LOWER(department)
                = LOWER(:department)
            """),
            {"department": department}
        )

        count = count_result.scalar()

        result = conn.execute(
            text("""
                SELECT *
                FROM employees
                WHERE LOWER(department)
                = LOWER(:department)
                LIMIT 100
            """),
            {"department": department}
        )

        rows = result.fetchall()

    return {
        "count": count,
        "rows": [
            dict(row._mapping)
            for row in rows
        ]
    }


# -----------------------------------
# DESIGNATION API
# -----------------------------------
@app.get("/employees/designation")
def get_by_designation(
    designation: str
):

    with engine.connect() as conn:

        count_result = conn.execute(
            text("""
                SELECT COUNT(*)
                FROM employees
                WHERE LOWER(designation)
                = LOWER(:designation)
            """),
            {"designation": designation}
        )

        count = count_result.scalar()

        result = conn.execute(
            text("""
                SELECT *
                FROM employees
                WHERE LOWER(designation)
                = LOWER(:designation)
                LIMIT 100
            """),
            {"designation": designation}
        )

        rows = result.fetchall()

    return {
        "count": count,
        "rows": [
            dict(row._mapping)
            for row in rows
        ]
    }


# -----------------------------------
# EXPERIENCE API
# -----------------------------------
@app.get("/employees/experience")
def get_by_experience(
    years: int
):

    with engine.connect() as conn:

        count_result = conn.execute(
            text("""
                SELECT COUNT(*)
                FROM employees
                WHERE experience_years
                > :years
            """),
            {"years": years}
        )

        count = count_result.scalar()

        result = conn.execute(
            text("""
                SELECT *
                FROM employees
                WHERE experience_years
                > :years
                LIMIT 100
            """),
            {"years": years}
        )

        rows = result.fetchall()

    return {
        "count": count,
        "rows": [
            dict(row._mapping)
            for row in rows
        ]
    }


# -----------------------------------
# EMPLOYMENT TYPE API
# -----------------------------------
@app.get("/employees/employment_type")
def get_by_employment_type(
    employment_type: str
):

    with engine.connect() as conn:

        count_result = conn.execute(
            text("""
                SELECT COUNT(*)
                FROM employees
                WHERE LOWER(employment_type)
                = LOWER(:employment_type)
            """),
            {"employment_type": employment_type}
        )

        count = count_result.scalar()

        result = conn.execute(
            text("""
                SELECT *
                FROM employees
                WHERE LOWER(employment_type)
                = LOWER(:employment_type)
                LIMIT 100
            """),
            {"employment_type": employment_type}
        )

        rows = result.fetchall()

    return {
        "count": count,
        "rows": [
            dict(row._mapping)
            for row in rows
        ]
    }


# -----------------------------------
# MANAGER API
# -----------------------------------
@app.get("/employees/manager")
def get_by_manager(
    manager: str
):

    with engine.connect() as conn:

        count_result = conn.execute(
            text("""
                SELECT COUNT(*)
                FROM employees
                WHERE LOWER(manager_name)
                = LOWER(:manager)
            """),
            {"manager": manager}
        )

        count = count_result.scalar()

        result = conn.execute(
            text("""
                SELECT *
                FROM employees
                WHERE LOWER(manager_name)
                = LOWER(:manager)
                LIMIT 100
            """),
            {"manager": manager}
        )

        rows = result.fetchall()

    return {
        "count": count,
        "rows": [
            dict(row._mapping)
            for row in rows
        ]
    }


# -----------------------------------
# GENDER API
# -----------------------------------
@app.get("/employees/gender")
def get_by_gender(
    gender: str
):

    with engine.connect() as conn:

        count_result = conn.execute(
            text("""
                SELECT COUNT(*)
                FROM employees
                WHERE LOWER(gender)
                = LOWER(:gender)
            """),
            {"gender": gender}
        )

        count = count_result.scalar()

        result = conn.execute(
            text("""
                SELECT *
                FROM employees
                WHERE LOWER(gender)
                = LOWER(:gender)
                LIMIT 100
            """),
            {"gender": gender}
        )

        rows = result.fetchall()

    return {
        "count": count,
        "rows": [
            dict(row._mapping)
            for row in rows
        ]
    }


# ==================================================
# CREATE MODEL
# ==================================================
class EmployeeCreate(BaseModel):
    employee_name: str
    age: int
    gender: str
    department: str
    designation: str
    salary: int
    experience_years: int
    city: str
    manager_name: str
    employment_type: str


# ==================================================
# UPDATE MODEL
# ==================================================
class EmployeeUpdate(BaseModel):
    employee_name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    department: Optional[str] = None
    designation: Optional[str] = None
    salary: Optional[int] = None
    experience_years: Optional[int] = None
    city: Optional[str] = None
    manager_name: Optional[str] = None
    employment_type: Optional[str] = None


# -----------------------------------
# CREATE EMPLOYEE API
# -----------------------------------
@app.post("/employees")
def create_employee(
    employee: EmployeeCreate
):

    try:

        with engine.begin() as conn:

            result = conn.execute(
                text("""
                    INSERT INTO employees (
                        employee_name,
                        age,
                        gender,
                        department,
                        designation,
                        salary,
                        experience_years,
                        city,
                        manager_name,
                        employment_type
                    )
                    VALUES (
                        :employee_name,
                        :age,
                        :gender,
                        :department,
                        :designation,
                        :salary,
                        :experience_years,
                        :city,
                        :manager_name,
                        :employment_type
                    )
                    RETURNING employee_id
                """),
                {
                    "employee_name": employee.employee_name,
                    "age": employee.age,
                    "gender": employee.gender,
                    "department": employee.department,
                    "designation": employee.designation,
                    "salary": employee.salary,
                    "experience_years": employee.experience_years,
                    "city": employee.city,
                    "manager_name": employee.manager_name,
                    "employment_type": employee.employment_type
                }
            )

            row = result.fetchone()
            employee_id = row[0] if row else None

        return {
            "message": "Employee added successfully",
            "employee_id": int(employee_id) if employee_id else None
        }

    except Exception as e:

        return {"error": str(e)}


# -----------------------------------
# UPDATE EMPLOYEE API
# -----------------------------------
@app.put("/employees/{employee_id}")
def update_employee(
    employee_id: int,
    employee: dict
):

    try:

        with engine.begin() as conn:

            # -------------------------
            # Get existing employee
            # -------------------------
            result = conn.execute(
                text("""
                    SELECT *
                    FROM employees
                    WHERE employee_id = :id
                """),
                {
                    "id": employee_id
                }
            )

            existing = result.fetchone()

            if not existing:

                return {
                    "error":
                    "Employee not found"
                }

            existing = dict(
                existing._mapping
            )

            # -------------------------
            # Keep old values
            # Update only sent fields
            # -------------------------
            merged_data = existing.copy()

            for key, value in employee.items():

                if value is not None:

                    merged_data[key] = value

            # remove employee_id
            merged_data.pop(
                "employee_id",
                None
            )

            update_query = text("""
                UPDATE employees
                SET
                    employee_name = :employee_name,
                    age = :age,
                    gender = :gender,
                    department = :department,
                    designation = :designation,
                    salary = :salary,
                    experience_years = :experience_years,
                    city = :city,
                    manager_name = :manager_name,
                    employment_type = :employment_type
                WHERE employee_id = :employee_id
            """)

            merged_data[
                "employee_id"
            ] = employee_id

            conn.execute(
                update_query,
                merged_data
            )

        return {
            "message":
            "Employee updated successfully",

            "employee_id":
            employee_id,

            "updated_fields":
            list(employee.keys())
        }

    except Exception as e:

        return {
            "error":
            str(e)
        }

# -----------------------------------
# DELETE EMPLOYEE API
# -----------------------------------
@app.delete("/employees/{employee_id}")
def delete_employee(
    employee_id: int,
    confirm: bool = Query(False)
):

    try:

        with engine.begin() as conn:

            employee_result = conn.execute(
                text("""
                    SELECT employee_name
                    FROM employees
                    WHERE employee_id = :employee_id
                """),
                {"employee_id": employee_id}
            )

            employee = employee_result.fetchone()

            if not employee:

                return {"error": "Employee ID not found"}

            employee_name = employee[0]

            if not confirm:

                return {
                    "message":
                    f"Are you sure you want to delete "
                    f"'{employee_name}' (ID: {employee_id})?",
                    "confirmation_required": True,
                    "how_to_confirm":
                    f"/employees/{employee_id}?confirm=true"
                }

            conn.execute(
                text("""
                    DELETE FROM employees
                    WHERE employee_id = :employee_id
                """),
                {"employee_id": employee_id}
            )

        return {
            "message": "Employee deleted successfully",
            "employee_id": employee_id,
            "employee_name": employee_name
        }

    except Exception as e:

        return {"error": str(e)}


# -----------------------------------
# GET EMPLOYEE BY ID API
# KEEP THIS LAST — dynamic route must
# come after all static routes
# -----------------------------------
@app.get("/employees/{employee_id}")
def get_employee_by_id(
    employee_id: int
):

    try:

        with engine.connect() as conn:

            result = conn.execute(
                text("""
                    SELECT *
                    FROM employees
                    WHERE employee_id = :employee_id
                """),
                {"employee_id": employee_id}
            )

            row = result.fetchone()

        if not row:

            return {"error": "Employee ID not found"}

        return dict(row._mapping)

    except Exception as e:

        return {"error": str(e)}