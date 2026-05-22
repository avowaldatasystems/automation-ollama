from fastapi import FastAPI
from sqlalchemy import text

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
            {
                "age": age
            }
        )

        count = count_result.scalar()

        result = conn.execute(
            text("""
                SELECT *
                FROM employees
                WHERE age > :age
                LIMIT 100
            """),
            {
                "age": age
            }
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
            {
                "salary": salary
            }
        )

        count = count_result.scalar()

        result = conn.execute(
            text("""
                SELECT *
                FROM employees
                WHERE salary > :salary
                LIMIT 100
            """),
            {
                "salary": salary
            }
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
            {
                "city": city
            }
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
            {
                "city": city
            }
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
            {
                "department": department
            }
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
            {
                "department": department
            }
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
            {
                "designation": designation
            }
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
            {
                "designation": designation
            }
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
            {
                "years": years
            }
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
            {
                "years": years
            }
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
            {
                "employment_type":
                employment_type
            }
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
            {
                "employment_type":
                employment_type
            }
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
            {
                "manager": manager
            }
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
            {
                "manager": manager
            }
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
            {
                "gender": gender
            }
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
            {
                "gender": gender
            }
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
# NORMAL SQL QUERY API
# -----------------------------------
@app.post("/query")
def execute_query(
    sql: str
):

    try:

        dangerous_words = [
            "drop",
            "delete",
            "update",
            "alter",
            "truncate",
            "insert",
            "create"
        ]

        sql_lower = sql.lower()

        if not sql_lower.startswith(
            "select"
        ):
            return {
                "error":
                "Only SELECT queries are allowed."
            }

        if any(
            word in sql_lower
            for word
            in dangerous_words
        ):
            return {
                "error":
                "Unsafe SQL blocked."
            }

        with engine.connect() as conn:

            result = conn.execute(
                text(sql)
            )

            rows = result.fetchall()

            columns = result.keys()

        formatted_rows = []

        for row in rows:

            row_dict = {}

            for col, val in zip(
                columns,
                row
            ):
                row_dict[col] = val

            formatted_rows.append(
                row_dict
            )

        return {
            "rows":
            formatted_rows
        }

    except Exception as e:

        return {
            "error":
            str(e)
        }