from langgraph.graph import (
    StateGraph,
    END
)

import re
import json
import requests

from sqlalchemy import text
from db import engine

from llm import llm
from state import AgentState

BASE_URL = "http://127.0.0.1:8000"


# ==================================================
# HELPER FUNCTIONS
# ==================================================
def execute_sql(sql_query):

    response = requests.post(
        f"{BASE_URL}/query",
        params={
            "sql": sql_query
        }
    )

    return response.json()


def format_rows(rows):

    if not rows:
        return "No matching records found."

    formatted = []

    for row in rows:

        row_text = []

        for k, v in row.items():

            row_text.append(
                f"{k}: {v}"
            )

        formatted.append(
            "\n".join(row_text)
        )

    return "\n\n".join(formatted)


def format_count_response(data):

    return (
        f"count: {data['count']}\n\n"
        + format_rows(data.get("rows", []))
    )


def format_employee(data):
    """
    Formats a single employee record
    showing all 11 fields clearly.
    """

    fields = [
        ("Employee ID",      data.get("employee_id")),
        ("Name",             data.get("employee_name")),
        ("Age",              data.get("age")),
        ("Gender",           data.get("gender")),
        ("Department",       data.get("department")),
        ("Designation",      data.get("designation")),
        ("Salary",           data.get("salary")),
        ("Experience Years", data.get("experience_years")),
        ("City",             data.get("city")),
        ("Manager",          data.get("manager_name")),
        ("Employment Type",  data.get("employment_type")),
    ]

    return "\n".join(
        f"{label}: {value}"
        for label, value in fields
        if value is not None
    )


def extract_numbers(text):

    return [
        int(n)
        for n in text.split()
        if n.isdigit()
    ]


def extract_employee_id(text):
    """
    Extract employee ID only when
    explicitly mentioned.

    Examples:
    - employee id 42
    - id 42
    - employee 42
    """

    match = re.search(
        r'(?:employee\s+id|employee|id)\s+(\d+)',
        text,
        re.IGNORECASE
    )

    return (
        int(match.group(1))
        if match else None
    )


# ==================================================
# EMPLOYEE QUERY NODE
# ==================================================
def employee_query(state):

    original_question = state["question"]
    question = original_question
    question_lower = question.lower().strip()

    # --------------------------------------------------
    # SMALL TALK / GREETINGS
    # --------------------------------------------------
    greetings = [
        "hi", "hello", "hey", "hii",
        "helo", "good morning",
        "good afternoon", "good evening"
    ]

    how_are_you = [
        "how are you", "how are you?",
        "how r u", "how do you do"
    ]

    thanks = [
        "thanks", "thank you", "thx"
    ]

    bye_words = [
        "bye", "goodbye",
        "see you", "bye bye"
    ]

    if question_lower in greetings:

        state["response"] = (
            "Hello! How can I help you "
            "with employee database information today?"
        )

        return state

    if question_lower in how_are_you:

        state["response"] = (
            "I'm doing well, thank you! "
            "How can I assist you with "
            "employee database information?"
        )

        return state

    if question_lower in thanks:

        state["response"] = (
            "You're welcome! Let me know "
            "if you need any employee "
            "database information."
        )

        return state

    if question_lower in bye_words:

        state["response"] = (
            "Goodbye! Have a great day."
        )

        return state

    # --------------------------------------------------
    # MEMORY / CHAT HISTORY
    # --------------------------------------------------
    chat_history = state.get("chat_history", [])

    memory_text = ""

    for msg in chat_history[-5:]:

        memory_text += (
            f"User: {msg['user']}\n"
            f"Assistant: {msg['assistant']}\n"
        )

    # --------------------------------------------------
    # FOLLOW-UP CONTEXT RESOLUTION
    # e.g. "what are they?" after asking about cities
    # --------------------------------------------------
    last_user_question = ""

    if chat_history:

        last_user_question = (
            chat_history[-1]["user"]
            .lower()
            .strip()
        )

    if question_lower in [
        "what are they",
        "what are they?",
        "what are those",
        "what are them"
    ]:

        topic_map = {
            ("city", "cities"): "what are the cities",
            ("department", "departments"): "what are departments",
            ("designation", "designations"): "what are designations",
            ("gender", "genders"): "what are the genders",
            ("employment_type", "employment type"): "what are employment types",
            ("manager", "managers", "manager_name"): "what are the managers",
        }

        for keywords, resolved_question in topic_map.items():

            if any(kw in last_user_question for kw in keywords):

                question = resolved_question
                question_lower = question.lower().strip()
                break

    # ==================================================
    # CAPABILITY QUESTIONS
    # e.g. "can you delete an employee?"
    # ==================================================
    capability_keywords = [
        "can you", "can i", "how do i",
        "how to", "what can you", "what can i",
        "is it possible", "are you able",
        "do you support", "do you have"
    ]

    capability_actions = [
        "delete", "remove", "create", "add",
        "update", "edit", "change", "modify",
        "search", "find", "get", "show",
        "fetch", "list", "view"
    ]

    if any(kw in question_lower for kw in capability_keywords):

        if any(
            action in question_lower
            for action in capability_actions
        ):

            state["response"] = (
                "Yes! Here's what I can do:\n\n"
                "READ:\n"
                "- Search employees by age, salary, city, "
                "department, designation, gender, "
                "employment type, manager, or experience\n"
                "- Get full employee details by ID: "
                "'show employee 42' or 'get employee id 42'\n"
                "- Ask complex questions like 'top 5 highest paid' "
                "or 'average salary by department'\n\n"
                "WRITE:\n"
                "- Create employee: 'create employee name John age 30...'\n"
                "- Update employee: 'update employee id 42 salary to 90000'\n"
                "- Delete employee: 'delete employee 42'"
            )

            _save_memory(
                state, chat_history,
                original_question
            )

            return state

    # ==================================================
    # INTENT DETECTION — WRITE OPERATIONS
    # ==================================================

    # --------------------------------------------------
    # CREATE EMPLOYEE
    # --------------------------------------------------
    create_keywords = [
        "add employee", "create employee",
        "create an employee", "add an employee",
        "new employee", "add a new", "insert employee",
        "register employee", "onboard employee",
        "create a new employee"
    ]

    if any(kw in question_lower for kw in create_keywords):

        state["response"] = _handle_create(
            question, question_lower
        )

        _save_memory(
            state, chat_history,
            original_question
        )

        return state

    # --------------------------------------------------
    # UPDATE EMPLOYEE
    # --------------------------------------------------
    update_keywords = [
        "update",
        "edit",
        "change",
        "modify",

        "update employee",
        "edit employee",
        "change employee",
        "modify employee",

        "name to",
        "salary to",
        "city to",
        "department to",
        "designation to",
        "gender to",
        "manager to",
        "experience to",
        "employment type to",
        "employment to",
        "age to"
    ]

    is_update_request = (
        any(
            kw in question_lower
            for kw in update_keywords
        )
        and (
            "employee" in question_lower
            or "id" in question_lower
        )
        and extract_employee_id(
            question_lower
        ) is not None
    )

    if is_update_request:

        print(
            "UPDATE ROUTE TRIGGERED"
        )

        state["response"] = (
            _handle_update(
                question,
                question_lower
            )
        )

        _save_memory(
            state,
            chat_history,
            original_question
        )

        return state

    # --------------------------------------------------
    # DELETE EMPLOYEE
    # --------------------------------------------------
    delete_keywords = [
        "delete employee", "remove employee",
        "fire employee", "terminate employee",
        "delete id", "remove id"
    ]

    if any(kw in question_lower for kw in delete_keywords):

        state["response"] = _handle_delete(
            question, question_lower
        )

        _save_memory(
            state, chat_history,
            original_question
        )

        return state
        # ==================================================
    # GET EMPLOYEE BY ID
    # Triggers on: "employee 42", "show employee 42",
    # "get employee id 42", "details of 42",
    # "employee id 42", "fetch employee 42"
    # ==================================================
    id_keywords = [
        "employee id", "employee no",
        "employee number", "show employee",
        "get employee", "fetch employee",
        "details of employee", "info of employee",
        "tell me about employee", "who is employee",
        "employee details", "employee info"
    ]

    id_triggered = any(
        kw in question_lower
        for kw in id_keywords
    )

    # Also trigger on bare "employee <number>"
    # e.g. "employee 42"
    bare_employee_id = re.search(
        r'\bemployee\s+(\d+)\b',
        question_lower
    )

    if id_triggered or bare_employee_id:

        emp_id = (
            int(bare_employee_id.group(1))
            if bare_employee_id
            else extract_employee_id(question_lower)
        )

        if emp_id:

            response = requests.get(
                f"{BASE_URL}/employees/{emp_id}"
            )

            data = response.json()

            if "error" in data:

                state["response"] = (
                    f"Employee ID {emp_id} not found."
                )

            else:

                state["response"] = format_employee(data)

            _save_memory(
                state, chat_history,
                original_question
            )

            return state


    # ==================================================
    # DIRECT API ROUTING — READ OPERATIONS
    # ==================================================

    # --------------------------------------------------
    # AGE ROUTE
    # --------------------------------------------------
    age_keywords = [
        "age",
        "older than",
        "greater than",
        "more than",
        "above",
        "less than",
        "younger than",
        "below",
        "<",
        ">"
    ]

    if any(
        kw in question_lower
        for kw in age_keywords
    ):

        numbers = extract_numbers(
            question_lower
        )

        if numbers:

            age = numbers[0]

            # -------------------------
            # LESS THAN (<)
            # -------------------------
            if (
                "<" in question_lower
                or "less than" in question_lower
                or "younger than" in question_lower
                or "below" in question_lower
            ):

                sql_query = f"""
                    SELECT *
                    FROM employees
                    WHERE age < {age}
                    LIMIT 100
                """

            # -------------------------
            # GREATER THAN (>)
            # -------------------------
            elif (
                ">" in question_lower
                or "greater than" in question_lower
                or "older than" in question_lower
                or "more than" in question_lower
                or "above" in question_lower
            ):

                sql_query = f"""
                    SELECT *
                    FROM employees
                    WHERE age > {age}
                    LIMIT 100
                """

            # -------------------------
            # EQUAL (=)
            # -------------------------
            else:

                sql_query = f"""
                    SELECT *
                    FROM employees
                    WHERE age = {age}
                    LIMIT 100
                """

            with engine.connect() as conn:

                result = conn.execute(
                    text(sql_query)
                )

                rows = result.fetchall()

            formatted_rows = [
                dict(row._mapping)
                for row in rows
            ]

            state["response"] = format_rows(
                formatted_rows
            )

            _save_memory(
                state,
                chat_history,
                original_question
            )

            return state

    # --------------------------------------------------
    # SALARY ROUTE
    # --------------------------------------------------
    if (
        "salary" in question_lower
        or "earning" in question_lower
        or "earns" in question_lower
        or "income" in question_lower
        or "pay" in question_lower
        or "lakh" in question_lower
    ):
        
        numbers = extract_numbers(
            question_lower
        )

        if numbers:

            salary = numbers[0]

            # lakh support
            if "lakh" in question_lower:
                salary *= 100000

            # -------------------------
            # LESS THAN (<)
            # -------------------------
            if (
                "<" in question_lower
                or "less than" in question_lower
                or "below" in question_lower
            ):

                sql_query = f"""
                    SELECT *
                    FROM employees
                    WHERE salary < {salary}
                    LIMIT 100
                """

            # -------------------------
            # GREATER THAN (>)
            # -------------------------
            elif (
                ">" in question_lower
                or "greater than" in question_lower
                or "more than" in question_lower
                or "above" in question_lower
            ):

                sql_query = f"""
                    SELECT *
                    FROM employees
                    WHERE salary > {salary}
                    LIMIT 100
                """

            # -------------------------
            # EQUAL (=)
            # -------------------------
            else:

                sql_query = f"""
                    SELECT *
                    FROM employees
                    WHERE salary = {salary}
                    LIMIT 100
                """

            with engine.connect() as conn:

                result = conn.execute(
                    text(sql_query)
                )

                rows = result.fetchall()

            formatted_rows = [
                dict(row._mapping)
                for row in rows
            ]

            state["response"] = format_rows(
                formatted_rows
            )

            _save_memory(
                state,
                chat_history,
                original_question
            )

            return state

    # --------------------------------------------------
    # EXPERIENCE ROUTE
    # --------------------------------------------------
    if (
        "experience" in question_lower
        or "years of experience" in question_lower
        or "experienced" in question_lower
        or "worked for" in question_lower
    ):

        print("EXPERIENCE ROUTE TRIGGERED")

        numbers = extract_numbers(
            question_lower
        )

        if numbers:

            years = numbers[0]

            print(
                "Experience extracted:",
                years
            )

            # -------------------------
            # LESS THAN (<)
            # -------------------------
            if (
                "<" in question_lower
                or "less than" in question_lower
                or "below" in question_lower
            ):

                sql_query = f"""
                    SELECT *
                    FROM employees
                    WHERE experience_years < {years}
                    LIMIT 100
                """

            # -------------------------
            # GREATER THAN (>)
            # -------------------------
            elif (
                ">" in question_lower
                or "greater than" in question_lower
                or "more than" in question_lower
                or "above" in question_lower
            ):

                sql_query = f"""
                    SELECT *
                    FROM employees
                    WHERE experience_years > {years}
                    LIMIT 100
                """

            # -------------------------
            # EQUAL (=)
            # -------------------------
            else:

                sql_query = f"""
                    SELECT *
                    FROM employees
                    WHERE experience_years = {years}
                    LIMIT 100
                """

            with engine.connect() as conn:

                result = conn.execute(
                    text(sql_query)
                )

                rows = result.fetchall()

            formatted_rows = [
                dict(row._mapping)
                for row in rows
            ]

            state["response"] = format_rows(
                formatted_rows
            )

            _save_memory(
                state,
                chat_history,
                original_question
            )

            return state

    # --------------------------------------------------
    # CITY ROUTE
    # --------------------------------------------------
    cities = [
        "hyderabad", "bangalore", "chennai",
        "mumbai", "pune", "delhi", "noida"
    ]

    for city in cities:

        if city in question_lower:

            data = requests.get(
                f"{BASE_URL}/employees/city",
                params={"city": city}
            ).json()

            state["response"] = format_count_response(data)

            _save_memory(state, chat_history, original_question)

            return state

    # --------------------------------------------------
    # DEPARTMENT ROUTE
    # --------------------------------------------------
    departments = [
        "hr", "it", "finance", "marketing",
        "operations", "sales", "engineering",
        "legal", "admin", "support"
    ]

    for dept in departments:

        if dept in question_lower:

            data = requests.get(
                f"{BASE_URL}/employees/department",
                params={"department": dept}
            ).json()

            state["response"] = format_count_response(data)

            _save_memory(state, chat_history, original_question)

            return state

    # --------------------------------------------------
    # DESIGNATION ROUTE
    # --------------------------------------------------
    designations = [
        "manager", "engineer", "analyst",
        "developer", "director", "executive",
        "associate", "consultant", "lead",
        "intern", "senior", "junior", "vp", "ceo", "cto"
    ]

    for designation in designations:

        if designation in question_lower:

            data = requests.get(
                f"{BASE_URL}/employees/designation",
                params={"designation": designation}
            ).json()

            state["response"] = format_count_response(data)

            _save_memory(state, chat_history, original_question)

            return state

    # --------------------------------------------------
    # MANAGER ROUTE
    # --------------------------------------------------
    manager_keywords = [
        "manager",
        "manager name",
        "under manager",
        "reports to",
        "managed by",
        "under"
    ]

    if any(
        kw in question_lower
        for kw in manager_keywords
    ):

        m = re.search(
            r'(?:manager(?:\s+name)?\s+|managed\s+by\s+|reports\s+to\s+|under\s+)([A-Za-z]+(?:\s+[A-Za-z]+)*)',
            question,
            re.IGNORECASE
        )

        if m:

            manager_name = (
                m.group(1)
                .strip()
            )

            print(
                "Manager extracted:",
                manager_name
            )

            response = requests.get(
                f"{BASE_URL}/employees/manager",
                params={
                    "manager": manager_name
                }
            )

            data = response.json()

            state["response"] = (
                format_count_response(data)
            )

            _save_memory(
                state,
                chat_history,
                original_question
            )

            return state

    # ==================================================
    # SQL FALLBACK — LLM GENERATES QUERY
    # ==================================================
    prompt = f"""
You are a PostgreSQL SQL expert assistant
for an employee management system.

Your ONLY job is to generate valid
PostgreSQL SELECT queries for the
employees table — nothing else.

===================================
DATABASE SCHEMA
===================================
Table: employees

Columns:
  employee_id        INTEGER (primary key)
  employee_name      TEXT
  age                INTEGER
  gender             TEXT        (e.g. Male, Female, Other)
  department         TEXT        (e.g. IT, HR, Finance, Marketing,
                                  Operations, Sales, Engineering)
  designation        TEXT        (e.g. Manager, Engineer, Analyst,
                                  Developer, Director, Intern)
  salary             INTEGER     (annual, in rupees)
  experience_years   INTEGER
  city               TEXT        (e.g. Hyderabad, Bangalore, Chennai,
                                  Mumbai, Pune, Delhi, Noida)
  manager_name       TEXT
  employment_type    TEXT        (Full-time, Part-time,
                                  Contract, Freelance, Intern)

===================================
STRICT RULES
===================================
1.  Return ONLY raw SQL. No explanation.
    No markdown. No backticks. No preamble.

2.  ONLY generate SELECT queries.
    NEVER generate: UPDATE, DELETE, DROP,
    ALTER, INSERT, TRUNCATE, CREATE.

3.  Use ONLY the employees table and
    exact column names listed above.
    NEVER hallucinate columns.

4.  Always use LIMIT 100 for row-returning
    queries (not needed for COUNT).

5.  Use COUNT(*) for counting rows.

6.  Use COUNT(DISTINCT col) for counting
    unique values.

7.  Use GROUP BY for aggregations.

8.  Use ORDER BY for ranking.

9.  Use LOWER() for case-insensitive
    text comparisons.

10. For salary in lakhs, multiply by 100000.
    (e.g. "5 lakh" = 500000)

11. Use IS NULL / IS NOT NULL for
    null checks.

12. Use ILIKE for partial/fuzzy
    name matching.

13. Understand follow-up questions
    using conversation context below.

14. If the question is unrelated to
    employees or is not a valid data query,
    return exactly:

    INVALID_QUERY

===================================
CONVERSATION CONTEXT (last 5 turns)
===================================
{memory_text}

===================================
QUERY EXAMPLES
===================================

Question: how many employees
SQL: SELECT COUNT(*) AS count FROM employees;

Question: how many departments
SQL: SELECT COUNT(DISTINCT department) AS count FROM employees;

Question: what are the departments
SQL: SELECT DISTINCT department FROM employees;

Question: what are the cities
SQL: SELECT DISTINCT city FROM employees;

Question: how many cities
SQL: SELECT COUNT(DISTINCT city) AS count FROM employees;

Question: what are the designations
SQL: SELECT DISTINCT designation FROM employees;

Question: how many designations
SQL: SELECT COUNT(DISTINCT designation) AS count FROM employees;

Question: what are the genders
SQL: SELECT DISTINCT gender FROM employees;

Question: what are the employment types
SQL: SELECT DISTINCT employment_type FROM employees;

Question: what are the managers
SQL: SELECT DISTINCT manager_name FROM employees;

Question: highest salary employee
SQL: SELECT * FROM employees ORDER BY salary DESC LIMIT 1;

Question: lowest salary employee
SQL: SELECT * FROM employees ORDER BY salary ASC LIMIT 1;

Question: top 5 highest paid employees
SQL: SELECT employee_name, salary FROM employees ORDER BY salary DESC LIMIT 5;

Question: top 5 most experienced employees
SQL: SELECT employee_name, experience_years FROM employees ORDER BY experience_years DESC LIMIT 5;

Question: department wise employee count
SQL: SELECT department, COUNT(*) AS employee_count FROM employees GROUP BY department ORDER BY employee_count DESC;

Question: average salary
SQL: SELECT AVG(salary) AS average_salary FROM employees;

Question: average salary by department
SQL: SELECT department, AVG(salary) AS average_salary FROM employees GROUP BY department ORDER BY average_salary DESC;

Question: highest salary in IT department
SQL: SELECT * FROM employees WHERE LOWER(department) = 'it' ORDER BY salary DESC LIMIT 1;

Question: employees with more than 10 years experience
SQL: SELECT * FROM employees WHERE experience_years > 10 LIMIT 100;

Question: full time employees in hyderabad
SQL: SELECT * FROM employees WHERE LOWER(employment_type) = 'full-time' AND LOWER(city) = 'hyderabad' LIMIT 100;

Question: female managers
SQL: SELECT * FROM employees WHERE LOWER(gender) = 'female' AND LOWER(designation) ILIKE '%manager%' LIMIT 100;

Question: employees under manager John
SQL: SELECT * FROM employees WHERE LOWER(manager_name) ILIKE '%john%' LIMIT 100;

Question: average age by gender
SQL: SELECT gender, AVG(age) AS average_age FROM employees GROUP BY gender;

Question: salary distribution by employment type
SQL: SELECT employment_type, AVG(salary) AS avg_salary, MIN(salary) AS min_salary, MAX(salary) AS max_salary FROM employees GROUP BY employment_type;

Question: count of employees per city
SQL: SELECT city, COUNT(*) AS count FROM employees GROUP BY city ORDER BY count DESC;

Question: youngest employee
SQL: SELECT * FROM employees ORDER BY age ASC LIMIT 1;

Question: oldest employee
SQL: SELECT * FROM employees ORDER BY age DESC LIMIT 1;

Question: engineers with salary above 80000
SQL: SELECT * FROM employees WHERE LOWER(designation) ILIKE '%engineer%' AND salary > 80000 LIMIT 100;

Question: employees hired under Ravi Kumar
SQL: SELECT * FROM employees WHERE LOWER(manager_name) ILIKE '%ravi kumar%' LIMIT 100;

Question: contract employees in finance
SQL: SELECT * FROM employees WHERE LOWER(employment_type) = 'contract' AND LOWER(department) = 'finance' LIMIT 100;

Question: hello
SQL: INVALID_QUERY

Question: what is the weather today
SQL: INVALID_QUERY

===================================
CURRENT QUESTION
===================================
{question}

SQL:"""

    # --------------------------------------------------
    # Invoke LLM
    # --------------------------------------------------
    sql_response = llm.invoke(prompt)

    sql_query = (
        sql_response.content
        .replace("```sql", "")
        .replace("```", "")
        .strip()
    )

    print(f"\nGenerated SQL:\n{sql_query}")

    # --------------------------------------------------
    # Handle INVALID_QUERY
    # --------------------------------------------------
    if sql_query == "INVALID_QUERY":

        state["response"] = (
            "I can only answer employee "
            "database questions. Please ask "
            "something related to employee data."
        )

    else:

        try:

            data = execute_sql(sql_query)

            if "error" in data:

                state["response"] = data["error"]

            else:

                state["response"] = format_rows(
                    data.get("rows", [])
                )

        except Exception as e:

            state["response"] = (
                f"API Error: {str(e)}"
            )

    _save_memory(state, chat_history, original_question)

    return state


# ==================================================
# WRITE OPERATION HANDLERS
# ==================================================

# Required fields for creating an employee
REQUIRED_EMPLOYEE_FIELDS = [
    "employee_name",
    "age",
    "gender",
    "department",
    "designation",
    "salary",
    "experience_years",
    "city",
    "manager_name",
    "employment_type"
]

FIELD_LABELS = {
    "employee_name":    "name",
    "age":              "age",
    "gender":           "gender (Male/Female/Other)",
    "department":       "department (e.g. IT, HR, Finance, Sales)",
    "designation":      "designation (e.g. Engineer, Manager, Analyst)",
    "salary":           "salary (integer)",
    "experience_years": "experience in years (integer)",
    "city":             "city (e.g. Hyderabad, Bangalore, Mumbai)",
    "manager_name":     "manager name",
    "employment_type":  "employment type (Full-time/Part-time/Contract/Freelance/Intern)"
}


def _extract_create_fields(question):
    """
    Extracts employee fields directly using regex.
    No LLM involved — fast and reliable.
    """

    q = question.lower()
    data = {}

    # employee_name
    m = re.search(
        r'name\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)',
        question,
        re.IGNORECASE
    )
    if m:
        data["employee_name"] = m.group(1).strip()

    # age
    m = re.search(r'age\s+(\d+)', q)
    if m:
        data["age"] = int(m.group(1))

    # gender
    for g in ["male", "female", "other"]:
        if g in q:
            data["gender"] = g.capitalize()
            break

    # salary
    m = re.search(r'salary\s+(\d+)', q)
    if m:
        sal = int(m.group(1))
        if "lakh" in q:
            sal *= 100000
        data["salary"] = sal

    # experience_years
    m = re.search(
        r'experience[_\s]*years?\s+(\d+)|(\d+)\s+years?\s+(?:of\s+)?experience',
        q
    )
    if m:
        data["experience_years"] = int(m.group(1) or m.group(2))

    # city
    for city in [
        "hyderabad", "bangalore", "chennai",
        "mumbai", "pune", "delhi", "noida"
    ]:
        if city in q:
            data["city"] = city.capitalize()
            break

    # department
    for dept in [
        "hr", "it", "finance", "marketing",
        "operations", "sales", "engineering",
        "legal", "admin", "support",
        "cybersecurity", "data engineering", "ai/ml"
    ]:
        if dept in q:
            data["department"] = dept.upper() if dept in ["hr", "it"] else dept.title()
            break

    # designation
    for desig in [
        "engineer", "manager", "analyst",
        "developer", "director", "executive",
        "associate", "consultant", "lead",
        "intern", "scientist", "architect"
    ]:
        if desig in q:
            data["designation"] = desig.title()
            break

    # employment_type
    emp_map = {
        "full-time": "Full-time",
        "full time": "Full-time",
        "part-time": "Part-time",
        "part time": "Part-time",
        "contract":  "Contract",
        "freelance": "Freelance",
        "intern":    "Intern",
    }
    for key, val in emp_map.items():
        if key in q:
            data["employment_type"] = val
            break

    # manager_name — look for "manager <name>" or "manager_name <name>"
    m = re.search(
        r'manager[_\s]*name\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)',
        question,
        re.IGNORECASE
    )
    if m:
        data["manager_name"] = m.group(1).strip()

    return data


def _handle_create(question, question_lower):
    """
    Extracts employee fields using regex (no LLM).
    If any required field is missing, asks the
    user to provide it instead of sending nulls.
    """

    employee_data = _extract_create_fields(question)

    # Find missing required fields
    missing = [
        FIELD_LABELS[field]
        for field in REQUIRED_EMPLOYEE_FIELDS
        if employee_data.get(field) is None
    ]

    if missing:

        missing_list = "\n".join(
            f"  - {m}" for m in missing
        )

        return (
            f"I need a few more details to create "
            f"the employee. Please provide:\n"
            f"{missing_list}"
        )

    try:

        # All fields present — call the API
        api_response = requests.post(
            f"{BASE_URL}/employees",
            json=employee_data
        )

        result = api_response.json()

        if "error" in result:

            return f"Failed to create employee: {result['error']}"

        emp_id = result.get("employee_id")

        return (
            f"Employee created successfully!\n"
            f"Employee ID: {emp_id}\n"
            f"Name: {employee_data.get('employee_name')}\n"
            f"Department: {employee_data.get('department')}\n"
            f"Designation: {employee_data.get('designation')}\n"
            f"City: {employee_data.get('city')}\n"
            f"Salary: {employee_data.get('salary')}\n"
            f"Employment Type: {employee_data.get('employment_type')}"
        )

    except Exception as e:

        return (
            f"Could not create employee.\n"
            f"Error: {str(e)}"
        )



def _handle_update(
    question,
    question_lower
):

    emp_id = extract_employee_id(
        question_lower
    )

    if not emp_id:

        return (
            "Please provide an "
            "employee ID.\n"
            "Example:\n"
            "'Update employee "
            "42 salary to 90000'"
        )

    update_data = {}

    # -------------------------
    # NAME
    # -------------------------
    m = re.search(
        r'name\s+to\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)',
        question,
        re.IGNORECASE
    )

    if m:

        update_data[
            "employee_name"
        ] = (
            m.group(1)
            .strip()
            .title()
        )

    # -------------------------
    # AGE
    # -------------------------
    m = re.search(
        r'age\s+to\s+(\d+)',
        question_lower
    )

    if m:

        update_data[
            "age"
        ] = int(
            m.group(1)
        )

    # -------------------------
    # SALARY
    # -------------------------
    m = re.search(
        r'salary\s+to\s+(\d+)',
        question_lower
    )

    if m:

        salary = int(
            m.group(1)
        )

        if "lakh" in question_lower:
            salary *= 100000

        update_data[
            "salary"
        ] = salary

    # -------------------------
    # CITY
    # -------------------------
    m = re.search(
        r'city\s+to\s+([A-Za-z\s]+?)(?=\s+(?:salary|department|designation|gender|manager|experience|employment|age|name|$))',
        question,
        re.IGNORECASE
    )

    if m:

        update_data[
            "city"
        ] = (
            m.group(1)
            .strip()
            .title()
        )

    # -------------------------
    # DEPARTMENT
    # -------------------------
    m = re.search(
        r'department\s+to\s+([A-Za-z\s]+?)(?=\s+(?:salary|city|designation|gender|manager|experience|employment|age|name|$))',
        question,
        re.IGNORECASE
    )

    if m:

        dept = (
            m.group(1)
            .strip()
        )

        update_data[
            "department"
        ] = (
            dept.upper()
            if dept.lower()
            in ["it", "hr"]
            else dept.title()
        )

    # -------------------------
    # DESIGNATION
    # -------------------------
    m = re.search(
        r'designation\s+to\s+([A-Za-z\s]+?)(?=\s+(?:salary|city|department|gender|manager|experience|employment|age|name|$))',
        question,
        re.IGNORECASE
    )

    if m:

        update_data[
            "designation"
        ] = (
            m.group(1)
            .strip()
            .title()
        )

    # -------------------------
    # GENDER
    # -------------------------
    m = re.search(
        r'gender\s+to\s+(male|female|other)',
        question_lower
    )

    if m:

        update_data[
            "gender"
        ] = (
            m.group(1)
            .capitalize()
        )

    # -------------------------
    # EXPERIENCE
    # -------------------------
    m = re.search(
        r'(?:experience|experience years)\s+to\s+(\d+)',
        question_lower
    )

    if m:

        update_data[
            "experience_years"
        ] = int(
            m.group(1)
        )

    # -------------------------
    # MANAGER
    # -------------------------
    m = re.search(
        r'manager(?:\s+name)?\s+to\s+([A-Za-z\s]+?)(?=\s+(?:salary|city|department|designation|gender|experience|employment|age|name|$))',
        question,
        re.IGNORECASE
    )

    if m:

        update_data[
            "manager_name"
        ] = (
            m.group(1)
            .strip()
            .title()
        )

    # -------------------------
    # EMPLOYMENT TYPE
    # -------------------------
    m = re.search(
        r'(?:employment(?:\s+type)?)\s+to\s+(full-time|full time|part-time|part time|contract|freelance|intern)',
        question_lower
    )

    if m:

        emp_type = (
            m.group(1)
            .replace(
                " ",
                "-"
            )
            .title()
        )

        update_data[
            "employment_type"
        ] = emp_type

    if not update_data:

        return (
            "No fields to update "
            "were identified.\n"
            "Example:\n"
            "'update employee "
            "1001 salary to 90000'"
        )

    try:

        response = requests.put(
            f"{BASE_URL}/employees/{emp_id}",
            json=update_data
        )

        result = (
            response.json()
        )

        if "error" in result:

            return (
                f"Update failed:\n"
                f"{result['error']}"
            )

        return (
            f"Employee "
            f"{emp_id} "
            f"updated successfully!\n"
            f"Updated fields: "
            f"{', '.join(update_data.keys())}"
        )

    except Exception as e:

        return (
            f"Update failed:\n"
            f"{str(e)}"
        )


def _handle_delete(question, question_lower):
    """
    Extracts employee ID and calls
    DELETE /employees/{id}?confirm=true.
    """

    emp_id = extract_employee_id(question_lower)

    if not emp_id:

        return (
            "Please provide an employee ID "
            "to delete. Example: "
            "'Delete employee 42'"
        )

    try:

        api_response = requests.delete(
            f"{BASE_URL}/employees/{emp_id}",
            params={"confirm": True}
        )

        result = api_response.json()

        if "error" in result:

            return f"Failed to delete: {result['error']}"

        return (
            f"Employee deleted successfully!\n"
            f"ID: {result.get('employee_id')}\n"
            f"Name: {result.get('employee_name')}"
        )

    except Exception as e:

        return f"Delete failed: {str(e)}"


# ==================================================
# MEMORY HELPER
# ==================================================

def _save_memory(state, chat_history, original_question):

    chat_history.append(
        {
            "user": original_question,
            "assistant": state["response"]
        }
    )

    state["chat_history"] = chat_history


# ==================================================
# BUILD LANGGRAPH
# ==================================================

builder = StateGraph(AgentState)

builder.add_node(
    "employee_query",
    employee_query
)

builder.set_entry_point("employee_query")

builder.add_edge("employee_query", END)

graph = builder.compile()