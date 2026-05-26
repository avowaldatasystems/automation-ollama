from langgraph.graph import (
    StateGraph,
    END
)

import re
import json
import requests

from llm import llm
from state import AgentState


BASE_URL = "http://127.0.0.1:8001"


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


def extract_numbers(text):

    return [
        int(n)
        for n in text.split()
        if n.isdigit()
    ]


def extract_employee_id(text):
    """
    Extracts a numeric employee ID
    from a question string.
    """

    match = re.search(r'\b(\d+)\b', text)

    return int(match.group(1)) if match else None


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
    # INTENT DETECTION — WRITE OPERATIONS
    # ==================================================

    # --------------------------------------------------
    # CREATE EMPLOYEE
    # --------------------------------------------------
    create_keywords = [
        "add employee", "create employee",
        "new employee", "add a new", "insert employee",
        "register employee", "onboard employee"
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
        "update employee", "edit employee",
        "change employee", "modify employee",
        "update salary", "change salary",
        "update department", "change department",
        "update designation", "change designation",
        "update city", "change city",
        "update age", "change age",
        "update manager", "change manager",
        "update employment", "change employment"
    ]

    if any(kw in question_lower for kw in update_keywords):

        state["response"] = _handle_update(
            question, question_lower
        )

        _save_memory(
            state, chat_history,
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
    # DIRECT API ROUTING — READ OPERATIONS
    # ==================================================

    # --------------------------------------------------
    # AGE ROUTE
    # --------------------------------------------------
    age_keywords = [
        "age", "older than",
        "above age", "age greater than",
        "aged above", "aged more than"
    ]

    if any(kw in question_lower for kw in age_keywords):

        numbers = extract_numbers(question_lower)

        if numbers:

            data = requests.get(
                f"{BASE_URL}/employees/age",
                params={"age": numbers[0]}
            ).json()

            state["response"] = format_count_response(data)

            _save_memory(state, chat_history, original_question)

            return state

    # --------------------------------------------------
    # SALARY ROUTE
    # --------------------------------------------------
    salary_keywords = [
        "salary", "earning",
        "earns", "lakh", "income", "pay"
    ]

    if any(kw in question_lower for kw in salary_keywords):

        numbers = extract_numbers(question_lower)

        if numbers:

            salary = numbers[0]

            if "lakh" in question_lower:
                salary *= 100000

            data = requests.get(
                f"{BASE_URL}/employees/salary",
                params={"salary": salary}
            ).json()

            state["response"] = format_count_response(data)

            _save_memory(state, chat_history, original_question)

            return state

    # --------------------------------------------------
    # EXPERIENCE ROUTE
    # --------------------------------------------------
    experience_keywords = [
        "experience", "years of experience",
        "experienced more than", "worked for more than",
        "experience greater than", "more than",
    ]

    if any(kw in question_lower for kw in experience_keywords):

        numbers = extract_numbers(question_lower)

        if numbers:

            data = requests.get(
                f"{BASE_URL}/employees/experience",
                params={"years": numbers[0]}
            ).json()

            state["response"] = format_count_response(data)

            _save_memory(state, chat_history, original_question)

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
    # EMPLOYMENT TYPE ROUTE
    # --------------------------------------------------
    employment_types = [
        "full-time", "full time",
        "part-time", "part time",
        "contract", "freelance", "intern"
    ]

    for emp_type in employment_types:

        if emp_type in question_lower:

            normalized = emp_type.replace(" ", "-")

            data = requests.get(
                f"{BASE_URL}/employees/employment_type",
                params={"employment_type": normalized}
            ).json()

            state["response"] = format_count_response(data)

            _save_memory(state, chat_history, original_question)

            return state

    # --------------------------------------------------
    # GENDER ROUTE
    # --------------------------------------------------
    genders = ["male", "female", "other"]

    for gender in genders:

        if gender in question_lower:

            data = requests.get(
                f"{BASE_URL}/employees/gender",
                params={"gender": gender}
            ).json()

            state["response"] = format_count_response(data)

            _save_memory(state, chat_history, original_question)

            return state

    # --------------------------------------------------
    # MANAGER ROUTE
    # e.g. "employees under John Smith"
    # --------------------------------------------------
    manager_keywords = [
        "under manager", "reports to",
        "managed by", "under"
    ]

    if any(kw in question_lower for kw in manager_keywords):

        # Let LLM extract manager name via SQL fallback
        pass

    # --------------------------------------------------
    # GET BY EMPLOYEE ID ROUTE
    # e.g. "show employee 42", "details of id 7"
    # --------------------------------------------------
    id_keywords = [
        "employee id", "employee number",
        "show employee", "details of employee",
        "fetch employee", "get employee"
    ]

    if any(kw in question_lower for kw in id_keywords):

        emp_id = extract_employee_id(question_lower)

        if emp_id:

            response = requests.get(
                f"{BASE_URL}/employees/{emp_id}"
            )

            data = response.json()

            if "error" in data:

                state["response"] = data["error"]

            else:

                state["response"] = format_rows([data])

            _save_memory(state, chat_history, original_question)

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

def _handle_create(question, question_lower):
    """
    Uses LLM to extract employee fields
    from the question and calls POST /employees.
    """

    extract_prompt = f"""
Extract employee details from this request
and return a JSON object with these exact keys:

  employee_name, age, gender, department,
  designation, salary, experience_years,
  city, manager_name, employment_type

Rules:
- Return ONLY valid JSON. No explanation.
- No markdown. No backticks. No preamble.
- If a field is missing, use null.
- age, salary, experience_years must be integers.
- employment_type must be exactly one of:
  Full-time, Part-time, Contract, Freelance, Intern
- Do NOT invent values not mentioned in the request.

Request: {question}

Output:"""

    response = llm.invoke(extract_prompt)

    raw = (
        response.content
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    try:

        employee_data = json.loads(raw)

        # Remove null fields
        employee_data = {
            k: v
            for k, v in employee_data.items()
            if v is not None
        }

        api_response = requests.post(
            f"{BASE_URL}/employees",
            json=employee_data
        )

        result = api_response.json()

        print(f"[CREATE] API result: {result}")

        if "error" in result:

            return f"Failed to create employee: {result['error']}"

        emp_id = (
            result.get("employee_id")
            or result.get("id")
        )

        return (
            f"Employee created successfully!\n"
            f"Employee ID: {emp_id}"
        )

    except Exception as e:

        return (
            f"Could not parse employee details "
            f"from your request. Please provide: "
            f"name, age, gender, department, "
            f"designation, salary, experience, "
            f"city, manager, and employment type.\n"
            f"Error: {str(e)}"
        )


def _handle_update(question, question_lower):
    """
    Uses LLM to extract employee ID and
    update fields, then calls PUT /employees/{id}.
    """

    emp_id = extract_employee_id(question_lower)

    if not emp_id:

        return (
            "Please provide an employee ID "
            "to update. Example: "
            "'Update employee 42 salary to 90000'"
        )

    extract_prompt = f"""
Extract ONLY the fields explicitly mentioned
to be changed in this request.

Available fields:
  employee_name, age, gender, department,
  designation, salary, experience_years,
  city, manager_name, employment_type

Rules:
- Return ONLY valid JSON. No explanation.
- No markdown. No backticks. No preamble.
- Include ONLY fields the user explicitly mentions changing.
- Do NOT invent or guess values for other fields.
- salary, age, experience_years must be integers.

Example:
Request: update employee id 42 name shivani
Output: {{"employee_name": "shivani"}}

Example:
Request: update employee 10 salary to 90000
Output: {{"salary": 90000}}

Request: {question}

Output:"""

    response = llm.invoke(extract_prompt)

    raw = (
        response.content
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    try:

        update_data = json.loads(raw)

        if not update_data:

            return "No fields to update were identified."

        api_response = requests.put(
            f"{BASE_URL}/employees/{emp_id}",
            json=update_data
        )

        result = api_response.json()

        if "error" in result:

            return f"Failed to update: {result['error']}"

        updated = result.get("updated_fields", [])

        return (
            f"Employee {emp_id} updated successfully!\n"
            f"Updated fields: {', '.join(updated)}"
        )

    except Exception as e:

        return (
            f"Could not parse update details. "
            f"Example: 'Update employee 42 salary to 90000'\n"
            f"Error: {str(e)}"
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