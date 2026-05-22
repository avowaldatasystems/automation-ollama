from langgraph.graph import (
    StateGraph,
    END
)

import requests

from llm import llm
from state import AgentState


# -----------------------------
# Helper Functions
# -----------------------------
def execute_sql(sql_query):

    response = requests.post(
        "http://127.0.0.1:8001/query",
        params={
            "sql": sql_query
        }
    )

    return response.json()


def format_rows(rows):

    if not rows:
        return "No matching data found."

    formatted = []

    for row in rows:

        row_text = []

        for k, v in row.items():

            row_text.append(
                f"{k}: {v}"
            )

        formatted.append(
            "\n".join(
                row_text
            )
        )

    return "\n\n".join(
        formatted
    )


# -----------------------------
# Employee Query Node
# -----------------------------
def employee_query(state):

    original_question = (
        state["question"]
    )

    question = (
        original_question
    )

    question_lower = (
        question.lower().strip()
    )

    # ==================================================
    # BASIC GREETINGS / SMALL TALK
    # ==================================================
    greetings = [
        "hi",
        "hello",
        "hey",
        "hii",
        "helo",
        "good morning",
        "good afternoon",
        "good evening"
    ]

    how_are_you = [
        "how are you",
        "how are you?",
        "how r u",
        "how do you do"
    ]

    thanks = [
        "thanks",
        "thank you",
        "thx"
    ]

    bye_words = [
        "bye",
        "goodbye",
        "see you",
        "bye bye"
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

    # -----------------------------
    # MEMORY
    # -----------------------------
    chat_history = state.get(
        "chat_history",
        []
    )

    memory_text = ""

    for msg in chat_history[-5:]:

        memory_text += (
            f"User: "
            f"{msg['user']}\n"
        )

        memory_text += (
            f"Assistant: "
            f"{msg['assistant']}\n"
        )

    # -----------------------------
    # Follow-up Context
    # -----------------------------
    last_user_question = ""

    if chat_history:

        last_user_question = (
            chat_history[-1]["user"]
            .lower()
            .strip()
        )

    # -----------------------------
    # Handle:
    # what are they?
    # -----------------------------
    if question_lower in [
        "what are they",
        "what are they?",
        "what are those",
        "what are them"
    ]:

        if (
            "city"
            in last_user_question
            or "cities"
            in last_user_question
        ):

            question = (
                "what are the cities"
            )

        elif (
            "department"
            in last_user_question
            or "departments"
            in last_user_question
        ):

            question = (
                "what are departments"
            )

        elif (
            "designation"
            in last_user_question
            or "designations"
            in last_user_question
        ):

            question = (
                "what are designations"
            )

    # ==================================================
    # AGE API ROUTE
    # ==================================================
    if any(
        phrase in question_lower
        for phrase in [
            "age",
            "older than",
            "above age",
            "age greater than",
            "more than"
        ]
    ):

        numbers = [
            int(word)
            for word
            in question_lower.split()
            if word.isdigit()
        ]

        if numbers:

            age = numbers[0]

            response = requests.get(
                "http://127.0.0.1:8001/employees/age",
                params={
                    "age":
                    age
                }
            )

            data = (
                response.json()
            )

            state["response"] = (
                f"count: "
                f"{data['count']}\n\n"
                + format_rows(
                    data["rows"]
                )
            )

            return state

    # ==================================================
    # SALARY API ROUTE
    # ==================================================
    if any(
        phrase in question_lower
        for phrase in [
            "salary",
            "earning",
            "earns",
            "lakh",
            "income"
        ]
    ):

        numbers = [
            int(word)
            for word
            in question_lower.split()
            if word.isdigit()
        ]

        if numbers:

            salary = (
                numbers[0]
            )

            # lakh support
            if (
                "lakh"
                in question_lower
            ):

                salary *= (
                    100000
                )

            response = requests.get(
                "http://127.0.0.1:8001/employees/salary",
                params={
                    "salary":
                    salary
                }
            )

            data = (
                response.json()
            )

            state["response"] = (
                f"count: "
                f"{data['count']}\n\n"
                + format_rows(
                    data["rows"]
                )
            )

            return state
            # ==================================================
    # CITY API ROUTE
    # ==================================================
    cities = [
        "hyderabad",
        "bangalore",
        "chennai",
        "mumbai",
        "pune",
        "delhi",
        "noida"
    ]

    for city in cities:

        if (
            city
            in question_lower
        ):

            response = requests.get(
                "http://127.0.0.1:8001/employees/city",
                params={
                    "city":
                    city
                }
            )

            data = (
                response.json()
            )

            state["response"] = (
                f"count: "
                f"{data['count']}\n\n"
                + format_rows(
                    data["rows"]
                )
            )

            return state

    # ==================================================
    # SQL PROMPT (Fallback)
    # ==================================================
    prompt = f"""
You are a PostgreSQL SQL expert.

Your ONLY job is to generate
PostgreSQL SELECT queries
for the employee database.

Database Table:

employees

Columns:

employee_id
employee_name
age
gender
department
designation
salary
experience_years
city
manager_name
employment_type

Conversation Context:
{memory_text}

IMPORTANT RULES:

1. Return ONLY SQL.

2. ONLY generate
SELECT queries.

3. NEVER generate:
UPDATE
DELETE
DROP
ALTER
INSERT
TRUNCATE

4. NEVER explain.

5. NEVER use markdown.

6. ONLY use
employees table.

7. NEVER hallucinate
columns.

8. Use exact schema names.

9. Use LIMIT 100
for large outputs.

10. Use COUNT(*)
for counts.

11. Use
COUNT(DISTINCT ...)
for categories.

12. Use GROUP BY
for aggregations.

13. Use ORDER BY
for ranking.

14. If unrelated:

Return EXACTLY:

INVALID_QUERY

15. Understand
follow-up questions.

Examples:

Question:
hello

Answer:
INVALID_QUERY


Question:
temperature on sun

Answer:
INVALID_QUERY


Question:
how many employees

SQL:
SELECT COUNT(*)
AS count
FROM employees;


Question:
how many departments

SQL:
SELECT
COUNT(DISTINCT department)
AS count
FROM employees;


Question:
what are departments

SQL:
SELECT DISTINCT
department
FROM employees;


Question:
how many cities

SQL:
SELECT
COUNT(DISTINCT city)
AS count
FROM employees;


Question:
what are cities

SQL:
SELECT DISTINCT
city
FROM employees;


Question:
how many designations

SQL:
SELECT
COUNT(DISTINCT designation)
AS count
FROM employees;


Question:
what are designations

SQL:
SELECT DISTINCT
designation
FROM employees;


Question:
highest salary employee

SQL:
SELECT *
FROM employees
ORDER BY salary DESC
LIMIT 1;


Question:
lowest salary employee

SQL:
SELECT *
FROM employees
ORDER BY salary ASC
LIMIT 1;


Question:
top 5 highest salaries

SQL:
SELECT
employee_name,
salary
FROM employees
ORDER BY salary DESC
LIMIT 5;


Question:
department wise employee count

SQL:
SELECT
department,
COUNT(*)
AS employee_count
FROM employees
GROUP BY department;


Question:
average salary

SQL:
SELECT
AVG(salary)
AS average_salary
FROM employees;


Question:
average salary by department

SQL:
SELECT
department,
AVG(salary)
AS average_salary
FROM employees
GROUP BY department;


Question:
highest salary in IT

SQL:
SELECT *
FROM employees
WHERE department='IT'
ORDER BY salary DESC
LIMIT 1;


Question:
employees with
more than 10 years
experience

SQL:
SELECT *
FROM employees
WHERE experience_years > 10
LIMIT 100;


Question:
full time employees

SQL:
SELECT *
FROM employees
WHERE employment_type
= 'Full-time'
LIMIT 100;


Current Question:
{question}

SQL:
"""

    # ==================================================
    # Generate SQL
    # ==================================================
    sql_response = llm.invoke(
        prompt
    )

    sql_query = (
        sql_response.content
        .replace(
            "```sql",
            ""
        )
        .replace(
            "```",
            ""
        )
        .strip()
    )

    print(
        "\nGenerated SQL:"
    )

    print(
        sql_query
    )

    # ==================================================
    # Invalid Query
    # ==================================================
    if sql_query == (
        "INVALID_QUERY"
    ):

        state["response"] = (
            "I can only answer "
            "employee database questions."
        )

    else:

        try:

            data = execute_sql(
                sql_query
            )

            if (
                "error"
                in data
            ):

                state["response"] = (
                    data["error"]
                )

            else:

                state["response"] = (
                    format_rows(
                        data["rows"]
                    )
                )

        except Exception as e:

            state["response"] = (
                f"API Error: "
                f"{str(e)}"
            )

    # ==================================================
    # SAVE MEMORY
    # ==================================================
    chat_history.append(
        {
            "user":
            original_question,

            "assistant":
            state["response"]
        }
    )

    state[
        "chat_history"
    ] = chat_history

    return state


# -----------------------------
# BUILD GRAPH
# -----------------------------
builder = StateGraph(
    AgentState
)

builder.add_node(
    "employee_query",
    employee_query
)

builder.set_entry_point(
    "employee_query"
)

builder.add_edge(
    "employee_query",
    END
)

graph = builder.compile()
