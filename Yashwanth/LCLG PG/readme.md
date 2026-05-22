# Employee AI Database Assistant

An AI-powered Employee Database Assistant built using:

- Streamlit
- LangGraph
- Ollama (Llama 3.2)
- FastAPI
- PostgreSQL
- SQLAlchemy

This project allows users to query employee data using natural language. The system intelligently routes user requests through predefined APIs or AI-generated SQL queries.

---

# Architecture

```text
User
 ↓
Streamlit
 ↓
LangGraph
 ↓
Ollama (Llama 3.2)
 ↓
FastAPI
 ↓
PostgreSQL
 ↓
Results
 ↓
Streamlit
```

---

# Features

### Fast Employee Filtering

Supports employee filtering by:

- Age
- Salary
- City
- Department
- Designation
- Experience
- Employment Type
- Manager
- Gender

### AI SQL Querying

For complex questions, the system uses Ollama + LangGraph to generate SQL dynamically.

Example queries:

```text
highest salary employee
department wise employee count
average salary by department
```

---

# Tech Stack

| Technology | Purpose |
|------------|----------|
| Streamlit | Frontend UI |
| LangGraph | Workflow Engine |
| Ollama | Local LLM |
| Llama 3.2 | AI Model |
| FastAPI | API Layer |
| PostgreSQL | Database |
| SQLAlchemy | Database Connection |

---

# Project Structure

```text
project/
│── app.py
│── graph.py
│── db_api.py
│── db.py
│── llm.py
│── state.py
│── load.py
│── employee_dataset.xlsx
│── requirements.txt
│── README.md
```

---

# Installation

## 1. Clone Repository

```bash
git clone <your-repository-url>
cd project-name
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Install Ollama

Download Ollama:

https://ollama.com/download

Verify installation:

```bash
ollama --version
```

Install model:

```bash
ollama pull llama3.2
```

---

# PostgreSQL Setup

Install PostgreSQL and pgAdmin.

Create a database:

```sql
CREATE DATABASE employee_db;
```

Update database connection in:

```text
db.py
```

Example:

```python
from sqlalchemy import create_engine

DATABASE_URL = (
    "postgresql://postgres:password@localhost:5432/employee_db"
)

engine = create_engine(
    DATABASE_URL
)
```

---

# Load Dataset

Place:

```text
employee_dataset.xlsx
```

inside the project folder.

Run:

```bash
python load.py
```

This uploads employee data into PostgreSQL.

---

# Run FastAPI

Start the API server:

```bash
uvicorn db_api:app --reload --port 8001
```

Swagger UI:

```text
http://127.0.0.1:8001/docs
```

---

# APIs

### GET APIs

Employee filtering APIs:

```text
/age
/salary
/city
/department
/designation
/experience
/employment_type
/manager
/gender
```

### POST API

Dynamic SQL query execution:

```text
/query
```

Only safe `SELECT` queries are allowed.

---

# Run Streamlit

Start the application:

```bash
streamlit run app.py
```

---

# Example Queries

```text
employees in hyderabad
```

```text
salary above 10 lakh
```

```text
show IT employees
```

```text
highest salary employee
```

```text
department wise employee count
```

---

# Requirements

streamlit
fastapi
uvicorn
sqlalchemy
psycopg2
pandas
openpyxl
langgraph
langchain
langchain-ollama
typing-extensions


---

# Future Improvements

- Multi-filter APIs
- Vector Database (pgvector)
- Better memory handling
- Dashboard analytics

---

# Built With

```text
LangGraph + Ollama + FastAPI + PostgreSQL + Streamlit
```
