import json
import os
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from sqlalchemy.orm import Session

from app.config import get_settings
from app.crud import search_employee_records
from app.vector_store import retrieve_documents


Route = Literal["sql", "vector", "both"]


class OfficeRagState(TypedDict, total=False):
    question: str
    route: Route
    sql_context: list[dict[str, Any]]
    vector_context: list[dict[str, Any]]
    answer: str


SQL_KEYWORDS = {
    "employee",
    "salary",
    "leave",
    "attendance",
    "department",
    "designation",
    "manager",
    "joining",
    "email",
    "phone",
    "code",
    "emp",
    "varshith",
    "personal",
    "office",
    "bonus",
    "paid",
    "present",
    "absent",
}

VECTOR_KEYWORDS = {
    "policy",
    "handbook",
    "rule",
    "procedure",
    "company",
    "document",
    "pdf",
    "doc",
    "docs",
    "guideline",
    "benefit",
    "training",
    "about",
    "explain",
    "theory",
}


def configure_langsmith() -> None:
    settings = get_settings()
    if settings.langsmith_tracing:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        if settings.langsmith_api_key:
            os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key


def get_llm() -> ChatOllama:
    settings = get_settings()
    return ChatOllama(
        model=settings.ollama_chat_model,
        base_url=settings.ollama_base_url,
        temperature=0.1,
    )


def route_question(question: str) -> Route:
    lowered = question.lower()
    has_sql = any(keyword in lowered for keyword in SQL_KEYWORDS)
    has_vector = any(keyword in lowered for keyword in VECTOR_KEYWORDS)

    if has_sql and has_vector:
        return "both"
    if has_sql:
        return "sql"
    if has_vector:
        return "vector"

    # Default to both for ambiguous office questions so the answer can use all context.
    return "both"


def build_graph(db: Session):
    configure_langsmith()

    def classify(state: OfficeRagState) -> OfficeRagState:
        state["route"] = route_question(state["question"])
        return state

    def retrieve_sql(state: OfficeRagState) -> OfficeRagState:
        state["sql_context"] = search_employee_records(db, state["question"])
        return state

    def retrieve_vector(state: OfficeRagState) -> OfficeRagState:
        try:
            state["vector_context"] = retrieve_documents(state["question"])
        except Exception as exc:
            state["vector_context"] = [
                {
                    "title": "Vector search unavailable",
                    "source_type": "system",
                    "source_path": "",
                    "content": (
                        "Qdrant/Ollama document retrieval is unavailable. "
                        f"Details: {exc}"
                    ),
                }
            ]
        return state

    def retrieve_both(state: OfficeRagState) -> OfficeRagState:
        state["sql_context"] = search_employee_records(db, state["question"])
        try:
            state["vector_context"] = retrieve_documents(state["question"])
        except Exception as exc:
            state["vector_context"] = [
                {
                    "title": "Vector search unavailable",
                    "source_type": "system",
                    "source_path": "",
                    "content": (
                        "Qdrant/Ollama document retrieval is unavailable. "
                        f"Details: {exc}"
                    ),
                }
            ]
        return state

    def generate_answer(state: OfficeRagState) -> OfficeRagState:
        sql_context = json.dumps(state.get("sql_context", []), default=str, indent=2)
        vector_context = json.dumps(state.get("vector_context", []), default=str, indent=2)
        messages = [
            SystemMessage(
                content=(
                    "You are an office assistant for an employee management system. "
                    "Answer only from the provided SQL employee data and Qdrant document context. "
                    "If the context is missing, say what data needs to be added. "
                    "Be concise, accurate, and mention whether the answer came from SQL, documents, or both."
                )
            ),
            HumanMessage(
                content=(
                    f"Question:\n{state['question']}\n\n"
                    f"SQL employee context:\n{sql_context}\n\n"
                    f"Document/vector context:\n{vector_context}"
                )
            ),
        ]
        try:
            state["answer"] = get_llm().invoke(messages).content
        except Exception as exc:
            state["answer"] = build_fallback_answer(state, exc)
        return state

    def route_edge(state: OfficeRagState) -> str:
        return state["route"]

    graph = StateGraph(OfficeRagState)
    graph.add_node("classify", classify)
    graph.add_node("retrieve_sql", retrieve_sql)
    graph.add_node("retrieve_vector", retrieve_vector)
    graph.add_node("retrieve_both", retrieve_both)
    graph.add_node("generate_answer", generate_answer)

    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify",
        route_edge,
        {"sql": "retrieve_sql", "vector": "retrieve_vector", "both": "retrieve_both"},
    )
    graph.add_edge("retrieve_sql", "generate_answer")
    graph.add_edge("retrieve_vector", "generate_answer")
    graph.add_edge("retrieve_both", "generate_answer")
    graph.add_edge("generate_answer", END)
    return graph.compile()


def ask_office_rag(db: Session, question: str) -> OfficeRagState:
    graph = build_graph(db)
    return graph.invoke({"question": question})


def build_fallback_answer(state: OfficeRagState, exc: Exception) -> str:
    parts = [
        "Ollama is not reachable, so I returned the retrieved context without LLM generation.",
        f"Route: {state.get('route', 'both')}.",
    ]
    sql_context = state.get("sql_context", [])
    if sql_context:
        parts.append("SQL employee matches:")
        for row in sql_context[:5]:
            name = " ".join(
                part for part in [str(row.get("first_name") or ""), str(row.get("last_name") or "")] if part
            )
            details = [
                f"ID {row.get('employee_id')}",
                name.strip() or "Unnamed employee",
                str(row.get("department") or "No department"),
                str(row.get("designation") or "No designation"),
            ]
            if row.get("salary_per_month") is not None:
                details.append(f"monthly salary {row.get('salary_per_month')}")
            if row.get("final_salary") is not None:
                details.append(f"final salary {row.get('final_salary')}")
            parts.append("- " + "; ".join(details))
    else:
        parts.append("No matching SQL employee rows were found.")

    vector_context = state.get("vector_context", [])
    if vector_context:
        parts.append("Document context:")
        for item in vector_context[:3]:
            title = item.get("title") or "Untitled"
            content = str(item.get("content") or "").strip().replace("\n", " ")
            if len(content) > 240:
                content = content[:237] + "..."
            parts.append(f"- {title}: {content}")

    parts.append(f"LLM error: {exc}")
    return "\n".join(parts)
