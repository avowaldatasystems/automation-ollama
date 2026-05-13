"""Minimal LangGraph scaffold (extend or import patterns from app.py)."""

from typing import TypedDict

from langgraph.graph import END, StateGraph


class GraphState(TypedDict):
    question: str
    context: str
    answer: str


def retrieve(state: GraphState) -> GraphState:
    return state


def generate(state: GraphState) -> GraphState:
    return state


workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")

workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()
