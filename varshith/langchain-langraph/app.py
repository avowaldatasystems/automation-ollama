"""Interactive RAG chat: LangGraph retrieve -> generate over Qdrant + Ollama."""

from __future__ import annotations

import os
import sys
from typing import TypedDict

import httpx
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import END, StateGraph
from ollama import ResponseError
from qdrant_client.http.exceptions import UnexpectedResponse

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_collection")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL")


def _ollama_kwargs() -> tuple[dict, dict]:
    llm_kw: dict = {"model": LLM_MODEL}
    embed_kw: dict = {"model": EMBED_MODEL}
    if OLLAMA_BASE:
        llm_kw["base_url"] = OLLAMA_BASE
        embed_kw["base_url"] = OLLAMA_BASE
    return llm_kw, embed_kw


class GraphState(TypedDict):
    question: str
    context: str
    answer: str
    k: int


def create_rag_graph():
    """Create vector store + compiled graph. Call after Qdrant has the collection (run ingest.py)."""
    llm_kw, embed_kw = _ollama_kwargs()
    llm = ChatOllama(**llm_kw)
    embeddings = OllamaEmbeddings(**embed_kw)

    try:
        vectorstore = QdrantVectorStore.from_existing_collection(
            collection_name=COLLECTION,
            embedding=embeddings,
            url=QDRANT_URL,
        )
    except UnexpectedResponse as exc:
        if exc.status_code == 404:
            raise RuntimeError(
                f"Qdrant has no collection named `{COLLECTION}`. "
                "With Qdrant running, index documents first:\n"
                "  python ingest.py"
            ) from exc
        raise
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        raise RuntimeError(
            f"Cannot connect to Qdrant at {QDRANT_URL}. "
            "Start Qdrant (for example: docker run -p 6333:6333 qdrant/qdrant) and retry."
        ) from exc

    def retrieve(state: GraphState) -> dict:
        question = state["question"]
        k = int(state.get("k") or 6)
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(question)
        context = "\n\n".join(
            f"[source: {doc.metadata.get('source','unknown')}]\n{doc.page_content}"
            for doc in docs
        )
        return {"question": question, "context": context}

    def generate(state: GraphState) -> dict:
        question = state["question"]
        context = state["context"]

        prompt = f"""
You are a helpful AI assistant.

Answer the question using the provided context only.

Context:
{context}

Question:
{question}
"""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            answer = response.content
        except ResponseError as exc:
            answer = (
                f"[Ollama error] {exc}\n\n"
                "Fix: run `ollama serve`, pull the model (`ollama pull "
                f"{LLM_MODEL}`), or set OLLAMA_LLM_MODEL in .env to a tag from `ollama list`. "
                "If the error mentions system memory, use a smaller model (for example "
                "`ollama pull llama3.2:1b` and set OLLAMA_LLM_MODEL=llama3.2:1b)."
            )

        return {"question": question, "context": context, "answer": answer}

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


def main() -> None:
    try:
        app = create_rag_graph()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    while True:
        question = input("\nAsk: ")
        if question.strip().lower() == "exit":
            break
        result = app.invoke({"question": question})
        print("\nAnswer:")
        print(result["answer"])


if __name__ == "__main__":
    main()
