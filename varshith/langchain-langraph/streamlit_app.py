"""Streamlit UI for the LangGraph RAG pipeline (Qdrant + Ollama)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from app import create_rag_graph


def _ensure_graph():
    if "rag_graph" not in st.session_state:
        try:
            st.session_state.rag_graph = create_rag_graph()
            st.session_state.rag_graph_error = None
        except RuntimeError as exc:
            st.session_state.rag_graph = None
            st.session_state.rag_graph_error = str(exc)


def main() -> None:
    st.set_page_config(page_title="Simple RAG", page_icon="📄", layout="centered")
    st.title("Simple RAG")
    st.caption("Ask questions about documents you indexed with `python ingest.py`.")

    _ensure_graph()

    if st.session_state.rag_graph_error:
        st.error(st.session_state.rag_graph_error)
        st.info(
            "**Before using the app:** start Qdrant and Ollama, pull models, then run "
            "`python ingest.py` from this folder."
        )
        if st.button("Retry connection"):
            for key in ("rag_graph", "rag_graph_error"):
                st.session_state.pop(key, None)
            st.rerun()
        return

    question = st.text_input("Your question", placeholder="What is this document about?")
    k = st.slider("Top-K retrieval", min_value=1, max_value=12, value=6, step=1)
    show_context = st.checkbox("Show retrieved context", value=False)

    ask = st.button("Ask", type="primary")

    if ask and question.strip():
        with st.spinner("Retrieving and generating answer…"):
            # pass through k so the graph can use it if nodes support it
            result = st.session_state.rag_graph.invoke({"question": question.strip(), "k": k})

        st.subheader("Answer")
        st.write(result["answer"])

        if show_context and result.get("context"):
            with st.expander("Retrieved context"):
                st.text(result["context"])
    elif ask and not question.strip():
        st.warning("Enter a question first.")


if __name__ == "__main__":
    main()
