# Multi-PDF RAG Chatbot using LangGraph and Qdrant

An AI-powered multi-document chatbot built using LangGraph, Qdrant, OpenAI, and Streamlit.

This project allows users to upload multiple PDFs, store them as vector embeddings in Qdrant Cloud, and ask natural language questions across all uploaded documents using Retrieval-Augmented Generation (RAG).

---

# Features

* Multi-PDF ingestion
* Semantic search using Qdrant Vector Database
* Retrieval-Augmented Generation (RAG)
* LangGraph workflow orchestration
* OpenAI-powered responses
* Streamlit chatbot interface
* Context-aware document retrieval
* Embedding-based semantic search

---

# Tech Stack

| Technology   | Purpose                     |
| ------------ | --------------------------- |
| LangGraph    | Workflow orchestration      |
| LangChain    | LLM and retrieval framework |
| OpenAI       | Embeddings and LLM          |
| Qdrant Cloud | Vector database             |
| Streamlit    | Frontend UI                 |
| PyPDF        | PDF text extraction         |

---

# Project Architecture

```text
PDFs
 ↓
Text Extraction
 ↓
Chunking
 ↓
Embeddings
 ↓
Qdrant Vector Database
 ↓
Retriever
 ↓
OpenAI LLM
 ↓
Final Response
```

---

# Folder Structure

```text
pdf-chatbot/
│
├── app.py
├── graph.py
├── ingest.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── pdfs/
│    ├── sample.pdf
│    └── generative_ai_guide.pdf
```

---

# Requirements

## Software Requirements

* Python 3.11+
* Git
* Qdrant Cloud account
* OpenAI API key

## Python Libraries

Install all required libraries manually:

```bash
pip install langgraph
pip install langchain
pip install langchain-openai
pip install langchain-community
pip install langchain-qdrant
pip install langchain-text-splitters
pip install qdrant-client
pip install streamlit
pip install python-dotenv
pip install pypdf

```

Main libraries used:

* langgraph
* langchain
* langchain-openai
* langchain-community
* langchain-qdrant
* langchain-text-splitters
* qdrant-client
* streamlit
* python-dotenv
* pypdf

---

# Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key

QDRANT_URL=your_qdrant_cluster_url

QDRANT_API_KEY=your_qdrant_api_key
```

---

# Installation

## Clone Repository

```bash
git clone <repository_url>
cd pdf-chatbot
```

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Mac/Linux

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install langgraph langchain langchain-openai langchain-community langchain-qdrant langchain-text-splitters qdrant-client streamlit python-dotenv pypdf
```

---

# Add PDFs

Place all PDFs inside the `pdfs/` folder.

Example:

```text
pdfs/
 ├── sample.pdf
 ├── generative_ai_guide.pdf
 └── another_document.pdf
```

---

# Run PDF Ingestion

This converts PDFs into vector embeddings and stores them in Qdrant Cloud.

```bash
python ingest.py
```

Expected Output:

```text
Loading: pdfs/sample.pdf
Loading: pdfs/generative_ai_guide.pdf

Total chunks: 40+
All PDFs stored successfully!
```

---

# Run Chatbot

```bash
streamlit run app.py
```

---

# Example Questions

* What is Generative AI?
* Explain embeddings
* What is LangGraph?
* What is RAG architecture?
* Which place is famous for beaches?

---

# Core Concepts Used

* Retrieval-Augmented Generation (RAG)
* Vector Embeddings
* Semantic Search
* Vector Databases
* LangGraph Workflows
* Prompt Engineering
* Multi-Document Retrieval
* Conversational AI

---

# Professional Summary

Developed a multi-document Retrieval-Augmented Generation (RAG) chatbot using LangGraph, Qdrant, OpenAI, and Streamlit for semantic document retrieval and conversational question answering.
