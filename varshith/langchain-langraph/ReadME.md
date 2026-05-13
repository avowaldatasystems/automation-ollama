# README.md

````markdown
# Simple RAG Pipeline using LangGraph, LangChain, Qdrant, Ollama, and Streamlit

A fully local Retrieval-Augmented Generation (RAG) application built using:

- LangGraph for workflow orchestration
- LangChain for AI integrations
- Qdrant as the vector database
- Ollama for local LLMs and embeddings
- Streamlit for the user interface

The system allows users to:
- Ingest PDF documents
- Convert documents into vector embeddings
- Store embeddings in Qdrant
- Retrieve semantically relevant chunks
- Generate contextual answers using local LLMs

---

# Project Architecture

```text
                ┌─────────────────────┐
                │     PDF Files       │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   PyPDFLoader/OCR   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Text Chunking       │
                │ Recursive Splitter  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Ollama Embeddings   │
                │ nomic-embed-text    │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Qdrant Vector DB    │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ LangGraph Workflow  │
                │ retrieve → generate │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Ollama LLM          │
                │ llama3              │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Streamlit UI        │
                └─────────────────────┘
````

---

# Technologies Used

| Technology                     | Purpose                        |
| ------------------------------ | ------------------------------ |
| LangChain                      | AI integrations and retrievers |
| LangGraph                      | Workflow orchestration         |
| Qdrant                         | Vector database                |
| Ollama                         | Local LLM and embeddings       |
| Streamlit                      | Web UI                         |
| PyPDFLoader                    | PDF text extraction            |
| OCR (Tesseract)                | Scanned PDF extraction         |
| RecursiveCharacterTextSplitter | Chunking                       |

---

# Folder Structure

```text
simple-rag/
│
├── app.py
├── graph.py
├── ingest.py
├── streamlit_app.py
├── .env
├── requirements.txt
│
├── data/
│   ├── sample.pdf
│   ├── Precision.pdf
│   └── Varshith_Thungapalli_AI_ML_Engineer.pdf
│
└── README.md
```

---

# Components Explanation

# 1. app.py

Main RAG workflow implementation.

Responsibilities:

* Connect to Qdrant
* Connect to Ollama
* Create LangGraph workflow
* Retrieve relevant documents
* Generate answers

Workflow:

```text
retrieve() → generate() → END
```

LangGraph manages:

* state passing
* execution flow
* node orchestration

---

# 2. graph.py

Minimal LangGraph scaffold.

Used as:

* starter workflow template
* learning scaffold
* experimentation graph

Contains:

* GraphState
* retrieve node
* generate node
* graph compilation

---

# 3. ingest.py

Responsible for document ingestion.

Pipeline:

```text
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Embeddings
 ↓
Qdrant Storage
```

Features:

* Multiple PDF ingestion
* OCR support
* Chunk splitting
* Embedding generation
* Qdrant storage

---

# 4. streamlit_app.py

Frontend UI for interacting with the RAG system.

Features:

* Question input
* Top-K retrieval slider
* Context display
* Answer generation
* Error handling

---

# 5. .env

Stores configuration variables.

Includes:

* Ollama URL
* Qdrant URL
* Model names
* PDF paths
* OCR configuration

---

# LangChain vs LangGraph

# LangChain

LangChain handles:

* LLM integration
* Embeddings
* Retrievers
* Vector stores
* Message formatting

Examples:

```python
ChatOllama
OllamaEmbeddings
QdrantVectorStore
HumanMessage
```

---

# LangGraph

LangGraph handles:

* Workflow execution
* Node orchestration
* State management
* Flow control

Example workflow:

```text
START
  ↓
retrieve
  ↓
generate
  ↓
END
```

---

# Retrieval-Augmented Generation (RAG)

RAG = Retrieve + Generate

Pipeline:

```text
User Question
      ↓
Semantic Retrieval
      ↓
Relevant Context
      ↓
LLM Generation
      ↓
Final Answer
```

Benefits:

* Reduces hallucinations
* Uses external knowledge
* Allows custom knowledge bases
* Works with private/local documents

---

# How Retrieval Works

# Step 1 — User Question

Example:

```text
What AI projects has Varshith worked on?
```

---

# Step 2 — Embedding Generation

The question is converted into vectors using:

```text
nomic-embed-text
```

---

# Step 3 — Vector Similarity Search

Qdrant searches for semantically similar chunks.

---

# Step 4 — Retrieve Relevant Chunks

Example:

```text
Projects:
- Underwater Waste Detection
- Blockchain Fraud Detection
- LangGraph RAG System
```

---

# Step 5 — Generate Final Answer

The retrieved context is sent to:

```text
llama3
```

to generate the final response.

---

# OCR Support

The system supports scanned/image PDFs using:

* pdf2image
* pytesseract
* poppler

Workflow:

```text
Scanned PDF
      ↓
Convert PDF Pages to Images
      ↓
OCR Text Extraction
      ↓
Chunking
      ↓
Embedding
```

---

# Environment Variables

Example `.env`

```env
# Ollama
OLLAMA_BASE_URL=http://localhost:11434

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=rag_collection

# Models
OLLAMA_LLM_MODEL=llama3
OLLAMA_EMBED_MODEL=nomic-embed-text

# PDF ingestion
DATA_PATHS=data/sample.pdf;data/Precision.pdf;data/Varshith_Thungapalli_AI_ML_Engineer.pdf

# OCR
ENABLE_OCR=1
POPPLER_PATH=C:/Users/VARSHITH/Downloads/Release-26.02.0-0/poppler-26.02.0/Library/bin
TESSERACT_CMD=C:/Program Files/Tesseract-OCR/tesseract.exe
```

---

# Installation

# 1. Clone Repository

```bash
git clone <repo_url>
cd simple-rag
```

---

# 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate:

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

---

# 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 4. Install Ollama

Download:
[https://ollama.com/download](https://ollama.com/download)

---

# 5. Pull Models

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

---

# 6. Start Ollama

```bash
ollama serve
```

---

# 7. Start Qdrant

```bash
docker run -d --name qdrant ^
-p 6333:6333 ^
-p 6334:6334 ^
qdrant/qdrant
```

---

# 8. Ingest Documents

```bash
python ingest.py
```

Expected:

```text
Documents embedded successfully!
```

---

# 9. Run Streamlit App

```bash
streamlit run streamlit_app.py
```

---

# Example Queries

```text
What AI skills does Varshith have?

Summarize Varshith's resume.

What is LangGraph?

Explain RAG architecture.

What projects are mentioned in the resume?
```

---

# Features

* Local/offline AI
* Multi-PDF ingestion
* OCR support
* Semantic search
* Streamlit UI
* LangGraph workflows
* Qdrant vector search
* Local embeddings
* Modular architecture

---

# Future Improvements

Possible extensions:

* Multi-agent workflows
* Conversational memory
* Hybrid search
* Metadata filtering
* Reranking
* Source citations
* Web search integration
* Multi-modal RAG
* Audio transcription
* GPU acceleration

---

# Common Errors

# Qdrant Connection Error

```text
WinError 10061
```

Fix:

```bash
docker start qdrant
```

---

# Ollama Connection Error

```text
Failed to connect to Ollama
```

Fix:

```bash
ollama serve
```

---

# Missing Embedding Model

Fix:

```bash
ollama pull nomic-embed-text
```

---

# OCR Failure

Install:

* Tesseract OCR
* Poppler

Set:

```env
TESSERACT_CMD=
POPPLER_PATH=
```

---

# Example Workflow Internals

```text
User Query
   ↓
LangGraph
   ↓
retrieve()
   ↓
Qdrant similarity search
   ↓
Retrieved context
   ↓
generate()
   ↓
llama3 generation
   ↓
Final response
```

---

# Author

Varshith Thungapalli

AI/ML Engineer | Data Science | Generative AI | LangChain | LangGraph | Computer Vision | RAG Systems

---

# License

MIT License

```
```
