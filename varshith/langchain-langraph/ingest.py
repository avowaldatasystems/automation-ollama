"""Load documents, chunk, embed with Ollama, and upsert into Qdrant."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import httpx
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import ResponseError

from sample_pdf import build_minimal_pdf_bytes

load_dotenv()

# Use DATA_PATHS to ingest multiple PDFs (separated by ';' or ',').
# Backward compatible: DATA_PATH is still supported.
DATA_PATH = Path(os.getenv("DATA_PATH", "data/sample.pdf"))
DATA_PATHS_RAW = os.getenv("DATA_PATHS", "")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_collection")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
ENABLE_OCR = os.getenv("ENABLE_OCR", "0").strip().lower() in {"1", "true", "yes", "y"}
POPPLER_PATH = os.getenv("POPPLER_PATH")
TESSERACT_CMD = os.getenv("TESSERACT_CMD")


def _ocr_pdf_to_documents(pdf_path: Path) -> list[Document]:
    """OCR a scanned PDF into text documents (requires poppler + tesseract)."""
    try:
        from pdf2image import convert_from_path  # type: ignore
        import pytesseract  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "OCR dependencies missing. Install: pip install pdf2image pytesseract pillow"
        ) from exc

    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    images = convert_from_path(str(pdf_path), poppler_path=POPPLER_PATH)
    out: list[Document] = []
    for idx, img in enumerate(images, start=1):
        text = (pytesseract.image_to_string(img) or "").strip()
        if text:
            out.append(
                Document(
                    page_content=text,
                    metadata={"source": str(pdf_path), "page": idx, "ocr": True},
                )
            )
    return out


def _parse_paths() -> list[Path]:
    if DATA_PATHS_RAW.strip():
        parts = [p.strip().strip("\"'") for p in DATA_PATHS_RAW.replace(",", ";").split(";")]
        paths = [Path(p) for p in parts if p]
    else:
        paths = [DATA_PATH]
    # De-dupe while preserving order
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def ensure_sample_pdf(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        "Sample knowledge base for RAG. "
        "This document explains that LangGraph orchestrates retrieve-then-generate flows, "
        "Qdrant stores vectors for similarity search, and Ollama runs local LLMs and embeddings offline."
    )
    path.write_bytes(build_minimal_pdf_bytes(text))


def main() -> None:
    paths = _parse_paths()

    # Only auto-create the sample PDF if it's one of the requested inputs.
    for p in paths:
        if str(p).replace("\\", "/").endswith("data/sample.pdf"):
            ensure_sample_pdf(p)

    docs = []
    for p in paths:
        if not p.exists():
            print(f"Missing file: {p}", file=sys.stderr)
            raise SystemExit(1)
        loader = PyPDFLoader(str(p))
        loaded = loader.load()
        extracted_chars = sum(len((d.page_content or "").strip()) for d in loaded)
        if extracted_chars < 200:
            print(
                f"Warning: extracted very little text from {p} (chars={extracted_chars}). "
                "If this PDF is scanned/image-only, you will need OCR to use it for RAG.",
                file=sys.stderr,
            )
            if ENABLE_OCR:
                try:
                    ocr_docs = _ocr_pdf_to_documents(p)
                except Exception as exc:
                    print(f"OCR failed for {p}: {exc}", file=sys.stderr)
                    ocr_docs = []
                if ocr_docs:
                    print(
                        f"OCR extracted {sum(len(d.page_content) for d in ocr_docs)} chars from {p}.",
                        file=sys.stderr,
                    )
                    loaded = ocr_docs
        docs.extend(loaded)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embed_kw: dict = {"model": EMBED_MODEL}
    if os.getenv("OLLAMA_BASE_URL"):
        embed_kw["base_url"] = os.getenv("OLLAMA_BASE_URL")

    try:
        embeddings = OllamaEmbeddings(**embed_kw)
        QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            url=QDRANT_URL,
            collection_name=COLLECTION,
        )
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        print(
            f"Cannot connect to Qdrant at {QDRANT_URL}. "
            "Start Qdrant first, then run this script again.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    except ResponseError as exc:
        print(
            f"Ollama embedding error: {exc}\n"
            "Pull the embedding model: ollama pull nomic-embed-text",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    print("Documents embedded successfully!")


if __name__ == "__main__":
    main()
