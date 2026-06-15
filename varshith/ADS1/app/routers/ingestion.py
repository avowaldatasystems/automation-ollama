from pathlib import Path
from shutil import copyfileobj
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from app import schemas
from app.config import get_settings
from app.vector_store import ingest_file, ingest_text

router = APIRouter(prefix="/knowledge", tags=["ingestion"])


@router.post("/text")
def add_knowledge_text(payload: schemas.IngestTextRequest) -> dict[str, Any]:
    try:
        chunks = ingest_text(payload.title, payload.text, payload.source_type)
        return {"chunks_added": chunks}
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Knowledge ingestion is unavailable. Start Qdrant and Ollama, "
                "then pull the embedding model configured in .env. "
                f"Details: {exc}"
            ),
        ) from exc


@router.post("/upload")
def upload_knowledge_file(file: UploadFile = File(...)) -> dict[str, Any]:
    settings = get_settings()
    target = settings.upload_dir / file.filename
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as out:
        copyfileobj(file.file, out)

    try:
        chunks = ingest_file(Path(target))
        return {"file": str(target), "chunks_added": chunks}
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Knowledge ingestion is unavailable. Start Qdrant and Ollama, "
                "then pull the embedding model configured in .env. "
                f"Details: {exc}"
            ),
        ) from exc
