from fastapi import APIRouter, HTTPException

from app.vector_store import retrieve_documents

router = APIRouter(prefix="/knowledge", tags=["retrieval"])


@router.get("/search")
def search_knowledge(q: str, k: int = 5) -> list[dict[str, str]]:
    try:
        return retrieve_documents(q, k=k)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Knowledge search is unavailable. Start Qdrant and Ollama, "
                "then pull the embedding model configured in .env. "
                f"Details: {exc}"
            ),
        ) from exc
