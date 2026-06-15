from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app import schemas
from app.database import get_db
from app.graph import ask_office_rag

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=schemas.ChatResponse)
def chat(payload: schemas.ChatRequest, db: Session = Depends(get_db)):
    result = ask_office_rag(db, payload.question)
    return schemas.ChatResponse(
        route=result.get("route", "both"),
        answer=result.get("answer", ""),
        sql_context=result.get("sql_context", []),
        vector_sources=result.get("vector_context", []),
    )
