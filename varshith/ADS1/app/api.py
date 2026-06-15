"""Optional all-in-one gateway (single port). Prefer scripts/start_apis.cmd for micro-APIs."""

from fastapi import FastAPI

from app.routers.chat import router as chat_router
from app.routers.employees import router as employees_router
from app.routers.health import router as health_router
from app.routers.ingestion import router as ingestion_router
from app.routers.retrieval import router as retrieval_router

app = FastAPI(title="Employee Office RAG Gateway", version="1.0.0")
app.include_router(health_router)
app.include_router(employees_router)
app.include_router(ingestion_router)
app.include_router(retrieval_router)
app.include_router(chat_router)
