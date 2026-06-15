from app.routers.chat import router as chat_router
from app.routers.employees import router as employees_router
from app.routers.health import router as health_router
from app.routers.ingestion import router as ingestion_router
from app.routers.retrieval import router as retrieval_router

__all__ = [
    "chat_router",
    "employees_router",
    "health_router",
    "ingestion_router",
    "retrieval_router",
]
