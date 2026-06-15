from fastapi import FastAPI

from app.routers.ingestion import router as ingestion_router

app = FastAPI(title="Ingestion API", version="1.0.0")
app.include_router(ingestion_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ingestion"}
