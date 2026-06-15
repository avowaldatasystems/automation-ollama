from fastapi import FastAPI

from app.routers.retrieval import router as retrieval_router

app = FastAPI(title="Retrieval API", version="1.0.0")
app.include_router(retrieval_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "retrieval"}
