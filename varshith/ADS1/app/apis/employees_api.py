from fastapi import FastAPI

from app.routers.employees import router as employees_router

app = FastAPI(title="Employees API", version="1.0.0")
app.include_router(employees_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "employees"}
