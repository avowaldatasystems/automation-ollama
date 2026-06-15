from typing import Any

from fastapi import APIRouter

from app.diagnostics import run_all_checks

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "health"}


@router.get("/health/full")
def full_health() -> dict[str, Any]:
    return run_all_checks()
