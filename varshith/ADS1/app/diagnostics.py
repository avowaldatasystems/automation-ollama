from typing import Any

import requests
from qdrant_client import QdrantClient
from sqlalchemy import create_engine, text

from app.config import get_settings


def check_mysql() -> dict[str, Any]:
    settings = get_settings()
    try:
        engine = create_engine(settings.mysql_server_url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"ok": True, "message": "MySQL connection is working."}
    except Exception as exc:
        return {
            "ok": False,
            "message": (
                "MySQL is not reachable. Install MySQL Server locally (no Docker), "
                "start the MySQL80 Windows service, and check MYSQL_USER/MYSQL_PASSWORD "
                f"in .env. Details: {exc}"
            ),
        }


def check_qdrant() -> dict[str, Any]:
    settings = get_settings()
    try:
        client = QdrantClient(url=settings.qdrant_url)
        client.get_collections()
        return {"ok": True, "message": "Qdrant connection is working."}
    except Exception as exc:
        return {
            "ok": False,
            "message": (
                f"Qdrant is not reachable at {settings.qdrant_url}. "
                "Install qdrant.exe locally and run scripts/start_qdrant.cmd. "
                f"Details: {exc}"
            ),
        }


def check_ollama() -> dict[str, Any]:
    settings = get_settings()
    try:
        response = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        response.raise_for_status()
        models = [item.get("name", "") for item in response.json().get("models", [])]
        missing = [
            model
            for model in (settings.ollama_chat_model, settings.ollama_embed_model)
            if not any(name.startswith(model) for name in models)
        ]
        if missing:
            return {
                "ok": False,
                "message": "Ollama is running, but missing models: " + ", ".join(missing),
                "installed_models": models,
            }
        return {"ok": True, "message": "Ollama connection and models are working.", "installed_models": models}
    except Exception as exc:
        return {
            "ok": False,
            "message": (
                f"Ollama is not reachable at {settings.ollama_base_url}. "
                "Install Ollama, start it, then run: ollama pull llama3.1 "
                "and ollama pull nomic-embed-text. "
                f"Details: {exc}"
            ),
        }


def run_all_checks() -> dict[str, Any]:
    checks = {
        "mysql": check_mysql(),
        "qdrant": check_qdrant(),
        "ollama": check_ollama(),
    }
    checks["ready"] = all(item["ok"] for item in checks.values())
    return checks
