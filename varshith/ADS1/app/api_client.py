from __future__ import annotations

from typing import Any

import requests
from pydantic import BaseModel

from app.config import get_settings


class ApiError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class OfficeRagClient:
    """Routes each operation to the correct local micro-API."""

    def __init__(self, timeout: float = 300.0):
        settings = get_settings()
        self.timeout = timeout
        self.services: dict[str, str] = {
            "health": settings.api_health_url.rstrip("/"),
            "employees": settings.api_employees_url.rstrip("/"),
            "ingestion": settings.api_ingestion_url.rstrip("/"),
            "retrieval": settings.api_retrieval_url.rstrip("/"),
            "chat": settings.api_chat_url.rstrip("/"),
        }

    @property
    def base_url(self) -> str:
        return self.services["health"]

    def _request(
        self,
        service: str,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | list[Any] | None = None,
        params: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        base = self.services[service]
        url = f"{base}{path}"
        try:
            response = requests.request(
                method,
                url,
                json=json,
                params=params,
                files=files,
                timeout=timeout or self.timeout,
            )
        except requests.ConnectionError as exc:
            raise ApiError(
                f"Cannot reach {service} API at {base}. "
                "Start all APIs with: .\\scripts\\start_apis.cmd"
            ) from exc
        except requests.Timeout as exc:
            raise ApiError(f"API request timed out ({url})") from exc

        if response.ok:
            if not response.content:
                return None
            return response.json()

        detail = response.text
        try:
            body = response.json()
            if isinstance(body, dict) and "detail" in body:
                detail = body["detail"]
                if isinstance(detail, list):
                    detail = "; ".join(str(item) for item in detail)
        except ValueError:
            pass
        raise ApiError(f"[{service}] {detail}", status_code=response.status_code)

    def all_services_ok(self) -> bool:
        statuses = self.health_all()
        return bool(statuses) and all(
            info.get("status") == "ok" for info in statuses.values()
        )

    def health_all(self) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        for name, base in self.services.items():
            try:
                response = requests.get(f"{base}/health", timeout=10)
                if response.ok:
                    results[name] = response.json()
                else:
                    results[name] = {"status": "error", "code": response.status_code}
            except requests.RequestException as exc:
                results[name] = {"status": "offline", "error": str(exc)}
        return results

    def health(self) -> dict[str, str]:
        return self._request("health", "GET", "/health", timeout=10)

    def full_health(self) -> dict[str, Any]:
        return self._request("health", "GET", "/health/full", timeout=60)

    def chat(self, question: str) -> dict[str, Any]:
        return self._request("chat", "POST", "/chat", json={"question": question})

    def list_employees(self) -> list[dict[str, Any]]:
        return self._request("employees", "GET", "/employees", timeout=30)

    def get_employee(self, employee_id: int) -> dict[str, Any]:
        return self._request("employees", "GET", f"/employees/{employee_id}", timeout=30)

    def create_employee(self, payload: BaseModel) -> dict[str, Any]:
        return self._request(
            "employees",
            "POST",
            "/employees",
            json=payload.model_dump(mode="json", exclude_none=True),
            timeout=30,
        )

    def create_office_details(self, payload: BaseModel) -> dict[str, Any]:
        return self._request(
            "employees",
            "POST",
            "/employees/office",
            json=payload.model_dump(mode="json", exclude_none=True),
            timeout=30,
        )

    def create_salary_leave(self, payload: BaseModel) -> dict[str, Any]:
        return self._request(
            "employees",
            "POST",
            "/employees/salary-leave",
            json=payload.model_dump(mode="json", exclude_none=True),
            timeout=30,
        )

    def create_attendance(self, payload: BaseModel) -> dict[str, Any]:
        return self._request(
            "employees",
            "POST",
            "/employees/attendance",
            json=payload.model_dump(mode="json", exclude_none=True),
            timeout=30,
        )

    def create_document(self, payload: BaseModel) -> dict[str, Any]:
        return self._request(
            "employees",
            "POST",
            "/employees/documents",
            json=payload.model_dump(mode="json", exclude_none=True),
            timeout=30,
        )

    def ingest_text(self, title: str, text: str, source_type: str = "manual") -> dict[str, Any]:
        return self._request(
            "ingestion",
            "POST",
            "/knowledge/text",
            json={"title": title, "text": text, "source_type": source_type},
            timeout=120,
        )

    def upload_knowledge_file(self, filename: str, content: bytes) -> dict[str, Any]:
        return self._request(
            "ingestion",
            "POST",
            "/knowledge/upload",
            files={"file": (filename, content)},
            timeout=300,
        )

    def search_knowledge(self, query: str, k: int = 5) -> list[dict[str, str]]:
        return self._request(
            "retrieval",
            "GET",
            "/knowledge/search",
            params={"q": query, "k": k},
            timeout=60,
        )


def get_client() -> OfficeRagClient:
    return OfficeRagClient()
