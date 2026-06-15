from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "employee_management_system"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "office_knowledge"

    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.1"
    ollama_embed_model: str = "nomic-embed-text"

    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "employee-office-rag"

    upload_dir: Path = Path("data/uploads")

    api_health_url: str = "http://127.0.0.1:8000"
    api_employees_url: str = "http://127.0.0.1:8001"
    api_ingestion_url: str = "http://127.0.0.1:8002"
    api_retrieval_url: str = "http://127.0.0.1:8003"
    api_chat_url: str = "http://127.0.0.1:8004"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def api_base_url(self) -> str:
        """Backward-compatible alias for the health API URL."""
        return self.api_health_url

    @property
    def mysql_url(self) -> str:
        password_part = f":{self.mysql_password}" if self.mysql_password else ""
        return (
            f"mysql+pymysql://{self.mysql_user}{password_part}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )

    @property
    def mysql_server_url(self) -> str:
        password_part = f":{self.mysql_password}" if self.mysql_password else ""
        return (
            f"mysql+pymysql://{self.mysql_user}{password_part}"
            f"@{self.mysql_host}:{self.mysql_port}"
        )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    return settings
