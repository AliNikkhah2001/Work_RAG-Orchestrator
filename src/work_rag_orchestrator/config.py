"""Configuration for Orchestrator service."""

from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server
    orchestrator_port: int = 8100

    # Dependencies
    kb_base_url: str = "http://127.0.0.1:8000"
    guardrails_base_url: str = "http://127.0.0.1:8200"

    # Timeouts
    request_timeout_seconds: int = 120

    # Retrieval
    retrieval_top_k: int = 5

    # Upstream LLM model (Vast Gemma)
    upstream_llm_model: str = "unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL"

    @property
    def kb_search_url(self) -> str:
        return f"{self.kb_base_url.rstrip('/')}/search/api"

    @property
    def guardrails_rails_check_url(self) -> str:
        return f"{self.guardrails_base_url.rstrip('/')}/v1/rails/check"

    @property
    def guardrails_chat_url(self) -> str:
        return f"{self.guardrails_base_url.rstrip('/')}/v1/chat/completions"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()