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

    # Server — bind host configurable for Docker (0.0.0.0) vs local (127.0.0.1)
    orchestrator_host: str = "0.0.0.0"
    orchestrator_port: int = 8100

    # Dependencies
    kb_base_url: str = "http://127.0.0.1:8000"
    guardrails_base_url: str = "http://127.0.0.1:8200"

    # Timeouts
    request_timeout_seconds: int = 120

    # Retrieval
    retrieval_top_k: int = 20

    subspace_threshold: float = 0.8
    subspace_general_exempt: bool = True
    subspace_map: dict = {"cheque": "گزارش اعتباری چک", "facilities_person": "اشخاص حقیقی/اعتبارسنجی تسهیلات", "facilities_legal": "اشخاص حقوقی/اعتبارسنجی تسهیلات", "general": "پایگاه دانش عمومی"}

    # Response detail: "production" hides the ground-truth retrieval block
    # (source labels, titles, scores) from user-visible text; "verbose" appends
    # it for debugging. Citations metadata + traces always keep full detail.
    response_detail: str = "production"

    # Evidence threshold for hallucination guard (G5 fix):
    # if top retrieved rerank/hybrid score < this, treat as no evidence and refuse
    # instead of synthesizing "یکی از سطوح...". Tunable via EVIDENCE_MIN_SCORE env.
    evidence_min_score: float = 0.15

    # T3: Per-subspace evidence thresholds (env overrides: EVIDENCE_MIN_SCORE_CHEQUE etc.)
    evidence_min_score_cheque: float = 0.05
    evidence_min_score_facilities_person: float = 0.10
    evidence_min_score_facilities_legal: float = 0.15
    evidence_min_score_general: float = 0.15

    def get_evidence_threshold(self, applied_subspace: str | None) -> float:
        """Return evidence threshold for the given subspace (T3).

        Maps both canonical labels (cheque, facilities_person, facilities_legal,
        general) and Persian filter_path substrings (چک، حقیقی، حقوقی، عمومی)
        to per-subspace thresholds.  Falls back to global evidence_min_score
        for None/unknown subspaces.  Env overrides via EVIDENCE_MIN_SCORE_*
        are handled by pydantic field defaults.
        """
        s = (applied_subspace or "").strip()
        if not s:
            return float(self.evidence_min_score)
        low = s.lower()
        # Persian filter_path contains these substrings
        if "چک" in s or low == "cheque":
            return float(self.evidence_min_score_cheque)
        if "حقوقی" in s or low == "facilities_legal":
            return float(self.evidence_min_score_facilities_legal)
        if "حقیقی" in s or low == "facilities_person":
            return float(self.evidence_min_score_facilities_person)
        if "عمومی" in s or "پایگاه دانش عمومی" in s or low == "general":
            return float(self.evidence_min_score_general)
        # English label fallback (exact match)
        if low == "cheque":
            return float(self.evidence_min_score_cheque)
        if low == "facilities_person":
            return float(self.evidence_min_score_facilities_person)
        if low == "facilities_legal":
            return float(self.evidence_min_score_facilities_legal)
        if low == "general":
            return float(self.evidence_min_score_general)
        return float(self.evidence_min_score)

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