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
    subspace_keyword_boost: dict = {"cheque": 3.5, "facilities_person": 3.0, "facilities_legal": 3.0, "general": 2.5}
    subspace_map: dict = {"cheque": "گزارش اعتبارسنجی چک", "facilities_person": "اشخاص حقیقی/اعتبارسنجی تسهیلات", "facilities_legal": "اشخاص حقوقی/اعتبارسنجی تسهیلات", "general": "پایگاه دانش عمومی"}

    # Embedding subspace ambiguity score (see docs/SUBSPACE_AMBIGUITY.md).
    # `enabled` gates compute + audit (stage 1); `route` gates actual
    # routing/clarification from the score and defaults False.
    subspace_ambiguity_enabled: bool = True
    subspace_ambiguity_route: bool = False
    subspace_protos_path: str = "data/subspace_protos.npz"
    subspace_clf_path: str = "data/subspace_clf.joblib"
    subspace_ambiguity_margin: float = 0.08
    subspace_ambiguity_min_top1: float = 0.30
    subspace_ambiguity_entropy_max: float = 0.85
    subspace_softmax_temp: float = 0.05
    subspace_score_cache_size: int = 512
    embed_base_url: str = "http://127.0.0.1:8003"
    embed_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # Multi-prototype subspace scoring (see docs/SUBSPACE_AMBIGUITY.md).
    subspace_proto_k: int = 4
    subspace_proto_weight: float = 0.7
    subspace_intra_spread_thr: float = 0.15
    subspace_proto_coverage_thr: float = 0.3
    subspace_ambiguity_weights: dict = {
        "margin": 0.4,
        "entropy": 0.3,
        "top1": 0.2,
        "intra_spread": 0.1,
    }

    # D: Per-subspace BM25 keyword_boost passed to the KB on scoped (filtered)
    # retrieval calls. Cheque docs are Q&A pairs where exact keywords
    # (چک، برگشتی، صیاد) dominate relevance, so that subspace gets the
    # strongest keyword bias (3.5); facilities Q&A pairs sit at 3.0; the
    # general subspace is narrative prose and needs less keyword bias (2.5)
    # so the dense/semantic leg contributes more. Whole-KB (unfiltered)
    # calls keep the KB server default by passing keyword_boost=None.
    subspace_keyword_boost: dict = {"cheque": 3.5, "facilities_person": 3.0, "facilities_legal": 3.0, "general": 2.5}

    # B: Sticky decay — max consecutive generic user turns (no subspace
    # keyword, or long) allowed before the sticky subspace is considered
    # stale and dropped (env STICKY_DECAY_TURNS). Above this many generic
    # turns in a row, the query re-enters the normal classify/infer path.
    sticky_decay_turns: int = 3

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