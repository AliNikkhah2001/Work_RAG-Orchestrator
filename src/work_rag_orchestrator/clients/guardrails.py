"""Guardrails client for Orchestrator."""

from __future__ import annotations

import httpx
import logging
from typing import Optional

from ..config import get_settings
from ..schemas import (
    RailCheckRequest,
    RailCheckResponse,
    GuardrailsChatRequest,
    GuardrailsChatResponse,
)

log = logging.getLogger(__name__)


class GuardrailsClient:
    """Client for communicating with Guardrails service."""

    def __init__(self, timeout: Optional[float] = None):
        settings = get_settings()
        self.base_url = settings.guardrails_base_url.rstrip("/")
        self.rails_check_url = settings.guardrails_rails_check_url
        self.chat_url = settings.guardrails_chat_url
        self.timeout = timeout or settings.request_timeout_seconds
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "GuardrailsClient":
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                self.timeout,
                connect=10.0,
                read=self.timeout,
                write=self.timeout,
                pool=10.0,
            ),
            trust_env=False,  # Critical: bypass proxy for localhost
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    self.timeout,
                    connect=10.0,
                    read=self.timeout,
                    write=self.timeout,
                    pool=10.0,
                ),
                trust_env=False,
            )
        return self._client

    async def check_rails(
        self, stage: str, text: str, request_id: str
    ) -> RailCheckResponse:
        """Call Guardrails /v1/rails/check endpoint."""
        client = self._get_client()
        request = RailCheckRequest(stage=stage, text=text, request_id=request_id)

        try:
            response = await client.post(
                self.rails_check_url,
                json=request.model_dump(),
            )
            response.raise_for_status()
            return RailCheckResponse(**response.json())
        except httpx.HTTPStatusError as e:
            log.error("Guardrails rails check failed: %s", e)
            # Fail closed on policy engine errors
            return RailCheckResponse(
                allowed=False,
                action="refuse",
                categories=["engine_error"],
                reason=f"Guardrails service error: {e.response.status_code}",
                policy_version="mvp-1",
                request_id=request_id,
            )
        except httpx.TimeoutException:
            log.error("Guardrails rails check timeout")
            return RailCheckResponse(
                allowed=False,
                action="refuse",
                categories=["timeout"],
                reason="Guardrails service timeout",
                policy_version="mvp-1",
                request_id=request_id,
            )
        except Exception as e:
            log.exception("Guardrails rails check error: %s", e)
            return RailCheckResponse(
                allowed=False,
                action="refuse",
                categories=["engine_error"],
                reason=f"Guardrails service error: {str(e)}",
                policy_version="mvp-1",
                request_id=request_id,
            )

    async def chat_completion(
        self, request: GuardrailsChatRequest, request_id: str
    ) -> GuardrailsChatResponse:
        """Call Guardrails /v1/chat/completions endpoint."""
        client = self._get_client()

        # Add request_id to headers for tracing
        headers = {"X-Request-ID": request_id}

        try:
            response = await client.post(
                self.chat_url,
                json=request.model_dump(exclude_none=True),
                headers=headers,
            )
            response.raise_for_status()
            return GuardrailsChatResponse(**response.json())
        except httpx.HTTPStatusError as e:
            log.error("Guardrails chat completion failed: %s", e)
            raise
        except httpx.TimeoutException:
            log.error("Guardrails chat completion timeout")
            raise
        except Exception as e:
            log.exception("Guardrails chat completion error: %s", e)
            raise

    async def health_check(self) -> bool:
        """Check if Guardrails service is healthy."""
        client = self._get_client()
        try:
            response = await client.get(f"{self.base_url}/health", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    async def readiness_check(self) -> dict:
        """Check if Guardrails service is ready."""
        client = self._get_client()
        try:
            response = await client.get(f"{self.base_url}/ready", timeout=5.0)
            if response.status_code == 200:
                return response.json()
            return {"status": "not_ready", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "not_ready", "error": str(e)}