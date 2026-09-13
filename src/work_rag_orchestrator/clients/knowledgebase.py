"""Knowledgebase client for Orchestrator."""

from __future__ import annotations

import httpx
import logging
from typing import Optional, List

from ..config import get_settings
from ..schemas import (
    KBRetrievalRequest,
    KBRetrievalResponse,
    KBRetrievalResult,
)

log = logging.getLogger(__name__)


class KnowledgebaseClient:
    """Client for communicating with KB Manager service."""

    def __init__(self, timeout: Optional[float] = None):
        settings = get_settings()
        self.base_url = settings.kb_base_url.rstrip("/")
        self.search_url = settings.kb_search_url
        self.timeout = timeout or settings.request_timeout_seconds
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "KnowledgebaseClient":
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

    async def retrieve(
        self, query: str, top_k: int, request_id: str
    ) -> List[KBRetrievalResult]:
        """Call KB POST /search/api endpoint and normalize results."""
        client = self._get_client()
        request = KBRetrievalRequest(query=query, top_k=top_k)

        headers = {"X-Request-ID": request_id}

        try:
            response = await client.post(
                self.search_url,
                json=request.model_dump(),
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

            # The KB returns SearchSteps with final_results
            # We need to normalize to our KBRetrievalResult format
            final_results = data.get("final_results", [])

            normalized = []
            for item in final_results:
                # KB returns content_preview (300 chars) - we use that as content for MVP
                normalized.append(KBRetrievalResult(
                    chunk_id=item.get("chunk_id", ""),
                    document_id=item.get("doc_id", ""),
                    title=item.get("doc_title", ""),
                    heading=item.get("heading_path", ""),
                    content=item.get("content_preview", ""),
                    score=item.get("rerank_score", item.get("hybrid_score", 0.0)),
                ))

            return normalized

        except httpx.HTTPStatusError as e:
            log.error("KB retrieval failed: %s", e)
            raise
        except httpx.TimeoutException:
            log.error("KB retrieval timeout")
            raise
        except Exception as e:
            log.exception("KB retrieval error: %s", e)
            raise

    async def health_check(self) -> bool:
        """Check if KB service is healthy."""
        client = self._get_client()
        try:
            # KB doesn't have /health, try root
            response = await client.get(f"{self.base_url}/", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False