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
from typing import Dict

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
    ) -> Dict[str, List[KBRetrievalResult]]:
        """Call KB POST /search/api and return top-10 RRF + top-10 CE sets.

        The KB truncates BOTH ``merged_candidates`` (RRF-ranked by
        ``hybrid_score``) and ``final_results`` (cross-encoder order,
        ``rerank_score``) to the requested ``top_k``. Callers request
        ``top_k=20`` and this method slices each list to its first 10
        entries. RRF items score by ``hybrid_score``, CE items by
        ``rerank_score``.
        """
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

            # The KB returns SearchSteps with merged_candidates (RRF-ranked by
            # hybrid_score) and final_results (cross-encoder order, rerank_score),
            # BOTH truncated to the requested top_k. Take top 10 of each.
            merged_candidates = data.get("merged_candidates", [])
            final_results = data.get("final_results", [])

            rrf: List[KBRetrievalResult] = []
            for rank, item in enumerate(merged_candidates[:10], start=1):
                hybrid = item.get("hybrid_score", 0.0)
                rrf.append(KBRetrievalResult(
                    chunk_id=item.get("chunk_id", ""),
                    document_id=item.get("doc_id", ""),
                    title=item.get("doc_title", ""),
                    heading=item.get("heading_path", ""),
                    # KB returns content_preview (300 chars) - we use that as content for MVP
                    content=item.get("content_preview", ""),
                    score=hybrid if hybrid is not None else 0.0,
                    source="rrf",
                    rank_rrf=rank,
                    rank_ce=None,
                    hybrid_score=hybrid,
                    rerank_score=item.get("rerank_score"),
                ))

            ce: List[KBRetrievalResult] = []
            for rank, item in enumerate(final_results[:10], start=1):
                rerank = item.get("rerank_score", item.get("hybrid_score", 0.0))
                ce.append(KBRetrievalResult(
                    chunk_id=item.get("chunk_id", ""),
                    document_id=item.get("doc_id", ""),
                    title=item.get("doc_title", ""),
                    heading=item.get("heading_path", ""),
                    # KB returns content_preview (300 chars) - we use that as content for MVP
                    content=item.get("content_preview", ""),
                    score=rerank if rerank is not None else 0.0,
                    source="ce",
                    rank_rrf=None,
                    rank_ce=rank,
                    hybrid_score=item.get("hybrid_score"),
                    rerank_score=item.get("rerank_score"),
                ))

            return {"rrf": rrf, "ce": ce}

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