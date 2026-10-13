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
        self, query: str, top_k: int, request_id: str, filter_path: Optional[str] = None,
        keyword_boost: Optional[float] = None,
    ) -> Dict[str, List[KBRetrievalResult]]:
        """Call KB POST /search/api and return top-10 RRF + top-10 CE sets.

        The KB truncates BOTH ``merged_candidates`` (RRF-ranked by
        ``hybrid_score``) and ``final_results`` (cross-encoder order,
        ``rerank_score``) to the requested ``top_k``. Callers request
        ``top_k=20`` and this method slices each list to its first 10
        entries. RRF items score by ``hybrid_score``, CE items by
        ``rerank_score``.
        ``keyword_boost`` if given is forwarded as ``keyword_boost`` in the JSON body
        (KB ``search.py`` respects per-request override, default 3.0; T2 uses 1.5 to relax keyword bias).
        ``filter_path`` forwarded if not None.
        """
        client = self._get_client()
        request = KBRetrievalRequest(query=query, top_k=top_k)
        payload = request.model_dump()
        if filter_path is not None:
            payload["filter_path"] = filter_path
        if keyword_boost is not None:
            try:
                payload["keyword_boost"] = float(keyword_boost)
            except Exception:
                pass

        headers = {"X-Request-ID": request_id}

        try:
            response = await client.post(
                self.search_url,
                json=payload,
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
                    folder_hierarchy=item.get("folder_hierarchy", []),
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
                    folder_hierarchy=item.get("folder_hierarchy", []),
                ))

            return {"rrf": rrf, "ce": ce, "injected_entities": data.get("injected_entities", [])}

        except httpx.HTTPStatusError as e:
            log.error("KB retrieval failed: %s", e)
            raise
        except httpx.TimeoutException:
            log.error("KB retrieval timeout")
            raise
        except Exception as e:
            log.exception("KB retrieval error: %s", e)
            raise

    async def supplementary_search(self, query: str, threshold: float = 0.4, request_id: str = "") -> List[Dict]:
        """Direct EntityList search as tool-call (not hidden step-8 inject).

        Supplements (5 EntityLists: سهامداران, لیست و عناوین وام ها, etc.)
        are normally injected via search.py step 8 with cosine >0.6 on
        description_embedding. This tool directly queries entity_lists via
        embedding similarity on list_name + description (threshold 0.4, lower
        than 0.6) so list content like "سهامداران" matches even when
        description alone fails. Endpoint: POST /search/supplement (tool).
        Falls back to empty list if KB endpoint unavailable (graceful, G5→DLQ).
        """
        headers = {"X-Request-ID": request_id} if request_id else {}
        # Canonical tool endpoint first (POST /search/supplement with list_name+description embedding, thr 0.4)
        for path in ("/search/supplement", "/entity_lists/search", "/supplementary/search", "/supplemental/search"):
            try:
                client = self._get_client()
                is_search_supplement = path == "/search/supplement"
                # Prefer POST; GET as fallback for legacy
                for method in ("POST", "GET"):
                    if method == "GET" and is_search_supplement:
                        # GET /search/supplement?query=&threshold=
                        try:
                            resp = await client.get(
                                f"{self.base_url}{path}",
                                params={"query": query, "threshold": threshold},
                                headers=headers,
                                timeout=5.0,
                            )
                        except Exception:
                            continue
                    else:
                        try:
                            resp = await client.post(
                                f"{self.base_url}{path}",
                                json={"query": query, "threshold": threshold},
                                headers=headers,
                                timeout=5.0,
                            )
                        except Exception:
                            continue
                    if resp.status_code == 200:
                        data = resp.json()
                        # normalize shapes: {"results": [...] } or list
                        if isinstance(data, list):
                            return data
                        if isinstance(data, dict):
                            for key in ("results", "injected_entities", "entity_lists", "data"):
                                if key in data and isinstance(data[key], list):
                                    return data[key]
                            # single entity
                            if "list_name" in data:
                                return [data]
                        return []
                    # 404/405 → try next path/method
                    if resp.status_code in (404, 405, 422):
                        continue
                # try next path
            except Exception:
                continue
        # Fallback: no endpoint — caller will rely on normal injected_entities
        # from /search/api. Return empty so threshold guard can decide (G5→DLQ).
        log.debug("supplementary_search no endpoint for query %r -> empty", query[:80])
        return []

    async def health_check(self) -> bool:
        """Check if KB service is healthy."""
        client = self._get_client()
        try:
            response = await client.get(f"{self.base_url}/health", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False