"""Unit tests for progressive scope-broaden helpers (no network)."""

from __future__ import annotations

from work_rag_orchestrator.subspace_routing import (
    filter_path_broaden_ladder,
    infer_subspace_from_chunks,
    subspace_topic_label,
)


def test_ladder_leaf_to_parent_to_corpus():
    ladder = filter_path_broaden_ladder("اشخاص حقیقی/اعتبارسنجی تسهیلات")
    assert ladder == ["اشخاص حقیقی", None]


def test_ladder_cheque_single_segment():
    ladder = filter_path_broaden_ladder("اعتبارسنجی چک")
    assert ladder == [None]


def test_ladder_empty_is_whole_corpus():
    assert filter_path_broaden_ladder(None) == [None]
    assert filter_path_broaden_ladder("") == [None]


def test_infer_subspace_majority_cheque():
    chunks = [
        {"folder_hierarchy": ["اعتبارسنجی چک", "رتبه"]},
        {"folder_hierarchy": ["اعتبارسنجی چک"]},
        {"folder_hierarchy": ["اشخاص حقیقی", "اعتبارسنجی تسهیلات"]},
    ]
    assert infer_subspace_from_chunks(chunks) == "cheque"


def test_infer_subspace_inconclusive():
    chunks = [
        {"folder_hierarchy": ["اعتبارسنجی چک"]},
        {"folder_hierarchy": ["اشخاص حقیقی", "اعتبارسنجی تسهیلات"]},
    ]
    assert infer_subspace_from_chunks(chunks) is None


def test_topic_label_known_and_none():
    assert "چک" in subspace_topic_label("cheque")
    assert "عمومی" in subspace_topic_label(None)


import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


def test_progressive_broaden_adopts_parent_then_updates_sticky():
    """Weak leaf → parent hit updates sticky/applied and audit flags."""

    async def _run():
        state = {
            "retrieved_chunks": [],
            "query": "رتبه آبی یعنی چه؟",
            "applied_subspace": "facilities_person",
            "sticky_subspace": "facilities_person",
        }
        parent_chunks = [
            {
                "chunk_id": "c1",
                "content": "رتبه آبی در گزارش چک",
                "score": 0.9,
                "folder_hierarchy": ["اعتبارسنجی چک", "رتبه"],
                "hybrid_score": 0.9,
                "rerank_score": 0.9,
            },
            {
                "chunk_id": "c2",
                "content": "گروه A آبی",
                "score": 0.85,
                "folder_hierarchy": ["اعتبارسنجی چک"],
                "hybrid_score": 0.85,
                "rerank_score": 0.85,
            },
        ]

        class _FakeKB:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def retrieve(self, q, top_k, request_id, filter_path=None):
                # Parent "اشخاص حقیقی" still empty; whole corpus hits
                if filter_path is None:
                    return {"rrf": parent_chunks, "ce": parent_chunks, "injected_entities": []}
                return {"rrf": [], "ce": [], "injected_entities": []}

        with patch(
            "work_rag_orchestrator.clients.knowledgebase.KnowledgebaseClient",
            _FakeKB,
        ), patch(
            "work_rag_orchestrator.nodes.retrieve._get_ev_threshold",
            return_value=0.3,
        ), patch(
            "work_rag_orchestrator.nodes.retrieve.get_settings",
        ) as gs:
            gs.return_value.subspace_map = {
                "cheque": "اعتبارسنجی چک",
                "facilities_person": "اشخاص حقیقی/اعتبارسنجی تسهیلات",
            }
            from work_rag_orchestrator.nodes.retrieve import _progressive_scope_broaden

            ok = await _progressive_scope_broaden(
                state,
                "رتبه آبی یعنی چه؟",
                "req-test",
                active_filter="اشخاص حقیقی/اعتبارسنجی تسهیلات",
                origin_subspace="facilities_person",
            )
        assert ok is True
        assert state.get("scope_broadened") is True
        assert state.get("scope_broadened_from") == "facilities_person"
        assert state.get("scope_broadened_to") == "cheque"
        assert state.get("applied_subspace") == "cheque"
        assert state.get("sticky_subspace") == "cheque"
        assert len(state.get("retrieved_chunks") or []) >= 1

    asyncio.run(_run())


def test_maybe_record_scope_broadened_on_subspace_shift():
    from work_rag_orchestrator.nodes.retrieve import (
        _maybe_record_scope_broadened_from_adopted_chunks,
    )

    state = {
        "applied_subspace": "facilities_person",
        "sticky_subspace": "facilities_person",
        "retrieved_chunks": [
            {"folder_hierarchy": ["اعتبارسنجی چک", "رتبه"]},
            {"folder_hierarchy": ["اعتبارسنجی چک"]},
        ],
    }
    _maybe_record_scope_broadened_from_adopted_chunks(state, scope_filter=None)
    assert state.get("scope_broadened") is True
    assert state.get("scope_broadened_from") == "facilities_person"
    assert state.get("scope_broadened_to") == "cheque"
    assert state.get("scope_broadened_filter") is None

    same_topic = {
        "applied_subspace": "cheque",
        "retrieved_chunks": [{"folder_hierarchy": ["اعتبارسنجی چک"]}],
    }
    _maybe_record_scope_broadened_from_adopted_chunks(same_topic, scope_filter=None)
    assert same_topic.get("scope_broadened") is not True


def test_t1_whole_kb_fallback_sets_scope_broadened_when_topic_shifts():
    """T1 fallback_used path must set scope_broadened for build_context disclosure."""

    async def _run():
        cheque_chunks = [
            {
                "chunk_id": "c1",
                "content": "رتبه چک",
                "score": 0.92,
                "folder_hierarchy": ["اعتبارسنجی چک", "رتبه"],
                "hybrid_score": 0.92,
                "rerank_score": 0.92,
            },
            {
                "chunk_id": "c2",
                "content": "گروه چک",
                "score": 0.88,
                "folder_hierarchy": ["اعتبارسنجی چک"],
                "hybrid_score": 0.88,
                "rerank_score": 0.88,
            },
        ]
        state = {
            "retrieved_chunks": [
                {
                    "chunk_id": "w1",
                    "content": "weak",
                    "score": 0.05,
                    "folder_hierarchy": ["اشخاص حقیقی", "اعتبارسنجی تسهیلات"],
                    "hybrid_score": 0.05,
                }
            ],
            "applied_subspace": "facilities_person",
            "sticky_subspace": "facilities_person",
            "query": "رتبه آبی چک",
        }

        class _FakeKB:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def retrieve(self, q, top_k, request_id, filter_path=None, **kw):
                if filter_path is None:
                    return {"rrf": cheque_chunks, "ce": cheque_chunks, "injected_entities": []}
                return {"rrf": [], "ce": [], "injected_entities": []}

        with patch(
            "work_rag_orchestrator.clients.knowledgebase.KnowledgebaseClient",
            _FakeKB,
        ), patch(
            "work_rag_orchestrator.nodes.retrieve._get_ev_threshold",
            return_value=0.3,
        ):
            from work_rag_orchestrator.nodes.retrieve import _t1_try_whole_kb_fallback

            ok = await _t1_try_whole_kb_fallback(
                state,
                "رتبه آبی چک",
                "req-t1",
                active_filter="اشخاص حقیقی/اعتبارسنجی تسهیلات",
            )
        assert ok is True
        assert state.get("fallback_used") is True
        assert state.get("scope_broadened") is True
        assert state.get("scope_broadened_from") == "facilities_person"
        assert state.get("scope_broadened_to") == "cheque"

    asyncio.run(_run())
