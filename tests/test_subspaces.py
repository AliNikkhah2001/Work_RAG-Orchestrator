"""Focused tests for the embedding subspace ambiguity scorer.

Exercises ``work_rag_orchestrator.subspaces`` with synthetic, in-memory data
only: no network, no database, and no reliance on ``data/subspace_protos.npz``.
Also covers the ``config``/``state`` additions from
``docs/SUBSPACE_AMBIGUITY.md``.

The suites/subspaces tuple under test is fixed
``("cheque", "facilities_person", "facilities_legal", "general")``.
"""

from __future__ import annotations

import numpy as np
import pytest

from work_rag_orchestrator.config import Settings, get_settings
from work_rag_orchestrator.graph import should_clarify
from work_rag_orchestrator.state import RAGState
from work_rag_orchestrator.subspaces import (
    SUBSPACES,
    ProtoBundle,
    SubspaceScorer,
    load_protos,
)

# All keys guaranteed by AmbiguityResult.as_dict().
EXPECTED_RESULT_KEYS = {
    "sims",
    "top1",
    "top2",
    "margin",
    "entropy",
    "score",
    "ambiguous",
    "subspace",
    "source",
}


class CountingEmbedder:
    """Fixed embedding callable that records how many times it was called."""

    def __init__(self, vector: list[float]) -> None:
        self.vector = np.asarray(vector, dtype="float32")
        self.calls = 0

    def __call__(self, query: str) -> np.ndarray:
        self.calls += 1
        return self.vector


def make_bundle(centroids: np.ndarray, prototypes: np.ndarray | None = None) -> ProtoBundle:
    """Build a synthetic ProtoBundle (no whitening, no prototypes by default)."""
    arr = np.asarray(centroids, dtype="float32")
    return ProtoBundle(
        subspaces=SUBSPACES,
        centroids=arr,
        prototypes=prototypes,
        whiten_mean=None,
        whiten_matrix=None,
        dim=arr.shape[-1],
        model="test-model",
        kb_version="test-kb",
    )


def scorer(
    settings: Settings,
    centroids: np.ndarray,
    vector: list[float],
    *,
    classifier: dict | None = None,
) -> SubspaceScorer:
    """Scorer with an injected constant embedder (deterministic, offline)."""
    return SubspaceScorer(
        make_bundle(centroids),
        settings,
        embedder=CountingEmbedder(vector),
        classifier=classifier,
    )


@pytest.fixture
def settings() -> Settings:
    """Real Settings carrying the documented defaults."""
    return Settings()


# ---------------------------------------------------------------------------
# 1. load_protos robustness
# ---------------------------------------------------------------------------


def test_load_protos_missing_path_returns_none(tmp_path):
    """A missing artifact path degrades to None instead of raising."""
    missing = tmp_path / "does_not_exist.npz"
    assert load_protos(str(missing)) is None


# ---------------------------------------------------------------------------
# 2. Core scoring behavior
# ---------------------------------------------------------------------------


def test_orthogonal_prototypes_clear_subspace(settings):
    """A query on one centroid yields a confident, unambiguous label."""
    res = scorer(settings, np.eye(4), [1, 0, 0, 0]).score("q")
    assert res.ambiguous is False
    assert res.subspace == "cheque"
    assert res.source == "protos"
    assert res.margin == pytest.approx(1.0)  # large margin
    assert res.score == pytest.approx(0.0)  # no ambiguity


def test_near_tied_prototypes_ambiguous(settings):
    """Two equidistant centroids trip the margin rule and clear the label."""
    res = scorer(settings, np.eye(4), [1, 1, 0, 0]).score("q")
    assert res.ambiguous is True
    assert res.subspace is None
    assert res.margin == pytest.approx(0.0, abs=1e-6)


def test_low_top1_out_of_domain_ambiguous(settings):
    """A weak best match (below min_top1) is treated as out-of-domain."""
    res = scorer(settings, 0.2 * np.eye(4), [1, 0, 0, 0]).score("q")
    assert res.top1 < settings.subspace_ambiguity_min_top1
    assert res.ambiguous is True
    assert res.subspace is None


def test_high_entropy_triggers_ambiguity():
    """A near-uniform score distribution triggers ambiguity via entropy.

    The other two gates are disabled so entropy is the sole trigger.
    """
    settings = Settings(subspace_ambiguity_min_top1=0.0, subspace_ambiguity_margin=0.0)
    res = scorer(settings, np.eye(4), [1.0, 0.99, 0.98, 0.97]).score("q")
    assert res.entropy > settings.subspace_ambiguity_entropy_max
    assert res.ambiguous is True
    assert res.subspace is None


def test_score_bounds_and_monotonic(settings):
    """`score` stays in [0, 1] and grows with ambiguity."""
    clear = scorer(settings, np.eye(4), [1, 0, 0, 0]).score("q")
    moderate = scorer(settings, np.eye(4), [1.0, 0.96, 0, 0]).score("q")
    ambiguous = scorer(settings, np.eye(4), [1, 1, 0, 0]).score("q")

    for res in (clear, moderate, ambiguous):
        assert 0.0 <= res.score <= 1.0

    assert clear.score < moderate.score < ambiguous.score
    assert clear.ambiguous is False
    assert ambiguous.ambiguous is True


def test_as_dict_contains_documented_keys(settings):
    """as_dict() exposes exactly the documented result keys."""
    res = scorer(settings, np.eye(4), [1, 0, 0, 0]).score("q")
    assert set(res.as_dict()) == EXPECTED_RESULT_KEYS


# ---------------------------------------------------------------------------
# 3. `source` semantics
# ---------------------------------------------------------------------------


def test_source_disabled_without_protos(settings):
    """No prototype bundle -> disabled neutral result, defer to other logic."""
    res = SubspaceScorer(None, settings).score("anything")
    assert res.source == "disabled"
    assert res.ambiguous is False
    assert res.subspace is None


def test_source_fallback_when_embedder_raises(settings):
    """An embedding failure yields a neutral fallback result."""

    def boom(query: str) -> np.ndarray:
        raise RuntimeError("embedding service down")

    sc = SubspaceScorer(make_bundle(np.eye(4)), settings, embedder=boom)
    res = sc.score("anything")
    assert res.source == "fallback"
    assert res.ambiguous is False
    assert res.subspace is None


def test_source_protos_normally(settings):
    """The normal path reports source='protos'."""
    assert scorer(settings, np.eye(4), [1, 0, 0, 0]).score("q").source == "protos"


def test_classifier_overrides_decision(settings):
    """A classifier label wins over the prototype argmax."""

    class StubClassifier:
        def __init__(self, label: str) -> None:
            self._label = label

        def predict(self, X):
            return [self._label]

    clf = {"clf": StubClassifier("general"), "feature_names": list(SUBSPACES)}
    # Protos would pick "cheque"; the classifier overrides to "general".
    res = scorer(settings, np.eye(4), [1, 0, 0, 0], classifier=clf).score("q")
    assert res.source == "classifier"
    assert res.subspace == "general"
    assert res.ambiguous is False


def test_classifier_unknown_label_is_ambiguous(settings):
    """A classifier label outside the subspaces set means ambiguous."""

    class StubClassifier:
        def predict(self, X):
            return ["__unknown__"]

    clf = {"clf": StubClassifier(), "feature_names": list(SUBSPACES)}
    res = scorer(settings, np.eye(4), [1, 0, 0, 0], classifier=clf).score("q")
    assert res.source == "classifier"
    assert res.ambiguous is True
    assert res.subspace is None


# ---------------------------------------------------------------------------
# 4. LRU cache
# ---------------------------------------------------------------------------


def test_lru_cache_normalizes_query_and_embeds_once(settings):
    """Whitespace variants share one cache entry and one embedder call."""
    embedder = CountingEmbedder([1, 0, 0, 0])
    sc = SubspaceScorer(make_bundle(np.eye(4)), settings, embedder=embedder)

    first = sc.score("hello   world")
    second = sc.score("  hello world ")

    assert first == second
    assert embedder.calls == 1


# ---------------------------------------------------------------------------
# 5. Config defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    """Declared Settings defaults match the frozen spec."""
    fields = Settings.model_fields
    assert fields["subspace_ambiguity_enabled"].default is True
    assert fields["subspace_ambiguity_route"].default is False
    assert fields["subspace_ambiguity_margin"].default == 0.08
    assert fields["subspace_ambiguity_min_top1"].default == 0.30
    assert fields["subspace_ambiguity_entropy_max"].default == 0.85
    assert fields["subspace_softmax_temp"].default == 0.05


# ---------------------------------------------------------------------------
# 6. State additions
# ---------------------------------------------------------------------------


def test_rag_state_accepts_ambiguity_keys():
    """RAGState declares the new ambiguity keys and they are usable."""
    state: RAGState = {
        "ambiguity_score": 0.42,
        "ambiguity_margin": 0.01,
        "ambiguity_entropy": 0.9,
        "ambiguity_ambiguous": True,
        "ambiguity_source": "protos",
        "subspace_sims": {"cheque": 0.5, "general": 0.4},
    }
    assert state["ambiguity_score"] == pytest.approx(0.42)
    assert state["ambiguity_ambiguous"] is True
    assert state["subspace_sims"]["cheque"] == pytest.approx(0.5)

    annotated = RAGState.__annotations__
    for key in (
        "ambiguity_score",
        "ambiguity_margin",
        "ambiguity_entropy",
        "ambiguity_ambiguous",
        "ambiguity_source",
        "subspace_sims",
    ):
        assert key in annotated


# ---------------------------------------------------------------------------
# 7. Route gating contract (graph.should_clarify)
# ---------------------------------------------------------------------------


def test_should_clarify_gating_contract(monkeypatch):
    """Ambiguous + no resolved subspace clarifies only when route is enabled."""
    state = {
        "ambiguity_ambiguous": True,
        "applied_subspace": None,
        "needs_clarification": False,
        "query": "",
    }
    try:
        monkeypatch.setenv("SUBSPACE_AMBIGUITY_ROUTE", "true")
        get_settings.cache_clear()
        assert should_clarify(dict(state)) == "clarify"

        monkeypatch.setenv("SUBSPACE_AMBIGUITY_ROUTE", "false")
        get_settings.cache_clear()
        assert should_clarify(dict(state)) == "continue"
    finally:
        # Never leak an env-driven Settings into other tests.
        get_settings.cache_clear()


# ---------------------------------------------------------------------------
# 8. M1 regression: no-whiten artifact must not collapse every query
# ---------------------------------------------------------------------------


def test_no_whiten_artifact_does_not_collapse_scores(tmp_path, settings):
    """A (D, 0) whiten_matrix + zero whiten_mean must be treated as absent.

    Before the fix the zero-column matrix projected the query to a zero-length
    vector, every similarity became 0.0 and the query was flagged ambiguous.
    """
    path = tmp_path / "protos.npz"
    np.savez(
        path,
        subspaces=np.asarray(SUBSPACES),
        centroids=np.eye(4, dtype="float32"),
        prototypes=np.zeros((4, 0, 4), dtype="float32"),
        whiten_mean=np.zeros(4, dtype="float32"),
        whiten_matrix=np.zeros((4, 0), dtype="float32"),
        dim=np.int32(4),
        model=np.asarray("test-model"),
        kb_version=np.asarray("test-kb"),
    )

    bundle = load_protos(str(path))
    assert bundle is not None
    assert bundle.whiten_mean is None
    assert bundle.whiten_matrix is None
    assert bundle.prototypes is None

    res = SubspaceScorer(bundle, settings, embedder=CountingEmbedder([1, 0, 0, 0])).score("q")
    assert res.source == "protos"
    assert res.subspace == "cheque"
    assert any(v != 0.0 for v in res.sims.values())


# ---------------------------------------------------------------------------
# 9. Multi-prototype branch
# ---------------------------------------------------------------------------


def test_multi_prototype_uses_best_prototype(settings):
    """When prototypes are present the max over k prototypes decides the label."""
    centroids = np.zeros((4, 4), dtype="float32")  # useless if (wrongly) used
    prototypes = np.zeros((4, 2, 4), dtype="float32")
    prototypes[0, 0] = [1, 0, 0, 0]
    prototypes[0, 1] = [0, 0, 0, 1]  # second cheque prototype matches the query
    prototypes[1, 0] = [0, 1, 0, 0]
    prototypes[2, 0] = [0, 0, 1, 0]
    prototypes[3, 0] = [0.05, 0, 0, 0]

    sc = SubspaceScorer(
        make_bundle(centroids, prototypes),
        settings,
        embedder=CountingEmbedder([0, 0, 0, 1]),
    )
    res = sc.score("q")
    assert res.source == "protos"
    assert res.subspace == "cheque"
    assert res.sims["cheque"] == pytest.approx(1.0)
    assert res.ambiguous is False


# ---------------------------------------------------------------------------
# 10. Async scoring (ascore) — hot-path contract
# ---------------------------------------------------------------------------


async def test_ascore_disabled_short_circuits_without_embedding():
    """Disabled feature returns 'disabled' and never calls the embedder."""
    settings = Settings(subspace_ambiguity_enabled=False)
    embedder = CountingEmbedder([1, 0, 0, 0])
    sc = SubspaceScorer(make_bundle(np.eye(4)), settings, embedder=embedder)

    res = await sc.ascore("anything")

    assert res.source == "disabled"
    assert res.ambiguous is False
    assert res.subspace is None
    assert embedder.calls == 0


async def test_ascore_matches_sync_score_with_injected_embedder(settings):
    """ascore shares the sync path semantics when an embedder is injected."""
    sc = SubspaceScorer(make_bundle(np.eye(4)), settings, embedder=CountingEmbedder([1, 0, 0, 0]))
    res = await sc.ascore("q")
    assert res.source == "protos"
    assert res.subspace == "cheque"
    assert res.ambiguous is False
