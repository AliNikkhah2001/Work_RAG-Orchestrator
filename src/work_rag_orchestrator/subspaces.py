"""Embedding subspace ambiguity scoring.

Implements the frozen runtime interface from
``docs/SUBSPACE_AMBIGUITY.md``: encode the query once, score its cosine
similarity against the four knowledge-subspace prototypes/centroids, and turn
the resulting distribution into a calibrated ambiguity signal.

Behavior is gated by ``subspace_ambiguity_enabled`` (compute + audit) and,
for routing, by ``subspace_ambiguity_route`` (owned by the integration layer).
``available`` is ``False`` whenever no prototype bundle could be loaded, so the
caller can safely fall back to the existing keyword logic.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

log = logging.getLogger(__name__)

# Canonical knowledge subspaces (also the softmax support size S).
SUBSPACES: tuple[str, ...] = ("cheque", "facilities_person", "facilities_legal", "general")

# Timeout for the HTTP embedding fallback path (seconds).
_EMBED_TIMEOUT_S = 10.0

# Repository root (the `components/orchestrator` dir). Artifact paths are
# configured relative to it but may also be passed relative to the process CWD.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Process-lifetime shared async HTTP client for the embedding fallback. Reused
# across requests (never created per call) and proxy-bypassed via trust_env=False
# so localhost embedding calls never hang behind the corporate Squid proxy.
_ASYNC_EMBED_CLIENT: Any = None


def _get_async_client() -> "Any":
    """Return the shared async HTTP client, creating it on first use.

    A single event loop serves the API, so the check/create is race-free without
    a lock. Closed only when the process exits (no per-request teardown).
    """
    global _ASYNC_EMBED_CLIENT
    if _ASYNC_EMBED_CLIENT is None:
        import httpx

        _ASYNC_EMBED_CLIENT = httpx.AsyncClient(timeout=_EMBED_TIMEOUT_S, trust_env=False)
    return _ASYNC_EMBED_CLIENT


@dataclass(frozen=True)
class AmbiguityResult:
    """Calibrated ambiguity signal for one query."""

    sims: dict[str, float]  # subspace -> cosine similarity (best prototype)
    all_sims: dict[str, list[float]]  # subspace -> list of prototype similarities
    top1: float  # max sim (best prototype)
    top2: float  # second max sim
    margin: float  # top1 - top2
    entropy: float  # normalized softmax entropy in [0, 1]
    score: float  # ambiguity in [0, 1]; higher = more ambiguous
    ambiguous: bool  # decision
    subspace: str | None  # argmax when not ambiguous, else None
    source: str  # "protos" | "classifier" | "fallback" | "disabled"
    intra_spread: dict[str, float]  # subspace -> std over prototypes
    proto_coverage: dict[str, float]  # subspace -> fraction of protos > threshold

    def as_dict(self) -> dict:
        """Serialize to a plain dict for state/audit emission."""
        return {
            "sims": self.sims,
            "all_sims": self.all_sims,
            "top1": self.top1,
            "top2": self.top2,
            "margin": self.margin,
            "entropy": self.entropy,
            "score": self.score,
            "ambiguous": self.ambiguous,
            "subspace": self.subspace,
            "source": self.source,
            "intra_spread": self.intra_spread,
            "proto_coverage": self.proto_coverage,
        }


@dataclass
class ProtoBundle:
    """Loaded prototype/centroid artifact for the subspace classifier."""

    subspaces: tuple[str, ...]
    centroids: "np.ndarray"  # (S, D) L2-normalized
    prototypes: "np.ndarray | None"  # (S, K, D) optional multi-prototype
    whiten_mean: "np.ndarray | None"  # (D,)
    whiten_matrix: "np.ndarray | None"  # (D, r)
    dim: int
    model: str
    kb_version: str


def _resolve_artifact_path(path: str) -> str:
    """Resolve an artifact path as-is first, then relative to the repo root.

    Config defaults (``data/subspace_protos.npz``) are relative to
    ``components/orchestrator``; resolving against CWD alone silently disables
    the scorer when the process starts elsewhere. Relative paths that do not
    exist as-is are retried under the repo root; the original string is
    returned when neither exists (``load_protos`` logs and degrades).
    """
    p = Path(path)
    if p.exists():
        return str(p)
    candidate = _REPO_ROOT / path
    if candidate.exists():
        return str(candidate)
    return str(p)


def load_protos(path: str) -> Optional[ProtoBundle]:
    """Load a :class:`ProtoBundle` from an ``.npz`` artifact.

    Returns ``None`` (never raises) when the file is missing or malformed, so
    the scorer degrades to the disabled path instead of breaking callers.
    """
    try:
        import numpy as np

        # All keys are numeric or unicode arrays (no object/pickle payloads).
        with np.load(path, allow_pickle=False) as data:
            subspaces = tuple(str(s) for s in data["subspaces"].tolist())
            centroids = np.asarray(data["centroids"], dtype="float32")
            prototypes = None
            if "prototypes" in data:
                prototypes = np.asarray(data["prototypes"], dtype="float32")
                # Tolerate a flattened (S*K, D) layout by reshaping per subspace.
                if prototypes.ndim == 2 and centroids.ndim == 2:
                    s, d = centroids.shape
                    if s and prototypes.shape[0] % s == 0:
                        prototypes = prototypes.reshape(s, prototypes.shape[0] // s, d)
                # A zero-size block means "no multi-prototypes"; use centroids.
                if prototypes.size == 0:
                    prototypes = None
            # Treat zero-size whiten_mean/whiten_matrix as absent, mirroring the
            # offline loader's `.size` guard. Without it a (D, 0) matrix
            # projects every query to a zero-length vector, collapsing all
            # similarities to 0.0 and flagging every query ambiguous (M1).
            whiten_mean = None
            if "whiten_mean" in data:
                wm = np.asarray(data["whiten_mean"], dtype="float32")
                whiten_mean = wm if wm.size else None
            whiten_matrix = None
            if "whiten_matrix" in data:
                wmat = np.asarray(data["whiten_matrix"], dtype="float32")
                whiten_matrix = wmat if wmat.size else None
            # Whitening needs both halves; a half-present bundle is treated as
            # no-whitening so scoring stays geometrically valid.
            if whiten_mean is None or whiten_matrix is None:
                whiten_mean = None
                whiten_matrix = None
            dim = int(data["dim"]) if "dim" in data else int(centroids.shape[-1])
            model = str(data["model"]) if "model" in data else ""
            kb_version = str(data["kb_version"]) if "kb_version" in data else ""
        return ProtoBundle(
            subspaces, centroids, prototypes, whiten_mean, whiten_matrix, dim, model, kb_version
        )
    except Exception as e:  # missing file, bad keys, dtype errors — never raise
        log.warning("subspace protos unavailable at %s (%s); scorer disabled", path, e)
        return None


class SubspaceScorer:
    """Scores queries against subspace prototypes with optional whitening.

    ``embedder`` may be injected as a ``query -> np.ndarray`` callable (used by
    tests/offline tooling); otherwise embeddings are fetched over HTTP from the
    configured embedding service. Results are LRU-cached by normalized query
    text when the embedding is not supplied by the caller.
    """

    def __init__(
        self,
        protos: Optional[ProtoBundle],
        settings: Any,
        embedder: Optional[Callable[[str], Any]] = None,
        classifier: Optional[dict] = None,
    ) -> None:
        self._protos = protos
        self._settings = settings
        self._embedder = embedder
        self._classifier = classifier
        self._cache: "OrderedDict[str, AmbiguityResult]" = OrderedDict()
        try:
            self._cache_size = max(1, int(getattr(settings, "subspace_score_cache_size", 512)))
        except Exception:
            self._cache_size = 512

    @classmethod
    def from_settings(cls, settings: Any) -> "SubspaceScorer":
        """Build a scorer from :class:`Settings`, loading artifacts if enabled."""
        enabled = bool(getattr(settings, "subspace_ambiguity_enabled", True))
        protos: Optional[ProtoBundle] = None
        classifier: Optional[dict] = None
        if enabled:
            protos = load_protos(
                _resolve_artifact_path(
                    getattr(settings, "subspace_protos_path", "data/subspace_protos.npz")
                )
            )
            classifier = _load_classifier(
                _resolve_artifact_path(
                    getattr(settings, "subspace_clf_path", "data/subspace_clf.joblib")
                )
            )
        return cls(protos, settings, classifier=classifier)

    @property
    def available(self) -> bool:
        """True when a prototype bundle is loaded and scoring can run."""
        return self._protos is not None

    def score(self, query: str, query_embedding: "np.ndarray | None" = None) -> AmbiguityResult:
        """Compute the ambiguity signal for ``query``.

        ``query_embedding`` short-circuits embedding (and caching); otherwise a
        normalized-text LRU cache is consulted. Embedding failures yield a
        ``source="fallback"`` result rather than raising.
        """
        if self._protos is None:
            return self._disabled_result()

        key: Optional[str] = None
        if query_embedding is None:
            key = self._normalize(query)
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached

        try:
            vec = self._embedding(query, query_embedding)
        except Exception as e:
            log.warning("subspace embedding failed (%s); falling back", e)
            return self._fallback_result()

        result = self._score_vector(vec)
        if key is not None and result.source != "fallback":
            self._remember(key, result)
        return result

    async def ascore(
        self, query: str, query_embedding: "np.ndarray | None" = None
    ) -> AmbiguityResult:
        """Async twin of :meth:`score` for the request hot path.

        Same semantics and LRU cache as :meth:`score`, but the HTTP embedding is
        fetched via the shared module-level ``httpx.AsyncClient`` so the event
        loop is never blocked by a synchronous socket call. The disabled
        short-circuit happens before any work, so a disabled feature never
        performs a network embed.
        """
        # Gate BEFORE any work: disabled => no embed, no network, no cache.
        if not bool(getattr(self._settings, "subspace_ambiguity_enabled", True)):
            return self._disabled_result()
        if self._protos is None:
            return self._disabled_result()

        key: Optional[str] = None
        if query_embedding is None:
            key = self._normalize(query)
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached

        try:
            vec = await self._embedding_async(query, query_embedding)
        except Exception as e:
            log.warning("subspace embedding failed (%s); falling back", e)
            return self._fallback_result()

        result = self._score_vector(vec)
        if key is not None and result.source != "fallback":
            self._remember(key, result)
        return result

    # -- internals ---------------------------------------------------------

    def _embedding(self, query: str, query_embedding: "np.ndarray | None") -> "Any":
        """Resolve the query embedding (caller-provided, injected, or HTTP)."""
        import numpy as np

        if query_embedding is not None:
            return np.asarray(query_embedding, dtype="float32")
        if self._embedder is not None:
            return np.asarray(self._embedder(query), dtype="float32")
        return self._http_embed(query)

    async def _embedding_async(self, query: str, query_embedding: "Any") -> "Any":
        """Async twin of :meth:`_embedding` using the shared async HTTP client."""
        import numpy as np

        if query_embedding is not None:
            return np.asarray(query_embedding, dtype="float32")
        if self._embedder is not None:
            return np.asarray(self._embedder(query), dtype="float32")
        return await self._http_embed_async(query)

    def _http_embed(self, query: str) -> "Any":
        """POST to the OpenAI-compatible embedding service (proxy bypassed)."""
        import httpx
        import numpy as np

        base = str(getattr(self._settings, "embed_base_url", "http://127.0.0.1:8003")).rstrip("/")
        model = str(
            getattr(self._settings, "embed_model", "paraphrase-multilingual-MiniLM-L12-v2")
        )
        # trust_env=False is critical: never route localhost calls via the
        # corporate proxy (they would hang).
        with httpx.Client(timeout=_EMBED_TIMEOUT_S, trust_env=False) as client:
            resp = client.post(f"{base}/v1/embeddings", json={"input": query, "model": model})
            resp.raise_for_status()
            data = resp.json()
        return np.asarray(data["data"][0]["embedding"], dtype="float32")

    async def _http_embed_async(self, query: str) -> "Any":
        """POST to the embedding service via the shared, reused async client."""
        import numpy as np

        base = str(getattr(self._settings, "embed_base_url", "http://127.0.0.1:8003")).rstrip("/")
        model = str(
            getattr(self._settings, "embed_model", "paraphrase-multilingual-MiniLM-L12-v2")
        )
        client = _get_async_client()
        resp = await client.post(f"{base}/v1/embeddings", json={"input": query, "model": model})
        resp.raise_for_status()
        data = resp.json()
        return np.asarray(data["data"][0]["embedding"], dtype="float32")

    def _prepare(self, vec: "Any") -> "Any":
        """Flatten, optionally whiten, then L2-normalize the query embedding."""
        import numpy as np

        e = np.asarray(vec, dtype="float32").reshape(-1)
        p = self._protos
        if p is not None and p.whiten_mean is not None and p.whiten_matrix is not None:
            mean = np.asarray(p.whiten_mean, dtype="float32").reshape(-1)
            matrix = np.asarray(p.whiten_matrix, dtype="float32")
            # Defensive: only whiten when shapes actually compose. A (D, 0)
            # matrix would otherwise project to a zero-length vector (M1).
            if (
                mean.shape[0] == e.shape[0]
                and matrix.ndim == 2
                and matrix.shape[0] == e.shape[0]
                and matrix.shape[1] > 0
            ):
                e = (e - mean) @ matrix
        norm = float(np.linalg.norm(e))
        if norm > 0:
            e = e / norm
        return e

    def _score_vector(self, vec: "Any") -> AmbiguityResult:
        """Score a prepared query embedding against the prototype bundle."""
        import numpy as np

        s = self._settings
        protos = self._protos
        assert protos is not None  # guarded by score()/available
        e = self._prepare(vec)

        # Compute similarities for all prototypes per subspace
        # protos.prototypes shape: (S, K, D) or None
        # protos.centroids shape: (S, D)
        sims: Dict[str, float] = {}
        all_sims: Dict[str, list[float]] = {}
        intra_spread: Dict[str, float] = {}
        proto_coverage: Dict[str, float] = {}

        for i, name in enumerate(protos.subspaces):
            try:
                if protos.prototypes is not None:
                    # Multi-prototype: (K, D)
                    proto_k = np.asarray(protos.prototypes[i], dtype="float32")
                    # Cosine similarities: (K,)
                    k_sims = proto_k @ e
                    sims[name] = float(k_sims.max())
                    all_sims[name] = k_sims.tolist()
                    intra_spread[name] = float(k_sims.std())
                    # Coverage: fraction of prototypes above threshold
                    coverage_thr = float(getattr(s, "subspace_proto_coverage_thr", 0.3))
                    proto_coverage[name] = float((k_sims > coverage_thr).mean())
                else:
                    # Single centroid fallback
                    centroid = np.asarray(protos.centroids[i], dtype="float32")
                    sim = float(centroid @ e)
                    sims[name] = sim
                    all_sims[name] = [sim]
                    intra_spread[name] = 0.0
                    proto_coverage[name] = 1.0 if sim > 0.3 else 0.0
            except Exception as exc:
                log.warning("subspace %s scoring failed (%s); falling back", name, exc)
                return self._fallback_result()

        # Weighted combination of max and mean similarity per subspace
        proto_weight = float(getattr(s, "subspace_proto_weight", 0.7))
        combined: Dict[str, float] = {}
        for name in protos.subspaces:
            k_sims = all_sims[name]
            max_sim = max(k_sims)
            mean_sim = float(np.mean(k_sims))
            combined[name] = proto_weight * max_sim + (1.0 - proto_weight) * mean_sim

        # Overall best subspace by combined score
        values = sorted(combined.values(), reverse=True)
        top1 = values[0] if values else 0.0
        top2 = values[1] if len(values) > 1 else 0.0
        margin = top1 - top2
        entropy = _normalized_entropy(
            list(combined.values()), float(getattr(s, "subspace_softmax_temp", 0.05))
        )
        argmax = max(combined, key=combined.get) if combined else None

        # Ambiguity thresholds
        min_top1 = float(getattr(s, "subspace_ambiguity_min_top1", 0.30))
        margin_thr = float(getattr(s, "subspace_ambiguity_margin", 0.08))
        entropy_max = float(getattr(s, "subspace_ambiguity_entropy_max", 0.85))
        intra_spread_thr = float(getattr(s, "subspace_intra_spread_thr", 0.15))

        # Intra-spread signal: max spread across subspaces
        max_intra_spread = max(intra_spread.values()) if intra_spread else 0.0

        # Weighted ambiguity score
        weights = getattr(s, "subspace_ambiguity_weights", None)
        if weights is None:
            weights = {"margin": 0.4, "entropy": 0.3, "top1": 0.2, "intra_spread": 0.1}

        score = _ambiguity_score_multi(
            top1=top1,
            margin=margin,
            entropy=entropy,
            max_intra_spread=max_intra_spread,
            min_top1=min_top1,
            margin_thr=margin_thr,
            entropy_max=entropy_max,
            intra_spread_thr=intra_spread_thr,
            weights=weights,
        )

        ambiguous = (top1 < min_top1) or (margin < margin_thr) or (entropy > entropy_max) or (max_intra_spread > intra_spread_thr)
        subspace = argmax if not ambiguous else None
        source = "protos"

        if self._classifier is not None:
            features = dict(sims)
            features.update({"top1": top1, "top2": top2, "margin": margin, "entropy": entropy})
            label = _classify(self._classifier, features, protos.subspaces)
            if label is not None:
                ambiguous = label not in protos.subspaces
                subspace = label if not ambiguous else None
                source = "classifier"

        return AmbiguityResult(
            sims=sims,
            all_sims=all_sims,
            top1=top1,
            top2=top2,
            margin=margin,
            entropy=entropy,
            score=score,
            ambiguous=ambiguous,
            subspace=subspace,
            source=source,
            intra_spread=intra_spread,
            proto_coverage=proto_coverage,
        )

    def _disabled_result(self) -> AmbiguityResult:
        """Neutral result used when no prototype bundle is available."""
        return AmbiguityResult(
            sims={},
            all_sims={},
            top1=0.0,
            top2=0.0,
            margin=0.0,
            entropy=0.0,
            score=0.0,
            ambiguous=False,
            subspace=None,
            source="disabled",
            intra_spread={},
            proto_coverage={},
        )

    def _fallback_result(self) -> AmbiguityResult:
        """Neutral result used when embedding failed."""
        return AmbiguityResult(
            sims={},
            all_sims={},
            top1=0.0,
            top2=0.0,
            margin=0.0,
            entropy=0.0,
            score=0.0,
            ambiguous=False,
            subspace=None,
            source="fallback",
            intra_spread={},
            proto_coverage={},
        )

    @staticmethod
    def _normalize(query: str) -> str:
        """Collapse whitespace for stable cache keys."""
        return " ".join((query or "").split())

    def _remember(self, key: str, result: AmbiguityResult) -> None:
        """Insert into the bounded LRU cache, evicting the oldest entry."""
        self._cache[key] = result
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)


def _normalized_entropy(sims: list[float], temp: float) -> float:
    """Softmax entropy of ``sims`` at temperature ``temp``, divided by ln(S)."""
    import numpy as np

    if not sims:
        return 0.0
    t = temp if temp and temp > 0 else 1.0
    x = np.asarray(sims, dtype="float64") / t
    x = x - x.max()
    p = np.exp(x)
    total = float(p.sum())
    if total <= 0:
        return 0.0
    p = p / total
    n = int(p.shape[0])
    if n <= 1:
        return 0.0
    h = float(-(p * np.log(p + 1e-12)).sum())
    return float(h / np.log(n))


def _ambiguity_score(
    top1: float,
    margin: float,
    entropy: float,
    min_top1: float,
    margin_thr: float,
    entropy_max: float,
) -> float:
    """Max normalized threshold excess, clipped to [0, 1]."""
    excesses = []
    if min_top1 > 0:
        excesses.append((min_top1 - top1) / min_top1)
    if margin_thr > 0:
        excesses.append((margin_thr - margin) / margin_thr)
    denom = 1.0 - entropy_max
    if denom > 0:
        excesses.append((entropy - entropy_max) / denom)
    if not excesses:
        return 0.0
    return float(min(1.0, max(0.0, max(excesses))))


def _ambiguity_score_multi(
    top1: float,
    margin: float,
    entropy: float,
    max_intra_spread: float,
    min_top1: float,
    margin_thr: float,
    entropy_max: float,
    intra_spread_thr: float,
    weights: dict[str, float],
) -> float:
    """Weighted ambiguity score with intra-spread signal, clipped to [0, 1].

    Following the spec:
      score = w1 * (1 - margin/margin_thr) + w2 * (entropy/entropy_max)
            + w3 * (1 - top1/min_top1) + w4 * (intra_spread/intra_spread_thr)
    Each term is clipped to [0, 1] before weighting.
    """
    w = weights or {"margin": 0.4, "entropy": 0.3, "top1": 0.2, "intra_spread": 0.1}

    signal_margin = 0.0
    if margin_thr > 0:
        signal_margin = max(0.0, min(1.0, (margin_thr - margin) / margin_thr))

    signal_entropy = 0.0
    if entropy_max > 0:
        signal_entropy = min(1.0, entropy / entropy_max)

    signal_top1 = 0.0
    if min_top1 > 0:
        signal_top1 = max(0.0, min(1.0, (min_top1 - top1) / min_top1))

    signal_spread = 0.0
    if intra_spread_thr > 0:
        signal_spread = min(1.0, max_intra_spread / intra_spread_thr)

    score = (
        w.get("margin", 0.4) * signal_margin
        + w.get("entropy", 0.3) * signal_entropy
        + w.get("top1", 0.2) * signal_top1
        + w.get("intra_spread", 0.1) * signal_spread
    )
    score = float(min(1.0, max(0.0, score)))
    # Clip near-zero scores to exactly 0 for test stability
    if score < 1e-7:
        score = 0.0
    return score


def _load_classifier(path: str) -> Optional[dict]:
    """Load the optional joblib classifier dict; ``None`` if unavailable.

    ``joblib.load`` is required because the payload is a fitted sklearn
    estimator; the file is a locally generated, gitignored artifact, not an
    untrusted user input stream (m4).
    """
    try:
        import joblib

        obj = joblib.load(path)
        if isinstance(obj, dict) and obj.get("clf") is not None:
            return obj
    except Exception as e:
        log.debug("subspace classifier unavailable at %s (%s)", path, e)
    return None


def _classify(classifier: dict, features: dict, subspaces: tuple[str, ...]) -> Optional[str]:
    """Return the classifier's predicted label, or ``None`` on failure."""
    try:
        import numpy as np

        names = classifier.get("feature_names") or list(subspaces)
        row = [[float(features.get(str(name), 0.0)) for name in names]]
        pred = classifier["clf"].predict(np.asarray(row, dtype="float32"))[0]
        return str(pred)
    except Exception as e:
        log.debug("subspace classifier failed (%s); using protos", e)
        return None


_SCORER_SINGLETON: Dict[int, "SubspaceScorer"] = {}


def get_scorer() -> SubspaceScorer:
    """Return the process-wide cached :class:`SubspaceScorer`.

    Cached per settings object so a reloaded :class:`Settings` yields a fresh
    scorer while normal callers reuse a single instance.
    """
    from .config import get_settings

    settings = get_settings()
    key = id(settings)
    scorer = _SCORER_SINGLETON.get(key)
    if scorer is None:
        scorer = SubspaceScorer.from_settings(settings)
        _SCORER_SINGLETON[key] = scorer
    return scorer
