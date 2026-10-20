# Subspace Ambiguity Score — Design Spec (frozen interface)

Goal: replace brittle keyword-only ambiguity detection with a **calibrated embedding
similarity score** over the 4 knowledge subspaces, then use it to route or clarify.

## Stages (recommended plan)

1. **Instrument** — compute the score, store it in state, emit it in `audit`. No behavior change.
2. **Cheap wins** — normalized prototypes, optional whitening, one shared query encode, LRU cache.
3. **Calibrate + late fusion** — fit thresholds on the labeled bench; combine pre-retrieval
   embedding score with post-retrieval distribution.
4. **Upgrade prototypes** — multi-prototype k-means, try BGE-m3.
5. **Learn** — swap thresholds for a small classifier distilled from the LLM judge; feedback loop.

Behavior is gated: `subspace_ambiguity_enabled` (compute + audit) and
`subspace_ambiguity_route` (actually route/clarify from the score; default **False** so
stage 1 is instrument-only).

## Subspaces

`("cheque", "facilities_person", "facilities_legal", "general")`

`folder_hierarchy` → subspace rules (handle both Persian spellings):
- contains `چک` (or `گزارش اعتباری چک` / `گزارش اعتبارسنجی چک`) → `cheque`
- contains `حقوقی` → `facilities_legal`
- contains `حقیقی` → `facilities_person`
- contains `عمومی` → `general`
- else → `facilities_person`

## Frozen runtime interface — `src/work_rag_orchestrator/subspaces.py`

```python
SUBSPACES: tuple[str, ...] = ("cheque", "facilities_person", "facilities_legal", "general")

@dataclass(frozen=True)
class AmbiguityResult:
    sims: dict[str, float]        # subspace -> cosine similarity
    top1: float                   # max sim
    top2: float                   # second max sim
    margin: float                 # top1 - top2
    entropy: float                # normalized softmax entropy in [0, 1]
    score: float                  # ambiguity in [0, 1]; higher = more ambiguous
    ambiguous: bool               # decision
    subspace: str | None          # argmax when not ambiguous, else None
    source: str                   # "protos" | "classifier" | "fallback" | "disabled"
    def as_dict(self) -> dict: ...   # sims, top1, top2, margin, entropy, score, ambiguous, subspace, source

@dataclass
class ProtoBundle:
    subspaces: tuple[str, ...]
    centroids: "np.ndarray"       # (S, D) L2-normalized
    prototypes: "np.ndarray | None"   # (S, K, D) optional multi-prototype
    whiten_mean: "np.ndarray | None"  # (D,)
    whiten_matrix: "np.ndarray | None"# (D, r)
    dim: int
    model: str
    kb_version: str

def load_protos(path: str) -> ProtoBundle: ...

class SubspaceScorer:
    def __init__(self, protos: ProtoBundle | None, settings, embedder=None, classifier=None): ...
    @classmethod
    def from_settings(cls, settings) -> "SubspaceScorer": ...
    @property
    def available(self) -> bool: ...
    def score(self, query: str, query_embedding: "np.ndarray | None" = None) -> AmbiguityResult: ...

def get_scorer() -> SubspaceScorer: ...   # process-wide cached singleton (respects settings)
```

**Scoring math**
- `e_q` = L2-normalized query embedding (optional whitening applied if bundle has it).
- `sims[s] = max over prototypes k of dot(e_q, proto[s,k])` (or `dot(e_q, centroid[s])`).
- `margin = top1 - top2`; `p = softmax(sims / T)` (T = `subspace_softmax_temp`, default 0.05);
  `entropy = H(p) / ln(S)`.
- `ambiguous = (top1 < subspace_ambiguity_min_top1) or (margin < subspace_ambiguity_margin)
   or (entropy > subspace_ambiguity_entropy_max)`.
- `subspace = argmax(sims)` when not ambiguous.
- `score` = max of the three normalized excesses, clipped to [0,1].
- If a classifier is loaded, it overrides `ambiguous`/`subspace` (`source="classifier"`).
- If `not available` → `source="disabled"`, `ambiguous=False` (defer to existing logic).
- Embedding is fetched via `embedder` callable if injected, else POST `{embed_base_url}/v1/embeddings`
  (httpx, `trust_env=False`); on failure → `source="fallback"`, `ambiguous=False`.
- `score()` results are LRU-cached by normalized query text.

## Config additions — `config.py` `Settings`

```
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
```

## State additions — `state.py` `RAGState`

```
ambiguity_score: Optional[float]
ambiguity_margin: Optional[float]
ambiguity_entropy: Optional[float]
ambiguity_ambiguous: Optional[bool]
ambiguity_source: Optional[str]
subspace_sims: Optional[Dict[str, float]]
```

## Artifact format — `data/subspace_protos.npz`

keys: `subspaces` (S,), `centroids` (S,D), `prototypes` (S,K,D) optional,
`prototype_subspaces` (S*K,), `whiten_mean` (D,), `whiten_matrix` (D,r) optional,
`dim` (int), `model` (str), `kb_version` (str), `built_at` (str). Artifacts are **not**
committed (`.gitignore` covers `*.npz`); regenerate via the builder script.

`data/subspace_clf.joblib` = dict `{clf, feature_names, subspaces, model, trained_at}`.

## Data facts (for tooling)

- KB Postgres: `postgresql://postgres:postgres@127.0.0.1:5433/kb_manager` (asyncpg).
  `chunks.embedding` (vector 384), `chunks.document_id`; `documents.metadata->>'folder_hierarchy'` (JSON list).
- Dense npz: `components/knowledgebase/kb-manager/data/dense_embeddings.npz`
  keys `ids` (1110,), `vectors` (1110,384), `fingerprint`.
- Embedding model: `paraphrase-multilingual-MiniLM-L12-v2` (384-d).
- Labeled turns: `evaluation/bench_runner.py` `CONVERSATIONS` (`expected_finish`, `expected_subspace`).
- Python: `/tmp/kb-venv/bin/python` (numpy, sklearn, sentence_transformers, asyncpg, joblib, pytest, httpx).
- Orchestrator tests: `PYTHONPATH=./src /tmp/kb-venv/bin/python -m pytest tests/ -q`
  (2 pre-existing failures: `test_memory_coref.py::TestRetrieveUsesRewrite`, `test_orchestrator.py::TestRetrieve` — ignore).

## Ownership (avoid file conflicts)

- **Agent A (offline tooling):** `scripts/build_subspace_protos.py`, `scripts/calibrate_subspace.py`,
  `scripts/train_subspace_clf.py`, and generated `data/*.npz|*.json|*.joblib`.
- **Agent B (core):** `src/work_rag_orchestrator/subspaces.py`, `config.py`, `state.py`.
- **Agent C (integration):** `nodes/retrieve.py`, `graph.py`, `api.py`.
- **Agent D (tests):** `tests/test_subspaces.py`.

No agent commits/pushes. Preserve unrelated WIP (e.g. the uncommitted sticky-decay block in `api.py`).
