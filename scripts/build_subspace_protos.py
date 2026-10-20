#!/usr/bin/env python3
"""Build embedding-subspace prototypes for the ambiguity scorer.

Offline tooling for ``docs/SUBSPACE_AMBIGUITY.md``: load chunk embeddings +
subspace labels (Postgres first, dense ``.npz`` fallback), map each document's
``folder_hierarchy`` to one of the four knowledge subspaces, and persist
per-subspace centroids and k-means multi-prototypes (plus optional PCA
whitening) to ``data/subspace_protos.npz``.

The artifact matches the frozen format exactly:
    subspaces (S,), centroids (S,D), prototypes (S,K,D), prototype_subspaces
    (S*K,), whiten_mean (D,), whiten_matrix (D,r), dim (int), model (str),
    kb_version (str), built_at (str).

This module also exposes the small helpers used by ``calibrate_subspace.py`` and
``train_subspace_clf.py`` (label mapping, proto loading, embedding, feature
scoring) so the three scripts share one implementation.

Usage:
    python scripts/build_subspace_protos.py --out data/subspace_protos.npz
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

# Canonical knowledge subspaces. Mirrors src/work_rag_orchestrator/subspaces.py
# (owned by another agent); duplicated here so offline tooling never imports src.
SUBSPACES: tuple[str, ...] = ("cheque", "facilities_person", "facilities_legal", "general")

# Embedding model stored in the artifact. The scorer owns the query-side model
# via config.embed_model; this is metadata only.
DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

DEFAULT_DB_URL = "postgresql://postgres:postgres@127.0.0.1:5433/kb_manager"
DEFAULT_NPZ = (
    "/splunk-data/v1/Work_Credit-RAG_Phase1/components/knowledgebase/"
    "kb-manager/data/dense_embeddings.npz"
)

# Repo root (…/components/orchestrator) and canonical data dir. Paths are
# resolved against this so artifacts always land in the right place regardless
# of the caller's CWD.
ORCH_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ORCH_ROOT / "data"

# Offline-provisioned HF mirror — same root kb_manager.model_cache searches.
OFFLINE_MODEL_ROOTS = [
    "/splunk-data/v1/Work_RAG-Server-Setup/offline-prep/models/huggingface",
]

# Persisted id -> subspace map so the ``--npz`` fallback stays usable when the
# database is unreachable (the npz itself carries no labels).
LABELS_SIDECAR = DATA_DIR / "subspace_chunk_labels.json"


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

def map_folder_to_subspace(folder_hierarchy: Any) -> str:
    """Map a document ``folder_hierarchy`` to a subspace.

    Rule order is significant and comes straight from the frozen spec: cheque
    first (so ``گزارش اعتبارسنجی چک`` never falls through to حقیقی/حقوقی),
    then legal, person, general, else default to ``facilities_person``. Both
    Persian cheque spellings are accepted (``گزارش اعتباری چک`` /
    ``گزارش اعتبارسنجی چک``), as well as a bare ``چک``.
    """
    if isinstance(folder_hierarchy, (list, tuple)):
        text = " ".join(str(part) for part in folder_hierarchy)
    else:
        text = str(folder_hierarchy or "")

    if (
        "گزارش اعتباری چک" in text
        or "گزارش اعتبارسنجی چک" in text
        or "چک" in text
    ):
        return "cheque"
    if "حقوقی" in text:
        return "facilities_legal"
    if "حقیقی" in text:
        return "facilities_person"
    if "عمومی" in text:
        return "general"
    return "facilities_person"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _decode_id(raw: Any) -> str:
    """Normalize an npz id (often a ``|S`` bytes scalar) to a plain string."""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", "replace")
    return str(raw)


def _parse_vector(raw: Any) -> Optional[np.ndarray]:
    """Parse an embedding from pgvector text, list, or ndarray into float32."""
    if raw is None:
        return None
    if isinstance(raw, np.ndarray):
        return raw.astype(np.float32).reshape(-1)
    if isinstance(raw, (list, tuple)):
        arr = np.asarray(raw, dtype=np.float32).reshape(-1)
        return arr if arr.size else None
    text = str(raw).strip().strip("[]")
    if not text:
        return None
    try:
        arr = np.fromstring(text, sep=",", dtype=np.float32)
    except Exception:
        return None
    return arr if arr.size else None


async def _load_from_db(db_url: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Load embeddings + subspace labels from KB Postgres (asyncpg).

    Returns ``(vectors, labels, chunk_ids)``. Raises on any connection/query
    failure so the caller can fall back to npz.
    """
    import asyncpg

    conn = await asyncpg.connect(db_url, timeout=10)
    try:
        rows = await conn.fetch(
            "SELECT c.id, c.embedding, d.metadata FROM chunks c "
            "JOIN documents d ON c.document_id = d.id "
            "WHERE c.embedding IS NOT NULL"
        )
    finally:
        await conn.close()

    vectors: list[np.ndarray] = []
    labels: list[str] = []
    chunk_ids: list[str] = []
    for row in rows:
        vec = _parse_vector(row["embedding"])
        if vec is None or vec.size == 0:
            continue
        meta = row["metadata"]
        if isinstance(meta, (str, bytes)):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        folder = meta.get("folder_hierarchy") if isinstance(meta, dict) else None
        vectors.append(vec)
        labels.append(map_folder_to_subspace(folder))
        chunk_ids.append(str(row["id"]))

    if not vectors:
        raise RuntimeError("database returned no usable embeddings")
    return np.stack(vectors).astype(np.float32), labels, chunk_ids


def _load_label_sidecar(path: Path) -> dict[str, str]:
    """Load an id -> subspace map from JSON or an ``.npz`` sidecar."""
    if not path or not path.is_file():
        return {}
    try:
        if path.suffix == ".npz":
            data = np.load(path, allow_pickle=True)
            ids = [_decode_id(x) for x in data["ids"].tolist()]
            raw = data["labels"].tolist() if "labels" in data else data["folder_hierarchy"].tolist()
            out: dict[str, str] = {}
            for i, r in zip(ids, raw):
                if isinstance(r, bytes):
                    r = r.decode("utf-8", "replace")
                # A bare subspace token is used directly; anything else is a
                # folder hierarchy and gets mapped.
                out[i] = r if r in SUBSPACES else map_folder_to_subspace(r)
            return out
        with open(path, encoding="utf-8") as fh:
            obj = json.load(fh)
        return {str(k): str(v) for k, v in obj.items()}
    except Exception as exc:  # malformed sidecar is non-fatal
        print(f"[warn] could not read labels sidecar {path}: {exc}")
        return {}


def _load_from_npz(
    npz_path: str, labels_path: Optional[str]
) -> tuple[np.ndarray, list[str], int]:
    """Load embeddings from the dense index npz and join them to labels.

    Labels are resolved from: explicit ``--labels`` file, a sibling
    ``*_labels.json``/``.npz``, the canonical sidecar, else every chunk falls
    back to the default subspace with a warning. Returns (vectors, labels,
    n_missing_labels).
    """
    with np.load(npz_path, allow_pickle=True) as data:
        vectors = np.asarray(data["vectors"], dtype=np.float32)
        ids = [_decode_id(x) for x in data["ids"].tolist()]

    candidates = []
    if labels_path:
        candidates.append(Path(labels_path))
    candidates.append(Path(str(npz_path).replace(".npz", "_labels.json")))
    candidates.append(Path(str(npz_path).replace(".npz", "_labels.npz")))
    candidates.append(LABELS_SIDECAR)

    label_map: dict[str, str] = {}
    for cand in candidates:
        label_map = _load_label_sidecar(cand)
        if label_map:
            break

    labels: list[str] = []
    missing = 0
    for chunk_id in ids:
        sub = label_map.get(chunk_id)
        if sub is None:
            sub = "facilities_person"  # spec default when hierarchy is unknown
            missing += 1
        labels.append(sub)
    return vectors, labels, missing


# ---------------------------------------------------------------------------
# Proto construction
# ---------------------------------------------------------------------------

def build_protos(
    vectors: np.ndarray,
    labels: list[str],
    k: int,
    whiten_dim: int,
    model: str,
    kb_version: str,
) -> SimpleNamespace:
    """Turn labeled chunk vectors into centroids, prototypes, and whitening.

    Vectors are L2-normalized before any geometry, so dot products are cosine
    similarities and the scorer's plain-dot math is valid. When whitening is
    enabled the corpus is projected into the whitened space *before* centroids
    and prototypes are computed, because the runtime whitens the query and then
    dots it directly against these stored prototypes — both sides must share the
    same basis.
    """
    from sklearn.cluster import KMeans

    vecs = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms

    whiten_mean = None
    whiten_matrix = None
    working = vecs
    if whiten_dim and whiten_dim > 0 and vecs.shape[0] > 1:
        whiten_mean, whiten_matrix = _fit_whitening(vecs, whiten_dim)
        working = (vecs - whiten_mean) @ whiten_matrix
        wnorm = np.linalg.norm(working, axis=1, keepdims=True)
        wnorm[wnorm == 0] = 1.0
        working = working / wnorm

    n_sub = len(SUBSPACES)
    dim = working.shape[1]
    centroids = np.zeros((n_sub, dim), dtype=np.float32)
    prototypes = np.zeros((n_sub, k, dim), dtype=np.float32)
    prototype_subspaces: list[str] = []
    counts: dict[str, int] = {}

    for si, name in enumerate(SUBSPACES):
        member_idx = [i for i, lab in enumerate(labels) if lab == name]
        counts[name] = len(member_idx)

        # Every (subspace, prototype) slot is tagged even when padded so the
        # flattened axis stays aligned with `subspaces`.
        prototype_subspaces.extend([name] * k)
        if not member_idx:
            continue

        sub = working[member_idx]
        centroid = sub.mean(axis=0)
        cnorm = float(np.linalg.norm(centroid))
        centroids[si] = centroid / cnorm if cnorm > 0 else centroid

        # Fewer members than prototypes -> pad with zeros (cosine 0, harmless
        # to the max-over-prototypes reduction).
        kk = max(1, min(k, len(member_idx)))
        km = KMeans(n_clusters=kk, n_init=10, random_state=42).fit(sub)
        centers = km.cluster_centers_
        pnorm = np.linalg.norm(centers, axis=1, keepdims=True)
        pnorm[pnorm == 0] = 1.0
        prototypes[si, :kk] = (centers / pnorm).astype(np.float32)

    return SimpleNamespace(
        subspaces=SUBSPACES,
        centroids=centroids,
        prototypes=prototypes,
        prototype_subspaces=np.asarray(prototype_subspaces),
        whiten_mean=whiten_mean,
        whiten_matrix=whiten_matrix,
        dim=dim,
        model=model,
        kb_version=kb_version,
        counts=counts,
    )


def _fit_whitening(vecs: np.ndarray, whiten_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit a PCA transform; return (mean (D,), matrix (D, r)).

    The matrix is the PCA rotation/reduction (top-``r`` components), *not*
    scaled by ``1/sqrt(eigenvalue)``. Dividing by the per-component standard
    deviation is the classical "whitening" form, but on 384-d sentence
    embeddings with a handful of near-zero eigenvalues it amplifies noise
    directions and collapsed subspace discrimination on the labeled bench
    (balanced accuracy 0.94 -> 0.64). The rotation keeps geometry stable while
    still satisfying the runtime's ``(e - mean) @ matrix`` contract.
    """
    from sklearn.decomposition import PCA

    r = int(min(whiten_dim, vecs.shape[1], vecs.shape[0]))
    # Full SVD keeps the artifact byte-reproducible; sklearn's "auto" would pick
    # randomized SVD for this shape, whose random_state is not exposed.
    pca = PCA(n_components=r, whiten=False, svd_solver="full").fit(vecs)
    mean = pca.mean_.astype(np.float32)
    matrix = pca.components_.T.astype(np.float32)
    return mean, matrix


def save_protos(bundle: SimpleNamespace, out_path: Path) -> None:
    """Persist the bundle to npz with the exact frozen key set."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        subspaces=np.asarray(bundle.subspaces),
        centroids=bundle.centroids.astype(np.float32),
        prototypes=bundle.prototypes.astype(np.float32),
        prototype_subspaces=np.asarray(bundle.prototype_subspaces),
        whiten_mean=(
            bundle.whiten_mean.astype(np.float32)
            if bundle.whiten_mean is not None
            else np.zeros(bundle.dim, dtype=np.float32)
        ),
        whiten_matrix=(
            bundle.whiten_matrix.astype(np.float32)
            if bundle.whiten_matrix is not None
            else np.zeros((bundle.dim, 0), dtype=np.float32)
        ),
        dim=np.int32(bundle.dim),
        model=np.asarray(bundle.model),
        kb_version=np.asarray(bundle.kb_version),
        built_at=np.asarray(datetime.now(timezone.utc).isoformat()),
    )


def load_protos(path: str | Path) -> SimpleNamespace:
    """Load a proto bundle from npz into a lightweight namespace.

    Mirrors ``ProtoBundle`` from the runtime; returns ``None`` when the file is
    missing/malformed so callers can degrade gracefully.
    """
    try:
        with np.load(path, allow_pickle=True) as data:
            prototypes = None
            if "prototypes" in data:
                prototypes = np.asarray(data["prototypes"], dtype=np.float32)
            return SimpleNamespace(
                subspaces=tuple(str(s) for s in data["subspaces"].tolist()),
                centroids=np.asarray(data["centroids"], dtype=np.float32),
                prototypes=prototypes,
                whiten_mean=(
                    np.asarray(data["whiten_mean"], dtype=np.float32)
                    if "whiten_mean" in data and data["whiten_mean"].size
                    else None
                ),
                whiten_matrix=(
                    np.asarray(data["whiten_matrix"], dtype=np.float32)
                    if "whiten_matrix" in data and data["whiten_matrix"].size
                    else None
                ),
                dim=int(data["dim"]) if "dim" in data else 0,
                model=str(data["model"]) if "model" in data else "",
                kb_version=str(data["kb_version"]) if "kb_version" in data else "",
            )
    except Exception as exc:
        print(f"[warn] could not load protos {path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Shared scoring helpers (used by calibrate + train)
# ---------------------------------------------------------------------------

def resolve_model_path(name: str) -> str:
    """Resolve a model id to a local directory for offline loading."""
    if not name:
        return name
    candidates = [Path(name)]
    flat = name.replace("/", "_")
    candidates.extend(Path(root) / flat for root in OFFLINE_MODEL_ROOTS)
    for cand in candidates:
        if cand.is_dir():
            return str(cand)

    # Short alias (config uses "paraphrase-…") -> full repo id in HF cache.
    if "/" not in name:
        try:
            from huggingface_hub import snapshot_download

            return snapshot_download(
                f"sentence-transformers/{name}", local_files_only=True
            )
        except Exception:
            pass
    return name


def embed_texts(
    texts: list[str],
    model: str = DEFAULT_MODEL,
    embed_url: Optional[str] = None,
) -> np.ndarray:
    """Embed texts locally (default) or via an OpenAI-compatible HTTP service."""
    if embed_url:
        import httpx

        out: list[np.ndarray] = []
        base = embed_url.rstrip("/")
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            for text in texts:
                resp = client.post(
                    f"{base}/v1/embeddings", json={"input": text, "model": model}
                )
                resp.raise_for_status()
                out.append(
                    np.asarray(resp.json()["data"][0]["embedding"], dtype=np.float32)
                )
        return np.stack(out).astype(np.float32)

    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(resolve_model_path(model))
    return np.asarray(st.encode(texts, normalize_embeddings=False), dtype=np.float32)


def prepare_vectors(vecs: np.ndarray, protos: SimpleNamespace) -> np.ndarray:
    """Apply proto whitening (if any) then L2-normalize query/corpus vectors."""
    arr = np.asarray(vecs, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if protos is not None and protos.whiten_mean is not None and protos.whiten_matrix is not None:
        if protos.whiten_mean.shape[0] == arr.shape[1]:
            arr = (arr - protos.whiten_mean) @ protos.whiten_matrix
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def normalized_entropy(sims: np.ndarray, temp: float) -> np.ndarray:
    """Row-wise softmax entropy at temperature ``temp``, divided by ln(S)."""
    t = temp if temp and temp > 0 else 1.0
    x = np.asarray(sims, dtype=np.float64) / t
    x = x - x.max(axis=1, keepdims=True)
    p = np.exp(x)
    p = p / p.sum(axis=1, keepdims=True)
    n = p.shape[1]
    if n <= 1:
        return np.zeros(p.shape[0], dtype=np.float64)
    h = -(p * np.log(p + 1e-12)).sum(axis=1)
    return h / np.log(n)


def score_features(
    vectors: np.ndarray, protos: SimpleNamespace, temp: float = 0.05
) -> dict[str, np.ndarray]:
    """Compute sims/top1/top2/margin/entropy exactly like the runtime scorer."""
    e = prepare_vectors(vectors, protos)
    if protos.prototypes is not None:
        # (N, S, K) -> max over prototypes -> (N, S).
        sims = np.einsum("nd,skd->nsk", e, protos.prototypes).max(axis=2)
    else:
        sims = e @ protos.centroids.T

    ordered = np.sort(sims, axis=1)
    top1 = ordered[:, -1]
    top2 = ordered[:, -2] if sims.shape[1] > 1 else np.zeros(sims.shape[0])
    return {
        "sims": sims,
        "top1": top1,
        "top2": top2,
        "margin": top1 - top2,
        "entropy": normalized_entropy(sims, temp),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_out(path: str) -> Path:
    """Resolve an output path, anchoring relative paths at the orchestrator root."""
    p = Path(path)
    return p if p.is_absolute() else ORCH_ROOT / p


def _read_npz_fingerprint(npz_path: str) -> Optional[str]:
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            if "fingerprint" in data:
                fp = data["fingerprint"]
                value = fp.tolist()[0] if hasattr(fp, "tolist") else fp
                return value.decode() if isinstance(value, bytes) else str(value)
    except Exception:
        return None
    return None


async def _amain(args: argparse.Namespace) -> int:
    vectors: Optional[np.ndarray] = None
    labels: Optional[list[str]] = None
    db_ids: list[str] = []
    source = ""

    # Primary source: Postgres (embeddings + labels in one shot).
    try:
        vectors, labels, db_ids = await _load_from_db(args.db_url)
        source = f"db ({args.db_url})"
    except Exception as exc:
        print(f"[warn] DB load failed ({type(exc).__name__}: {exc}); trying npz")

    # Fallback: dense index npz + labels sidecar/CSV.
    missing = 0
    if vectors is None:
        try:
            vectors, labels, missing = _load_from_npz(args.npz, args.labels)
            source = f"npz ({args.npz})"
            if missing:
                print(
                    f"[warn] {missing}/{len(labels)} npz chunks had no label; "
                    "defaulted to facilities_person"
                )
        except Exception as exc:
            print(f"[error] npz load failed ({type(exc).__name__}: {exc})")
            return 1

    assert vectors is not None and labels is not None

    # Persist the id->label sidecar so future npz-only runs keep their labels.
    if args.labels_out and db_ids:
        try:
            _write_label_sidecar(db_ids, labels, Path(args.labels_out))
        except Exception as exc:
            print(f"[warn] could not write labels sidecar: {exc}")

    kb_version = _read_npz_fingerprint(args.npz) or f"db:{vectors.shape[0]}"
    bundle = build_protos(
        vectors, labels, k=args.k, whiten_dim=args.whiten_dim,
        model=args.model, kb_version=kb_version,
    )
    out_path = _resolve_out(args.out)
    save_protos(bundle, out_path)

    print(f"\nsource     : {source}")
    print(f"chunks     : {vectors.shape[0]}  dim={bundle.dim}  k={args.k}"
          f"  whiten_dim={args.whiten_dim}")
    print(f"model      : {bundle.model}")
    print(f"kb_version : {bundle.kb_version}")
    print(f"out        : {out_path}")
    print("\nper-subspace counts / shapes:")
    print(f"  {'subspace':<18}{'chunks':>8}   centroid        prototypes")
    for si, name in enumerate(bundle.subspaces):
        print(
            f"  {name:<18}{bundle.counts.get(name, 0):>8}"
            f"   {tuple(bundle.centroids[si].shape)!s:<16} {tuple(bundle.prototypes[si].shape)!s}"
        )
    print(f"\nshapes: centroids={bundle.centroids.shape}"
          f" prototypes={bundle.prototypes.shape}"
          f" prototype_subspaces={bundle.prototype_subspaces.shape}")
    if bundle.whiten_matrix is not None:
        print(f"whitening: mean={bundle.whiten_mean.shape}"
              f" matrix={bundle.whiten_matrix.shape}")
    return 0


def _write_label_sidecar(ids: list[str], labels: list[str], out_path: Path) -> None:
    """Write chunk id -> subspace so the npz fallback can join labels offline."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({i: l for i, l in zip(ids, labels)}, fh, ensure_ascii=False, indent=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build subspace prototype artifact")
    parser.add_argument("--db-url", default=DEFAULT_DB_URL, help="KB Postgres DSN")
    parser.add_argument("--npz", default=DEFAULT_NPZ, help="dense embeddings npz fallback")
    parser.add_argument("--labels", default=None, help="optional id->subspace JSON/NPZ labels")
    parser.add_argument(
        "--labels-out",
        default=str(LABELS_SIDECAR),
        help="where to cache the id->subspace sidecar after a DB load",
    )
    parser.add_argument("--out", default="data/subspace_protos.npz", help="output npz")
    parser.add_argument("--k", type=int, default=4, help="prototypes per subspace")
    parser.add_argument("--whiten-dim", type=int, default=128, help="PCA whitening dim (0=off)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="embedding model id metadata")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.exit(asyncio.run(_amain(args)))


if __name__ == "__main__":
    main()
