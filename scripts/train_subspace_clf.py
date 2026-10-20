#!/usr/bin/env python3
"""Train the optional subspace classifier distilled from the labeled bench.

Features are the same vector the runtime scorer already computes:
``[sims(4), margin, top1, entropy]``. Two logistic-regression models are fit —
a binary "ambiguous" gate and a 4-way subspace router (trained on non-ambiguous
rows only) — plus a combined 5-class model stored under the spec's single
``clf`` key for compatibility with ``src/work_rag_orchestrator/subspaces.py``.

Output: ``data/subspace_clf.joblib`` =
    {clf_ambiguous, clf_subspace, feature_names, subspaces, model, trained_at, clf}

Usage:
    python scripts/train_subspace_clf.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_subspace_protos import (  # noqa: E402  (sibling import)
    DEFAULT_MODEL,
    ORCH_ROOT,
    SUBSPACES,
    embed_texts,
    load_protos,
    score_features,
)
from calibrate_subspace import (  # noqa: E402  (sibling import)
    dedupe_rows,
    find_bench_runner,
    load_bench_labels,
    load_query_jsonl,
)

# Feature order is frozen: the four subspace sims then margin/top1/entropy.
# Naming the sims after the subspaces lets the runtime `_classify` map its
# features dict (keyed by subspace name) directly onto the model input.
FEATURE_NAMES = [*SUBSPACES, "margin", "top1", "entropy"]


def build_feature_matrix(feats: dict[str, np.ndarray]) -> np.ndarray:
    """Assemble the (N, 7) feature matrix in :data:`FEATURE_NAMES` order."""
    cols = [feats["sims"]]
    cols.append(feats["margin"].reshape(-1, 1))
    cols.append(feats["top1"].reshape(-1, 1))
    cols.append(feats["entropy"].reshape(-1, 1))
    return np.concatenate(cols, axis=1).astype(np.float32)


def _fit_classifier(X: np.ndarray, y: np.ndarray) -> Any:
    """Fit multinomial logistic regression, falling back to a dummy model when
    the training fold contains a single class."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier

    if len(set(y.tolist())) < 2:
        return DummyClassifier(strategy="most_frequent").fit(X, y)
    return LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, y)


def _report(name: str, clf: Any, X: np.ndarray, y: np.ndarray) -> None:
    """Print train accuracy and a confusion matrix for one classifier."""
    from sklearn.metrics import accuracy_score, confusion_matrix

    pred = clf.predict(X)
    acc = accuracy_score(y, pred)
    labels = sorted(set(y.tolist()) | set(np.asarray(pred).tolist()))
    cm = confusion_matrix(y, pred, labels=labels)
    print(f"\n[{name}] train accuracy = {acc:.3f}  (n={len(y)})")
    print("  confusion (rows=true, cols=pred):")
    header = "".join(f"{str(l)[:10]:>12}" for l in labels)
    print(f"  {'':>12}{header}")
    for i, lab in enumerate(labels):
        row = "".join(f"{int(v):>12}" for v in cm[i])
        print(f"  {str(lab)[:12]:>12}{row}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train subspace classifier")
    parser.add_argument("--protos", default="data/subspace_protos.npz", help="proto npz")
    parser.add_argument("--queries", default=None, help="extra labeled queries JSONL")
    parser.add_argument("--embed-url", default=None, help="embedding service base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="embedding model id")
    parser.add_argument("--temp", type=float, default=0.05, help="softmax temperature")
    parser.add_argument("--out", default="data/subspace_clf.joblib", help="output joblib")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    protos_path = Path(args.protos)
    if not protos_path.is_absolute():
        protos_path = ORCH_ROOT / protos_path
    protos = load_protos(protos_path)
    if protos is None:
        print(f"[error] no protos at {args.protos}; run build_subspace_protos.py first")
        return 1

    rows = load_bench_labels(find_bench_runner())
    if args.queries:
        rows.extend(load_query_jsonl(Path(args.queries)))
    rows = dedupe_rows(rows)
    if not rows:
        print("[error] no labeled queries; nothing to train on")
        return 1

    vectors = embed_texts([r["query"] for r in rows], model=args.model, embed_url=args.embed_url)
    feats = score_features(vectors, protos, temp=args.temp)
    X = build_feature_matrix(feats)

    y_amb = np.array([1 if r["ambiguous"] else 0 for r in rows])
    subspace_for_row = np.array([r["subspace"] for r in rows], dtype=object)

    # (a) binary ambiguity gate over every row.
    clf_ambiguous = _fit_classifier(X, y_amb)

    # (b) 4-way router on non-ambiguous rows only.
    route_mask = y_amb == 0
    X_route = X[route_mask]
    y_route = subspace_for_row[route_mask]
    if len(y_route) == 0:
        print("[error] no non-ambiguous training rows; cannot fit subspace router")
        return 1
    clf_subspace = _fit_classifier(X_route, y_route)

    # Combined 5-class model for the spec's single `clf` slot: predicts either a
    # subspace name or the literal "ambiguous", which the runtime treats as
    # ambiguous (label not in subspaces).
    y_combined = np.array(
        ["ambiguous" if amb else sub for amb, sub in zip(y_amb, subspace_for_row)],
        dtype=object,
    )
    clf_combined = _fit_classifier(X, y_combined)

    _report("clf_ambiguous", clf_ambiguous, X, y_amb)
    _report("clf_subspace", clf_subspace, X_route, y_route)
    _report("clf (combined)", clf_combined, X, y_combined)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ORCH_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import joblib

    joblib.dump(
        {
            "clf_ambiguous": clf_ambiguous,
            "clf_subspace": clf_subspace,
            "clf": clf_combined,
            "feature_names": list(FEATURE_NAMES),
            "subspaces": list(SUBSPACES),
            "model": protos.model or args.model,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        },
        out_path,
    )
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
