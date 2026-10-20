#!/usr/bin/env python3
"""Calibrate the subspace ambiguity thresholds against the labeled bench.

Loads ``data/subspace_protos.npz``, builds a labeled query set from
``evaluation/bench_runner.py`` (``expected_finish == "clarify"`` -> ambiguous;
``expected_finish == "stop"`` with an ``expected_subspace`` -> route label),
embeds the queries locally with MiniLM, and grid-searches
``margin``/``min_top1``/``entropy_max`` to maximize balanced accuracy for the
clarify decision. Writes ``data/subspace_calibration.json`` and prints a table.

Usage:
    python scripts/calibrate_subspace.py                # bench labels
    python scripts/calibrate_subspace.py --queries q.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

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

# Config defaults (see src/work_rag_orchestrator/config.py) — reported for
# reference so the reader can see how far calibration moved the thresholds.
DEFAULT_THRESHOLDS = {"margin": 0.08, "min_top1": 0.30, "entropy_max": 0.85}
DEFAULT_TEMP = 0.05


# ---------------------------------------------------------------------------
# Labeled query set
# ---------------------------------------------------------------------------

def find_bench_runner() -> Optional[Path]:
    """Locate ``evaluation/bench_runner.py`` across known repo layouts."""
    candidates = [
        ORCH_ROOT / "evaluation" / "bench_runner.py",
        ORCH_ROOT.parent / "evaluation" / "bench_runner.py",
        Path("/splunk-data/v1/Work_Credit-RAG_Phase1/evaluation/bench_runner.py"),
        Path("/splunk-data/v1/Work_Credit-RAG_Phase1/components/evaluation/bench_runner.py"),
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def load_bench_labels(bench_path: Optional[Path]) -> list[dict[str, Any]]:
    """Extract (query, ambiguous, subspace) rows from bench CONVERSATIONS.

    Only turns that carry an informative label are kept: clarify turns are the
    positive (ambiguous) class; stop turns with an ``expected_subspace`` are the
    route class. Greetings and unlabeled stop turns are skipped.
    """
    if bench_path is None:
        return []
    spec = importlib.util.spec_from_file_location("_bench_runner", bench_path)
    if spec is None or spec.loader is None:
        return []
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rows: list[dict[str, Any]] = []
    for conv in getattr(module, "CONVERSATIONS", []):
        for turn in conv.get("turns", []):
            finish = turn.get("expected_finish")
            subspace = turn.get("expected_subspace")
            query = turn.get("user")
            if not query:
                continue
            if finish == "clarify":
                rows.append({"query": query, "ambiguous": True, "subspace": None})
            elif finish == "stop" and subspace:
                rows.append({"query": query, "ambiguous": False, "subspace": subspace})
    return rows


def load_query_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load extra labeled queries from JSONL: {query, ambiguous, subspace}."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(
                {
                    "query": str(obj["query"]),
                    "ambiguous": bool(obj.get("ambiguous", False)),
                    "subspace": obj.get("subspace"),
                }
            )
    return rows


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one row per normalized query; later rows overwrite earlier ones."""
    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = " ".join(row["query"].split())
        merged[key] = row
    return list(merged.values())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Precision/recall/F1 for the positive (ambiguous) class + balanced acc."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    tpr = recall
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float((tpr + tnr) / 2.0),
    }


def route_accuracy(
    predicted_ambiguous: np.ndarray,
    argmax: np.ndarray,
    y_amb: np.ndarray,
    expected_subspace: np.ndarray,
) -> float:
    """Subspace routing accuracy over rows that are truly non-ambiguous.

    A route row counts as correct only when the scorer both keeps it
    non-ambiguous and its argmax subspace matches the expected one.
    """
    mask = y_amb == 0
    if not mask.any():
        return 0.0
    correct = (~predicted_ambiguous[mask]) & (argmax[mask] == expected_subspace[mask])
    return float(np.mean(correct))


def evaluate_thresholds(
    margin: float,
    min_top1: float,
    entropy_max: float,
    feats: dict[str, np.ndarray],
    y_amb: np.ndarray,
    expected_subspace: np.ndarray,
    argmax: np.ndarray,
) -> dict[str, Any]:
    """Evaluate one threshold triple and return metrics + ambiguity predictions."""
    pred_amb = (
        (feats["top1"] < min_top1)
        | (feats["margin"] < margin)
        | (feats["entropy"] > entropy_max)
    )
    metrics = binary_metrics(y_amb.astype(int), pred_amb.astype(int))
    metrics["route_accuracy"] = route_accuracy(pred_amb, argmax, y_amb, expected_subspace)
    metrics["n_ambiguous_pred"] = int(pred_amb.sum())
    return {
        "thresholds": {"margin": margin, "min_top1": min_top1, "entropy_max": entropy_max},
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_out(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ORCH_ROOT / p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate subspace ambiguity thresholds")
    parser.add_argument("--protos", default="data/subspace_protos.npz", help="proto npz")
    parser.add_argument("--queries", default=None, help="extra labeled queries JSONL")
    parser.add_argument("--embed-url", default=None, help="embedding service base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="embedding model id")
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP, help="softmax temperature")
    parser.add_argument("--out", default="data/subspace_calibration.json", help="output JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    protos = load_protos(_resolve_out(args.protos))
    if protos is None:
        print(f"[error] no protos at {args.protos}; run build_subspace_protos.py first")
        return 1

    bench_path = find_bench_runner()
    rows = load_bench_labels(bench_path)
    if bench_path is None:
        print("[warn] evaluation/bench_runner.py not found; using --queries only")
    if args.queries:
        rows.extend(load_query_jsonl(Path(args.queries)))
    rows = dedupe_rows(rows)
    if not rows:
        print("[error] no labeled queries; nothing to calibrate")
        return 1

    n_amb = sum(1 for r in rows if r["ambiguous"])
    print(f"label source : {bench_path or '(--queries only)'}")
    print(f"queries      : {len(rows)}  ({n_amb} ambiguous / {len(rows) - n_amb} route)")

    texts = [r["query"] for r in rows]
    vectors = embed_texts(texts, model=args.model, embed_url=args.embed_url)

    feats = score_features(vectors, protos, temp=args.temp)
    y_amb = np.array([1 if r["ambiguous"] else 0 for r in rows], dtype=int)
    expected_subspace = np.array(
        [r["subspace"] if r["subspace"] else "" for r in rows], dtype=object
    )
    argmax = np.array(
        [protos.subspaces[i] for i in np.asarray(feats["sims"]).argmax(axis=1)],
        dtype=object,
    )

    # Grid-search the three thresholds to maximize balanced accuracy; break ties
    # by route accuracy then F1 so a spurious all-ambiguous corner loses.
    margins = np.round(np.arange(0.00, 0.301, 0.02), 4)
    min_top1s = np.round(np.arange(0.05, 0.651, 0.02), 4)
    entropy_maxs = np.round(np.arange(0.40, 1.001, 0.02), 4)

    best: Optional[dict[str, Any]] = None
    best_key = (-1.0, -1.0, -1.0)
    for margin in margins:
        for min_top1 in min_top1s:
            for entropy_max in entropy_maxs:
                cand = evaluate_thresholds(
                    float(margin), float(min_top1), float(entropy_max),
                    feats, y_amb, expected_subspace, argmax,
                )
                metrics = cand["metrics"]
                key = (
                    metrics["balanced_accuracy"],
                    metrics["route_accuracy"],
                    metrics["f1"],
                )
                if key > best_key:
                    best_key = key
                    best = cand

    assert best is not None
    default_eval = evaluate_thresholds(
        DEFAULT_THRESHOLDS["margin"],
        DEFAULT_THRESHOLDS["min_top1"],
        DEFAULT_THRESHOLDS["entropy_max"],
        feats, y_amb, expected_subspace, argmax,
    )

    # Persist thresholds + metrics for the runtime config.
    out_path = _resolve_out(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "thresholds": {
            **best["thresholds"],
            "softmax_temp": args.temp,
        },
        "metrics": best["metrics"],
        "defaults": {
            **DEFAULT_THRESHOLDS,
            "softmax_temp": args.temp,
            "metrics": default_eval["metrics"],
        },
        "n_queries": len(rows),
        "n_ambiguous": n_amb,
        "n_route": len(rows) - n_amb,
        "subspaces": list(SUBSPACES),
        "model": protos.model or args.model,
        "protos": str(args.protos),
        "label_source": str(bench_path) if bench_path else "queries-jsonl",
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    # Console summary table.
    print("\nthresholds                P       R       F1      BAcc    RouteAcc  #amb")
    print("-" * 78)
    for label, ev in (("config default", default_eval), ("calibrated", best)):
        t = ev["thresholds"]
        m = ev["metrics"]
        print(
            f"{label:<12} m={t['margin']:.2f} t1={t['min_top1']:.2f} e={t['entropy_max']:.2f} "
            f"{m['precision']:.3f}  {m['recall']:.3f}  {m['f1']:.3f}  "
            f"{m['balanced_accuracy']:.3f}  {m['route_accuracy']:.3f}   {m['n_ambiguous_pred']}"
        )
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
