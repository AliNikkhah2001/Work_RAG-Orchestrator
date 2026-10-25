#!/usr/bin/env python3
"""Evaluate Mizan-v3-3B on the ambiguity resolver dataset.

Sends each query to Mizan System-1 API for subspace classification,
compares against ground truth, computes accuracy, per-subspace F1, latency.

Mizan is a Persian-native System-1 model, so questions are asked in Persian.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("mizan-eval")

SUBSPACES = ["cheque", "facilities_person", "facilities_legal", "general"]
SUBSPACE_FA = {
    "cheque": "گزارش اعتبارسنجی چک (صیاد، برگشتی، نقدشده)",
    "facilities_person": "تسهیلات اشخاص حقیقی (افراد، وام، قسط)",
    "facilities_legal": "تسهیلات اشخاص حقوقی (شرکت‌ها، سهامداران)",
    "general": "پایگاه دانش عمومی (معرفی، هیات مدیره، کاربردها)",
}
LABELS_5CLASS = [*SUBSPACES, "ambiguous"]

MIZAN_URL = "http://127.0.0.1:8083/v1/systemone"
EVAL_FILE = Path("/splunk-data/v1/Work_Credit-RAG_Phase1/components/orchestrator/data/ambiguity_eval.jsonl")
OUTPUT_DIR = Path("/splunk-data/v1/Work_Credit-RAG_Phase1/components/orchestrator/data/mizan_eval")


def load_eval(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ground_truth_label(row: dict) -> str:
    if row.get("ambiguous") or row.get("subspace") is None:
        return "ambiguous"
    return row["subspace"]


def parse_mizan_answer(ans: dict) -> str:
    """Parse Mizan answer to a subspace label or 'ambiguous'."""
    qid = ans.get("question_id", "")
    qtype = ans.get("type", "")

    if qtype == "choice":
        choice = ans.get("choice", "")
        if choice:
            choice_lower = choice.lower()
            for s in SUBSPACES:
                if s in choice_lower or choice_lower in s:
                    return s
            # Also check Persian labels
            for s, fa in SUBSPACE_FA.items():
                if fa in choice or choice in fa:
                    return s
            return choice
    elif qtype == "noul":
        noul = ans.get("noul")
        if noul is False:
            return "ambiguous"
    elif qtype == "score":
        score = ans.get("score", 0)
        if score is not None and score > 5.0:
            return "ambiguous"

    return "unknown"


async def call_mizan(client: httpx.AsyncClient, text: str, request_id: str) -> tuple[str, float]:
    """Call Mizan System-1 API for subspace classification."""
    options = [f"{s} ({SUBSPACE_FA[s]})" for s in SUBSPACES]

    questions = [
        {
            "id": "subspace",
            "question": f"سوال کاربر به کدام یکی از فضاهای دانش زیر تعلق دارد؟ گزینه‌ها: {', '.join(options)}",
            "type": "choice",
            "options": options,
        },
        {
            "id": "ambiguous",
            "question": "آیا این سوال بین چند فضا مبهم است و بدون تعیین فضا نمی‌توان پاسخ داد؟",
            "type": "noul",
        },
        {
            "id": "severity",
            "question": "شدت inquiries از 0 تا 10 ارزیابی کنید.",
            "type": "score",
            "min_value": 0.0,
            "max_value": 10.0,
        },
    ]

    payload = {
        "model": "mizan-v3-3b",
        "record": {"text": text, "stage": "input"},
        "questions": questions,
        "temperature": 0.0,
    }

    headers = {"Authorization": "Bearer sk-local-dev", "X-Request-ID": request_id}

    t0 = time.monotonic()
    try:
        resp = await client.post(MIZAN_URL, json=payload, headers=headers, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        elapsed_ms = (time.monotonic() - t0) * 1000

        ans_map = {a.get("question_id"): a for a in data.get("answers", [])}

        # Get subspace prediction
        subspace_ans = ans_map.get("subspace", {})
        pred = parse_mizan_answer(subspace_ans)

        # Override with ambiguous if noul=true or severity high
        ambig_ans = ans_map.get("ambiguous", {})
        if ambig_ans.get("noul") is True:
            pred = "ambiguous"
        sev_ans = ans_map.get("severity", {})
        if sev_ans.get("score", 0) > 5.0:
            pred = "ambiguous"

        return pred, elapsed_ms

    except Exception as e:
        log.warning("Mizan request failed for %r: %s", text[:50], e)
        elapsed_ms = (time.monotonic() - t0) * 1000
        return "error", elapsed_ms


async def main():
    rows = load_eval(EVAL_FILE)
    labels_true = [ground_truth_label(r) for r in rows]
    queries = [r["query"] for r in rows]

    log.info("Loaded %d eval queries", len(rows))

    preds: list[str] = []
    latencies: list[float] = []

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=5.0, read=30.0, pool=10.0),
        trust_env=False,
    ) as client:
        for i, (q, true_label) in enumerate(zip(queries, labels_true)):
            pred, latency = await call_mizan(client, q, f"eval-{i}")
            preds.append(pred)
            latencies.append(latency)

            if (i + 1) % 10 == 0:
                correct = sum(1 for p, t in zip(preds, labels_true[:i + 1]) if p == t)
                log.info("  %d/%d done, running acc=%.3f", i + 1, len(rows), correct / (i + 1))

    # Compute metrics
    valid = [(t, p) for t, p in zip(labels_true, preds) if p != "error"]
    if not valid:
        log.error("No valid predictions!")
        return

    yt, yp = zip(*valid)
    n_missing = len(rows) - len(valid)

    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, classification_report,
        confusion_matrix, f1_score, precision_score, recall_score,
    )

    labels_present = sorted(set(yt) | set(yp))
    report = classification_report(yt, yp, labels=labels_present, output_dict=True, zero_division=0)

    metrics = {
        "n_total": len(rows),
        "n_valid": len(valid),
        "n_missing": n_missing,
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "precision_macro": float(precision_score(yt, yp, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(yt, yp, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "per_class": {},
        "confusion_matrix": confusion_matrix(yt, yp, labels=labels_present).tolist(),
        "confusion_matrix_labels": labels_present,
        "latency_mean_ms": float(sum(latencies) / len(latencies)),
        "latency_p50_ms": float(sorted(latencies)[len(latencies) // 2]),
        "latency_p95_ms": float(sorted(latencies)[int(len(latencies) * 0.95)]),
    }

    for lab in labels_present:
        if lab in report:
            metrics["per_class"][lab] = {
                "precision": float(report[lab].get("precision", 0.0)),
                "recall": float(report[lab].get("recall", 0.0)),
                "f1": float(report[lab].get("f1-score", report[lab].get("f1", 0.0))),
                "support": int(report[lab].get("support", 0)),
            }

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    results = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "eval_file": str(EVAL_FILE),
            "mizan_url": MIZAN_URL,
            "n_eval": len(rows),
            "labels": LABELS_5CLASS,
        },
        "metrics": metrics,
        "predictions": [
            {"query": q, "true": t, "pred": p}
            for q, t, p in zip(queries, labels_true, preds)
        ],
    }

    json_path = OUTPUT_DIR / f"mizan_eval_{ts}.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("MIZAN EVALUATION RESULTS")
    print("=" * 60)
    print(f"N total:       {metrics['n_total']}")
    print(f"N valid:       {metrics['n_valid']}")
    print(f"N missing:     {metrics['n_missing']}")
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    print(f"Balanced Acc:  {metrics['balanced_accuracy']:.4f}")
    print(f"F1 Macro:      {metrics['f1_macro']:.4f}")
    print(f"Latency mean:  {metrics['latency_mean_ms']:.1f}ms")
    print(f"Latency P50:   {metrics['latency_p50_ms']:.1f}ms")
    print(f"Latency P95:   {metrics['latency_p95_ms']:.1f}ms")
    print("\nPer-class F1:")
    for lab, m in metrics["per_class"].items():
        print(f"  {lab:20s} F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} n={m['support']}")
    print(f"\nSaved to: {json_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
