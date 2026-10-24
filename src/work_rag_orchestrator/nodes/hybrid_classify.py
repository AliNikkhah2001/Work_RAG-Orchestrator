"""Thresholded Hybrid Classifier + Gemma node.

Replaces pure heuristic/LLM subspace classification with a calibrated hybrid:

1. Run trained SVM-RBF classifier (local, ~0.17ms) on the query
2. If classifier confidence >= HYBRID_CLASSIFIER_CONFIDENCE_THRESHOLD:
     → accept classifier's subspace label (fast path)
3. Else → escalate to Gemma-4-31B TRT-LLM via Guardrails (slow path, ~128ms)
     → accept Gemma's subspace label

This combines classifier speed with Gemma reasoning for uncertain cases,
measured at 78% accuracy (vs 73% classifier alone, 72% Gemma alone)
at threshold 0.60 with 42% escalation rate and 53.9ms mean latency.

Sets state fields:
  - applied_subspace: final subspace label (cheque/facilities_person/facilities_legal/general/ambiguous)
  - hybrid_classifier_source: "classifier" | "gemma" | "fallback"
  - hybrid_classifier_confidence: float in [0,1]
  - hybrid_classifier_escalated: bool
  - ambiguity_*: from SubspaceScorer (audit trail)
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..state import RAGState, record_timing
from ..clients.guardrails import GuardrailsClient
from ..schemas import GuardrailsChatRequest
from ..config import get_settings
from ..subspaces import get_scorer, SUBSPACES

# Ensure scripts directory is on path for build_subspace_protos import
# File is at src/work_rag_orchestrator/nodes/hybrid_classify.py
# scripts dir is at components/orchestrator/scripts/ (3 levels up from nodes/)
_SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from build_subspace_protos import embed_texts, score_features  # noqa: E402

log = logging.getLogger(__name__)

# Persian subspace classification prompt for Gemma TRT-LLM
HYBRID_GEMMA_PROMPT = """شما یک klasyfikاتور فضاء‌های دانش برای سامانه اعتبارسنجی ایران هستید.
سوال کاربر را به یکی از این فضاها वर्ग کنید:

- cheque: گزارش اعتباری چک (صیاد، برگشتی، نقدشده، Награده چک)
- facilities_person: تسهیلات اشخاص حقیقی (افراد، وام، قسط، اعتبارسنجی افراد)
- facilities_legal: تسهیلات اشخاص حقوقی (شرکت‌ها، سهامداران، اعتبارسنجی شرکت‌ها)
- general: پایگاه دانش عمومی (معرفی شرکت، هیات مدیره، کاربردها)
- ambiguous: سوال مبهم که نمی‌توانwithout specifying الفضاء تعیین کرد

فقط نام فضا را بنویسید (cheque، facilities_person، facilities_legal، general، ambiguous).

سوال: {text}

فاصله:"""


async def hybrid_classify_node(state: RAGState) -> RAGState:
    """Run trained SVM-RBF classifier, escalate to Gemma TRT if confidence low."""
    t0 = time.monotonic()
    request_id = state["request_id"]
    text = (state.get("query") or state.get("messages", [{}])[-1].get("content", "")).strip()

    settings = get_settings()

    # Default outputs
    state["hybrid_classifier_source"] = "fallback"
    state["hybrid_classifier_confidence"] = 0.0
    state["hybrid_classifier_escalated"] = False
    state["applied_subspace"] = None

    if not text:
        record_timing(state, "hybrid_classify", round((time.monotonic() - t0) * 1000, 1))
        return state

    # Skip hybrid if disabled — leave classify's heuristic result in place
    if not settings.hybrid_classifier_enabled:
        record_timing(state, "hybrid_classify", round((time.monotonic() - t0) * 1000, 1))
        return state

    # ── Step 1: Trained SVM-RBF classifier (local, ~0.17ms) ──
    try:
        scorer = get_scorer()
        vectors = embed_texts([text], model=settings.embed_model)

        feats = score_features(vectors, scorer.protos, temp=settings.subspace_softmax_temp)

        cols = [feats["sims"]]
        cols.append(feats["margin"].reshape(-1, 1))
        cols.append(feats["top1"].reshape(-1, 1))
        cols.append(feats["entropy"].reshape(-1, 1))
        X = np.concatenate(cols, axis=1).astype(np.float32)

        # Load the trained classifier
        import joblib

        clf_path = settings.subspace_clf_path
        obj = joblib.load(clf_path)
        clf = obj["model"]
        classes = list(obj["classes"])

        pred = clf.predict(X)[0]
        pred_label = str(pred)
        proba = clf.predict_proba(X)[0]
        confidence = float(np.max(proba))

        state["hybrid_classifier_source"] = "classifier"
        state["hybrid_classifier_confidence"] = confidence
        state["applied_subspace"] = pred_label

        # Also record ambiguity signals from the scorer for audit
        state["ambiguity_score"] = float(feats.get("ambiguity_score", 0.0)) if isinstance(feats, dict) else 0.0
        state["ambiguity_margin"] = float(feats["margin"][0])
        state["ambiguity_entropy"] = float(feats["entropy"][0])
        state["ambiguity_top1"] = float(feats["top1"][0])
        state["ambiguity_source"] = "classifier"

        log.info(
            "Hybrid classify [%s]: classifier=%s conf=%.3f threshold=%.2f",
            request_id, pred_label, confidence, settings.hybrid_classifier_confidence_threshold,
        )

    except Exception as e:
        log.warning("Hybrid classifier failed for request %s: %s", request_id, e)
        confidence = 0.0
        pred_label = None

    # ── Step 2: Escalate to Gemma TRT-LLM if confidence below threshold ──
    if (
        settings.hybrid_classifier_enabled
        and confidence < settings.hybrid_classifier_confidence_threshold
    ):
        state["hybrid_classifier_escalated"] = True

        try:
            async with GuardrailsClient() as client:
                req = GuardrailsChatRequest(
                    model=settings.hybrid_classifier_gemma_model,
                    messages=[
                        {
                            "role": "user",
                            "content": HYBRID_GEMMA_PROMPT.format(text=text[:500]),
                        }
                    ],
                    max_tokens=settings.hybrid_classifier_gemma_max_tokens,
                    temperature=settings.hybrid_classifier_gemma_temperature,
                )
                resp = await client.chat_completion(req, request_id)
                ans = resp.choices[0].message.get("content", "").strip().lower() if resp.choices else ""

                # Parse Gemma's subspace label
                gemma_label = None
                for label in SUBSPACES:
                    if label in ans:
                        gemma_label = label
                        break
                # Also check Persian variants
                if gemma_label is None:
                    low = ans
                    if "چک" in low or "صیاد" in low or "برگشتی" in low:
                        gemma_label = "cheque"
                    elif "تسهیلات" in low and ("حقوقی" in low or "شرکت" in low or "سهامدار" in low):
                        gemma_label = "facilities_legal"
                    elif "تسهیلات" in low and ("حقیقی" in low or "فردی" in low or "افراد" in low):
                        gemma_label = "facilities_person"
                    elif "عمومی" in low or "معرفی" in low or "هیات مدیره" in low or "کاربرد" in low:
                        gemma_label = "general"

                if gemma_label:
                    state["applied_subspace"] = gemma_label
                    state["hybrid_classifier_source"] = "gemma"
                    state["hybrid_classifier_confidence"] = 0.0  # Gemma doesn't give calibrated probs here
                    log.info(
                        "Hybrid classify [%s]: escalated to Gemma, label=%s raw=%r",
                        request_id, gemma_label, ans[:80],
                    )
                else:
                    log.warning("Hybrid classify [%s]: Gemma unparseable %r, keeping classifier", request_id, ans[:80])

        except Exception as e:
            log.warning("Hybrid Gemma escalation failed for request %s: %s", request_id, e)
            # Keep classifier result on Gemma failure

    record_timing(state, "hybrid_classify", round((time.monotonic() - t0) * 1000, 1))
    return state
