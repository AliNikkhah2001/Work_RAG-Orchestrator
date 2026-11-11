"""LLM classifier: greeting vs subspace specification vs question needing retrieve."""
import re, logging
from ..state import RAGState
from ..clients.guardrails import GuardrailsClient
from ..schemas import GuardrailsChatRequest
from ..config import get_settings
log = logging.getLogger(__name__)

# T5 scattered helper (reuse infer_specification logic to avoid duplication)
try:
    from .infer_specification import is_scattered_ambiguous as _is_scattered
except Exception:  # pragma: no cover
    _is_scattered = lambda q: False  # type: ignore

# Heuristic fallback when LLM down — mirrors detect_greeting_kind but as 3-way
_GREETING = {"سلام","درود","hello","hi","hey","صبح بخیر","وقت بخیر"}
_SPEC_KEYWORDS = {
  "cheque": ["چک","صیاد","پیچک","برگشت"],
  "facilities_person": ["تسهیلات","وام","قسط"],
  "facilities_legal": ["حقوقی","سهامدار"],
  "credit_ladder": ["نردبان"],
  "guarantor": ["کفیل","کفالت"],
  "supplier_legal": ["تامین کننده","تامین‌کننده"],
  "customers_person": ["مشتریان من","پنل کسب"],
  "customers_legal": ["مشتریان من حقوقی"],
  "general": ["معرفی","کاربرد","عمومی","عام","مدیر","هیات مدیره","هیئت مدیره","کاربردها","شرکت"],
}

# Patterns that indicate an entity lookup question (person/company name)
# even without explicit question words
_ENTITY_LOOKUP_PATTERNS = [
    r"^[\u0600-\u06FF\s]{3,20}$",  # Persian name-like (3-20 chars, Persian letters/spaces)
]
def _looks_like_entity_lookup(text: str) -> bool:
    """Check if text looks like a bare entity name (person/company) without question words."""
    t = text.strip()
    # Has question words -> definitely a question
    if any(w in t for w in ["چیست","چطور","چگونه","آیا","چرا","کیه","کجاست","کیست"]):
        return True
    # Short (2-4 words), no subspace keywords, looks like a name
    words = t.split()
    if 2 <= len(words) <= 4:
        has_spec_kw = any(kw in t for lst in _SPEC_KEYWORDS.values() for kw in lst)
        if not has_spec_kw:
            # Looks like a Persian name (no Latin, no digits, reasonable length)
            if all('\u0600' <= ch <= '\u06FF' or ch in ' \u200c' for ch in t):
                return True
    return False

def _heuristic_classify(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"\.+$","",t).strip()
    for ch in "؟?!.,،؛:\"'«»()…-": t=t.replace(ch,"")
    t = " ".join(t.replace("‌"," ").split())
    if t in _GREETING: return "greeting"
    # specification: short (<=4 words) + subspace keyword, no question mark
    if len(t.split()) <=4 and any(kw in t for lst in _SPEC_KEYWORDS.values() for kw in lst) and "؟" not in text and "?" not in text and not any(w in text for w in ["چیست","چطور","چگونه","آیا","چرا"]):
        return "specification"
    return "question"

def _classify_prompt(text: str) -> str:
    from ..content_registry import require_prompt

    return require_prompt("classify", text=text[:500])

async def classify(state: RAGState) -> RAGState:
    text = (state.get("query") or state.get("messages",[{}])[-1].get("content","")).strip()
    if not text:
        state["classify"]="question"
        return state
    # Heuristic: bare entity names (person/company) without question words -> question
    if _looks_like_entity_lookup(text):
        cat = "question"
    else:
        # Try LLM
        try:
            settings=get_settings()
            async with GuardrailsClient() as client:
                req=GuardrailsChatRequest(model=settings.upstream_llm_model, messages=[{"role":"user","content":_classify_prompt(text)}], max_tokens=16, temperature=0.0)
                resp=await client.chat_completion(req, state["request_id"])
                ans=resp.choices[0].message.get("content","").strip().lower() if resp.choices else ""
                if "greeting" in ans: cat="greeting"
                elif "specification" in ans: cat="specification"
                elif "question" in ans: cat="question"
                else: cat=_heuristic_classify(text)
        except Exception as e:
            log.warning("classify LLM failed: %s", e)
            cat=_heuristic_classify(text)
    state["classify"]=cat
    # If specification, also set the subspace label
    if cat=="specification":
        low=text.lower()
        has_facility_kw = "تسهیلات" in text or "وام" in text or "قسط" in text
        has_cheque_kw = "چک" in text
        is_general_kw = any(kw in text for kw in ["عام","عمومی","معرفی","مدیر","هیات مدیره","هیئت مدیره","کاربرد"]) or "general" in low
        if "نردبان" in text:
            state["applied_subspace"]="credit_ladder"
        elif any(kw in text for kw in ["کفیل","کفالت","وجه‌الکفاله"]):
            state["applied_subspace"]="guarantor"
        elif "تامین" in text:
            state["applied_subspace"]="supplier_legal"
        elif "مشتریان من" in text or "پنل" in text:
            state["applied_subspace"]="customers_legal" if ("حقوقی" in text or "شرکت" in text) else "customers_person"
        elif has_cheque_kw:
            state["applied_subspace"]="cheque"
        elif is_general_kw:
            state["applied_subspace"]="general"
        elif "شخصی" in text or "حقیقی" in text:
            state["applied_subspace"]="facilities_person"
        elif "شرکتی" in text or "حقوقی" in text or ("شرکت" in text and has_facility_kw):
            state["applied_subspace"]="facilities_legal"
        elif "شرکت" in text and not has_facility_kw and not has_cheque_kw:
            # "شرکت" alone without تسهیلات/چک → عمومی (معرفی شرکت)
            state["applied_subspace"]="general"
        else:
            state["applied_subspace"]="facilities_person"
    # Also set subspace for question that is clearly general (person name + کیه)
    if cat=="question":
        low2 = text.lower()
        if "قاسم" in text or "بور" in text or "هیات مدیره" in text or "هیئت مدیره" in text:
            state["applied_subspace"] = "general"
        # T5 scattered / cross-product impact → clarify (do not sticky-commit)
        try:
            from ..subspace_routing import L2_SUBSPACES, is_cross_product_impact
            from ..content_registry import clarify_option_labels

            if is_cross_product_impact(text):
                state["applied_subspace"] = None
                state["needs_clarification"] = True
                state["available_subspaces"] = clarify_option_labels() or list(L2_SUBSPACES)
                state["clarifying_question"] = None
                log.info("classify cross-product impact -> clarify for query=%r", text[:80])
            elif not state.get("applied_subspace") and _is_scattered(text):
                state["needs_clarification"] = True
                state["available_subspaces"] = clarify_option_labels() or list(L2_SUBSPACES)
                state["clarifying_question"] = None
                log.info("classify scattered ambiguous -> pre-set clarify for query=%r", text[:80])
        except Exception:
            pass
    elif cat=="greeting":
        state["greeting_only"]="greeting"
        state["needs_clarification"]=False
    try:
        from ..tracing import trace_span
        trace_span(state["request_id"], "classify", span_input=text, span_output=cat)
    except: pass
    return state
