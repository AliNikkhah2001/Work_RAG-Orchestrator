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
  "general": ["معرفی","کاربرد","عمومی","عام","مدیر","هیات مدیره","هیئت مدیره","کاربردها","شرکت"],
}
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

CLASSIFY_PROMPT = """دسته‌بندی کن — فقط یک کلمه: greeting / specification / question
- greeting: سلام، تشکر، خداحافظی، تعارف کوتاه بدون نیاز به پایگاه دانش
- specification: کاربر فقط شاخه را مشخص می‌کند (1-4 کلمه شامل چک/تسهیلات/شرکت/عمومی)، مثل "چک" یا "تسهیلات اشخاص حقیقی" یا "شرکت" یا "عمومی"
  شاخه‌ها: 1. تسهیلات افراد 2. گزارش چک 3. تسهیلات شرکت‌ها 4. عمومی (معرفی، هیات مدیره، کاربردها)
  نکته: "مدیر"، "هیات مدیره"، "هیئت مدیره"، "معرفی شرکت" و سوال درباره اشخاص (مثلا "قاسم بور کیه") → عمومی است نه تسهیلات. "شرکت" تنها بدون تسهیلات/چک → عمومی؛ "شرکت" + تسهیلات/حقوقی → تسهیلات شرکت‌ها
- question: سوال واقعی که جواب می‌خواهد، حتی اگر کلمه چک داشته باشد: "چک برگشتی چیست؟" → question، "قاسم بور کیه" → question (درباره شخص)

متن: "{text}"
فقط بنویس: greeting یا specification یا question"""

async def classify(state: RAGState) -> RAGState:
    text = (state.get("query") or state.get("messages",[{}])[-1].get("content","")).strip()
    if not text:
        state["classify"]="question"
        return state
    # Try LLM
    try:
        settings=get_settings()
        async with GuardrailsClient() as client:
            req=GuardrailsChatRequest(model=settings.upstream_llm_model, messages=[{"role":"user","content":CLASSIFY_PROMPT.format(text=text[:500])}], max_tokens=16, temperature=0.0)
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
        if has_cheque_kw:
            state["applied_subspace"]="cheque"
        elif is_general_kw:
            state["applied_subspace"]="general"
        elif "حقوقی" in text or ("شرکت" in text and has_facility_kw):
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
        # T5 scattered gate: vague query without subspace keywords => mark ambiguous for retrieve/graph
        # Do not override already-set general sticky; only mark when no subspace decided
        try:
            if not state.get("applied_subspace") and _is_scattered(text):
                state["needs_clarification"] = True
                state["available_subspaces"] = ["cheque", "facilities_person", "facilities_legal", "general"]
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
