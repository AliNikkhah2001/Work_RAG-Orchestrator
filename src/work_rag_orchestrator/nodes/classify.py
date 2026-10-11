"""LLM classifier: greeting vs subspace specification vs question needing retrieve."""
import re, logging
from ..state import RAGState
from ..clients.guardrails import GuardrailsClient
from ..schemas import GuardrailsChatRequest
from ..config import get_settings
log = logging.getLogger(__name__)

# Heuristic fallback when LLM down — mirrors detect_greeting_kind but as 3-way
_GREETING = {"سلام","درود","hello","hi","hey","صبح بخیر","وقت بخیر"}
_SPEC_KEYWORDS = {
  "cheque": ["چک","صیاد","پیچک","برگشت"],
  "facilities_person": ["تسهیلات","وام","قسط"],
  "facilities_legal": ["شرکت","حقوقی","سهامدار"],
  "general": ["معرفی","کاربرد","عمومی"],
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
- specification: کاربر فقط شاخه را مشخص می‌کند (1-4 کلمه شامل چک/تسهیلات/شرکت/عمومی)، مثل "چک" یا "تسهیلات اشخاص حقیقی" یا "شرکت"
- question: سوال واقعی که جواب می‌خواهد، حتی اگر کلمه چک داشته باشد: "چک برگشتی چیست؟" → question

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
        if "چک" in text: state["applied_subspace"]="cheque"
        elif "حقوقی" in text or "شرکت" in text: state["applied_subspace"]="facilities_legal"
        elif "عام" in text or "general" in low: state["applied_subspace"]="general"
        else: state["applied_subspace"]="facilities_person"
    elif cat=="greeting":
        state["greeting_only"]="greeting"
        state["needs_clarification"]=False
    try:
        from ..tracing import trace_span
        trace_span(state["request_id"], "classify", span_input=text, span_output=cat)
    except: pass
    return state
