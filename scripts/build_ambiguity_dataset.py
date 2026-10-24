#!/usr/bin/env python3
"""Build a labeled ambiguity dataset for the subspace resolver.

Pipeline
--------
1. **Extract** every real `qa_pair` question from the KB, labeling it by the
   subspace of its document's `folder_hierarchy` (ground truth from the folders).
2. **Judge** each question with the local LLM (Gemma via the manager) to confirm
   it truly belongs to that subspace (rejects cross-subspace drift).
3. **Top up** each subspace to the target quota with LLM-generated questions,
   each judged before acceptance.
4. **Generate + judge** genuinely ambiguous questions (no single subspace).
5. **Sophisticated synthetic generation** - negation patterns, aliases, paraphrases,
   cross-subspace ambiguous, negation with correction, varied question types.

Output (JSONL, one object per line):

    {"id": ..., "query": ..., "subspace": "cheque"|..., "ambiguous": false,
     "source": "kb"|"generated", "verdict": ..., "created_at": ...}

Ambiguous rows carry `"subspace": null, "ambiguous": true`.

Usage
-----
    python scripts/build_ambiguity_dataset.py                 # full run (200/subspace + 200 ambiguous = 1000)
    python scripts/build_ambiguity_dataset.py --total-samples 1000 --concurrency 2
    python scripts/build_ambiguity_dataset.py --per-subspace 5 --ambiguous 5 --dry-run
    python scripts/build_ambiguity_dataset.py --resume        # keep existing rows, fill gaps
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

ORCH_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ORCH_ROOT / "data" / "ambiguity_dataset.jsonl"
DEFAULT_DB_URL = "postgresql://postgres:postgres@127.0.0.1:5433/kb_manager"
DEFAULT_LLM_URL = "http://127.0.0.1:9000/v1/chat/completions"
DEFAULT_MODEL = "gemma-4-31b"

SUBSPACES = ("cheque", "facilities_person", "facilities_legal", "general")

_QUESTION_RE = re.compile(r"سوال\s*[:：]\s*(.*?)\s*(?:پاسخ|$)", re.DOTALL)
_WS_RE = re.compile(r"\s+")

# ---------------------------------------------------------------------------
# Subspace mapping (folder_hierarchy -> subspace)
# ---------------------------------------------------------------------------
def subspace_of(folder_hierarchy) -> str:
    """Map a KB `folder_hierarchy` list to one of the four subspaces."""
    s = " / ".join(folder_hierarchy or [])
    if "چک" in s:
        return "cheque"
    if "حقوقی" in s:
        return "facilities_legal"
    if "حقیقی" in s:
        return "facilities_person"
    if "عمومی" in s:
        return "general"
    return "facilities_person"


def normalize(q: str) -> str:
    """Normalize a question for dedup / comparison."""
    q = (q or "").replace("\u200c", " ").replace("ي", "ی").replace("ك", "ک")
    return _WS_RE.sub(" ", q).strip()


# ---------------------------------------------------------------------------
# Subspace aliases and keywords for sophisticated generation
# ---------------------------------------------------------------------------
_SUBSPACE_FA = {
    "cheque": "گزارش اعتبارسنجی چک",
    "facilities_person": "اعتبارسنجی تسهیلات برای اشخاص حقیقی (افراد)",
    "facilities_legal": "اعتبارسنجی تسهیلات برای اشخاص حقوقی (شرکت‌ها)",
    "general": "پایگاه دانش عمومی (معرفی شرکت، هیئت‌مدیره، کاربردها)",
}

# Primary keywords per subspace (for detection)
_SUBSPACE_PRIMARY_KW = {
    "cheque": ("چک", "صیاد", "پیچک", "برگشتی چک", "امتیاز چک", "رتبه چک", "گزارش چک"),
    "facilities_person": ("تسهیلات اشخاص حقیقی", "تسهیلات افراد", "وام اشخاص حقیقی", "اعتبارسنجی افراد", "اعتبار سنجی اشخاص حقیقی"),
    "facilities_legal": ("تسهیلات اشخاص حقوقی", "تسهیلات شرکت‌ها", "اعتبارسنجی شرکت‌ها", "وام اشخاص حقوقی", "اعتبار سنجی اشخاص حقوقی"),
    "general": ("معرفی شرکت", "هیات مدیره", "کاربردها", "عمومی", "طبیعت شرکت"),
}

# Extended aliases per subspace (for generation variety)
_SUBSPACE_ALIASES = {
    "cheque": [
        "چک", "صیاد", "پیچک", "برگشتی چک", "امتیاز چک", "رتبه چک", "گزارش چک",
        "گزارش اعتباری چک", "گزارش اعتبارسنجی چک", "وضعیت چک", "بازگشت چک"
    ],
    "facilities_person": [
        "تسهیلات اشخاص حقیقی", "تسهیلات افراد", "وام اشخاص حقیقی",
        "اعتبارسنجی افراد", "اعتبار سنجی اشخاص حقیقی", "تسهیلات شخصی",
        "وام فردی", "اعتبار فردی", "بازپرداخت تسهیلات", "قسط تسهیلات"
    ],
    "facilities_legal": [
        "تسهیلات اشخاص حقوقی", "تسهیلات شرکت‌ها", "اعتبارسنجی شرکت‌ها",
        "وام اشخاص حقوقی", "اعتبار سنجی اشخاص حقوقی", "تسهیلات شرکتی",
        "اعتبار شرکت", "وام تجاری", "سهامدار", "هیات مدیره شرکت"
    ],
    "general": [
        "معرفی شرکت", "هیات مدیره", "کاربردها", "عمومی", "طبیعت شرکت",
        "درباره شرکت", "شرکت چیست", "فعالیت شرکت", "خدمات شرکت"
    ],
}

# Question type templates
_QUESTION_TYPES = {
    "factual": {
        "cheque": ["چک {alias} چیست؟", "تعریف {alias} را توضیح دهید.", "{alias} چه معنایی دارد؟"],
        "facilities_person": ["{alias} چه ویژگی‌هایی دارد؟", "شرایط {alias} چیست؟", "تعریف {alias} را بیان کنید."],
        "facilities_legal": ["{alias} شامل چه مواردی است؟", "معیارهای {alias} چیستند؟", "تعریف {alias} چگونه است؟"],
        "general": ["{alias} به چه معنی است؟", "مفهوم {alias} را توضیح دهید.", "تعریف {alias} چیست؟"],
    },
    "procedural": {
        "cheque": ["چگونه {alias} را دریافت کنیم؟", "مراحل درخواست {alias} چیست؟", "روش查询 {alias} چگونه است؟"],
        "facilities_person": ["نحوه درخواست {alias} چگونه است؟", "مراحل دریافت {alias} چیست؟", "چطور می‌توان {alias} گرفت؟"],
        "facilities_legal": ["فرآیند {alias} چگونه است؟", "مراحل انجام {alias} را توضیح دهید.", "نحوه ثبت {alias} چیست؟"],
        "general": ["چگونه به {alias} دسترسی پیدا کنیم؟", "مراحل {alias} چیست؟", "روش انجام {alias} چگونه است؟"],
    },
    "hypothetical": {
        "cheque": ["اگر {alias} برگشتی شود چه می‌شود؟", "فرض کنید {alias} مشکلی داشته باشد، راهکار چیست؟", "در صورت بروز خطا در {alias} چه کنید؟"],
        "facilities_person": ["اگر در {alias} دیرکرد رخ دهد چه اتفاقی می‌افتد؟", "فرض کنید {alias} رد شود، دلیل چیست؟", "در صورت عدم پرداخت {alias} چه پیش می‌آید؟"],
        "facilities_legal": ["اگر شرکت در {alias} مشکلی داشته باشد؟", "فرض کنید {alias} لغو شود، پیامدها چیستند؟", "در صورت تغییر در {alias} چه باید کرد؟"],
        "general": ["اگر {alias} تغییر کند چه می‌شود؟", "فرض کنید {alias} موجود نباشد، جایگزین چیست؟", "در صورت بروز مشکل در {alias} راهکار چیست؟"],
    },
    "comparative": {
        "cheque": ["تفاوت {alias} با سایر گزارش‌ها چیست؟", "{alias} چگونه از موارد مشابه متمایز است؟", "مقایسه {alias} با موارد دیگر"],
        "facilities_person": ["تفاوت {alias} با تسهیلات حقوقی چیست؟", "{alias} نسبت به وام‌های دیگر چه مزایایی دارد؟", "مقایسه {alias} با تسهیلات شرکت‌ها"],
        "facilities_legal": ["تفاوت {alias} با تسهیلات أفراد چیست؟", "{alias} چه مزایایی نسبت به تسهیلات شخصی دارد؟", "مقایسه {alias} با وام‌های فردی"],
        "general": ["تفاوت {alias} با سایر بخش‌ها چیست؟", "{alias} چگونه از موارد عمومی متمایز است؟", "مقایسه {alias} با کلیه خدمات"],
    },
}

# Cross-subspace ambiguous patterns (mix keywords from multiple subspaces)
_CROSS_SUBSPACE_PATTERNS = [
    ("cheque", "facilities_person", "وضعیت {kw1} و {kw2} من چطور است؟"),
    ("cheque", "facilities_legal", "گزارش {kw1} برای {kw2} چه می‌گوید؟"),
    ("facilities_person", "facilities_legal", "تفاوت {kw1} و {kw2} چیست؟"),
    ("cheque", "general", "آیا {kw1} مربوط به {kw2} است؟"),
    ("facilities_person", "general", "چگونه {kw1} در {kw2} اعمال می‌شود؟"),
    ("facilities_legal", "general", "مشخصات {kw1} در {kw2} چیست؟"),
]

# Negation with correction patterns
_NEGATION_CORRECTION_PATTERNS = [
    "منظورم {kw1} نبود، {kw2} بود.",
    "نه {kw1}، می‌خواستم بپرسم در مورد {kw2}.",
    "{kw1} نه، {kw2} منظورم است.",
    "اشتباه پرسیدم درباره {kw1}، منظور {kw2} بود.",
    "نه {kw1}، سوال من در مورد {kw2} است.",
]


def _has_any_subspace_kw(q: str) -> bool:
    ql = normalize(q)
    return any(kw in ql for kws in _SUBSPACE_PRIMARY_KW.values() for kw in kws)


def _has_other_subspace_kw(q: str, subspace: str) -> bool:
    ql = normalize(q)
    for name, kws in _SUBSPACE_PRIMARY_KW.items():
        if name == subspace:
            continue
        if any(kw in ql for kw in kws):
            return True
    return False


def _split_lines(text: str) -> list[str]:
    """Extract clean question lines from an LLM generation block."""
    out = []
    for line in (text or "").splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line.strip())
        line = normalize(line)
        if len(line) < 6 or ("؟" not in line and "?" not in line):
            continue
        out.append(line)
    return out


def _parse_bool_array(text: str) -> list[bool]:
    """Parse a JSON-ish boolean array from a judge response."""
    m = re.search(r"\[[^\[\]]*\]", text or "", re.DOTALL)
    if not m:
        return []
    body = m.group(0)
    out = []
    for tok in re.findall(r"true|false|بله|خیر|yes|no|1|0", body, re.IGNORECASE):
        out.append(tok.lower() in ("true", "بله", "yes", "1"))
    return out


# ---------------------------------------------------------------------------
# Step 1: extract real Q&A questions from the KB
# ---------------------------------------------------------------------------
async def extract_kb_questions(db_url: str) -> dict[str, list[str]]:
    """Return {subspace: [question, ...]} from the KB's qa_pair chunks."""
    import asyncpg

    out: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    conn = await asyncpg.connect(db_url)
    try:
        rows = await conn.fetch(
            "SELECT c.content, d.metadata FROM chunks c "
            "JOIN documents d ON c.document_id = d.id WHERE c.chunk_type = 'qa_pair'"
        )
    finally:
        await conn.close()

    for r in rows:
        md = r["metadata"]
        if isinstance(md, str):
            md = json.loads(md)
        sub = subspace_of((md or {}).get("folder_hierarchy"))
        m = _QUESTION_RE.search(r["content"] or "")
        q = normalize(m.group(1)) if m else ""
        if not q or len(q) < 6 or q in seen:
            continue
        seen.add(q)
        out[sub].append(q)
    return dict(out)


# ---------------------------------------------------------------------------
# LLM client (Gemma via the manager)
# ---------------------------------------------------------------------------
class Judge:
    """Async LLM judge / generator with a bounded connection pool."""

    def __init__(
        self,
        url: str,
        model: str,
        concurrency: int = 2,
        timeout: float = 120.0,
        judge_batch: int = 10,
        gen_batch: int = 15,
    ):
        self.url = url
        self.model = model
        self.sem = asyncio.Semaphore(concurrency)
        self.client = httpx.AsyncClient(timeout=timeout, trust_env=False)
        self.judge_batch = judge_batch
        self.gen_batch = gen_batch

    async def close(self):
        await self.client.aclose()

    async def _chat(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, retries: int = 4) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max(16, min(max_tokens, 2048)),
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        last: Exception | None = None
        for attempt in range(retries):
            try:
                async with self.sem:
                    r = await self.client.post(self.url, json=payload)
                    r.raise_for_status()
                    data = r.json()
                return (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
            except Exception as e:  # transient 502/timeout — back off and retry
                last = e
                await asyncio.sleep(1.5 * (attempt + 1))
        raise last if last else RuntimeError("judge call failed")

    # -- generation ------------------------------------------------------
    async def generate(self, subspace: str, n: int, examples: list[str]) -> list[str]:
        """Generate n natural Persian questions for a subspace (batched)."""
        out: list[str] = []
        while len(out) < n:
            batch = min(self.gen_batch, n - len(out))
            ex = "\n".join(f"- {q}" for q in examples[:5])
            prompt = (
                f"شما یک تولیدکنندهٔ پرسش فارسی هستید. {batch} پرسش طبیعی و متنوع بنویسید که "
                f"قطعاً مربوط به حوزهٔ «{_SUBSPACE_FA[subspace]}» در سامانهٔ اعتبارسنجی ایران باشد.\n"
                f"نمونه‌های همان حوزه:\n{ex}\n\n"
                f"قواعد: هر پرسش یک خط، بدون شماره‌گذاری، بدون توضیح، فقط متن پرسش. "
                f"پرسش‌ها باید فقط با کلیدواژه‌های همین حوزه ساخته شوند و به حوزه‌های دیگر اشاره نکنند."
            )
            text = await self._chat(prompt, max_tokens=48 * batch, temperature=0.9)
            got = _split_lines(text)[:batch]
            if not got:
                break
            out.extend(got)
        return out[:n]

    async def generate_ambiguous(self, n: int) -> list[str]:
        """Generate n genuinely ambiguous questions (batched)."""
        out: list[str] = []
        while len(out) < n:
            batch = min(self.gen_batch, n - len(out))
            prompt = (
                f"شما یک تولیدکنندهٔ پرسش فارسی هستید. {batch} پرسش «مبهم» بنویسید که به‌طور هم‌زمان "
                f"می‌تواند به چند حوزهٔ مختلف مربوط باشد و بدون مشخص‌کردن حوزه نتوان پاسخ داد.\n"
                f"حوزه‌ها: تسهیلات افراد (اشخاص حقیقی)، گزارش چک، تسهیلات شرکت‌ها (اشخاص حقوقی)، عمومی.\n"
                f"مثال: «وضعیت اعتباری من چطور است؟» یا «امتیازم چنده؟»\n\n"
                f"قواعد: هر پرسش یک خط، بدون شماره‌گذاری، بدون توضیح، و هیچ کلیدواژهٔ حوزه‌ای "
                f"(چک، تسهیلات، حقوقی، حقیقی، شرکت، معرفی، هیئت‌مدیره) نداشته باشد."
            )
            text = await self._chat(prompt, max_tokens=48 * batch, temperature=0.95)
            got = _split_lines(text)[:batch]
            if not got:
                break
            out.extend(got)
        return out[:n]

    # -- sophisticated generation techniques -----------------------------
    async def generate_with_aliases(self, subspace: str, n: int, examples: list[str]) -> list[str]:
        """Generate questions using alternative names/aliases for the subspace."""
        aliases = _SUBSPACE_ALIASES.get(subspace, [])
        if not aliases:
            return []

        out: list[str] = []
        while len(out) < n:
            batch = min(self.gen_batch, n - len(out))
            # Pick random aliases for variety
            selected_aliases = random.sample(aliases, min(len(aliases), batch * 2))
            alias_prompt = "، ".join(selected_aliases[:10])

            ex = "\n".join(f"- {q}" for q in examples[:3])
            prompt = (
                f"شما یک تولیدکنندهٔ پرسش فارسی هستید. {batch} پرسش طبیعی بنویسید که "
                f"مربوط به حوزهٔ «{_SUBSPACE_FA[subspace]}» باشد و از نام‌های جایگزین زیر استفاده کند:\n"
                f"نام‌های جایگزین: {alias_prompt}\n\n"
                f"نمونه‌های حوزه:\n{ex}\n\n"
                f"قواعد: هر پرسش یک خط، بدون شماره‌گذاری، فقط متن پرسش. "
                f"باید حداقل یکی از نام‌های جایگزین را در پرسش استفاده کنید. "
                f"به حوزه‌های دیگر اشاره نکنید."
            )
            text = await self._chat(prompt, max_tokens=50 * batch, temperature=0.85)
            got = _split_lines(text)[:batch]
            if not got:
                break
            out.extend(got)
        return out[:n]

    async def generate_question_types(self, subspace: str, n: int, examples: list[str]) -> list[str]:
        """Generate questions of different types (factual, procedural, hypothetical, comparative)."""
        templates = _QUESTION_TYPES
        aliases = _SUBSPACE_ALIASES.get(subspace, [_SUBSPACE_PRIMARY_KW[subspace][0]])

        out: list[str] = []
        type_names = list(templates.keys())
        questions_per_type = max(1, n // len(type_names))

        for q_type in type_names:
            type_need = min(questions_per_type, n - len(out))
            if type_need <= 0:
                break

            batch = min(self.gen_batch, type_need)
            selected_aliases = random.sample(aliases, min(len(aliases), batch))
            type_templates = templates[q_type].get(subspace, [])

            if not type_templates:
                continue

            ex = "\n".join(f"- {q}" for q in examples[:3])
            prompt = (
                f"شما یک تولیدکنندهٔ پرسش فارسی هستید. {batch} پرسش از نوع «{q_type}\" بنویسید "
                f"برای حوزهٔ «{_SUBSPACE_FA[subspace]}» با استفاده از این الگوهای کلیدی:\n"
                f"الگوهای نوع {q_type}: {' | '.join(type_templates[:3])}\n"
                f"نام‌های کلیدی: {', '.join(selected_aliases[:5])}\n\n"
                f"نمونه‌های حوزه:\n{ex}\n\n"
                f"قواعد: هر پرسش یک خط، بدون شماره‌گذاری، فقط متن پرسش. "
                f"پرسش‌ها باید به حوزهٔ {subspace} منحصر باشند."
            )
            text = await self._chat(prompt, max_tokens=50 * batch, temperature=0.85)
            got = _split_lines(text)[:batch]
            if not got:
                continue
            out.extend(got)

        return out[:n]

    async def generate_negation_patterns(self, subspace: str, n: int, examples: list[str]) -> list[str]:
        """Generate negation patterns like 'not cheque, but facilities_person'."""
        other_subspaces = [s for s in SUBSPACES if s != subspace]
        out: list[str] = []

        while len(out) < n:
            batch = min(self.gen_batch, n - len(out))
            other_sub = random.choice(other_subspaces)

            this_aliases = _SUBSPACE_ALIASES.get(subspace, [_SUBSPACE_PRIMARY_KW[subspace][0]])
            other_aliases = _SUBSPACE_ALIASES.get(other_sub, [_SUBSPACE_PRIMARY_KW[other_sub][0]])

            this_kw = random.choice(this_aliases[:3])
            other_kw = random.choice(other_aliases[:3])

            patterns = [
                f"نه {this_kw}، بلکه {other_kw}.",
                f"منظورم {this_kw} نیست، {other_kw} است.",
                f"{this_kw} خیر، {other_kw} بله.",
                f"نه درباره {this_kw}، سوال من درباره {other_kw} است.",
                f"نه {this_kw}، منظور {other_kw} است.",
            ]

            selected = random.sample(patterns, min(len(patterns), batch))
            for p in selected:
                if len(p) >= 6:
                    out.append(normalize(p))

            if len(out) >= n:
                break

        return out[:n]

    async def generate_negation_with_correction(self, subspace: str, n: int, examples: list[str]) -> list[str]:
        """Generate 'negation with correction' patterns like 'منظورم چک نبود، تسهیلات بود'."""
        other_subspaces = [s for s in SUBSPACES if s != subspace]
        out: list[str] = []

        while len(out) < n:
            batch = min(self.gen_batch, n - len(out))
            other_sub = random.choice(other_subspaces)

            this_aliases = _SUBSPACE_ALIASES.get(subspace, [_SUBSPACE_PRIMARY_KW[subspace][0]])
            other_aliases = _SUBSPACE_ALIASES.get(other_sub, [_SUBSPACE_PRIMARY_KW[other_sub][0]])

            this_kw = random.choice(this_aliases[:3])
            other_kw = random.choice(other_aliases[:3])

            for pattern in _NEGATION_CORRECTION_PATTERNS:
                q = pattern.format(kw1=this_kw, kw2=other_kw)
                if len(q) >= 6:
                    out.append(normalize(q))
                if len(out) >= n:
                    break

            if len(out) >= n:
                break

        return out[:n]

    async def generate_cross_subspace_ambiguous(self, n: int) -> list[str]:
        """Generate ambiguous questions that deliberately mix keywords from multiple subspaces."""
        out: list[str] = []

        while len(out) < n:
            batch = min(self.gen_batch, n - len(out))

            for s1, s2, template in _CROSS_SUBSPACE_PATTERNS:
                if len(out) >= n:
                    break

                aliases1 = _SUBSPACE_ALIASES.get(s1, [_SUBSPACE_PRIMARY_KW[s1][0]])
                aliases2 = _SUBSPACE_ALIASES.get(s2, [_SUBSPACE_PRIMARY_KW[s2][0]])

                kw1 = random.choice(aliases1[:3])
                kw2 = random.choice(aliases2[:3])

                q = template.format(kw1=kw1, kw2=kw2)
                if len(q) >= 6:
                    out.append(normalize(q))

            # Also generate via LLM for more natural ambiguous questions
            if len(out) < n:
                remaining = n - len(out)
                batch = min(self.gen_batch, remaining)
                prompt = (
                    f"شما یک تولیدکنندهٔ پرسش فارسی هستید. {batch} پرسش «مبهم متقاطع» بنویسید که "
                    f"کلیدواژه‌های از دو حوزه‌ی مختلف را در یک پرسش ترکیب کند، طوری کهzona مشخص نشود.\n"
                    f"حوزه‌ها: چک، تسهیلات افراد، تسهیلات شرکت‌ها، عمومی.\n"
                    f"مثال: «چک تسهیلات أفراد چگونه است؟» یا «گزارش اعتباری شرکت‌ها چیست؟»\n\n"
                    f"قواعد: هر پرسش یک خط، بدون شماره‌گذاری، باید کلمات از حداقل دو حوزه داشته باشد."
                )
                text = await self._chat(prompt, max_tokens=50 * batch, temperature=0.9)
                got = _split_lines(text)[:batch]
                if not got:
                    break
                out.extend(got)

        return out[:n]

    async def generate_paraphrases(self, questions: list[str], n_per_question: int = 2) -> list[str]:
        """Generate diverse paraphrases of existing questions using LLM."""
        out: list[str] = []

        for q in questions:
            if len(out) >= len(questions) * n_per_question:
                break

            prompt = (
                f"این پرسش را به {n_per_question} صورت متفاوت بازنویسی کنید، با حفظ معنای دقیق:\n"
                f"پرسش اصلی: {q}\n\n"
                f"قواعد: هر بازنویسی یک خط، بدون شماره‌گذاری، فقط متن پرسش جدید. "
                f"کلیدواژه‌های اصلی حفظ شوند. متن طبیعی و روان باشد."
            )
            text = await self._chat(prompt, max_tokens=80 * n_per_question, temperature=0.8)
            got = _split_lines(text)[:n_per_question]
            out.extend(got)

        return out

    # -- judging ---------------------------------------------------------
    async def judge_membership(self, subspace: str, questions: list[str]) -> list[bool]:
        """Return True for each question that clearly belongs to *subspace*."""
        flags: list[bool] = []
        for i in range(0, len(questions), self.judge_batch):
            flags.extend(await self._judge_membership_chunk(subspace, questions[i:i + self.judge_batch]))
        return flags

    async def _judge_membership_chunk(self, subspace: str, questions: list[str]) -> list[bool]:
        numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        prompt = (
            f"حوزه: «{_SUBSPACE_FA[subspace]}».\n"
            f"برای هر پرسش مشخص کن آیا فقط به همین حوزه مربوط است (بله) یا به حوزهٔ دیگر/چند حوزه (خیر).\n"
            f"{numbered}\n\n"
            f"فقط یک آرایهٔ JSON از true/false به ترتیب پرسش‌ها برگردان، بدون توضیح. مثال: [true,false,true]"
        )
        text = await self._chat(prompt, max_tokens=16 * len(questions) + 32, temperature=0.0)
        flags = _parse_bool_array(text)
        if len(flags) != len(questions):
            # Conservative fallback: keep only questions with no other-subspace keyword.
            flags = [not _has_other_subspace_kw(q, subspace) for q in questions]
        return flags

    async def judge_ambiguous(self, questions: list[str]) -> list[bool]:
        """Return True for each question that is genuinely ambiguous."""
        flags: list[bool] = []
        for i in range(0, len(questions), self.judge_batch):
            flags.extend(await self._judge_ambiguous_chunk(questions[i:i + self.judge_batch]))
        return flags

    async def _judge_ambiguous_chunk(self, questions: list[str]) -> list[bool]:
        numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        prompt = (
            "برای هر پرسش مشخص کن آیا «مبهم» است؛ یعنی بدون مشخص‌کردن حوزه نمی‌توان پاسخ داد "
            "و می‌تواند به چند حوزه مربوط باشد (بله) یا به یک حوزهٔ مشخص اشاره دارد (خیر).\n"
            f"{numbered}\n\n"
            "فقط یک آرایهٔ JSON از true/false به ترتیب برگردان، بدون توضیح. مثال: [true,false,true]"
        )
        text = await self._chat(prompt, max_tokens=16 * len(questions) + 32, temperature=0.0)
        flags = _parse_bool_array(text)
        if len(flags) != len(questions):
            flags = [not _has_any_subspace_kw(q) for q in questions]
        return flags


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
async def build(args) -> None:
    judge = Judge(
        args.llm_url,
        args.model,
        concurrency=args.concurrency,
        judge_batch=args.judge_batch,
        gen_batch=args.gen_batch,
    )
    existing: dict[str, list[dict]] = defaultdict(list)
    if args.resume and args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                key = row["subspace"] or "__ambiguous__"
                existing[key].append(row)

    real = await extract_kb_questions(args.db_url)
    print("KB real questions:", {k: len(v) for k, v in real.items()})

    rows: list[dict] = list(sum(existing.values(), []))
    have = Counter(r["subspace"] or "__ambiguous__" for r in rows)
    now = datetime.now(timezone.utc).isoformat()

    def add(query, subspace, ambiguous, source, verdict):
        rows.append({
            "id": f"{subspace or 'ambiguous'}-{len(rows):04d}",
            "query": query, "subspace": subspace, "ambiguous": ambiguous,
            "source": source, "verdict": verdict, "created_at": now,
        })
        have[subspace or "__ambiguous__"] += 1

    # --- per-subspace positives (real first, then generated with sophisticated techniques) ---
    for sub in SUBSPACES:
        need = args.per_subspace - have[sub]
        if need <= 0:
            continue

        candidates = [q for q in real.get(sub, []) if q not in {r["query"] for r in rows}]
        if candidates:
            flags = await judge.judge_membership(sub, candidates)
            for q, ok in zip(candidates, flags):
                if ok and have[sub] < args.per_subspace:
                    add(q, sub, False, "kb", "in_subspace")

        still = args.per_subspace - have[sub]
        if still > 0:
            # Use multiple generation techniques for variety - pool all candidates then judge
            techniques = [
                ("generated_basic", judge.generate(sub, still * 3, real.get(sub, []))),
                ("generated_aliases", judge.generate_with_aliases(sub, still, real.get(sub, []))),
                ("generated_question_types", judge.generate_question_types(sub, still, real.get(sub, []))),
            ]

            # Run all generation techniques in parallel
            technique_results = await asyncio.gather(*[t[1] for t in techniques])

            # Pool all generated candidates
            all_generated = []
            for (tech_name, _), gen_list in zip(techniques, technique_results):
                for q in gen_list:
                    if q not in {r["query"] for r in rows}:
                        all_generated.append((q, tech_name))

            if all_generated:
                # Shuffle to ensure diversity from all techniques
                random.shuffle(all_generated)

                # Judge all at once
                questions = [q for q, _ in all_generated]
                tech_names = [t for _, t in all_generated]
                flags = await judge.judge_membership(sub, questions)
                for q, tech_name, ok in zip(questions, tech_names, flags):
                    if ok and have[sub] < args.per_subspace:
                        add(q, sub, False, "generated", tech_name)

        print(f"  {sub}: {have[sub]}/{args.per_subspace}")

    # --- ambiguous ---
    need = args.ambiguous - have["__ambiguous__"]
    while need > 0:
        # Use multiple ambiguous generation techniques - pool all then judge
        techniques = [
            ("ambiguous_basic", judge.generate_ambiguous(min(need * 2, 40))),
            ("ambiguous_cross_subspace", judge.generate_cross_subspace_ambiguous(min(need * 2, 40))),
        ]

        # Run all generation techniques in parallel
        technique_results = await asyncio.gather(*[t[1] for t in techniques])

        # Pool all generated candidates
        all_generated = []
        for (tech_name, _), gen_list in zip(techniques, technique_results):
            for q in gen_list:
                if q not in {r["query"] for r in rows}:
                    all_generated.append((q, tech_name))

        if all_generated:
            # Shuffle to ensure diversity from all techniques
            random.shuffle(all_generated)

            # Judge all at once
            questions = [q for q, _ in all_generated]
            tech_names = [t for _, t in all_generated]
            flags = await judge.judge_ambiguous(questions)
            for q, tech_name, ok in zip(questions, tech_names, flags):
                if ok and have["__ambiguous__"] < args.ambiguous:
                    add(q, None, True, "generated", tech_name)

        need = args.ambiguous - have["__ambiguous__"]
        print(f"  ambiguous: {have['__ambiguous__']}/{args.ambiguous}")
        if args.dry_run:
            break

    await judge.close()

    if args.dry_run:
        print("DRY RUN — not writing. Sample:")
        for r in rows[:5]:
            print("  ", r["subspace"], "|", r["query"][:70])
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "total": len(rows),
        "per_subspace": {s: have[s] for s in SUBSPACES},
        "ambiguous": have["__ambiguous__"],
        "by_source": dict(Counter(r["source"] for r in rows)),
        "by_verdict": dict(Counter(r["verdict"] for r in rows)),
        "model": args.model,
        "built_at": now,
        "out": str(args.out),
    }
    (args.out.parent / "ambiguity_dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Wrote", args.out, "->", stats)

    # --- Train/Test/Eval split (80/10/10) ---
    if not args.dry_run:
        await export_splits(rows, args.out.parent)


async def export_splits(rows: list[dict], out_dir: Path) -> None:
    """Export train/test/eval splits (80/10/10) as JSONL files."""
    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    shuffled = rows.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * 0.8)
    test_end = int(n * 0.9)

    splits = {
        "train": shuffled[:train_end],
        "test": shuffled[train_end:test_end],
        "eval": shuffled[test_end:],
    }

    for split_name, split_rows in splits.items():
        split_file = out_dir / f"ambiguity_{split_name}.jsonl"
        with split_file.open("w", encoding="utf-8") as fh:
            for r in split_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {split_name}: {len(split_rows)} samples -> {split_file}")

    # Also create a combined stats file for splits
    split_stats = {
        "total": n,
        "train": len(splits["train"]),
        "test": len(splits["test"]),
        "eval": len(splits["eval"]),
        "per_subspace_train": dict(Counter(r["subspace"] or "__ambiguous__" for r in splits["train"])),
        "per_subspace_test": dict(Counter(r["subspace"] or "__ambiguous__" for r in splits["test"])),
        "per_subspace_eval": dict(Counter(r["subspace"] or "__ambiguous__" for r in splits["eval"])),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "ambiguity_splits_stats.json").write_text(
        json.dumps(split_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Split stats:", split_stats)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db-url", default=os.getenv("KB_DB_URL_ASYNC", DEFAULT_DB_URL))
    ap.add_argument("--llm-url", default=DEFAULT_LLM_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--per-subspace", type=int, default=None)
    ap.add_argument("--ambiguous", type=int, default=None)
    ap.add_argument("--total-samples", type=int, default=1000,
                    help="Total samples (auto-calculates per-subspace and ambiguous if not specified)")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--judge-batch", type=int, default=10)
    ap.add_argument("--gen-batch", type=int, default=15)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Auto-calculate from total-samples if per-subspace and ambiguous not explicitly set
    if args.per_subspace is None or args.ambiguous is None:
        # Default: 200 ambiguous, rest divided among 4 subspaces
        if args.ambiguous is None:
            args.ambiguous = 200
        if args.per_subspace is None:
            args.per_subspace = max(0, (args.total_samples - args.ambiguous) // 4)
        print(f"Auto-calculated: per_subspace={args.per_subspace}, ambiguous={args.ambiguous}, total={args.per_subspace * 4 + args.ambiguous}")

    # Set defaults if still None
    if args.per_subspace is None:
        args.per_subspace = 200
    if args.ambiguous is None:
        args.ambiguous = 200

    asyncio.run(build(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())