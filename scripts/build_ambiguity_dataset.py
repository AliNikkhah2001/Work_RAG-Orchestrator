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

Output (JSONL, one object per line):

    {"id": ..., "query": ..., "subspace": "cheque"|..., "ambiguous": false,
     "source": "kb"|"generated", "verdict": ..., "created_at": ...}

Ambiguous rows carry `"subspace": null, "ambiguous": true`.

Usage
-----
    python scripts/build_ambiguity_dataset.py                 # full run (100/subspace + 200 ambiguous)
    python scripts/build_ambiguity_dataset.py --per-subspace 5 --ambiguous 5 --dry-run
    python scripts/build_ambiguity_dataset.py --resume        # keep existing rows, fill gaps
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
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

    def __init__(self, url: str, model: str, concurrency: int = 2, timeout: float = 120.0,
                 judge_batch: int = 10, gen_batch: int = 15):
        self.url = url
        self.model = model
        self.sem = asyncio.Semaphore(concurrency)
        self.client = httpx.AsyncClient(timeout=timeout, trust_env=False)
        self.judge_batch = judge_batch
        self.gen_batch = gen_batch

    async def close(self):
        await self.client.aclose()

    async def _chat(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7,
                    retries: int = 4) -> str:
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


_SUBSPACE_FA = {
    "cheque": "گزارش اعتبارسنجی چک",
    "facilities_person": "اعتبارسنجی تسهیلات برای اشخاص حقیقی (افراد)",
    "facilities_legal": "اعتبارسنجی تسهیلات برای اشخاص حقوقی (شرکت‌ها)",
    "general": "پایگاه دانش عمومی (معرفی شرکت، هیئت‌مدیره، کاربردها)",
}
_SUBSPACE_KW = {
    "cheque": ("چک", "صیاد", "پیچک", "برگشت"),
    "facilities_person": ("تسهیلات", "وام", "قسط", "حقیقی", "بازپرداخت"),
    "facilities_legal": ("حقوقی", "شرکت", "سهامدار"),
    "general": ("معرفی", "هیئت مدیره", "هیات مدیره", "کاربرد", "عمومی"),
}


def _has_any_subspace_kw(q: str) -> bool:
    ql = normalize(q)
    return any(kw in ql for kws in _SUBSPACE_KW.values() for kw in kws)


def _has_other_subspace_kw(q: str, subspace: str) -> bool:
    ql = normalize(q)
    for name, kws in _SUBSPACE_KW.items():
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
        if len(line) < 6 or "؟" not in line and "?" not in line:
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
# Orchestration
# ---------------------------------------------------------------------------
async def build(args) -> None:
    judge = Judge(args.llm_url, args.model, concurrency=args.concurrency,
                  judge_batch=args.judge_batch, gen_batch=args.gen_batch)
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

    # --- per-subspace positives (real first, then generated) ---
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
            gen = await judge.generate(sub, still * 2, real.get(sub, []))
            gen = [q for q in gen if q not in {r["query"] for r in rows}]
            if gen:
                flags = await judge.judge_membership(sub, gen)
                for q, ok in zip(gen, flags):
                    if ok and have[sub] < args.per_subspace:
                        add(q, sub, False, "generated", "in_subspace")
        print(f"  {sub}: {have[sub]}/{args.per_subspace}")

    # --- ambiguous ---
    need = args.ambiguous - have["__ambiguous__"]
    while need > 0:
        batch = await judge.generate_ambiguous(min(need * 2, 40))
        batch = [q for q in batch if q not in {r["query"] for r in rows}]
        if not batch:
            break
        flags = await judge.judge_ambiguous(batch)
        for q, ok in zip(batch, flags):
            if ok and have["__ambiguous__"] < args.ambiguous:
                add(q, None, True, "generated", "ambiguous")
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
        "model": args.model,
        "built_at": now,
        "out": str(args.out),
    }
    (args.out.parent / "ambiguity_dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Wrote", args.out, "->", stats)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db-url", default=os.getenv("KB_DB_URL_ASYNC", DEFAULT_DB_URL))
    ap.add_argument("--llm-url", default=DEFAULT_LLM_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--per-subspace", type=int, default=100)
    ap.add_argument("--ambiguous", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--judge-batch", type=int, default=10)
    ap.add_argument("--gen-batch", type=int, default=15)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    asyncio.run(build(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
