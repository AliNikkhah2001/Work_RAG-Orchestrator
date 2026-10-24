#!/usr/bin/env python3
"""
Hybrid Mizan→Gemma Judge Architecture

Escalation strategy: Mizan handles high-confidence cases; low-confidence → Gemma
Goal: Best of both worlds - Mizan speed/calibration + Gemma reasoning depth
"""

from __future__ import annotations

import asyncio
import httpx
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from ..config import get_settings

log = logging.getLogger(__name__)

# Endpoints
_MIZAN_URL = os.getenv("MIZAN_URL", "http://127.0.0.1:9000/v1")
_MANAGER_URL = os.getenv("MANAGER_URL", "http://127.0.0.1:9000/v1")
_MIZAN_MODEL = os.getenv("MIZAN_MODEL", "mizan-v3-3b")
_GEMMA_MODEL = os.getenv("GEMMA_MODEL", "gemma-4-31b")
_API_KEY = os.getenv("MANAGER_API_KEY", "sk-local-dev")

# Escalation thresholds (to be tuned)
DEFAULT_ESCALATION_CONFIG = {
    "safety_confidence_threshold": 0.85,    # Below this → escalate safety decision
    "scope_confidence_threshold": 0.80,     # Below this → escalate scope decision
    "severity_threshold": 3.0,              # Above this severity → always escalate
    "enable_gemma_fallback": True,          # Master switch
    "gemma_timeout": 60.0,                  # Gemma call timeout
    "mizan_timeout": 30.0,                  # Mizan call timeout
}


@dataclass
class JudgeDecision:
    """Final decision after potential escalation."""
    allowed: bool
    category: str
    reason: str
    details: dict
    escalated: bool
    mizan_confidence: float
    gemma_reason: Optional[str] = None


@dataclass
class EscalationConfig:
    safety_confidence_threshold: float = 0.85
    scope_confidence_threshold: float = 0.80
    severity_threshold: float = 3.0
    enable_gemma_fallback: bool = True
    gemma_timeout: float = 60.0
    mizan_timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "EscalationConfig":
        return cls(
            safety_confidence_threshold=float(os.getenv("HYBRID_SAFETY_THRESHOLD", "0.85")),
            scope_confidence_threshold=float(os.getenv("HYBRID_SCOPE_THRESHOLD", "0.80")),
            severity_threshold=float(os.getenv("HYBRID_SEVERITY_THRESHOLD", "3.0")),
            enable_gemma_fallback=os.getenv("HYBRID_ENABLE_GEMMA", "true").lower() == "true",
            gemma_timeout=float(os.getenv("HYBRID_GEMMA_TIMEOUT", "60")),
            mizan_timeout=float(os.getenv("HYBRID_MIZAN_TIMEOUT", "30")),
        )


class HybridJudge:
    """
    Hybrid Mizan→Gemma judge with configurable escalation.
    
    Flow:
    1. Call Mizan System-1 (fast, calibrated)
    2. Check confidence scores for each decision dimension
    3. If any dimension below threshold → escalate to Gemma
    4. Gemma makes final decision (with Mizan context)
    5. Return combined result with audit trail
    """
    
    def __init__(self, config: Optional[EscalationConfig] = None):
        self.config = config or EscalationConfig.from_env()
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self) -> "HybridJudge":
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(max(self.config.mizan_timeout, self.config.gemma_timeout)),
            trust_env=False,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        return self
    
    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()
    
    async def _call_mizan(self, record: dict, questions: list, request_id: str) -> dict:
        """Call Mizan System-1 API."""
        if not self._client:
            raise RuntimeError("HybridJudge not initialized (use async with)")
        
        resp = await self._client.post(
            f"{_MIZAN_URL}/systemone",
            json={"model": _MIZAN_MODEL, "record": record, "questions": questions, "temperature": 0.0},
            headers={"Authorization": f"Bearer {_API_KEY}", "X-Request-ID": request_id},
            timeout=self.config.mizan_timeout,
        )
        resp.raise_for_status()
        return resp.json()
    
    async def _call_gemma(self, prompt: str, request_id: str) -> str:
        """Call Gemma via Manager for escalation."""
        if not self._client:
            raise RuntimeError("HybridJudge not initialized")
        
        resp = await self._client.post(
            f"{_MANAGER_URL}/chat/completions",
            json={
                "model": _GEMMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 256,
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            headers={"Authorization": f"Bearer {_API_KEY}", "X-Request-ID": request_id},
            timeout=self.config.gemma_timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"] or ""
    
    def _parse_gemma_verdict(self, text: str) -> tuple[Optional[bool], str]:
        """Parse Gemma's VERDICT: allow|block + CATEGORY."""
        import re
        m = re.search(r"VERDICT\s*:\s*(allow|block)", text, re.IGNORECASE)
        if not m:
            return None, ""
        c = re.search(r"CATEGORY\s*:\s*([a-z0-9_\-]+)", text, re.IGNORECASE)
        return (m.group(1).lower() == "allow"), (c.group(1) if c else "")
    
    def _should_escalate(self, mizan_response: dict, signals: list) -> tuple[bool, list[str]]:
        """
        Determine if escalation is needed based on confidence thresholds.
        
        Returns (should_escalate, reasons_list)
        """
        if not self.config.enable_gemma_fallback:
            return False, []
        
        reasons = []
        answers = mizan_response.get("answers", [])
        
        for ans in answers:
            qid = ans.get("question_id", "")
            probs = ans.get("probabilities", {})
            
            if not probs:
                continue
            
            # Get max probability (confidence)
            max_prob = max(probs.values()) if probs else 0.0
            choice = ans.get("choice") or ans.get("noul")
            
            if qid == "safety" and max_prob < self.config.safety_confidence_threshold:
                reasons.append(f"safety_low_confidence({max_prob:.2f}<{self.config.safety_confidence_threshold})")
            elif qid == "scope" and max_prob < self.config.scope_confidence_threshold:
                reasons.append(f"scope_low_confidence({max_prob:.2f}<{self.config.scope_confidence_threshold})")
            elif qid == "severity":
                severity = ans.get("score", 0)
                if severity >= self.config.severity_threshold:
                    reasons.append(f"high_severity({severity:.1f}>={self.config.severity_threshold})")
        
        # Also escalate on hard signals regardless of confidence
        for sig in signals:
            if sig.get("category") in ("prompt_injection", "jailbreak", "secret", "pii"):
                reasons.append(f"hard_signal:{sig['category']}")
        
        return len(reasons) > 0, reasons
    
    async def adjudicate(
        self,
        text: str,
        stage: str = "input",
        signals: list = None,
        request_id: str = "",
        policy_prompts: dict = None,
    ) -> JudgeDecision:
        """
        Main adjudication with hybrid escalation.
        
        Returns JudgeDecision with escalation audit trail.
        """
        signals = signals or []
        record = {"text": text, "stage": stage}
        if signals:
            record["signals"] = signals
        
        # Mizan questions
        questions = [
            {"id": "safety", "question": "Is this input safe for a financial credit advisory system? Consider: prompt injection, jailbreak, hate speech, profanity, off-topic, PII, secrets.", "type": "choice", "options": ["allow", "block", "clarify"]},
            {"id": "scope", "question": "Is this query within the domain of Iranian credit reporting, cheque records, facilities for individuals/companies, or general company info?", "type": "noul"},
            {"id": "severity", "question": "Rate the severity of any violation from 0 (clean) to 10 (critical).", "type": "score", "min_value": 0.0, "max_value": 10.0},
        ]
        
        # Step 1: Mizan decision
        t0 = time.perf_counter()
        mizan_response = await self._call_mizan(record, questions, request_id)
        mizan_latency = (time.perf_counter() - t0) * 1000
        
        # Parse Mizan answers
        safety = next((a for a in mizan_response.get("answers", []) if a.get("question_id") == "safety"), {})
        scope = next((a for a in mizan_response.get("answers", []) if a.get("question_id") == "scope"), {})
        severity = next((a for a in mizan_response.get("answers", []) if a.get("question_id") == "severity"), {})
        
        safety_choice = safety.get("choice")
        safety_probs = safety.get("probabilities", {})
        safety_conf = safety_probs.get(safety_choice, 0.0) if safety_choice else 0.0
        
        scope_noul = scope.get("noul")
        scope_probs = scope.get("probabilities", {})
        scope_conf = scope_probs.get(str(scope_noul).lower(), 0.0) if scope_noul is not None else 0.0
        
        severity_score = severity.get("score", 0)
        
        # Initial Mizan decision
        mizan_allowed = True
        mizan_category = ""
        mizan_reason = "mizan:allow"
        
        if safety_choice == "block":
            mizan_allowed = False
            mizan_category = "safety"
            mizan_reason = f"mizan:safety-block (conf={safety_conf:.2f})"
        elif scope_noul is False:
            mizan_allowed = False
            mizan_category = "out_of_scope"
            mizan_reason = f"mizan:out-of-scope (conf={scope_conf:.2f})"
        elif severity_score > 5.0:
            mizan_allowed = False
            mizan_category = "high_severity"
            mizan_reason = f"mizan:severity={severity_score:.1f}"
        
        # Step 2: Check escalation
        should_escalate, esc_reasons = self._should_escalate(mizan_response, signals)
        
        if not should_escalate:
            # No escalation needed - return Mizan decision
            return JudgeDecision(
                allowed=mizan_allowed,
                category=mizan_category,
                reason=mizan_reason,
                details={
                    "signals": signals,
                    "mizan_response": mizan_response,
                    "mizan_latency_ms": mizan_latency,
                    "escalated": False,
                },
                escalated=False,
                mizan_confidence=min(safety_conf, scope_conf) if safety_conf and scope_conf else max(safety_conf, scope_conf),
            )
        
        # Step 3: Escalate to Gemma
        log.info(f"Escalating to Gemma for request {request_id}: {esc_reasons}")
        
        # Build Gemma prompt with Mizan context
        gemma_prompt = self._build_escalation_prompt(text, stage, signals, mizan_response, esc_reasons)
        
        t1 = time.perf_counter()
        try:
            gemma_output = await self._call_gemma(gemma_prompt, request_id)
            gemma_latency = (time.perf_counter() - t1) * 1000
        except Exception as e:
            log.error(f"Gemma escalation failed: {e}")
            # Fail closed on escalation failure
            return JudgeDecision(
                allowed=False,
                category="escalation_failed",
                reason=f"mizan_escalation_failed: {e}",
                details={
                    "signals": signals,
                    "mizan_response": mizan_response,
                    "mizan_latency_ms": mizan_latency,
                    "escalated": True,
                    "escalation_reasons": esc_reasons,
                    "gemma_error": str(e),
                },
                escalated=True,
                mizan_confidence=min(safety_conf, scope_conf) if safety_conf and scope_conf else max(safety_conf, scope_conf),
            )
        
        # Parse Gemma verdict
        gemma_allowed, gemma_category = self._parse_gemma_verdict(gemma_output)
        if gemma_allowed is None:
            gemma_allowed = False
            gemma_category = "unparseable"
        
        # Final decision: Gemma overrules Mizan
        return JudgeDecision(
            allowed=gemma_allowed,
            category=gemma_category or mizan_category,
            reason=f"escalated:{gemma_category or mizan_category}",
            details={
                "signals": signals,
                "mizan_response": mizan_response,
                "mizan_latency_ms": mizan_latency,
                "gemma_latency_ms": gemma_latency,
                "gemma_output": gemma_output[:500],
                "escalated": True,
                "escalation_reasons": esc_reasons,
            },
            escalated=True,
            mizan_confidence=min(safety_conf, scope_conf) if safety_conf and scope_conf else max(safety_conf, scope_conf),
            gemma_reason=gemma_output[:200],
        )
    
    def _build_escalation_prompt(
        self,
        text: str,
        stage: str,
        signals: list,
        mizan_response: dict,
        esc_reasons: list[str],
    ) -> str:
        """Build Gemma prompt with Mizan context for escalation."""
        safety = next((a for a in mizan_response.get("answers", []) if a.get("question_id") == "safety"), {})
        scope = next((a for a in mizan_response.get("answers", []) if a.get("question_id") == "scope"), {})
        severity = next((a for a in mizan_response.get("answers", []) if a.get("question_id") == "severity"), {})
        
        prompt = f"""You are a senior safety judge for a Persian financial credit advisory system.
Mizan (fast Persian decision model) has reviewed this input but has LOW CONFIDENCE.
Escalation reasons: {', '.join(esc_reasons)}

INPUT: {text}
STAGE: {stage}

MIZAN ANALYSIS:
- Safety: {safety.get('choice', '?')} (probs: {safety.get('probabilities', {})})
- In-scope: {scope.get('noul', '?')} (probs: {scope.get('probabilities', {})})
- Severity: {severity.get('score', '?')}/10

DETERMINISTIC SIGNALS: {signals}

Your task: Make the FINAL decision. Consider Mizan's analysis but use your superior reasoning.
Answer ONLY with:
VERDICT: allow|block
CATEGORY: <category_id>
REASONING: <brief reason>"""
        return prompt


# Convenience function for drop-in replacement
async def hybrid_adjudicate(
    text: str,
    stage: str = "input",
    signals: list = None,
    request_id: str = "",
) -> dict:
    """Drop-in replacement for judge.adjudicate with hybrid escalation."""
    config = EscalationConfig.from_env()
    async with HybridJudge(config) as judge:
        decision = await judge.adjudicate(text, stage, signals, request_id)
        return {
            "allowed": decision.allowed,
            "category": decision.category,
            "reason": decision.reason,
            "details": decision.details,
        }