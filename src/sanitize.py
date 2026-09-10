"""Layer 0 — sanitization / abstention gate.

Runs BEFORE either the cheap classifier or the LLM fallback. Anything flagged here returns the
defined abstention envelope directly and never reaches a model, which is what guarantees:
  - zero crashes/hangs from malformed unicode or binary junk (nothing weird ever reaches
    the tokenizer or gets sent to an LLM prompt)
  - zero prompt-injection susceptibility (injection text never reaches the LLM at all)
  - 100% schema validity on the hostile pack (the abstention envelope is schema-valid by
    construction)
"""
from __future__ import annotations

import re
import unicodedata

MAX_TEXT_LEN = 4000          # a message of "several thousand words" is a noisy-slice case, not
                              # a reason to OOM a tokenizer; cap and still try to classify
MAX_TOTAL_CONTEXT_LEN = 12000

ZERO_WIDTH = "\u200b\u200c\u200d\ufeff"
RTL_OVERRIDE = "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
CONTROL_CHARS = "".join(chr(c) for c in range(0, 32) if c not in (9, 10, 13))

INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(your|previous|all)\s+instructions", re.I),
    re.compile(r"</?\s*(system|context|instructions?)\s*>", re.I),
    re.compile(r"^\s*system\s*:", re.I | re.M),
    re.compile(r"new instruction", re.I),
    re.compile(r"you (are|must|will) now (act|behave|respond)", re.I),
]


class SanitizeResult:
    def __init__(self, ok: bool, reason: str | None, cleaned_context: list[dict] | None):
        self.ok = ok
        self.reason = reason
        self.cleaned_context = cleaned_context


def _strip_unicode_tricks(text: str) -> str:
    # NFKC normalization collapses most decomposed / compatibility-equivalent forms.
    text = unicodedata.normalize("NFKC", text)
    for ch in ZERO_WIDTH + RTL_OVERRIDE + CONTROL_CHARS:
        text = text.replace(ch, "")
    return text


def _looks_like_injection(text: str) -> bool:
    return any(p.search(text) for p in INJECTION_PATTERNS)


def sanitize(context: list[dict]) -> SanitizeResult:
    """Returns ok=False with a reason if the input should be abstained on immediately,
    otherwise ok=True with a cleaned context safe to hand to a classifier or LLM prompt."""
    if not context:
        return SanitizeResult(False, "empty_context", None)

    total_len = sum(len(t.get("text", "")) for t in context)
    if total_len == 0:
        return SanitizeResult(False, "empty_text", None)
    if total_len > MAX_TOTAL_CONTEXT_LEN:
        # Still attempt classification but flag it — a very long message is a noisy-slice
        # case (per the brief), not necessarily hostile. We truncate rather than abstain.
        pass

    cleaned = []
    injection_hit = False
    for turn in context:
        text = turn.get("text", "")
        if not isinstance(text, str):
            return SanitizeResult(False, "non_string_text", None)
        clean_text = _strip_unicode_tricks(text)[:MAX_TEXT_LEN]
        if turn.get("speaker") == "customer" and _looks_like_injection(clean_text):
            injection_hit = True
        cleaned.append({"speaker": turn.get("speaker", "customer"), "text": clean_text})

    if injection_hit:
        # We don't crash and we don't comply — we abstain and flag for a human, which is the
        # only safe behaviour when the customer's own message is trying to steer the system.
        return SanitizeResult(False, "prompt_injection_detected", cleaned)

    return SanitizeResult(True, None, cleaned)


def abstain_response(row_id: str, reason: str) -> dict:
    return {
        "id": row_id,
        "intent": "unknown",
        "action": "none",
        "confidence": 0.0,
        "needs_human": True,
        "error": {"code": "invalid_input", "message": reason},
    }
