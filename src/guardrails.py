"""Step 5 - Guardrails: safety checks written in plain code."""
import re

INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above|your) (instructions|rules)",
    r"disregard .{0,30}(rules|instructions)",
    r"reveal .{0,20}system prompt",
    r"system override",
    r"you are now",
    r"say hacked",
]

PII_PATTERNS = {
    "CARD": r"\b(?:\d[ -]?){13,16}\b",
    "EMAIL": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "PHONE": r"\b\d{10}\b",
}


def is_injection(text):
    """True if the text contains a known attack phrase."""
    return any(re.search(p, text.lower()) for p in INJECTION_PATTERNS)


def redact_pii(text):
    """Replace emails, phone numbers and card numbers with a label."""
    for label, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f"[{label}_REMOVED]", text)
    return text


def check_input(question):
    """Checkpoint 1: returns (ok, text). Blocks attacks, hides private data."""
    if is_injection(question):
        return False, "Blocked: this looks like a prompt injection attack."
    return True, redact_pii(question)


def clean_context(chunks):
    """Checkpoint 2: remove attack sentences hidden inside documents."""
    cleaned = []
    for c in chunks:
        sentences = re.split(r"(?<=[.!?])\s+", c["text"])
        safe = [s for s in sentences if not is_injection(s)]
        if safe:
            cleaned.append({**c, "text": " ".join(safe)})
    return cleaned


def check_output(answer, n_chunks):
    """Checkpoint 3: an answer should cite a source, or honestly refuse."""
    if "don't have enough information" in answer.lower():
        return True
    return any(f"[{i + 1}]" in answer for i in range(n_chunks))