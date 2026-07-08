"""Safety Agent: detects sensitive content so it can be pinned to the
local tier - sensitive data never leaves the customer's infrastructure."""

import re
from dataclasses import dataclass, field

# PII patterns (compiled once) 
PII_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
    "phone": re.compile(r"(?:\+?\d{1,3}[\s-]?)?(?:\d[\s-]?){9,11}\d\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "us_ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "aadhaar": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    "pan_card": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "api_key": re.compile(r"\b(?:sk|pk|api|key|token)[-_][A-Za-z0-9_-]{16,}\b", re.I),
}

# domain keyword sets 
DOMAIN_KEYWORDS: dict[str, set[str]] = {
    "financial": {"salary", "bank account", "iban", "swift", "invoice", "tax return",
                  "net worth", "loan", "credit score", "upi", "ifsc"},
    "legal": {"nda", "lawsuit", "plaintiff", "defendant", "settlement", "contract breach",
              "confidential agreement", "litigation"},
    "healthcare": {"diagnosis", "prescription", "patient", "medical record", "symptoms",
                   "blood test", "mental health", "therapy notes", "hipaa"},
    "credentials": {"password", "secret key", "private key", "access token", "otp"},
}


@dataclass
class SafetyVerdict:
    sensitive: bool = False
    categories: list[str] = field(default_factory=list)
    reason: str = "clean"


class SafetyAgent:
    """Deterministic first line of defense. Fast (regex + sets), runs on
    every request before any routing decision."""

    def inspect(self, text: str) -> SafetyVerdict:
        found: list[str] = []

        for name, pattern in PII_PATTERNS.items():
            if pattern.search(text):
                found.append(f"pii:{name}")

        lowered = text.lower()
        for domain, words in DOMAIN_KEYWORDS.items():
            if any(w in lowered for w in words):
                found.append(f"domain:{domain}")

        if found:
            return SafetyVerdict(sensitive=True, categories=found,
                                 reason=f"sensitive content detected ({', '.join(found[:4])})")
        return SafetyVerdict()