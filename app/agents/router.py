"""Router Agent: decides local vs cloud for every request.
M3 = layered heuristics + safety pinning. M5 adds memory-based learning,
M7 adds evaluator feedback - both plug into this same class."""

import logging
import re
from dataclasses import dataclass, field
from app.agents.safety import SafetyAgent, SafetyVerdict
from app.config import get_settings
from app.schemas import ChatMessage

log = logging.getLogger("flywheel.router")

# Signals that a prompt needs frontier-level reasoning
HARD_PATTERNS = [
    (re.compile(r"\b(prove|theorem|derivative|integral|complexity analysis)\b", re.I), 2.5, "math/proof"),
    (re.compile(r"\b(refactor|debug|implement|write a (function|class|program)|stack trace)\b", re.I), 2.0, "coding"),
    (re.compile(r"\b(step[- ]by[- ]step|chain of thought|reason (through|about)|multi[- ]step)\b", re.I), 1.5, "multi-step reasoning"),
    (re.compile(r"\b(compare and contrast|trade[- ]?offs|pros and cons|architecture design)\b", re.I), 1.5, "analysis"),
    (re.compile(r"\b(legal advice|medical advice|financial analysis)\b", re.I), 1.5, "expert domain"),
    (re.compile(r"```", re.M), 1.5, "code block present"),
]

# Signals a prompt is easy for a small model
EASY_PATTERNS = [
    (re.compile(r"^(hi|hello|hey|thanks|thank you)\b", re.I), -2.0, "greeting"),
    (re.compile(r"\b(what is|define|meaning of|who is|when was)\b", re.I), -1.0, "simple lookup"),
    (re.compile(r"\b(summarize|tl;?dr|shorten|rephrase|translate)\b", re.I), -0.5, "transform task"),
]


@dataclass
class RouteDecision:
    route: str                      # "local" | "cloud"
    reason: str
    complexity: float = 0.0
    sensitive: bool = False
    categories: list[str] = field(default_factory=list)


class RouterAgent:
    def __init__(self) -> None:
        self.safety = SafetyAgent()
        self.settings = get_settings()

    def _complexity_score(self, text: str) -> tuple[float, list[str]]:
        score, hits = 0.0, []
        for pattern, weight, label in HARD_PATTERNS + EASY_PATTERNS:
            if pattern.search(text):
                score += weight
                hits.append(label)
        # length pressure: long prompts usually mean harder tasks
        words = len(text.split())
        if words > 400:
            score += 2.0; hits.append("very long prompt")
        elif words > 150:
            score += 1.0; hits.append("long prompt")
        return score, hits

    def decide(self, model: str, messages: list[ChatMessage]) -> RouteDecision:
        # 1. client override always wins
        if model == "flywheel-cloud":
            return RouteDecision(route="cloud", reason="client forced cloud")
        if model == "flywheel-local":
            return RouteDecision(route="local", reason="client forced local")

        text = " ".join(m.content for m in messages if m.role != "system")

        # 2. safety pinning: sensitive never leaves the box
        verdict: SafetyVerdict = self.safety.inspect(text)
        if verdict.sensitive and self.settings.pin_sensitive_local:
            return RouteDecision(route="local",
                                 reason=f"pinned local by safety agent: {verdict.reason}",
                                 sensitive=True, categories=verdict.categories)

        # 3. complexity heuristics
        score, hits = self._complexity_score(text)
        threshold = self.settings.complexity_threshold
        if score >= threshold:
            return RouteDecision(route="cloud", complexity=score,
                                 sensitive=verdict.sensitive, categories=verdict.categories,
                                 reason=f"complexity {score:.1f} >= {threshold} ({', '.join(hits[:3])})")
        return RouteDecision(route="local", complexity=score,
                             sensitive=verdict.sensitive, categories=verdict.categories,
                             reason=f"complexity {score:.1f} < {threshold}"
                                    + (f" ({', '.join(hits[:3])})" if hits else ""))