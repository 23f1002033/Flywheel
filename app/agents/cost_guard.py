"""Cost Agent: watches monthly spend against the configured budget and
tightens routing as the cap approaches. Past the cap, cloud requires a
much higher complexity bar (client overrides still win - control first)."""

import logging
from typing import Optional
from app.config import get_settings
from app.memory.store import MemoryStore

log = logging.getLogger("flywheel.costguard")


class CostAgent:
    def __init__(self, memory: MemoryStore) -> None:
        self.memory = memory
        self.settings = get_settings()

    def pressure(self) -> Optional[tuple[float, str]]:
        """Returns (threshold_delta, reason) or None."""
        budget = self.settings.monthly_budget_usd
        if budget <= 0:
            return None
        spend = self.memory.month_spend()
        ratio = spend / budget
        if ratio >= 1.0:
            return (100.0, f"budget exhausted ({spend:.2f}/{budget:.2f} USD) - cloud locked")
        if ratio >= 0.8:
            return (1.5, f"budget pressure ({ratio:.0%} used)")
        return None

    def status(self) -> dict:
        spend = self.memory.month_spend()
        budget = self.settings.monthly_budget_usd
        return {"month_spend_usd": round(spend, 6),
                "monthly_budget_usd": budget,
                "used_ratio": round(spend / budget, 4) if budget > 0 else None}