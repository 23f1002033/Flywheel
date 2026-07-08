"""Cost engine: actual cost, counterfactual (what a frontier model would
have charged), and rough CO2 savings for locally served tokens."""

from dataclasses import dataclass
from app.config import get_settings


@dataclass
class CostReport:
    cost_usd: float
    counterfactual_usd: float
    saved_usd: float
    co2_saved_grams: float


def compute(route: str, prompt_tokens: int, completion_tokens: int) -> CostReport:
    s = get_settings()
    counterfactual = (prompt_tokens * s.frontier_price_input
                      + completion_tokens * s.frontier_price_output) / 1_000_000
    if route == "cloud":
        cost = (prompt_tokens * s.cloud_price_input
                + completion_tokens * s.cloud_price_output) / 1_000_000
        co2 = 0.0
    else:  # local or cache: zero marginal API cost
        cost = 0.0
        co2 = (prompt_tokens + completion_tokens) / 1000 * s.co2_grams_per_1k_cloud_tokens
    return CostReport(cost_usd=round(cost, 8),
                      counterfactual_usd=round(counterfactual, 8),
                      saved_usd=round(counterfactual - cost, 8),
                      co2_saved_grams=round(co2, 4))
