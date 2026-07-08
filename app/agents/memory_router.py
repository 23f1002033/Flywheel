"""Learning hint for the Router Agent: look up similar past requests and
their outcomes. If requests like this one consistently succeeded locally,
nudge toward local; if they consistently went to cloud, nudge toward cloud.
The router literally gets smarter as traffic accumulates."""

import logging
from typing import Optional
from app.config import get_settings
from app.memory.store import MemoryStore

log = logging.getLogger("flywheel.memrouter")

GOOD_QUALITY = 0.6   


class MemoryHint:
    def __init__(self, memory: MemoryStore) -> None:
        self.memory = memory
        self.settings = get_settings()

    def __call__(self, text: str) -> Optional[tuple[float, str]]:
        if not self.settings.memory_routing_enabled:
            return None
        neighbors = [n for n in self.memory.similar(text, k=self.settings.memory_k)
                     if n["similarity"] >= 0.55]
        if len(neighbors) < self.settings.memory_min_samples:
            return None

        local_ok = sum(1 for n in neighbors
                       if n["route"] in ("local", "cache")
                       and (n["quality_score"] is None or n["quality_score"] >= GOOD_QUALITY))
        cloud = sum(1 for n in neighbors if n["route"] == "cloud")
        rate_local = local_ok / len(neighbors)
        rate_cloud = cloud / len(neighbors)

        if rate_local >= 0.7:
            return (-1.5, f"memory: {local_ok}/{len(neighbors)} similar requests fine locally")
        if rate_cloud >= 0.7:
            return (1.5, f"memory: {cloud}/{len(neighbors)} similar requests needed cloud")
        return None