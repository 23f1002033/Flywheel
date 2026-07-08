"""Semantic cache: if a request is nearly identical to one already answered,
serve the stored answer - zero cost, ~zero latency. Reuses the memory
store's embeddings; no separate storage needed."""

import logging
from typing import Optional
from app.config import get_settings
from app.memory.store import MemoryStore

log = logging.getLogger("flywheel.cache")


class SemanticCache:
    def __init__(self, memory: MemoryStore) -> None:
        self.memory = memory
        self.settings = get_settings()

    def lookup(self, prompt_text: str) -> Optional[dict]:
        if not self.settings.cache_enabled:
            return None
        neighbors = self.memory.similar(prompt_text, k=1)
        if not neighbors:
            return None
        best = neighbors[0]
        if (best["similarity"] >= self.settings.cache_similarity_threshold
                and best["response"] and not best["cached"]):
            log.info("cache hit: sim=%.4f against request #%s",
                     best["similarity"], best["id"])
            return best
        return None