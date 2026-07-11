"""Evaluator Agent: samples local responses and has the cloud model grade
them 0.0-1.0. Scores feed the memory router and dataset filtering.
Fire-and-forget: never blocks or fails a user request."""

import logging
import random
from app.config import get_settings
from app.memory.store import MemoryStore
from app.providers.base import ProviderError
from app.providers.registry import get_provider
from app.schemas import ChatMessage

log = logging.getLogger("flywheel.evaluator")

JUDGE_PROMPT = """You are grading an AI assistant's answer. Score it 0.0-1.0 for
correctness, helpfulness and clarity given the user prompt. Respond with ONLY
the number.

User prompt:
{prompt}

Assistant answer:
{answer}"""


class EvaluatorAgent:
    def __init__(self, memory: MemoryStore) -> None:
        self.memory = memory
        self.settings = get_settings()

    def should_sample(self) -> bool:
        return random.random() < self.settings.judge_sample_rate

    async def judge(self, request_id: int, prompt: str, answer: str) -> None:
        try:
            result = await get_provider("cloud").complete(
                [ChatMessage(role="user",
                             content=JUDGE_PROMPT.format(prompt=prompt[:2000],
                                                         answer=answer[:2000]))],
                temperature=0.0, max_tokens=8)
            score = max(0.0, min(1.0, float(result.content.strip())))
            self.memory.set_quality(request_id, score)
            log.info("judged request #%s -> %.2f", request_id, score)
        except (ProviderError, ValueError) as e:
            log.warning("judge skipped for #%s: %s", request_id, e)
