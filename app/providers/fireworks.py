import json
import logging
from typing import AsyncIterator
import httpx
from app.config import get_settings
from app.providers.base import BaseProvider, ProviderError, ProviderResult
from app.schemas import ChatMessage

log = logging.getLogger("flywheel.fireworks")


class FireworksProvider(BaseProvider):
    """Cloud tier. Fireworks exposes an OpenAI-compatible API, so this
    doubles as a template for any OpenAI-style backend."""
    name = "fireworks"

    def __init__(self) -> None:
        s = get_settings()
        if not s.fireworks_api_key:
            raise ProviderError(self.name, "FLYWHEEL_FIREWORKS_API_KEY is not set")
        self.model = s.fireworks_model
        self.client = httpx.AsyncClient(
            base_url=s.fireworks_base_url,
            headers={"Authorization": f"Bearer {s.fireworks_api_key}"},
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    def _payload(self, messages, temperature, max_tokens, stream=False) -> dict:
        return {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

    async def complete(self, messages: list[ChatMessage],
                       temperature: float, max_tokens: int) -> ProviderResult:
        try:
            r = await self.client.post("/chat/completions",
                                       json=self._payload(messages, temperature, max_tokens))
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise ProviderError(self.name, f"upstream {e.response.status_code}: {e.response.text[:200]}")
        except httpx.HTTPError as e:
            raise ProviderError(self.name, f"network error: {e}")

        data = r.json()
        usage = data.get("usage", {})
        return ProviderResult(
            content=data["choices"][0]["message"]["content"],
            model=data.get("model", self.model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=data["choices"][0].get("finish_reason", "stop"),
            raw=data,
        )

    async def stream(self, messages: list[ChatMessage],
                     temperature: float, max_tokens: int) -> AsyncIterator[str]:
        try:
            async with self.client.stream(
                "POST", "/chat/completions",
                json=self._payload(messages, temperature, max_tokens, stream=True),
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if line.startswith("data:"):
                        yield line + "\n\n"
        except httpx.HTTPError as e:
            raise ProviderError(self.name, f"stream error: {e}")