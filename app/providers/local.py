import asyncio
import json
import logging
import time
import uuid
from typing import AsyncIterator
import httpx
from app.config import get_settings
from app.providers.base import BaseProvider, ProviderError, ProviderResult
from app.schemas import ChatMessage

log = logging.getLogger("flywheel.local")


class LocalProvider(BaseProvider):
    """Local tier. Talks to a vLLM server (OpenAI-compatible) on the AMD GPU.
    If FLYWHEEL_LOCAL_BASE_URL is empty, runs in STUB MODE so the whole
    pipeline is developable on any laptop with no GPU."""
    name = "local"

    def __init__(self) -> None:
        s = get_settings()
        self.base_url = s.local_base_url.rstrip("/") if s.local_base_url else ""
        self.model = s.local_model
        self.stub = not self.base_url
        if self.stub:
            log.warning("LocalProvider in STUB mode (FLYWHEEL_LOCAL_BASE_URL empty)")
        else:
            self.client = httpx.AsyncClient(
                base_url=self.base_url, timeout=httpx.Timeout(120.0, connect=10.0))

    def _stub_reply(self, messages: list[ChatMessage]) -> str:
        last = next((m.content for m in reversed(messages) if m.role == "user"), "")
        return (f"[local-stub answer] I understood your request about: "
                f"'{last[:60]}'. (Real Gemma-on-ROCm replies arrive once "
                f"FLYWHEEL_LOCAL_BASE_URL points to a vLLM server.)")

    async def complete(self, messages: list[ChatMessage],
                       temperature: float, max_tokens: int) -> ProviderResult:
        if self.stub:
            reply = self._stub_reply(messages)
            prompt_toks = sum(len(m.content.split()) for m in messages)
            return ProviderResult(content=reply, model="local-stub",
                                  prompt_tokens=prompt_toks,
                                  completion_tokens=len(reply.split()))
        try:
            r = await self.client.post("/v1/chat/completions", json={
                "model": self.model,
                "messages": [m.model_dump() for m in messages],
                "temperature": temperature,
                "max_tokens": max_tokens,
            })
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise ProviderError(self.name, f"vLLM error: {e}")

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
        if self.stub:
            rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            for word in self._stub_reply(messages).split():
                chunk = {"id": rid, "object": "chat.completion.chunk",
                         "created": int(time.time()), "model": "local-stub",
                         "choices": [{"index": 0, "delta": {"content": word + " "},
                                      "finish_reason": None}]}
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.02)
            yield "data: [DONE]\n\n"
            return
        try:
            async with self.client.stream("POST", "/v1/chat/completions", json={
                "model": self.model,
                "messages": [m.model_dump() for m in messages],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if line.startswith("data:"):
                        yield line + "\n\n"
        except httpx.HTTPError as e:
            raise ProviderError(self.name, f"vLLM stream error: {e}")