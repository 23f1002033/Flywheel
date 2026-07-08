from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator
from app.schemas import ChatMessage


@dataclass
class ProviderResult:
    """Normalized output every provider returns, whatever the backend."""
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    raw: dict = field(default_factory=dict)


class ProviderError(Exception):
    """Raised on upstream failure; carries a safe, loggable message."""
    def __init__(self, provider: str, detail: str):
        self.provider = provider
        self.detail = detail
        super().__init__(f"[{provider}] {detail}")


class BaseProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def complete(self, messages: list[ChatMessage],
                       temperature: float, max_tokens: int) -> ProviderResult: ...

    @abstractmethod
    def stream(self, messages: list[ChatMessage],
               temperature: float, max_tokens: int) -> AsyncIterator[str]:
        """Yields raw SSE lines (already 'data: {...}' formatted) to pass through."""
        ...