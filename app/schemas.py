import time
import uuid
from typing import Literal, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible request — clients switch to us by changing one URL."""
    model: str = "flywheel-auto"         
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    stream: bool = False


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class FlywheelMeta(BaseModel):
    """Our extension block - route decision transparency for every response."""
    route: Literal["local", "cloud", "cache", "stub"] = "stub"
    reason: str = ""
    sensitive: bool = False          
    cached: bool = False
    model_version: str = ""          
    quality_score: float | None = None
    cost_usd: float = 0.0
    counterfactual_cost_usd: float = 0.0
    latency_ms: int = 0


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage = Usage()
    flywheel: Optional[FlywheelMeta] = None