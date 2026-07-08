import logging
import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.providers.base import ProviderError
from app.providers.registry import get_provider
from app.schemas import (
    ChatCompletionRequest, ChatCompletionResponse,
    ChatMessage, Choice, FlywheelMeta, Usage,
)

log = logging.getLogger("flywheel.routes")
router = APIRouter()


def resolve_route(model: str) -> tuple[str, str]:
    """Map requested model name to (route, reason).
    M2: 'flywheel-auto' defaults to local. M3 replaces this with the router brain."""
    if model == "flywheel-cloud":
        return "cloud", "client forced cloud"
    if model == "flywheel-local":
        return "local", "client forced local"
    return "local", "auto (M2 default: local; router brain lands in M3)"


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    route, reason = resolve_route(req.model)
    provider = get_provider(route)
    start = time.perf_counter()

    if req.stream:
        try:
            gen = provider.stream(req.messages, req.temperature, req.max_tokens)
            return StreamingResponse(gen, media_type="text/event-stream")
        except ProviderError as e:
            raise HTTPException(status_code=502, detail=str(e))

    try:
        result = await provider.complete(req.messages, req.temperature, req.max_tokens)
    except ProviderError as e:
        log.error("provider failure: %s", e)
        raise HTTPException(status_code=502, detail=str(e))

    latency_ms = int((time.perf_counter() - start) * 1000)
    return ChatCompletionResponse(
        model=result.model,
        choices=[Choice(message=ChatMessage(role="assistant", content=result.content),
                        finish_reason=result.finish_reason)],
        usage=Usage(prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    total_tokens=result.prompt_tokens + result.completion_tokens),
        flywheel=FlywheelMeta(route=route, reason=reason, latency_ms=latency_ms),
    )