import logging
import time
from functools import lru_cache
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app import costing
from app.memory.store import MemoryStore
from app.agents.router import RouterAgent
from app.providers.base import ProviderError
from app.providers.registry import get_provider
from app.schemas import (
    ChatCompletionRequest, ChatCompletionResponse,
    ChatMessage, Choice, FlywheelMeta, Usage,
)

log = logging.getLogger("flywheel.routes")
router = APIRouter()

_memory = MemoryStore()


@lru_cache
def get_router_agent() -> RouterAgent:
    return RouterAgent()


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    decision = get_router_agent().decide(req.model, req.messages)
    log.info("route=%s | %s", decision.route, decision.reason)
    provider = get_provider(decision.route)
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
    report = costing.compute(decision.route, result.prompt_tokens, result.completion_tokens)

    prompt_text = " ".join(m.content for m in req.messages if m.role != "system")
    _memory.log_request(
        prompt=prompt_text,
        response=result.content,
        route=decision.route,
        reason=decision.reason,
        sensitive=decision.sensitive,
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        latency_ms=latency_ms,
        cost_usd=report.cost_usd,
        counterfactual_usd=report.counterfactual_usd,
        saved_usd=report.saved_usd,
        co2_saved_grams=report.co2_saved_grams,
    )

    return ChatCompletionResponse(
        model=result.model,
        choices=[Choice(message=ChatMessage(role="assistant", content=result.content),
                        finish_reason=result.finish_reason)],
        usage=Usage(prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    total_tokens=result.prompt_tokens + result.completion_tokens),
        flywheel=FlywheelMeta(route=decision.route, reason=decision.reason,
                              sensitive=decision.sensitive, latency_ms=latency_ms,
                              cost_usd=report.cost_usd,
                              counterfactual_cost_usd=report.counterfactual_usd),
    )


@router.get("/api/stats")
async def api_stats():
    return _memory.stats()


@router.get("/api/requests/recent")
async def api_recent(n: int = 50):
    return _memory.recent(min(n, 200))