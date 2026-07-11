import asyncio
import logging
import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app import costing
from app.agents.cache import SemanticCache
from app.agents.cost_guard import CostAgent
from app.agents.memory_router import MemoryHint
from app.agents.router import RouterAgent
from app.memory.store import MemoryStore
from app.providers.base import ProviderError
from app.providers.registry import get_provider
from app.schemas import (
    ChatCompletionRequest, ChatCompletionResponse,
    ChatMessage, Choice, FlywheelMeta, Usage,
)
from app.agents.evaluator import EvaluatorAgent
from app.agents.trainer import TrainerAgent
from app.flywheel.dataset import build_dataset

log = logging.getLogger("flywheel.routes")
router = APIRouter()

_memory = MemoryStore()
_cache = SemanticCache(_memory)
_cost_agent = CostAgent(_memory)
_router_agent = RouterAgent(memory_hint=MemoryHint(_memory),
                            budget_pressure=_cost_agent.pressure)

_trainer = TrainerAgent()
_evaluator = EvaluatorAgent(_memory)

def _prompt_text(req: ChatCompletionRequest) -> str:
    return " ".join(m.content for m in req.messages if m.role != "system")

@router.get("/api/flywheel/status")
async def flywheel_status():
    return _trainer.status()


@router.post("/api/flywheel/build-dataset")
async def flywheel_build_dataset():
    return build_dataset()

@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    decision = _router_agent.decide(req.model, req.messages)
    log.info("route=%s | %s", decision.route, decision.reason)
    prompt_text = _prompt_text(req)
    start = time.perf_counter()

    if not req.stream and not decision.sensitive and req.model == "flywheel-auto":
        hit = _cache.lookup(prompt_text)
        if hit:
            latency_ms = int((time.perf_counter() - start) * 1000)
            est_tokens = len(hit["response"].split())
            report = costing.compute("cache", len(prompt_text.split()), est_tokens)
            _memory.log_request(
                prompt=prompt_text, response=hit["response"], route="cache",
                reason=f"semantic cache hit (sim={hit['similarity']})",
                cached=True, model="flywheel-cache",
                prompt_tokens=len(prompt_text.split()), completion_tokens=est_tokens,
                latency_ms=latency_ms, cost_usd=0.0,
                counterfactual_usd=report.counterfactual_usd,
                saved_usd=report.counterfactual_usd,
                co2_saved_grams=report.co2_saved_grams)
            return ChatCompletionResponse(
                model="flywheel-cache",
                choices=[Choice(message=ChatMessage(role="assistant",
                                                    content=hit["response"]))],
                usage=Usage(),
                flywheel=FlywheelMeta(route="cache", cached=True,
                                      reason=f"semantic cache hit (sim={hit['similarity']})",
                                      counterfactual_cost_usd=report.counterfactual_usd,
                                      latency_ms=latency_ms))

    provider = get_provider(decision.route)

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

    rid = _memory.log_request(
        prompt=prompt_text, response=result.content,
        route=decision.route, reason=decision.reason, sensitive=decision.sensitive,
        model=result.model, prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens, latency_ms=latency_ms,
        cost_usd=report.cost_usd, counterfactual_usd=report.counterfactual_usd,
        saved_usd=report.saved_usd, co2_saved_grams=report.co2_saved_grams)

    if decision.route == "local" and not decision.sensitive and _evaluator.should_sample():
        asyncio.create_task(_evaluator.judge(rid, prompt_text, result.content))

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
                              counterfactual_cost_usd=report.counterfactual_usd))


@router.get("/api/stats")
async def api_stats():
    return {**_memory.stats(), "budget": _cost_agent.status()}


@router.get("/api/requests/recent")
async def api_recent(n: int = 50):
    return _memory.recent(min(n, 200))

@router.get("/api/stats/timeline")
async def api_timeline(days: int = 14):
    return _memory.timeline(min(days, 90))
