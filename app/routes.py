import time
from fastapi import APIRouter
from app.config import get_settings
from app.schemas import (
    ChatCompletionRequest, ChatCompletionResponse,
    ChatMessage, Choice, FlywheelMeta, Usage,
)

router = APIRouter()


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
    start = time.perf_counter()
    settings = get_settings()

    last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    reply = f"[flywheel-stub:{settings.env}] received {len(req.messages)} message(s). Last: {last_user[:80]!r}"

    latency_ms = int((time.perf_counter() - start) * 1000)
    return ChatCompletionResponse(
        model="flywheel-stub",
        choices=[Choice(message=ChatMessage(role="assistant", content=reply))],
        usage=Usage(prompt_tokens=len(last_user.split()), completion_tokens=len(reply.split()),
                    total_tokens=len(last_user.split()) + len(reply.split())),
        flywheel=FlywheelMeta(route="stub", reason="M1 skeleton - providers arrive in M2",
                              latency_ms=latency_ms),
    )