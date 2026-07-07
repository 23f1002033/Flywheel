# Flywheel

**The gateway that makes itself cheaper.**

A self-improving hybrid inference gateway: simple requests are served by a
small local model on your own AMD GPU (~$0), complex requests escalate to
frontier models via Fireworks AI - and every escalation is logged as training
data to continuously fine-tune the local model on *your* traffic. Your
inference bill shrinks automatically, week over week.

Built for the AMD Developer Hackathon: ACT II (Unicorn Track).

## Why

Inference is 60–80% of AI operational spend, and it's killing margins across
the industry. Routing startups (worth $500M–$1.3B) solve half the problem -
they optimize *which API* you rent. Flywheel closes the loop: it converts
your API spend into a model you **own**, served on hardware you control.

## Architecture

Client -> Flywheel Gateway (OpenAI-compatible)
       -> Router -> Local tier (Gemma on vLLM / ROCm, AMD GPU)
                 -> Cloud tier (Fireworks AI)
       -> Traffic log -> dataset builder -> LoRA fine-tune (AMD GPU) -> hot-swap

## Quick start

```bash
cp .env.example .env        # add your Fireworks API key
docker compose up --build
curl http://localhost:8000/health
```

Send an OpenAI-style request:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}]}'
```

## Tech stack

FastAPI · Pydantic · vLLM on ROCm (AMD Developer Cloud) · Fireworks AI API ·
Gemma · PEFT/TRL (LoRA) · Docker

## Status

- [x] M1 — Gateway skeleton (OpenAI-compatible endpoint, config, Docker)
- [ ] M2 — Provider layer (Fireworks + local vLLM, streaming)
- [ ] M3 — Router brain (complexity + confidence routing)
- [ ] M4 — Traffic logger + live cost engine
- [ ] M5 — Flywheel loop (auto LoRA fine-tuning on AMD GPUs)
- [ ] M6 — Dashboard
- [ ] M7 — Deployment + demo

## License

MIT