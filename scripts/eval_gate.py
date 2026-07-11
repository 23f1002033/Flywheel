"""Eval gate: candidate must beat (or match) the current local model on a
held-out sample before promotion. Judge = cloud model via Fireworks.
Usage (GPU box, with candidate vLLM on :8002 and current on :8001):
    python scripts/eval_gate.py gemma-v2 --candidate http://localhost:8002 --current http://localhost:8001
"""

import argparse
import json
import os
import random
import sys

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import get_settings
from app.flywheel import registry

JUDGE_PROMPT = """You are grading an AI assistant's answer. Score it 0.0-1.0 for
correctness, helpfulness and clarity given the user prompt. Respond with ONLY
the number.

User prompt:
{prompt}

Assistant answer:
{answer}"""

N_SAMPLES = 20


def ask(base_url: str, prompt: str) -> str:
    r = httpx.post(f"{base_url}/v1/chat/completions", timeout=120, json={
        "model": "local", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512})
    return r.json()["choices"][0]["message"]["content"]


def judge(prompt: str, answer: str) -> float:
    s = get_settings()
    r = httpx.post(f"{s.fireworks_base_url}/chat/completions",
                   headers={"Authorization": f"Bearer {s.fireworks_api_key}"},
                   timeout=60, json={
                       "model": s.fireworks_model, "max_tokens": 8,
                       "messages": [{"role": "user", "content":
                                     JUDGE_PROMPT.format(prompt=prompt, answer=answer)}]})
    try:
        return max(0.0, min(1.0, float(r.json()["choices"][0]["message"]["content"].strip())))
    except (ValueError, KeyError):
        return 0.5


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("version_name")
    p.add_argument("--candidate", required=True)
    p.add_argument("--current", required=True)
    p.add_argument("--dataset", default=None, help="held-out jsonl; defaults to newest")
    args = p.parse_args()

    s = get_settings()
    dataset = args.dataset or sorted(
        f for f in os.listdir(s.dataset_dir) if f.endswith(".jsonl"))[-1]
    dataset = os.path.join(s.dataset_dir, dataset) if not os.path.isabs(dataset) else dataset

    prompts = [json.loads(line)["messages"][0]["content"] for line in open(dataset)]
    random.seed(42)
    sample = random.sample(prompts, min(N_SAMPLES, len(prompts)))

    cand_scores, curr_scores = [], []
    for i, prompt in enumerate(sample, 1):
        cand_scores.append(judge(prompt, ask(args.candidate, prompt)))
        curr_scores.append(judge(prompt, ask(args.current, prompt)))
        print(f"[{i}/{len(sample)}] candidate={cand_scores[-1]:.2f} current={curr_scores[-1]:.2f}")

    cand, curr = sum(cand_scores) / len(cand_scores), sum(curr_scores) / len(curr_scores)
    outcome = registry.set_eval_and_maybe_promote(args.version_name, cand, curr)
    print(f"\ncandidate avg={cand:.3f} vs current avg={curr:.3f} -> {outcome.upper()}")
    if outcome == "promoted":
        print("Restart vLLM with the new adapter (scripts/run_vllm.sh) to hot-swap.")


if __name__ == "__main__":
    main()