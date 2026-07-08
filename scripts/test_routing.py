"""M3 test gate: sends labeled prompts through the gateway and checks
each one lands on the expected route. Run with the server up:
    python scripts/test_routing.py
"""
import httpx

BASE = "http://localhost:8000"

CASES = [
    ("hello there!", "local"),
    ("what is the capital of France?", "local"),
    ("summarize this: the meeting moved to Friday.", "local"),
    ("Prove that the sum of two even numbers is even, step by step.", "cloud"),
    ("Refactor this function and explain the trade-offs:\n```python\ndef f(x): return x\n```", "cloud"),
    ("Compare and contrast microservices vs monolith architecture design.", "cloud"),
    ("My email is priya@example.com and phone 98765 43210, draft a reply.", "local"),   # PII pinned
    ("Analyze this patient medical record for diagnosis history.", "local"),            # healthcare pinned
]

passed = 0
for prompt, expected in CASES:
    r = httpx.post(f"{BASE}/v1/chat/completions", timeout=120,
                   json={"messages": [{"role": "user", "content": prompt}]})
    meta = r.json().get("flywheel", {})
    ok = meta.get("route") == expected
    passed += ok
    print(f"[{'PASS' if ok else 'FAIL'}] expected={expected:5} got={meta.get('route'):5} "
          f"sensitive={meta.get('sensitive')} :: {meta.get('reason', '')[:70]}")

print(f"\n{passed}/{len(CASES)} cases passed")