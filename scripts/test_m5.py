"""M5 test gate: cache hits, learning router influence, budget status."""
import httpx

BASE = "http://localhost:8000"


def ask(prompt: str) -> dict:
    r = httpx.post(f"{BASE}/v1/chat/completions", timeout=120,
                   json={"messages": [{"role": "user", "content": prompt}]})
    return r.json().get("flywheel", {})


print("1. CACHE TEST")
first = ask("what is the capital of Japan?")
print(f"   first ask : route={first['route']} (expect local/cloud)")
second = ask("what is the capital of Japan?")
print(f"   second ask: route={second['route']} cached={second.get('cached')} "
      f"(expect cache, True)")

print("2. NEAR-DUPLICATE CACHE TEST")
third = ask("what is the capital of Japan??")
print(f"   near-dup  : route={third['route']} (expect cache - semantic, not exact match)")

print("3. LEARNING ROUTER TEST (seed 5 similar, then check reason)")
for i in range(5):
    ask(f"translate 'good morning friend number {i}' to French")
final = ask("translate 'good morning my dear friend' to French")
print(f"   route={final['route']} reason={final['reason'][:90]}")
print("   (expect reason to mention 'memory:' once enough similar traffic exists)")

print("4. BUDGET STATUS")
stats = httpx.get(f"{BASE}/api/stats").json()
print(f"   budget={stats['budget']}")
print(f"   by_route={stats['by_route']}  saved=${stats['total_saved_usd']}")