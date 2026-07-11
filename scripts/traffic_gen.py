"""Demo traffic generator. Usage: python scripts/traffic_gen.py 40"""
import random
import sys
import time
import httpx

BASE = "http://localhost:8000"
EASY = ["what is kubernetes?", "define machine learning", "hello, how are you?",
        "translate 'thank you' to Spanish", "who is the CEO of Tesla?",
        "summarize: our standup moved to 10am daily."]
HARD = ["Prove step by step that sqrt(2) is irrational.",
        "Refactor this and explain trade-offs:\n```python\ndef f(l):\n  return [x*2 for x in l]\n```",
        "Compare and contrast event sourcing vs CRUD architecture design.",
        "Analyze the pros and cons of microservices for a 5-person startup."]
SENSITIVE = ["My email is raj@example.com, phone 98765 43210 - draft a reply.",
             "Review this patient medical record: diagnosis was hypertension.",
             "My salary is 12 LPA and my bank account needs a loan letter."]

n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
for i in range(n):
    bucket = random.choices([EASY, HARD, SENSITIVE], weights=[6, 3, 1])[0]
    prompt = random.choice(bucket)
    try:
        r = httpx.post(f"{BASE}/v1/chat/completions", timeout=180,
                       json={"messages": [{"role": "user", "content": prompt}]})
        meta = r.json().get("flywheel", {})
        print(f"[{i+1}/{n}] {str(meta.get('route')):6} cached={meta.get('cached')} :: {prompt[:50]}")
    except Exception as e:
        print("error:", e)
    time.sleep(0.4)
