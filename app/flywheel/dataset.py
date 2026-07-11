"""Dataset builder: converts escalated (cloud-answered) traffic into a
Gemma-format SFT dataset. This is the moment API spend becomes an owned asset."""

import datetime
import json
import os
from sqlalchemy import select
from app.config import get_settings
from app.flywheel import registry
from app.memory.db import RequestLog, SessionLocal

MIN_QUALITY = 0.6   # skip cloud answers the evaluator flagged as bad


def build_dataset() -> dict:
    s = get_settings()
    since = registry.last_trained_id()
    with SessionLocal() as db:
        rows = db.execute(
            select(RequestLog)
            .where(RequestLog.route == "cloud", RequestLog.id > since,
                   RequestLog.response != "")
            .order_by(RequestLog.id)).scalars().all()

    rows = [r for r in rows
            if r.quality_score is None or r.quality_score >= MIN_QUALITY]
    if not rows:
        return {"rows": 0, "path": None, "up_to_id": since}

    os.makedirs(s.dataset_dir, exist_ok=True)
    stamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(s.dataset_dir, f"sft_{stamp}.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps({"messages": [
                {"role": "user", "content": r.prompt},
                {"role": "assistant", "content": r.response},
            ]}) + "\n")

        
    with open(path + ".meta", "w") as f:
        json.dump({"up_to_id": rows[-1].id}, f)
    return {"rows": len(rows), "path": path, "up_to_id": rows[-1].id}