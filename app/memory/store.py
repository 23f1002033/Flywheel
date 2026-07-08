import datetime
import logging
import numpy as np
from sqlalchemy import func, select
from app.memory.db import RequestLog, SessionLocal, init_db
from app.memory import embeddings as emb

log = logging.getLogger("flywheel.memory")


class MemoryStore:
    def __init__(self) -> None:
        init_db()

    def log_request(self, **kw) -> None:
        text = kw.pop("prompt", "")
        record = RequestLog(prompt=text, embedding=emb.to_bytes(emb.embed(text)), **kw)
        with SessionLocal() as s:
            s.add(record)
            s.commit()

    def similar(self, text: str, k: int = 8, scan: int = 500) -> list[dict]:
        """Nearest neighbors among the last `scan` records (brute force -
        fine at hackathon scale; FAISS is a roadmap line)."""
        query_vec = emb.embed(text)
        with SessionLocal() as s:
            rows = s.execute(select(RequestLog)
                             .where(RequestLog.embedding.is_not(None))
                             .order_by(RequestLog.id.desc()).limit(scan)).scalars().all()
        scored = []
        for r in rows:
            sim = emb.cosine(query_vec, emb.from_bytes(r.embedding))
            scored.append((sim, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"similarity": round(sim, 4), "route": r.route, "cached": r.cached,
                 "quality_score": r.quality_score, "response": r.response,
                 "prompt": r.prompt, "id": r.id}
                for sim, r in scored[:k]]

    def month_spend(self) -> float:
        start = datetime.datetime.utcnow().replace(day=1, hour=0, minute=0,
                                                   second=0, microsecond=0)
        with SessionLocal() as s:
            total = s.execute(select(func.coalesce(func.sum(RequestLog.cost_usd), 0.0))
                              .where(RequestLog.created_at >= start)).scalar()
        return float(total or 0.0)

    def stats(self) -> dict:
        with SessionLocal() as s:
            total = s.execute(select(func.count(RequestLog.id))).scalar() or 0
            by_route = dict(s.execute(
                select(RequestLog.route, func.count(RequestLog.id))
                .group_by(RequestLog.route)).all())
            sums = s.execute(select(
                func.coalesce(func.sum(RequestLog.cost_usd), 0.0),
                func.coalesce(func.sum(RequestLog.counterfactual_usd), 0.0),
                func.coalesce(func.sum(RequestLog.saved_usd), 0.0),
                func.coalesce(func.sum(RequestLog.co2_saved_grams), 0.0),
                func.coalesce(func.avg(RequestLog.latency_ms), 0.0),
            )).one()
        local_like = by_route.get("local", 0) + by_route.get("cache", 0)
        return {
            "total_requests": total,
            "by_route": by_route,
            "local_ratio": round(local_like / total, 4) if total else 0.0,
            "total_cost_usd": round(sums[0], 6),
            "counterfactual_usd": round(sums[1], 6),
            "total_saved_usd": round(sums[2], 6),
            "co2_saved_grams": round(sums[3], 2),
            "avg_latency_ms": round(sums[4], 1),
            "month_spend_usd": round(self.month_spend(), 6),
        }

    def recent(self, n: int = 50) -> list[dict]:
        with SessionLocal() as s:
            rows = s.execute(select(RequestLog)
                             .order_by(RequestLog.id.desc()).limit(n)).scalars().all()
        return [{"id": r.id, "created_at": r.created_at.isoformat(),
                 "route": r.route, "reason": r.reason, "sensitive": r.sensitive,
                 "cached": r.cached, "latency_ms": r.latency_ms,
                 "cost_usd": r.cost_usd, "saved_usd": r.saved_usd,
                 "quality_score": r.quality_score,
                 "prompt_preview": r.prompt[:80]} for r in rows]