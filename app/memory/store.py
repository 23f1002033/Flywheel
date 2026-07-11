import datetime
import logging
import numpy as np
from sqlalchemy import case, func, select
from app.memory.db import RequestLog, SessionLocal, init_db
from app.memory import embeddings as emb

log = logging.getLogger("flywheel.memory")


class MemoryStore:
    def __init__(self) -> None:
        init_db()

    def log_request(self, **kw) -> int:
        text = kw.pop("prompt", "")
        record = RequestLog(prompt=text, embedding=emb.to_bytes(emb.embed(text)), **kw)
        with SessionLocal() as s:
            s.add(record)
            s.commit()
            return record.id

    def set_quality(self, request_id: int, score: float) -> None:
        with SessionLocal() as s:
            row = s.get(RequestLog, request_id)
            if row:
                row.quality_score = score
                s.commit()

    def timeline(self, days: int = 14) -> list[dict]:
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=days)
        with SessionLocal() as s:
            rows = s.execute(
                select(func.date(RequestLog.created_at).label("day"),
                       func.sum(RequestLog.saved_usd),
                       func.sum(RequestLog.cost_usd),
                       func.count(RequestLog.id),
                       func.sum(case((RequestLog.route != "cloud", 1), else_=0)))
                .where(RequestLog.created_at >= cutoff)
                .group_by("day").order_by("day")).all()
        return [{"day": str(d), "saved_usd": round(sv or 0, 6),
                 "cost_usd": round(c or 0, 6), "requests": n,
                 "local_ratio": round((loc or 0) / n, 4) if n else 0}
                for d, sv, c, n, loc in rows]

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