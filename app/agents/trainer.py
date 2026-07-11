"""Trainer Agent: watches escalation volume and decides when a fine-tune
cycle is worth running. Training itself runs via scripts/ on the GPU box."""

from sqlalchemy import func, select
from app.config import get_settings
from app.flywheel import registry
from app.memory.db import RequestLog, SessionLocal


class TrainerAgent:
    def pending_escalations(self) -> int:
        since = registry.last_trained_id()
        with SessionLocal() as s:
            return s.execute(
                select(func.count(RequestLog.id))
                .where(RequestLog.route == "cloud", RequestLog.id > since)
            ).scalar() or 0

    def status(self) -> dict:
        s = get_settings()
        pending = self.pending_escalations()
        active = registry.active_version()
        return {
            "pending_escalations": pending,
            "train_trigger": s.train_trigger_escalations,
            "ready_to_train": pending >= s.train_trigger_escalations,
            "active_version": active["name"] if active else "gemma-base",
            "versions": registry.all_versions(),
        }
    