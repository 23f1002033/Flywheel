"""Model version registry: gemma-v1 -> v2 -> v3. JSON on disk - simple,
inspectable, survives restarts. Tracks eval scores and rollout status."""

import datetime
import json
import os
from app.config import get_settings


def _load() -> dict:
    path = get_settings().registry_path
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"versions": [], "last_trained_id": 0}


def _save(data: dict) -> None:
    path = get_settings().registry_path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def active_version() -> dict | None:
    for v in reversed(_load()["versions"]):
        if v["status"] == "active":
            return v
    return None


def all_versions() -> list[dict]:
    return _load()["versions"]


def last_trained_id() -> int:
    return _load().get("last_trained_id", 0)


def register_candidate(adapter_path: str, trained_rows: int, up_to_id: int) -> dict:
    data = _load()
    version = {
        "name": f"gemma-v{len(data['versions']) + 1}",
        "adapter_path": adapter_path,
        "created_at": datetime.datetime.utcnow().isoformat(),
        "trained_rows": trained_rows,
        "eval_score": None,
        "status": "candidate",
    }
    data["versions"].append(version)
    data["last_trained_id"] = up_to_id
    _save(data)
    return version


def set_eval_and_maybe_promote(name: str, score: float, baseline: float) -> str:
    data = _load()
    outcome = "rejected"
    for v in data["versions"]:
        if v["name"] == name:
            v["eval_score"] = score
            if score >= baseline:
                for other in data["versions"]:
                    if other["status"] == "active":
                        other["status"] = "retired"
                v["status"] = "active"
                outcome = "promoted"
            else:
                v["status"] = "rejected"
    _save(data)
    return outcome