"""Step 7 - LLMOps monitoring.

Every interaction with the assistant is appended to a JSONL log
(``outputs/interactions.jsonl``) with latency, token usage, the provider that
served it, and which documents were retrieved.  :func:`load_interactions` and
:func:`usage_report` read that log back for the "Monitoring" tab of the
Streamlit app.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from insightforge.config import settings


def log_interaction(record: dict[str, Any], path: Path | None = None) -> None:
    """Append one interaction record to the JSONL log."""
    path = Path(path) if path is not None else settings.log_file
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def load_interactions(path: Path | None = None) -> list[dict]:
    """Read the interaction log back (skipping any corrupt line)."""
    path = Path(path) if path is not None else settings.log_file
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def usage_report(records: list[dict] | None = None) -> dict:
    """Aggregate the log into the handful of numbers worth showing."""
    records = load_interactions() if records is None else records
    if not records:
        return {
            "interactions": 0,
            "avg_latency_s": None,
            "p95_latency_s": None,
            "total_tokens": 0,
            "errors": 0,
            "providers": {},
        }

    latencies = sorted(r["latency_s"] for r in records if r.get("latency_s") is not None)
    tokens = sum(r.get("total_tokens") or 0 for r in records)
    providers: dict[str, int] = {}
    for record in records:
        provider = record.get("provider", "unknown")
        providers[provider] = providers.get(provider, 0) + 1

    def percentile(values: list[float], pct: float) -> float | None:
        if not values:
            return None
        index = min(len(values) - 1, int(round(pct * (len(values) - 1))))
        return round(values[index], 3)

    return {
        "interactions": len(records),
        "avg_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "p95_latency_s": percentile(latencies, 0.95),
        "max_latency_s": round(latencies[-1], 3) if latencies else None,
        "total_tokens": tokens,
        "errors": sum(1 for r in records if r.get("error")),
        "providers": providers,
    }


@contextmanager
def timed() -> Iterator[dict]:
    """``with timed() as t:`` -> ``t['elapsed']`` holds the duration in seconds."""
    marker: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield marker
    finally:
        marker["elapsed"] = round(time.perf_counter() - start, 3)


def extract_token_usage(response: Any) -> dict[str, int | None]:
    """Pull token counts out of a LangChain message, if the provider sent them."""
    meta = getattr(response, "usage_metadata", None) or {}
    if not meta:
        meta = (getattr(response, "response_metadata", {}) or {}).get("token_usage", {}) or {}
    return {
        "input_tokens": meta.get("input_tokens") or meta.get("prompt_tokens"),
        "output_tokens": meta.get("output_tokens") or meta.get("completion_tokens"),
        "total_tokens": meta.get("total_tokens"),
    }
