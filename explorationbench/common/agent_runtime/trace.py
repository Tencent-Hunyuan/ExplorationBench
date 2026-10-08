from __future__ import annotations

from typing import Any

from common.agent_client import JsonlTraceStore
from common.agent_client.types import json_copy


def append_runtime_event(
    trace_store: JsonlTraceStore | None,
    *,
    event: str,
    framework: str,
    track: str,
    session_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    """Write a framework/runtime audit event without changing trajectory schema."""

    if trace_store is None:
        return
    trace_store.append({
        "record_type": "runtime_event",
        "event": event,
        "framework": framework,
        "track": track,
        "session_id": session_id,
        "payload": json_copy(payload or {}),
    })
