from __future__ import annotations

import hashlib
import json
from typing import Any

from common.agent_client.types import json_copy

from .protocol import AgentSessionLike, ArtifactSnapshot, PhaseContext


def stable_digest(value: Any) -> str:
    encoded = json.dumps(
        json_copy(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def closed_book_fork(
    session: AgentSessionLike,
    *,
    artifact: ArtifactSnapshot | None = None,
    context: PhaseContext | None = None,
) -> AgentSessionLike:
    """Fork a graded branch with all environment tools removed."""

    child = session.fork(tools=[])
    # Framework snapshots are recorded by the runtime trace. The session fork
    # itself remains provider-native and tool-free.
    _ = artifact, context
    return child
