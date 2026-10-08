"""Cross-process cap on how many requests we aim at one endpoint at a time.

A self-hosted deployment serves a fixed pool of instances and answers HTTP 500
``all instances hit concurrency limit`` once every instance is busy. Retrying
does not help when the saturation is caused by our own sibling runs, so the cap
has to be enforced before the request leaves, and it has to hold across
processes because each evaluation cell runs as its own process.

Slots are files under ``logs/endpoint_gate/<key>``; holding one means holding an
exclusive ``flock`` on it. Enable with ``AGENT_GATE_SLOTS``; ``AGENT_GATE_KEY``
separates unrelated endpoints. When unset the gate is a no-op, so providers that
do not need it are untouched.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import random
import time
from pathlib import Path

_GATE_ROOT = Path(__file__).resolve().parents[3] / "logs" / "endpoint_gate"
_POLL_SECONDS = 0.25


def _settings() -> tuple[int, str] | None:
    raw = os.environ.get("AGENT_GATE_SLOTS", "").strip()
    if not raw:
        return None
    try:
        slots = int(raw)
    except ValueError:
        return None
    if slots < 1:
        return None
    key = os.environ.get("AGENT_GATE_KEY", "default").strip() or "default"
    return slots, "".join(c if c.isalnum() or c in "-_" else "_" for c in key)


@contextlib.contextmanager
def hold_slot():
    """Occupy one slot for the duration of the block."""
    config = _settings()
    if config is None:
        yield
        return
    slots, key = config
    folder = _GATE_ROOT / key
    folder.mkdir(parents=True, exist_ok=True)
    order = list(range(slots))
    handle = None
    # Randomising avoids every waiter stampeding the same low-numbered slot.
    random.shuffle(order)
    while handle is None:
        for index in order:
            candidate = open(folder / f"slot_{index}", "a+")
            try:
                fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                candidate.close()
                continue
            handle = candidate
            break
        if handle is None:
            time.sleep(_POLL_SECONDS + random.uniform(0, _POLL_SECONDS))
            random.shuffle(order)
    try:
        yield
    finally:
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            handle.close()
