"""Process-lifetime lock for one evaluation run id."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import TextIO

ROOT = Path(__file__).resolve().parents[2]
LOCK_DIR = ROOT / "logs" / "control_locks"


def acquire_run_lock(run_id: str) -> TextIO | None:
    """Hold an exclusive lock until this process exits.

    New launchers wrap evaluations with the system ``flock`` command and set
    ``EVAL_RUN_LOCK_HELD``. This internal fallback protects evaluations started
    by an older, already-running launcher that does not know about that wrapper.
    """

    if not run_id or os.environ.get("EVAL_RUN_LOCK_HELD") == "1":
        return None
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    handle = (LOCK_DIR / f"{run_id}.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise SystemExit(75)
    return handle
