"""One definition of whether a run measured what the protocol asked for.

Both sandboxes write a result file even when the provider never answered part
of the held-out set, and the failure is recorded as a wrong answer: an hour of
throttling once produced a 0/90 that read like a real score and sat in the
tables for days. The harnesses stamp a ``validity`` block onto new results, and
this module derives the same verdict for files written before the stamp
existed, so a single rule decides what the analysis is allowed to average.

The evidence differs by sandbox but means the same thing:

* AlienCode files an unanswered question with an ``error_type``.
* AlienLogic files one with ``diagnostic.reason_class == "WORKER_ERROR"``.
* An AlienCode control round records how many donor probes it owed and how
  many it delivered; a short round saw less evidence than the run it is
  matched against. AlienLogic raises instead of finishing short.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

WORKER_ERROR = "WORKER_ERROR"


def _run_day(path: Path) -> str:
    """When the run in this file was evaluated, for ordering rival files."""

    try:
        stamp = str(json.loads(path.read_text(encoding="utf-8")).get(
            "timestamp") or "")
    except (OSError, ValueError):
        stamp = ""
    if stamp[:2] == "20":
        return stamp
    return dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat()


def _redone_depth(path: Path) -> int:
    """How many repair passes a file carries.

    Repairing a file that was already repaired appends another ``_redone``, so
    the deepest name is the most recent rescore of that run.
    """

    return path.name.count("_redone")


def latest_result(paths) -> Path | None:
    """The newest run among rival files, and within it the latest rescore.

    A ``_redone`` file is the same run rescored after questions lost to
    infrastructure errors were re-asked, so it supersedes its own base file --
    but not a later rerun. Relaunching under the same run id overwrites the
    base file and leaves the old ``_redone`` beside it, so preferring the
    repaired name outright would keep reporting the episode that was replaced.

    Unlike :func:`authoritative_result` this takes the whole candidate set
    rather than a base path, because a base file need not exist: one Qwen
    control survives only as its repairs, and a locator that starts from the
    base name cannot see that repeat at all.
    """

    candidates = [Path(path) for path in paths]
    if not candidates:
        return None
    dated = sorted((_run_day(path), str(path)) for path in candidates)
    newest = dated[-1][0]
    tied = [Path(path) for day, path in dated if day == newest]
    tied.sort(key=lambda path: (-_redone_depth(path), str(path)))
    return tied[0]


def result_variants(directory: Path | str, prefix: str, run_id: str) -> list[Path]:
    """Every scored file for one run id, repairs included.

    The base name is not required to exist: repairs are written beside it and
    outlive a base that was cleaned up or never landed.
    """

    directory = Path(directory)
    if not directory.is_dir():
        return []
    pattern = re.compile(rf"_{re.escape(run_id)}(?:_redone)*\.json$")
    return [
        path
        for path in directory.glob(f"{prefix}*_{run_id}*.json")
        if pattern.search(path.name)
    ]


def authoritative_result(base: Path | str) -> Path:
    """The file that holds a run's real scores: the repair, unless it is stale.

    A repaired ``_redone.json`` re-asks the questions a run lost and normally
    supersedes the file beside it. But re-running the whole cell writes a fresh
    base result and leaves the old repair in place, and preferring it blindly
    then resurrects the damage that was just fixed -- a clean Gemini rerun kept
    being relaunched because a four-hour-old repair still read as its scores.
    A repair is always written after the run it repairs, so a base that is
    newer means the repair belongs to a run that no longer exists.
    """

    base = Path(base)
    repaired = base.with_name(base.stem + "_redone.json")
    if not repaired.is_file():
        return base
    try:
        if base.stat().st_mtime > repaired.stat().st_mtime:
            return base
    except OSError:
        return base
    return repaired

# A run whose provider dropped a couple of calls out of ~450 still measures the
# model; one that lost dozens does not. The threshold only affects the summary
# verdict -- the exact count always travels with it.
DEFAULT_TOLERANCE = 5


def _unanswered(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Graded questions the provider never answered, across both sandboxes."""

    lost: list[dict[str, Any]] = []
    for snapshot in data.get("snapshots") or []:
        for result in snapshot.get("test_results") or []:
            if result.get("error_type"):
                lost.append({
                    "milestone": snapshot.get("milestone_idx"),
                    "task_id": result.get("task_id"),
                    "error_type": result.get("error_type"),
                })
    for record in data.get("milestones") or []:
        for result in record.get("test_results") or []:
            diagnostic = result.get("diagnostic") or {}
            if diagnostic.get("reason_class") == WORKER_ERROR:
                lost.append({
                    "milestone": record.get("milestone"),
                    "task_id": result.get("theorem_id"),
                    "error_type": diagnostic.get("reason_id"),
                })
    return lost


def _short_control_rounds(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Control rounds that delivered fewer donor probes than they owed."""

    return [
        {"loop_idx": record.get("loop_idx"),
         "round_in_loop": record.get("round_in_loop"),
         "expected": record.get("locked_probes_expected"),
         "delivered": record.get("locked_probes_delivered")}
        for record in data.get("explore_records") or []
        if record.get("locked_probes_expected") is not None
        and (record.get("locked_probes_delivered")
             != record.get("locked_probes_expected"))
    ]


def _graded_calls(data: dict[str, Any]) -> int:
    return sum(
        len(record.get("test_results") or [])
        for key in ("snapshots", "milestones")
        for record in (data.get(key) or [])
    )


def _delivered_observations(data: dict[str, Any]) -> int | None:
    """Observations the environment actually returned during exploration.

    ``None`` when the sandbox does not record them, so the check stays silent
    for AlienLogic and for the archived code protocol.
    """

    records = data.get("explore_records") or []
    counted = [
        record["show_count_executed"] for record in records
        if record.get("show_count_executed") is not None
    ]
    return sum(counted) if counted else None


def derive(data: dict[str, Any], *,
           tolerance: int = DEFAULT_TOLERANCE) -> dict[str, Any]:
    """Compute the verdict from the run's own records, ignoring any stamp."""

    unanswered = _unanswered(data)
    short_rounds = _short_control_rounds(data)
    observations = _delivered_observations(data)
    reasons = []
    if len(unanswered) > tolerance:
        reasons.append(f"{len(unanswered)} graded calls never answered")
    if short_rounds:
        reasons.append(f"{len(short_rounds)} control rounds under-delivered")
    # An exploration arm that received nothing measured no exploration. One
    # provider filled every tool call with empty arguments for a whole run and
    # still answered all 630 questions, so the scores looked complete while the
    # model had been told nothing about the world.
    if (observations == 0
            and (data.get("config") or {}).get("control_mode") != "none"):
        reasons.append("exploration returned no observations")
    return {
        "ok": not reasons,
        "reasons": reasons,
        "graded_calls": _graded_calls(data),
        "unanswered_graded_calls": len(unanswered),
        "unanswered": unanswered[:100],
        "short_control_rounds": short_rounds,
        "delivered_observations": observations,
    }


def validity_of(data: dict[str, Any], *,
                tolerance: int = DEFAULT_TOLERANCE) -> dict[str, Any]:
    """The run's verdict: its own stamp when it has one, else derived.

    A stamped run still gets its counts re-checked against the tolerance,
    because the harness stamps any shortfall at all while the analysis is
    willing to keep a run that lost a call or two.
    """

    stamp = data.get("validity")
    if not isinstance(stamp, dict):
        return derive(data, tolerance=tolerance)
    lost = int(stamp.get("unanswered_graded_calls") or 0)
    short = list(stamp.get("short_control_rounds") or [])
    reasons = []
    if lost > tolerance:
        reasons.append(f"{lost} graded calls never answered")
    if short:
        reasons.append(f"{len(short)} control rounds under-delivered")
    # Re-derived rather than read from the stamp: runs written before this
    # check existed carry an "ok" that never looked at whether exploration
    # returned anything.
    observations = _delivered_observations(data)
    if (observations == 0
            and (data.get("config") or {}).get("control_mode") != "none"):
        reasons.append("exploration returned no observations")
    return {**stamp, "ok": not reasons, "reasons": reasons,
            "delivered_observations": observations}


def is_usable(data: dict[str, Any], *,
              tolerance: int = DEFAULT_TOLERANCE) -> bool:
    return bool(validity_of(data, tolerance=tolerance)["ok"])
