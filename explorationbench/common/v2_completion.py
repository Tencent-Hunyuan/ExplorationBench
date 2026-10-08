"""What it means for an AlienCode v2 run to be finished.

This lived in three copies -- the launcher, the cohort replay driver and the
dashboard -- and they drifted. Two runs whose exploration executed nothing at
all were counted as complete by the launcher and excluded by the analysis at
the same time, so the scheduler never replaced them and the leaderboard quietly
lost a repeat. One definition, called from all three.

    python3 -c 'import sys; sys.path.insert(0, "dev");
                from common.v2_completion import main; main()' RESULT.json
"""
from __future__ import annotations

import json
import sys
from typing import Any

from . import run_validity


def _batch_complete(row: dict[str, Any], expected: int) -> bool:
    results = row.get('test_results') or []
    return (
        row.get('test_total') == expected
        and len(results) == expected
        and not any(item.get('error_type') for item in results)
    )


def _task_count(data: dict[str, Any]) -> int:
    """How many questions this run was supposed to answer per batch.

    Read from the run's own manifest rather than assumed, because the task set
    changed size: the legacy set asked 90 and v2 asks 60. A run is complete
    when it answered its own set, not someone else's.
    """
    manifest = data.get('protocol_manifest') or {}
    recorded = manifest.get('task_count')
    if isinstance(recorded, int) and recorded > 0:
        return recorded
    sizes = {row.get('test_total') for row in (data.get('snapshots') or [])}
    sizes.discard(None)
    return max(sizes) if sizes else 90


def state(data: dict[str, Any]) -> tuple[str, str]:
    """Return (state, detail) where state is complete | damaged."""
    if not (data.get('config') or {}).get('protocol_v2'):
        return 'damaged', 'not a v2 run'
    snapshots = data.get('snapshots') or []
    oracle = data.get('oracle_diagnostics') or []
    if len(data.get('milestones') or []) < 5 or len(snapshots) != 5:
        return 'damaged', f'{len(snapshots)}/5 milestones'
    expected = _task_count(data)
    bad = [f'M{row.get("milestone_idx")}' for row in snapshots
           if not _batch_complete(row, expected)]
    if bad:
        return 'damaged', 'incomplete held-out: ' + ', '.join(bad)
    if len(oracle) != 2:
        return 'damaged', f'{len(oracle)}/2 oracle diagnostics'
    bad = [str(row.get('label')) for row in oracle
           if not _batch_complete(row, expected)]
    if bad:
        return 'damaged', 'incomplete oracle: ' + ', '.join(bad)
    # Derived, never read from the run's own stamp: a run written before a
    # check existed carries an "ok" that never applied it.
    verdict = run_validity.validity_of(data)
    if not verdict['ok']:
        return 'damaged', '; '.join(verdict['reasons'])
    return 'complete', 'done'


def is_complete(data: dict[str, Any]) -> bool:
    return state(data)[0] == 'complete'


def exploration_done(checkpoint: dict[str, Any]) -> bool:
    """True when a checkpoint has walked M0..M4 but not yet scored anything.

    This is the state an explore-first sweep leaves behind: every milestone
    session is snapshotted and the held-out questions are queued against those
    snapshots, so the 450 answers can be collected later without the model
    needing to explore again.
    """

    phases = set(checkpoint.get('done_phases') or [])
    return 'milestone_4' in phases and 'heldout_flushed' not in phases


def main() -> None:
    """Exit 0 when the result file named in argv[1] is a complete run."""
    try:
        with open(sys.argv[1], encoding='utf-8') as handle:
            data = json.load(handle)
    except Exception:                                        # noqa: BLE001
        sys.exit(1)
    verdict, detail = state(data)
    if verdict != 'complete':
        print(detail, file=sys.stderr)
    sys.exit(0 if verdict == 'complete' else 1)


def explored_main() -> None:
    """Exit 0 when the checkpoint named in argv[1] has finished exploring."""
    try:
        with open(sys.argv[1], encoding='utf-8') as handle:
            checkpoint = json.load(handle)
    except Exception:                                        # noqa: BLE001
        sys.exit(1)
    sys.exit(0 if exploration_done(checkpoint) else 1)
