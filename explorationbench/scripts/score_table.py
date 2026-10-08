#!/usr/bin/env python3
"""Every finished run's milestone scores, one line each.

The viewer (``tools/run_viewer.html``) is where a run gets read: curves, turns,
what the API lost. This is the other question, the one the viewer answers badly
because it needs files dragged in one at a time -- how did all of them do. It
reads the score files directly, prefers a run's repaired scores over its
original ones, and says which runs those are.

    python scripts/score_table.py                 # both sandboxes
    python scripts/score_table.py --sandbox logic
    python scripts/score_table.py --since 2026-08-10
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

DEV = Path(__file__).resolve().parent.parent
ROOT = DEV.parent

# The run's own name carries the model, which the logic score files don't store.
NAME = re.compile(
    r"^eval_results_alien_(?:code|logic)_"
    r"(?:\d{8}_\d{6}_)?"          # some runs still carry a launch stamp
    r"(?P<model>.+?)"
    r"(?:_demo_seeded|_public_demo)?"          # the logic track's fixed world
    r"_(?P<run>[^_]+(?:_[^_]+)*)$"
)


def named(stem: str) -> tuple[str, str]:
    """The model and the run id, as the file name spells them."""

    found = NAME.match(stem)
    if not found:
        return "?", stem
    return found.group("model"), found.group("run")


def day_of(data: dict[str, Any], path: Path) -> str:
    """When the run happened, falling back to the file for runs with no stamp."""

    stamp = str(data.get("timestamp") or data.get("started_at") or "")
    if stamp[:2] == "20":
        return stamp[:10]
    return date.fromtimestamp(path.stat().st_mtime).isoformat()


def milestones(data: dict[str, Any]) -> list[tuple[str, int, int]]:
    """The graded milestones as (label, passed, asked).

    AlienCode files them under `snapshots` and counts `test_correct`; AlienLogic
    under `milestones`, counting `test_pass`.
    """

    out = []
    for index, record in enumerate(data.get("milestones") or data.get("snapshots") or []):
        passed = record.get("test_correct", record.get("test_pass"))
        asked = record.get("test_total")
        if passed is None or not asked:
            continue
        label = record.get("milestone", record.get("milestone_idx", index))
        out.append((str(label), passed, asked))
    return out


def runs(sandbox: str, since: str) -> list[dict[str, Any]]:
    found = []
    for path in sorted((ROOT / f"logs/{sandbox}/results").glob("eval_results_*.json")):
        if path.stem.endswith("_redone"):
            continue
        repaired = path.with_name(path.stem + "_redone.json")
        source = repaired if repaired.exists() else path
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        day = day_of(data, path)
        if since and day < since:
            continue
        scores = milestones(data)
        if not scores:
            continue
        model, run = named(path.stem)
        found.append({
            "model": model,
            "run": run,
            "effort": (data.get("config") or {}).get("reasoning_effort", "-"),
            "day": day,
            "repaired": repaired.exists(),
            "scores": scores,
        })
    return sorted(found, key=lambda row: (row["model"], row["effort"], row["run"]))


def show(sandbox: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n{'=' * 104}\n{sandbox.upper()}  {len(rows)} 轮\n{'=' * 104}")
    for row in rows:
        tail = row["scores"][-1]
        print(f"{row['model'][:32]:<33} {str(row['effort']):<7} "
              f"{row['run'][:22]:<23} {row['day']}"
              f"{'  已补做' if row['repaired'] else ''}")
        print("    " + "  ".join(f"M{l}:{p}/{t}" for l, p, t in row["scores"])
              + f"   末档 {tail[1] / tail[2] * 100:.0f}%")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sandbox", choices=["code", "logic"], action="append",
                        default=[])
    parser.add_argument("--since", default="",
                        help="skip runs before this day, e.g. 2026-08-10")
    args = parser.parse_args()

    for sandbox in args.sandbox or ["code", "logic"]:
        show(sandbox, runs(sandbox, args.since))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
