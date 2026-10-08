#!/usr/bin/env python3
"""Rescore the sampled artifacts so a missing sample counts as wrong.

The three-sample passes used to average each question over however many
samples arrived. That reads reasonably until you notice what it rewards:
a question answered once, correctly, scored 100 while one answered three
times and right twice scored 67. The questions that cost the provider the
most trouble were the ones whose denominators shrank.

The convention is now the blunt one -- an unfilled slot is a wrong answer,
and every question is divided by `trials`. The samplers write it that way
from here; this pass rewrites the artifacts already on disk so the board,
the figures and the paper all read one number.

Nothing measured is discarded. `verdicts` keeps exactly the samples that
came back and `short_jobs` keeps the list of questions that were padded,
so the old number is still recoverable from the artifact.

    python3 dev/scripts/rescore_missing_as_wrong.py --dry-run
    python3 dev/scripts/rescore_missing_as_wrong.py
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE_VAR = ROOT / 'logs' / 'code' / 'variance'
LOGIC_VAR = ROOT / 'logs' / 'logic' / 'variance'
ORACLE = ROOT / 'logs' / 'code' / 'oracle'
MARK = 'missing-sample-counts-wrong'


def padded(payload: dict) -> dict[str, list[bool]]:
    """Every question's samples, topped up to `trials` with wrong answers.

    The job set is the union of what was answered and what was recorded
    short, so a question that never came back at all is still scored.
    """

    trials = int(payload.get('trials') or 3)
    verdicts = payload.get('verdicts') or {}
    jobs = set(verdicts) | set(payload.get('short_jobs') or [])
    return {
        job: list(verdicts.get(job) or [])
        + [False] * max(trials - len(verdicts.get(job) or []), 0)
        for job in sorted(jobs)
    }


def milestone_mean(scored: dict[str, list[bool]], prefix: str,
                   pick=None) -> float:
    votes = [statistics.mean(v) for job, v in scored.items()
             if job.startswith(prefix) and v
             and (pick is None or pick(job.split('|', 1)[1]))]
    return statistics.mean(votes) * 100 if votes else 0.0


def rescore_code(payload: dict) -> dict[str, float]:
    scored = padded(payload)
    return {name: milestone_mean(scored, f'{name}|')
            for name in payload.get('mean_score') or {}}


def rescore_logic(payload: dict) -> dict[str, dict[str, float]]:
    scored = padded(payload)
    meta = payload.get('theorem_meta') or {}
    aware = (lambda t: meta.get(t, {}).get('alien_aware_required'))
    unprov = (lambda t: meta.get(t, {}).get('alien_provable') is False)
    out = {}
    for field, pick in (('mean_score', None), ('mean_aware', aware),
                        ('mean_unprovable', unprov)):
        if field in payload:
            out[field] = {
                name: milestone_mean(scored, f'{name}|', pick)
                for name in payload[field]}
    return out


def rescore_oracle(payload: dict) -> dict[str, float | None]:
    scored = padded(payload)
    return {label: milestone_mean(scored, f'{label}|')
            for label in payload.get('mean_score') or {}}


def agreement(payload: dict) -> float:
    scored = padded(payload)
    stable = sum(1 for v in scored.values() if len(set(v)) == 1)
    return stable / max(len(scored), 1)


def report(path: Path, before: dict, after: dict, short: int) -> str:
    moved = [
        f'{k} {before[k]:.1f}->{after[k]:.1f}'
        for k in after
        if isinstance(before.get(k), (int, float))
        and abs(before[k] - after[k]) >= 0.05]
    tail = ('  '.join(moved) if moved else '无变化')
    return f'  {path.stem:<18} 缺{short:<3} {tail}'


def walk(paths: list[Path], kind: str, write: bool) -> tuple[int, int]:
    touched = changed = 0
    for path in sorted(paths):
        try:
            with path.open(encoding='utf-8') as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            continue
        short = len(payload.get('short_jobs') or [])
        before = dict(payload.get('mean_score') or {})
        if kind == 'logic':
            fresh = rescore_logic(payload)
            after = fresh.get('mean_score', {})
        elif kind == 'oracle':
            fresh = {'mean_score': rescore_oracle(payload)}
            after = fresh['mean_score']
        else:
            fresh = {'mean_score': rescore_code(payload)}
            after = fresh['mean_score']
        line = report(path, before, after, short)
        moved = '无变化' not in line
        if moved:
            changed += 1
        if moved or payload.get('scoring') != MARK:
            print(line)
        touched += 1
        if not write:
            continue
        payload.update(fresh)
        payload['agreement'] = agreement(payload)
        payload['scoring'] = MARK
        tmp = path.with_suffix('.json.tmp')
        with tmp.open('w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        tmp.replace(path)
    return touched, changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    write = not args.dry_run

    plan = (
        ('code 三遍采样', list(CODE_VAR.glob('v2_*_n*.json')), 'code'),
        ('logic 三遍采样', list(LOGIC_VAR.glob('v2_*_n*.json')), 'logic'),
        ('oracle', list(ORACLE.glob('v2_*_n*.json')), 'oracle'),
    )
    for title, paths, kind in plan:
        print(f'\n════ {title}（{len(paths)} 份）════')
        touched, changed = walk(paths, kind, write)
        print(f'  ── 看过 {touched} 份，分数有变化 {changed} 份')
    if args.dry_run:
        print('\n（空跑，没有写盘）')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
