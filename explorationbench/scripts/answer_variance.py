#!/usr/bin/env python3
"""answer_variance.py — ask the same question from the same state, repeatedly.

The spread between a model's repeats is wider here than the spread between
models, and so far that has been read as a fact about exploration: one run
infers the rules, another does not. But a repeat differs from its siblings in
two places, not one. It explores differently, and it also answers
differently, because every graded question is its own sampled continuation.

Only the first of those is the thing the benchmark means to measure. This
separates them by holding exploration fixed: one milestone snapshot, the same
sixty-odd questions, asked N times each from N independent forks of that one
state. Whatever disagreement shows up is answering noise, and whatever is
left of the run-to-run spread after subtracting it belongs to exploration.

    python3 dev/scripts/answer_variance.py --run v2_grok46_n3 --trials 3
    python3 dev/scripts/answer_variance.py --run v2_gpt56_n2 --milestones 0,4
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / 'explorationbench'
CODE = DEV / 'sandboxes' / 'code'
for path in (str(DEV), str(CODE)):
    if path not in sys.path:
        sys.path.insert(0, path)

RESULT_DIR = ROOT / 'logs' / 'code' / 'results'
OUT_DIR = ROOT / 'logs' / 'code' / 'variance'


CHECKPOINTS = ROOT / 'logs' / 'code' / 'checkpoints'
BACKFILL = ROOT / 'logs' / 'code' / 'backfill'


def find_result(run_id: str) -> Path:
    matches = sorted(RESULT_DIR.glob(f'eval_results_alien_code_*_{run_id}.json'),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise SystemExit(f'no result file for {run_id!r}')
    return matches[0]


def existing_verdicts(run_id: str, short: str,
                      wanted: list[int]) -> dict[str, bool]:
    """What this run already answered, as the first trial.

    Every question here has been asked once under the protocol and the answer
    was scored and kept. Asking it a third time to obtain a first sample
    would pay for a measurement already on disk, so the run's own verdict is
    trial one and only the rest are bought.

    A run's answers live in its ledger, or -- for the ones whose v2 score was
    assembled from milestone snapshots -- in its backfill file. Both are read
    so the two kinds of run can be compared on the same footing.
    """

    got: dict[str, bool] = {}
    ledger = CHECKPOINTS / f'heldout_alien_code_{short}_{run_id}.jsonl'
    if ledger.exists():
        with ledger.open(encoding='utf-8', errors='replace') as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                job = str(row.get('job') or '')
                milestone = job.partition('|')[0]
                if milestone[1:].isdigit() and int(milestone[1:]) in wanted:
                    result = row.get('result') or {}
                    if not result.get('error_type'):
                        got[job] = bool(result.get('correct'))
    prior = BACKFILL / f'{run_id}.json'
    if prior.exists():
        with prior.open(encoding='utf-8') as handle:
            done = json.load(handle)
        for batch in done.get('snapshots') or []:
            if batch.get('milestone_idx') in wanted:
                for row in batch.get('test_results') or []:
                    got[f"M{batch['milestone_idx']}|{row['task_id']}"] = bool(
                        row.get('correct'))
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--trials', type=int, default=3)
    ap.add_argument('--milestones', default='4')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--extra-rounds', type=int, default=4,
                    help='further passes for questions the provider never '
                         'answered; they are not scored as wrong')
    ap.add_argument('--limit', type=int, default=0,
                    help='only the first N questions, for a quick look')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    source = find_result(args.run)
    with source.open(encoding='utf-8') as fh:
        result = json.load(fh)
    short = result['model_short']
    wanted = [int(x) for x in args.milestones.split(',') if x.strip()]

    os.environ.setdefault('ALIENCODE_PROTOCOL_V2', '1')
    os.environ.setdefault('ALIENCODE_TASK_SET', 'v2')
    # The comparison is only meaningful if the new samples are drawn under
    # the settings the first one was: a different reasoning tier is a
    # different model. The run recorded what it asked for, and asking for
    # anything else here is either a 400 -- grok rejects the default 'max'
    # outright -- or, worse, a silent downgrade.
    asked = (result.get('config') or {}).get('reasoning_effort_asked_for')
    if asked and 'EVAL_REASONING_EFFORT' not in os.environ:
        os.environ['EVAL_REASONING_EFFORT'] = str(asked)
    import run_eval as h

    h.MODEL = result['model']
    h.MODEL_SHORT = short
    # A distinct id keeps this off the run's own ledger: these are extra
    # samples of a question already answered, not a repair of it.
    h.RUN_ID = f'{args.run}_var'

    tasks = list(h.TEST_TASKS)[:args.limit or None]
    snapshots = {s['milestone_idx']: s for s in result.get('snapshots') or []}
    missing = [m for m in wanted if not (
        snapshots.get(m, {}).get('session_snapshot_path')
        and os.path.exists(snapshots[m]['session_snapshot_path']))]
    if missing:
        raise SystemExit(f'{args.run}: no snapshot for milestones {missing}')

    seed = existing_verdicts(args.run, short, wanted)
    # Topped up per question, not per run. A run part-way through its
    # backfill has an answer for the original tasks and none for the newest
    # ones, and a single global trial count gives that second group one
    # sample fewer than everything it will be averaged against.
    wanted_jobs = [f"M{m}|{t['id']}" for m in wanted for t in tasks]
    total = sum(args.trials - (1 if job in seed else 0)
                for job in wanted_jobs)
    # Only the questions this job will actually ask. A ledger also holds
    # answers to tasks the set has since dropped, and counting those reads
    # as more coverage than there is.
    have = sum(1 for job in wanted_jobs if job in seed)
    print(f'{args.run}  ({short})')
    print(f'  {have}/{len(wanted_jobs)} answers already on disk '
          f'count as trial 1')
    print(f'  {len(tasks)} questions × {len(wanted)} milestones, '
          f'topped up to {args.trials} samples each = {total} calls')
    if args.dry_run:
        return 0

    h._init_agent_runtime(None)
    client = h._AGENT_CLIENT

    started = time.time()
    verdicts: dict[str, list[bool]] = collections.defaultdict(list)
    for milestone in wanted:
        for task in tasks:
            job = f"M{milestone}|{task['id']}"
            if job in seed:
                verdicts[job].append(seed[job])

    # Every milestone and every missing sample goes out in one pool. Asking
    # one (milestone, trial) batch at a time made each batch wait for its
    # slowest question -- three capped attempts, about an hour -- so a cell
    # spent most of a day on ten tails. Each trial still gets its own fork of
    # the milestone state, and each job keeps the name the batch-at-a-time
    # loop gave it, so a ledger written either way resumes.
    bases: dict[tuple[int, int], object] = {}

    def base_for(milestone: int, trial: int):
        # A fresh fork per trial: the point is an independent sample of the
        # same state, so the trials must not see each other.
        if (milestone, trial) not in bases:
            snapshot = snapshots[milestone]['session_snapshot_path']
            bases[(milestone, trial)] = h._closed_book_fork(
                client.restore_session(snapshot))
        return bases[(milestone, trial)]

    def ask(plan: list[tuple[int, dict, int]], banner: str) -> None:
        jobs, names, owners = {}, {}, {}
        for index, (milestone, task, trial) in enumerate(plan):
            key = (f'M{milestone}t{trial}', index)
            jobs[key] = {
                'task': task,
                'base_session': base_for(milestone, trial),
                'pending_feedback': None,
                'milestone_idx': milestone,
            }
            names[key] = f"M{milestone}|{task['id']}|t{trial}"
            owners[key] = f"M{milestone}|{task['id']}"
        flat = h._dispatch_code_jobs(
            jobs, names, kind='heldout', n_workers=args.workers,
            banner=banner)
        for key, job in owners.items():
            row = flat.get(key) or {}
            if row.get('error_type'):
                # Not recorded, so the question stays short and a later round
                # asks it again on a fresh fork. Scoring it here would spend a
                # sample slot on the provider's outage and stop the retry.
                # Once the rounds run out the hole is scored as wrong -- see
                # below. A timeout carries no error_type, because running out
                # of the budget is a verdict.
                continue
            verdicts[job].append(bool(row.get('correct')))

    # Only what is still short of a full set of samples. A question whose
    # original answer was reused needs one sample fewer than one added after
    # that run was scored.
    first = [
        (milestone, task, trial)
        for milestone in wanted for task in tasks
        for trial in range(max(
            args.trials - len(verdicts[f"M{milestone}|{task['id']}"]), 0))
    ]
    if first:
        print(f'\n[采样] 一次提交 {len(first)} 个样本'
              f'（{len(wanted)} 个里程碑，每题补足 {args.trials} 遍）',
              flush=True)
        ask(first, f'all milestones ({len(first)} 个样本)')
    # More rounds than samples, because a round can come back without one:
    # those questions are asked again, on fresh forks, against a pool that
    # may have recovered. Trial numbers continue past the first pass so no
    # two samples of a question share a fork or a ledger name.
    for extra in range(args.extra_rounds):
        pending = [
            (milestone, task, args.trials + extra)
            for milestone in wanted for task in tasks
            if len(verdicts[f"M{milestone}|{task['id']}"]) < args.trials
        ]
        if not pending:
            break
        print(f'\n[补题] 第 {extra + 1} 轮：{len(pending)} 题仍缺样本',
              flush=True)
        ask(pending, f'补题 {extra + 1} ({len(pending)} 题)')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f'{args.run}.json'
    reused = sorted(job for job in verdicts if job in seed)
    undersampled = sorted(job for job in wanted_jobs
                          if len(verdicts.get(job) or []) < args.trials)
    # A slot the provider never filled is scored as a wrong answer. The
    # loop above works hard to avoid that -- extra rounds, a fresh fork
    # each time -- but once it is out of rounds, averaging over however
    # many samples arrived would quietly reward the questions that failed
    # most: a question answered once, correctly, would outscore one
    # answered three times with two right. Every question therefore
    # carries the same denominator, and `short_jobs` says which ones were
    # padded so the reader can see the cost.
    scored = {
        job: (verdicts.get(job) or [])
        + [False] * max(args.trials - len(verdicts.get(job) or []), 0)
        for job in wanted_jobs
    }
    stable = sum(1 for v in scored.values() if len(set(v)) == 1)
    payload = {
        'run_id': args.run, 'model_short': short,
        'trials': args.trials, 'reused_first_trial': bool(seed),
        # Which questions opened with the answer the run was scored on.
        # It is per question now, so a reader cannot infer it from a vote
        # count: with the set topped up, every question has `trials` votes
        # whether or not the first one was reused.
        'reused_jobs': reused,
        # Questions the provider never answered enough times. Their
        # missing samples are scored as wrong, so a reader has to be told
        # which questions paid that price.
        'short_jobs': undersampled,
        'complete': not undersampled,
        'scoring': 'missing-sample-counts-wrong',
        'fresh_trials': max(args.trials - 1, 0), 'milestones': wanted,
        'questions': len(tasks), 'elapsed_seconds': time.time() - started,
        'agreement': stable / max(len(scored), 1),
        'mean_score': {
            f'M{m}': (
                sum(sum(v) for k, v in scored.items()
                    if k.startswith(f'M{m}|'))
                / max(sum(args.trials for k in scored
                          if k.startswith(f'M{m}|')), 1)
                * 100
            )
            for m in wanted
        },
        'verdicts': {k: v for k, v in sorted(verdicts.items())},
    }
    with out.open('w', encoding='utf-8') as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print(f"\n  {stable}/{len(verdicts)} questions gave the same verdict "
          f"every time ({payload['agreement'] * 100:.1f}%)")
    if undersampled:
        print(f'  [未采满] {len(undersampled)} 题不足 {args.trials} 个样本：'
              f'{", ".join(undersampled[:8])}'
              f'{" ..." if len(undersampled) > 8 else ""}')
    else:
        print(f'  [采满] 全部 {len(wanted_jobs)} 题各 {args.trials} 个样本')
    for milestone in wanted:
        rows = [v for k, v in verdicts.items() if k.startswith(f'M{milestone}|')]
        depth = min(len(r) for r in rows) if rows else 0
        per_trial = [sum(r[t] for r in rows) for t in range(depth)]
        print(f'  M{milestone} score per trial: {per_trial}  of {len(rows)}')
    print(f'  wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
