#!/usr/bin/env python3
"""Re-ask a run's oracle diagnostics on the current task set, three times.

The diagnostics stored inside each result file cannot be compared across
the cohort. They were computed when the run was launched, against
whichever held-out set was current then, so the thirty-three v2 runs
carry oracle numbers over sixty, seventy and ninety tasks -- only one of
them over the seventy that the rest of the paper now reports. An oracle
delta taken against a differently-sized M4 is not a delta.

They are also single answers, while every other number here is a mean of
three. Roughly one question in six changes its verdict between samples,
so a single-sample oracle score carries noise the endpoint it is
subtracted from no longer has.

So both are re-done together: the same two disclosures, from the same
milestone snapshots the run recorded, over the current seventy tasks,
three independent times each.

    python3 dev/scripts/oracle_variance.py --run v2_opus5_n1
    python3 dev/scripts/oracle_variance.py --run v2_opus5_n1 --dry-run
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / 'explorationbench'
for entry in (DEV, DEV / 'sandboxes' / 'code'):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

RESULTS = ROOT / 'logs' / 'code' / 'results'
OUT_DIR = ROOT / 'logs' / 'code' / 'oracle'

#: Which milestone each disclosure forks from. O@M0 asks what the rules
#: are worth to a system that has not explored; A4+O asks what is left
#: after exploration once the rules are no longer the obstacle.
LABELS = (('O@M0', 0), ('A4+O', 4))
CHECKPOINTS = ROOT / 'logs' / 'code' / 'checkpoints'


def prior_samples(run: str, short: str | None,
                  labels: list[str]) -> dict[str, list[bool]]:
    """Samples this pass already took, read back from its own ledger.

    Every answer here was drawn under the same disclosure from the same
    forked state, so it is a sample of exactly the quantity being
    measured -- there is no reason to buy it twice. Without this a restart
    began again from nothing, which is what a supervisor does whenever a
    lane dies, so two lanes spent two and a half hours reaching halfway
    and would have lost all of it to one more crash.

    Trials are capped by the caller, so a ledger holding more than
    `--trials` samples of a question simply contributes the first few.
    """

    got: dict[str, list[bool]] = {}
    if not short:
        return got
    ledger = CHECKPOINTS / f'oracle_alien_code_{short}_{run}_oracle.jsonl'
    if not ledger.is_file():
        return got
    with ledger.open(encoding='utf-8', errors='replace') as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            # A ledger row is named per trial -- `O@M0|A47|t0` -- because
            # each trial is its own dispatch. The question is the first two
            # fields; the trial index is what makes it a separate sample.
            parts = str(row.get('job') or '').split('|')
            if len(parts) < 2 or parts[0] not in labels:
                continue
            result = row.get('result') or {}
            if result.get('error_type'):
                continue
            got.setdefault(f'{parts[0]}|{parts[1]}', []).append(
                bool(result.get('correct')))
    return got

#: How stubbornly to chase the disclosure itself. Generous, because it is
#: one call that gates seventy, and the failures it sees are transient:
#: a token-rate 429 on the Anthropic routes, an empty account pool on the
#: gateway. Both clear in minutes.
ACK_ATTEMPTS = int(os.environ.get('ORC_ACK_ATTEMPTS', '6'))
ACK_BACKOFF = float(os.environ.get('ORC_ACK_BACKOFF', '45'))


def find_result(run: str) -> Path:
    matches = sorted(RESULTS.glob(f'eval_results_alien_code_*_{run}.json'),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise SystemExit(f'no result for {run}')
    return matches[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--trials', type=int, default=3)
    ap.add_argument('--workers', type=int, default=30)
    ap.add_argument('--extra-rounds', type=int, default=3)
    ap.add_argument('--dry-run', action='store_true')
    # The two disclosures are independent measurements that happen to share
    # a run: one forks the state before any exploration, the other the
    # state after all of it, and neither reads the other. Running them in
    # one process made a lane spend two hours on the first before starting
    # the second, for no reason but the order of a loop. Naming one lets
    # them run side by side and merge into the same artifact.
    ap.add_argument('--labels', default=','.join(l for l, _ in LABELS),
                    help='which disclosures to measure, comma separated')
    args = ap.parse_args()

    source = find_result(args.run)
    with source.open(encoding='utf-8') as handle:
        result = json.load(handle)
    config = result.get('config') or {}

    os.environ.setdefault('ALIENCODE_PROTOCOL_V2', '1')
    os.environ.setdefault('ALIENCODE_TASK_SET', 'v2')
    asked = config.get('reasoning_effort_asked_for')
    if asked and 'EVAL_REASONING_EFFORT' not in os.environ:
        os.environ['EVAL_REASONING_EFFORT'] = str(asked)

    import run_eval as harness
    from oracle_rules import build_oracle_message

    asked_labels = [name.strip() for name in args.labels.split(',')
                    if name.strip()]
    unknown = [name for name in asked_labels
               if name not in {label for label, _ in LABELS}]
    if unknown:
        raise SystemExit(f'unknown disclosure {unknown}; '
                         f'known: {[label for label, _ in LABELS]}')
    chosen = [(label, index) for label, index in LABELS
              if label in asked_labels]

    snapshots = {s['milestone_idx']: s for s in result.get('snapshots') or []}
    wanted = [(label, index) for label, index in chosen
              if snapshots.get(index, {}).get('session_snapshot_path')
              and os.path.exists(snapshots[index]['session_snapshot_path'])]
    missing = [label for label, _ in chosen
               if label not in {name for name, _ in wanted}]
    if missing:
        raise SystemExit(
            f'{args.run}: no milestone snapshot for {missing}; the '
            f'disclosure has to fork from the state it is measuring')

    tasks = list(harness.TEST_TASKS)
    total = len(wanted) * len(tasks) * args.trials
    print(f'{args.run}  ({result.get("model_short")})')
    print(f'  {len(wanted)} 个披露 × {len(tasks)} 题 × {args.trials} 次 '
          f'= {total} 次调用')
    if args.dry_run:
        return 0

    harness.MODEL = result['model']
    harness.MODEL_SHORT = result['model_short']
    # Its own id, so these never land in the run's graded ledger: an
    # oracle answer is a diagnostic, not a held-out score.
    harness.RUN_ID = f'{args.run}_oracle'

    harness._init_agent_runtime(None)
    client = harness._AGENT_CLIENT
    oracle_message = build_oracle_message()

    started = time.time()
    verdicts: dict[str, list[bool]] = collections.defaultdict(list)
    for job, votes in prior_samples(args.run, result.get('model_short'),
                                    [label for label, _ in wanted]).items():
        verdicts[job].extend(votes[:args.trials])
    if verdicts:
        got = sum(len(v) for v in verdicts.values())
        print(f'  ledger 里已有 {got} 个样本，算作前几次采样')
    for label, index in wanted:
        snapshot = snapshots[index]['session_snapshot_path']
        for trial in range(args.trials + args.extra_rounds):
            pending = [t for t in tasks
                       if len(verdicts[f'{label}|{t["id"]}']) < args.trials]
            if not pending:
                break
            print(f'\n[oracle] {label} trial {trial + 1}/{args.trials}：'
                  f'{len(pending)} 题待问', flush=True)
            # A fresh disclosure per trial, not a shared one: the ack is
            # part of what is being repeated, and reusing a single acked
            # session would make the three samples share its wording.
            #
            # Retried here because it is the one call in this pass with no
            # retry behind it, and it runs before any question is graded:
            # a single 429 on the disclosure used to kill the whole run
            # and cost all 420 answers. Four traces died that way.
            base = ack = None
            for attempt in range(ACK_ATTEMPTS):
                try:
                    base = harness._closed_book_fork(
                        client.restore_session(snapshot))
                    ack = harness._chat(
                        base, oracle_message, label=f'{label} Oracle Ack')
                    break
                except Exception as exc:  # noqa: BLE001 - message is the data
                    wait = ACK_BACKOFF * (attempt + 1)
                    print(f'  [ack 失败 {attempt + 1}/{ACK_ATTEMPTS}] '
                          f'{type(exc).__name__}: {str(exc)[:120]}'
                          f'  {wait:.0f}s 后重试', flush=True)
                    time.sleep(wait)
            if ack is None:
                # Skipping the round leaves the questions short and the
                # outer loop asks them again; aborting would throw away
                # the label that did work.
                print(f'  [跳过] {label} trial {trial + 1} 的规则披露没成功',
                      flush=True)
                continue
            jobs, names = {}, {}
            for position, task in enumerate(pending):
                key = (label, trial, position)
                jobs[key] = {
                    'task': task,
                    'base_session': base,
                    'pending_feedback': None,
                    'milestone_idx': index,
                }
                names[key] = f'{label}|{task["id"]}|t{trial}'
            flat = harness._dispatch_code_jobs(
                jobs, names, kind='oracle', n_workers=args.workers,
                banner=f'{label} trial {trial + 1}/{args.trials} '
                       f'({len(pending)} 题)')
            for key, task in zip(jobs, pending):
                row = flat.get(key) or {}
                if row.get('error_type'):
                    # Left unrecorded so the loop above asks again on a
                    # fresh fork; the hole is scored as wrong at the end.
                    continue
                verdicts[f'{label}|{task["id"]}'].append(
                    bool(row.get('correct')))

    # A slot the provider never filled is scored as a wrong answer, the
    # same rule the three-sample passes use: averaging over however many
    # samples arrived would reward the tasks that failed most.
    every_job = [f'{label}|{t["id"]}' for label, _ in wanted for t in tasks]
    scored = {
        job: (verdicts.get(job) or [])
        + [False] * max(args.trials - len(verdicts.get(job) or []), 0)
        for job in every_job
    }

    def mean_of(label: str) -> float | None:
        votes = [statistics.mean(v) for job, v in scored.items()
                 if job.startswith(f'{label}|') and v]
        return statistics.mean(votes) * 100 if votes else None

    short_jobs = sorted(job for job in every_job
                        if len(verdicts.get(job) or []) < args.trials)
    measured = [label for label, _ in wanted]
    payload = {
        'run_id': args.run,
        'model_short': result.get('model_short'),
        'trials': args.trials,
        'labels': measured,
        'n_tasks': len(tasks),
        'mean_score': {label: mean_of(label) for label in measured},
        'short_jobs': short_jobs,
        'complete': not short_jobs and len(measured) == len(LABELS),
        'scoring': 'missing-sample-counts-wrong',
        'verdicts': {k: v for k, v in sorted(verdicts.items())},
        'seconds': round(time.time() - started, 1),
        'written_at': time.strftime('%F %T'),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f'{args.run}.json'
    # Merge rather than replace. Each disclosure may be measured by its own
    # process now, and a plain write would have the second one erase the
    # first: the artifact is per run, the measurement is per disclosure. So
    # a pass only ever adds its own labels, and `complete` is the union's
    # verdict rather than this pass's.
    if out.is_file():
        try:
            with out.open(encoding='utf-8') as handle:
                prior = json.load(handle)
        except (OSError, ValueError):
            prior = {}
        keep = [label for label in (prior.get('labels') or [])
                if label not in measured]
        payload['labels'] = sorted(
            set(keep) | set(measured),
            key=lambda name: [l for l, _ in LABELS].index(name))
        for label in keep:
            value = (prior.get('mean_score') or {}).get(label)
            if value is not None:
                payload['mean_score'][label] = value
        payload['verdicts'].update({
            job: votes for job, votes in (prior.get('verdicts') or {}).items()
            if job.split('|', 1)[0] in keep})
        payload['verdicts'] = dict(sorted(payload['verdicts'].items()))
        payload['short_jobs'] = sorted(set(short_jobs) | {
            job for job in (prior.get('short_jobs') or [])
            if job.split('|', 1)[0] in keep})
        payload['complete'] = (not payload['short_jobs']
                               and len(payload['labels']) == len(LABELS))

    tmp = out.with_suffix('.json.tmp')
    with tmp.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    tmp.replace(out)
    for label, _ in wanted:
        value = payload['mean_score'][label]
        print(f'  {label} 均分 '
              + (f'{value:.1f}' if value is not None else '—'))
    if short_jobs:
        print(f'  [未采满] {len(short_jobs)} 题不足 {args.trials} 个样本')
    print(f'  -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
