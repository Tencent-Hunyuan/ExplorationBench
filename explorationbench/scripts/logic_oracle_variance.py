#!/usr/bin/env python3
"""The knowing-versus-doing diagnostic for AlienLogic, three samples deep.

AlienCode has had this for a while: fork a run's own frozen state, hand
the model the true rules, and re-ask every held-out question. The gap
between that and what the run reached on its own separates "never worked
the rule out" from "worked it out and still could not use it". AlienLogic
had no equivalent, so half of RQ3 was missing and the oracle figure could
only be drawn for one sandbox.

Two disclosures, the same pair AlienCode measures:

* ``O@M0`` forks the state before any exploration. The model has the
  calibration demos and the rules, and nothing it discovered.
* ``A4+O`` forks the state after all four rounds. The model has
  everything it found *and* the rules.

Both are asked of all seventy theorems, three times each, from a fresh
fork per trial -- so the disclosure itself is repeated rather than shared,
and the three samples are independent the way the scoring pass's are.

    python3 dev/scripts/logic_oracle_variance.py --run v2_opus5_n1 --dry-run
    python3 dev/scripts/logic_oracle_variance.py --run v2_opus5_n1
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
for path in (str(DEV), str(DEV / 'sandboxes' / 'logic')):
    if path not in sys.path:
        sys.path.insert(0, path)

RESULTS = ROOT / 'logs' / 'logic' / 'results'
CHECKPOINTS = ROOT / 'logs' / 'logic' / 'checkpoints'
OUT_DIR = ROOT / 'logs' / 'logic' / 'oracle'
EPISODE = os.environ.get('ALIENLOGIC_EPISODE', 'public_demo')
#: Disclosure -> the milestone whose frozen state it forks.
LABELS = (('O@M0', '0'), ('A4+O', '4'))

#: How stubbornly to chase the disclosure turn itself. Generous for the
#: reason AlienCode's is: it is one call that gates seventy, and a single
#: 429 on it used to cost a whole pass.
ACK_ATTEMPTS = int(os.environ.get('LORC_ACK_ATTEMPTS', '6'))
ACK_BACKOFF = float(os.environ.get('LORC_ACK_BACKOFF', '45'))


def find_result(run: str) -> Path:
    matches = sorted(
        RESULTS.glob(f'eval_results_alien_logic_*_{EPISODE}_{run}.json'),
        key=lambda p: p.stat().st_mtime)
    if not matches:
        raise SystemExit(f'{run}: no logic result artifact')
    return matches[-1]


def prior_samples(run: str, short: str,
                  labels: list[str]) -> dict[str, list[bool]]:
    """Samples already taken, read back from this pass's own ledger.

    Each was drawn under the same disclosure from the same forked state,
    so it measures exactly the quantity being measured. Without this a
    restart begins from nothing, and a supervisor restarts whatever dies.
    """

    got: dict[str, list[bool]] = {}
    ledger = CHECKPOINTS / f'heldout_{run}_oracle.jsonl'
    if not ledger.is_file():
        return got
    with ledger.open(encoding='utf-8', errors='replace') as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            # Named per trial -- `O@M0|T04|t0` -- because each trial is its
            # own dispatch; the question is the first two fields.
            parts = str(row.get('job') or '').split('|')
            if len(parts) < 2 or parts[0] not in labels:
                continue
            result = row.get('result') or {}
            if result.get('error_type') or result.get('accepted') is None:
                continue
            got.setdefault(f'{parts[0]}|{parts[1]}', []).append(
                bool(result['accepted']))
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--trials', type=int, default=3)
    ap.add_argument('--workers', type=int, default=48)
    ap.add_argument('--extra-rounds', type=int, default=3)
    ap.add_argument('--labels', default=','.join(l for l, _ in LABELS),
                    help='which disclosures to measure, comma separated')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    source = find_result(args.run)
    with source.open(encoding='utf-8') as handle:
        result = json.load(handle)
    config = result.get('config') or {}
    short = result.get('model_short') or args.run

    asked = config.get('reasoning_effort_asked_for')
    if asked and 'EVAL_REASONING_EFFORT' not in os.environ:
        os.environ['EVAL_REASONING_EFFORT'] = str(asked)
    os.environ.setdefault('ALIENLOGIC_PROTOCOL_V2', '1')

    from sandboxes.logic import run_eval as _entry  # noqa: F401
    from sandboxes.logic import harness
    from oracle_rules import build_oracle_message

    episode = harness._apply_v2_task_set(
        harness.get_episode(result.get('episode_id') or EPISODE))
    theorems = {t.id: t for t in episode.heldout_theorems}

    names = [name.strip() for name in args.labels.split(',') if name.strip()]
    unknown = [n for n in names if n not in {l for l, _ in LABELS}]
    if unknown:
        raise SystemExit(f'unknown disclosure {unknown}')

    stones = {str(m.get('milestone')): m
              for m in (result.get('milestones') or [])}
    wanted = []
    for label, stone in LABELS:
        if label not in names:
            continue
        row = stones.get(stone) or {}
        path = row.get('session_snapshot_path')
        if path and os.path.exists(path):
            wanted.append((label, path))
    missing = [l for l in names if l not in {w for w, _ in wanted}]
    if missing:
        raise SystemExit(
            f'{args.run}: no milestone snapshot for {missing}; the '
            f'disclosure has to fork from the state it is measuring')

    print(f'{args.run}  ({short})')
    print(f'  {len(wanted)} 个披露 × {len(theorems)} 题 × {args.trials} 次 '
          f'= {len(wanted) * len(theorems) * args.trials} 次调用')
    if args.dry_run:
        return 0

    harness.MODEL = result['model']
    harness.MODEL_SHORT = short
    # Its own id, so these never land in the trace's graded ledger: an
    # oracle answer is a diagnostic, not a held-out score.
    harness.RUN_ID = f'{args.run}_oracle'
    harness._TRANSCRIPT_PATH = str(
        ROOT / 'logs' / 'logic' / 'traces'
        / f'agent_trace_alien_logic_{short}_{EPISODE}_'
          f'{args.run}_oracle.jsonl')

    oracle_message = build_oracle_message(episode.alien_rules)

    started = time.time()
    verdicts: dict[str, list[bool]] = collections.defaultdict(list)
    for job, votes in prior_samples(args.run, short,
                                    [l for l, _ in wanted]).items():
        verdicts[job].extend(votes[:args.trials])
    if verdicts:
        got = sum(len(v) for v in verdicts.values())
        print(f'  ledger 里已有 {got} 个样本，算作前几次采样')

    for label, snapshot_path in wanted:
        with Path(snapshot_path).open(encoding='utf-8') as handle:
            state = json.load(handle)
        state['model'] = harness.MODEL
        restored = harness.build_runtime_for_redo(
            snapshot=state, snapshot_dir=str(Path(snapshot_path).parent))

        for trial in range(args.trials + args.extra_rounds):
            pending = [t for t in theorems
                       if len(verdicts[f'{label}|{t}']) < args.trials]
            if not pending:
                break
            print(f'\n[oracle] {label} trial {trial + 1}/{args.trials}：'
                  f'{len(pending)} 题待问', flush=True)
            # A fresh disclosure per trial rather than one shared ack: the
            # disclosure turn is part of what is being repeated, and three
            # samples drawn after one ack would share its wording.
            base = ack = None
            for attempt in range(ACK_ATTEMPTS):
                try:
                    base = harness._closed_book_fork(restored)
                    ack, _ = harness._chat_with_empty_retry(
                        base, oracle_message,
                        label=f'{label} Oracle Ack',
                        max_tokens=harness.SUMMARY_MAX_TOKENS)
                    break
                except Exception as exc:          # noqa: BLE001
                    wait = ACK_BACKOFF * (attempt + 1)
                    print(f'  [ack 失败 {attempt + 1}/{ACK_ATTEMPTS}] '
                          f'{type(exc).__name__}: {str(exc)[:120]}'
                          f'  {wait:.0f}s 后重试', flush=True)
                    time.sleep(wait)
            if ack is None:
                # The questions stay short and the outer loop asks them
                # again; aborting would throw away the rounds that worked.
                print(f'  [跳过] {label} trial {trial + 1}：披露没成功',
                      flush=True)
                continue

            jobs, job_names = {}, {}
            for theorem_id in pending:
                key = (label, theorem_id, trial)
                jobs[key] = {
                    'theorem': theorems[theorem_id],
                    'base': base,
                    'pending_feedback': None,
                    'milestone_label': label,
                    'alien_rules': episode.alien_rules,
                }
                job_names[key] = f'{label}|{theorem_id}|t{trial}'
            flat = harness._dispatch_heldout(
                jobs, n_workers=args.workers, names=job_names,
                banner=f'{label} trial {trial + 1}/{args.trials} '
                       f'({len(pending)} 题)')
            for key, theorem_id in zip(jobs, pending):
                row = flat.get(key) or {}
                if harness._worker_failed(row):
                    continue
                verdicts[f'{label}|{theorem_id}'].append(
                    bool(row['accepted']))

    # A slot the provider never filled is scored as a failed proof, the
    # convention both sandboxes' sampling passes use.
    measured = [label for label, _ in wanted]
    every_job = [f'{label}|{t}' for label in measured for t in theorems]
    scored = {
        job: (verdicts.get(job) or [])
        + [False] * max(args.trials - len(verdicts.get(job) or []), 0)
        for job in every_job
    }
    meta = {t.id: t for t in episode.heldout_theorems}

    def rate(label: str, pick=None) -> float | None:
        votes = [statistics.mean(v) for job, v in scored.items()
                 if job.startswith(f'{label}|') and v
                 and (pick is None or pick(job.split('|', 1)[1]))]
        return statistics.mean(votes) * 100 if votes else None

    aware = (lambda t: getattr(meta.get(t), 'alien_aware_required', False))
    unprov = (lambda t: getattr(meta.get(t), 'alien_provable', True) is False)

    short_jobs = sorted(job for job in every_job
                        if len(verdicts.get(job) or []) < args.trials)
    payload = {
        'run_id': args.run,
        'model_short': short,
        'sandbox': 'logic',
        'trials': args.trials,
        'labels': measured,
        'n_theorems': len(theorems),
        'mean_score': {label: rate(label) for label in measured},
        'mean_aware': {label: rate(label, aware) for label in measured},
        'mean_unprovable': {label: rate(label, unprov)
                            for label in measured},
        'short_jobs': short_jobs,
        'complete': not short_jobs and len(measured) == len(LABELS),
        'scoring': 'missing-sample-counts-wrong',
        'verdicts': {k: v for k, v in sorted(verdicts.items())},
        'seconds': round(time.time() - started, 1),
        'written_at': time.strftime('%F %T'),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f'{args.run}.json'
    # Merge, because each disclosure may be measured by its own process:
    # the artifact is per run, the measurement is per disclosure.
    if out.is_file():
        try:
            with out.open(encoding='utf-8') as handle:
                prior = json.load(handle)
        except (OSError, ValueError):
            prior = {}
        keep = [l for l in (prior.get('labels') or []) if l not in measured]
        payload['labels'] = sorted(
            set(keep) | set(measured),
            key=lambda name: [l for l, _ in LABELS].index(name))
        for field in ('mean_score', 'mean_aware', 'mean_unprovable'):
            for label in keep:
                value = (prior.get(field) or {}).get(label)
                if value is not None:
                    payload[field][label] = value
        payload['verdicts'].update({
            job: votes
            for job, votes in (prior.get('verdicts') or {}).items()
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

    for label in measured:
        value = payload['mean_score'][label]
        print(f'  {label} 均分 '
              + (f'{value:.1f}' if value is not None else '—'))
    if short_jobs:
        print(f'  [未采满] {len(short_jobs)} 题不足 {args.trials} 个样本')
    print(f'  -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
