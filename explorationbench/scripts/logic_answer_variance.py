#!/usr/bin/env python3
"""Ask one AlienLogic trace's held-out theorems three times each.

The AlienLogic counterpart of ``answer_variance.py``. Both boards already
report three traces per model, but only AlienCode also sampled each
question three times: a logic trace answered every theorem once, so a
milestone's pass rate carried the variance of a single draw and the two
sandboxes were not comparable at the question level.

This re-asks each theorem from the milestone's own base snapshot -- the
session as it stood when that milestone was graded, closed-book -- so the
extra samples are drawn from the same state the original answer was, and
not from a model that has since seen four more rounds of exploration.

The trace's own verdict counts as the first sample, exactly as on the code
side: it was produced under the protocol and scored, so paying to
reproduce it would buy a measurement already on disk.

    python3 dev/scripts/logic_answer_variance.py --run v2_qwen38_n3
    python3 dev/scripts/logic_answer_variance.py --run v2_hy4_n1 --dry-run
    python3 dev/scripts/logic_answer_variance.py --run ctrl_passive_v2_hy4_logic_n2 --milestones 4
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

DEV = Path(__file__).resolve().parents[1]
if str(DEV) not in sys.path:
    sys.path.insert(0, str(DEV))

ROOT = DEV.parent
RESULTS = ROOT / 'logs' / 'logic' / 'results'
OUT_DIR = ROOT / 'logs' / 'logic' / 'variance'
EPISODE = os.environ.get('ALIENLOGIC_EPISODE', 'public_demo')

# Read at import time by the harness, so both have to be set before it is
# pulled in: without the first the held-out set is the old 85 theorems, and
# a sample drawn against a different task set is not a repeat.
os.environ.setdefault('ALIENLOGIC_PROTOCOL_V2', '1')


#: Traces produced before the logic routes were corrected recorded the
#: evaluation gateway's model id. That account is out of budget and answers
#: 402 to everything, so replaying a theorem through it gets no sample at
#: all. These are the same weights behind GatewayA's door -- the door the
#: AlienCode runs for these models already used -- so the extra samples go
#: there. The first sample still came through the old one; same model,
#: different endpoint.
GATEWAY_A_EQUIVALENT = {
    'api_azure_openai_gpt-5.6-sol': 'gateway_a/azure/gpt-5.6-sol',
    'messages/api_deepseek_deepseek-v4-pro': 'gateway_a/deepseek/deepseek-v4-pro',
    'api_aws_third_anthropic.claude-opus-4-8':
        'gateway_a/aws_third/anthropic.claude-opus-4-8',
}


def prune_ledger(path: Path) -> int:
    """Drop calls the provider never answered from the resume journal.

    The journal exists so a restart does not pay twice for an answer, and
    it is keyed per trial. A call that failed transport wrote a row too,
    and that row is indistinguishable from an answer to the resume logic:
    the next launch finds it, asks nothing, and this pass skips it for not
    being a verdict. The sample can then never be collected, however many
    times the supervisor restarts -- which is how one trace sat at 18
    restarts with a whole milestone permanently one sample short.

    A failed call is not a measurement, so it does not belong in a journal
    of measurements.
    """

    if not path.exists():
        return 0
    kept, dropped = [], 0
    with path.open(encoding='utf-8', errors='replace') as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            result = row.get('result')
            if isinstance(result, dict) and (
                    (result.get('diagnostic') or {}).get('reason_class')
                    == 'WORKER_ERROR'):
                dropped += 1
                continue
            kept.append(line)
    if dropped:
        scratch = path.with_suffix('.jsonl.tmp')
        with scratch.open('w', encoding='utf-8') as handle:
            handle.writelines(kept)
        scratch.replace(path)
    return dropped


def find_result(run: str) -> Path:
    matches = sorted(
        RESULTS.glob(f'eval_results_alien_logic_*_{EPISODE}_{run}.json'),
        key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise SystemExit(f'no result for {run}')
    return matches[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--trials', type=int, default=3)
    ap.add_argument('--milestones', default='0,1,2,3,4',
                    help='comma-separated milestones to sample; a board '
                         'cell needs all five, an endpoint check only 4')
    ap.add_argument('--workers', type=int, default=64)
    ap.add_argument('--extra-rounds', type=int, default=3,
                    help='further passes for theorems the provider never '
                         'answered; they are not scored as failures')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    source = find_result(args.run)
    with source.open(encoding='utf-8') as handle:
        result = json.load(handle)
    config = result.get('config') or {}

    # The comparison only holds if the new samples are drawn at the tier the
    # original was. The run recorded what it asked for; asking for anything
    # else is a different model.
    asked = config.get('reasoning_effort_asked_for')
    if asked and 'EVAL_REASONING_EFFORT' not in os.environ:
        os.environ['EVAL_REASONING_EFFORT'] = str(asked)

    from sandboxes.logic import run_eval as _entry  # noqa: F401
    from sandboxes.logic import harness

    episode = harness._apply_v2_task_set(
        harness.get_episode(result.get('episode_id') or EPISODE))
    theorems = {t.id: t for t in episode.heldout_theorems}

    picked = {x.strip() for x in args.milestones.split(',') if x.strip()}
    stones = [m for m in (result.get('milestones') or [])
              if str(m.get('milestone')) in picked]
    stones.sort(key=lambda m: int(m['milestone']))
    missing = [m['milestone'] for m in stones
               if not (m.get('session_snapshot_path')
                       and os.path.exists(m['session_snapshot_path']))]
    if missing:
        raise SystemExit(
            f'{args.run}: milestones {missing} have no base snapshot; '
            f're-asking them from another session would hand the model '
            f'what it learned afterwards')

    # The trace's own answers are trial one.
    seed: dict[str, bool] = {}
    meta: dict[str, dict] = {}
    for stone in stones:
        for row in stone.get('test_results') or []:
            theorem_id = row.get('theorem_id')
            if not theorem_id:
                continue
            seed[f"M{stone['milestone']}|{theorem_id}"] = bool(row['accepted'])
            meta.setdefault(theorem_id, {
                'role': row.get('role'),
                'alien_aware_required': bool(row.get('alien_aware_required')),
                'alien_provable': row.get('alien_provable'),
            })

    recorded = result['model']
    model = GATEWAY_A_EQUIVALENT.get(recorded, recorded)
    if model != recorded:
        # Only if the new door speaks the dialect the saved history is
        # written in. The two DeepSeek routes do not: the gateway's is
        # Anthropic-shaped and GatewayA's is Responses-shaped, so replaying
        # one through the other is not a slower sample, it is no sample.
        from common.agent_client import AgentClientConfig
        was = AgentClientConfig(model=recorded).provider.value
        now = AgentClientConfig(model=model).provider.value
        if was != now:
            raise SystemExit(
                f'{args.run}: 原路由 {recorded}（{was}）欠费，而 GatewayA 的 '
                f'{model} 说的是 {now}，存档的历史重放不过去。\n'
                f'  ROUTE_BLOCKED：这条 trace 要等账号充值才能补采样。')

    wanted = [f"M{s['milestone']}|{t}" for s in stones for t in theorems]
    total = sum(args.trials - (1 if job in seed else 0) for job in wanted)
    have = sum(1 for job in wanted if job in seed)
    print(f'{args.run}  ({result.get("model_short")})')
    if model != recorded:
        print(f'  路由改走 GatewayA：{recorded} -> {model}（原网关账号欠费）')
    print(f'  {have}/{len(wanted)} 道已有作答，算作第 1 个样本')
    print(f'  {len(theorems)} 题 × {len(stones)} 个 milestone，'
          f'补到每题 {args.trials} 个样本 = {total} 次调用')
    if args.dry_run:
        return 0

    harness.MODEL = model
    harness.MODEL_SHORT = result.get('model_short') or harness.MODEL
    # A distinct id keeps these off the trace's own ledger: they are extra
    # samples of a graded question, not a repair of it.
    harness.RUN_ID = f'{args.run}_var'
    harness._TRANSCRIPT_PATH = str(
        ROOT / 'logs' / 'logic' / 'traces'
        / f'agent_trace_alien_logic_{harness.MODEL_SHORT}_{EPISODE}_'
          f'{args.run}_var.jsonl')

    ledger = (ROOT / 'logs' / 'logic' / 'checkpoints'
              / f'heldout_{harness.RUN_ID}.jsonl')
    shed = prune_ledger(ledger)
    if shed:
        print(f'  账本清掉 {shed} 条未答成功的调用记录，这些题会重问')

    started = time.time()
    verdicts: dict[str, list[bool]] = collections.defaultdict(list)
    for job in wanted:
        if job in seed:
            verdicts[job].append(seed[job])

    restored = {}
    for stone in stones:
        label = str(stone['milestone'])
        snapshot_path = Path(stone['session_snapshot_path'])
        with snapshot_path.open(encoding='utf-8') as handle:
            state = json.load(handle)
        # The harness refuses a snapshot whose model is not the configured
        # one, which is the right default. Here the difference is the door
        # rather than the model, and it is deliberate.
        state['model'] = harness.MODEL
        restored[label] = harness.build_runtime_for_redo(
            snapshot=state, snapshot_dir=str(snapshot_path.parent))

    # Every milestone and every missing sample goes out in one pool, as in
    # AlienCode's sampler: a batch per (milestone, trial) waited for its
    # slowest theorem each time. Each trial keeps its own fork and each job
    # its old name, so a ledger written either way resumes.
    bases: dict[tuple[str, int], object] = {}

    def base_for(label: str, trial: int):
        # A fresh fork per trial: the point is an independent sample of the
        # same state, so the trials must not see each other.
        if (label, trial) not in bases:
            bases[(label, trial)] = harness._closed_book_fork(restored[label])
        return bases[(label, trial)]

    def ask(plan: list[tuple[str, str, int]], banner: str) -> None:
        jobs, names = {}, {}
        for label, theorem_id, trial in plan:
            key = (label, theorem_id, trial)
            jobs[key] = {
                'theorem': theorems[theorem_id],
                'base': base_for(label, trial),
                'pending_feedback': None,
                'milestone_label': label,
                'alien_rules': episode.alien_rules,
            }
            names[key] = f'M{label}|{theorem_id}|t{trial}'
        flat = harness._dispatch_heldout(
            jobs, n_workers=args.workers, names=names, banner=banner)
        for label, theorem_id, trial in plan:
            row = flat.get((label, theorem_id, trial)) or {}
            # Not recorded, so the theorem stays short and a later round
            # asks it again on a fresh fork. Scoring it here would spend a
            # sample slot on the provider's outage and end the retry; the
            # hole is scored as a failed proof at the end instead -- see
            # `scored` below.
            if harness._worker_failed(row):
                continue
            verdicts[f'M{label}|{theorem_id}'].append(bool(row['accepted']))

    labels = [str(stone['milestone']) for stone in stones]
    first = [
        (label, theorem_id, trial)
        for label in labels for theorem_id in theorems
        for trial in range(max(
            args.trials - len(verdicts[f'M{label}|{theorem_id}']), 0))
    ]
    # Printed up front: the harness only announces its banner when it is
    # chasing failures, so a healthy run said nothing about where it was.
    if first:
        print(f'\n[采样] 一次提交 {len(first)} 个样本'
              f'（{len(labels)} 个里程碑，每题补足 {args.trials} 遍）',
              flush=True)
        ask(first, f'all milestones ({len(first)} 个样本)')
    # Trial numbers continue past the first pass so no two samples of a
    # theorem share a fork or a ledger name.
    for extra in range(args.extra_rounds):
        pending = [
            (label, theorem_id, args.trials + extra)
            for label in labels for theorem_id in theorems
            if len(verdicts[f'M{label}|{theorem_id}']) < args.trials
        ]
        if not pending:
            break
        print(f'\n[补题] 第 {extra + 1} 轮：{len(pending)} 题仍缺样本',
              flush=True)
        ask(pending, f'补题 {extra + 1} ({len(pending)} 题)')

    # A slot the provider never filled is scored as a failed proof, the
    # same rule AlienCode's pass uses. Averaging over however many samples
    # arrived would reward the theorems that failed most: one proved once,
    # accepted, would outscore one attempted three times and accepted
    # twice. `short_jobs` records which theorems were padded.
    scored = {
        job: (verdicts.get(job) or [])
        + [False] * max(args.trials - len(verdicts.get(job) or []), 0)
        for job in wanted
    }

    def rate(label: str, pick) -> float:
        votes = [statistics.mean(v) for job, v in scored.items()
                 if job.startswith(f'M{label}|') and v
                 and pick(job.split('|', 1)[1])]
        return statistics.mean(votes) * 100 if votes else 0.0

    every = (lambda _: True)
    aware = (lambda t: meta.get(t, {}).get('alien_aware_required'))
    unprov = (lambda t: meta.get(t, {}).get('alien_provable') is False)

    stable = sum(1 for v in scored.values() if len(set(v)) == 1)
    undersampled = sorted(job for job in wanted
                          if len(verdicts.get(job) or []) < args.trials)
    payload = {
        'run_id': args.run,
        'model_short': result.get('model_short'),
        'sandbox': 'logic',
        'trials': args.trials,
        'reused_first_trial': bool(seed),
        'reused_jobs': sorted(job for job in verdicts if job in seed),
        'milestones': [int(s['milestone']) for s in stones],
        'n_theorems': len(theorems),
        'agreement': stable / max(len(scored), 1),
        'scoring': 'missing-sample-counts-wrong',
        'mean_score': {f'M{s["milestone"]}': rate(str(s['milestone']), every)
                       for s in stones},
        'mean_aware': {f'M{s["milestone"]}': rate(str(s['milestone']), aware)
                       for s in stones},
        'mean_unprovable': {f'M{s["milestone"]}':
                            rate(str(s['milestone']), unprov)
                            for s in stones},
        'short_jobs': undersampled,
        'complete': not undersampled,
        'theorem_meta': meta,
        'verdicts': {k: v for k, v in sorted(verdicts.items())},
        'seconds': round(time.time() - started, 1),
        'written_at': time.strftime('%F %T'),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f'{args.run}.json'
    with out.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f'\n  {stable}/{len(scored)} 题三次判定一致 '
          f'({payload["agreement"]:.1%})')
    for stone in stones:
        key = f'M{stone["milestone"]}'
        once = (stone.get('pass_rate') or 0.0) * 100
        print(f'  {key} 均分 {payload["mean_score"][key]:.1f}'
              f'（原单次 {once:.1f}）')
    if undersampled:
        print(f'  [未采满] {len(undersampled)} 题不足 {args.trials} 个样本')
    print(f'  -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
