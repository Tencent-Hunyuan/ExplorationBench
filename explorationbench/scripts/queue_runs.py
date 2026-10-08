#!/usr/bin/env python3
"""Run independent repeat evaluations with bounded high concurrency.

Repeats exist to separate a real gap between models from run-to-run noise, so
they may run concurrently, including repeats on the same provider. A single
trajectory remains sequential, while its closed-book forks and independent
repeat trajectories fan out. Health checks still matter: a run whose model
never called the tool can finish and write a score file without measuring
learning.

Both sandboxes are driven from here, so a full sweep is one command and every
run leaves the same artefacts behind (messages archive, HTML replay, index).

Usage:
    python3 dev/scripts/queue_runs.py --sandbox code --repeats 2
    python3 dev/scripts/queue_runs.py --sandbox logic --models gemini,gpt-max
    python3 dev/scripts/queue_runs.py --sandbox both --repeats 3 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "explorationbench"
RUN_LOCKS = ROOT / "logs" / ".run_locks"
if str(DEV) not in sys.path:
    sys.path.insert(0, str(DEV))

from frameworks.registry import FRAMEWORK_SPECS, normalize_framework_key

# What differs between the two sandboxes: where the run writes, how it names
# its files, and which knob bounds its concurrency. Both keep traces in their
# own traces/ directory now, and both name artifacts after the run id alone --
# the globs still tolerate the launch timestamp older runs carry.
SANDBOXES = {
    "code": {
        "script": "sandboxes/code/run_eval.py",
        "results": ROOT / "logs" / "code" / "results",
        "traces": ROOT / "logs" / "code" / "traces",
        "queue_logs": ROOT / "logs" / "code" / "queue",
        "exports": ROOT / "logs" / "code" / "exports",
        "short_flag": "--short",
        "extra_args": [],
        "parallel_env": "ALIENCODE_PARALLEL_TESTS",
        "trace_glob": "agent_trace_alien_code_*{short}_{run}.jsonl",
        "score_glob": "eval_results_alien_code_{short}_{run}.json",
    },
    "logic": {
        "script": "sandboxes/logic/run_eval.py",
        "results": ROOT / "logs" / "logic" / "results",
        "traces": ROOT / "logs" / "logic" / "traces",
        "legacy_traces": ROOT / "logs" / "logic" / "results",
        "queue_logs": ROOT / "logs" / "logic" / "queue",
        "exports": ROOT / "logs" / "logic" / "exports",
        "short_flag": "--model-short",
        # Deferring the held-out batches is score-equivalent (every batch still
        # forks from its own milestone snapshot) and removes the per-milestone
        # barrier, which is most of a logic run's wall clock.
        # The shared paper endpoint is M4. Existing M8 runs remain a separate
        # appendix diagnostic, but n=3 replication must not spend half its
        # budget beyond the primary horizon.
        # --resume: relaunching a run id here means picking a killed or reaped
        # run back up, so continue from its last completed phase.
        "extra_args": ["--defer-tests", "--explore-loops", "4", "--resume"],
        "parallel_env": "ALIENLOGIC_DEFER_WORKERS",
        "trace_glob": "agent_trace_alien_logic_*{short}_*{run}.jsonl",
        "score_glob": "eval_results_alien_logic_*{short}_*{run}.json",
    },
}

# The top reasoning tier each model actually offers, and the run id its first
# run used. Gemini 3.x stops at "high" and Doubao at "high"; asking for more is
# a 400. DeepSeek takes a thinking budget rather than a named tier, so its
# effort is only there to keep the launch uniform.
MODELS = [
    {
        "key": "opus",
        "model": "api_aws_third_anthropic.claude-opus-4-8",
        "short": "opus-4-8",
        "effort": "max",
        "base": {"code": "full_opus48", "logic": "opus48_l"},
        # The first code run predates stable repeat naming.
        "run_ids": {
            "code": {
                1: "full_opus48",
                2: "opus48_n2",
                3: "opus48_n3",
            },
        },
    },
    {
        "key": "gpt-max",
        "model": "api_azure_openai_gpt-5.6-sol",
        "short": "gpt-5.6-sol-max",
        "effort": "max",
        "base": {"code": "gpt56_max", "logic": "gpt56_max_l"},
    },
    {
        "key": "gpt-xhigh",
        "model": "api_azure_openai_gpt-5.6-sol",
        "short": "gpt-5.6-sol-xhigh",
        "effort": "xhigh",
        "base": {"code": "gpt56_xhigh", "logic": "gpt56_xhigh_l"},
    },
    {
        "key": "gpt-high",
        "model": "api_azure_openai_gpt-5.6-sol",
        "short": "gpt-5.6-sol-high",
        "effort": "high",
        "base": {"code": "gpt56_high", "logic": "gpt56_high_l"},
    },
    {
        "key": "gemini",
        "model": "api_google_gemini-3.6-flash",
        "short": "gemini-3.6-flash-high",
        "effort": "high",
        "base": {"code": "gemini36_high", "logic": "gemini36_high_l"},
    },
    {
        "key": "deepseek",
        "model": "messages/api_deepseek_deepseek-v4-flash",
        "short": "deepseek-v4-flash",
        "effort": "max",
        "base": {"code": "dsv4_flash", "logic": "dsv4_flash_l"},
    },
    {
        "key": "deepseek-pro",
        "model": "messages/api_deepseek_deepseek-v4-pro",
        "short": "deepseek-v4-pro-native",
        "effort": "max",
        "base": {
            "code": "dsv4_pro_native_max",
            "logic": "dsv4_pro_native_max_l",
        },
    },
    {
        # The same model on the vendor passthrough rather than GatewayA's
        # standard chat door, which is what makes it comparable with the
        # reported cohort: Responses with tool calls, prompt cache and
        # reasoning, instead of a flattened chat conversation.
        "key": "deepseek-flash41-pt",
        # The dated alias was retired mid-cohort and now fails intermittently
        # as the retirement rolls across accounts. Both names resolve to the
        # same upstream -- each answers with model='deepseek-flash' -- so this
        # is a rename, and runs on either name stay comparable.
        "model": "gateway_a/deepseek/deepseek-flash",
        "short": "deepseek-v4.1-flash-pt",
        "effort": "max",
        "base": {"code": "dsv41pt", "logic": "dsv41pt_l"},
        # The prompt cache absorbs most of the replayed history, so a held-out
        # question costs a fraction of its metered prompt and this key takes
        # far more width than the default before the upstream pushes back.
        "parallel": 32,
    },
    {
        # Exploratory only, not part of the reported systems: this one is
        # served by GatewayA's standard endpoint rather than the evaluation
        # gateway, so its numbers are not comparable with the rest.
        "key": "deepseek-flash41",
        "model": "gateway_a/deepseek-v4.1-flash-expires-on-0910",
        "short": "deepseek-v4.1-flash",
        "effort": "max",
        "base": {"code": "dsv41_flash", "logic": "dsv41_flash_l"},
        # The prompt cache absorbs 98% of the replayed history here, so the
        # metered cost of a held-out question is a fraction of its 30-50k
        # prompt and this key takes far more width than the default.
        "parallel": 12,
        # Nothing reported here reads the pre-exploration baseline, and it
        # costs a sixth of a logic run, so this one is scored from M0.
        "extra_args": {"logic": ["--skip-pre"]},
    },
    {
        "key": "hy3-gateway_b",
        "model": "api_gateway_b_hy3",
        "short": "hy3-gateway_b-reasoning",
        "effort": "high",
        "base": {
            "code": "hy3_gateway_b_reasoning",
            "logic": "hy3_gateway_b_reasoning_l",
        },
    },
    {
        "key": "doubao",
        "model": "api_doubao_doubao-seed-2-1-pro-260628",
        "short": "doubao-seed-2.1-pro",
        "effort": "high",
        "base": {"code": "doubao21_high", "logic": "doubao21_high_l"},
    },
    {
        "key": "qwen",
        "model": "api_ali_qwen3.8-max",
        "short": "qwen3.8-max",
        "effort": "max",
        "base": {"code": "qwen38_max", "logic": "qwen38_max_l"},
    },
    {
        # A fresh replication of the reported qwen cohort on the same
        # passthrough. Its own run ids, so the paper's three runs stay put.
        "key": "qwen-rt",
        "model": "api_ali_qwen3.8-max",
        "short": "qwen3.8-max-rt",
        "effort": "max",
        "base": {"code": "qwen38_rt", "logic": "qwen38_rt_l"},
        # This upstream caps concurrent requests rather than queueing past
        # them, and answers 400 Throttling.Concurrency -- which is a lost
        # question, not a retryable stall. The cap is shared across every run
        # on this vendor: three repeats at 8 stay inside it, while three at 16
        # cost one run nine graded calls. A single run alone has the whole cap
        # to itself.
        "parallel": 24,
    },
    {
        "key": "qwen-0902",
        "model": "api_ali_qwen3.8-max-0902",
        "short": "qwen3.8-max-0902",
        # The route collapses max onto the vendor's native xhigh, which is its
        # top tier.
        "effort": "max",
        "base": {"code": "qwen0902", "logic": "qwen0902_l"},
    },
    {
        # Reachable only through GatewayA's own Bedrock door: the evaluation
        # gateway stocks no account for this model.
        "key": "opus5",
        "model": "gateway_a/aws_third/anthropic.claude-opus-5",
        "short": "claude-opus-5",
        "effort": "max",
        "base": {"code": "opus5", "logic": "opus5_l"},
    },
    {
        # The same model as `qwen`, re-tested on GatewayA's own passthrough
        # instead of the evaluation gateway's Responses route. Exploratory:
        # that endpoint collapses max onto xhigh, so the tier is not the one
        # the reported qwen runs asked for.
        "key": "qwen-gateway_a",
        "model": "gateway_a/ali/qwen3.8-max",
        "short": "qwen3.8-max-gateway_a",
        "effort": "max",
        "base": {"code": "qwen38_gateway_a", "logic": "qwen38_gateway_a_l"},
        "parallel": 16,
        "extra_args": {"logic": ["--skip-pre"]},
    },
    {
        "key": "kimi",
        "model": "api_moonshot_kimi-k3",
        "short": "kimi-k3",
        "effort": "max",
        "base": {"code": "kimik3_max", "logic": "kimik3_max_l"},
        # The gateway's account pool for this vendor runs dry under load and
        # answers 500 until one frees up, so this one asks for fewer workers.
        "parallel": 2,
    },
    {
        "key": "grok",
        "model": "api_xai_grok-4.5",
        "short": "grok-4.5",
        # Reasoning cannot be turned off and the tiers stop here.
        "effort": "high",
        "base": {"code": "grok45_high", "logic": "grok45_high_l"},
    },
]

BY_KEY = {spec["key"]: spec for spec in MODELS}


def run_id(spec: dict, sandbox: str, n: int) -> str:
    """The run id of the n-th run (1-based).

    AlienCode's first runs predate this script and are already named, so only
    the repeats get a suffix. AlienLogic numbers every run from the start.
    """

    explicit = (spec.get("run_ids") or {}).get(sandbox, {}).get(n)
    if explicit:
        return explicit
    base = spec["base"][sandbox]
    if sandbox == "logic":
        return f"{base}{n}"
    return base if n == 1 else f"{base}_n{n}"


def framework_run_id(base_run: str, framework: str, track: str) -> str:
    framework = normalize_framework_key(framework)
    track = (track or "controlled").strip().lower()
    if framework == "baseline" and track == "controlled":
        return base_run
    return f"{base_run}_fw_{framework}_{track}"


def running_runs() -> list[str]:
    """Run ids of every evaluation currently alive."""

    try:
        out = subprocess.run(
            ["pgrep", "-af", "run_eval.py"],
            capture_output=True, text=True, check=False,
        ).stdout
    except OSError:
        return []
    ids = []
    for line in out.splitlines():
        if "--run-id" not in line or "queue_runs" in line:
            continue
        parts = line.split("--run-id", 1)[1].split()
        if parts:
            ids.append(parts[0])
    # pgrep matches both the shell wrapper and python; collapse duplicates.
    return sorted(set(ids))


def sandbox_of_run(run: str) -> str | None:
    """Which sandbox a live run id belongs to, or None if it is not ours.

    The bases overlap -- a logic id extends the code one -- so the longest
    match wins.
    """

    best: tuple[int, str] | None = None
    for spec in MODELS:
        for sandbox, base in spec["base"].items():
            if run.startswith(base) and (best is None or len(base) > best[0]):
                best = (len(base), sandbox)
    return best[1] if best else None


def wait_for(run_ids: list[str], *, poll: int = 60) -> None:
    pending = set(run_ids)
    while True:
        alive = pending & set(running_runs())
        if not alive:
            return
        print(
            f"[{time.strftime('%H:%M:%S')}] waiting on {', '.join(sorted(alive))}",
            flush=True,
        )
        time.sleep(poll)


def _newest(directory: Path, pattern: str) -> Path | None:
    matches = sorted(
        directory.glob(pattern), key=lambda p: p.stat().st_mtime,
    )
    return matches[-1] if matches else None


def latest_trace(sandbox: str, short: str, run: str) -> Path | None:
    box = SANDBOXES[sandbox]
    pattern = box["trace_glob"].format(short=short, run=run)
    # Logic traces used to be written beside the scores; keep looking there so
    # runs from before the move still export and health-check.
    for directory in (box["traces"], box.get("legacy_traces")):
        if directory is None:
            continue
        found = _newest(directory, pattern)
        if found:
            return found
    return None


def latest_scores(sandbox: str, short: str, run: str) -> Path | None:
    box = SANDBOXES[sandbox]
    pattern = box["score_glob"].format(short=short, run=run)
    # A `_redone` file is the same run rescored after questions that failed on
    # infrastructure errors were retried, so it reflects the model, not the
    # outage.
    redone = _newest(box["results"], pattern.replace(".json", "_redone.json"))
    return redone or _newest(box["results"], pattern)


def tool_calls_in(trace: Path) -> int:
    count = 0
    for line in trace.open(encoding="utf-8", errors="replace"):
        if '"tool_call"' not in line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("record_type") != "trajectory":
            continue
        count += sum(
            1
            for event in (record.get("history_delta") or [])
            if event.get("kind") == "tool_call"
        )
    return count


def runtime_feedback_events_in(trace: Path) -> int:
    count = 0
    for line in trace.open(encoding="utf-8", errors="replace"):
        if '"environment_feedback_observed"' not in line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("record_type") != "runtime_event":
            continue
        if record.get("event") == "environment_feedback_observed":
            count += 1
    return count


def health(sandbox: str, short: str, run: str) -> tuple[bool, str]:
    """Whether a finished run is worth repeating, and why not if it isn't."""

    scores_path = latest_scores(sandbox, short, run)
    if scores_path is None:
        return False, "no score file"
    try:
        data = json.loads(scores_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"unreadable score file: {exc}"

    milestones = data.get("milestones") or []
    if not milestones:
        return False, "no milestones scored"

    trace = latest_trace(sandbox, short, run)
    if trace is None:
        return False, "no trace"
    calls = tool_calls_in(trace)
    runtime_feedback = runtime_feedback_events_in(trace)
    if not calls and not runtime_feedback:
        # The failure mode that silently invalidated three runs: the model
        # answered in prose, nothing executed, and it learned nothing.
        return False, "model never called the tool"

    last = milestones[-1]
    if sandbox == "logic":
        rate = last.get("pass_rate")
        detail = (
            f"final {last.get('test_pass')}/{last.get('test_total')} proofs"
            + (f" ({100 * rate:.1f}%)" if isinstance(rate, (int, float)) else "")
        )
    else:
        detail = (
            f"final {last.get('found')}/{last.get('total')} rules "
            f"{last.get('test_correct')}/{last.get('test_total')} tests"
        )
    return True, (
        f"{len(milestones)} milestones, {calls} tool calls, "
        f"{runtime_feedback} runtime feedback events, {detail}"
    )


def completed_score(sandbox: str, short: str, run: str) -> bool:
    """Whether this exact paper run already has its complete milestone set.

    This intentionally reads only the compact score file. Using `health` here
    would rescan every multi-hundred-megabyte trace before a missing-run queue
    can start. Explicit run ids and the expected milestone counts keep old or
    partial artifacts from satisfying the check.
    """

    scores_path = latest_scores(sandbox, short, run)
    if scores_path is None:
        return False
    try:
        data = json.loads(scores_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    milestones = data.get("milestones") or []
    if sandbox == "code":
        return len(milestones) >= 5
    # AlienLogic's record count says nothing on its own: it prepends Mpre
    # unless --skip-pre was passed, and legacy long-horizon runs continue to
    # M8. Both sandboxes are scored on M0--M4, so ask for those by name.
    labels = {
        str(record.get("milestone", record.get("milestone_idx")))
        for record in milestones
    }
    return {"0", "1", "2", "3", "4"}.issubset(labels)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True
    return True


def claim_run(sandbox: str, run: str) -> Path | None:
    """Atomically claim a run id so concurrent old queues cannot duplicate it."""

    RUN_LOCKS.mkdir(parents=True, exist_ok=True)
    path = RUN_LOCKS / f"{sandbox}_{run}.lock"
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                owner = int(path.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                owner = -1
            if _pid_alive(owner):
                return None
            # A killed queue can leave a lock behind. Never clear it while an
            # untracked child with the same run id is still alive.
            if run in running_runs():
                return None
            path.unlink(missing_ok=True)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        return path
    return None


def release_run(path: Path) -> None:
    path.unlink(missing_ok=True)


def omit_completed(
    plan: list[tuple[str, list[tuple[dict, str, str]]]],
) -> list[tuple[str, list[tuple[dict, str, str]]]]:
    """Drop completed or currently running ids; queue only genuinely absent."""

    active = set(running_runs())
    filtered = []
    for sandbox, batch in plan:
        missing = []
        for spec, run, framework in batch:
            if run in active:
                print(f"  skip in-flight {sandbox}/{run}", flush=True)
            elif completed_score(sandbox, spec["short"], run):
                print(f"  skip complete {sandbox}/{run}", flush=True)
            else:
                missing.append((spec, run, framework))
        filtered.append((sandbox, missing))
    return filtered


def launch(
    spec: dict,
    sandbox: str,
    run: str,
    parallel: int,
    *,
    framework: str,
    track: str,
    budget_profile: str,
) -> subprocess.Popen:
    box = SANDBOXES[sandbox]
    box["queue_logs"].mkdir(parents=True, exist_ok=True)
    log = box["queue_logs"] / f"{run}.log"
    env = {
        **os.environ,
        "EVAL_TOOL_MODE": "1",
        "EVAL_REASONING_EFFORT": spec["effort"],
        "EVAL_FRAMEWORK": framework,
        "EVAL_TRACK": track,
        "EVAL_BUDGET_PROFILE": budget_profile,
        # 1200s because that is the ceiling every route enforces; anything
        # larger is clamped on the way out. The code sandbox learned this the
        # expensive way -- at 600s, 229 of its graded questions ran out of
        # clock and scored zero, three quarters of them at M0, where a model
        # that has only seen the demos reasons longest -- and a deadline has
        # to mean the same thing in both sandboxes for their numbers to sit in
        # one table. 1800s sits above that ceiling on purpose: a 1200s cut is
        # then a fault and retried, and only three capped attempts score wrong.
        "EVAL_HTTP_TIMEOUT": os.environ.get("EVAL_HTTP_TIMEOUT", "1800"),
        "EVAL_MAX_RETRIES": os.environ.get("EVAL_MAX_RETRIES", "2"),
        # A model whose upstream cannot take the usual width says so, and gets
        # the narrower of the two.
        box["parallel_env"]: str(min(parallel, spec.get("parallel", parallel))),
    }
    handle = log.open("w", encoding="utf-8")
    print(f"  start {sandbox}/{run} -> {log}", flush=True)
    return subprocess.Popen(
        [
            sys.executable, "-u", box["script"],
            "--model", spec["model"],
            "--run-id", run,
            box["short_flag"], spec["short"],
            "--framework", framework,
            "--track", track,
            "--budget-profile", budget_profile,
            *box["extra_args"],
            *(spec.get("extra_args") or {}).get(sandbox, []),
        ],
        cwd=str(DEV), env=env, stdout=handle, stderr=subprocess.STDOUT,
    )


def export(sandbox: str, runs: list[tuple[str, str]]) -> None:
    """Write the messages archive, HTML replay and cross-run index."""

    traces = [
        str(path)
        for path in (latest_trace(sandbox, short, run) for short, run in runs)
        if path is not None
    ]
    if not traces:
        print(f"nothing to export for {sandbox}", flush=True)
        return
    print(f"\n=== exporting {len(traces)} {sandbox} runs ===", flush=True)
    subprocess.run(
        [sys.executable, "scripts/export_run.py", *traces,
         "--out-dir", str(SANDBOXES[sandbox]["exports"])],
        cwd=str(DEV), check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sandbox", choices=["code", "logic", "both"], default="code",
    )
    parser.add_argument(
        "--models",
        default="",
        help=f"comma-separated subset of: {', '.join(BY_KEY)} (default: all)",
    )
    parser.add_argument(
        "--frameworks",
        default="baseline",
        help=(
            "comma-separated framework adapters: "
            f"{', '.join(sorted(FRAMEWORK_SPECS))} (default: baseline)"
        ),
    )
    parser.add_argument(
        "--track",
        choices=["controlled", "native", "open"],
        default="controlled",
        help="framework evaluation track",
    )
    parser.add_argument(
        "--budget-profile",
        default="c1",
        help="framework budget profile: c1/1x/2x/4x",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="runs to add on top of the first one",
    )
    parser.add_argument(
        "--from-run",
        type=int,
        default=2,
        help="number the queued runs from here (1 starts a model from scratch)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=8,
        help="held-out questions in flight per run",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=12,
        help="independent runs in flight at once",
    )
    parser.add_argument(
        "--stagger",
        type=int,
        default=0,
        help="optional seconds between starts (default: no staggering)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-wait",
        action="store_true",
        help="start now instead of waiting for the in-flight runs",
    )
    parser.add_argument(
        "--skip-health",
        action="store_true",
        help="queue repeats even for models with no usable first run",
    )
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="skip exact run ids that already have a complete score file",
    )
    parser.add_argument(
        "--force-existing",
        action="store_true",
        help="intentionally replace a completed run id (active ids still skip)",
    )
    args = parser.parse_args()

    if args.models:
        unknown = [k for k in args.models.split(",") if k not in BY_KEY]
        if unknown:
            print(f"unknown model keys: {', '.join(unknown)}", file=sys.stderr)
            return 2
        specs = [BY_KEY[k] for k in args.models.split(",")]
    else:
        specs = list(MODELS)
    frameworks = [
        normalize_framework_key(item)
        for item in args.frameworks.split(",")
        if item.strip()
    ]
    unknown_frameworks = [fw for fw in frameworks if fw not in FRAMEWORK_SPECS]
    if unknown_frameworks:
        print(
            f"unknown framework keys: {', '.join(unknown_frameworks)}",
            file=sys.stderr,
        )
        return 2
    boxes = ["code", "logic"] if args.sandbox == "both" else [args.sandbox]

    plan: list[tuple[str, list[tuple[dict, str, str]]]] = []
    for sandbox in boxes:
        # Repeats are independent experimental units. Keep every requested
        # repeat in one sandbox-level pool so a slow n2 cannot block n3 runs
        # from using otherwise idle provider capacity.
        batch = []
        for n in range(args.from_run, args.from_run + args.repeats):
            for spec in specs:
                base = run_id(spec, sandbox, n)
                for framework in frameworks:
                    batch.append((
                        spec,
                        framework_run_id(base, framework, args.track),
                        framework,
                    ))
        plan.append((sandbox, batch))
    if args.dry_run:
        if args.missing_only:
            plan = omit_completed(plan)
        for index, (sandbox, batch) in enumerate(plan, 1):
            print(f"batch {index} [{sandbox}]: "
                  + ", ".join(run for _, run, _ in batch))
        return 0

    if not args.skip_wait:
        # Another sandbox's queue is no reason to hold this one back; an
        # unrecognised run id might be, so it still counts.
        alive = [
            run for run in running_runs()
            if (sandbox_of_run(run) or "") in boxes
            or sandbox_of_run(run) is None
        ]
        if alive:
            print(f"in-flight: {', '.join(alive)}", flush=True)
            wait_for(alive)

    # Filter after the wait: a run that was in flight when this queue started
    # may have produced its final score while we were waiting.
    if args.missing_only:
        print("\n=== completed-run filter ===", flush=True)
        plan = omit_completed(plan)

    if args.from_run <= 1:
        args.skip_health = True
        print("starting from run 1; skipping baseline health precheck", flush=True)

    if not args.skip_health:
        print("\n=== baseline health ===", flush=True)
        healthy: set[tuple[str, str, str]] = set()
        for sandbox in boxes:
            for spec in specs:
                for framework in frameworks:
                    first = framework_run_id(
                        run_id(spec, sandbox, 1), framework, args.track)
                    ok, why = health(sandbox, spec["short"], first)
                    print(f"  {'OK  ' if ok else 'FAIL'} {sandbox}/{first}: {why}",
                          flush=True)
                    if ok:
                        healthy.add((sandbox, spec["key"], framework))
        if not healthy:
            print(
                "\nEvery baseline is unusable, so repeats would just measure "
                "the same fault. Fix it and rerun this queue.",
                flush=True,
            )
            return 1
        # One broken baseline shouldn't cost the others their repeats; a model
        # with no usable first run has nothing to compare a repeat against.
        skipped = sorted(
            f"{sandbox}/{spec['key']}:{framework}"
            for sandbox in boxes for spec in specs for framework in frameworks
            if (sandbox, spec["key"], framework) not in healthy
        )
        if skipped:
            print(f"  skipping repeats for: {', '.join(skipped)}", flush=True)
        plan = [
            (sandbox, [(spec, run, framework) for spec, run, framework in batch
                       if (sandbox, spec["key"], framework) in healthy])
            for sandbox, batch in plan
        ]

    done: dict[str, list[tuple[str, str]]] = {sandbox: [] for sandbox in boxes}
    for sandbox in boxes:
        for spec in specs:
            for framework in frameworks:
                first = framework_run_id(
                    run_id(spec, sandbox, 1), framework, args.track)
                if latest_trace(sandbox, spec["short"], first) is not None:
                    done[sandbox].append((spec["short"], first))

    for index, (sandbox, batch) in enumerate(plan, 1):
        if not batch:
            continue
        print(f"\n=== batch {index}/{len(plan)} [{sandbox}] ===", flush=True)
        # Runs are independent and may share a provider. `--concurrent` is the
        # explicit capacity bound; each completion immediately pulls the next
        # repeat into the available slot.
        pending = list(batch)
        active: list[tuple[dict, str, str, subprocess.Popen, Path]] = []
        while pending or active:
            while pending and len(active) < args.concurrent:
                spec, run, framework = pending.pop(0)
                if run in running_runs():
                    print(f"  skip in-flight {sandbox}/{run}", flush=True)
                    continue
                if (not args.force_existing
                        and completed_score(sandbox, spec["short"], run)):
                    print(f"  skip complete {sandbox}/{run}", flush=True)
                    done[sandbox].append((spec["short"], run))
                    continue
                lock = claim_run(sandbox, run)
                if lock is None:
                    print(f"  skip claimed {sandbox}/{run}", flush=True)
                    continue
                try:
                    proc = launch(
                        spec,
                        sandbox,
                        run,
                        args.parallel,
                        framework=framework,
                        track=args.track,
                        budget_profile=args.budget_profile,
                    )
                except BaseException:
                    release_run(lock)
                    raise
                active.append((
                    spec,
                    run,
                    framework,
                    proc,
                    lock,
                ))
                if (pending or len(active) > 1) and args.stagger:
                    time.sleep(args.stagger)
            for item in list(active):
                spec, run, framework, proc, lock = item
                code = proc.poll()
                if code is None:
                    continue
                active.remove(item)
                release_run(lock)
                print(
                    f"  done {run} (exit {code}) at {time.strftime('%H:%M:%S')}"
                    f"  [{len(pending)} queued, {len(active)} running]",
                    flush=True,
                )
                if code == 0:
                    done[sandbox].append((spec["short"], run))
            if active:
                time.sleep(20)

    print("\nall repeats finished", flush=True)
    for sandbox in boxes:
        export(sandbox, done[sandbox])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
