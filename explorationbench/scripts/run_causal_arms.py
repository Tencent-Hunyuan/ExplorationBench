#!/usr/bin/env python3
"""Launch the causal control arms under the native-tool protocol.

The contrasts this enables are the ones the paper currently cannot make:

    self   - none      total effect of the exploration protocol
    think  - none      repeated inference and extra calls, with zero evidence
    random - think     generic environment feedback
    passive- random    evidence quality
    self   - passive   choosing your own experiments
    oracle - self      remaining discovery gap, given the true rules

``passive`` and ``random`` replay a donor's probes through the *same* tool path
the treatment uses: the model still issues the tool calls and still receives
tool results, and only the probe is substituted. That is what makes
``self - passive`` a probe-choice contrast rather than a comparison between the
native tool protocol and a text one. Their donors are built by
``build_control_donors.py`` and verified against
``dev/causal_controls/donor_manifest.json`` before any paid call.

The default registry covers the two primary contrast systems and two faster
DeepSeek controls; ``--systems`` selects the subset launched in a wave.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "explorationbench"

# The harness loads the developer's local config itself, so this launcher
# never needed to. It does now: a system may be pinned to its own account,
# and the launcher is what puts that key in the child's environment, so it
# has to be able to read it before the child exists.
if str(DEV) not in sys.path:
    sys.path.insert(0, str(DEV))
from common import local_config  # noqa: E402

local_config.load()


#: The v2 control cohort: four systems rather than eleven, because the
#: contrast needs range rather than coverage and each arm costs a full
#: campaign. One near the top, one mid, one at the floor, plus hy4.
#:
#: Chosen on reliability as well as rank. GPT-5.6 sits second but refused
#: a third of its sampled calls behind an Azure rate limit; Qwen, Doubao
#: and DeepSeek Flash answer in a median 500--630s a question. None of
#: them survives four arms times three traces.
#:
#: Routes are GatewayA's, matching the autonomous runs these are subtracted
#: from. The evaluation gateway below is out of budget and answers 402.
V2_SYSTEMS = {
    "v2_gpt56": {
        "model": "gateway_a/azure/gpt-5.6-sol",
        "short": "gpt-5.6-sol-max",
        "effort": "max",
        # The narrow lane the autonomous campaign used. This route answers
        # a token-rate 429 well below the width the others take, and it is
        # the one system whose sampled calls were refused in bulk, so the
        # arms get the same allowance the treatment had rather than more.
        "parallel": 12,
    },
    "v2_opus5": {
        "model": "gateway_a/anthropic/claude-opus-5",
        "short": "opus-5",
        "effort": "max",
        "parallel": 48,
        # Its own GatewayA account, so the strongest system's arms do not
        # queue behind the other nine on one pool. Only the variable name
        # is here; the key lives in the untracked local config. On
        # 2026-09-23 that account answered PlatformNoAvailableAccount to
        # every call for hours while the shared one served Opus fine, so a
        # launcher can set CTRL_OPUS5_SHARED_KEY=1 to fall back to it.
        "key_env": (None if os.environ.get("CTRL_OPUS5_SHARED_KEY") == "1"
                    else "GATEWAY_A_API_KEY_OPUS5"),
    },
    "v2_grok46": {
        "model": "gateway_a/grok-4.6",
        "short": "grok-4.6",
        # This route carries a tier above 'high' and rejects 'max'.
        "effort": "xhigh",
        "parallel": 30,
    },
    "v2_dspro": {
        "model": "gateway_a/deepseek/deepseek-v4-pro",
        "short": "deepseek-v4-pro-native",
        "effort": "max",
        "parallel": 30,
    },
    "v2_hy4": {
        "model": "gateway_a/gateway_b/hy4-preview",
        "short": "hy4-preview",
        # The vendor ships deep reasoning on by default with no tier above it.
        "effort": "high",
        "parallel": 30,
    },
    # The remaining six of the reported ten. The study began with four --
    # enough to span the range of autonomous outcomes -- and is widened
    # here so the control table covers the same cohort as every other
    # table in the paper, rather than a subset a reader has to be told
    # about. Routes, tiers and lane widths are the autonomous campaign's,
    # because these arms are subtracted from those numbers.
    "v2_qwen38": {
        "model": "gateway_a/ali/qwen3.8-max-0902",
        "short": "qwen3.8-max-0902",
        "effort": "max",
        "parallel": 48,
    },
    "v2_dsflash41": {
        "model": "gateway_a/deepseek/deepseek-flash",
        "short": "deepseek-flash",
        "effort": "max",
        "parallel": 48,
    },
    "v2_kimik3": {
        "model": "gateway_a/kimi-k3",
        "short": "kimi-k3",
        "effort": "max",
        "parallel": 48,
    },
    "v2_gemini38": {
        "model": "gateway_a/gemini-3.8-flash",
        "short": "gemini-3.8-flash-high",
        # Gemini carries its tier in thinking.thinkingLevel rather than
        # reasoning_effort, so the wire shows no tier for this one.
        "effort": "max",
        "parallel": 48,
    },
    "v2_doubao21": {
        "model": "gateway_a/doubao-seed-2-1-pro-260915",
        "short": "doubao-seed-2.1-pro-0915",
        # Four tiers ending at high; there is nothing above it.
        "effort": "high",
        "parallel": 48,
    },
}

SYSTEMS = {
    "gpt56": {
        "model": "api_azure_openai_gpt-5.6-sol",
        "short": "gpt-5.6-sol-max",
        "effort": "max",
    },
    "opus48": {
        "model": "api_aws_third_anthropic.claude-opus-4-8",
        "short": "opus-4-8",
        "effort": "max",
    },
    "dsv4-flash": {
        "model": "messages/api_deepseek_deepseek-v4-flash",
        "short": "deepseek-v4-flash",
        "effort": "max",
    },
    "dsv4-pro": {
        "model": "messages/api_deepseek_deepseek-v4-pro",
        "short": "deepseek-v4-pro-native",
        "effort": "max",
    },
    "gemini36": {
        "model": "api_google_gemini-3.6-flash",
        "short": "gemini-3.6-flash-high",
        "effort": "high",
    },
    "grok45": {
        "model": "api_xai_grok-4.5",
        "short": "grok-4.5",
        "effort": "high",
    },
    "hy3": {
        "model": "api_gateway_b_hy3",
        "short": "hy3-gateway_b-reasoning",
        "effort": "high",
    },
    # The self-hosted replacement, kept as its own system so its artifacts sit
    # beside the gateway ones rather than overwriting them: the two are
    # different services and a column must not mix them silently.
    "hy3local": {
        "model": "explorationbench-experiment",
        "short": "hy3-local",
        "effort": "high",
        # The deployment answers `all instances hit concurrency limit` well
        # below the fifty requests it is sized for, and dropping the question
        # workers from six to four did not reduce the rate. AlienCode carries
        # far more context per request than AlienLogic and is the arm that
        # keeps hitting it, so the ceiling looks like occupancy rather than
        # request count.
        "parallel": 2,
    },
    "qwen38": {
        "model": "api_ali_qwen3.8-max",
        "short": "qwen3.8-max",
        "effort": "max",
        # This was 1. The gateway used to answer four workers with
        # Throttling.Concurrency and eight workers across three runs managed
        # 16 graded questions an hour, so one in-flight request was all the
        # allowance there was. The account now carries a 200 RPM grant and
        # measures clean at far more than that, and the old ceiling was
        # silently clamping every --parallel the caller asked for.
        "parallel": 48,
    },
    "kimi3": {
        "model": "api_moonshot_kimi-k3",
        "short": "kimi-k3",
        "effort": "max",
    },
    "doubao21": {
        "model": "api_doubao_doubao-seed-2-1-pro-260628",
        "short": "doubao-seed-2.1-pro",
        "effort": "high",
    },
    "hy4": {
        # The vendor ships two tiers for this model, deep reasoning and off, and
        # deep reasoning is the default. OpenRouter's low/medium/high/xhigh/max
        # enum is its own abstraction over providers; anything above "high"
        # has no native tier to map to here, so "high" is already the ceiling.
        "model": "openrouter/tencent/hy4-preview",
        "short": "hy4-preview",
        "effort": "high",
        # The account 429s hard past a couple of concurrent requests.
        "parallel": 2,
    },
}

SANDBOXES = {
    "code": {
        "script": "sandboxes/code/run_eval.py",
        "short_flag": "--short",
        "extra_args": [],
        "parallel_env": "ALIENCODE_PARALLEL_TESTS",
        "queue_logs": ROOT / "logs" / "code" / "queue" / "causal",
    },
    "logic": {
        "script": "sandboxes/logic/run_eval.py",
        "short_flag": "--model-short",
        "extra_args": ["--defer-tests", "--explore-loops", "4"],
        "parallel_env": "ALIENLOGIC_DEFER_WORKERS",
        "queue_logs": ROOT / "logs" / "logic" / "queue" / "causal",
    },
}

SYSTEMS.update(V2_SYSTEMS)

ARMS = ("none", "think", "random", "passive", "oracle")
DONOR_ARMS = ("random", "passive")
DONOR_DIR = DEV / "causal_controls" / "donors"
MANIFEST = DEV / "causal_controls" / "donor_manifest.json"
LOCK_DIR = ROOT / "logs" / "control_locks"


#: Which trace each v2 system's passive arm replays: its own best one,
#: measured after three-sample scoring. Passive hands the model back the
#: probes it chose itself when it did best, so `self - passive` is about
#: choosing as you go rather than about which probes were chosen -- the
#: probes are the same either way.
#: Per sandbox, because a model's best exploration is not the same run in
#: both worlds: Grok's best AlienCode trace is n3 and its best AlienLogic
#: trace is n2, and replaying the wrong one would hand the arm probes that
#: were never this model's best in the world being scored.
#: Each system's passive arm replays its own best trace, chosen by that
#: trace's M4 under three-sample scoring, so `self - passive` isolates
#: choosing the probes from reading them: same model, same probes, same
#: number of them, and only the deciding removed.
V2_PASSIVE_DONOR = {
    ("v2_gpt56", "code"): "v2_gpt56_n2",
    ("v2_opus5", "code"): "v2_opus5_n3",
    ("v2_qwen38", "code"): "v2_qwen38_n1",
    ("v2_grok46", "code"): "v2_grok46_n3",
    ("v2_dspro", "code"): "v2_dspro_n3",
    ("v2_dsflash41", "code"): "v2_dsflash41_n3",
    ("v2_kimik3", "code"): "v2_kimik3_n3",
    ("v2_gemini38", "code"): "v2_gemini38_n1",
    ("v2_hy4", "code"): "v2_hy4_n2",
    ("v2_doubao21", "code"): "v2_doubao21_n1",
    ("v2_gpt56", "logic"): "v2_gpt56_n2",
    ("v2_opus5", "logic"): "v2_opus5_n2",
    ("v2_qwen38", "logic"): "v2_qwen38_n3",
    ("v2_grok46", "logic"): "v2_grok46_n2",
    ("v2_dspro", "logic"): "v2_dspro_n3",
    ("v2_dsflash41", "logic"): "v2_dsflash41_n1",
    ("v2_kimik3", "logic"): "v2_kimik3_n3",
    ("v2_gemini38", "logic"): "v2_gemini38_n2",
    ("v2_hy4", "logic"): "v2_hy4_n1",
    ("v2_doubao21", "logic"): "v2_doubao21_n1",
}
V2_CONTROL_SYSTEMS = frozenset(
    system for system, _sandbox in V2_PASSIVE_DONOR)

#: The random arm is one sequence for the whole cohort, not one per model.
#: Its probes are model-independent already, but their count per call was
#: taken from each model's own donor and the seed from its own run id, so
#: two models' random arms differed by their draw as well as by the model
#: and the difference could not be attributed. Both now come from a single
#: profile, so the arm is a fixed baseline every system meets identically.
#:
#: Grok's best trace supplies the volume: it is the mid-ranked system of
#: the four, so the allowance is neither the strongest model's nor the
#: weakest.
V2_RANDOM_PROFILE = {"code": "v2_grok46_n3", "logic": "v2_grok46_n2"}
V2_RANDOM_PROFILE_SHORT = "grok-4.6"
V2_RANDOM_SEED_TAG = "v2-shared-random"


def donor_for(arm: str, system: str, sandbox: str) -> Path:
    """The attested donor whose probes this control replays.

    These arms replay the free-seed autonomous runs, so the locked-seed banks
    that the `ls_` cohort produced are excluded by name. They sit in the same
    folder, and taking the alphabetically last match quietly picked one of them
    for every model except qwen, whose free-seed run id happens to sort after
    `ls_`.
    """

    if system in V2_CONTROL_SYSTEMS:
        if arm == "random":
            run = V2_RANDOM_PROFILE[sandbox]
            short = V2_RANDOM_PROFILE_SHORT
        else:
            run = V2_PASSIVE_DONOR[(system, sandbox)]
            short = SYSTEMS[system]["short"]
        named = DONOR_DIR / f"donor_alien_{sandbox}_{short}_{run}.jsonl"
        if not named.is_file():
            raise SystemExit(
                f"no donor {named.name}; run "
                f"scripts/build_control_donors.py --v2 first")
        return named

    matches = sorted(
        path
        for path in DONOR_DIR.glob(
            f"donor_alien_{sandbox}_{SYSTEMS[system]['short']}_*.jsonl")
        if "_ls_" not in path.name
    )
    if not matches:
        raise SystemExit(
            f"no donor for {system}/{sandbox}; run "
            f"scripts/build_control_donors.py first")
    return matches[-1]


def command(arm: str, system: str, sandbox: str, run_id: str, parallel: int,
            fresh: bool = False):
    """Build the argv and environment for one control run."""

    spec = SYSTEMS[system]
    box = SANDBOXES[sandbox]
    env = {
        **os.environ,
        "EVAL_TOOL_MODE": "1",
        "EVAL_REASONING_EFFORT": spec["effort"],
        "EVAL_FRAMEWORK": "baseline",
        "EVAL_TRACK": "controlled",
        "EVAL_BUDGET_PROFILE": "c1",
        # Named, not left to the harness default. A control arm exists to
        # be subtracted from the autonomous arm, and the autonomous arm is
        # v2: seventy tasks, fixed calibration, twelve tool calls a round.
        # Without these the harness quietly runs the previous generation
        # and the contrast compares two different benchmarks. AlienLogic
        # lost eighteen traces to exactly this omission.
        "ALIENCODE_PROTOCOL_V2": "1",
        "ALIENCODE_TASK_SET": os.environ.get("ALIENCODE_TASK_SET", "v2"),
        "ALIENLOGIC_PROTOCOL_V2": "1",
        "EVAL_MAX_TOOL_CALLS": os.environ.get("EVAL_MAX_TOOL_CALLS", "12"),
        "EVAL_EXPLORE_ROUNDS": os.environ.get("EVAL_EXPLORE_ROUNDS", "1"),
        box["parallel_env"]: str(min(parallel, spec.get("parallel", parallel))),
    }
    if sandbox == "logic":
        # A resumed logic run scores each milestone as it reaches it, and
        # that path reads its own width -- default one -- rather than the
        # deferred pool's; Opus arms graded seventy theorems serially.
        env["ALIENLOGIC_HELDOUT_WORKERS"] = env[box["parallel_env"]]
    if sandbox == "code":
        # Every route cuts a request at 1200s. An 1800s deadline makes such a
        # cut a retryable fault rather than a spent budget, which is how the
        # first control answers were scored, and the autonomous timeouts are
        # re-asked under the same rule. Two retries bound a question at about
        # an hour; eval.local.toml's twelve let one hang for six.
        env["EVAL_HTTP_TIMEOUT"] = os.environ.get("EVAL_HTTP_TIMEOUT", "1800")
        env["EVAL_TRAJECTORY_TIMEOUT"] = os.environ.get(
            "EVAL_TRAJECTORY_TIMEOUT", "1800")
        env["EVAL_MAX_RETRIES"] = os.environ.get("EVAL_MAX_RETRIES", "2")
    # A system may draw on its own account. The routing layer reads one
    # variable, so the override is applied here rather than there: the
    # per-system name is resolved and its value copied onto the name the
    # client already looks for. Nothing is logged -- this is a credential.
    key_env = spec.get("key_env")
    if key_env:
        key = os.environ.get(key_env, "").strip()
        if not key:
            raise SystemExit(
                f"{system} is configured to use {key_env}, which is not "
                f"set; add it under [env] in dev/eval.local.toml")
        env["GATEWAY_A_API_KEY"] = key
    if system == "opus48" and sandbox == "logic":
        # Long controlled contexts can spend the ordinary output allowance on
        # adaptive thinking and return no proof. Keep the protocol unchanged,
        # but give thinking plus the visible proof enough room.
        env.update({
            "EVAL_DEFAULT_MAX_TOKENS": "24000",
            "EVAL_SUMMARY_MAX_TOKENS": "32000",
            "EVAL_TEST_MAX_TOKENS": "32000",
            "EVAL_TEST_RETRY_MAX_TOKENS": "48000",
        })
    argv = [
        sys.executable, "-u", box["script"],
        "--model", spec["model"],
        "--run-id", run_id,
        box["short_flag"], spec["short"],
        "--framework", "baseline",
        "--track", "controlled",
        "--budget-profile", "c1",
    ]

    if arm == "oracle":
        # Oracle is an intervention rather than a control mode, so it stays on
        # the self path. AlienLogic has a dedicated flag that stops after the
        # single post-injection milestone. AlienCode always walks its explore
        # loops, so the loop count is cut to the minimum the harness accepts
        # and only the post-injection M0 is read as the ceiling.
        if sandbox == "logic":
            argv.append("--oracle-only")
        else:
            env["EVAL_ORACLE_ONLY"] = "1"
            env["EVAL_EXPLORE_LOOPS"] = "1"
    else:
        argv += ["--control-mode", arm]
        if arm in DONOR_ARMS:
            argv += [
                "--control-bank", str(donor_for(arm, system, sandbox)),
                "--control-manifest", str(MANIFEST),
            ]
        if arm == "random" and system in V2_CONTROL_SYSTEMS:
            # Keyed on the cohort rather than the run, so all four systems
            # draw the same sequence.
            env["EVAL_RANDOM_SEED_TAG"] = V2_RANDOM_SEED_TAG

    argv += box["extra_args"]
    if not fresh:
        # A relaunch of the same id is nearly always picking a killed or reaped
        # run back up, so resume unless the caller explicitly asked to start
        # over; --fresh is what says "discard what is there".
        #
        # Both sandboxes are told the same thing. AlienCode would also resume
        # on its own, since it looks for a checkpoint named after the run id
        # whenever --fresh is absent, but leaving that implicit meant the two
        # sandboxes recovered by different routes and only one of them said so
        # at the call site. An edit to either side could then drop resume
        # silently, which on a model as slow as Hy4 preview costs hours.
        argv.append("--resume")
    return argv, env


def run_id_for(arm: str, system: str, sandbox: str, repeat: int) -> str:
    return f"ctrl_{arm}_{system}_{sandbox}_n{repeat}"


def result_exists(sandbox: str, run_id: str) -> bool:
    results = ROOT / "logs" / sandbox / "results"
    return any(results.glob(f"eval_results_alien_{sandbox}_*_{run_id}.json"))


def active_run_ids() -> set[str]:
    listing = subprocess.run(
        ["ps", "-eo", "args"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {
        match.group(1)
        for match in re.finditer(r"--run-id\s+([^\s]+)", listing)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument(
        "--systems", nargs="+", default=list(SYSTEMS), choices=list(SYSTEMS))
    parser.add_argument(
        "--sandboxes", nargs="+", default=list(SANDBOXES),
        choices=list(SANDBOXES))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--repeat-start", type=int, default=1,
        help="first repeat index to launch (inclusive)")
    parser.add_argument("--parallel", type=int, default=8,
                        help="held-out questions in flight per run")
    parser.add_argument("--concurrent", type=int, default=2,
                        help="control runs in flight at once")
    parser.add_argument(
        "--fresh", action="store_true",
        help="restart existing AlienCode run ids instead of resuming checkpoints")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="do not launch run ids that already have a result artifact")
    parser.add_argument(
        "--skip-active", action="store_true",
        help="do not launch run ids already active in another queue")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.repeat_start < 1 or args.repeat_start > args.repeats:
        parser.error("--repeat-start must be between 1 and --repeats")
    # `--fresh` used to be rejected for AlienLogic because that sandbox had no
    # checkpoint and every launch was a fresh run anyway. Now that it does,
    # the flag means the same thing in both: ignore what is on disk.

    jobs = [
        (arm, system, sandbox, repeat)
        for arm in args.arms
        for system in args.systems
        for sandbox in args.sandboxes
        for repeat in range(args.repeat_start, args.repeats + 1)
    ]

    for box in SANDBOXES.values():
        box["queue_logs"].mkdir(parents=True, exist_ok=True)
    LOCK_DIR.mkdir(parents=True, exist_ok=True)

    running: list[tuple[str, subprocess.Popen, object]] = []
    failures: list[str] = []
    active = active_run_ids() if args.skip_active else set()

    def reap(block: bool) -> None:
        while running and (block or len(running) >= args.concurrent):
            for index, (name, proc, handle) in enumerate(running):
                if proc.poll() is not None:
                    handle.close()
                    status = (
                        "ok" if proc.returncode == 0
                        else "skip-locked" if proc.returncode == 75
                        else "FAILED"
                    )
                    print(f"  {status} {name} (rc={proc.returncode})",
                          flush=True)
                    if proc.returncode not in (0, 75):
                        failures.append(name)
                    running.pop(index)
                    break
            else:
                time.sleep(15)
                continue
            if not block:
                return

    for arm, system, sandbox, repeat in jobs:
        run = run_id_for(arm, system, sandbox, repeat)
        if args.skip_existing and result_exists(sandbox, run):
            print(f"  skip {run}: result exists", flush=True)
            continue
        if args.skip_active and run in active:
            print(f"  skip {run}: already active", flush=True)
            continue
        argv, env = command(arm, system, sandbox, run, args.parallel,
                            fresh=args.fresh)
        if args.fresh and sandbox == "code":
            argv.append("--fresh")
        env["EVAL_RUN_LOCK_HELD"] = "1"
        argv = [
            "flock", "-n", "-E", "75",
            str(LOCK_DIR / f"{run}.lock"),
            *argv,
        ]
        if args.dry_run:
            print(f"{run}\n  {' '.join(argv)}", flush=True)
            continue
        reap(block=False)
        log = SANDBOXES[sandbox]["queue_logs"] / f"{run}.log"
        # Appended, not truncated: this file is where a crashing child's
        # traceback lands, and the watchdog relaunches automatically. Opening
        # it "w" meant every retry erased the reason the last attempt died,
        # which left a run looping with nothing on disk to explain it.
        handle = log.open("a", encoding="utf-8")
        handle.write(
            f"\n{'=' * 72}\n"
            f"[launch] {time.strftime('%Y-%m-%d %H:%M:%S')}  {run}\n"
            f"{'=' * 72}\n"
        )
        handle.flush()
        print(f"  start {run} -> {log}", flush=True)
        running.append((
            run,
            subprocess.Popen(argv, cwd=str(DEV), env=env,
                             stdout=handle, stderr=subprocess.STDOUT),
            handle,
        ))

    reap(block=True)
    if failures:
        print("failed: " + ", ".join(failures), flush=True)
        return 1
    print("all causal control runs complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
