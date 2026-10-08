#!/usr/bin/env python3
"""Build attested donor banks for the passive and random control arms.

A donor is one clean ``self`` run whose executable probes the control replays.
The banks shipped before this script were captured under the archived
fenced-block protocol, where a round's probe was a code block inside the
assistant's prose. A native-tool round instead carries one probe per tool call,
so the bank records them as an ordered ``codes`` list and the harness replays
them through the same tool path the treatment uses.

The manifest binds each donor to the model, sandbox, episode, loop and round
structure, the manual, and the environment and rule digests it was recorded
against, so a later edit to any of them fails preflight instead of silently
producing an incomparable control.

    python3 dev/scripts/build_control_donors.py --list
    python3 dev/scripts/build_control_donors.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
from pathlib import Path

# Set from --force; a rewrite that a replay arm is pinned to invalidates it.
_FORCE = False

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "explorationbench"
OUT_DIR = DEV / "causal_controls" / "donors"
MANIFEST = DEV / "causal_controls" / "donor_manifest.json"

# One donor per (system, sandbox): the run whose probes the matched passive
# control replays. Both systems the control arms cover are included.
# The locked-seed AlienCode cohort. Its controls cannot replay the banks above:
# those were recorded when each system chose how much seed evidence to carry
# into M0, so replaying one into a locked-seed run would pair a standardized
# baseline with probes designed against a different one. Every locked arm
# replays a locked autonomous run of the same system instead.
LOCKED_SEED_DONORS = [
    {"sandbox": "code", "model_short": "gpt-5.6-sol-max", "run_id": "ls_gpt56",
     "episode": "code_default"},
    {"sandbox": "code", "model_short": "opus-4-8", "run_id": "ls_opus48",
     "episode": "code_default"},
    {"sandbox": "code", "model_short": "qwen3.8-max", "run_id": "ls_qwen38",
     "episode": "code_default"},
    {"sandbox": "code", "model_short": "kimi-k3", "run_id": "ls_kimik3",
     "episode": "code_default"},
    {"sandbox": "code", "model_short": "hy4-preview", "run_id": "ls_hy4",
     "episode": "code_default"},
    {"sandbox": "code", "model_short": "grok-4.5", "run_id": "ls_grok45",
     "episode": "code_default"},
    {"sandbox": "code", "model_short": "deepseek-v4-pro-native",
     "run_id": "ls_dspro", "episode": "code_default"},
    {"sandbox": "code", "model_short": "deepseek-v4-flash",
     "run_id": "ls_dsflash", "episode": "code_default"},
    {"sandbox": "code", "model_short": "doubao-seed-2.1-pro",
     "run_id": "ls_doubao21", "episode": "code_default"},
    {"sandbox": "code", "model_short": "gemini-3.6-flash-high",
     "run_id": "ls_gemini36", "episode": "code_default"},
]

#: The v2 control cohort's passive banks: each system's own best trace,
#: ranked after three-sample scoring rather than on its single answers.
#: Passive then replays the probes the model itself chose on its best run,
#: so `self - passive` isolates choosing as you go from what was chosen.
#:
#: Grok's is also the random arm's volume profile -- the mid-ranked system
#: of the four, so the allowance is neither the strongest model's nor the
#: weakest -- and with EVAL_RANDOM_SEED_TAG fixed the four random arms are
#: one sequence rather than four draws.
#: Ranked per sandbox, not once: a model's best exploration is a different
#: run in the two worlds. Grok's best AlienCode trace is n3 and its best
#: AlienLogic trace is n2.
V2_DONORS = [
    {"sandbox": "code", "model_short": "gpt-5.6-sol-max",
     "run_id": "v2_gpt56_n2", "episode": "code_default"},
    {"sandbox": "code", "model_short": "grok-4.6",
     "run_id": "v2_grok46_n3", "episode": "code_default"},
    {"sandbox": "code", "model_short": "deepseek-v4-pro-native",
     "run_id": "v2_dspro_n3", "episode": "code_default"},
    {"sandbox": "code", "model_short": "hy4-preview",
     "run_id": "v2_hy4_n2", "episode": "code_default"},
    # The other six of the reported ten, added when the control study was
    # widened from four systems to the whole cohort. Each is that
    # system's own best sampled trace, the same rule the first four use.
    {"sandbox": "code", "model_short": "opus-5",
     "run_id": "v2_opus5_n3", "episode": "code_default"},
    {"sandbox": "code", "model_short": "qwen3.8-max-0902",
     "run_id": "v2_qwen38_n1", "episode": "code_default"},
    {"sandbox": "code", "model_short": "deepseek-flash",
     "run_id": "v2_dsflash41_n3", "episode": "code_default"},
    {"sandbox": "code", "model_short": "kimi-k3",
     "run_id": "v2_kimik3_n3", "episode": "code_default"},
    {"sandbox": "code", "model_short": "gemini-3.8-flash-high",
     "run_id": "v2_gemini38_n1", "episode": "code_default"},
    {"sandbox": "code", "model_short": "doubao-seed-2.1-pro-0915",
     "run_id": "v2_doubao21_n1", "episode": "code_default"},
    {"sandbox": "logic", "model_short": "gpt-5.6-sol-max",
     "run_id": "v2_gpt56_n2", "episode": "demo_seeded"},
    {"sandbox": "logic", "model_short": "grok-4.6",
     "run_id": "v2_grok46_n2", "episode": "demo_seeded"},
    {"sandbox": "logic", "model_short": "deepseek-v4-pro-native",
     "run_id": "v2_dspro_n3", "episode": "demo_seeded"},
    {"sandbox": "logic", "model_short": "hy4-preview",
     "run_id": "v2_hy4_n1", "episode": "demo_seeded"},
    {"sandbox": "logic", "model_short": "opus-5",
     "run_id": "v2_opus5_n2", "episode": "demo_seeded"},
    {"sandbox": "logic", "model_short": "qwen3.8-max-0902",
     "run_id": "v2_qwen38_n3", "episode": "demo_seeded"},
    {"sandbox": "logic", "model_short": "deepseek-flash",
     "run_id": "v2_dsflash41_n1", "episode": "demo_seeded"},
    {"sandbox": "logic", "model_short": "kimi-k3",
     "run_id": "v2_kimik3_n3", "episode": "demo_seeded"},
    {"sandbox": "logic", "model_short": "gemini-3.8-flash-high",
     "run_id": "v2_gemini38_n2", "episode": "demo_seeded"},
    {"sandbox": "logic", "model_short": "doubao-seed-2.1-pro-0915",
     "run_id": "v2_doubao21_n1", "episode": "demo_seeded"},
]

DONORS = [
    {
        "sandbox": "code",
        "model_short": "gpt-5.6-sol-max",
        "run_id": "gpt56_max",
        "episode": "code_default",
    },
    {
        "sandbox": "code",
        "model_short": "opus-4-8",
        "run_id": "full_opus48",
        "episode": "code_default",
    },
    {
        "sandbox": "logic",
        "model_short": "gpt-5.6-sol-max",
        "run_id": "gpt56_max_l2",
        "episode": "demo_seeded",
    },
    {
        "sandbox": "logic",
        "model_short": "opus-4-8",
        "run_id": "opus48_l2",
        "episode": "demo_seeded",
    },
    {
        "sandbox": "code",
        "model_short": "deepseek-v4-flash",
        "run_id": "dsv4_flash",
        "episode": "code_default",
    },
    {
        "sandbox": "logic",
        "model_short": "deepseek-v4-flash",
        "run_id": "dsv4_flash_l2",
        "episode": "demo_seeded",
    },
    {
        "sandbox": "code",
        "model_short": "deepseek-v4-pro-native",
        "run_id": "dsv4_pro_native_max",
        "episode": "code_default",
    },
    {
        "sandbox": "logic",
        "model_short": "deepseek-v4-pro-native",
        "run_id": "dsv4_pro_native_max_l2",
        "episode": "demo_seeded",
    },
    {
        "sandbox": "code",
        "model_short": "gemini-3.6-flash-high",
        "run_id": "gemini36_high",
        "episode": "code_default",
    },
    {
        "sandbox": "logic",
        "model_short": "gemini-3.6-flash-high",
        "run_id": "gemini36_high_l2",
        "episode": "demo_seeded",
    },
    {
        "sandbox": "code",
        "model_short": "grok-4.5",
        "run_id": "grok45_high",
        "episode": "code_default",
    },
    {
        "sandbox": "logic",
        "model_short": "grok-4.5",
        "run_id": "grok45_high_l2",
        "episode": "demo_seeded",
    },
    {
        "sandbox": "code",
        "model_short": "hy3-gateway_b-reasoning",
        "run_id": "hy3_gateway_b_reasoning_n3",
        "episode": "code_default",
    },
    {
        "sandbox": "logic",
        "model_short": "hy3-gateway_b-reasoning",
        "run_id": "hy3_gateway_b_reasoning_l2",
        "episode": "demo_seeded",
    },
    # The self-hosted Hy 3 is its own system: a control replays the probes of
    # an autonomous run of the same system, so it cannot borrow the gateway
    # build's bank.
    {
        "sandbox": "logic",
        "model_short": "hy3-local",
        "run_id": "hy3local_l1",
        "episode": "demo_seeded",
    },
    {
        "sandbox": "code",
        "model_short": "hy3-local",
        "run_id": "hy3local",
        "episode": "code_default",
    },
    {
        "sandbox": "code",
        "model_short": "hy4-preview",
        "run_id": "hy4probe",
        "episode": "code_default",
    },
    {
        "sandbox": "logic",
        "model_short": "hy4-preview",
        "run_id": "hy4_l1",
        "episode": "demo_seeded",
    },
    {
        "sandbox": "code",
        "model_short": "qwen3.8-max",
        "run_id": "qwen38_max",
        "episode": "code_default",
    },
    {
        "sandbox": "logic",
        "model_short": "qwen3.8-max",
        "run_id": "qwen38_max_l2",
        "episode": "demo_seeded",
    },
    {
        "sandbox": "code",
        "model_short": "kimi-k3",
        "run_id": "kimik3_max",
        "episode": "code_default",
    },
    {
        "sandbox": "logic",
        "model_short": "kimi-k3",
        "run_id": "kimik3_max_l2",
        "episode": "demo_seeded",
    },
    {
        "sandbox": "code",
        "model_short": "doubao-seed-2.1-pro",
        "run_id": "doubao21_high",
        "episode": "code_default",
    },
    {
        "sandbox": "logic",
        "model_short": "doubao-seed-2.1-pro",
        "run_id": "doubao21_high_l2",
        "episode": "demo_seeded",
    },
    *LOCKED_SEED_DONORS,
    *V2_DONORS,
]

# The shared paper horizon is four exploration loops. Some AlienLogic runs
# continued to M8 as a long-horizon diagnostic; a donor from one of those must
# be cut back, because preflight matches the loop structure of the run it is
# replayed into and a longer donor is not a matched control.
MAX_LOOPS = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Digests preflight compares against. Each sandbox names its own files, and
# getting one wrong turns into a preflight rejection at launch rather than a
# quiet mismatch, so they are resolved from one table.
_ENVIRONMENT = {"code": "execution.py", "logic": "engine.py"}
_RULES = {"code": "engine.py", "logic": "episodes.py"}


def environment_file(sandbox: str) -> Path:
    return DEV / "sandboxes" / sandbox / _ENVIRONMENT[sandbox]


def rule_file(sandbox: str) -> Path:
    return DEV / "sandboxes" / sandbox / _RULES[sandbox]


def manual_digest(sandbox: str) -> str:
    """Hash of the wrong manual the model is shown, as the harness hashes it."""

    if sandbox == "code":
        text = (DEV / "sandboxes" / "code" / "manual.md").read_text(
            encoding="utf-8").strip()
    else:
        if str(DEV) not in sys.path:
            sys.path.insert(0, str(DEV))
        from sandboxes.logic.engine import REFERENCE_MANUAL

        text = REFERENCE_MANUAL
    return hashlib.sha256(text.encode()).hexdigest()


def find_result(sandbox: str, model_short: str, run_id: str) -> Path | None:
    """The authoritative result file for a run id, repaired copy preferred."""

    results = ROOT / "logs" / sandbox / "results"
    pattern = re.compile(
        rf"^eval_results_alien_{sandbox}_(?:\d{{8}}_\d{{6}}_)?"
        rf"{re.escape(model_short)}_(?:demo_seeded_|task_warmup_)?"
        rf"{re.escape(run_id)}$"
    )
    found = []
    for path in sorted(results.glob("*.json")):
        stem = path.stem
        if stem.endswith("_redone") or not pattern.match(stem):
            continue
        repaired = path.with_name(stem + "_redone.json")
        found.append(repaired if repaired.exists() else path)
    return found[-1] if found else None


def probes(sandbox: str, record: dict) -> list[str]:
    """The executable probes of one explore round, in submission order.

    AlienCode submits one program per tool call. AlienLogic submits a batch of
    proofs per round and records each with its verifier verdict, so the proof
    text is read back out of those diagnostics.
    """

    if sandbox == "code":
        calls = record.get("tool_calls") or []
        codes = [str(call.get("code") or "").strip() for call in calls]
    else:
        entries = record.get("diagnostics") or []
        codes = [str(entry.get("proof") or "").strip() for entry in entries]
    return [code for code in codes if code]


def build(spec: dict) -> dict | None:
    path = find_result(spec["sandbox"], spec["model_short"], spec["run_id"])
    if path is None:
        print(f"  !  no result for {spec['sandbox']}/{spec['run_id']}")
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    config = data.get("config") or {}
    if config.get("control_mode") not in (None, "self"):
        print(f"  !  {spec['run_id']} is not a self run; skipped")
        return None

    entries = []
    for record in data.get("explore_records") or []:
        # AlienCode names them loop_idx/round_in_loop, AlienLogic loop/round.
        loop = record.get("loop_idx", record.get("loop"))
        index = record.get("round_in_loop", record.get("round"))
        if loop is None or index is None:
            continue
        if int(loop) > MAX_LOOPS:
            continue
        codes = probes(spec["sandbox"], record)
        if not codes:
            print(f"  !  {spec['run_id']} {loop}-{index} has no probe; skipped")
            return None
        entries.append({
            # AlienCode labels rounds `Explore 1-1`, AlienLogic `Explore 1.1`.
            "label": (
                f"Explore {loop}-{index}" if spec["sandbox"] == "code"
                else f"Explore {loop}.{index}"
            ),
            "model": spec["model_short"],
            "codes": codes,
            "response": (
                record.get("response")
                or record.get("model_update")
                or record.get("code")
                or ""
            ),
        })
    if not entries:
        print(f"  !  {spec['run_id']} has no explore rounds")
        return None

    loops = len({e["label"].split()[1].replace(".", "-").split("-")[0]
                 for e in entries})
    per_loop = len(entries) // max(1, loops)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / (
        f"donor_alien_{spec['sandbox']}_{spec['model_short']}_"
        f"{spec['run_id']}.jsonl"
    )
    body = "".join(
        json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries
    )
    holders = [] if _FORCE else replays_pinned_to(out)
    if holders and hashlib.sha256(body.encode()).hexdigest() != sha256_file(out):
        # A passive or undirected arm replays this exact bank and records its
        # digest. Rewriting it silently re-points those runs at probes they
        # never saw, and the delivery audit then compares them against a bank
        # that did not exist when they ran.
        print(f"  !  {out.name} 已被 {len(holders)} 个回放臂引用，内容有变，"
              f"跳过（用 --force 覆盖，但那些臂必须重跑）")
        for name in holders[:3]:
            print(f"       {name}")
        return None
    out.write_text(body, encoding="utf-8")

    print(f"  wrote {out.relative_to(ROOT)}  "
          f"({len(entries)} rounds, {loops} loops x {per_loop})")
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "transcript": os.path.relpath(out, MANIFEST.parent),
        "sha256": sha256_file(out),
        "model_short": spec["model_short"],
        "sandbox": "AlienCode" if spec["sandbox"] == "code" else "AlienLogic",
        "episode": spec["episode"],
        "explore_loops": loops,
        "rounds_per_loop": per_loop,
        "source_result": os.path.relpath(path, ROOT),
        "protocol": "native_tool",
        "manual_sha256": manual_digest(spec["sandbox"]),
        "environment_sha256": sha256_file(environment_file(spec["sandbox"])),
    }


def replays_pinned_to(bank: Path) -> list[str]:
    """Control results whose recorded bank digest matches this file today."""

    if not bank.is_file():
        return []
    digest = sha256_file(bank)
    found = []
    for sandbox in ("code", "logic"):
        folder = ROOT / "logs" / sandbox / "results"
        for path in folder.glob("eval_results_alien_*_ctrl_*.json"):
            try:
                config = json.loads(path.read_text(encoding="utf-8")).get(
                    "config") or {}
            except (OSError, ValueError):
                continue
            preflight = config.get("control_preflight") or {}
            if preflight.get("transcript_sha256") == digest:
                found.append(path.name)
    return sorted(found)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true",
                        help="show the donors that would be built")
    parser.add_argument(
        "--force", action="store_true",
        help="rewrite banks that replay arms are pinned to; every pinned arm "
             "has to be re-run afterwards",
    )
    args = parser.parse_args()
    global _FORCE
    _FORCE = args.force

    if args.list:
        for spec in DONORS:
            path = find_result(
                spec["sandbox"], spec["model_short"], spec["run_id"])
            print(f"  {spec['sandbox']:5s} {spec['model_short']:16s} "
                  f"{spec['run_id']:16s} -> "
                  f"{path.name if path else 'MISSING'}")
        return 0

    built = [entry for entry in (build(spec) for spec in DONORS) if entry]
    if not built:
        print("no donors built")
        return 1

    episodes = {
        entry["sandbox"]: entry["episode"] for entry in built
    }
    manifest = {
        "schema_version": 2,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "notes": (
            "Native-tool donors. Probes are replayed through the same tool "
            "path the treatment uses, so passive and self differ only in "
            "which probe runs."
        ),
        "donors": built,
        "evaluator_stamps": {
            "AlienCode": "AlienCode-self",
            "AlienLogic": (
                f"AlienLogic-{episodes.get('AlienLogic', 'demo_seeded')}-self"
            ),
        },
        "rule_digests": {
            "AlienCode": sha256_file(rule_file("code")),
            "AlienLogic": sha256_file(rule_file("logic")),
        },
    }
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {MANIFEST.relative_to(ROOT)} with {len(built)} donors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
