"""
AlienLogic evaluation harness — proof-script production.

Protocol (mirrors the MVP protocol in shape, replaces 1-bit prediction with proof artifacts):

  1. Pre-baseline: model submits an initial structured guess of the alien
     rule set, with no observations.
  2. Seed phase: model writes one proof per seed example. The verifier
     returns a *fully detailed* diagnostic (reason_class, reason_id, detail)
     so the model gets grounded on both proof format and the existence of
     hidden side conditions.
  3. Milestone 0: structured rule report + forked held-out tests.
  4. Explore loops: model issues CHECK_PROOF probes (multiple proofs per
     round). The verifier returns *opaque* diagnostics (reason_class,
     reason_id only — detail hidden) to force differential probing.
  5. After each explore loop: another milestone with rule report + tests.

Primary metric: TheoremPassRate per milestone, plus evolution curve.
Secondary metrics: ProofMinimality, AlienAwareness vs classical baseline,
ProbeEfficiency (PassRate gain per probe).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import re
import shutil
import sys
import threading
import time
from datetime import datetime
from typing import Any, Callable

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.agent_client import (
    AgentClient,
    AgentClientConfig,
    AgentSession,
    JsonlTraceStore,
    SessionState,
    ToolResult,
)
from common.agent_client.routing import (
    STANDARD_MESSAGES_PREFIX,
    gateway_a_vendor,
    chat_route_from_model,
    legacy_route,
    responses_route_from_model,
)
from common.agent_runtime import BudgetProfile, PhaseContext
from common import local_config
from common.run_artifacts import export_run_artifacts
from common.run_lock import acquire_run_lock
from common.run_validity import derive as derive_validity
from frameworks.registry import build_runtime
from sandboxes.logic import protocol_v2
from sandboxes.logic.engine import (
    REFERENCE_MANUAL,
    STANDARD_RULES,
    ALIEN_RULE_REGISTRY,
    proof_diagnostic,
    parse_proof,
    verify_proof,
)
from sandboxes.logic.legacy_episodes import (
    EPISODES,
    EpisodeSpec,
    HeldoutTheorem,
    SeedExample,
    get_episode,
    validate_episode,
)


# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════

# Credentials, endpoints and transport knobs live in eval.local.toml. Loading
# is idempotent and never overwrites an exported variable, so importing this
# harness directly works the same as going through run_eval.py.
_LOCAL_SETTINGS = local_config.load()

MODEL = os.environ.get("ALIENLOGIC_MODEL", "")
MODEL_SHORT = os.environ.get("ALIENLOGIC_MODEL_SHORT", "")
RUN_ID = os.environ.get("ALIENLOGIC_RUN_ID", "")
EPISODE_ID = os.environ.get("ALIENLOGIC_EPISODE", "no_explosion_to_compounds")
CONTROL_MODE = os.environ.get("EVAL_CONTROL_MODE", "self").lower()
CONTROL_BANK_PATH = os.environ.get("EVAL_CONTROL_BANK", "")
CONTROL_MANIFEST_PATH = os.environ.get("EVAL_CONTROL_MANIFEST", "")
_CONTROL_BANK_CACHE: dict[str, dict] | None = None
_CONTROL_PREFLIGHT: dict | None = None
_VALID_CONTROL_MODES = {"self", "none", "think", "passive", "random"}

# Provider-native audit trace. The path is unique per run (timestamped) and is
# set in main(); run_eval also supplies a fallback for direct programmatic use.
_TRANSCRIPT_PATH: str | None = None
_AGENT_CLIENT: AgentClient | None = None
_AGENT_RUNTIME = None
FRAMEWORK = os.environ.get(
    "EVAL_FRAMEWORK", str(_LOCAL_SETTINGS.get("framework", "baseline")))
EVAL_TRACK = os.environ.get(
    "EVAL_TRACK", str(_LOCAL_SETTINGS.get("track", "controlled")))
BUDGET_PROFILE = os.environ.get(
    "EVAL_BUDGET_PROFILE", str(_LOCAL_SETTINGS.get("budget_profile", "c1")))

N_EXPLORE_LOOPS = 4
N_PROBES_PER_LOOP = 1
#: Under v2 these widen to match the code sandbox, which allows twelve tool
#: calls a block. The two sandboxes are aligned on how many independent
#: experiments a block permits rather than on what one experiment costs: a
#: probe here is a whole proof and buys far more than one observed value
#: does there, and flattening that difference would misrepresent both.
PROTOCOL_V2 = os.environ.get(
    "ALIENLOGIC_PROTOCOL_V2", "0").lower() in ("1", "true", "yes")
MAX_PROBES_PER_ROUND = 12 if PROTOCOL_V2 else 4
MAX_TOTAL_PROBES = 48 if PROTOCOL_V2 else 24
# All four budgets are env-overridable so ultra-verbose reasoning models
# (e.g. A20B-High spends ~50K reasoning tokens per logic proof) can be given a
# large enough cap to avoid thinking-truncated empty content, without changing
# the defaults for normal models.
DEFAULT_MAX_TOKENS = int(os.environ.get("EVAL_DEFAULT_MAX_TOKENS", "12000"))
SUMMARY_MAX_TOKENS = int(os.environ.get("EVAL_SUMMARY_MAX_TOKENS", "16000"))
# Held-out proof budget. Claude adaptive/high thinking counts thinking tokens
# against this output cap; at late milestones (long context) a small cap gets
# fully consumed by thinking → empty content. Keep generous so thinking + the
# proof both fit; the retry uses an even larger cap for the stragglers.
TEST_MAX_TOKENS = int(os.environ.get("EVAL_TEST_MAX_TOKENS", "16000"))
TEST_RETRY_MAX_TOKENS = int(os.environ.get("EVAL_TEST_RETRY_MAX_TOKENS", "24000"))
LOG_RESPONSE_CHARS = 4000

# Design A (deferred pooled testing). When on, each milestone snapshots the
# exact base context that held-out tests would fork from, but does NOT run the
# tests inline; exploration proceeds immediately. After all explore loops
# finish, every milestone's held-out batch is executed in ONE merged
# ThreadPoolExecutor so concurrency is maximized and the per-milestone "drain"
# barriers disappear. This is score-equivalent to the inline path because both
# the summary and held-out bases fork from the same live AgentSession and never
# mutate it. Opt-in via env or --defer-tests.
DEFER_HELDOUT = os.environ.get("ALIENLOGIC_DEFER_TESTS", "0") == "1"

#: Providers that keep the reasoning state on their side, behind a response
#: id that expires. Deferring their held-out pool to the end of the run means
#: every question is asked from a milestone base the upstream has since
#: forgotten: gpt-5.6 lost 175 of 280 that way, all of them to
#: "Previous response with id ... not found" rather than to anything about
#: the proofs. These score each milestone as they reach it instead.
#: The same list the code sandbox keeps, for the same reason.
_STREAMING_MARKERS = (
    "gpt-5", "gpt5", "gemini", "opus", "claude",
    "deepseek-v4-pro", "deepseek-flash", "deepseek-v4-flash",
)


def _streams_heldout(model: str, short: str) -> bool:
    name = f"{model} {short}".lower()
    return any(marker in name for marker in _STREAMING_MARKERS)

# The pre-exploration baseline costs a full held-out batch -- one sixth of a
# run -- and every primary scoring path already drops it: the paper compares
# M0--M4, and only SeedGain (M0 minus Pre) and the trajectory display read it.
# Skipping it is therefore a cost decision, not a scoring one.
#: The pre-baseline asks for a rule guess before any evidence at all, which
#: the code sandbox has no counterpart to: there a run starts at M0, after
#: the fixed demos. v2 drops it so the two milestone ladders are the same
#: length and M0 means the same thing in both.
SKIP_PRE = (PROTOCOL_V2
            or os.environ.get("EVAL_SKIP_PRE", "0").lower()
            in ("1", "true", "yes"))

# Resume from the last completed phase when a checkpoint for this run id
# exists. Off by default so an intentional re-run of an id starts clean; the
# supervisor and the queue scripts pass --resume.
RESUME = os.environ.get("ALIENLOGIC_RESUME", "0") == "1"

# Tool mode. Explore rounds hand the verifier over as a callable tool, so the
# model decides when to probe and how many probes to spend, and sees each
# diagnostic before choosing the next one -- instead of emitting a fixed batch
# of proof blocks and reading the feedback a round later. Only exploration
# opens it: seed rounds measure a blind ACCEPT/REJECT prediction and the
# milestone phases are closed-book, so neither may call the verifier.
# This is the protocol now, so it is on unless asked otherwise;
# EVAL_TOOL_MODE=0 restores the proof-block protocol of the archived cohorts.
TOOL_MODE = os.environ.get("EVAL_TOOL_MODE", "1").strip().lower() in {
    "1", "true", "yes", "on",
}
PROOF_TOOL_NAME = "check_proof"
# How many turns a model may keep requesting the tool after its budget is
# spent before the round is treated as broken.
_TOOL_LOOP_GRACE = 2

_REPO_ROOT = os.path.dirname(_PROJECT_ROOT)
_LOG_DIR = os.path.join(_REPO_ROOT, "logs", "logic", "run_logs", "live")
_OUT_DIR = os.path.join(_REPO_ROOT, "logs", "logic", "results")
# Traces live beside the code sandbox's, so results/ holds nothing but scores.
_TRACE_DIR = os.path.join(_REPO_ROOT, "logs", "logic", "traces")


# Cohort freeze: seed and held-out answers still use fenced proof blocks, while
# native-tool exploration overrides probe submission in its round prompt and
# executes only check_proof calls. The template also retains a legacy sentence
# about multiple fenced probes. Do not clean that wording until the current
# n=3 cohort is complete; changing it now would create a prompt confound between
# the existing first runs and their replications.
SYSTEM_PROMPT_TEMPLATE = """\
你是 AlienLogicBenchmark 中的自动定理证明者。

环境提供一个看似标准的 Fitch-style 自然演绎证明检查器，但部分规则可能带有
隐藏 side condition（"alien rule"）。每当某条规则在某 step 上触发隐藏约束，
verifier 会返回一个不透明诊断：
    reason_class : 大类（RULE_VIOLATION / ALIEN_SIDE_CONDITION /
                   ALIEN_AXIOM_DISABLED / STRUCTURE / PARSE_ERROR）
    reason_id    : 不透明编码（如 "K3" / "K7"），不直接告诉你规则名

你的目标：
  1. 通过提交"探针证明"（complete PROOF...END blocks）观察拒绝模式，
     反推每条 reason_id 背后的 alien rule。
  2. 在 milestone 时输出结构化 JSON 总结你已发现的 alien rule。
  3. 对每道 held-out theorem 输出一段完整证明，让 verifier 接受。

重要约定：
  - 所有证明必须放在 ```alienlogic_proof``` 代码块中。
  - 一条命令一行；行号从 1 开始递增；`|` 表示进入 assumption box；
    `[<rule> <refs>]` 标注规则及引用。
  - 多条 probe 可并列在多个独立 ```alienlogic_proof``` 块中提交。
  - 不要修改 PROOF/END 包络格式；不要在代码块外用自然语言伪造证明。
  - 如果你认为某 held-out 在当前 alien rule 下不可证，回答行：
        ANSWER: UNPROVABLE
    然后给出一段简短论证（不会被 verifier 跑，仅作记录）。

参考手册（标准证明系统；真实 verifier 可能稳定偏离）:
─────────────────────────────────────────
{manual}
─────────────────────────────────────────
"""


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

_PROOF_BLOCK_RE = re.compile(
    r"```\s*alienlogic_proof\s*\n(.*?)```", re.DOTALL | re.IGNORECASE
)
_GENERIC_BLOCK_RE = re.compile(r"```\s*\n(.*?)```", re.DOTALL)


def extract_proof_blocks(text: str) -> list[str]:
    """Return all proof scripts from the response, in order of appearance."""
    blocks = _PROOF_BLOCK_RE.findall(text)
    if blocks:
        return [b.strip() for b in blocks if b.strip()]
    # Fallback: any fenced block whose content looks like a proof envelope.
    fallback = []
    for raw in _GENERIC_BLOCK_RE.findall(text):
        s = raw.strip()
        if "PROOF" in s.upper() and "END" in s.upper():
            fallback.append(s)
    return fallback


def extract_first_proof(text: str) -> str:
    blocks = extract_proof_blocks(text)
    return blocks[0] if blocks else ""


def extract_unprovable(text: str) -> bool:
    return bool(re.search(r"^\s*ANSWER\s*[：:]\s*UNPROVABLE\s*$",
                          text, re.IGNORECASE | re.MULTILINE))


def extract_json_object(text: str) -> dict[str, Any] | None:
    m = re.search(r"```json\s*(.*?)```", text, flags=re.DOTALL)
    payload = m.group(1) if m else text
    start = payload.find("{")
    end = payload.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        return json.loads(payload[start:end + 1])
    except json.JSONDecodeError:
        return None


def _trim(text: str, n: int = LOG_RESPONSE_CHARS) -> str:
    return text if len(text) <= n else text[:n] + f"\n...[truncated {len(text) - n} chars]"


def _requested_effort() -> str:
    """The reasoning tier this route will actually ask for.

    EVAL_REASONING_EFFORT alone does not say: several routes default to a tier
    of their own when it is unset, and a couple pin the top tier through a
    thinking budget without ever reading it. Recording the bare variable has
    already made runs at the top tier look like they were something else, so
    read the answer back out of the request.

    'unset' means the request carries no reasoning knob at all, which is not
    the same as asking for none: a route with no off switch drops that request
    and thinks anyway. The asked-for value is recorded separately.
    """

    kwargs = _chat_kwargs(DEFAULT_MAX_TOKENS)
    effort = kwargs.get("reasoning_effort")
    if effort:
        return str(effort)
    nested = (kwargs.get("extra_body") or {}).get("output_config") or {}
    if nested.get("effort"):
        return str(nested["effort"])
    thinking = kwargs.get("thinking") or {}
    if thinking.get("type") == "disabled":
        return "none"
    if thinking.get("budget_tokens"):
        return f"budget:{thinking['budget_tokens']}"
    if thinking.get("type"):
        return str(thinking["type"])
    return "unset"


def _chat_kwargs(max_tokens: int) -> dict[str, Any]:
    model_lower = MODEL.lower()
    kwargs: dict[str, Any] = {"max_tokens": max_tokens}
    # Unified non-thinking switch for routes that expose one. Claude omits
    # thinking; Qwen/Doubao use native off controls; the reasoning_effort family
    # passes "none" through. Kimi has no explicit off switch on this route.
    _nothink = os.environ.get("EVAL_REASONING_EFFORT", "").lower() == "none"
    if any(k in model_lower for k in ["claude", "anthropic"]):
        # Claude Opus 4.7+ on Bedrock dropped `thinking.type=enabled`+budget_tokens.
        # New API: `thinking.type=adaptive` + `extra_body.output_config.effort`.
        adaptive_opus = any(tag in model_lower for tag in (
            "opus-4-7", "opus-4.7",
            "opus-4-8", "opus-4.8",
            "opus-5", "opus.5",
        ))
        if _nothink:
            if "opus-5" in model_lower or "opus.5" in model_lower:
                kwargs["thinking"] = {"type": "disabled"}
        elif adaptive_opus:
            kwargs["thinking"] = {
                "type": "adaptive",
                "display": "summarized",
            }
            _opus_effort = os.environ.get("EVAL_REASONING_EFFORT", "high").lower()
            if _opus_effort not in ("low", "medium", "high", "xhigh", "max"):
                _opus_effort = "high"
            kwargs["extra_body"] = {"output_config": {"effort": _opus_effort}}
        else:
            thinking_budget = min(4000, max(1024, max_tokens // 2))
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
    elif MODEL.startswith(STANDARD_MESSAGES_PREFIX):
        # A non-Anthropic vendor speaking Anthropic Messages. DeepSeek has no
        # budget_tokens of its own; the adapter quantises the number onto its
        # reasoning tiers, and 60000 is the top one.
        kwargs["thinking"] = (
            {"type": "disabled"} if _nothink
            else {"type": "enabled", "budget_tokens": 60000}
        )
    elif "gemini" in model_lower:
        # Gemini 3.x takes named thinking levels only, so the 'max' every other
        # vendor accepts is a 400, and temperature/topP are deprecated fields
        # the vendor warns will start failing. Send neither.
        _level = os.environ.get("EVAL_REASONING_EFFORT", "high").lower()
        if _level not in ("minimal", "low", "medium", "high"):
            _level = "high"
        kwargs["thinking"] = {
            "thinkingLevel": _level,
            "includeThoughts": True,
        }
        kwargs["max_tokens"] = min(max_tokens, 65536)
    elif responses_route_from_model(MODEL) == "doubao":
        # Four tiers topping out at 'high'; thinking is a separate switch the
        # tier may only ride along with when it is on. Its prompt cache is
        # ignored entirely when instructions are set, so the system prompt
        # travels as the leading input item instead.
        _level = os.environ.get("EVAL_REASONING_EFFORT", "high").lower()
        if _level not in ("minimal", "low", "medium", "high"):
            _level = "high"
        if _nothink:
            kwargs["thinking"] = {"type": "disabled"}
        else:
            kwargs["thinking"] = {"type": "enabled"}
            kwargs["reasoning_effort"] = _level
        kwargs["temperature"] = 1.0
        kwargs["system_in_input"] = True
    elif responses_route_from_model(MODEL):
        # The GatewayA Responses upstreams name the tier by effort, up to 'max'.
        kwargs["temperature"] = 1.0
        kwargs["reasoning_effort"] = os.environ.get(
            "EVAL_REASONING_EFFORT", "max").lower()
        kwargs["max_tokens"] = min(max_tokens, 131072)
    elif gateway_a_vendor(MODEL) == "ali":
        # Matched on the route rather than on the word 'qwen' in the id, and
        # asked before the keyword branch below, which would otherwise answer
        # first and leave this route on the vendor's default tier.
        # reasoning_effort and thinking_budget are a 400 together, and
        # max_completion_tokens has to be strictly greater than any budget
        # sent, so the tier is named and the budget left implicit. This
        # endpoint has three tiers, not seven: max and high both land on xhigh.
        # Thinking returns as reasoning_content and is replayed verbatim,
        # which is what preserve_thinking expects.
        _level = os.environ.get("EVAL_REASONING_EFFORT", "max").lower()
        if _level not in (
            "none", "minimal", "low", "medium", "high", "xhigh", "max"
        ):
            _level = "max"
        if _level == "none" or _nothink:
            kwargs["enable_thinking"] = False
        else:
            kwargs["enable_thinking"] = True
            kwargs["reasoning_effort"] = _level
        kwargs["temperature"] = 1.0
        kwargs.pop("max_tokens", None)
        kwargs["max_completion_tokens"] = min(
            max(max_tokens, 65536), 131072
        )
    elif "qwen" in model_lower:
        # Legacy chat route: qwen toggles thinking via enable_thinking there,
        # not reasoning_effort.
        kwargs["temperature"] = 1.0
        kwargs["extra_body"] = {"enable_thinking": not _nothink}
    elif "doubao" in model_lower:
        # Legacy chat route: doubao rejects reasoning_effort=none there; its
        # native off switch is thinking.disabled.
        kwargs["temperature"] = 1.0
        if _nothink:
            kwargs["thinking"] = {"type": "disabled"}
        else:
            kwargs["reasoning_effort"] = os.environ.get("EVAL_REASONING_EFFORT", "high")
    elif chat_route_from_model(MODEL) == "xai":
        # Three tiers topping out at 'high'; 'max' is not in the enum and
        # 'none' is a 400 because reasoning cannot be turned off. The budget
        # field is max_completion_tokens, which the reasoning does not spend.
        _level = os.environ.get("EVAL_REASONING_EFFORT", "high").lower()
        if _level not in ("low", "medium", "high"):
            _level = "high"
        kwargs["reasoning_effort"] = _level
        kwargs["temperature"] = 1.0
        kwargs.pop("max_tokens", None)
        kwargs["max_completion_tokens"] = max_tokens
        kwargs["prompt_cache_key"] = f"alienlogic-{RUN_ID}"
    elif chat_route_from_model(MODEL) == "moonshot":
        # Thinking is always on and the tier tops out at 'max'. The sampling
        # knobs are fixed server-side and rejected if sent, and max_tokens is
        # retired in favour of max_completion_tokens -- one budget covering the
        # reasoning and the answer.
        _level = os.environ.get("EVAL_REASONING_EFFORT", "max").lower()
        if _level not in ("none", "low", "medium", "high", "max"):
            _level = "max"
        kwargs["reasoning_effort"] = _level
        kwargs.pop("max_tokens", None)
        kwargs["max_completion_tokens"] = min(max_tokens, 131072)
    elif legacy_route(MODEL) == "gateway_a_standard":
        # DeepSeek on GatewayA's standard endpoint. Seven named tiers up to 'max'
        # collapsed onto the vendor's own three, 'none' turning thinking off.
        # The thinking rides back in message.reasoning_content and only stays
        # in scope because the chat adapter replays that message verbatim.
        _level = os.environ.get("EVAL_REASONING_EFFORT", "max").lower()
        if _level not in (
            "none", "minimal", "low", "medium", "high", "xhigh", "max"
        ):
            _level = "max"
        kwargs["reasoning_effort"] = _level
        kwargs["temperature"] = 1.0
        kwargs.pop("max_tokens", None)
        # The thinking spends this budget too, and at 'max' effort it can run
        # past 32768 on its own, ending the turn at finish_reason=length with
        # no answer written. A floor rather than the phase budget keeps a long
        # proof attempt from being cut off mid-thought.
        kwargs["max_completion_tokens"] = min(
            max(max_tokens, 131072), 393216
        )
    elif "kimi" in model_lower:
        # Kimi exposes built-in reasoning on this route, but no explicit effort
        # or non-thinking control. Match the historical Kimi K3 launch config.
        kwargs["temperature"] = 1.0
    elif any(k in model_lower for k in
             ["reasoner", "thinking", "gpt-5", "o1", "o3", "hy3", "hy4", "opd",
              "a20b", "deepseek", "minimax", "gemini", "grok"]) or MODEL in [
                  m for m in os.environ.get("HY_GATEWAY_C_MODELS", "").split(",") if m]:
        # Reasoning models on the gateway proxy: explicitly request high effort.
        # Verified accepted (no 400) for doubao-seed-2.0 / deepseek-v4-pro /
        # MiniMax-M2.5 — all are reasoning-on by default and honor this knob.
        # EVAL_REASONING_EFFORT env overrides the default (e.g. "max" for the
        # deepseek-v4-pro (max) ablation entry); leave unset to keep "high".
        kwargs["temperature"] = 1.0
        kwargs["reasoning_effort"] = os.environ.get("EVAL_REASONING_EFFORT", "high")
    else:
        kwargs["temperature"] = 0.0
    return kwargs


#: What fraction of the deadline a failed call has to have consumed before it
#: counts as the model's time rather than the provider's. Short of this the
#: gateway dropped the request; at or past it the model was still thinking.
#: Matched to the code sandbox so a timeout means the same thing in both.
_DEADLINE_CREDIT = 0.9


def _request_deadline() -> float:
    """The per-request budget this run was launched with, in seconds."""
    try:
        return max(1.0, float(os.environ.get(
            "EVAL_HTTP_TIMEOUT", _LOCAL_SETTINGS.get("timeout", 600))))
    except ValueError:
        return 600.0


#: Whether the call this worker just made ended by running out of clock.
#: Held-out theorems are graded in a thread pool, so the flag has to be per
#: worker; a module global would let one theorem's timeout mark another's
#: answer.
_budget = threading.local()


def _clear_deadline_flag() -> None:
    _budget.timed_out = False


def _deadline_was_spent() -> bool:
    return bool(getattr(_budget, "timed_out", False))


def _chat(
    session: AgentSession,
    prompt: str,
    *,
    label: str = "",
    max_tokens: int | None = None,
    extra_overrides: dict[str, Any] | None = None,
) -> str:
    # run_eval.py overrides the module-level budget after importing this
    # harness. Resolve it at call time rather than freezing the import-time
    # default in the function signature.
    if max_tokens is None:
        max_tokens = DEFAULT_MAX_TOKENS
    history = session.history
    turns = (
        int(session.system is not None)
        + sum(1 for event in history if event.get("role") in {"user", "assistant"})
        + 1
    )
    chars = (
        len(str(session.system or ""))
        + sum(
            len(str(event.get("payload", {}).get("content", "")))
            for event in history
        )
        + len(prompt)
    )
    print(f"\n[Chat -> {MODEL_SHORT or MODEL} ({label})] "
          f"turns={turns} chars={chars} max_tokens={max_tokens}")
    # A failed call must not abort the batch, but it must not silently become a
    # wrong answer either. Returning "" for everything filed a gateway 500 as
    # NO_PROOF_BLOCK -- indistinguishable from a model that had nothing to say,
    # and invisible to the repair pass that exists precisely for provider
    # failures. So the clock decides, the same way the code sandbox decides it:
    # a call that consumed the thinking budget is the model's answer, wrong;
    # anything that failed faster is the provider's, and is raised so the
    # worker wrapper can file it as WORKER_ERROR and the repair sweep can
    # re-ask it.
    started = time.time()
    _clear_deadline_flag()
    try:
        response = session.send_user(
            prompt,
            label=label,
            request_overrides={
                **_chat_kwargs(max_tokens),
                **(extra_overrides or {}),
            },
        )
    except Exception as exc:
        elapsed = time.time() - started
        spent_budget = elapsed >= _DEADLINE_CREDIT * _request_deadline()
        verdict = "spent its budget" if spent_budget else "provider failure"
        print(f"  [Chat Error] {type(exc).__name__} after {elapsed:.0f}s "
              f"({verdict}): {str(exc)[:200]}")
        if spent_budget:
            # An empty string on its own is the one thing a model that simply
            # had nothing to say also produces, and the grader files both as
            # NO_PROOF_BLOCK. Flag it so the record can say which happened.
            _budget.timed_out = True
            return ""
        raise
    return response.text or ""


def _chat_with_empty_retry(
    session: AgentSession,
    prompt: str,
    *,
    label: str = "",
    max_tokens: int | None = None,
    retry_prompt: str | None = None,
    retry_max_tokens: int | None = None,
    answered: Callable[[str], bool] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Ask once, and ask again if nothing usable came back.

    `answered` decides what usable means for the caller. Held-out theorems pass
    one, because a graded turn can come back full of text and still hold no
    answer: the model, forked from a history where probing was allowed, says it
    will verify with a probe first and ends the turn. There is no tool on this
    fork, so nothing is coming -- the re-ask says so and asks for the answer now.

    A `retry_prompt` has to stand on its own. The failed turn is rolled off the
    history before the re-ask, which takes the question with it, so a prompt that
    only refers back to it ("give the proof for T07") asks about something the
    model can no longer see.
    """

    if max_tokens is None:
        max_tokens = DEFAULT_MAX_TOKENS
    if answered is None:
        answered = lambda text: bool(text.strip())  # noqa: E731
    before = session.state()
    previous_response = session.last_response
    response = _chat(
        session, prompt, label=label, max_tokens=max_tokens,
    )
    first_response = session.last_response
    if first_response is previous_response:
        first_response = None
    first_call_meta = session.last_call_meta if first_response is not None else {}
    meta = {"empty_initial": not response.strip(),
            "unanswered_initial": not answered(response),
            "retry_attempted": False, "retry_success": False,
            "transport_retry_count": int(
                first_call_meta.get("transport_retry_count", 0) or 0),
            "effective_max_tokens": first_call_meta.get(
                "effective_max_tokens")}
    if answered(response):
        return response, meta
    meta["retry_attempted"] = True
    session.restore_in_place(
        before,
        preserve_usage=first_response.usage if first_response is not None else None,
    )
    # A graded turn offers no tool, but the history it forked from is full of
    # them, and some models answer out of that habit: a real call, no text.
    # `restore_in_place` above already dropped it, so the re-ask only has to
    # say why no result is coming.
    called_a_tool = bool(
        first_response is not None and first_response.tool_calls
    )
    if called_a_tool:
        names = ", ".join(
            sorted({call.name for call in first_response.tool_calls})
        )
        print(f"  [Empty Retry] {label} (模型调用了闭卷不可用的 {names})")
        # Prepended rather than substituted so the milestone feedback the
        # caller baked into the retry prompt still travels.
        retry_prompt = (
            f"⚠️ 本轮为闭卷作答，{names} 不可用，你的调用不会被执行，"
            f"也不会有任何返回。请直接给出最终答案。\n\n"
            + (retry_prompt if retry_prompt is not None else prompt)
        )
    elif response.strip():
        # Text came back, just no answer in it -- the model announced it would
        # probe first and stopped. Same dead end as an actual call, so say the
        # same thing: nothing is coming, answer now.
        print(f"  [Empty Retry] {label} (模型想先用探针，闭卷阶段没有工具)")
        retry_prompt = (
            "⚠️ 本轮为闭卷作答，验证器/探针不可用，你无法在本轮取得任何返回。"
            "请仅凭已掌握的规则直接给出最终答案。\n\n"
            + (retry_prompt if retry_prompt is not None else prompt)
        )
    else:
        print(f"  [Empty Retry] {label}")
        retry_prompt = (
            "⚠️ 上一轮返回为空（可能是思考占满了输出预算）。"
            "请把预算留给答案本身。\n\n"
            + (retry_prompt if retry_prompt is not None else prompt)
        )
    previous_retry_response = session.last_response
    response = _chat(
        session,
        retry_prompt if retry_prompt is not None else prompt,
        label=f"{label} retry",
        max_tokens=retry_max_tokens or max_tokens,
    )
    retry_response = session.last_response
    retry_call_meta = (
        session.last_call_meta
        if retry_response is not previous_retry_response
        else {}
    )
    meta["transport_retry_count"] += int(
        retry_call_meta.get("transport_retry_count", 0) or 0)
    meta["effective_max_tokens"] = retry_call_meta.get(
        "effective_max_tokens")
    meta["retry_success"] = answered(response)
    return response, meta


def _heldout_retry_prompt(theorem_id: str, prompt: str) -> str:
    """The question asked over again, not referred back to.

    The turn being replaced was rolled off the history and took the theorem with
    it, so a re-ask that only names the theorem asks about something the model
    can no longer see. It says so and the answer is ungradeable a second time,
    which is how one milestone lost 62 of 85 questions.
    """

    return (
        f"请直接在 ```alienlogic_proof``` 代码块中给出 {theorem_id} 的完整证明；"
        f"若确实不可证，只回一行 ANSWER: UNPROVABLE。题目重述如下。\n\n"
        + prompt
    )


def _format_diagnostic(diag: dict[str, Any], hide_detail: bool = True) -> str:
    """Render a diagnostic dict as concise feedback for the model."""
    if hide_detail:
        keys = ("accepted", "verifier_passed", "goal_match",
                "rejected_step", "rejected_rule", "reason_class", "reason_id",
                "n_steps")
    else:
        keys = ("accepted", "verifier_passed", "goal_match",
                "rejected_step", "rejected_rule", "reason_class", "reason_id",
                "detail", "n_steps")
    return json.dumps({k: diag.get(k) for k in keys}, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════
#  Phase: pre-baseline
# ═══════════════════════════════════════════════════════════════════════

def run_pre_baseline(state: dict[str, Any], episode: EpisodeSpec,
                     records: list[dict[str, Any]],
                     deferred: list[dict[str, Any]] | None = None
                     ) -> dict[str, Any] | None:
    print("\n" + "#" * 72)
    print("  Milestone PRE")
    print("#" * 72)
    prompt = (
        "在尚未观察任何 verifier 反馈前，仅基于参考手册输出一个非常初步的"
        "alien-rule 假设 JSON。\n"
        "字段：rules (list), uncertainties (list), planned_probes (list of strings)。\n"
        "如果没有任何证据猜测，rules 可以为空数组。只输出 JSON。"
    )
    summary_session: AgentSession = _closed_book_fork(state["session"])
    response, meta = _chat_with_empty_retry(
        summary_session, prompt,
        label="M_pre Summary", max_tokens=SUMMARY_MAX_TOKENS,
    )
    summary = extract_json_object(response) or {}
    print(f"\n[M_pre Summary]\n{_trim(response)}")
    base = _heldout_base(state)
    base_snapshot = _milestone_base_snapshot(base, "pre")
    if deferred is not None:
        deferred.append({
            "milestone_id": "pre", "milestone_label": "pre",
            "summary_text": response, "summary": summary, "retry_meta": meta,
            "base": base,
            "pending_feedback": state.get("pending_feedback"),
            "base_snapshot": base_snapshot,
        })
        print("[M_pre] held-out tests deferred to merged pool")
        return None
    test_results = run_heldout_tests(
        state, episode, milestone_label="pre", base=base,
    )
    rec = _build_milestone_record(
        milestone_id="pre", summary_text=response, summary=summary,
        test_results=test_results, retry_meta=meta,
        base_snapshot=base_snapshot,
    )
    records.append(rec)
    _print_milestone_summary("pre", rec)
    return rec


# ═══════════════════════════════════════════════════════════════════════
#  Phase: seed
# ═══════════════════════════════════════════════════════════════════════

#: The v2 held-out set: the original set with the giveaways removed.
#:
#: Selection is by measured M0 rate, not by judgement, with one constraint:
#: every patched rule keeps at least one theorem, even where the cohort
#: already solves it. That is the same bar the code set is held to.
_V2_IDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "heldout_v2_ids.json")


_HARD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hard_theorems.json")


def _load_hard_theorems() -> list:
    """The authored band above the old ceiling, if it has been built.

    Each was checked against this verifier at build time: its reference proof
    accepted and its naive one refused, or both refused where declining is
    the answer. See build_hard_theorems.py.
    """

    if not os.path.exists(_HARD_PATH):
        return []
    with open(_HARD_PATH, encoding="utf-8") as handle:
        rows = json.load(handle)
    # Built against whichever HeldoutTheorem the run is using: the harness
    # binds the legacy one, which has no key_rules, while the authored set
    # carries them for the coverage audit. Pass only the fields the bound
    # class declares and set the rest afterwards.
    known = {field.name for field in dataclasses.fields(HeldoutTheorem)}
    built = []
    for row in rows:
        fields = {
            "id": row["id"],
            "cluster_id": f"hard_{row['id']}",
            "role": row["role"],
            "premises": list(row["premises"]),
            "goal": row["goal"],
            "alien_provable": bool(row["provable"]),
            "min_alien_steps": int(row.get("min_alien_steps") or 0),
            "alien_aware_required": True,
            "reference_alien_proof": (row["reference"] if row["provable"]
                                      else None),
            "unprovable_naive_proof": (None if row["provable"]
                                       else row["naive"]),
            "key_rules": tuple(row.get("key_rules") or ()),
        }
        theorem = HeldoutTheorem(
            **{k: v for k, v in fields.items() if k in known})
        if "key_rules" not in known:
            object.__setattr__(theorem, "key_rules", fields["key_rules"])
        built.append(theorem)
    return built


def _apply_v2_task_set(episode: EpisodeSpec) -> EpisodeSpec:
    """Narrow the held-out set under v2; leave it alone otherwise."""

    if not PROTOCOL_V2 or not os.path.exists(_V2_IDS_PATH):
        return episode
    with open(_V2_IDS_PATH, encoding="utf-8") as handle:
        keep = set(json.load(handle))
    kept = [t for t in episode.heldout_theorems if t.id in keep]
    if not kept:
        return episode
    missing = keep - {t.id for t in kept}
    if missing:
        raise RuntimeError(
            f"v2 held-out ids not present in episode: {sorted(missing)}")
    hard = _load_hard_theorems()
    clash = {t.id for t in kept} & {t.id for t in hard}
    if clash:
        raise RuntimeError(f"hard theorem ids collide with kept: {clash}")
    kept = kept + hard
    print(f"[Task set] v2: {len(kept)} held-out theorems "
          f"({len(kept) - len(hard)} kept from the original set, "
          f"{len(hard)} authored)")
    return dataclasses.replace(episode, heldout_theorems=kept)


def _demo_checker(episode: EpisodeSpec):
    """A verdict function bound to this episode's patched rules.

    Read from the same checker the held-out phase uses rather than from a
    verdict written beside the example, so the block cannot drift away from
    the engine that scores the run.
    """

    def verdict(proof_text: str) -> dict[str, Any]:
        try:
            return proof_diagnostic(proof_text, episode.alien_rules,
                                    hide_reason_detail=True)
        except Exception:                                    # noqa: BLE001
            return {"accepted": False, "reason_class": "PARSE_ERROR",
                    "reason_id": "DEMO_UNCHECKED"}

    return verdict


def _run_fixed_demos(state: dict[str, Any], episode: EpisodeSpec,
                     records: list[dict[str, Any]]) -> None:
    """Record the fixed evidence. The model already has it in its system."""

    rows = protocol_v2.demo_rows(_demo_checker(episode),
                                 episode.seed_examples)
    print("\n" + "━" * 72)
    print(f"  Phase: fixed calibration demos (AlienLogic v2) — "
          f"{len(rows)} demos")
    print("━" * 72)
    for row in rows:
        records.append({
            "phase": "fixed_demo",
            "seed_id": row["id"],
            "description": row["description"],
            "premises": row["premises"],
            "goal": row["goal"],
            "proof_text": row["proof"],
            "accepted": row["accepted"],
            "reason_class": row["reason_class"],
            "reason_id": row["reason_id"],
            "model_prediction": None,
            "time_seconds": 0.0,
            "protocol_version": protocol_v2.PROTOCOL_VERSION,
        })
        verdict = "ACCEPTED" if row["accepted"] else (
            f"REJECTED {row['reason_class']}/{row['reason_id']}")
        print(f"[{row['id']}] {verdict}")
    state["pending_feedback"] = None


def run_seed(state: dict[str, Any], episode: EpisodeSpec,
             records: list[dict[str, Any]]) -> None:
    if PROTOCOL_V2:
        _run_fixed_demos(state, episode, records)
        return
    print("\n" + "━" * 72)
    print("  Seed Phase")
    print("━" * 72)
    hide_detail = bool(getattr(episode, "seed_hide_detail", False))
    feedback = None
    session: AgentSession = state["session"]
    for example in episode.seed_examples:
        t0 = time.time()
        prompt_parts: list[str] = []
        if feedback:
            prompt_parts.append(feedback)
            prompt_parts.append("---")
        prompt_parts.append(f"【Seed {example.id}】{example.description}")
        task_mode = getattr(example, "goal", None) is not None
        if task_mode:
            # Task mode (task_warmup): show only the bare theorem; the model must
            # author its own proof and gets the same opaque feedback as the
            # held-out phase. No proof and no rule is ever shown.
            premise_str = ", ".join(example.premises) if example.premises else "（无）"
            prompt_parts.append(
                "下面是一个待证定理。请你自己写出一段完整证明，让 verifier 接受它。"
                "verifier 只会返回不透明诊断（reason_class + reason_id），"
                "不会解释拒绝的具体原因。"
            )
            prompt_parts.append(f"premises: {premise_str}")
            prompt_parts.append(f"goal: {example.goal}")
            prompt_parts.append(
                "先在你回答的开头说明你预测 verifier 会 ACCEPT 还是 REJECT，"
                "然后给出一个 ```alienlogic_proof``` 代码块（premises 与 goal "
                "必须与上面给定的完全一致）。"
            )
        else:
            prompt_parts.append(
                "请按 PROOF...END 格式提交以下证明的副本（你可以在保留逻辑等价的"
                "前提下重写步骤），verifier 会返回完整诊断。"
            )
            prompt_parts.append("```alienlogic_proof")
            prompt_parts.append(example.proof_text.strip())
            prompt_parts.append("```")
            prompt_parts.append("先在你回答的开头说明你预测 verifier 会 ACCEPT 还是 REJECT，"
                                "然后给出 ```alienlogic_proof``` 代码块（可以是上面那段，"
                                "也可以是你认为更合理的替代版本）。")
        prompt = "\n".join(prompt_parts)
        response, retry_meta = _chat_with_empty_retry(
            session, prompt, label=f"Seed {example.id}",
        )
        if not response.strip():
            response = "（模型返回了空回复）"
        if task_mode:
            proof_text = extract_first_proof(response)
            diag = proof_diagnostic(
                proof_text, episode.alien_rules,
                hide_reason_detail=hide_detail,
                expected_goal=example.goal, expected_premises=example.premises,
            )
        else:
            proof_text = extract_first_proof(response) or example.proof_text
            diag = proof_diagnostic(proof_text, episode.alien_rules,
                                    hide_reason_detail=hide_detail)
        elapsed = time.time() - t0
        records.append({
            "phase": "seed",
            "example_id": example.id,
            "expected_accepted": example.expected_accepted,
            "expected_reason_id": example.expected_reason_id,
            "diagnostic": diag,
            "predicted_label": _extract_predicted_label(response),
            "actual_accepted": bool(diag.get("accepted")),
            "response": response,
            "proof_submitted": proof_text,
            "time_seconds": elapsed,
            **retry_meta,
        })
        feedback = (
            f"上一条 seed (Seed {example.id}) verifier 反馈：\n"
            f"{json.dumps(diag, ensure_ascii=False, indent=2)}"
        )
        print(f"\n[Seed {example.id} Response]\n{_trim(response)}")
        print(f"[Seed {example.id} Diag] {_format_diagnostic(diag, hide_detail=hide_detail)}")
        print(f"[Seed {example.id}] time={elapsed:.1f}s")
    state["pending_feedback"] = feedback


def _extract_predicted_label(text: str) -> str:
    if re.search(r"\bACCEPT(ED)?\b", text, re.IGNORECASE):
        return "ACCEPT"
    if re.search(r"\bREJECT(ED)?\b", text, re.IGNORECASE):
        return "REJECT"
    return "UNKNOWN"


# ═══════════════════════════════════════════════════════════════════════
#  Phase: explore (probe)
# ═══════════════════════════════════════════════════════════════════════

def _is_donor_trajectory(rec: dict[str, Any]) -> bool:
    """Accept legacy transcript rows and provider-native trajectory rows."""
    return rec.get("record_type") in (None, "trajectory")


def _load_control_bank() -> dict[str, dict]:
    """Load standard-run donor responses from a transcript JSONL."""
    global _CONTROL_BANK_CACHE
    if _CONTROL_BANK_CACHE is not None:
        return _CONTROL_BANK_CACHE
    if not CONTROL_BANK_PATH:
        raise RuntimeError(
            "EVAL_CONTROL_MODE=passive requires --control-bank or "
            "EVAL_CONTROL_BANK pointing to a standard-run transcript JSONL"
        )
    bank: dict[str, dict] = {}
    with open(CONTROL_BANK_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict) or not _is_donor_trajectory(rec):
                continue
            label = str(rec.get("label", ""))
            if label.startswith("Explore ") and not label.endswith(" retry"):
                bank[label] = rec
    _CONTROL_BANK_CACHE = bank
    return bank


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preflight_control_bank(episode: EpisodeSpec) -> dict | None:
    """Fail before paid calls if a passive/random donor is not comparable."""
    global _CONTROL_PREFLIGHT
    if CONTROL_MODE not in {"passive", "random"}:
        _CONTROL_PREFLIGHT = None
        return None
    if not CONTROL_MANIFEST_PATH:
        raise RuntimeError(
            "passive/random controls require --control-manifest so donor "
            "model, sandbox, version, and hashes can be verified")

    expected = {
        f"Explore {loop}.{round_idx}"
        for loop in range(1, N_EXPLORE_LOOPS + 1)
        for round_idx in range(1, N_PROBES_PER_LOOP + 1)
    }
    bank = _load_control_bank()
    labels = set(bank)
    if labels != expected:
        raise RuntimeError(
            "Control donor explore structure mismatch: "
            f"missing={sorted(expected - labels)}, "
            f"unexpected={sorted(labels - expected)}")

    retry_labels: list[str] = []
    with open(CONTROL_BANK_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict) or not _is_donor_trajectory(rec):
                continue
            label = str(rec.get("label", ""))
            if label.startswith("Explore ") and label.endswith(" retry"):
                retry_labels.append(label)
    if retry_labels:
        raise RuntimeError(
            f"Control donor contains retry explore calls: {retry_labels}")

    aliases = {MODEL, MODEL_SHORT, MODEL.rsplit("_", 1)[-1]}
    donor_models = {str(rec.get("model", "")) for rec in bank.values()}
    if not donor_models:
        raise RuntimeError("Control donor names no model")
    # Passive replays the donor's proofs, so it has to be this model's own
    # run. Random replays none of them -- it takes the per-round probe
    # count and generates its own -- so one shared profile across the
    # cohort is the point: it makes the arm a single baseline every system
    # meets identically rather than one draw per model. The structural and
    # digest checks below still apply to both.
    if (CONTROL_MODE != "random"
            and any(model not in aliases for model in donor_models)):
        raise RuntimeError(
            f"Control donor model mismatch: donor={sorted(donor_models)}, "
            f"target={sorted(x for x in aliases if x)}")
    empty = [
        label for label, rec in bank.items()
        if not str(rec.get("response", "")).strip()
        and not (rec.get("codes") or [])
    ]
    if empty:
        raise RuntimeError(f"Control donor has empty responses: {empty}")
    no_probe = [
        label for label in expected if _donor_probe_count(label) <= 0
    ]
    if no_probe:
        raise RuntimeError(f"Control donor has no executable probes: {no_probe}")

    transcript_digest = _sha256_file(CONTROL_BANK_PATH)
    manifest_verified = False
    manifest_entry = None
    with open(CONTROL_MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    # The stamp names the episode the donor was recorded under. It used to be
    # pinned to task_warmup, which silently rejected a demo_seeded donor even
    # when every other digest matched.
    evaluator_stamp = f"AlienLogic-{episode.id}-self"
    if manifest.get("evaluator_stamps", {}).get(
            "AlienLogic") != evaluator_stamp:
        raise RuntimeError(
            "Control donor evaluator stamp is missing or incompatible")
    rule_digest = _sha256_file(os.path.join(
        _PROJECT_ROOT, "sandboxes", "logic", "episodes.py"))
    if manifest.get("rule_digests", {}).get("AlienLogic") != rule_digest:
        raise RuntimeError(
            "Control donor hidden-rule digest is missing or incompatible")
    manifest_dir = os.path.dirname(
        os.path.abspath(CONTROL_MANIFEST_PATH))
    target_path = os.path.realpath(CONTROL_BANK_PATH)
    for entry in manifest.get("donors", []):
        entry_path = os.path.realpath(os.path.join(
            manifest_dir, entry["transcript"]))
        if entry_path == target_path:
            manifest_entry = entry
            break
    if manifest_entry is None:
        raise RuntimeError(
            f"Control donor is absent from manifest: {CONTROL_BANK_PATH}")
    checks = {
        "sha256": transcript_digest,
        "model_short": MODEL_SHORT,
        "sandbox": "AlienLogic",
        "episode": episode.id,
        "explore_loops": N_EXPLORE_LOOPS,
        "rounds_per_loop": N_PROBES_PER_LOOP,
        "manual_sha256": hashlib.sha256(
            REFERENCE_MANUAL.encode()).hexdigest(),
        "environment_sha256": _sha256_file(os.path.join(
            _PROJECT_ROOT, "sandboxes", "logic", "engine.py")),
    }
    if CONTROL_MODE == "random":
        # Deliberately another model's donor: random takes its probe volume
        # and generates its own proofs, so one shared profile is what makes
        # the arm a common baseline. The provenance is still recorded in
        # _CONTROL_PREFLIGHT below; it is simply not a mismatch.
        checks.pop("model_short")
    mismatches = {
        key: (manifest_entry.get(key), value)
        for key, value in checks.items()
        if manifest_entry.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Control donor manifest mismatch: {mismatches}")
    manifest_verified = True

    _CONTROL_PREFLIGHT = {
        "status": "verified" if manifest_verified else "unattested",
        "sandbox": "AlienLogic",
        "episode": episode.id,
        "model_short": MODEL_SHORT,
        "explore_loops": N_EXPLORE_LOOPS,
        "rounds_per_loop": N_PROBES_PER_LOOP,
        "transcript_sha256": transcript_digest,
        # Whose run supplied the bank. Equal to model_short for passive and
        # deliberately not for random, which is the one thing a reader of a
        # random arm needs to be able to check.
        "donor_model_short": sorted(donor_models),
        "manifest": CONTROL_MANIFEST_PATH or None,
        "manifest_generated_at": (
            manifest_entry.get("generated_at") if manifest_entry else None
        ),
        "evaluator_stamp": evaluator_stamp,
        "rule_sha256": rule_digest,
    }
    print(f"[Control Preflight] {_CONTROL_PREFLIGHT}")
    return _CONTROL_PREFLIGHT


def _passive_response(label: str) -> str:
    rec = _load_control_bank().get(label)
    if not rec or not str(rec.get("response", "")).strip():
        raise LookupError(
            f"Passive donor transcript has no non-empty response for {label!r}: "
            f"{CONTROL_BANK_PATH}"
        )
    return str(rec["response"])


def _donor_proofs(label: str) -> list[str]:
    """Executable donor proofs, preferring the native-tool argument archive."""

    rec = _load_control_bank().get(label)
    if not rec:
        raise LookupError(
            f"Passive donor transcript has no entry for {label!r}: "
            f"{CONTROL_BANK_PATH}"
        )
    proofs = [
        str(proof).strip()
        for proof in (rec.get("codes") or [])
        if str(proof).strip()
    ]
    if proofs:
        return proofs[:MAX_PROBES_PER_ROUND]
    return extract_proof_blocks(_passive_response(label))[:MAX_PROBES_PER_ROUND]


def _donor_probe_count(label: str) -> int:
    """Actual accepted-budget probe count in the paired self transcript."""
    return len(_donor_proofs(label))


def _random_probe_response(loop_idx: int, round_idx: int,
                           n_probes: int) -> str:
    """Deterministic parse-valid proof probes with no model-dependent design."""
    seed_text = f"AlienLogic:{RUN_ID}:{loop_idx}:{round_idx}"
    seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    atoms = rng.sample(["p", "q", "r", "s", "t", "u"], 4)
    p, q, r, s = atoms
    templates = [
        # Standard-form controls.
        f"""PROOF
premises: {p}, {q}
goal: AND({p}, {q})
1. {p} [premise]
2. {q} [premise]
3. AND({p}, {q}) [AND_I 1, 2]
END""",
        f"""PROOF
premises: IMPL({p}, {q}), {p}
goal: {q}
1. IMPL({p}, {q}) [premise]
2. {p} [premise]
3. {q} [IMPL_E 1, 2]
END""",
        f"""PROOF
premises: {p}, NOT({p})
goal: BOT
1. {p} [premise]
2. NOT({p}) [premise]
3. BOT [NOT_E 1, 2]
END""",
        f"""PROOF
premises: BOT
goal: {q}
1. BOT [premise]
2. {q} [BOT_E 1]
END""",
        # Grammar-valid structural variations sampled independently of the
        # target model's current hypothesis.
        f"""PROOF
premises: {p}, NOT({p})
goal: AND({q}, {r})
1. {p} [premise]
2. NOT({p}) [premise]
3. BOT [NOT_E 1, 2]
4. AND({q}, {r}) [BOT_E 3]
END""",
        f"""PROOF
premises: {p}
goal: OR({p}, AND({q}, {r}))
1. {p} [premise]
2. OR({p}, AND({q}, {r})) [OR_I1 1]
END""",
        f"""PROOF
premises: AND({p}, {q}), NOT(AND({p}, {q}))
goal: BOT
1. AND({p}, {q}) [premise]
2. NOT(AND({p}, {q})) [premise]
3. BOT [NOT_E 1, 2]
END""",
        f"""PROOF
premises: IMPL(AND({p}, {q}), {r}), AND({p}, {q})
goal: {r}
1. IMPL(AND({p}, {q}), {r}) [premise]
2. AND({p}, {q}) [premise]
3. {r} [IMPL_E 1, 2]
END""",
        f"""PROOF
premises: {p}, NOT({p})
goal: BOT
1. {p} [premise]
2. NOT({p}) [premise]
3. BOT [NOT_E 2, 1]
END""",
        f"""PROOF
premises: NOT(NOT({p}))
goal: {p}
1. NOT(NOT({p})) [premise]
2. {p} [DNE 1]
END""",
        f"""PROOF
premises:
goal: OR({p}, NOT({p}))
1. OR({p}, NOT({p})) [LEM]
END""",
        f"""PROOF
premises: {p}
goal: OR({q}, {p})
1. {p} [premise]
2. OR({q}, {p}) [OR_I2 1]
END""",
        f"""PROOF
premises: {p}, {q}
goal: AND({q}, {p})
1. {p} [premise]
2. {q} [premise]
3. AND({q}, {p}) [AND_I 2, 1]
END""",
        f"""PROOF
premises: AND({p}, {q})
goal: AND({q}, {p})
1. AND({p}, {q}) [premise]
2. {p} [AND_E1 1]
3. {q} [AND_E2 1]
4. AND({q}, {p}) [AND_I 3, 2]
END""",
        f"""PROOF
premises: {p}, {q}, {r}, {s}
goal: AND(AND(AND({p}, {q}), {r}), {s})
1. {p} [premise]
2. {q} [premise]
3. {r} [premise]
4. {s} [premise]
5. AND({p}, {q}) [AND_I 1, 2]
6. AND(AND({p}, {q}), {r}) [AND_I 5, 3]
7. AND(AND(AND({p}, {q}), {r}), {s}) [AND_I 6, 4]
END""",
        f"""PROOF
premises: {p}, NOT({p})
goal: AND({q}, {r})
1. {p} [premise]
2. NOT({p}) [premise]
3. BOT [NOT_E 1, 2]
4. {q} [BOT_E 3]
5. {r} [BOT_E 3]
6. AND({q}, {r}) [AND_I 4, 5]
END""",
    ]
    rng.shuffle(templates)
    proofs = [templates[i % len(templates)] for i in range(max(0, n_probes))]
    return (
        "Deterministically sampled proof probes (independent of the current "
        "hypothesis):\n\n"
        + "\n\n".join(f"```alienlogic_proof\n{x}\n```" for x in proofs)
    )


def _require_agent_client() -> AgentClient:
    if _AGENT_CLIENT is None:
        raise RuntimeError("AgentClient has not been initialized")
    return _AGENT_CLIENT


def _require_agent_runtime():
    if _AGENT_RUNTIME is None:
        raise RuntimeError("AgentRuntime has not been initialized")
    return _AGENT_RUNTIME


def _usage_delta(before: dict[str, Any]) -> dict[str, int]:
    after = _require_agent_client().get_usage()
    return {
        key: max(0, int(after.get(key, 0) or 0)
                 - int(before.get(key, 0) or 0))
        for key in (
            "prompt_tokens", "completion_tokens", "reasoning_tokens",
            "total_tokens",
        )
    }


def _closed_book_fork(session: AgentSession) -> AgentSession:
    """Branch for a graded phase, with the verifier taken away.

    Summaries and held-out proofs are scored on what the model inferred during
    exploration, so a branch must not be able to test a proof before answering.
    Exploration only holds the tool inside its own `using_tools` block, so these
    branches are already toolless; saying so here keeps it that way.
    """

    return _require_agent_runtime().fork_closed_book(
        session,
        context=PhaseContext(
            sandbox="logic",
            phase="closed_book",
            track=EVAL_TRACK,
            framework=FRAMEWORK,
        ),
    )


def _proof_tool_definition(max_calls: int) -> dict[str, Any]:
    return {
        "name": PROOF_TOOL_NAME,
        "description": (
            "把一条完整的 alienlogic proof 交给 verifier 真实执行，立刻返回不透明"
            "诊断（reason_class + reason_id）。用它做差分实验：每次只改动一个你"
            "想验证的因素，看诊断如何变化。诊断不会解释规则名，也不会给出修复建议。"
            f"本轮最多可调用 {max_calls} 次；调用之间可以先想清楚再继续。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "proof": {
                    "type": "string",
                    "description": (
                        "一条完整证明，PROOF...END 格式，不要包含 markdown 代码"
                        "围栏。一条命令一行；行号从 1 递增；`|` 进入 assumption "
                        "box；`[<rule> <refs>]` 标注规则与引用。"
                    ),
                },
                "purpose": {
                    "type": "string",
                    "description": "一句话说明这条 probe 想验证什么 hypothesis。",
                },
            },
            "required": ["proof"],
        },
    }


def _exec_probe_call(
    call,
    episode: EpisodeSpec,
    *,
    label: str = "",
    locked_proof: str | None = None,
) -> dict[str, Any]:
    """Run one model-issued probe through the verifier."""

    args = call.arguments if isinstance(call.arguments, dict) else {}
    model_proof = str(args.get("proof") or "")
    proof_text = model_proof if locked_proof is None else locked_proof
    # Models sometimes fence the proof anyway; the parser wants it bare.
    stripped = extract_first_proof(f"```alienlogic_proof\n{proof_text}\n```")
    diag = proof_diagnostic(
        stripped or proof_text, episode.alien_rules,
        hide_reason_detail=True,
    )
    feedback = _format_diagnostic(diag)
    return {
        "proof": stripped or proof_text,
        "model_proof": model_proof,
        "locked": locked_proof is not None,
        "purpose": str(args.get("purpose") or ""),
        "diagnostic": diag,
        "result": ToolResult(
            call_id=call.call_id,
            output=feedback,
            name=call.name,
        ),
    }


def _run_probe_round(
        session: AgentSession,
        prompt: str,
        episode: EpisodeSpec,
        *,
        label: str = "",
        max_calls: int,
        locked_proofs: list[str] | None = None,
) -> tuple[str, list[dict[str, Any]], int]:
    """Let the model probe the verifier until it stops calling the tool.

    Returns the model's prose, one record per probe in the order it made them,
    and how many model calls the round cost.
    """

    if locked_proofs:
        max_calls = len(locked_proofs)
    # Forcing a call is what keeps a locked round from silently delivering no
    # probe, but two upstreams reject the parameter outright: Qwen, and
    # DeepSeek whenever thinking is on ("Thinking mode does not support this
    # tool_choice"). Match them by vendor rather than by one route's prefix.
    # The prefix test this replaces named the gateway's Messages door and
    # Ali's direct door, so it stopped recognising either vendor once the
    # control arms moved to GatewayA: every locked round of DeepSeek-V4-Pro's
    # logic arms died on its first probe. AlienCode carries the same guard,
    # and this is the fix it already had.
    locked_tool_override = (
        None
        if (
            "qwen" in MODEL.lower()
            or "deepseek" in MODEL.lower()
        )
        else {"tool_choice": "required"}
    )
    text = _chat(
        session,
        prompt,
        label=label,
        extra_overrides=(
            locked_tool_override if locked_proofs else None
        ),
    )
    response = session.last_response
    probes: list[dict[str, Any]] = []
    texts: list[str] = [text.strip()] if text.strip() else []
    model_calls = 1
    grace = 0
    rounds = 0
    nudges = 0

    def require_locked_call(current):
        nonlocal model_calls, nudges
        while (
            locked_proofs
            and len(probes) < max_calls
            and (current is None or not current.tool_calls)
            and nudges < 4
        ):
            nudges += 1
            follow_text = _chat(
                session,
                (
                    "还有系统锁定的 proof 尚未执行。请现在调用 "
                    f"{PROOF_TOOL_NAME}；你填写的 proof 会被锁定 proof 替换。"
                ),
                label=f"{label} locked retry {nudges}",
                extra_overrides=locked_tool_override,
            )
            model_calls += 1
            if follow_text.strip():
                texts.append(follow_text.strip())
            current = session.last_response
        return current

    response = require_locked_call(response)

    while response is not None and response.tool_calls:
        rounds += 1
        allowed = response.tool_calls[:max(0, max_calls - len(probes))]
        refused = response.tool_calls[len(allowed):]
        results: list[ToolResult] = []

        for call in allowed:
            if call.name != PROOF_TOOL_NAME:
                message = f"[ToolError] 未知工具 {call.name}"
                print(f"  [Probe] {label}: {message}")
                results.append(ToolResult(
                    call_id=call.call_id,
                    output=message,
                    name=call.name,
                    is_error=True,
                ))
                continue
            record = _exec_probe_call(
                call,
                episode,
                label=label,
                locked_proof=(
                    locked_proofs[len(probes)] if locked_proofs else None
                ),
            )
            results.append(record.pop("result"))
            probes.append(record)
            _require_agent_runtime().after_environment_feedback(
                PhaseContext(
                    sandbox="logic",
                    phase="explore",
                    label=label,
                    track=EVAL_TRACK,
                    framework=FRAMEWORK,
                ),
                probe={"tool": call.name, "proof": record.get("proof", "")},
                feedback=record.get("diagnostic", {}),
            )
            print(
                f"  [Probe {len(probes)}/{max_calls}] {label}\n"
                f"  → {_format_diagnostic(record['diagnostic'])}"
            )

        for call in refused:
            results.append(ToolResult(
                call_id=call.call_id,
                output=(
                    f"[ToolError] 本轮 probe 预算已用完（上限 {max_calls} 次）。"
                    "请不要再调用工具，直接总结你从本轮 probe 中学到了什么。"
                ),
                name=call.name,
                is_error=True,
            ))

        extra_overrides = None
        suffix = ""
        # Rounds count too: a model asking for an unknown tool never fills
        # `probes`, so it could otherwise loop untouched.
        if len(probes) >= max_calls or rounds >= max_calls:
            grace += 1
            if grace > _TOOL_LOOP_GRACE:
                raise RuntimeError(
                    f"{label}: model kept calling {PROOF_TOOL_NAME} for "
                    f"{grace} turns after spending its {max_calls}-probe "
                    "budget, ignoring tool_choice=none")
            extra_overrides = {"tool_choice": "none"}
            suffix = " (budget spent)"

        try:
            # GatewayB rejects tool_choice=none while tools are present, and Qwen's
            # Responses route may ignore it unless the tool list is also empty.
            # Assign directly rather than nesting using_tools([]): a provider
            # can still return one last tool call, and the outer loop must be
            # allowed to consume that call's error result before either context
            # checks for pending calls.
            force_without_tools = (
                extra_overrides
                and extra_overrides.get("tool_choice") == "none"
                and (MODEL.startswith(
                        ("messages/api_gateway_b_", "api_gateway_b_"))
                     or MODEL.startswith("api_ali_"))
            )
            if force_without_tools:
                previous_tools = session.tools
                previous_response_id = session.last_response_id
                session.tools = []
                # Ali's Responses endpoint can inherit the prior turn's tool
                # schema through previous_response_id even when this request
                # sends no tools.  Replay the accumulated wire history inline
                # to sever that server-side capability while preserving the
                # conversation and the pending function-call outputs.
                detach_qwen_continuation = MODEL.startswith("api_ali_")
                if detach_qwen_continuation:
                    session.last_response_id = None
                submitted = False
                try:
                    overrides = _chat_kwargs(DEFAULT_MAX_TOKENS)
                    if detach_qwen_continuation:
                        overrides["tool_choice"] = "none"
                    follow = session.submit_tool_results(
                        results,
                        label=f"{label} probe-results{suffix}",
                        request_overrides=overrides,
                    )
                    submitted = True
                finally:
                    if detach_qwen_continuation and not submitted:
                        session.last_response_id = previous_response_id
                    session.tools = previous_tools
            else:
                follow = session.submit_tool_results(
                    results,
                    label=f"{label} probe-results{suffix}",
                    request_overrides={
                        **_chat_kwargs(DEFAULT_MAX_TOKENS),
                        **(extra_overrides or {}),
                    },
                )
        except Exception as exc:
            print(f"  [Chat Error] {type(exc).__name__}: {str(exc)[:200]}")
            break
        model_calls += 1
        response = session.last_response
        if follow.text and follow.text.strip():
            texts.append(follow.text.strip())
        response = require_locked_call(response)

    if locked_proofs and len(probes) != len(locked_proofs):
        raise RuntimeError(
            f"{label}: delivered {len(probes)}/{len(locked_proofs)} "
            "locked proofs"
        )
    return "\n\n".join(texts), probes, model_calls


def run_explore(state: dict[str, Any], episode: EpisodeSpec, loop_idx: int,
                records: list[dict[str, Any]]) -> None:
    print("\n" + "━" * 72)
    print(f"  Explore Loop {loop_idx}")
    print("━" * 72)
    remaining = MAX_TOTAL_PROBES - state.get("total_probes_used", 0)
    if remaining <= 0 and CONTROL_MODE != "think":
        print("  [Skip] probe budget exhausted")
        return
    session: AgentSession = state["session"]
    for r in range(1, N_PROBES_PER_LOOP + 1):
        t0 = time.time()
        label = f"Explore {loop_idx}.{r}"
        controlled = CONTROL_MODE in {"think", "passive", "random"}
        budget = max(
            0, MAX_TOTAL_PROBES - state.get("total_probes_used", 0))
        per_round = (
            0 if CONTROL_MODE == "think"
            else min(MAX_PROBES_PER_ROUND, budget)
        )

        # Lock the harness-selected proof batch before the model update call.
        # Only executable proof blocks are exposed to the target live session;
        # donor prose and donor reasoning are never copied into it.
        control_source = None
        donor_response_hash = None
        if CONTROL_MODE == "passive":
            donor_response = _passive_response(label)
            donor_response_hash = hashlib.sha256(
                donor_response.encode()).hexdigest()
            proofs = _donor_proofs(label)
            control_source = CONTROL_BANK_PATH
        elif CONTROL_MODE == "random":
            n_random = _donor_probe_count(label)
            random_response = _random_probe_response(loop_idx, r, n_random)
            proofs = extract_proof_blocks(random_response)
            control_source = (
                f"sha256({RUN_ID}:{loop_idx}:{r}); "
                f"count_from={CONTROL_BANK_PATH}"
            )
        else:
            proofs = []

        if controlled and CONTROL_MODE != "think":
            proofs = proofs[:per_round]
            if not proofs:
                raise RuntimeError(
                    f"{CONTROL_MODE} produced no executable proofs for {label}")

        prompt_parts: list[str] = []
        if state.get("pending_feedback"):
            prompt_parts.append(state["pending_feedback"])
            prompt_parts.append("---")
            state["pending_feedback"] = None
        if CONTROL_MODE == "think":
            prompt_parts.append(
                "本轮不执行任何新 proof，也不会提供新 verifier 反馈。请只根据已有"
                "上下文更新你对 alien rules 的假设，记录目前最可信与最不确定的"
                "约束。不要提出或输出 proof probe。"
            )
        elif controlled:
            locked = "\n\n".join(
                f"```alienlogic_proof\n{proof}\n```" for proof in proofs)
            prompt_parts.append(
                "以下 proof batch 已由系统锁定，本轮只会执行这些 probes：\n"
                f"{locked}\n"
                f"请按顺序调用 {PROOF_TOOL_NAME} 共 {len(proofs)} 次。"
                "你不能修改、替换或提出其他 probe；工具参数中的 proof 会由系统"
                "替换为上面对应的锁定 proof。请先记录当前规则假设、最不确定的"
                "规则，并预测这个固定 batch 的结果。"
            )
        elif TOOL_MODE:
            prompt_parts.append(
                f"继续探索隐藏 alien 规则。本轮最多调用 {per_round} 次 "
                f"{PROOF_TOOL_NAME} 工具，episode 剩余 probe 预算 {budget}。"
                "每次调用都会真实执行并立刻返回诊断，你可以看到结果后再决定下一"
                "条 probe，也可以不用完预算。"
            )
            prompt_parts.append(
                "要点提示：\n"
                "  - 每条 probe 应该是为某个 hypothesis 设计的差分实验，目标是"
                "区分多个候选 alien rule。\n"
                "  - 不要重复 seed 中已经看过的 reason_id；优先 probe 你尚未观察过"
                "的规则。\n"
                "  - 用 purpose 参数写清这条 probe 想验证什么；最后再用一段话总结"
                "本轮学到了什么。\n"
            )
        else:
            prompt_parts.append(
                f"继续探索隐藏 alien 规则。本轮最多提交 {per_round} 条 probe（每条都是"
                "完整的 ```alienlogic_proof``` 代码块），episode 剩余 probe 预算 "
                f"{budget}。"
            )
            prompt_parts.append(
                "要点提示：\n"
                "  - 每条 probe 应该是为某个 hypothesis 设计的差分实验，目标是"
                "区分多个候选 alien rule。\n"
                "  - 不要重复 seed 中已经看过的 reason_id；优先 probe 你尚未观察过"
                "的规则。\n"
                "  - 请同时附一段简短自然语言说明：你这一轮想验证什么。\n"
            )
        prompt = "\n".join(prompt_parts)
        usage_before = _require_agent_client().get_usage()
        response_before = session.last_response
        tool_probes: list[dict[str, Any]] | None = None
        tool_model_calls = 1
        if controlled and CONTROL_MODE != "think" and TOOL_MODE:
            with session.using_tools([_proof_tool_definition(per_round)]):
                response, tool_probes, tool_model_calls = _run_probe_round(
                    session,
                    prompt,
                    episode,
                    label=f"{label} Control Update",
                    max_calls=per_round,
                    locked_proofs=proofs,
                )
            retry_meta = {
                "empty_initial": not response.strip(),
                "retry_attempted": False,
                "retry_success": False,
                "transport_retry_count": int(
                    (session.last_call_meta or {}).get(
                        "transport_retry_count", 0) or 0),
                "effective_max_tokens": (session.last_call_meta or {}).get(
                    "effective_max_tokens"),
            }
        elif controlled:
            # Exactly one logical update call per controlled explore round.
            # Do not add an empty-response retry: that would break call
            # matching against the preflighted donor self run.
            response = _chat(
                session, prompt, label=f"{label} Control Update")
            call_response = session.last_response
            transport_meta = (
                session.last_call_meta
                if call_response is not response_before
                else {}
            )
            retry_meta = {
                "empty_initial": not response.strip(),
                "retry_attempted": False,
                "retry_success": False,
                "transport_retry_count": int(
                    transport_meta.get("transport_retry_count", 0) or 0),
                "effective_max_tokens": transport_meta.get(
                    "effective_max_tokens"),
            }
        elif TOOL_MODE:
            with session.using_tools([_proof_tool_definition(per_round)]):
                response, tool_probes, tool_model_calls = _run_probe_round(
                    session, prompt, episode,
                    label=label, max_calls=per_round,
                )
            retry_meta = {
                "empty_initial": not response.strip(),
                "retry_attempted": False,
                "retry_success": False,
                "transport_retry_count": int(
                    (session.last_call_meta or {}).get(
                        "transport_retry_count", 0) or 0),
                "effective_max_tokens": (session.last_call_meta or {}).get(
                    "effective_max_tokens"),
            }
        else:
            response, retry_meta = _chat_with_empty_retry(
                session, prompt, label=label,
            )
        call_response = session.last_response
        if call_response is response_before:
            call_response = None
        if not response.strip():
            response = "（模型返回了空回复）"
        call_usage = _usage_delta(usage_before)

        if tool_probes is not None:
            # The verifier already ran inside the tool loop, one call at a
            # time, so there is nothing left to batch here.
            diags = [
                {
                    "proof": item["proof"],
                    "model_proof": item.get("model_proof", ""),
                    "locked": bool(item.get("locked")),
                    "diagnostic": item["diagnostic"],
                }
                for item in tool_probes
            ]
            accepted_proofs = [item["proof"] for item in tool_probes]
            skipped_proofs = []
            state["total_probes_used"] = (
                state.get("total_probes_used", 0) + len(tool_probes)
            )
        else:
            if not controlled:
                proofs = extract_proof_blocks(response)
            accepted_proofs = proofs[:per_round]
            skipped_proofs = proofs[per_round:]
            diags = []
            for proof_text in accepted_proofs:
                diag = proof_diagnostic(
                    proof_text, episode.alien_rules, hide_reason_detail=True,
                )
                diags.append({"proof": proof_text, "diagnostic": diag})
                state["total_probes_used"] = (
                    state.get("total_probes_used", 0) + 1
                )
        elapsed = time.time() - t0
        feedback = "本轮 probe 反馈（reason_id 是不透明编码）：\n"
        for i, item in enumerate(diags, 1):
            feedback += (
                f"[系统锁定 probe {i}] " if controlled else f"[probe {i}] "
            )
            feedback += _format_diagnostic(item["diagnostic"]) + "\n"
        probe_payload = json.dumps(accepted_proofs, ensure_ascii=False)
        call_metrics = {
            "model_call_count": (
                tool_model_calls if tool_probes is not None
                else 1 + int(retry_meta["retry_attempted"])
            ),
            "input_tokens": call_usage["prompt_tokens"],
            "output_tokens": call_usage["completion_tokens"],
            "reasoning_tokens": call_usage["reasoning_tokens"] or None,
            "configured_max_tokens": DEFAULT_MAX_TOKENS,
            "effective_max_tokens": retry_meta.get("effective_max_tokens"),
            "retry_count": int(retry_meta["retry_attempted"]),
            "transport_retry_count": retry_meta["transport_retry_count"],
            "response_length": len(response),
            "reasoning_chars": sum(
                len(item.text or "") for item in call_response.reasoning
            ) if call_response is not None else 0,
        }
        records.append({
            "phase": "explore",
            "loop": loop_idx,
            "round": r,
            "response": response,
            "n_probes_submitted": (
                len(accepted_proofs) if tool_probes is not None
                else len(proofs)
            ),
            "n_probes_accepted_for_eval": len(accepted_proofs),
            "n_probes_skipped_budget": len(skipped_proofs),
            "diagnostics": diags,
            "time_seconds": elapsed,
            "control_mode": CONTROL_MODE,
            "control_source": control_source,
            "model_update": response if controlled else None,
            "probe_hash": (
                hashlib.sha256(probe_payload.encode()).hexdigest()
                if accepted_proofs else None
            ),
            "donor_response_hash": donor_response_hash,
            "tool_mode": TOOL_MODE,
            "call_metrics": call_metrics,
            **retry_meta,
        })
        # Under tool mode each diagnostic already came back as a tool result
        # mid-round, so carrying it into the next prompt would show it twice.
        state["pending_feedback"] = (
            None if CONTROL_MODE == "think" or tool_probes is not None
            else feedback
        )
        label_kind = "Control Update" if controlled else "Response"
        print(f"\n[Explore {loop_idx}.{r} {label_kind}]\n{_trim(response)}")
        if CONTROL_MODE == "think":
            print("[Control think] no proof executed; no new feedback")
        for i, item in enumerate(diags, 1):
            print(f"[Probe {i}] {_format_diagnostic(item['diagnostic'])}")
        print(f"[Explore {loop_idx}.{r}] probes={len(accepted_proofs)} "
              f"skipped={len(skipped_proofs)} total_used="
              f"{state.get('total_probes_used', 0)}/{MAX_TOTAL_PROBES} "
              f"time={elapsed:.1f}s")


# ═══════════════════════════════════════════════════════════════════════
#  Phase: milestone (rule report + held-out tests)
# ═══════════════════════════════════════════════════════════════════════

def run_milestone(state: dict[str, Any], episode: EpisodeSpec,
                  milestone_id: int | str,
                  records: list[dict[str, Any]],
                  deferred: list[dict[str, Any]] | None = None) -> None:
    print("\n" + "#" * 72)
    print(f"  Milestone {milestone_id}")
    print("#" * 72)
    # Left pending on purpose. Summary and held-out sessions are throwaway
    # forks, so feedback consumed there would never reach the live session.
    # `run_explore` is the one live-session consumer.
    pending = state.get("pending_feedback")
    parts: list[str] = []
    if pending:
        parts.append(pending)
        parts.append("---")
    parts.append(
        "请基于至此为止的 seed/explore 反馈，输出结构化 JSON 总结当前你已"
        "发现的 alien rules。\n"
        "字段约定：\n"
        "{\n"
        '  "rules": [\n'
        '    {"target_rule": "<standard rule name like BOT_E / IMPL_E / ...>",\n'
        '     "constraint": "<one-sentence description of the side condition>",\n'
        '     "evidence_reason_ids": ["K7", ...],\n'
        '     "confidence": 0.0~1.0}\n'
        "  ],\n"
        '  "uncertainties": [...],\n'
        '  "next_strategy": "<brief plan for next probes or for held-out proofs>"\n'
        "}\n"
        "只输出 JSON，不要写散文。"
    )
    summary_prompt = "\n".join(parts)

    summary_session: AgentSession = _closed_book_fork(state["session"])
    summary_response, summary_meta = _chat_with_empty_retry(
        summary_session, summary_prompt, label=f"M{milestone_id} Summary",
        max_tokens=SUMMARY_MAX_TOKENS,
    )
    summary = extract_json_object(summary_response) or {}
    print(f"\n[M{milestone_id} Summary]\n{_trim(summary_response)}")

    # An independent provider-native base, taken before any later live call so
    # the theorems are answered from what this milestone knew. Exploration
    # continues afterwards, so the end-of-run snapshot holds knowledge the model
    # did not have here -- redoing a theorem the API ate needs *this* state.
    base = _heldout_base(state)
    base_snapshot = _milestone_base_snapshot(base, milestone_id)

    if deferred is not None:
        deferred.append({
            "milestone_id": milestone_id, "milestone_label": str(milestone_id),
            "summary_text": summary_response, "summary": summary,
            "retry_meta": summary_meta, "base": base,
            "pending_feedback": pending,
            "base_snapshot": base_snapshot,
        })
        print(f"[M{milestone_id}] held-out tests deferred to merged pool")
        return

    test_results = run_heldout_tests(
        state, episode, milestone_label=str(milestone_id), base=base,
    )
    rec = _build_milestone_record(
        milestone_id=milestone_id, summary_text=summary_response,
        summary=summary, test_results=test_results, retry_meta=summary_meta,
        base_snapshot=base_snapshot,
    )
    records.append(rec)
    _print_milestone_summary(milestone_id, rec)


def _graded_answer(response: str) -> bool:
    """Whether a held-out reply holds something to grade at all."""

    return bool(extract_first_proof(response) or extract_unprovable(response))


def _with_pending_feedback(prompt: str, pending_feedback: str | None) -> str:
    if not pending_feedback:
        return prompt
    return f"{pending_feedback}\n\n---\n\n{prompt}"


def _run_single_heldout(theorem: HeldoutTheorem, base: AgentSession,
                        *, pending_feedback: str | None,
                        milestone_label: str,
                        alien_rules: tuple[str, ...]) -> dict[str, Any]:
    """Process one theorem on its own provider-native session fork.

    Pure per-theorem work — safe to invoke concurrently because:
      * `base` is read-only and AgentSession.fork() is lock-protected;
      * every worker mutates only its own child session;
      * trace writes and usage aggregation are lock-protected by AgentClient;
      * `proof_diagnostic` is a pure function over its inputs;
      * `print` calls are coalesced into a single multi-line block so
        interleaving with parallel workers stays readable.
    """
    t0 = time.time()
    session = _closed_book_fork(base)
    prompt = _with_pending_feedback(
        _heldout_prompt(theorem), pending_feedback,
    )
    retry_prompt = _heldout_retry_prompt(theorem.id, prompt)
    response, retry_meta = _chat_with_empty_retry(
        session, prompt, label=f"M{milestone_label} {theorem.id}",
        max_tokens=TEST_MAX_TOKENS,
        retry_prompt=retry_prompt,
        retry_max_tokens=TEST_RETRY_MAX_TOKENS,
        # Exactly what grading looks for below, so a turn is re-asked when and
        # only when it would otherwise be scored as having said nothing.
        answered=_graded_answer,
    )
    if not response.strip():
        response = "（模型返回了空回复）"
    proof_text = extract_first_proof(response)
    unprovable = extract_unprovable(response)
    if proof_text:
        diag = proof_diagnostic(
            proof_text, alien_rules,
            hide_reason_detail=False,
            expected_goal=theorem.goal,
            expected_premises=theorem.premises,
        )
    elif unprovable:
        diag = {
            "accepted": False, "verifier_passed": False, "goal_match": False,
            "reason_class": "UNPROVABLE_CLAIM", "reason_id": "MODEL_DECLINED",
            "n_steps": 0,
        }
    elif _deadline_was_spent():
        # Out of clock, not out of ideas. Both arrive here with nothing to
        # parse, and filing them together made a slow model look like one that
        # cannot format a proof -- and hid from the audit that the deadline,
        # not the theorem, decided the score.
        diag = {
            "accepted": False, "verifier_passed": False, "goal_match": False,
            "reason_class": "TIMEOUT", "reason_id": "BUDGET_EXHAUSTED",
            "n_steps": 0,
        }
    else:
        diag = {
            "accepted": False, "verifier_passed": False, "goal_match": False,
            "reason_class": "PARSE_ERROR", "reason_id": "NO_PROOF_BLOCK",
            "n_steps": 0,
        }

    # Asymmetric scoring (see original docstring).
    if theorem.alien_provable:
        accepted = bool(diag.get("accepted"))
    else:
        verifier_accepted = bool(diag.get("accepted"))
        if verifier_accepted:
            accepted = False
            diag = dict(diag)
            diag["reason_class"] = "DESIGN_BUG"
            diag["reason_id"] = "UNPROVABLE_PASSED_VERIFIER"
        else:
            accepted = bool(unprovable)

    elapsed = time.time() - t0
    result = {
        "theorem_id": theorem.id,
        "cluster_id": theorem.cluster_id,
        "role": theorem.role,
        "alien_provable": theorem.alien_provable,
        "alien_aware_required": theorem.alien_aware_required,
        "min_alien_steps": theorem.min_alien_steps,
        "diagnostic": diag,
        "accepted": accepted,
        "n_steps": diag.get("n_steps", 0),
        "proof_text": proof_text,
        "response": response,
        "time_seconds": elapsed,
        "claimed_unprovable": unprovable,
        **retry_meta,
    }
    if _deadline_was_spent():
        # Same two fields the code sandbox writes, so a timeout means the same
        # thing in both and one repair rule can serve both. The budget travels
        # with the verdict: re-asking is worth it once the allowance goes up,
        # and pointless at the allowance that already refused it.
        result["timed_out"] = True
        result["budget_seconds"] = int(_request_deadline())
    # Coalesce into a single print so parallel workers don't shred each
    # other's output mid-line. Python's print is GIL-locked, so one call
    # per theorem is atomic.
    print(
        f"\n[M{milestone_label} {theorem.id} Response]\n{_trim(response)}"
        f"\n[M{milestone_label} {theorem.id}] "
        f"{'PASS' if accepted else 'FAIL'} "
        f"steps={result['n_steps']} "
        f"reason={diag.get('reason_id')} time={elapsed:.1f}s",
        flush=True,
    )
    return result


# Concurrency for the held-out test loop. Default = 1 (serial, original
# behaviour). Override with env var `ALIENLOGIC_HELDOUT_WORKERS=<int>` to
# parallelize via ThreadPoolExecutor. Each theorem is processed on an
# independent AgentSession fork (see `_run_single_heldout`); shared trace and
# usage stores are lock-protected.
def _heldout_workers() -> int:
    try:
        n = int(os.environ.get("ALIENLOGIC_HELDOUT_WORKERS", "1"))
    except ValueError:
        return 1
    return max(1, n)


def _defer_workers() -> int:
    """Worker count for the merged deferred pool.

    Defaults to ALIENLOGIC_HELDOUT_WORKERS, but ALIENLOGIC_DEFER_WORKERS can
    push it higher since the merged pool spans every milestone's batch at once
    (n_milestones × n_theorems jobs) and benefits from more concurrency.
    """
    v = os.environ.get("ALIENLOGIC_DEFER_WORKERS")
    if v:
        try:
            return max(1, int(v))
        except ValueError:
            pass
    return _heldout_workers()


def _heldout_base(state: dict[str, Any]) -> AgentSession:
    """Create the immutable, independent base used by theorem workers."""
    session: AgentSession = state["session"]
    return _closed_book_fork(session)


def _milestone_base_snapshot(
    base: AgentSession, milestone_id: int | str
) -> str | None:
    """Persist the session this milestone's theorems are answered from.

    It has to be the closed-book fork rather than the live session: snapshots
    are filed under the session id, so snapshotting the live one every
    milestone would just overwrite one file with the latest state. The fork is
    created per milestone and never advances, so its file keeps exactly what
    this milestone knew -- which is what makes a later redo honest.
    """

    try:
        return str(base.snapshot())
    except Exception as exc:  # noqa: BLE001 - a missing snapshot is not fatal
        print(f"  [warn] M{milestone_id} base snapshot failed: {exc}", flush=True)
        return None


def _safe_single_heldout(theorem: HeldoutTheorem, base: AgentSession,
                         *, pending_feedback: str | None,
                         milestone_label: str,
                         alien_rules: tuple[str, ...]) -> dict[str, Any]:
    """`_run_single_heldout` wrapped so a worker exception becomes a FAIL
    record instead of propagating and killing the pool."""
    try:
        return _run_single_heldout(
            theorem, base, pending_feedback=pending_feedback,
            milestone_label=milestone_label,
            alien_rules=alien_rules,
        )
    except Exception as exc:
        print(
            f"\n[M{milestone_label} {theorem.id}] WORKER_FAIL "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return {
            "theorem_id": theorem.id,
            "cluster_id": theorem.cluster_id,
            "role": theorem.role,
            "alien_provable": theorem.alien_provable,
            "alien_aware_required": theorem.alien_aware_required,
            "min_alien_steps": theorem.min_alien_steps,
            "diagnostic": {
                "accepted": False, "verifier_passed": False,
                "goal_match": False,
                "reason_class": "WORKER_ERROR",
                "reason_id": type(exc).__name__,
                "n_steps": 0,
            },
            "accepted": False,
            "n_steps": 0,
            "proof_text": "",
            "response": f"（worker 异常：{exc}）",
            "time_seconds": 0.0,
            "claimed_unprovable": False,
            "empty_retry_used": False,
            "empty_retry_succeeded": False,
        }


# How many extra sweeps a held-out batch spends chasing answers the provider
# never delivered, and the seconds between them. Throttling clears in minutes,
# so a few patient passes recover a run that would otherwise score as a wipeout.
# Tests set the backoff to zero rather than waiting out a real one.
_MAX_HELDOUT_REPAIR_PASSES = int(
    os.environ.get("EVAL_TEST_REPAIR_PASSES", "3"))
_HELDOUT_REPAIR_BACKOFF = float(
    os.environ.get("EVAL_TEST_REPAIR_BACKOFF", "30"))


def _worker_failed(result: dict[str, Any] | None) -> bool:
    """Whether the provider, not the model, is why this theorem has no proof."""

    return ((result or {}).get("diagnostic") or {}).get(
        "reason_class") == "WORKER_ERROR"


def _heldout_ledger_path() -> str:
    """Where finished held-out verdicts are journalled for this run."""

    folder = os.path.join(_REPO_ROOT, "logs", "logic", "checkpoints")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"heldout_{RUN_ID or 'unnamed'}.jsonl")


def _load_heldout_ledger(names: dict[Any, str]) -> dict[Any, dict[str, Any]]:
    """Verdicts this run already produced, keyed back to their job."""

    path = _heldout_ledger_path()
    if not os.path.exists(path):
        return {}
    by_name = {name: key for key, name in names.items()}
    done: dict[Any, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            key = by_name.get(row.get("job"))
            if key is not None and isinstance(row.get("result"), dict):
                done[key] = row["result"]
    if done:
        print(f"\n[heldout-resume] 从账本恢复 {len(done)} 道已答题目 "
              f"({os.path.basename(path)})", flush=True)
    return done


def _dispatch_heldout(jobs: dict[Any, dict[str, Any]], *, n_workers: int,
                      banner: str,
                      names: dict[Any, str] | None = None
                      ) -> dict[Any, dict[str, Any]]:
    """Run every held-out job, then chase the ones the provider never answered.

    A worker exception is filed as an unaccepted proof, which downstream reads
    exactly like a model that could not find one. Re-asking those from the same
    milestone base is what keeps a throttled stretch out of the score. Later
    passes use fewer workers, since load is usually what caused the failure.

    When `names` gives each job a stable name, every verdict is journalled as
    it lands. This sandbox answers all of its held-out theorems in one pool at
    the very end of a run, so without the journal a crash there discards hours
    of exploration and grading alike; with it, the next launch resumes.
    """

    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: dict[Any, dict[str, Any]] = {}
    ledger = None
    if names:
        results.update(_load_heldout_ledger(names))
        ledger = open(_heldout_ledger_path(), "a", encoding="utf-8")

    def record(key: Any, value: dict[str, Any]) -> None:
        results[key] = value
        if ledger is None:
            return
        ledger.write(json.dumps(
            {"job": names[key], "result": value}, ensure_ascii=False) + "\n")
        ledger.flush()

    todo = [key for key in jobs if key not in results]
    for attempt in range(_MAX_HELDOUT_REPAIR_PASSES + 1):
        if not todo:
            break
        workers = max(1, min(n_workers, len(todo)) >> attempt)
        if attempt:
            print(
                f"\n[heldout-repair] {banner}: {len(todo)} unanswered, "
                f"pass {attempt}/{_MAX_HELDOUT_REPAIR_PASSES} "
                f"(workers={workers})",
                flush=True,
            )
            time.sleep(_HELDOUT_REPAIR_BACKOFF * attempt)
        if workers <= 1:
            for key in todo:
                record(key, _safe_single_heldout(**jobs[key]))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_safe_single_heldout, **jobs[key]): key
                    for key in todo
                }
                for fut in as_completed(futures):
                    record(futures[fut], fut.result())
        todo = [key for key in jobs if _worker_failed(results.get(key))]
    if todo:
        print(
            f"\n[heldout-damaged] {banner}: {len(todo)} theorems never "
            f"answered; these scores are not the model's",
            flush=True,
        )
    if ledger is not None:
        ledger.close()
    return results


def _run_heldout_batch(base: AgentSession, episode: EpisodeSpec,
                       *, pending_feedback: str | None,
                       milestone_label: str,
                       n_workers: int) -> list[dict[str, Any]]:
    """Run one milestone's held-out batch over a fixed base snapshot."""
    theorems = list(episode.heldout_theorems)
    jobs = {
        i: {
            "theorem": theorem,
            "base": base,
            "pending_feedback": pending_feedback,
            "milestone_label": milestone_label,
            "alien_rules": episode.alien_rules,
        }
        for i, theorem in enumerate(theorems)
    }
    print(
        f"\n[heldout] M{milestone_label}: dispatching {len(theorems)} "
        f"theorems across {min(n_workers, len(theorems))} workers",
        flush=True,
    )
    results = _dispatch_heldout(
        jobs, n_workers=n_workers, banner=f"M{milestone_label}")
    return [results[i] for i in range(len(theorems))]


def run_heldout_tests(state: dict[str, Any], episode: EpisodeSpec,
                      *, milestone_label: str,
                      base: AgentSession | None = None) -> list[dict[str, Any]]:
    """Forked context — each theorem gets its own AgentSession child."""
    base = base if base is not None else _heldout_base(state)
    return _run_heldout_batch(
        base, episode, pending_feedback=state.get("pending_feedback"),
        milestone_label=milestone_label,
        n_workers=_heldout_workers(),
    )


def _flush_deferred(deferred: list[dict[str, Any]], episode: EpisodeSpec,
                    milestones: list[dict[str, Any]]) -> None:
    """Design A: execute every deferred milestone's held-out batch in ONE
    merged pool, then build + append the milestone records in milestone order.

    Each job forks from its own milestone AgentSession base (`task["base"]`),
    so scoring is identical to running the batches inline at milestone time —
    only scheduling differs (single high-concurrency pool, no per-milestone
    barrier, no explore stall waiting on tests)."""
    theorems = list(episode.heldout_theorems)
    n_workers = _defer_workers()
    jobs = {
        (ti, hi): {
            "theorem": theorems[hi],
            "base": deferred[ti]["base"],
            "pending_feedback": deferred[ti].get("pending_feedback"),
            "milestone_label": deferred[ti]["milestone_label"],
            "alien_rules": episode.alien_rules,
        }
        for ti in range(len(deferred))
        for hi in range(len(theorems))
    }

    labels = ", ".join(f"M{t['milestone_label']}" for t in deferred)
    print(
        f"\n[heldout-deferred] flushing {len(deferred)} milestones "
        f"[{labels}] × {len(theorems)} theorems = {len(jobs)} jobs "
        f"across {n_workers} workers",
        flush=True,
    )
    # Name each job by what it is rather than by its position, so the journal
    # still lines up if a later launch rebuilds the job table.
    names = {
        (ti, hi): f"M{deferred[ti]['milestone_label']}|{theorems[hi].id}"
        for ti in range(len(deferred))
        for hi in range(len(theorems))
    }
    flat = _dispatch_heldout(jobs, n_workers=n_workers, banner="deferred",
                             names=names)
    results: dict[int, list[dict[str, Any] | None]] = {
        ti: [flat[(ti, hi)] for hi in range(len(theorems))]
        for ti in range(len(deferred))
    }

    for ti, task in enumerate(deferred):
        rec = _build_milestone_record(
            milestone_id=task["milestone_id"],
            summary_text=task["summary_text"], summary=task["summary"],
            test_results=results[ti], retry_meta=task["retry_meta"],
            base_snapshot=task.get("base_snapshot"),
        )
        milestones.append(rec)
        _print_milestone_summary(task["milestone_id"], rec)


def _heldout_prompt(theorem: HeldoutTheorem) -> str:
    premises = ", ".join(theorem.premises) if theorem.premises else "(none)"
    return (
        f"【Held-out {theorem.id}】写出一段完整 PROOF...END 证明（放在 "
        "```alienlogic_proof``` 代码块），verifier 会在当前 alien rule 下评判。\n\n"
        f"premises: {premises}\n"
        f"goal: {theorem.goal}\n\n"
        "如果你认为该定理在当前 alien rule 下不可证，请回答一行：\n"
        "    ANSWER: UNPROVABLE\n"
        "再附一句话说明，但不需要写证明。\n"
        "否则，必须给出完整证明。verifier 会拒绝任何不合规则的步骤。"
    )


# ═══════════════════════════════════════════════════════════════════════
#  Milestone record + metrics
# ═══════════════════════════════════════════════════════════════════════

def _build_milestone_record(*, milestone_id: int | str, summary_text: str,
                            summary: dict[str, Any],
                            test_results: list[dict[str, Any]],
                            retry_meta: dict[str, Any],
                            base_snapshot: str | None = None) -> dict[str, Any]:
    n_total = len(test_results)
    n_pass = sum(1 for r in test_results if r["accepted"])
    aware_total = sum(1 for r in test_results if r["alien_aware_required"])
    aware_pass = sum(1 for r in test_results if r["alien_aware_required"] and r["accepted"])
    unprovable_total = sum(
        1 for r in test_results if r.get("alien_provable") is False
    )
    unprovable_correct = sum(
        1 for r in test_results
        if r.get("alien_provable") is False and r["accepted"]
    )
    # Minimality is only meaningful when an actual proof was accepted (n_steps>0).
    # UNPROVABLE-correct answers contribute n_steps=0 and are excluded.
    minimality_ratios = [
        (r["min_alien_steps"] / r["n_steps"])
        if r["accepted"] and r["n_steps"] and r.get("alien_provable") is not False
        else None
        for r in test_results
    ]
    valid_ratios = [x for x in minimality_ratios if x is not None]
    pass_by_role: dict[str, dict[str, int]] = {}
    for r in test_results:
        bucket = pass_by_role.setdefault(r["role"], {"pass": 0, "total": 0})
        bucket["total"] += 1
        if r["accepted"]:
            bucket["pass"] += 1
    pass_by_cluster: dict[str, dict[str, int]] = {}
    for r in test_results:
        bucket = pass_by_cluster.setdefault(r["cluster_id"], {"pass": 0, "total": 0})
        bucket["total"] += 1
        if r["accepted"]:
            bucket["pass"] += 1
    return {
        "milestone": milestone_id,
        "session_snapshot_path": base_snapshot,
        "summary_text": summary_text,
        "summary": summary,
        "summary_empty_initial": retry_meta.get("empty_initial"),
        "summary_retry_attempted": retry_meta.get("retry_attempted"),
        "summary_retry_success": retry_meta.get("retry_success"),
        "test_total": n_total,
        "test_pass": n_pass,
        "pass_rate": n_pass / n_total if n_total else 0.0,
        "alien_aware_total": aware_total,
        "alien_aware_pass": aware_pass,
        "alien_aware_pass_rate": aware_pass / aware_total if aware_total else 0.0,
        "unprovable_total": unprovable_total,
        "unprovable_correct": unprovable_correct,
        "unprovable_recognition_rate": (
            unprovable_correct / unprovable_total if unprovable_total else None
        ),
        "proof_minimality_mean": (
            sum(valid_ratios) / len(valid_ratios) if valid_ratios else None
        ),
        "pass_by_role": pass_by_role,
        "pass_by_cluster": pass_by_cluster,
        "test_results": test_results,
    }


def _print_milestone_summary(milestone_id: int | str, rec: dict[str, Any]) -> None:
    msg = (
        f"\n[Milestone {milestone_id}] "
        f"PassRate={rec['test_pass']}/{rec['test_total']} ({rec['pass_rate']:.2%}) "
        f"AlienAware={rec['alien_aware_pass']}/{rec['alien_aware_total']} "
        f"({rec['alien_aware_pass_rate']:.2%}) "
        f"Minimality={rec['proof_minimality_mean']}"
    )
    if rec.get("unprovable_total"):
        msg += (
            f" Unprovable={rec['unprovable_correct']}/{rec['unprovable_total']}"
            f" ({rec['unprovable_recognition_rate']:.2%})"
        )
    print(msg)
    for role, b in sorted(rec["pass_by_role"].items()):
        print(f"  by_role/{role}: {b['pass']}/{b['total']} "
              f"({b['pass'] / b['total']:.2%})")
    for cl, b in sorted(rec["pass_by_cluster"].items()):
        print(f"  by_cluster/{cl}: {b['pass']}/{b['total']} "
              f"({b['pass'] / b['total']:.2%})")


def compute_evolution_metrics(milestones: list[dict[str, Any]]) -> dict[str, Any]:
    by_idx: dict[Any, dict[str, Any]] = {m["milestone"]: m for m in milestones}
    pre = by_idx.get("pre")
    m0 = by_idx.get(0)
    final_idx = max((m for m in by_idx if isinstance(m, int)), default=None)
    final = by_idx.get(final_idx) if final_idx is not None else None
    explore = [m for m in milestones
               if isinstance(m["milestone"], int) and m["milestone"] >= 1]
    best = max(explore, key=lambda m: m["pass_rate"], default=None)

    def safe(m, k): return None if m is None else m.get(k)

    return {
        "PassRatePre": safe(pre, "pass_rate"),
        "PassRateM0": safe(m0, "pass_rate"),
        "PassRateFinal": safe(final, "pass_rate"),
        "PassRateBest": safe(best, "pass_rate"),
        "BestMilestone": best["milestone"] if best else None,
        "SeedGain": (
            safe(m0, "pass_rate") - safe(pre, "pass_rate")
            if safe(m0, "pass_rate") is not None and safe(pre, "pass_rate") is not None
            else None
        ),
        "ExploreGainFinal": (
            safe(final, "pass_rate") - safe(m0, "pass_rate")
            if safe(final, "pass_rate") is not None and safe(m0, "pass_rate") is not None
            else None
        ),
        "ExploreGainBest": (
            safe(best, "pass_rate") - safe(m0, "pass_rate")
            if safe(best, "pass_rate") is not None and safe(m0, "pass_rate") is not None
            else None
        ),
        "AlienAwareFinal": safe(final, "alien_aware_pass_rate"),
        "AlienAwareGain": (
            safe(final, "alien_aware_pass_rate") - safe(m0, "alien_aware_pass_rate")
            if safe(final, "alien_aware_pass_rate") is not None
            and safe(m0, "alien_aware_pass_rate") is not None
            else None
        ),
        "UnprovableRecognitionPre": safe(pre, "unprovable_recognition_rate"),
        "UnprovableRecognitionM0": safe(m0, "unprovable_recognition_rate"),
        "UnprovableRecognitionFinal": safe(final, "unprovable_recognition_rate"),
        "UnprovableRecognitionGain": (
            safe(final, "unprovable_recognition_rate") - safe(m0, "unprovable_recognition_rate")
            if safe(final, "unprovable_recognition_rate") is not None
            and safe(m0, "unprovable_recognition_rate") is not None
            else None
        ),
        "MinimalityFinal": safe(final, "proof_minimality_mean"),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Top-level run
# ═══════════════════════════════════════════════════════════════════════

def _default_agent_trace_path(episode_id: str) -> str:
    short = MODEL_SHORT or MODEL.rsplit("_", 1)[-1]
    run_suffix = f"_{RUN_ID}" if RUN_ID else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(
        _OUT_DIR,
        f"agent_trace_alien_logic_{ts}_{short}_{episode_id}{run_suffix}.jsonl",
    )


def _build_client_config(snapshot_dir: str) -> AgentClientConfig:
    """Transport settings for this run, shared by a fresh run and by a redo.

    Per-request knobs (thinking, max_tokens) come from `_chat_kwargs`, which
    keys off the model; only the transport settings are read here.
    """

    config = AgentClientConfig(
        model=MODEL,
        provider=_LOCAL_SETTINGS.get("provider") or None,
        # Same precedence as AlienCode: a launcher's variables win over the
        # local file, so one command sets the deadline for both sandboxes.
        timeout=float(os.environ.get(
            "EVAL_HTTP_TIMEOUT", _LOCAL_SETTINGS.get("timeout", 600))),
        max_retries=int(os.environ.get(
            "EVAL_MAX_RETRIES", _LOCAL_SETTINGS.get("max_retries", 4))),
        max_empty_retries=int(_LOCAL_SETTINGS.get("max_empty_retries", 2)),
        trace_path=_TRANSCRIPT_PATH,
        snapshot_dir=snapshot_dir,
        settings={},
    )
    # Seed plus every probe is re-sent by each later call, and each milestone
    # replays that history once per held-out theorem, so the prefix is worth
    # caching for the length of a run. The two protocols ask for it
    # differently: Anthropic needs an explicit breakpoint, while the Responses
    # API caches long prefixes itself and only wants a stable routing key.
    if config.provider.value == "anthropic":
        # Named TTLs are an Anthropic extension; a vendor borrowing the
        # protocol only knows the bare ephemeral breakpoint.
        config.settings["cache_breakpoints"] = (
            "default" if config.standard_messages
            else str(_LOCAL_SETTINGS.get("cache_ttl", "1h"))
        )
    elif config.provider.value == "openai_responses":
        # GatewayA's own Responses upstreams match on the prefix itself and reject
        # the routing key OpenAI wants, so only send it to OpenAI.
        if not config.responses_route:
            config.settings["prompt_cache_key"] = (
                f"alienlogic:{MODEL_SHORT or MODEL}:{RUN_ID or 'default'}"
            )
    return config


def build_runtime_for_redo(*, snapshot: dict[str, Any],
                           snapshot_dir: str) -> AgentSession:
    """Restore a finished run's session so single theorems can be re-asked.

    Same client config the run used -- cache key, timeouts, retries -- so a
    replayed question is charged and cached the way the original was.
    """

    global _AGENT_CLIENT, _AGENT_RUNTIME
    config = _build_client_config(snapshot_dir)
    state = SessionState.from_dict(snapshot)
    if state.model != MODEL:
        raise ValueError(
            f"snapshot model {state.model!r} does not match current {MODEL!r}"
        )
    runtime = build_runtime(
        framework=FRAMEWORK,
        track=EVAL_TRACK,
        config=config,
        trace_store=JsonlTraceStore(_TRANSCRIPT_PATH, snapshot_dir=snapshot_dir),
        budget_profile=BudgetProfile.from_name(BUDGET_PROFILE),
    )
    _AGENT_RUNTIME = runtime
    _AGENT_CLIENT = runtime.client
    return AgentSession.from_state(runtime.client, state)


def run_eval(episode: EpisodeSpec, oracle_only: bool = False) -> dict[str, Any]:
    global _AGENT_CLIENT, _AGENT_RUNTIME, _TRANSCRIPT_PATH
    if _TRANSCRIPT_PATH is None:
        _TRANSCRIPT_PATH = _default_agent_trace_path(episode.id)
    snapshot_dir = os.path.splitext(_TRANSCRIPT_PATH)[0] + "_snapshots"
    client_config = _build_client_config(snapshot_dir)
    trace_store = JsonlTraceStore(
        _TRANSCRIPT_PATH,
        snapshot_dir=snapshot_dir,
    )
    runtime = build_runtime(
        framework=FRAMEWORK,
        track=EVAL_TRACK,
        config=client_config,
        trace_store=trace_store,
        budget_profile=BudgetProfile.from_name(BUDGET_PROFILE),
    )
    _AGENT_RUNTIME = runtime
    _AGENT_CLIENT = runtime.client
    resume = _load_checkpoint() if RESUME else None
    if resume:
        reason = _resume_is_usable(resume)
        if reason:
            print(f"[Checkpoint] 忽略检查点（{reason}），从头开始", flush=True)
            resume = None
        else:
            print(f"[Checkpoint] 从 {resume.get('stage')} 之后续跑 "
                  f"（已完成 {len(resume.get('completed') or [])} 个阶段）",
                  flush=True)

    if resume:
        with open(resume["root_snapshot"], encoding="utf-8") as handle:
            root_session = AgentSession.from_state(
                runtime.client, SessionState.from_dict(json.load(handle)))
    else:
        system = SYSTEM_PROMPT_TEMPLATE.format(manual=REFERENCE_MANUAL)
        if PROTOCOL_V2:
            # Immutable context, exactly as the code sandbox delivers its
            # demos: the evidence is part of what every system is given, not
            # something a seed conversation produced and could vary.
            system += (
                "\n\n─────────────────────────────────────────\n"
                + protocol_v2.demo_block(_demo_checker(episode),
                                        episode.seed_examples)
                + "\n─────────────────────────────────────────\n"
            )
        root_session = runtime.create_session(system=system)
        root_session.snapshot()
    state: dict[str, Any] = {
        "session": root_session,
        "pending_feedback": resume.get("pending_feedback") if resume else None,
        "total_probes_used": (
            resume.get("total_probes_used", 0) if resume else 0),
    }
    seed_records: list[dict[str, Any]] = (
        list(resume.get("seed_records") or []) if resume else [])
    explore_records: list[dict[str, Any]] = (
        list(resume.get("explore_records") or []) if resume else [])
    milestones: list[dict[str, Any]] = (
        list(resume.get("milestones") or []) if resume else [])

    episode = _apply_v2_task_set(episode)

    print(f"\n[Episode] {episode.id} (level={episode.level})")
    print(f"[Episode Description] {episode.description}")
    print(f"[Active alien rules] {episode.alien_rules}")
    print(f"[Held-out count] {len(episode.heldout_theorems)}")
    print(f"[Standard rule pool] {STANDARD_RULES}")
    print(f"[Probe budget] {MAX_TOTAL_PROBES} (max {MAX_PROBES_PER_ROUND}/round)")
    if oracle_only:
        print("[ORACLE MODE] skipping pre-baseline + explore loops; will "
              "inject GT alien rules after seed and run a single 'oracle' "
              "milestone.")
    elif CONTROL_MODE != "self":
        print(f"[CONTROL MODE] {CONTROL_MODE}"
              + (f" bank={CONTROL_BANK_PATH}" if CONTROL_BANK_PATH else ""))

    print("\n[Validating reference proofs...]")
    validate_episode(episode)
    print("[Validation OK]")

    deferred: list[dict[str, Any]] | None = [] if DEFER_HELDOUT else None
    if DEFER_HELDOUT:
        print(f"[Defer] held-out testing deferred to merged pool "
              f"(workers={_defer_workers()})")
        if resume and resume.get("deferred"):
            # Each entry lost its live fork on the way through JSON. The fork
            # never advances after its milestone, so the snapshot it already
            # wrote reconstitutes it exactly.
            for entry in resume["deferred"]:
                if SKIP_PRE and entry.get("milestone_id") == "pre":
                    # A checkpoint written before the switch was set still
                    # carries the pre-baseline, and its held-out batch has not
                    # been spent yet. Dropping it here is what makes the
                    # switch worth setting on a resume.
                    print("  [Checkpoint] 丢弃已排队的探索前基线", flush=True)
                    continue
                path = entry.get("base_snapshot")
                if not path or not os.path.exists(path):
                    print(f"  [Checkpoint] M{entry.get('milestone_id')} "
                          f"缺基线快照，该里程碑将重跑", flush=True)
                    continue
                with open(path, encoding="utf-8") as handle:
                    entry = dict(entry)
                    entry["base"] = AgentSession.from_state(
                        runtime.client,
                        SessionState.from_dict(json.load(handle)))
                deferred.append(entry)
            print(f"[Checkpoint] 恢复 {len(deferred)} 个待评里程碑", flush=True)

    if oracle_only:
        # Oracle-only design (Wave 1): seed → GT rules → single 'oracle'
        # milestone. Measures pure exploitation given full rule disclosure.
        _require_agent_runtime().begin_phase(PhaseContext(
            sandbox="logic", phase="seed", label="Seed",
            track=EVAL_TRACK, framework=FRAMEWORK))
        run_seed(state, episode, seed_records)
        from oracle_rules import build_oracle_message
        oracle_msg = build_oracle_message(episode.alien_rules)
        # Brief ack turn so the assistant has "seen" the oracle in context.
        ack, _ = _chat_with_empty_retry(
            root_session,
            oracle_msg,
            label="Oracle Ack",
            max_tokens=SUMMARY_MAX_TOKENS,
        )
        ack_display = ack if ack.strip() else "（模型返回了空回复）"
        print(f"\n[Oracle Ack]\n{_trim(ack_display)}")
        _require_agent_runtime().begin_phase(PhaseContext(
            sandbox="logic", phase="milestone", label="oracle",
            track=EVAL_TRACK, framework=FRAMEWORK))
        run_milestone(state, episode, "oracle", milestones, deferred)
    else:
        def checkpoint(stage: str) -> None:
            _save_checkpoint(_checkpoint_payload(
                stage, state, seed_records, explore_records, milestones,
                deferred,
            ))

        if SKIP_PRE:
            print("[Pre] 跳过探索前基线，从 M0 开始计分")
        elif not _stage_done(resume, "pre"):
            _require_agent_runtime().begin_phase(PhaseContext(
                sandbox="logic", phase="pre_baseline", label="Pre",
                track=EVAL_TRACK, framework=FRAMEWORK))
            run_pre_baseline(state, episode, milestones, deferred)
            checkpoint("pre")
        if not _stage_done(resume, "seed"):
            _require_agent_runtime().begin_phase(PhaseContext(
                sandbox="logic", phase="seed", label="Seed",
                track=EVAL_TRACK, framework=FRAMEWORK))
            run_seed(state, episode, seed_records)
            checkpoint("seed")
        if not _stage_done(resume, "m0"):
            _require_agent_runtime().begin_phase(PhaseContext(
                sandbox="logic", phase="milestone", label="M0",
                track=EVAL_TRACK, framework=FRAMEWORK))
            run_milestone(state, episode, 0, milestones, deferred)
            checkpoint("m0")
        for loop in range(1, N_EXPLORE_LOOPS + 1):
            if not _stage_done(resume, f"explore{loop}"):
                if CONTROL_MODE == "none":
                    print(f"\n[Control none] skipping explore loop {loop}; "
                          "session remains unchanged")
                else:
                    _require_agent_runtime().begin_phase(PhaseContext(
                        sandbox="logic", phase="explore",
                        label=f"Explore {loop}",
                        track=EVAL_TRACK, framework=FRAMEWORK))
                    run_explore(state, episode, loop, explore_records)
                checkpoint(f"explore{loop}")
            if not _stage_done(resume, f"m{loop}"):
                _require_agent_runtime().begin_phase(PhaseContext(
                    sandbox="logic", phase="milestone", label=f"M{loop}",
                    track=EVAL_TRACK, framework=FRAMEWORK))
                run_milestone(state, episode, loop, milestones, deferred)
                checkpoint(f"m{loop}")

    if deferred is not None:
        _flush_deferred(deferred, episode, milestones)

    aggregate = compute_evolution_metrics(milestones)
    # The harness stamps any shortfall at all; the analysis decides later how
    # much it is willing to tolerate.
    validity = derive_validity({"milestones": milestones}, tolerance=0)
    if not validity["ok"]:
        print(
            "\n" + "!" * 72
            + f"\n  [Damaged run] {RUN_ID or MODEL_SHORT}: "
            + "; ".join(validity["reasons"])
            + "\n  结果仍会写盘以便修复，但已标记为不可用，分析脚本会跳过它。\n"
            + "!" * 72
        )
    _print_eval_summary(episode, milestones, seed_records, explore_records, aggregate)
    client = _require_agent_client()
    root_snapshot_path = root_session.snapshot()
    agent_trace_path = (
        str(client.trace_store.path) if client.trace_store.path else None
    )
    runtime_state = _require_agent_runtime().snapshot()

    return {
        "model": MODEL,
        "model_short": MODEL_SHORT,
        "run_id": RUN_ID,
        "episode_id": episode.id,
        "alien_rules": episode.alien_rules,
        "framework": FRAMEWORK,
        "track": EVAL_TRACK,
        "framework_version": runtime_state.get("framework_spec", {}),
        "budget_profile": runtime_state.get("budget_profile", {}),
        "artifact_hash": runtime_state.get("artifact_digest"),
        "fork_mode": "closed_book_tools_empty",
        "agent_trace_path": agent_trace_path,
        "root_session_id": root_session.session_id,
        "root_session_snapshot_path": (
            str(root_snapshot_path) if root_snapshot_path else None
        ),
        "root_session_history_event_count": len(root_session.history),
        "root_session_provider_history_item_count": len(
            root_session.provider_history
        ),
        "root_session_last_response_id": root_session.last_response_id,
        "agent_runtime_state": runtime_state,
        "config": {
            "tool_mode": TOOL_MODE,
            "n_explore_loops": N_EXPLORE_LOOPS,
            "max_probes_per_round": MAX_PROBES_PER_ROUND,
            "max_total_probes": MAX_TOTAL_PROBES,
            "default_max_tokens": DEFAULT_MAX_TOKENS,
            "summary_max_tokens": SUMMARY_MAX_TOKENS,
            "test_max_tokens": TEST_MAX_TOKENS,
            "api_protocol": client.config.provider.value,
            "framework": FRAMEWORK,
            "track": EVAL_TRACK,
            "budget_profile": BUDGET_PROFILE,
            "reasoning_effort": _requested_effort(),
            "reasoning_effort_asked_for": os.environ.get(
                "EVAL_REASONING_EFFORT"),
            "reasoning_mode": os.environ.get(
                "EVAL_REASONING_MODE", "standard"),
            "n_seeds": len(episode.seed_examples),
            "n_heldouts": len(episode.heldout_theorems),
            "defer_tests": DEFER_HELDOUT,
            "skip_pre": SKIP_PRE,
            "oracle_only": oracle_only,
            "control_mode": CONTROL_MODE,
            "control_bank": CONTROL_BANK_PATH or None,
            "control_preflight": _CONTROL_PREFLIGHT,
        },
        "seed_records": seed_records,
        "explore_records": explore_records,
        "milestones": milestones,
        "aggregate_metrics": aggregate,
        "validity": validity,
        "token_usage": client.get_usage(),
    }


def _print_eval_summary(episode, milestones, seed_records, explore_records, aggregate):
    print("\n" + "=" * 72)
    print("  AlienLogic Summary")
    print("=" * 72)
    print(f"Episode: {episode.id}")
    print(f"Active alien rules: {episode.alien_rules}")
    print(f"\nPassRate per milestone:")
    for m in milestones:
        print(f"  M{m['milestone']}: {m['test_pass']}/{m['test_total']} "
              f"({m['pass_rate']:.2%}) "
              f"AlienAware={m['alien_aware_pass']}/{m['alien_aware_total']} "
              f"({m['alien_aware_pass_rate']:.2%})")
    print(f"\nEvolution Metrics:")
    for k, v in aggregate.items():
        print(f"  {k}: {v}")
    # Calibration is a property of the interactive seed, where the model
    # predicted each verdict before submitting. v2's demos are shown rather
    # than attempted, so there is no prediction to score and the records
    # carry no label -- reading one crashed the summary after both runs had
    # finished all their real work.
    scored = [r for r in seed_records if "predicted_label" in r]
    if scored:
        seed_correct = sum(
            1 for r in scored
            if (r["predicted_label"] == "ACCEPT") == r["actual_accepted"]
        )
        print(f"\nSeed prediction calibration: "
              f"{seed_correct}/{len(scored)}")
    else:
        accepted = sum(1 for r in seed_records if r.get("accepted"))
        print(f"\nFixed calibration demos: {len(seed_records)} shown "
              f"({accepted} accepted, {len(seed_records) - accepted} refused)")
    n_probes = sum(r["n_probes_accepted_for_eval"] for r in explore_records)
    print(f"Total probes consumed: {n_probes}")


class _Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for f in self.files:
            f.write(text); f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


_CHECKPOINT_DIR = os.path.join(_REPO_ROOT, "logs", "logic", "checkpoints")


def _ckpt_path() -> str:
    short = MODEL_SHORT or MODEL.rsplit("_", 1)[-1]
    suffix = f"_{RUN_ID}" if RUN_ID else ""
    return os.path.join(_CHECKPOINT_DIR, f"ckpt_alien_logic_{short}{suffix}.json")


def _save_checkpoint(data: dict[str, Any]) -> None:
    """Record enough to resume at the next phase boundary.

    A milestone is the unit: the live session is snapshotted anyway so a redo
    can re-ask from it, and the records accumulated so far are plain data. What
    cannot be carried across processes is the session object itself, so the
    checkpoint stores the snapshot path and the resume rebuilds from it.
    """

    try:
        os.makedirs(_CHECKPOINT_DIR, exist_ok=True)
        path = _ckpt_path()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001 - a lost checkpoint is not fatal
        print(f"  [Checkpoint Warning] 保存失败: {exc}", flush=True)


def _load_checkpoint() -> dict[str, Any] | None:
    path = _ckpt_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # noqa: BLE001
        print(f"  [Checkpoint Warning] 读取失败，按全新运行处理: {exc}", flush=True)
        return None


def _clear_checkpoint() -> None:
    path = _ckpt_path()
    try:
        if os.path.exists(path):
            os.remove(path)
            print("  [Checkpoint] 评测完成，已清除检查点", flush=True)
    except OSError:
        pass


def _stage_done(resume: dict[str, Any] | None, stage: str) -> bool:
    if not resume:
        return False
    done = resume.get("completed") or []
    return stage in done


def _checkpoint_payload(stage: str, state: dict[str, Any],
                        seed_records: list[dict[str, Any]],
                        explore_records: list[dict[str, Any]],
                        milestones: list[dict[str, Any]],
                        deferred: list[dict[str, Any]] | None,
                        ) -> dict[str, Any]:
    """Everything the next process needs, with sessions reduced to paths.

    `deferred` carries a live closed-book fork per milestone so the merged pool
    can answer from it later. A fork cannot cross a process boundary, but it
    was already snapshotted for the redo tooling, so the path stands in for it
    and the resume rebuilds the fork from that file.
    """

    session = state.get("session")
    try:
        root_snapshot = str(session.snapshot()) if session else None
    except Exception as exc:  # noqa: BLE001
        print(f"  [Checkpoint Warning] 会话快照失败: {exc}", flush=True)
        root_snapshot = None
    packed_deferred = None
    if deferred is not None:
        packed_deferred = [
            {k: v for k, v in entry.items() if k != "base"}
            for entry in deferred
        ]
    previous = _load_checkpoint() or {}
    completed = list(previous.get("completed") or [])
    if stage not in completed:
        completed.append(stage)
    return {
        "schema": 1,
        "run_id": RUN_ID,
        "model": MODEL,
        "model_short": MODEL_SHORT,
        "control_mode": CONTROL_MODE,
        "explore_loops": N_EXPLORE_LOOPS,
        "completed": completed,
        "stage": stage,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "agent_trace_path": _TRANSCRIPT_PATH,
        "root_snapshot": root_snapshot,
        "pending_feedback": state.get("pending_feedback"),
        "total_probes_used": state.get("total_probes_used", 0),
        "seed_records": seed_records,
        "explore_records": explore_records,
        "milestones": milestones,
        "deferred": packed_deferred,
    }


def _resume_is_usable(resume: dict[str, Any]) -> str | None:
    """Why this checkpoint cannot be used, or None when it can."""

    if resume.get("model") != MODEL:
        return f"模型不符 {resume.get('model')!r} != {MODEL!r}"
    if resume.get("control_mode") != CONTROL_MODE:
        return (f"控制模式不符 {resume.get('control_mode')!r} "
                f"!= {CONTROL_MODE!r}")
    if int(resume.get("explore_loops") or 0) != N_EXPLORE_LOOPS:
        return "探索轮数不符"
    snapshot = resume.get("root_snapshot")
    if not snapshot or not os.path.exists(snapshot):
        return "会话快照缺失"
    return None


def _retire_run_artifacts(stem: str, *paths: str) -> str | None:
    """Set aside a previous run's files before a new one reuses its id.

    Only for a launch that starts over. A resume deliberately keeps the earlier
    trace and appends to it, since the two halves are one run; retiring them
    would strand the part already paid for.
    """

    candidates = list(paths) + [
        os.path.splitext(paths[-1])[0] + "_snapshots" if paths else ""
    ]
    existing = [path for path in candidates if path and os.path.exists(path)]
    if not existing:
        return None
    attic = os.path.join(
        _REPO_ROOT, "logs", "logic", "_superseded",
        f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(attic, exist_ok=True)
    for path in existing:
        shutil.move(path, os.path.join(attic, os.path.basename(path)))
    return attic


def main() -> None:
    global MODEL, MODEL_SHORT, EPISODE_ID, N_EXPLORE_LOOPS, RUN_ID, DEFER_HELDOUT
    global CONTROL_MODE, CONTROL_BANK_PATH, CONTROL_MANIFEST_PATH
    global FRAMEWORK, EVAL_TRACK, BUDGET_PROFILE, RESUME, SKIP_PRE

    parser = argparse.ArgumentParser(description="AlienLogic evaluation")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--model-short", default=MODEL_SHORT)
    parser.add_argument("--episode", default=EPISODE_ID)
    parser.add_argument("--explore-loops", type=int, default=N_EXPLORE_LOOPS)
    parser.add_argument(
        "--run-id", default=RUN_ID,
        help="Per-run tag appended to result/log filenames (e.g. l1/l2/l3). "
             "Empty = single-run mode, no suffix.",
    )
    parser.add_argument(
        "--defer-tests", action="store_true", default=DEFER_HELDOUT,
        help="Design A: defer all held-out tests to a single merged pool run "
             "after exploration (score-equivalent, much higher concurrency). "
             "Also enabled by ALIENLOGIC_DEFER_TESTS=1.",
    )
    parser.add_argument(
        "--skip-pre", action="store_true", default=SKIP_PRE,
        help="Skip the pre-exploration baseline and start scoring at M0. It "
             "costs a full held-out batch and only SeedGain (M0 minus Pre) "
             "reads it; M0--M4 are unaffected. Also enabled by EVAL_SKIP_PRE=1.",
    )
    parser.add_argument(
        "--oracle-only", action="store_true",
        default=os.environ.get("EVAL_ORACLE_ONLY", "0").lower() in ("1","true","yes"),
        help="Oracle intervention (Wave 1, Design A): skip pre-baseline + "
             "explore loops; after seed, inject GT alien-rule descriptions and "
             "run one milestone tagged 'oracle'. Also enabled by "
             "EVAL_ORACLE_ONLY=1. Compare to M0/M4 in analysis.",
    )
    parser.add_argument(
        "--control-mode", choices=sorted(_VALID_CONTROL_MODES),
        default=CONTROL_MODE,
        help="Causal control: self, none, passive donor replay, or random probes",
    )
    parser.add_argument(
        "--control-bank", default=CONTROL_BANK_PATH,
        help="Standard-run transcript JSONL used by --control-mode passive",
    )
    parser.add_argument(
        "--control-manifest", default=CONTROL_MANIFEST_PATH,
        help="Optional donor provenance manifest; mismatches fail preflight",
    )
    parser.add_argument(
        "--framework", default=os.environ.get("EVAL_FRAMEWORK", FRAMEWORK),
        help="Framework adapter: baseline, reflexion, ace, evotest, gepa, agent_factory",
    )
    parser.add_argument(
        "--track", choices=("controlled", "native", "open"),
        default=os.environ.get("EVAL_TRACK", EVAL_TRACK),
        help="Framework evaluation track",
    )
    parser.add_argument(
        "--budget-profile",
        default=os.environ.get("EVAL_BUDGET_PROFILE", BUDGET_PROFILE),
        help="Framework budget profile: c1/1x/2x/4x",
    )
    parser.add_argument(
        "--resume", action="store_true", default=RESUME,
        help="continue this run id from its last completed phase instead of "
             "starting over; without it an existing checkpoint is ignored",
    )
    args = parser.parse_args()

    MODEL = args.model
    MODEL_SHORT = args.model_short or (MODEL.rsplit("_", 1)[-1] if MODEL else "")
    EPISODE_ID = args.episode
    N_EXPLORE_LOOPS = args.explore_loops
    RUN_ID = args.run_id
    _RUN_LOCK_HANDLE = acquire_run_lock(RUN_ID)
    RESUME = args.resume
    DEFER_HELDOUT = args.defer_tests
    if DEFER_HELDOUT and _streams_heldout(MODEL, MODEL_SHORT):
        DEFER_HELDOUT = False
        print("[Defer] 该厂商把推理状态存在自己那边，response id 会过期；"
              "改为每个里程碑就地判分，避免整池失效。", flush=True)
    SKIP_PRE = args.skip_pre
    CONTROL_MODE = args.control_mode
    CONTROL_BANK_PATH = args.control_bank
    CONTROL_MANIFEST_PATH = args.control_manifest
    FRAMEWORK = args.framework
    EVAL_TRACK = args.track
    BUDGET_PROFILE = args.budget_profile
    os.environ["EVAL_FRAMEWORK"] = FRAMEWORK
    os.environ["EVAL_TRACK"] = EVAL_TRACK
    os.environ["EVAL_BUDGET_PROFILE"] = BUDGET_PROFILE
    if CONTROL_MODE in {"passive", "random"} and not CONTROL_BANK_PATH:
        parser.error(f"--control-mode {CONTROL_MODE} requires --control-bank "
                     "to pair the probe count with a self run")
    if args.oracle_only and CONTROL_MODE != "self":
        parser.error("--oracle-only cannot be combined with a non-self control mode")

    if not MODEL:
        raise SystemExit("Please pass --model or set ALIENLOGIC_MODEL.")

    episode = get_episode(EPISODE_ID)
    _preflight_control_bank(episode)

    os.makedirs(_LOG_DIR, exist_ok=True)
    os.makedirs(_OUT_DIR, exist_ok=True)
    os.makedirs(_TRACE_DIR, exist_ok=True)
    # No launch timestamp: a run id owns one log, one trace and one results
    # file, and relaunching under that id sets the earlier set aside rather
    # than minting a parallel one.
    short = MODEL_SHORT or MODEL.rsplit("_", 1)[-1]
    run_suffix = f"_{RUN_ID}" if RUN_ID else ""
    stem = f"alien_logic_{short}_{EPISODE_ID}{run_suffix}"
    log_path = os.path.join(_LOG_DIR, f"eval_{stem}.log")
    out_path = os.path.join(_OUT_DIR, f"eval_results_{stem}.json")

    global _TRANSCRIPT_PATH
    _TRANSCRIPT_PATH = os.path.join(_TRACE_DIR, f"agent_trace_{stem}.jsonl")
    # A resume is the same run continuing, so its trace and log stay in place
    # and get appended to; retiring them would strand the half already paid for.
    resuming = RESUME and _load_checkpoint() is not None
    retired = (
        None if resuming
        else _retire_run_artifacts(stem, log_path, out_path, _TRANSCRIPT_PATH)
    )
    if resuming:
        print(f"[Checkpoint] 续跑 {stem}，保留既有轨迹与日志", flush=True)

    original_stdout = sys.stdout
    with open(log_path, "a", encoding="utf-8") as log_f:
        log_f.write(
            f"\n{'=' * 72}\n"
            f"[start] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  "
            f"model={MODEL}  run={RUN_ID or '-'}  episode={EPISODE_ID}\n"
            f"{'=' * 72}\n"
        )
        log_f.flush()
        sys.stdout = _Tee(original_stdout, log_f)  # type: ignore[assignment]
        if retired:
            print(f"[Superseded] 同名 run 的旧产物已移至 {retired}")
        try:
            results = run_eval(episode, oracle_only=args.oracle_only)
            with open(out_path, "w", encoding="utf-8") as out_f:
                json.dump(results, out_f, ensure_ascii=False, indent=2)
            print(f"\nSaved results: {out_path}")
            _clear_checkpoint()
            # The journal only exists to survive a crash; the saved result now
            # supersedes it, and leaving it would let a re-run of this id
            # inherit the previous run's verdicts.
            ledger = _heldout_ledger_path()
            if os.path.exists(ledger):
                os.replace(ledger, ledger + ".done")
            if _TRANSCRIPT_PATH and os.path.exists(_TRANSCRIPT_PATH):
                n_turns = n_cot = 0
                with open(_TRANSCRIPT_PATH, "r", encoding="utf-8") as tf:
                    for line in tf:
                        if not line.strip():
                            continue
                        try:
                            rec = json.loads(line)
                            if rec.get("record_type") != "trajectory":
                                continue
                            n_turns += 1
                            if int(rec.get("reasoning_len", 0) or 0) > 0:
                                n_cot += 1
                        except Exception:
                            pass
                size_mb = os.path.getsize(_TRANSCRIPT_PATH) / 1e6
                print(f"轨迹已保存到 {_TRANSCRIPT_PATH}  "
                      f"({n_turns} 轮, {n_cot} 含CoT, {size_mb:.1f} MB)")
            export_run_artifacts(
                _TRANSCRIPT_PATH, os.path.join(_REPO_ROOT, "logs", "logic")
            )
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()
