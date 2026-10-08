'''
run_eval.py
==============

ExplorationBench — AlienCode harness (world data: sandboxes/code/world_data.py)

评估 LLM 在未知 AlienCode 环境中的三大核心能力:
  1. 探索力 (Exploration)   — 系统性发现隐藏规则偏差
  2. 总结力 (Summarization)  — 里程碑式归纳规则 (S-Expression)
  3. 应用力 (Application)    — 在扭曲规则下编写正确程序（含高难度算法）

执行流程:
  1. 播种轮 (Seed Phase): S01-S10, 10 轮，每轮给反馈
  2. Milestone 0 (基线): S-Expr 规则总结 + 80 道测试题 (无反馈)
  3. for i = 1..4:
       a. 自由探索 3 轮 (有执行反馈)
       b. Milestone i: S-Expr 规则总结 + 80 道测试题 (无反馈)

运行方式:
  python sandboxes/code/run_eval.py
'''
from __future__ import annotations
global _TRANSCRIPT_PATH, _CONTROL_MODE, _CONTROL_BANK_PATH, _CONTROL_BANK_CACHE, _CONTROL_PREFLIGHT, _PROMPT_VARIANT, _PROMPT_VARIANT, _PROMPT_VARIANT, _EXPLICIT_GENERALIZATION, _HIDDEN_TESTS_ONLY, _ORACLE_ONLY, _CONTROL_MODE, _CONTROL_BANK_PATH

import collections
import hashlib
import json
import os
import random
import re
import shutil
import socket
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import NamedTuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_DATA_DIR = os.path.join(os.path.dirname(_PROJECT_ROOT), 'logs', 'code')
_RESULTS_DIR = os.path.join(_DATA_DIR, 'results')
_LOG_DIR = os.path.join(_DATA_DIR, 'logs')
_TRACE_DIR = os.path.join(_DATA_DIR, 'traces')
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common import local_config
from common.agent_client import (
    AgentClient,
    AgentClientConfig,
    AgentHTTPError,
    AgentSession,
    JsonlTraceStore,
    SessionState,
    ToolResult,
)
from common.agent_client.types import normalize_tools
from common.agent_client.routing import (
    STANDARD_MESSAGES_PREFIX,
    gateway_a_vendor,
    chat_route_from_model,
    legacy_route,
    responses_route_from_model,
    same_model,
)
from common.agent_runtime import BudgetProfile, PhaseContext
from common.client import chat
from common.run_artifacts import export_run_artifacts
from common.run_lock import acquire_run_lock
from common.run_validity import derive as derive_validity
from frameworks.registry import build_runtime

# Credentials and run knobs live in eval.local.toml; anything already exported
# in the environment wins, so CI and one-off overrides keep working.
_LOCAL_SETTINGS = local_config.load()









from sandboxes.code.execution import (
    ALIEN_RULE_SPECS,
    FUNC_KEYWORD_MAP,
    REFERENCE_MANUAL,
    alien_exec,
)
from sandboxes.code import protocol_v2
from sandboxes.code import world_data















_SAVE_TRANSCRIPT = os.environ.get('EVAL_SAVE_TRANSCRIPT', '') not in ('', '0', 'false', 'False')
_TRANSCRIPT: list[dict] = []
_TRANSCRIPT_LOCK = threading.Lock()
_TRANSCRIPT_PATH = None
_AGENT_CLIENT: AgentClient | None = None
_AGENT_RUNTIME = None
_AGENT_TRACE_PATH: str | None = None
_AGENT_SNAPSHOT_DIR: str | None = None
FRAMEWORK = os.environ.get(
    'EVAL_FRAMEWORK', str(_LOCAL_SETTINGS.get('framework', 'baseline')))
EVAL_TRACK = os.environ.get(
    'EVAL_TRACK', str(_LOCAL_SETTINGS.get('track', 'controlled')))
BUDGET_PROFILE = os.environ.get(
    'EVAL_BUDGET_PROFILE', str(_LOCAL_SETTINGS.get('budget_profile', 'c1')))






_LAST_COT = threading.local()


def _take_last_cot() -> str:
    '''Pop the CoT of the latest _chat() on this thread; empty after one read.

    Pop rather than peek so a stale CoT can never be attached to a turn that did
    not come from that call (e.g. checkpoint-resumed turns, or a judge call
    landing between a chat and its append).
    '''
    cot = getattr(_LAST_COT, 'text', '') or ''
    _LAST_COT.text = ''
    return cot


def _assistant_turn(content: str) -> dict:
    '''History entry for a just-received assistant reply.'''
    msg = {'role': 'assistant', 'content': content}
    cot = _take_last_cot()
    if cot:
        msg['reasoning_content'] = cot
    return msg


def _transcript_path():
    short = MODEL_SHORT or MODEL.rsplit('_', 1)[-1]
    suffix = f'''_{RUN_ID}''' if RUN_ID else ''
    return os.path.join(_RESULTS_DIR, f'''transcript_alien_code_{short}{suffix}.jsonl''')


def _init_transcript(resuming: bool):
    '''Resolve the transcript path; truncate on a fresh run, keep it on resume.'''
    global _TRANSCRIPT_PATH
    if not _SAVE_TRANSCRIPT:
        return None
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    _TRANSCRIPT_PATH = _transcript_path()
    if not resuming:
        open(_TRANSCRIPT_PATH, 'w', encoding='utf-8').close()

MODEL = ''
MODEL_SHORT = ''
RUN_ID = ''
JUDGE_MODEL = 'api_azure_openai_gpt-5.2'


def _run_stem() -> str:
    '''The name every artifact of this run shares.

    No launch timestamp: one run id owns one log, one trace and one snapshot
    directory for its whole life, so picking a run up from a checkpoint keeps
    writing to the files it already has. Runs are told apart by their id, which
    the orchestrator numbers per repeat.
    '''
    short = MODEL_SHORT or MODEL.rsplit('_', 1)[-1]
    suffix = f'_{RUN_ID}' if RUN_ID else ''
    return f'alien_code_{short}{suffix}'


def _log_path() -> str:
    return os.path.join(_LOG_DIR, f'eval_{_run_stem()}.log')


def _retire_run_artifacts() -> str | None:
    '''Move a previous run's files aside before a fresh one reuses its id.

    Artifacts are keyed by run id now, and appending is what lets a resumed run
    continue its own log. A fresh start under an id that already has files is
    the one case where that would silently interleave two different runs, so
    the older set is set aside intact instead.
    '''
    stem = _run_stem()
    short = MODEL_SHORT or MODEL.rsplit('_', 1)[-1]
    suffix = f'_{RUN_ID}' if RUN_ID else ''
    result_path = os.path.join(
        _RESULTS_DIR, f'eval_results_alien_code_{short}{suffix}.json')
    existing = [
        path
        for path in (
            _log_path(),
            _default_agent_trace_path(),
            os.path.splitext(_default_agent_trace_path())[0] + '_snapshots',
            _ckpt_path(),
            _heldout_ledger_path('heldout'),
            _heldout_ledger_path('oracle'),
            result_path,
            result_path + '.tmp',
            _transcript_path(),
        )
        if os.path.exists(path)
    ]
    if not existing:
        return None
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    attic = os.path.join(_DATA_DIR, '_superseded', f'{stem}_{stamp}')
    os.makedirs(attic, exist_ok=True)
    for path in existing:
        shutil.move(path, os.path.join(attic, os.path.basename(path)))
    return attic


def _default_agent_trace_path() -> str:
    return os.path.join(_TRACE_DIR, f'agent_trace_{_run_stem()}.jsonl')


def _init_agent_runtime(checkpoint: dict | None) -> AgentSession:
    global _AGENT_CLIENT, _AGENT_RUNTIME, _AGENT_TRACE_PATH, _AGENT_SNAPSHOT_DIR

    saved_state_value = (
        checkpoint.get('agent_session_state') if checkpoint else None
    )
    saved_state = (
        SessionState.from_dict(saved_state_value)
        if isinstance(saved_state_value, dict)
        else None
    )
    if saved_state is not None:
        if not same_model(saved_state.model, MODEL):
            raise ValueError(
                f'checkpoint model {saved_state.model!r} does not match '
                f'current model {MODEL!r}')
        expected_provider = AgentClientConfig(model=MODEL).provider.value
        if saved_state.provider != expected_provider:
            raise ValueError(
                f'checkpoint provider {saved_state.provider!r} does not '
                f'match current provider {expected_provider!r}')

    configured_trace = (
        checkpoint.get('agent_trace_path') if checkpoint else None
    )
    if not configured_trace and saved_state is not None:
        configured_trace = saved_state.config.get('trace_path')
    _AGENT_TRACE_PATH = str(
        os.path.abspath(configured_trace or _default_agent_trace_path())
    )
    configured_snapshots = (
        checkpoint.get('agent_snapshot_dir') if checkpoint else None
    )
    if not configured_snapshots and saved_state is not None:
        configured_snapshots = saved_state.config.get('snapshot_dir')
    _AGENT_SNAPSHOT_DIR = str(os.path.abspath(
        configured_snapshots
        or os.path.splitext(_AGENT_TRACE_PATH)[0] + '_snapshots'
    ))

    # Per-request knobs (thinking, max_tokens) come from _request_overrides,
    # which keys off the model; only the transport settings are read here.
    # The shared file is tuned for long single sessions: 1800s a request and 12
    # retries, which is 4.5 hours before one graded question gives up. A cohort
    # answering 630 questions needs a dead endpoint to surface in minutes, so it
    # can say so per run rather than stalling the whole queue behind one socket.
    config = AgentClientConfig(
        model=MODEL,
        provider=_LOCAL_SETTINGS.get('provider') or None,
        timeout=float(os.environ.get(
            'EVAL_HTTP_TIMEOUT', _LOCAL_SETTINGS.get('timeout', 600))),
        max_retries=int(os.environ.get(
            'EVAL_MAX_RETRIES', _LOCAL_SETTINGS.get('max_retries', 4))),
        max_empty_retries=int(_LOCAL_SETTINGS.get('max_empty_retries', 2)),
        trace_path=_AGENT_TRACE_PATH,
        snapshot_dir=_AGENT_SNAPSHOT_DIR,
        settings={},
    )
    # The explored history is re-sent by every later call, and each milestone
    # replays it once per held-out question, so the prefix is worth caching
    # for the length of a run. The two protocols ask for that differently:
    # Anthropic needs an explicit breakpoint, while the Responses API caches
    # long prefixes on its own and only wants a stable key to route on.
    if config.provider.value == 'anthropic':
        # Named TTLs are an Anthropic extension; a vendor borrowing the
        # protocol only knows the bare ephemeral breakpoint.
        config.settings['cache_breakpoints'] = (
            'default' if config.standard_messages
            else str(_LOCAL_SETTINGS.get('cache_ttl', '1h'))
        )
    elif config.provider.value == 'openai_responses':
        # GatewayA's own Responses upstreams match on the prefix itself and reject
        # the routing key OpenAI wants, so only send it to OpenAI.
        if not config.responses_route:
            # The session layer appends its own id and clamps the result to
            # the length OpenAI accepts, so this only has to be stable.
            config.settings['prompt_cache_key'] = (
                f'aliencode:{MODEL_SHORT or MODEL}:{RUN_ID or "default"}')
    trace_store = JsonlTraceStore(
        _AGENT_TRACE_PATH,
        snapshot_dir=_AGENT_SNAPSHOT_DIR,
    )
    runtime_state = (
        checkpoint.get('agent_runtime_state') if checkpoint else None
    )
    runtime = build_runtime(
        framework=FRAMEWORK,
        track=EVAL_TRACK,
        config=config,
        trace_store=trace_store,
        budget_profile=BudgetProfile.from_name(BUDGET_PROFILE),
    )
    if isinstance(runtime_state, dict):
        runtime.restore_runtime_state(runtime_state)
    _AGENT_RUNTIME = runtime
    _AGENT_CLIENT = runtime.client
    usage_snapshot = checkpoint.get('usage_snapshot') if checkpoint else None
    if isinstance(usage_snapshot, dict):
        _AGENT_CLIENT.usage_ledger.restore_snapshot(usage_snapshot)

    if saved_state is not None:
        session = AgentSession.from_state(_AGENT_CLIENT, saved_state)
        # The saved state carries the tool definitions it was written with, so
        # a resume that crosses a protocol change advertises the old schema to
        # the model while the harness reads the new arguments. One run resumed
        # across the atomic-probe -> natural-code switch spent all 48 of its
        # exploration calls on `target`/`args`/`hypothesis` and delivered zero
        # observations. The protocol is whatever this process implements.
        session.tools = normalize_tools(
            [_alien_tool_definition(MAX_EXPLORE_TESTS)] if TOOL_MODE else None
        )
        session.snapshot()
        return session

    system = _initial_system_prompt()
    if checkpoint and checkpoint.get('conversation'):
        # Legacy checkpoints predate provider-native state. The old assistant
        # reasoning signatures/response IDs do not exist, so exact continuation
        # is impossible; preserve their visible transcript as migration context.
        legacy_messages = [
            message
            for message in checkpoint['conversation']
            if message.get('role') != 'system'
        ]
        if legacy_messages:
            system += (
                '\n\n以下是迁移前检查点中保留的可见会话记录。原生签名不可恢复；'
                '请将其作为此前上下文继续：\n'
                + json.dumps(legacy_messages, ensure_ascii=False)
            )
    # The exploration phases own the tool; summary and held-out test branches
    # drop it via _closed_book_fork so answers cannot be executed into being.
    tools = (
        [_alien_tool_definition(MAX_EXPLORE_TESTS)] if TOOL_MODE else None
    )
    return runtime.create_session(system=system, tools=tools)


def _require_agent_client() -> AgentClient:
    if _AGENT_CLIENT is None:
        raise RuntimeError('AgentClient has not been initialized')
    return _AGENT_CLIENT


def _require_agent_runtime():
    if _AGENT_RUNTIME is None:
        raise RuntimeError('AgentRuntime has not been initialized')
    return _AGENT_RUNTIME


def _target_usage() -> dict:
    return _require_agent_client().get_usage()


def _reasoning_text(session: AgentSession) -> str:
    response = session.last_response
    if response is None:
        return ''
    return '\n'.join(
        artifact.text
        for artifact in response.reasoning
        if artifact.text
    )






_CONTROL_MODE = os.environ.get('EVAL_CONTROL_MODE', 'self').lower()
_CONTROL_BANK_PATH = os.environ.get('EVAL_CONTROL_BANK', '')
_CONTROL_MANIFEST_PATH = os.environ.get('EVAL_CONTROL_MANIFEST', '')
_CONTROL_BANK_CACHE: dict[str, dict] | None = None
_CONTROL_PREFLIGHT: dict | None = None
_VALID_CONTROL_MODES = {'none', 'self', 'think', 'random', 'passive'}




_PROMPT_VARIANT = os.environ.get('EVAL_PROMPT_VARIANT', '').strip().lower()
if not _PROMPT_VARIANT:
    _PROMPT_VARIANT = 'generalization'

_VALID_PROMPT_VARIANTS = {'original', 'generalization', 'explicit_generalization'}


if _PROMPT_VARIANT not in _VALID_PROMPT_VARIANTS:
    raise ValueError(f'''Unknown EVAL_PROMPT_VARIANT={_PROMPT_VARIANT!r}; expected one of {sorted(_VALID_PROMPT_VARIANTS)}''')


_EXPLICIT_GENERALIZATION = _PROMPT_VARIANT != 'original'
_HIDDEN_TESTS_ONLY = os.environ.get('EVAL_HIDDEN_TESTS_ONLY', '0').lower() in ('1', 'true', 'yes')


try:
    N_EXPLORE_ROUNDS = max(
        1, int(os.environ.get('EVAL_EXPLORE_ROUNDS', '3')))
except ValueError:
    N_EXPLORE_ROUNDS = 3
try:
    N_EXPLORE_LOOPS = max(1, int(os.environ.get('EVAL_EXPLORE_LOOPS', '4')))
except ValueError:
    N_EXPLORE_LOOPS = 4

MAX_CONTEXT_TURNS = 0
MAX_EXPLORE_TESTS = 5


try:
    MAX_PARALLEL_TESTS = max(1, int(os.environ.get('ALIENCODE_PARALLEL_TESTS', '20')))
except ValueError:
    MAX_PARALLEL_TESTS = 20

TEST_CASE_EXEC_TIMEOUT = 8
MAX_SEED_SHOWS = 3
MAX_OUTPUT_LINES = 15
MAX_OUTPUT_CHARS = 600
MAX_CODE_LINES = 40
_LOOP_WEIGHT = 3

# Tool mode lets the model call the interpreter itself, deciding when to run
# code and how many probes a round is worth. This is the protocol now, so it is
# on unless asked otherwise; EVAL_TOOL_MODE=0 restores the code-block protocol
# for reproducing the archived cohorts, whose scores it is not comparable with.
TOOL_MODE = os.environ.get('EVAL_TOOL_MODE', '1').strip().lower() in {
    '1', 'true', 'yes', 'on'}
try:
    MAX_TOOL_CALLS_PER_ROUND = max(
        1, int(os.environ.get('EVAL_MAX_TOOL_CALLS', '6')))
except ValueError:
    MAX_TOOL_CALLS_PER_ROUND = 6

ALIEN_TOOL_NAME = 'run_aliencode'

#: Distinguishes "caller said nothing" from "caller asked for the run's own
#: deadline", which None already means by the time it reaches the transport.
_UNSET = object()

PROTOCOL_V2 = os.environ.get('ALIENCODE_PROTOCOL_V2', '0').strip().lower() in {
    '1', 'true', 'yes', 'on'}
if PROTOCOL_V2:
    N_EXPLORE_ROUNDS = 1
    if 'EVAL_MAX_TOOL_CALLS' not in os.environ:
        MAX_TOOL_CALLS_PER_ROUND = protocol_v2.DEFAULT_PROBES_PER_BLOCK

DEFER_HELDOUT = os.environ.get(
    'ALIENCODE_DEFER_TESTS',
    '1' if PROTOCOL_V2 else '0',
).strip().lower() in {'1', 'true', 'yes', 'on'}
V2_ORACLE_DIAGNOSTICS = os.environ.get(
    'ALIENCODE_V2_ORACLE_DIAGNOSTICS',
    '1' if PROTOCOL_V2 else '0',
).strip().lower() in {'1', 'true', 'yes', 'on'}
# Score each milestone as soon as its base exists instead of holding every
# question to the end. Only worth it for providers that keep reasoning state
# their side and can refuse to continue from an hours-old base.
STREAM_HELDOUT = os.environ.get(
    'ALIENCODE_STREAM_HELDOUT', '0').strip().lower() in {
        '1', 'true', 'yes', 'on'}
try:
    _stop_value = os.environ.get('ALIENCODE_STOP_AFTER_MILESTONE', '').strip()
    STOP_AFTER_MILESTONE = int(_stop_value) if _stop_value else None
except ValueError:
    raise ValueError(
        'ALIENCODE_STOP_AFTER_MILESTONE must be an integer from 0 to 4')
if STOP_AFTER_MILESTONE is not None and STOP_AFTER_MILESTONE not in range(5):
    raise ValueError(
        'ALIENCODE_STOP_AFTER_MILESTONE must be an integer from 0 to 4')

# Extra turns granted after the budget is spent, to let the model wrap up.
_TOOL_LOOP_GRACE = 3

# The seed prompts are fixed, but under the tool protocol each one still lets
# the model issue its own calls, so the evidence a system carries into M0 is
# its own choice and differs across systems. LOCKED_SEED pins that evidence:
# every seed task executes exactly the program the task describes, one call
# each, so every system reaches M0 having seen the same outputs. The model
# still writes its own prediction first -- only the executed probe is
# substituted, the same way the passive control substitutes explore probes.
LOCKED_SEED = os.environ.get('EVAL_LOCKED_SEED', '0').strip().lower() in {
    '1', 'true', 'yes', 'on'}

# One entry per SEED_TASKS id. Each is the minimal program its own prompt asks
# for, so the locked seed is derivable from the task definitions rather than
# donated by whichever system happened to probe well.
LOCKED_SEED_CODES = world_data.LOCKED_SEED_CODES

BASE_MAX_TOKENS = 4096

RULES_BY_KEY = {r['key']: r for r in ALIEN_RULE_SPECS}
ALL_RULE_KEYS = [r['key'] for r in ALIEN_RULE_SPECS]

SYSTEM_PROMPT = '你是一个在 AlienCode 执行环境中工作的程序员。\nAlienCode 是一种简单的计算语言，语法和语义详见下方参考手册。\n这个环境的某些操作行为可能与参考手册描述的标准行为不同。\n你可以通过写代码做实验，系统会返回真实执行结果。\n你的目标是发现并总结出每个操作的实际运行时行为。\n\n约定：\n1. 写代码时请使用 ```aliencode``` 代码块。\n2. 若需要预测输出，请明确写成"预测输出：xxx"。\n3. 若需要总结规则，请尽量写成结构化清单。\n\n以下是 AlienCode 语言的标准参考手册：\n─────────────────────────────────────────\n' + REFERENCE_MANUAL + '\n─────────────────────────────────────────\n'















_CODE_BLOCK_CONVENTION = '1. 写代码时请使用 ```aliencode``` 代码块。'
_TOOL_CONVENTION = (
    f'1. 需要执行代码时，请调用 {ALIEN_TOOL_NAME} 工具（参数 code 为 AlienCode '
    '源码），系统会返回真实执行输出。你可以在一轮内多次调用它：'
    '先跑一个探针，看到结果后再决定下一个。'
)
_V2_TOOL_CONVENTION = (
    f'1. 探索阶段请调用 {ALIEN_TOOL_NAME} 工具执行你自己写的 AlienCode'
    '（参数 code 为源码），系统会原样执行并返回真实输出。\n'
    '   每次调用按观测值计费：EMIT 的每个参数算一个观测值，'
    f'一次调用最多 {MAX_EXPLORE_TESTS} 个观测值。'
    '超出上限的调用会被整个拒绝、不执行任何代码，'
    '所以请把大批量观测拆成多次调用。\n'
    '   闭卷评测阶段没有工具，需要写程序时用 ```aliencode``` 代码块提交答案。'
)

# Without a code block closing the reply, the answer has no delimiter, so the
# tool protocol has to ask for a bare value.
_PREDICTION_CONVENTION = '2. 若需要预测输出，请明确写成"预测输出：xxx"。'
_TOOL_PREDICTION_CONVENTION = (
    '2. 若需要预测输出，请另起一行写成"预测输出：xxx"，其中 xxx 只写预测的'
    '输出值本身，不要附加解释、括号备注或 markdown 强调。'
)

# Same instructions, except experiments go through the tool instead of a code
# block the harness scrapes. _tool_system_prompt asserts the swap took effect.
def _tool_system_prompt() -> str:
    prompt = SYSTEM_PROMPT
    for original, replacement in (
            (_CODE_BLOCK_CONVENTION, _TOOL_CONVENTION),
            (_PREDICTION_CONVENTION, _TOOL_PREDICTION_CONVENTION)):
        if original not in prompt:
            raise RuntimeError(
                'SYSTEM_PROMPT no longer states the code-block convention; '
                f'update the tool protocol wording to match: {original!r}')
        prompt = prompt.replace(original, replacement)
    return prompt


def _v2_system_prompt() -> str:
    prompt = SYSTEM_PROMPT.replace(
        '你可以通过写代码做实验，系统会返回真实执行结果。',
        '你可以通过写代码做实验，系统会执行你提交的代码并返回真实结果；'
        '每次实验能观察的值有上限，所以要挑最有信息量的实验。',
    )
    for original, replacement in (
            (_CODE_BLOCK_CONVENTION, _V2_TOOL_CONVENTION),
            (_PREDICTION_CONVENTION, _TOOL_PREDICTION_CONVENTION)):
        if original not in prompt:
            raise RuntimeError(
                'SYSTEM_PROMPT convention changed; update AlienCode v2 '
                f'wording to match: {original!r}')
        prompt = prompt.replace(original, replacement)
    return prompt


def _initial_system_prompt() -> str:
    if PROTOCOL_V2:
        prompt = _v2_system_prompt()
        prompt += (
            '\n\n─────────────────────────────────────────\n'
            + protocol_v2.demo_block(alien_exec)
            + '\n─────────────────────────────────────────\n'
        )
    else:
        prompt = _tool_system_prompt() if TOOL_MODE else SYSTEM_PROMPT
    return prompt


SEED_LIMIT_NOTE = f'''\n\n⚠️ 执行环境限制：每段代码最多 {MAX_CODE_LINES} 行、最多 {MAX_SEED_SHOWS} 个观测值（EMIT 的每个参数算一个观测值，循环内每个按 {_LOOP_WEIGHT}× 计数），输出最多返回 {MAX_OUTPUT_LINES} 行。超出将被截断。'''

SEED_LIMIT_NOTE_TOOL = f'''\n\n⚠️ 执行环境限制：每次 {ALIEN_TOOL_NAME} 调用最多 {MAX_CODE_LINES} 行代码、最多 {MAX_SEED_SHOWS} 个观测值（EMIT 的每个参数算一个观测值，循环内每个按 {_LOOP_WEIGHT}× 计数），返回最多 {MAX_OUTPUT_LINES} 行输出，超出将被截断。本轮最多可调用 {MAX_TOOL_CALLS_PER_ROUND} 次。'''










SEED_TASKS = world_data.SEED_TASKS



















TEST_TASKS = []  # The legacy 90-task set is not released; v2 reads eval_set_v2.json.


















































































































































































_tc_data_path = os.path.join(_THIS_DIR, 'test_cases.json')
if os.path.exists(_tc_data_path):
    with open(_tc_data_path, encoding='utf-8') as _f:
        _tc_data = json.load(_f)
    _tc_map = {d['id']: d for d in _tc_data}
    for _task in TEST_TASKS:
        if _task['id'] in _tc_map:
            _td = _tc_map[_task['id']]
            _task['test_cases'] = _td['test_cases']
            _task['checker'] = _td.get('checker', 'exact')

# The v2 evaluation set supersedes the list above. It keeps the 37 legacy
# tasks the cohort still fails most often and spends the rest of its budget on
# two bands the legacy set never had: chained algorithm families, and tasks
# built on the one mechanism that defeats every system measured so far. Its
# file is self-contained -- prompts, checkers and cases travel together -- so
# nothing here has to be merged into it. Rebuild it with
# dev/sandboxes/code/assemble_task_set.py; score the legacy 90 instead with
# ALIENCODE_TASK_SET=legacy, which is what reproducing a pre-v2 run needs.
TASK_SET = os.environ.get('ALIENCODE_TASK_SET', 'v2').strip().lower()
#: Whether the operator named the set or just took the default. A resumed run
#: normally keeps the set its checkpoint was written under, but moving a
#: half-finished run onto v2 is a deliberate act: its deferred ledger is keyed
#: by milestone and task id, so the questions both sets share are already
#: answered and only the new ones get asked.
_TASK_SET_FORCED = 'ALIENCODE_TASK_SET' in os.environ
_v2_path = os.path.join(_THIS_DIR, 'eval_set_v2.json')
#: Kept addressable so a resumed run can be handed back the set it started on.
_LEGACY_TASKS = TEST_TASKS
if TASK_SET == 'v2' and os.path.exists(_v2_path):
    with open(_v2_path, encoding='utf-8') as _f:
        TEST_TASKS = json.load(_f)
elif TASK_SET not in ('v2', 'legacy'):
    raise ValueError(
        f'Unknown ALIENCODE_TASK_SET={TASK_SET!r}; expected v2 or legacy')
else:
    TASK_SET = 'legacy'

if _HIDDEN_TESTS_ONLY:
    TEST_TASKS = [task for task in TEST_TASKS if task.get('test_cases')]

# A prediction with nothing to compare against can never be scored right, so
# refuse the task set at load time rather than grade it wrong in every run.
_UNGRADABLE = [task['id'] for task in TEST_TASKS if task.get('question_type') == 'predict'
               and not task.get('given_code') and task.get('expected') is None]
if _UNGRADABLE:
    raise ValueError(f'predict tasks without an answer to compare with: {_UNGRADABLE}')





GROUND_TRUTH_SEXPR = world_data.GROUND_TRUTH_SEXPR
















_IDENTITY_RULE_IDS = {r['id'] for r in ALIEN_RULE_SPECS if r.get('identity')}



ALIAS_MAP = {
    'xor': '^', 'XOR': '^', 'bitxor': '^',
    'mul': '*', 'multiply': '*',
    'add': '+', 'plus': '+',
    'sub': '-', 'subtract': '-', 'minus': '-',
    'mod': '%', 'modulo': '%',
    'pow': '**', 'power': '**', 'exp': '**',
    'eq': '==', 'equal': '==',
    'neq': '!=', 'not_equal': '!=',
    'gt': '>', 'greater': '>',
    'lt': '<', 'less': '<',
    'rev': 'reverse', 'reversed': 'reverse',
    'index': 'get', 'nth': 'get',
    'desc': 'sort_desc', 'descending': 'sort_desc',
    'insert_head': 'prepend',
    'identity': 'IDENTITY', 'unchanged': 'IDENTITY', 'same': 'IDENTITY',
    'no_change': 'IDENTITY'}


COMMUTATIVE_OPS = {'*', 'or', 'and', '+', '!=', '=='}

SEXPR_RULE_IDS = list(GROUND_TRUTH_SEXPR.keys())


def _tokenize_sexpr(s: str) -> list[str]:
    tokens = []
    i = 0
    while i < len(s):
        c = s[i]
        if c in ' \t\n\r':
            i += 1
        elif c == '(':
            tokens.append('(')
            i += 1
        elif c == ')':
            tokens.append(')')
            i += 1
        else:
            j = i
            while j < len(s) and s[j] not in ' \t\n\r()':
                j += 1
            tokens.append(s[i:j])
            i = j
    return tokens


def parse_sexpr(s: str):
    s = s.strip()
    if not s:
        return None
    if s in ('UNKNOWN', 'IDENTITY'):
        return s
    tokens = _tokenize_sexpr(s)
    pos = [0]

    def _parse():
        if pos[0] >= len(tokens):
            return None
        tok = tokens[pos[0]]
        if tok == '(':
            pos[0] += 1
            items = []
            while pos[0] < len(tokens) and tokens[pos[0]] != ')':
                items.append(_parse())
            if pos[0] < len(tokens):
                pos[0] += 1
            return tuple(items)

        pos[    0] += 1
        try:
            return     int(tok)
        except     ValueError:
            try:
                return     float(tok)
            except     ValueError:
                return     tok


    result = _parse()
    return result

def _normalize_alias(node):
    if isinstance(node, str):
        return ALIAS_MAP.get(node, node)
    if isinstance(node, tuple):
        return tuple(_normalize_alias(x) for x in node)
    return node


def _match_structural(model, canon, var_map=None):
    if var_map is None:
        var_map = {}
    if isinstance(canon, (int, float)):
        return isinstance(model, type(canon)) and model == canon
    if isinstance(canon, str):
        if canon == 'IDENTITY':
            return isinstance(model, str) and _normalize_alias(model) == 'IDENTITY'
        if canon in ALIAS_MAP.values() or canon in ('+', '-', '*', '/', '//', '%', '^', '**', '==', '!=', '>', '<', '>=', '<=', 'and', 'or', 'not', 'reverse', 'get', 'slice', 'len', 'range', 'zip', 'enumerate', 'filter', 'sort_asc', 'sort_desc', 'max', 'min', 'any', 'all', 'prepend', 'pop_left', 'print'):




            return isinstance(model, str) and _normalize_alias(model) == canon
        if isinstance(model, str):
            if canon in var_map:
                return var_map[canon] == model
            var_map[canon] = model
            return True
        return False
    if isinstance(canon, tuple) and isinstance(model, tuple):
        if len(canon) != len(model):
            return False
        canon_op = _normalize_alias(canon[0]) if canon else None
        model_op = _normalize_alias(model[0]) if model else None
        if canon_op != model_op:
            return False
        if canon_op in COMMUTATIVE_OPS and len(canon) == 3:
            vm1 = dict(var_map)
            if  _match_structural(model[1], canon[1], vm1) and _match_structural(model[2], canon[2], vm1):

                var_map.update(vm1)
                return True
            vm2 = dict(var_map)
            if  _match_structural(model[1], canon[2], vm2) and _match_structural(model[2], canon[1], vm2):

                var_map.update(vm2)
                return True
            return False
        for m_elem, c_elem in zip(model[1:], canon[1:]):
            if not _match_structural(m_elem, c_elem, var_map):
                return False
        return True
    return False


def match_rule(model_expr: str, canonical_expr: str,
               rule_id: str | None = None) -> bool:
    try:
        m_ast = _normalize_alias(parse_sexpr(model_expr))
        c_ast = _normalize_alias(parse_sexpr(canonical_expr))
    except Exception:
        return False

    if m_ast is None or c_ast is None:
        return False
    if rule_id and rule_id in _IDENTITY_RULE_IDS and m_ast == 'IDENTITY':
        return     True
    if c_ast == 'IDENTITY':
        return m_ast == 'IDENTITY'
    return _match_structural(m_ast, c_ast)






SEXPR_MILESTONE_PROMPT = world_data.SEXPR_MILESTONE_PROMPT













































































def _extract_sexpr_rules(summary: str) -> dict[str, str]:
    rules = {}
    for m in re.finditer('(R\\d+\\+?)\\s*:=\\s*(.+?)(?=\\n(?:R\\d|---|$)|\\Z)', summary, re.DOTALL):
        rule_id = m.group(1).strip()
        expr = m.group(2).strip()
        expr = re.sub('#.*',  '', expr).strip()
        if expr:
            rules[rule_id] = expr
    return rules






_ALIEN_STATEMENT_STARTS = tuple(FUNC_KEYWORD_MAP) + (
    'SET', 'SWEEP', 'UPON', 'LEST', 'WHILE', 'CRAFT',
    'DELIVER', 'CEASE', 'BYPASS', 'IDLE', 'DEFAULT',
)

def _fenceless_code(response: str) -> str:
    """The AlienCode in a reply that arrived without a code fence.

    Most models fence their code and are read by the patterns above. Some
    answer a write-the-code question with the bare statement instead, and
    scoring that as "no code" would measure markdown habits rather than
    whether the model inferred the language. Take the first run of lines that
    open with a statement keyword, plus whatever is indented under them.
    """

    block: list[str] = []
    for raw in response.splitlines():
        line = raw.rstrip()
        opens = line.strip().startswith(_ALIEN_STATEMENT_STARTS)
        continues = bool(block) and line[:1].isspace()
        if opens or continues:
            block.append(line)
        elif block:
            break
    return '\n'.join(block).strip()


def extract_code(response: str) -> str:
    for pattern in ('```aliencode\\s*\\n(.*?)```', '```alien\\s*\\n(.*?)```', '```python\\s*\\n(.*?)```', '```\\s*\\n(.*?)```'):





        # Fence tags match case-insensitively: a reply that writes
        # ```AlienCode still submitted a program and should be graded on it.
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[0].strip()
    m = re.search(
        '```aliencode\\s*\\n(.+)', response, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return _fenceless_code(response)


def _count_show_statements(code: str) -> int:
    return len(re.findall('\\bEMIT\\s*\\(', code))


_EMIT_OPEN_RE = re.compile('\\bEMIT\\s*\\(')


def _emit_argument_count(code: str, open_paren_end: int) -> int:
    '''Top-level arguments of the EMIT call whose "(" ends at the given index.

    One EMIT statement used to cost one unit however many values it displayed,
    so a batch like ``EMIT(c1, c2, c3, c4, c5)`` read five conditions out of the
    environment for the price of one. Nested commas belong to the argument they
    sit inside and are not separate observations.
    '''
    depth = 1
    commas = 0
    has_content = False
    quote = None
    i = open_paren_end
    while i < len(code) and depth > 0:
        ch = code[i]
        if quote is not None:
            if ch == '\\':
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in '"\'':
            quote = ch
            has_content = True
        elif ch in '([{':
            depth += 1
            has_content = True
        elif ch in ')]}':
            depth -= 1
            if depth > 0:
                has_content = True
        elif ch == ',' and depth == 1:
            commas += 1
        elif not ch.isspace():
            has_content = True
        i += 1
    # A bare EMIT() observes nothing but still costs a unit, so an empty call
    # cannot be used to probe for free.
    return commas + 1 if has_content else 1


def _loop_depth_by_line(lines: list[str]) -> list[int]:
    depths = []
    depth = 0
    for line in lines:
        stripped = line.lstrip()
        is_indented = line != stripped and stripped != ''
        if re.match('(SWEEP\\b|SWEEP_WHILE\\b|WHILE\\b)', stripped):
            depth += 1
        elif depth > 0 and not is_indented and stripped:
            depth = 0
        depths.append(depth)
    return depths


def _show_units_by_line(code: str) -> list[int]:
    '''Weighted observation units charged to the line each EMIT starts on.

    A call may span several lines, so the scan runs over the whole submission
    and attributes the cost to the opening line; that is also the line whose
    loop depth decides the multiplier.
    '''
    lines = code.split('\n')
    depths = _loop_depth_by_line(lines)

    units = [0] * len(lines)
    for match in _EMIT_OPEN_RE.finditer(code):
        values = _emit_argument_count(code, match.end())
        line_idx = code.count('\n', 0, match.start())
        weight = _LOOP_WEIGHT if depths[line_idx] > 0 else 1
        units[line_idx] += values * weight
    return units


def _count_show_weighted(code: str) -> int:
    '''Count observed values with loop awareness.

    Each top-level EMIT argument is one observation, and observations inside
    SWEEP / SWEEP_WHILE / WHILE loops count as _LOOP_WEIGHT× to prevent
    brute-force enumeration via loop constructs.
    '''
    return sum(_show_units_by_line(code))


def _truncate_code_by_show(code: str, max_shows: int) -> tuple[str, int, int]:
    if not code or max_shows <= 0:
        return code, 0, 0
    units = _show_units_by_line(code)
    original_count = sum(units)
    if original_count <= max_shows:
        return code, original_count, original_count

    lines = code.split('\n')
    kept_lines = []
    show_seen = 0
    for line, weight in zip(lines, units):
        # A line is kept only if it fits whole. Keeping an overflowing line and
        # then reporting the cap would hand back more observations than the run
        # is charged for -- one packed EMIT could read seven values for five.
        if show_seen + weight > max_shows:
            break
        kept_lines.append(line)
        show_seen += weight

    return '\n'.join(kept_lines), original_count, show_seen


def _enforce_code_limits(code: str) -> tuple[str, list[str]]:
    '''Enforce MAX_CODE_LINES. Returns (code, warnings).'''
    warnings = []
    if not code:
        return code, warnings
    lines = code.split('\n')
    if len(lines) > MAX_CODE_LINES:
        code = '\n'.join(lines[:MAX_CODE_LINES])
        warnings.append(f'''代码超过 {MAX_CODE_LINES} 行限制 (原始 {len(lines)} 行)，已截断''')

    return code, warnings


def extract_prediction(response: str, *, tool_protocol: bool = False) -> str:
    '''Pull the answer the model committed to out of its reply.

    Under the tool protocol the code lives in tool calls, so a reply usually
    has no closing fence for the match to stop at and would otherwise run to
    the end of the message. End it at a blank line instead, and drop the
    markdown emphasis models tend to wrap the answer in.

    A tool round spans several messages, and the model typically predicts
    before running the probe that confirms it, so the *last* prediction in the
    round is the one it stands behind.
    '''
    stop = '```|\\n\\s*\\n' if tool_protocol else '```'
    pattern = f'预测[^\\n：:]*[：:]\\s*(.*?)(?={stop}|\\Z)'
    if tool_protocol:
        matches = list(re.finditer(pattern, response, re.DOTALL))
        m = matches[-1] if matches else None
    else:
        m = re.search(pattern, response, re.DOTALL)
    if m:
        pred = m.group(1).strip()
        if tool_protocol:
            # "**预测输出：100**（已验证）" leaves "100**（已验证）", so cut at
            # the closing emphasis. Splitting only when it is present keeps
            # multi-line answers intact.
            if '**' in pred:
                pred = pred.split('**', 1)[0].strip()
            pred = pred.strip('*`_ \t')
        if pred:
            return pred
    for pattern in ('输出[应会]*[该为是]*[：:]\\s*[`]*\\s*([^\\s`\\n]+)', '结果[应会]*[该为是]*[：:]\\s*[`]*\\s*([^\\s`\\n]+)'):



        m = re.search(pattern, response)
        if m:
            return m.group(1).strip()
    return ''


def values_equal(a: str, b: str) -> bool:
    a, b = (a or '').strip(), (b or '').strip()
    if a == b:
        return True
    try:
        if float(a) == float(b):
            return True
    except (ValueError, TypeError):
        pass

    a_lines = [line.strip() for line in a.splitlines() if line.strip()]
    b_lines = [line.strip() for line in b.splitlines() if line.strip()]
    return a_lines == b_lines


def predict_answer(task: dict) -> str:
    """The output a prediction task's answer is compared with.

    Task lists written into this file carry the program (`given_code`); the
    assembled v2 file carries only the engine's output (`expected`).
    """
    if task.get('given_code'):
        return alien_exec(task['given_code'])
    if task.get('expected') is not None:
        return str(task['expected'])
    raise ValueError(f"predict task {task['id']} has neither given_code nor expected")

_TOP_LEVEL_ASSIGN_RE = re.compile('(SET\\b|[A-Za-z_][\\w,\\s]*=(?!=))')


def _extract_func_defs(code: str) -> str:
    """Extract function definitions AND top-level assignments from model code.

    Keeps CRAFT/def function definitions plus any top-level constant/helper
    assignments (e.g. ``SET INF AS 999``) that the functions depend on, so the
    harness can append its own test call. Only top-level *driver* statements
    (the model's own EMIT calls and bare expression calls) are stripped.

    Previously this dropped every top-level statement, so code relying on a
    top-level constant failed with NameError instead of being judged on its
    actual logic.
    """
    if not code:
        return ''
    lines = code.split('\n')
    kept = []
    in_func = False
    for line in lines:
        stripped = line.lstrip()
        indented = line != stripped and stripped != ''

        if re.match('(CRAFT\\b|def\\b)', stripped):
            in_func = True
            kept.append(line)
            continue

        if in_func:
            if stripped == '' or indented:
                kept.append(line)
                continue
            in_func = False



        if stripped == '':
            kept.append(line); continue
        if not   indented and _TOP_LEVEL_ASSIGN_RE.match(stripped):
            kept.append(line)



    return '\n'.join(kept).strip()


def _check_topo_order(output: str, n: int, edges: list[tuple[int, int]]) -> bool:
    '''Verify output is a valid topological ordering.'''
    try:
        order = _parse_list_output(output)
        if len(order) != n:
            return False
        pos = {v: i for i, v in enumerate(order)}
        if len(pos) != n:
            return False
        for u, v in edges:
            if pos.get(u, n) >= pos.get(v, -1):
                return False
        return True
    except Exception:
        return False


def _check_perm_set(output: str, expected_set) -> bool:
    '''Verify output contains all permutations (order-independent).'''
    try:
        perms = _parse_list_output(output)
        got = set()
        for p in perms:
            if isinstance(p, (list, tuple)):
                got.add(tuple(p))
                continue
            return     False
        canonical = set()
        for p in expected_set:
            if isinstance(p, (list, tuple)):
                canonical.add(tuple(p))
                continue
            canonical.add(    p)
        return got == canonical
    except Exception:
        return False


def _parse_list_output(output: str):
    '''Parse a printed Python list/nested structure from output.'''
    import ast as _ast
    output = output.strip()
    try:
        return _ast.literal_eval(output)
    except (ValueError, SyntaxError):
        return None


def _run_test_cases(model_code: str, task: dict) -> tuple[bool, str]:
    """Run multiple test cases against model's function definitions.

    Returns (all_passed: bool, detail_str: str).
    """
    test_cases = task.get('test_cases', [])
    if not test_cases:
        return (True, '(no test_cases)')

    func_code = _extract_func_defs(model_code)
    if not func_code.strip():
        return (False, '(无法提取函数定义)')

    checker = task.get('checker', 'exact')
    results = []
    all_passed = True

    for tc in test_cases:
        call_code = tc['call']
        expected = tc['expected']
        full_code = func_code + '\n' + call_code
        try:
            output = alien_exec(full_code, timeout=TEST_CASE_EXEC_TIMEOUT)
        except Exception as e:
            output = f'''[AlienError] {e}'''

        if '[AlienError]' in output:
            passed = False
        elif checker == 'exact':
            passed = values_equal(output, expected)
        elif checker == 'topo':
            topo_meta = tc.get('topo_meta', {})
            passed = _check_topo_order(output, topo_meta.get('n', 0), topo_meta.get('edges', []))

        elif checker == 'perm':
            perm_meta = tc.get('perm_expected_set')
            passed = _check_perm_set(output, perm_meta) if perm_meta else False
        elif checker == 'sorted_match':
            try:
                got = sorted(_parse_list_output(output))
                exp = sorted(_parse_list_output(expected))
                passed = got == exp
            except Exception:
                passed = False

        elif checker == 'lines_set':
            got_lines = sorted(l.strip() for l in output.strip().splitlines() if l.strip())
            exp_lines = sorted(l.strip() for l in expected.strip().splitlines() if l.strip())
            passed = got_lines == exp_lines
        else:
            passed = values_equal(output, expected)
        if not passed:
            all_passed = False
        results.append({
                    'call': call_code[:80],
                        'expected': str(expected)[:80],
                   'got': output[:80],
                      'passed': passed})


    detail_parts = []
    for r in results:
        mark = '✓' if r['passed'] else '✗'
        detail_parts.append(f'''  {mark} {r["call"][:50]}  got={r["got"][:40]}''')
    detail_str = '\n'.join(detail_parts)

    return all_passed, detail_str


def _truncate_output(output: str,
                     max_lines: int | None = None,
                     max_chars: int | None = None) -> str:
    if not output:
        return '(无输出)'
    if max_lines is None:
        max_lines = MAX_OUTPUT_LINES
    if max_chars is None:
        max_chars = MAX_OUTPUT_CHARS
    lines = output.split('\n')
    if len(lines) > max_lines:
        output = '\n'.join(lines[:max_lines]) + f'''\n... (共 {len(lines)} 行，已截断至 {max_lines} 行)'''
    if len(output) > max_chars:
        output = output[:max_chars] + '... (已截断)'
    return output


_CKPT_EXEC_OUTPUT_MAX = 102400


def _cap_exec_output(output):
    '''落盘前裁剪 exec_output，避免单条无限循环输出撑爆 checkpoint。

    病理场景（例如 milestone 测试时模型写出无限 EMIT 循环）会产生几百 MB
    的字符串，原样塞进 checkpoint 会让 worker 进程被 OOM kill。

    仅影响落盘副本；回喂给模型的文本仍走 ``_truncate_output``。
    '''
    if not isinstance(output, str) or len(output) <= _CKPT_EXEC_OUTPUT_MAX:
        return output
    return output[:_CKPT_EXEC_OUTPUT_MAX] + f'''\n... (已截断以便存盘, 原始 {len(output)} 字节)'''





def _shows_divergence_reasoning(response: str) -> bool:
    patterns = world_data.DIVERGENCE_PATTERNS








    return any(re.search(p, response, re.IGNORECASE) for p in patterns)


def _classify_disorientation(prediction, real_answer, expected_answer, response):
    matches_real = values_equal(prediction, real_answer) if prediction else False
    matches_expected = values_equal(prediction, expected_answer) if prediction else False
    if matches_real and not matches_expected:
        return 'MISLED'
    if matches_expected:
        return 'CORRECT'
    if _shows_divergence_reasoning(response):
        return 'EXPLORING'
    return 'CONFUSED'






MAX_LOG_BYTES = 31457280


class LogSizeLimitExceeded(BaseException):
    '''单个模型日志超过 MAX_LOG_BYTES 时抛出。

    继承自 BaseException（而不是 Exception）：这样 worker 线程里的
    `except Exception:` 接不到它，能强制冒泡到 run_eval 的外层捕获，
    确保即使死循环发生在并行测试 worker 里也能真正中止。
    '''
    pass

class _Tee:
    def __init__(self, *files, max_log_bytes=None):
        self.files = files
        self.max_log_bytes = max_log_bytes
        self._bytes_written = 0
        self._tripped = False
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
        if  self.max_log_bytes is not None and not   self._tripped:
            self._bytes_written += (len(obj.encode('utf-8')) if isinstance(obj, str) else len(obj))


            if self._bytes_written > self.max_log_bytes:
                self._tripped = True
                limit_mb = self.max_log_bytes /  1048576
                raise LogSizeLimitExceeded(f'''日志文件已超过 {limit_mb:.0f} MB (已写入 {self._bytes_written / 1048576:.1f} MB)，模型可能陷入死循环，本模型评测中止。''')
            return None



    def flush(self):
        for f in self.files:
            f.flush()






_CHECKPOINT_DIR = os.path.join(_DATA_DIR, 'checkpoints')


def _ckpt_path():
    short = MODEL_SHORT or MODEL.rsplit('_', 1)[-1]
    suffix = f'''_{RUN_ID}''' if RUN_ID else ''
    return os.path.join(_CHECKPOINT_DIR, f'''ckpt_alien_code_{short}{suffix}.json''')


def _save_checkpoint(data):
    try:
        os.makedirs(_CHECKPOINT_DIR, exist_ok=True)
        path = _ckpt_path()
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception as e:
        print(f'''  [Checkpoint Warning] 保存失败: {e}''')
        return None

def _clear_checkpoint():
    path = _ckpt_path()
    try:
        if os.path.exists(path):
            os.remove(path)
            print('  [Checkpoint] 评测完成，已清除检查点')
            return None
        return None
    except OSError:
        return None

def _request_overrides() -> dict:
    '''Per-model sampling and reasoning knobs for one call.'''
    model_lower = MODEL.lower()
    kwargs = {}

    _CLAUDE_THINKING_BUDGET = 10000



    _NOTHINK = os.environ.get('EVAL_REASONING_EFFORT', '').lower() == 'none'

    if any(k in model_lower for k in ('claude', 'anthropic')):
        adaptive_opus = any(v in model_lower for v in (
            'opus-4-7', 'opus-4.7',
            'opus-4-8', 'opus-4.8',
            'opus-5', 'opus.5'))
        if _NOTHINK:
            if 'opus-5' in model_lower or 'opus.5' in model_lower:
                kwargs['thinking'] = {'type': 'disabled'}
            kwargs['max_tokens'] = BASE_MAX_TOKENS * 4

        elif adaptive_opus:
            kwargs['thinking'] = {
                'type': 'adaptive',
                'display': 'summarized'}
            effort = os.environ.get('EVAL_REASONING_EFFORT', 'max').lower()
            if effort not in ('low', 'medium', 'high', 'xhigh', 'max'):
                effort = 'max'
            kwargs['extra_body'] = {'output_config': {'effort': effort}}
            kwargs['max_tokens'] = BASE_MAX_TOKENS * 16
        else:
            kwargs['max_tokens'] = _CLAUDE_THINKING_BUDGET + BASE_MAX_TOKENS
            kwargs['thinking'] = {
                        'type': 'enabled',
                                 'budget_tokens': _CLAUDE_THINKING_BUDGET}

    elif MODEL.startswith(STANDARD_MESSAGES_PREFIX):
        # A non-Anthropic vendor speaking Anthropic Messages. DeepSeek has no
        # budget_tokens of its own, so the adapter layer quantises the number
        # onto its reasoning_effort enum: it picks a tier instead of capping
        # anything, and 60000 is the top one. Use max_tokens for a real limit.
        kwargs['thinking'] = (
            {'type': 'disabled'} if _NOTHINK
            else {'type': 'enabled', 'budget_tokens': 60000}
        )
        kwargs['max_tokens'] = BASE_MAX_TOKENS * 8

    elif 'gemini' in model_lower:
        # Gemini 3.x retired the integer thinking budget for named levels and
        # deprecated temperature/topP/topK, so an effort of 'max' -- valid for
        # every other vendor here -- is a 400, and the sampling knobs are
        # fields the vendor warns will start failing. Send neither.
        effort = os.environ.get('EVAL_REASONING_EFFORT', 'high').lower()
        if effort not in ('minimal', 'low', 'medium', 'high'):
            effort = 'high'
        kwargs['thinking'] = {
            'thinkingLevel': effort,
            # Asks for the readable thought summary alongside the signature.
            # The vendor only returns it about 7 times in 8, so the run is
            # scored on the signature round trip, not on this.
            'includeThoughts': True,
        }
        kwargs['max_tokens'] = min(BASE_MAX_TOKENS * 8, 65536)

    elif responses_route_from_model(MODEL) == 'doubao':
        # Four tiers here, topping out at 'high' -- the 'max' every other
        # vendor takes is a 400. Thinking is a separate switch from the tier,
        # and effort may only ride along with it when it is on. The cache is
        # off unless asked for, and the vendor reads none at all when
        # instructions are set, so the system prompt goes in the input where
        # it stays a cacheable prefix.
        effort = os.environ.get('EVAL_REASONING_EFFORT', 'high').lower()
        if effort not in ('minimal', 'low', 'medium', 'high'):
            effort = 'high'
        if _NOTHINK:
            kwargs['thinking'] = {'type': 'disabled'}
        else:
            kwargs['thinking'] = {'type': 'enabled'}
            kwargs['reasoning_effort'] = effort
        kwargs['temperature'] = 1.0
        # This budget covers the thinking as well as the answer.
        kwargs['max_tokens'] = min(BASE_MAX_TOKENS * 8, 262144)
        # The explicit cache refuses any request that both continues a cached
        # response and declares tools, which is every turn of a tool-mode
        # episode. Leave it off and take the implicit prefix cache, which
        # needs no opt-in. The system prompt still goes in the input rather
        # than instructions, since instructions suppress that cache too.
        kwargs['system_in_input'] = True

    elif responses_route_from_model(MODEL):
        # On the GatewayA Responses upstreams the tier is named by effort, over
        # seven steps up to 'max'; the vendor deprecated enable_thinking in
        # favour of it. Thinking mode caps output at 131072 tokens.
        effort = os.environ.get('EVAL_REASONING_EFFORT', 'max').lower()
        if effort not in (
            'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'
        ):
            effort = 'max'
        kwargs['reasoning_effort'] = effort
        kwargs['temperature'] = 1.0
        kwargs['max_tokens'] = min(BASE_MAX_TOKENS * 8, 131072)

    # These two are matched on the route rather than on words in the model id,
    # so they have to be asked before the keyword branches below: 'grok' and
    # 'kimi' appear there too and would otherwise answer first.
    elif chat_route_from_model(MODEL) == 'xai':
        # Three tiers topping out at 'high': 'max' is not in the enum and
        # 'none' is a 400, since reasoning cannot be turned off here. The
        # budget field is max_completion_tokens, which caps the visible answer
        # without the reasoning eating into it. Nothing hosts the thinking, and
        # the prompt cache is automatic -- prompt_cache_key only keeps a run on
        # the node holding its prefix.
        effort = os.environ.get('EVAL_REASONING_EFFORT', 'high').lower()
        if effort not in ('low', 'medium', 'high'):
            effort = 'high'
        kwargs['reasoning_effort'] = effort
        kwargs['temperature'] = 1.0
        kwargs['max_completion_tokens'] = BASE_MAX_TOKENS * 8
        kwargs['prompt_cache_key'] = f'aliencode-{RUN_ID}'

    elif chat_route_from_model(MODEL) == 'moonshot':
        # Thinking is always on and the tier tops out at 'max'. The sampling
        # knobs are fixed server-side and rejected if sent, max_tokens is
        # retired in favour of max_completion_tokens, and that budget covers
        # the reasoning as well as the answer.
        effort = os.environ.get('EVAL_REASONING_EFFORT', 'max').lower()
        if effort not in ('none', 'low', 'medium', 'high', 'max'):
            effort = 'max'
        kwargs['reasoning_effort'] = effort
        kwargs['max_completion_tokens'] = min(BASE_MAX_TOKENS * 8, 131072)

    elif gateway_a_vendor(MODEL) == 'ali':
        # Qwen on GatewayA's own passthrough. The tier is named, not budgeted:
        # reasoning_effort and thinking_budget are a 400 together, and
        # max_completion_tokens must be strictly greater than any budget sent,
        # so naming the tier and leaving the budget implicit avoids both.
        # This endpoint collapses the seven tiers onto three -- max and high
        # both become xhigh -- unlike the Responses route, where max is its own
        # tier. The thinking comes back as reasoning_content and the vendor
        # expects it replayed verbatim (preserve_thinking defaults true), which
        # the chat adapter does by replaying the message it parsed.
        effort = os.environ.get('EVAL_REASONING_EFFORT', 'max').lower()
        if effort not in (
            'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'
        ):
            effort = 'max'
        if effort == 'none' or _NOTHINK:
            kwargs['enable_thinking'] = False
        else:
            kwargs['enable_thinking'] = True
            kwargs['reasoning_effort'] = effort
        kwargs['temperature'] = 1.0
        kwargs['max_completion_tokens'] = min(BASE_MAX_TOKENS * 32, 131072)

    elif legacy_route(MODEL) == 'gateway_a_standard':
        # DeepSeek on GatewayA's standard endpoint. Seven named tiers up to 'max',
        # which the vendor collapses onto its own three, and 'none' turns
        # thinking off. Nothing hosts the thinking: it comes back as
        # message.reasoning_content and is only in scope next turn because the
        # chat adapter replays the assistant message byte-for-byte. json_schema
        # is a 400 here and parallel_tool_calls is ignored, so neither is sent.
        effort = os.environ.get('EVAL_REASONING_EFFORT', 'max').lower()
        if effort not in (
            'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'
        ):
            effort = 'max'
        kwargs['reasoning_effort'] = effort
        kwargs['temperature'] = 1.0
        # One budget covers the thinking and the answer, and at 'max' effort
        # over an explored history the thinking alone can exceed 32768: ten
        # closed-book turns of one run came back finish_reason=length with
        # reasoning present and no answer at all, which scores as a blank.
        # Re-asking does not help when the budget is what ran out.
        kwargs['max_completion_tokens'] = min(BASE_MAX_TOKENS * 32, 393216)

    elif any(k in model_lower for k in ('qwen',)):
        kwargs['temperature'] = 1.0
        kwargs['max_tokens'] = BASE_MAX_TOKENS * 8
        kwargs['extra_body'] = {'enable_thinking': not _NOTHINK}
    elif 'doubao' in model_lower:


        kwargs['temperature'] = 1.0
        kwargs['max_tokens'] = BASE_MAX_TOKENS * 8
        if _NOTHINK:
            kwargs['thinking'] = {'type': 'disabled'}
        else:
            kwargs['reasoning_effort'] = os.environ.get('EVAL_REASONING_EFFORT', 'high')




    elif any(k in model_lower for k in ('o1', 'o3', 'reasoner', 'thinking', 'mimo', 'hy3', 'hy4', 'opd', 'a20b', 'gpt-5', 'minimax', 'gemini', 'grok', 'deepseek-v4')) or MODEL in [m for m in os.environ.get('HY_GATEWAY_C_MODELS', '').split(',') if m]:

        kwargs['temperature'] = 1.0


        effort = os.environ.get('EVAL_REASONING_EFFORT', 'high')
        kwargs['reasoning_effort'] = effort



        kwargs['max_tokens'] = BASE_MAX_TOKENS * 16 if 'gemini' in model_lower and effort.lower() == 'max' else BASE_MAX_TOKENS * 8




    elif any(k in model_lower for k in ('kimi',)):
        kwargs['temperature'] = 1.0
        kwargs['max_tokens'] = BASE_MAX_TOKENS * 8
    else:
        kwargs['temperature'] = 0.0
        kwargs['max_tokens'] = BASE_MAX_TOKENS

    return kwargs


def _requested_effort():
    '''The reasoning tier this route will actually ask for.

    Reading EVAL_REASONING_EFFORT is not enough for the record: a route that
    the variable does not reach still asks for something (Anthropic's adaptive
    opus defaults to max, the Messages routes pin the top tier through a
    thinking budget and never look at the variable), and writing the unset
    variable into the results file has already made three max runs look like
    they were something else. Read it back out of the request instead.

    'unset' means the request carries no reasoning knob at all, which is not
    the same as asking for none: a route with no off switch drops that request
    on the floor and thinks anyway. The asked-for value is recorded separately
    so the two can be told apart afterwards.
    '''

    kwargs = _request_overrides()
    effort = kwargs.get('reasoning_effort')
    if effort:
        return str(effort)
    nested = (kwargs.get('extra_body') or {}).get('output_config') or {}
    if nested.get('effort'):
        return str(nested['effort'])
    thinking = kwargs.get('thinking') or {}
    if thinking.get('type') == 'disabled':
        return 'none'
    if thinking.get('budget_tokens'):
        return f"budget:{thinking['budget_tokens']}"
    if thinking.get('type'):
        return str(thinking['type'])
    return 'unset'


def _chat_turn(
        session: AgentSession,
        prompt: str,
        label='',
        *,
        tool_results=None,
        extra_overrides=None,
        timeout=_UNSET):
    '''Run one model call and return the full response.

    Pass ``tool_results`` to answer the session's pending tool calls instead of
    appending a new user message, and ``extra_overrides`` to adjust this call's
    request on top of the per-model defaults.

    ``timeout`` defaults to the trajectory budget, because all but one caller
    here is building the trajectory: summaries, exploration rounds, the tool
    round trips inside them, oracle acknowledgements. The exception is the
    graded question, which passes ``None`` to keep the run's own deadline --
    spending that budget is its wrong answer, and waiting longer only delays
    recording it.
    '''
    if timeout is _UNSET:
        timeout = _trajectory_deadline()
    history = session.history
    turns = (
        int(session.system is not None)
        + sum(
            1 for event in history
            if event.get('role') in {'user', 'assistant'}
        )
        + 1
    )
    chars = (
        len(str(session.system or ''))
        + sum(
            len(str(event.get('payload', {}).get('content', '')))
            for event in history
        )
        + len(prompt)
    )
    print(
        f'\n[Chat -> {MODEL_SHORT or MODEL} ({label})] '
        f'turns={turns} chars={chars}')
    kwargs = _request_overrides()
    if extra_overrides:
        kwargs.update(extra_overrides)

    if tool_results is not None:
        response = session.submit_tool_results(
            tool_results,
            label=label,
            request_overrides=kwargs,
        )
    else:
        response = session.send_user(
            prompt,
            label=label,
            request_overrides=kwargs,
            timeout=timeout,
        )
    result = response.text or ''
    reasoning = '\n'.join(
        artifact.text
        for artifact in response.reasoning
        if artifact.text
    )
    _LAST_COT.text = reasoning
    signature_count = sum(
        bool(artifact.signature) for artifact in response.reasoning
    )
    if signature_count:
        print(
            f'  [Native reasoning] {label}: '
            f'{signature_count} signed artifact(s) preserved')

    if _SAVE_TRANSCRIPT:
        rec = {
                     'label': label,
                                    'model': (MODEL_SHORT
                or                  MODEL),
                      'prompt': prompt,
                                  'response': (result
                or                ''),
                         'reasoning': reasoning,
                             'reasoning_len': len(reasoning),
                            'response_len': len(result or ''),
                            'control_mode': _CONTROL_MODE,
                         'call_meta': session.last_call_meta}
        with _TRANSCRIPT_LOCK:
            first = not _TRANSCRIPT
            _TRANSCRIPT.append(rec)
            if _TRANSCRIPT_PATH:
                try:
                    with open(_TRANSCRIPT_PATH, 'a', encoding='utf-8') as tf:
                        tf.write(json.dumps(rec, ensure_ascii=False) + '\n')
                        tf.flush()
                except Exception:
                    pass

        if first:
            print(f'''[TRANSCRIPT] first call captured: reasoning_len={len(reasoning)} response_len={len(result or '')} (CoT {"OK" if reasoning else "EMPTY"})''')
    return response


def _chat(session: AgentSession, prompt: str, label=''):
    return _chat_turn(session, prompt, label=label).text or ''


#: How many times a turn that carries a whole measurement is re-asked after
#: running out of its budget, before the run gives up and leaves the work to a
#: resume.
_DEADLINE_REASKS = 2


def _retry_on_deadline(call, *, label: str, reasks: int = _DEADLINE_REASKS):
    '''Re-ask a turn that ran out of its budget, rather than scoring the gap.

    Where a spent deadline is scored depends on how much rides on the one call.
    A held-out question is one of ninety, so losing it costs a ninetieth and
    counting it wrong is proportionate. A milestone summary is the entire rule
    score for that milestone, and an exploration round is a quarter of the
    whole budget -- scoring those as silence would throw away far more than
    the model failed to produce, and would read as a collapse it never had.

    So these are re-asked, and if the budget goes on being spent the deadline
    is allowed to escape: the run stops with its checkpoint intact and picks up
    from the last milestone instead of recording a measurement it never made.
    '''
    for attempt in range(reasks + 1):
        started = time.time()
        try:
            return call()
        except Exception as exc:
            elapsed = time.time() - started
            if not _answer_deadline_expired(exc, elapsed) or attempt == reasks:
                raise
            print(f'  TIMEOUT {label} ({elapsed:.0f}s): out of budget, '
                  f're-asking {attempt + 1}/{reasks}')


def _chat_within_budget(session: AgentSession, prompt: str, label='') -> str:
    return _retry_on_deadline(
        lambda: _chat_turn(session, prompt, label=label).text or '',
        label=label or 'turn')


def _alien_tool_definition(max_shows: int) -> dict:
    '''The interpreter, exposed so the model runs its own experiments.'''
    return {
        'name': ALIEN_TOOL_NAME,
        'description': (
            '在真实的 AlienCode 解释器中执行一段代码，返回真实执行输出。\n'
            '你可以在一轮内多次调用它来做实验：先跑一个探针，看到结果后再决定下一个探针。\n'
            f'限制：每次调用最多 {MAX_CODE_LINES} 行代码、'
            f'最多 {max_shows} 个观测值（EMIT 的每个参数算一个观测值，'
            f'循环内每个按 {_LOOP_WEIGHT}× 计数）、'
            f'返回最多 {MAX_OUTPUT_LINES} 行输出。\n'
            f'超过观测值上限的调用会被整个拒绝、不执行任何代码，请拆成多次调用。\n'
            f'一轮最多调用 {MAX_TOOL_CALLS_PER_ROUND} 次。'
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'code': {
                    'type': 'string',
                    'description': (
                        '要执行的 AlienCode 源码。用 EMIT(...) 输出你想观察的值。'
                    ),
                },
            },
            'required': ['code'],
        },
    }


def _exec_tool_call(
        call, *, label: str, max_shows: int | None,
        locked_code: str | None = None) -> dict:
    '''Run one tool call under the same limits as a code block.

    Returns a record of what was executed; ``result`` is the payload handed
    back to the model.

    ``locked_code`` is the passive/random control: the call still travels the
    provider's native tool path, so reasoning state, call ids and the result
    round trip are identical to a self run, but the harness substitutes the
    probe. What the model asked for is kept in ``model_code`` so the
    substitution stays auditable rather than silent.
    '''
    arguments = call.arguments if isinstance(call.arguments, dict) else {}
    model_code = str(arguments.get('code') or '')
    raw_code = model_code if locked_code is None else locked_code

    if not raw_code.strip():
        message = '[ToolError] 缺少 code 参数，未执行任何代码。'
        print(f'  [Tool {call.name}] {label}: 缺少 code 参数')
        return {
            'name': call.name,
            'raw_code': raw_code,
            'model_code': model_code,
            'locked': locked_code is not None,
            'code': '',
            'exec_output': message,
            'show_count_original': 0,
            'show_count_executed': 0,
            'is_error': True,
            'result': ToolResult(
                call_id=call.call_id,
                output=message,
                name=call.name,
                is_error=True,
            ),
        }

    code = raw_code
    orig_shows = kept_shows = _count_show_weighted(code)
    if max_shows is not None and max_shows > 0 and orig_shows > max_shows:
        # A submission over the per-call cap is rejected outright rather than
        # part-executed. Running the statements that happen to fit hands back a
        # half-answered experiment the model never designed, and the model
        # cannot tell which half it got.
        if locked_code is None:
            message = (
                f'[ToolError] 这次调用需要 {orig_shows} 个观测值，'
                f'超过单次上限 {max_shows}（EMIT 的每个参数算一个观测值，'
                f'循环内每个按 {_LOOP_WEIGHT}× 计数）。\n'
                '本次没有执行任何代码，也没有返回任何观测。'
                '请把实验拆成多次调用后重试。'
            )
            print(
                f'  [EMIT Limit] {label}: rejected '
                f'{orig_shows} > {max_shows}')
            return {
                'name': call.name,
                'raw_code': raw_code,
                'model_code': model_code,
                'locked': False,
                'code': '',
                'exec_output': message,
                'show_count_original': orig_shows,
                'show_count_executed': 0,
                'is_error': True,
                'result': ToolResult(
                    call_id=call.call_id,
                    output=message,
                    name=call.name,
                    is_error=True,
                ),
            }
        # A replayed donor probe is recorded evidence, not a request the model
        # made, so the per-call cap does not apply to it. Rejecting it would
        # leave the matched control with nothing, and truncating it would show
        # the control less evidence than the run it is paired against.
        print(
            f'  [EMIT Limit] {label}: donor probe replayed uncapped '
            f'({orig_shows} > {max_shows})')
    code, warns = _enforce_code_limits(code)
    for warning in warns:
        print(f'  [Code Limit] {label}: {warning}')

    exec_output = alien_exec(code)
    truncated = _truncate_output(exec_output)
    payload = truncated
    if orig_shows > kept_shows:
        payload += (
            f'\n[注意] 本轮锁定 probe 超过 {max_shows} 个观测值上限，'
            '只执行了前面的部分。')
    for warning in warns:
        payload += f'\n[注意] {warning}'

    return {
        'name': call.name,
        'raw_code': raw_code,
        'model_code': model_code,
        'locked': locked_code is not None,
        'code': code,
        'exec_output': exec_output,
        'show_count_original': orig_shows,
        'show_count_executed': kept_shows,
        'is_error': False,
        'result': ToolResult(
            call_id=call.call_id,
            output=payload,
            name=call.name,
        ),
    }


def _closed_book_fork(session: AgentSession) -> AgentSession:
    '''Branch the conversation without the interpreter tool.

    Rule summaries and held-out tests must be answered from the rules the
    model inferred during exploration. If these branches inherited
    ``run_aliencode`` the model could simply execute the task and read off the
    answer, which would measure nothing.
    '''
    return _require_agent_runtime().fork_closed_book(
        session,
        context=PhaseContext(
            sandbox='code',
            phase='closed_book',
            track=EVAL_TRACK,
            framework=FRAMEWORK,
        ),
    )


def _tool_call_records(executions) -> list[dict]:
    '''Shrink tool executions to a checkpoint-safe record.'''
    records = []
    for idx, record in enumerate(executions or [], 1):
        row = {
            'index': idx,
            'model_code': record.get('model_code', ''),
            'locked': bool(record.get('locked')),
            'code': record['code'],
            'raw_code': record['raw_code'],
            'exec_output': _cap_exec_output(record['exec_output']),
            'show_count_original': record['show_count_original'],
            'show_count_executed': record['show_count_executed'],
            'is_error': record['is_error'],
        }
        records.append(row)
    return records


class ToolRound(NamedTuple):
    '''What one tool-driven round produced.

    ``text`` joins every assistant message in the round, since the model
    narrates across several of them: it states a prediction, probes it, then
    summarises.
    '''

    text: str
    executions: list[dict]
    model_calls: int


# How many further turns a model that ignores tool_choice=none is answered for
# before the round gives up on it. Doubao asks for roughly fifty seed calls when
# left to itself, so a model meeting a one-call locked budget can take several
# turns to accept that the tool is closed.
_TOOL_CLOSE_OUT_TURNS = 8


def _close_out_tool_loop(session: AgentSession, results, *, label: str,
        budget: int):
    '''Answer outstanding calls until the model stops making them.

    Returns the first response that carries no tool call. Leaving a round with
    one still pending is what breaks the next fork, so every call is answered
    even once the round has decided to stop executing them: the reply is the
    budget-exhausted error, never a probe result.
    '''
    response = _chat_turn(
        session, '', label=f'{label} tool-results (budget spent)',
        tool_results=results, extra_overrides={'tool_choice': 'none'})
    for attempt in range(1, _TOOL_CLOSE_OUT_TURNS + 1):
        if not response.tool_calls:
            return response
        print(
            f'  [Tool] {label}: 模型在预算用尽后仍调用工具，'
            f'回以错误 {attempt}/{_TOOL_CLOSE_OUT_TURNS}')
        refusals = [
            ToolResult(
                call_id=call.call_id,
                output=(
                    f'[ToolError] 本轮工具调用预算已用完（上限 {budget} 次）。'
                    '工具已关闭，请直接根据已有结果给出你的结论。'
                ),
                name=call.name,
                is_error=True,
            )
            for call in response.tool_calls
        ]
        response = _chat_turn(
            session, '', label=f'{label} tool-results (closing {attempt})',
            tool_results=refusals, extra_overrides={'tool_choice': 'none'})
    if response.tool_calls:
        raise RuntimeError(
            f'{label}: model still called {ALIEN_TOOL_NAME} after '
            f'{_TOOL_CLOSE_OUT_TURNS} turns with the tool closed; leaving the '
            'round here would strand its pending calls')
    return response


def _run_tool_round(
        session: AgentSession,
        prompt: str,
        *,
        label: str = '',
        max_shows: int | None = None,
        max_calls: int | None = None,
        locked_codes: list[str] | None = None) -> ToolRound:
    '''Let the model drive the interpreter until it stops calling the tool.

    ``executions`` holds one record per tool call, in the order the model made
    them.

    Under ``locked_codes`` the model still issues the calls, but the harness
    substitutes the probe of the i-th call with the i-th locked probe and the
    round ends once the list is spent. This is what keeps the passive and
    random controls on the same wire protocol as the treatment: the difference
    between them is which probe runs, not whether a tool was used.
    '''
    budget = (
        len(locked_codes) if locked_codes
        else (max_calls or MAX_TOOL_CALLS_PER_ROUND)
    )
    # Forcing a call is what keeps a locked round from silently delivering no
    # probe, but two upstreams reject the parameter outright: qwen, and
    # DeepSeek whenever thinking is on ("Thinking mode does not support this
    # tool_choice"). Match DeepSeek by vendor rather than by one route's
    # prefix -- the same model is reachable through the gateway's Messages
    # door and GatewayA's Responses door, and only the first was excluded here,
    # so a locked round on the GatewayA route died on its first seed call.
    locked_tool_override = (
        None
        if (
            "qwen" in MODEL.lower()
            or "deepseek" in MODEL.lower()
        )
        else {"tool_choice": "required"}
    )
    response = _chat_turn(
        session,
        prompt,
        label=label,
        extra_overrides=(locked_tool_override if locked_codes else None),
    )
    executions: list[dict] = []
    texts: list[str] = []
    model_calls = 1
    grace = 0
    rounds = 0
    nudges = 0

    # A locked control must deliver the donor evidence even when the model's
    # first response is prose-only. The ordinary loop below cannot nudge that
    # case because it has no tool call to enter on.
    while (
        locked_codes
        and not response.tool_calls
        and nudges < _MAX_LOCKED_DELIVERY_RETRIES
    ):
        nudges += 1
        if response.text and response.text.strip():
            texts.append(response.text.strip())
        response = _chat_turn(
            session,
            _NO_TOOL_RETRY,
            label=f"{label} (locked retry {nudges})",
            extra_overrides=locked_tool_override,
        )
        model_calls += 1

    while response.tool_calls:
        rounds += 1
        if response.text and response.text.strip():
            texts.append(response.text.strip())

        remaining = max(0, budget - len(executions))
        allowed = response.tool_calls[:remaining]
        refused = response.tool_calls[len(allowed):]
        results = []

        for call in allowed:
            if call.name != ALIEN_TOOL_NAME:
                message = f'[ToolError] 未知工具 {call.name}'
                print(f'  [Tool] {label}: {message}')
                results.append(ToolResult(
                    call_id=call.call_id,
                    output=message,
                    name=call.name,
                    is_error=True,
                ))
                continue
            record = _exec_tool_call(
                call, label=label, max_shows=max_shows,
                locked_code=(
                    locked_codes[len(executions)] if locked_codes else None
                ),
            )
            results.append(record.pop('result'))
            executions.append(record)
            _require_agent_runtime().after_environment_feedback(
                PhaseContext(
                    sandbox='code',
                    phase='explore',
                    label=label,
                    track=EVAL_TRACK,
                    framework=FRAMEWORK,
                ),
                probe={'tool': call.name, 'code': record.get('code', '')},
                feedback=record.get('exec_output', ''),
            )
            print(
                f'  [Tool {len(executions)}/{budget}] {label}\n'
                f'{record["code"]}\n'
                f'  → {_truncate_output(record["exec_output"])}')

        for call in refused:
            message = (
                f'[ToolError] 本轮工具调用预算已用完（上限 {budget} 次）。'
                '请不要再调用工具，直接根据已有结果给出你的结论。'
            )
            results.append(ToolResult(
                call_id=call.call_id,
                output=message,
                name=call.name,
                is_error=True,
            ))

        extra_overrides = None
        suffix = ''
        # Rounds count too: calls the harness rejects never reach `executions`,
        # so a model asking for an unknown tool could otherwise loop untouched.
        # A locked round is exempt until its donor probes are all delivered,
        # since closing tools early there would truncate the matched evidence;
        # the wider round cap below still bounds the loop.
        spent = len(executions) >= budget
        exhausted = rounds >= (
            budget + _MAX_LOCKED_DELIVERY_RETRIES if locked_codes else budget
        )
        if spent or exhausted:
            # Without forcing text here, a model that keeps calling would spin
            # forever, and abandoning the loop with tool calls still pending
            # would break every later fork.
            grace += 1
            extra_overrides = {'tool_choice': 'none'}
            suffix = ' (budget spent)'
            if grace > _TOOL_LOOP_GRACE:
                # Some upstreams keep calling straight through
                # tool_choice=none. The round already holds all the evidence it
                # is allowed by this point, so the only thing still in dispute
                # is whether the model will stop asking, and killing the run
                # over that discards every milestone already scored. Answer the
                # outstanding calls until it settles and end the round instead.
                response = _close_out_tool_loop(
                    session, results, label=label, budget=budget)
                model_calls += 1
                break
        elif locked_codes:
            extra_overrides = locked_tool_override

        # GatewayB can emit textual XML tool syntax under tool_choice=none, and
        # its old Messages shim rejected that choice whenever tools remained.
        # Removing tools for the forced-summary turn expresses the intended
        # closed-tool state on both supported routes.
        if (extra_overrides
                and extra_overrides.get('tool_choice') == 'none'
                and MODEL.startswith(
                    ('messages/api_gateway_b_', 'api_gateway_b_'))):
            with session.using_tools([]):
                response = _chat_turn(
                    session,
                    '',
                    label=f'{label} tool-results{suffix}',
                    tool_results=results,
                )
        else:
            response = _chat_turn(
                session,
                '',
                label=f'{label} tool-results{suffix}',
                tool_results=results,
                extra_overrides=extra_overrides,
            )
        model_calls += 1

        # A locked round has to deliver its whole probe list, otherwise the
        # control sees less evidence than the run it is matched against and
        # the contrast stops being a probe-choice comparison. A model that
        # stops calling early is asked again rather than left short.
        if locked_codes and not response.tool_calls:
            while (len(executions) < budget
                   and nudges < _MAX_LOCKED_DELIVERY_RETRIES):
                nudges += 1
                print(
                    f'  [Locked] {label}: 模型未调用工具，'
                    f'重试 {nudges}/{_MAX_LOCKED_DELIVERY_RETRIES}')
                if response.text and response.text.strip():
                    texts.append(response.text.strip())
                response = _chat_turn(
                    session, _NO_TOOL_RETRY,
                    label=f'{label} (locked retry {nudges})',
                    extra_overrides=locked_tool_override)
                model_calls += 1
                if response.tool_calls:
                    break

    if response.text and response.text.strip():
        texts.append(response.text.strip())
    if locked_codes and len(executions) < budget:
        print(
            f'  [Locked] {label}: 只投递了 {len(executions)}/{budget} 个锁定探针')
    return ToolRound(
        text='\n\n'.join(texts),
        executions=executions,
        model_calls=model_calls,
    )


def _send_and_exec(conversation, session: AgentSession, prompt: str, *,
        label='', max_shows: int | None = None,
        locked_codes: list[str] | None = None):
    '''Send a prompt and execute whatever code came back.

    Returns ``(assistant_msg, code, exec_output, executions)``. Under tool mode
    ``executions`` has one entry per tool call and ``code`` concatenates them;
    otherwise the model's single code block is executed and ``executions`` is
    empty.

    ``locked_codes`` substitutes the executed probe the same way the passive
    control does, leaving the model's own prediction intact.
    '''

    conversation.append({'role': 'user', 'content': prompt})

    if TOOL_MODE:
        round_ = _run_tool_round(
            session, prompt, label=label, max_shows=max_shows,
            locked_codes=locked_codes)
        executions = list(round_.executions)
        replies = [round_.text]
        # A reply that describes code without calling the tool executes
        # nothing, so the model never sees a real output and the phase teaches
        # it nothing. Ask again rather than record an empty round, the same way
        # exploration does.
        retries = 0
        while not executions and retries < _MAX_NO_CODE_RETRIES:
            retries += 1
            print(
                f'  [Tool] {label}: 模型未调用工具，重试 '
                f'{retries}/{_MAX_NO_CODE_RETRIES}'
            )
            conversation.append(
                {'role': 'user', 'content': _NO_TOOL_RETRY}
            )
            retry = _run_tool_round(
                session, _NO_TOOL_RETRY, label=label, max_shows=max_shows,
                locked_codes=locked_codes)
            executions = list(retry.executions)
            replies.append(retry.text)
        assistant_msg = '\n\n'.join(
            text for text in replies if text
        ) or '（模型返回了空回复）'
        conversation.append(_assistant_turn(assistant_msg))
        if not executions:
            return assistant_msg, '', '(未调用工具)', executions
        combined = '\n\n'.join(
            f'# 第 {idx} 次调用\n{record["code"]}'
            for idx, record in enumerate(executions, 1)
        )
        return assistant_msg, combined, executions[-1]['exec_output'], executions

    assistant_msg = _chat(session, prompt, label=label)
    if not assistant_msg.strip():
        assistant_msg = '（模型返回了空回复）'
    conversation.append(_assistant_turn(assistant_msg))

    raw_code = extract_code(assistant_msg)
    if not raw_code:
        return assistant_msg, '', '(无代码)', []

    code = raw_code
    if max_shows is not None and max_shows > 0:
        code, orig, kept = _truncate_code_by_show(code, max_shows)
        if orig > kept:
            print(f'''  [EMIT Limit] {label}: {orig} → {kept} (max={max_shows})''')
    code, warns = _enforce_code_limits(code)
    for w in warns:
        print(f'''  [Code Limit] {label}: {w}''')

    exec_output = alien_exec(code)
    return assistant_msg, code, exec_output, []






_JUDGE_SYSTEM = '你是 AlienCode 规则评判器。判断以下两个 S-Expression 是否表达了相同的语义变换。\n操作符语义相同即可（如 (+ a b) ≡ (+ b a)）。\n变量名不影响判定。数值必须精确匹配。\nIDENTITY 表示操作与文档描述一致（无偏差），\n如果一边是 IDENTITY 另一边是与文档描述一致的具体表达式，也算等价。\n回答 EQUIVALENT 或 NOT_EQUIVALENT，附一句理由。'










def _llm_judge_sexpr(rule_id: str, canonical: str, model_expr: str) -> bool:
    rule_info = RULES_BY_KEY.get(next((r['key'] for r in ALIEN_RULE_SPECS if r['id'] == rule_id), ''), {})

    desc = rule_info.get('actual', rule_info.get('name', rule_id))

    prompt = f'''规则: {rule_id} — {desc}\n规范答案: {canonical}\n模型答案: {model_expr}\n\n这两个表达式是否描述了相同的语义变换？\n回答 EQUIVALENT 或 NOT_EQUIVALENT，附一句理由。'''






    messages = [
        {'role': 'system', 'content': _JUDGE_SYSTEM},
        {'role': 'user', 'content': prompt}]

    try:
        raw = chat(messages, model=JUDGE_MODEL, temperature=0.0, max_tokens=500)
        return 'EQUIVALENT' in raw and 'NOT_EQUIVALENT' not in raw
    except Exception as exc:
        print(f'''  [Judge Error] {rule_id}: {exc}''')
        return False


def _score_sexpr_discovery(summary: str, use_llm_fallback: bool = True) -> dict[str, bool]:
    model_rules = _extract_sexpr_rules(summary)
    scores = {}

    print('\n[S-Expression Rule Scoring]:')
    for rule_id, canonical in GROUND_TRUTH_SEXPR.items():
        model_expr = model_rules.get(rule_id, '').strip()

        if not model_expr or model_expr in ('UNKNOWN', '??'):
            scores[rule_id] = False
            print(f'''  0 {rule_id}: NOT_FOUND (missing/unknown)''')
            continue

        is_identity_rule = rule_id in _IDENTITY_RULE_IDS

        # The milestone prompt asks for S-expressions and shows the template, so
        # an infix answer like `op a b` is not a near miss to be repaired: it is
        # a report that did not follow the format it was given. Reading it
        # anyway would grade a different task than the one the model was set.
        if model_expr == 'IDENTITY' and not is_identity_rule:
            scores[rule_id] = False
            print(f'''  0 {rule_id}: FALSE_IDENTITY (claimed IDENTITY for non-identity rule)''')
            continue

        if match_rule(model_expr, canonical, rule_id=rule_id):
            scores[rule_id] = True
            tag = ' (identity rule)' if is_identity_rule else ''
            print(f'''  1 {rule_id}: MATCH (programmatic){tag}''')
            continue

        if use_llm_fallback:
            equiv = _llm_judge_sexpr(rule_id, canonical, model_expr)
            scores[rule_id] = equiv
            icon = '1' if equiv else '0'
            print(f'''  {icon} {rule_id}: LLM_JUDGE ({"EQUIVALENT" if equiv else "NOT_EQUIVALENT"})'''); continue

        scores[    rule_id] = False
        print(f'''  0 {  rule_id}: NO_MATCH (no fallback)''')

    return scores


def _score_freetext_discovery(summary: str) -> dict[str, bool]:
    rule_keys = [r['key'] for r in ALIEN_RULE_SPECS]
    rule_lines, json_lines = [], []
    for r in ALIEN_RULE_SPECS:
        rule_lines.append(f'''  - key="{r["key"]}" | L{r["layer"]} | {r["name"]} | 标准: {r["std"]} → 实际: {r["actual"]}''')



        json_lines.append(f'''  "{r["key"]}": {{"found": true/false, "reason": "简要依据"}}''')

    prompt = '以下是需要检查的真实规则偏差：\n' + '\n'.join(rule_lines) + '\n\n以下是被测模型的规则总结：\n---\n' + summary + '\n---\n\n请逐条判断这份总结是否正确描述了每条规则的实际行为。\n只要语义等价即可算发现；如果归因明显错误则不算。\n\n请严格输出 JSON：\n{\n' + ',\n'.join(json_lines) + '\n}'






    judge_system = '你是一个严谨的规则评审器。你需要判断被测模型的总结文本中，是否正确描述了 AlienCode 环境中各操作的实际运行时行为。请做语义判断。'




    messages = [
        {'role': 'system', 'content': judge_system},
        {'role': 'user', 'content': prompt}]

    try:
        raw = chat(messages, model=JUDGE_MODEL,
                                   response_format={'type': 'json_object'},
                   temperature=0.0, max_tokens=4000)
        payload = raw
        m = re.search('```json\\s*\\n?(.*?)```', raw, re.DOTALL)
        if m:
            payload = m.group(1)
        data = json.loads(payload)
        scores = {}
        for key in rule_keys:
            entry = data.get(key, {})
            scores[key] = bool(entry.get('found', False))
        return scores
    except Exception as exc:
        print(f'''[Judge Error] {type(exc).__name__}: {exc}''')
        return {key: False for key in rule_keys}


def _score_rule_precision(summary: str) -> dict[str, bool]:
    if ':=' in summary:
        return _score_sexpr_discovery(
            summary, use_llm_fallback=not PROTOCOL_V2)
    if PROTOCOL_V2:
        print('  [Scoring] No S-Expression format detected; v2 disables free-text LLM judge')
        return {rule_id: False for rule_id in GROUND_TRUTH_SEXPR}

    print('  [Scoring] No S-Expression format detected, using free-text LLM Judge')
    ft_scores = _score_freetext_discovery(summary)
    scores = {}
    for r in ALIEN_RULE_SPECS:
        if r['id'] in GROUND_TRUTH_SEXPR:
            scores[r['id']] = ft_scores.get(r['key'], False)
    return scores






def _run_seed_phase(state, results):
    if PROTOCOL_V2:
        print(f'''\n{"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"}\n  Phase: fixed calibration demos (AlienCode v2) — {len(protocol_v2.DEMOS)} demos\n{"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"}''')
        for row in protocol_v2.demo_rows(alien_exec):
            state['round'] += 1
            results.append({
                'round': state['round'],
                'phase': 'fixed_demo',
                'task_id': row['id'],
                'band': row['band'],
                'prompt': row['prompt'],
                'code': row['code'],
                'exec_output': _cap_exec_output(row['output']),
                'model_prediction': None,
                'classification': 'FIXED_DEMO',
                'rules': row['rules'],
                'time_seconds': 0.0,
                'tool_mode': False,
                'tool_calls': [],
                'protocol_version': protocol_v2.PROTOCOL_VERSION,
            })
            print(
                f'''[{row["id"]} | {row["band"]}]\n{row["code"]}\n→ {_truncate_output(row["output"])}'''
            )
        state['pending_feedback'] = None
        return

    print(f'''\n{"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"}\n  Phase: 播种轮 (Seed) — {len(SEED_TASKS)} 轮\n{"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"}''')
    pending_feedback = None
    for idx, task in enumerate(SEED_TASKS):
        state['round'] += 1
        prompt = task['prompt']
        if TOOL_MODE:
            # The seed wording asks the model to "give the code", which under
            # the tool protocol some models satisfy with a markdown block and
            # no tool call -- nothing runs, and they never see a real output.
            prompt = prompt.replace('再给出代码', f'再调用 {ALIEN_TOOL_NAME} 工具执行')
        if idx == 0:
            prompt += SEED_LIMIT_NOTE_TOOL if TOOL_MODE else SEED_LIMIT_NOTE
        if pending_feedback:
            prompt = pending_feedback + '\n\n---\n\n' + prompt
            pending_feedback = None
        print(f'''\n{"========================================================================"}\n  Seed Round {state["round"]} · {task["id"]}\n{"========================================================================"}''')
        t0 = time.time()
        # Under LOCKED_SEED the model still predicts, but the harness runs the
        # task's own canonical program instead of whatever the model asked to
        # run, so every system carries identical seed evidence into M0.
        seed_locked = (
            [LOCKED_SEED_CODES[task['id']]]
            if LOCKED_SEED and TOOL_MODE and task['id'] in LOCKED_SEED_CODES
            else None
        )
        assistant_msg, code, exec_output, executions = _send_and_exec(
            state['conversation'], state['session'], prompt,
            label=f'''Seed {task["id"]}''',
            max_shows=MAX_SEED_SHOWS,
            locked_codes=seed_locked)

        truncated_output = _truncate_output(exec_output)
        elapsed = time.time() - t0
        prediction = extract_prediction(assistant_msg, tool_protocol=TOOL_MODE)
        classification = _classify_disorientation(prediction, task['real_world_answer'], task['expected'], assistant_msg)

        results.append({
                     'round': state['round'], 'phase': 'seed', 'task_id': task['id'],
                    'code': code, 'exec_output': _cap_exec_output(exec_output),
                                'model_prediction': prediction, 'classification': classification,
                     'rules': task['rules'], 'time_seconds': elapsed,
                     'tool_mode': TOOL_MODE,
                     'tool_calls': _tool_call_records(executions)})

        print(f'''[Model]\n{assistant_msg[:800]}\n[AlienCode Exec]\n{truncated_output}''')
        print(f'''[Time] {elapsed:.1f}s  [Predicted] {prediction}  [Class] {classification}''')

        if TOOL_MODE:
            # The model already saw every result as a tool response; replaying
            # it as a user message would only duplicate the context.
            pending_feedback = None
            continue

        pending_feedback = f'''你的代码执行结果是：{truncated_output}\n（你的预测是 {prediction}，实际结果是 {truncated_output}）'''


        if prediction and not values_equal(prediction, exec_output):
            pending_feedback += '\n结果与你的预期不同。'
    state['pending_feedback'] = pending_feedback






_EXPLORE_NO_CODE_RETRY = '⚠️ 你的回复中没有包含 AlienCode 代码。\n请不要输出规则总结或纯文字分析。现在不是总结阶段。\n\n请 **必须** 写出可执行的 AlienCode 测试代码，放在 ```aliencode``` 代码块中。\n请测试你还不确定的操作。参考手册中列出了所有可用操作。\n'

_EXPLORE_NO_TOOL_RETRY = f'''⚠️ 你本轮没有调用 {ALIEN_TOOL_NAME} 工具，因此没有得到任何真实执行结果。\n请不要输出规则总结或纯文字分析。现在不是总结阶段。\n\n请 **必须** 调用 {ALIEN_TOOL_NAME} 工具来执行 AlienCode 测试代码。\n请测试你还不确定的操作。参考手册中列出了所有可用操作。\n'''






_MAX_NO_CODE_RETRIES = 2

# A locked control round is only comparable to its donor if every donor probe
# is delivered, so a model that stops calling early has to be asked back many
# more times than an ordinary round warrants. Two nudges left doubao-seed-2.1
# three probes short of the donor in the last exploration loop, where models
# tend to switch to summarising, and that silently broke the matched contrast.
_MAX_LOCKED_DELIVERY_RETRIES = 8

# Sent when a tool-mode reply contains no tool call at all. Models that were
# asked to "write code" often answer with a markdown block, which the tool
# protocol never executes.
_NO_TOOL_RETRY = (
    f'⚠️ 你刚才没有调用 {ALIEN_TOOL_NAME} 工具，所以代码没有真正执行，'
    f'你看到的不是真实输出。\n'
    f'请现在调用 {ALIEN_TOOL_NAME} 工具执行代码（参数 code 为 AlienCode 源码），'
    f'不要只把代码写在回复文本里。\n'
)

# The mirror image, for the closed-book turns. Those carry no tool, but the
# history is full of tool calls, and some models answer out of that habit --
# then stop, waiting for a result that will never come. Without this they score
# zero for never answering. Two shapes show up: a tool-call template printed as
# text, and a real tool call issued against a tool the request never offered.
# The separator is optional because vendors spell their own marker their own
# way: doubao emits a bare ``<|FunctionCallBegin|>`` token, and requiring the
# underscore let that one through as an ordinary unsubmitted answer.
_LEAKED_TOOL_CALL = re.compile(
    r'(?:invoke|tool[_\s-]?call|function[_\s-]?call)[^\n]{0,40}'
    + re.escape(ALIEN_TOOL_NAME),
    re.IGNORECASE,
)

_CLOSED_BOOK_RETRY = (
    f'⚠️ 本轮没有可用工具，{ALIEN_TOOL_NAME} 不会被执行，'
    f'你上面写的调用不会返回任何结果。\n'
    f'请直接根据你已经总结的规则给出最终答案：'
    f'需要写代码的题目，请把 AlienCode 代码放在 ```aliencode 代码块里；'
    f'需要预测输出的题目，请直接写出预测值。\n'
)

# Answers a call the request never offered. The protocol will not take another
# turn while that call hangs unanswered, so the refusal has to travel as its
# result rather than as a new user message.
_CLOSED_BOOK_TOOL_DENIAL = (
    f'[ToolError] 本轮为闭卷作答，{ALIEN_TOOL_NAME} 不可用，'
    f'本次调用没有被执行，也不会有执行结果。\n'
    + _CLOSED_BOOK_RETRY
)

# A turn that thought until the budget ran out and said nothing. The transport
# retries a genuinely empty completion, so what is left here is the model
# choosing to end the turn silently -- scored as a wrong answer for a question
# it never answered unless it is asked again.
_SILENT_ANSWER_RETRY = (
    '⚠️ 上一轮没有任何输出内容（可能是思考占满了输出预算）。'
    '请把预算留给答案本身，直接给出最终答案：'
    f'需要写代码的题目，请把 AlienCode 代码放在 ```aliencode 代码块里；'
    '需要预测输出的题目，请直接写出预测值。\n'
)

# A turn that reasoned all the way to an answer and then stopped short of
# submitting one, usually by announcing a verification step against a tool the
# closed book does not offer. It is neither silent nor a leaked call, so the
# earlier two shapes miss it and the answer is scored wrong for a question the
# model very nearly answered.
_UNSUBMITTED_ANSWER_RETRY = (
    '⚠️ 上一轮的回答里没有可评分的最终答案。本轮为闭卷作答，'
    '没有工具可以执行或验证，请不要等待执行结果。\n'
    '请直接给出最终答案：需要写代码的题目，请把 AlienCode 代码放在 '
    '```aliencode 代码块里；需要预测输出的题目，请直接写出预测值。\n'
)


def _unanswered(response: str, task: dict | None) -> bool:
    """Whether the grader would find nothing to score in this reply."""

    if task is None:
        return False
    if task.get('question_type') == 'predict':
        return not extract_prediction(response, tool_protocol=TOOL_MODE)
    return not extract_code(response)


def _needs_another_ask(turn, response: str, task: dict | None = None) -> bool:
    """Whether this graded turn holds nothing that can be scored.

    Four ways a turn arrives with no answer in it: a real call against a tool
    the closed book does not offer, the text of such a call printed instead of
    made, silence, and a reply that reasons toward an answer without ever
    submitting one. All four score as wrong for a question the model has not
    actually answered, so all four are asked again.
    """

    if not response.strip() and not turn.tool_calls:
        return True
    if TOOL_MODE and (
        turn.tool_calls or _LEAKED_TOOL_CALL.search(response)
    ):
        return True
    return _unanswered(response, task)


def _load_control_bank() -> dict[str, dict]:
    '''Load a donor JSONL transcript keyed by call label.'''
    global _CONTROL_BANK_CACHE
    if _CONTROL_BANK_CACHE is not None:
        return _CONTROL_BANK_CACHE
    if not _CONTROL_BANK_PATH:
        raise RuntimeError('EVAL_CONTROL_MODE=passive requires --control-bank or EVAL_CONTROL_BANK pointing to a standard-run transcript JSONL')



    bank = {}
    with open(_CONTROL_BANK_PATH, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            label = str(rec.get('label', ''))
            if label.startswith('Explore ') and '(retry' not in label:
                bank[label] = rec
    _CONTROL_BANK_CACHE = bank
    return bank


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1048576),     b''):
            digest.update(chunk)
    return digest.hexdigest()


def _preflight_control_bank() -> dict | None:
    '''Fail before paid calls if a passive/random donor is not comparable.'''
    global _CONTROL_PREFLIGHT, _CONTROL_PREFLIGHT
    if _CONTROL_MODE not in {'random', 'passive'}:
        _CONTROL_PREFLIGHT = None
        return None
    if not _CONTROL_MANIFEST_PATH:
        raise RuntimeError('passive/random controls require --control-manifest so donor model, sandbox, version, and hashes can be verified')



    expected = {f'''Explore {loop}-{round_idx}''' for loop in range(1, N_EXPLORE_LOOPS + 1) for round_idx in range(1, N_EXPLORE_ROUNDS + 1)}




    bank = _load_control_bank()
    labels = set(bank)
    if labels != expected:
        raise RuntimeError(f'''Control donor explore structure mismatch: missing={sorted(expected - labels)}, unexpected={sorted(labels - expected)}''')




    retry_labels = []
    with open(_CONTROL_BANK_PATH, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            label = str(json.loads(line).get('label', ''))
            if label.startswith('Explore ') and '(retry' in label:
                retry_labels.append(label)
    if retry_labels:
        raise RuntimeError(f'''Control donor contains retry explore calls: {retry_labels}''')


    aliases = {MODEL, MODEL_SHORT, MODEL.rsplit('_', 1)[-1]}
    donor_models = {str(rec.get('model', '')) for rec in bank.values()}
    if not donor_models:
        raise RuntimeError('Control donor names no model')
    # Passive replays the donor's probes verbatim, so a donor from another
    # model would hand this one someone else's experiments and the arm
    # would no longer be "the same probes, not chosen live". Random
    # replays nothing: it takes only the per-call probe *count* and
    # synthesises the probes, so one shared profile across a cohort is
    # deliberate -- it is what makes the arm a common baseline instead of
    # one arbitrary sequence per model. Every other check below still
    # applies to both, including the digests the manifest is pinned to.
    if (_CONTROL_MODE != 'random'
            and any(model not in aliases for model in donor_models)):
        raise RuntimeError(f'''Control donor model mismatch: donor={sorted(donor_models)}, target={sorted(x for x in aliases if x)}''')


    empty = [
        label for label, rec in bank.items()
        if not str(rec.get('response', '')).strip()
        and not (rec.get('codes') or [])
    ]



    if empty:
        raise RuntimeError(f'''Control donor has empty responses: {empty}''')
    no_probe = [label for label, rec in bank.items() if _donor_show_count(label) <= 0]



    if no_probe:
        raise RuntimeError(f'''Control donor has no executable probes: {no_probe}''')

    transcript_digest = _sha256_file(_CONTROL_BANK_PATH)
    manifest_verified = False
    manifest_entry = None
    with open(_CONTROL_MANIFEST_PATH, encoding='utf-8') as f:
        manifest = json.load(f)
    evaluator_stamp = 'AlienCode-self'
    if manifest.get('evaluator_stamps', {}).get('AlienCode') != evaluator_stamp:

        raise RuntimeError('Control donor evaluator stamp is missing or incompatible')

    rule_digest = _sha256_file(os.path.join(_THIS_DIR, 'engine.py'))
    if manifest.get('rule_digests', {}).get('AlienCode') != rule_digest:
        raise RuntimeError('Control donor hidden-rule digest is missing or incompatible')

    manifest_dir = os.path.dirname(os.path.abspath(_CONTROL_MANIFEST_PATH))

    target_path = os.path.realpath(_CONTROL_BANK_PATH)
    for entry in manifest.get('donors', []):
        entry_path = os.path.realpath(os.path.join(manifest_dir, entry['transcript']))

        if entry_path == target_path:
            manifest_entry = entry
            break
    if manifest_entry is None:
        raise RuntimeError(f'''Control donor is absent from manifest: {_CONTROL_BANK_PATH}''')

    checks = {
                  'sha256': transcript_digest,
                       'model_short': MODEL_SHORT,
                   'sandbox': 'AlienCode',
                   'episode': 'code_default',
                         'explore_loops': N_EXPLORE_LOOPS,
                           'rounds_per_loop': N_EXPLORE_ROUNDS,
                                       'manual_sha256': hashlib.sha256(REFERENCE_MANUAL.encode()).hexdigest(),
                              'environment_sha256': _sha256_file(os.path.join(_THIS_DIR, 'execution.py'))}



    if _CONTROL_MODE == 'random':
        # The donor is another model's on purpose here: random borrows its
        # probe volume and generates its own probes, so one shared profile
        # is what makes the arm a common baseline. Which model it came
        # from still has to be recorded, and it is -- in the run's own
        # control provenance below -- but it is not a mismatch.
        checks.pop('model_short')
    mismatches = {key: (
              manifest_entry.get(key), value) for key, value in checks.items() if manifest_entry.get(key) != value}



    if mismatches:
        raise RuntimeError(f'''Control donor manifest mismatch: {mismatches}''')

    manifest_verified = True

    _CONTROL_PREFLIGHT = {
        'status': 'verified' if manifest_verified else 'unattested',
        'sandbox': 'AlienCode',
        'episode': 'code_default',
        'model_short': MODEL_SHORT,
        'explore_loops': N_EXPLORE_LOOPS,
        'rounds_per_loop': N_EXPLORE_ROUNDS,
        'transcript_sha256': transcript_digest,
        # Whose run supplied the bank. Equal to model_short for passive and
        # deliberately not for random, which is the one thing a reader of a
        # random arm needs to be able to check.
        'donor_model_short': sorted(donor_models),
        'manifest': (_CONTROL_MANIFEST_PATH
                or None),
        'manifest_generated_at': manifest_entry.get('generated_at') if manifest_entry else None,
        'evaluator_stamp': evaluator_stamp,
        'rule_sha256': rule_digest }


    print(f'''[Control Preflight] {_CONTROL_PREFLIGHT}''')
    return _CONTROL_PREFLIGHT


def _passive_response(label: str) -> str:
    rec = _load_control_bank().get(label)
    if not rec or not str(rec.get('response', '')).strip():
        raise LookupError(f'''Passive donor transcript has no non-empty response for {label!r}: {_CONTROL_BANK_PATH}''')



    return str(rec['response'])


def _donor_probes(label: str) -> list[str]:
    """The donor round's probes, one per tool call it made.

    A native-tool donor records its probes as tool-call arguments, so the bank
    carries them as a list. Donors captured under the archived fenced-block
    protocol only have prose with a code block in it, and are read the old way
    so an older bank still loads.
    """

    rec = _load_control_bank().get(label)
    if not rec:
        raise LookupError(
            f'Passive donor transcript has no entry for {label!r}: '
            f'{_CONTROL_BANK_PATH}')
    codes = [
        str(code) for code in (rec.get('codes') or [])
        if str(code).strip()
    ]
    if codes:
        return codes
    fallback = extract_code(_passive_response(label))
    return [fallback] if fallback.strip() else []


def _donor_show_count(label: str) -> int:
    return sum(_donor_show_counts(label))


def _donor_show_counts(label: str) -> list[int]:
    return [
        _truncate_code_by_show(code, MAX_EXPLORE_TESTS)[2]
        for code in _donor_probes(label)
    ]


#: What the random arm's seed is keyed on. The run id by default, which
#: gives every run its own sequence. Set it to a fixed token and a whole
#: cohort draws the *same* sequence instead, which is what makes the arm a
#: shared baseline: with one sequence per run, two models' random arms
#: differ by their luck as well as by the model, and the difference cannot
#: be attributed. The probes stay model-independent either way; this
#: decides whether they are also run-independent.
_RANDOM_SEED_TAG = os.environ.get('EVAL_RANDOM_SEED_TAG', '').strip()


def _random_probe_code(
        loop_idx: int,
        round_idx: int,
        target_shows: int = MAX_EXPLORE_TESTS,
        call_idx: int = 1) -> str:
    '''Deterministic, grammar-valid, bounded random AlienCode probes.'''

    tag = _RANDOM_SEED_TAG or RUN_ID
    seed_text = f'''AlienCode:{tag}:{loop_idx}:{round_idx}:{call_idx}'''
    seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    ints = [rng.randint(0, 6) for _ in range(12)]
    a, b, c, d, e, f, g, h, i, j, k, m = ints
    pool = [
        f'''EMIT({a}, {b}, {c})''',
        f'''EMIT(SHATTER({a}, {b}), PARE({e}, {f}))''',
        'EMIT(WEAVE(YES, NO), WEAVE(NO, YES), WEAVE(1.0, 1.0))',
        f'''EMIT(FRACTURE({a + 1}, {b + 1}), RESIDUE({c + 2}, {d + 1}))''',
        f'''EMIT(COIL({a % 5 + 1}, {b % 4 + 1}), HALVE({c + 2}, {d % 4 + 1}))''',
        f'''EMIT(AKIN({a}, {b}), APART({c}, {d}), OVER({e}, {f}))''',
        f'''EMIT(UNDER({a}, {b}), ATOP({c}, {d}), BENEATH({e}, {f}))''',
        f'''EMIT(STRAND({a}, {b}, {c}), KNOT({d}, {e}, {f}))''',
        f'''EMIT(GAUGE(STRAND({a}, {b}, {c})), GAUGE(KNOT({d}, {e}, {f})))''',
        'EMIT(NEGATE(YES), BOND(YES, NO), BOND(YES, YES))',
        f'''EMIT(NADIR({a}, {b}, {c}), APEX({d}, {e}, {f}))''',
        'EMIT(SOME_OF(STRAND(YES, NO, YES)), EVERY_OF(STRAND(YES, NO, YES)))',
        f'''EMIT(ORDER(STRAND({a}, {b}, {c})))''',
        f'''EMIT(GATHER(EXTENT({a % 3}, {a % 3 + 4})))''',
        f'''EMIT({a + 0.25}, "probe-{g}{h}", {i})''',
        f'''EMIT("probe-{g}{h}", {i}, {j}, {k}, {m})''']


    return '\n'.join(rng.sample(pool, k=max(1, min(target_shows, len(pool)))))


def _random_probe_codes(
        loop_idx: int,
        round_idx: int,
        target_shows: list[int]) -> list[str]:
    """One model-independent probe per donor call, matched in probe volume."""

    return [
        _random_probe_code(loop_idx, round_idx, count, call_idx)
        for call_idx, count in enumerate(target_shows, 1)
        if count > 0
    ]


def _usage_delta(before: dict, after: dict) -> dict:
    return {key: max(0, int(after.get(key, 0) or 0) - int(before.get(key, 0) or 0)) for key in ('prompt_tokens', 'completion_tokens', 'reasoning_tokens', 'total_tokens')}









def _configured_explore_max_tokens() -> int:
    '''Harness-side cap (the proxy may enforce a larger family minimum).'''
    model_lower = MODEL.lower()
    if any(k in model_lower for k in ('claude', 'anthropic')):
        if any(v in model_lower for v in (
                'opus-4-7', 'opus-4.7',
                'opus-4-8', 'opus-4.8',
                'opus-5', 'opus.5')):

            return BASE_MAX_TOKENS * 16
        return 10000 + BASE_MAX_TOKENS
    if any(k in model_lower for k in ('qwen', 'doubao', 'o1', 'o3', 'reasoner', 'thinking', 'mimo', 'hy3', 'opd', 'a20b', 'gpt-5', 'minimax', 'gemini', 'grok', 'deepseek-v4', 'kimi')):




        return BASE_MAX_TOKENS * 8
    return BASE_MAX_TOKENS


def _run_explore_phase(state, loop_idx, results):
    print(f'''\n{"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"}\n  Phase: 自由探索 第 {loop_idx} 轮 ({N_EXPLORE_ROUNDS} rounds)\n{"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"}''')
    for r in range(N_EXPLORE_ROUNDS):
        state['round'] += 1
        label = f'''Explore {loop_idx}-{r + 1}'''
        controlled = _CONTROL_MODE in {'think', 'random', 'passive'}



        control_source = None
        donor_response_hash = None
        locked_probes: list[str] = []
        if _CONTROL_MODE == 'passive':
            donor_response = _passive_response(label)
            donor_response_hash = hashlib.sha256(donor_response.encode()).hexdigest()

            locked_probes = _donor_probes(label)
            raw_code = '\n'.join(locked_probes)
            control_source = _CONTROL_BANK_PATH
        elif _CONTROL_MODE == 'random':
            locked_probes = _random_probe_codes(
                loop_idx, r + 1, _donor_show_counts(label))
            raw_code = '\n'.join(locked_probes)
            control_source = f'''sha256({RUN_ID}:{loop_idx}:{r + 1}); count_from={_CONTROL_BANK_PATH}'''
        else:



            raw_code = ''

        if controlled and _CONTROL_MODE != 'think':
            display_probes = []
            orig_shows = kept_shows = 0
            warns = []
            for idx, locked_probe in enumerate(locked_probes, 1):
                normalized, original, kept = _truncate_code_by_show(
                    locked_probe, MAX_EXPLORE_TESTS)
                normalized, probe_warns = _enforce_code_limits(normalized)
                if not normalized:
                    raise RuntimeError(
                        f'''{_CONTROL_MODE} produced an empty locked probe '''
                        f'''for {label} call {idx}''')
                display_probes.append(
                    f'''# Locked probe {idx}\n{normalized}''')
                orig_shows += original
                kept_shows += kept
                warns.extend(
                    f'''call {idx}: {warning}''' for warning in probe_warns)
            code = '\n\n'.join(display_probes)
            if not display_probes:
                raise RuntimeError(
                    f'''{_CONTROL_MODE} produced no executable code for {label}''')
        else:

            code, orig_shows, kept_shows, warns = '', 0, 0, []

        prompt = ''
        if state['pending_feedback']:
            prompt = state['pending_feedback'] + '\n\n---\n\n'
            state['pending_feedback'] = None
        if TOOL_MODE:
            test_limit_note = f'''\n\n重要约束：请调用 {ALIEN_TOOL_NAME} 工具来执行代码。每次调用最多 {MAX_EXPLORE_TESTS} 个观测值（EMIT 的每个参数算一个观测值，循环内每个按 {_LOOP_WEIGHT}× 计数）、最多 {MAX_CODE_LINES} 行代码，返回最多 {MAX_OUTPUT_LINES} 行输出，超出部分会被截断。本轮最多可调用 {MAX_TOOL_CALLS_PER_ROUND} 次：你可以先跑一个探针，看到结果后再决定下一个。'''
        else:
            test_limit_note = f'''\n\n重要约束：每轮代码最多 {MAX_EXPLORE_TESTS} 个观测值（EMIT 的每个参数算一个观测值，循环内每个按 {_LOOP_WEIGHT}× 计数），最多 {MAX_CODE_LINES} 行代码，输出最多返回 {MAX_OUTPUT_LINES} 行。超出限制的部分将被截断。请精心设计每条测试以最大化信息量。'''




        if _CONTROL_MODE == 'think':
            prompt += '本轮不执行任何新实验，也不会提供新环境反馈。请只根据已有上下文更新你对 alien rules 的假设，记录目前最可信和最不确定的规则。不要提出或输出任何测试代码。'




        elif controlled and TOOL_MODE:
            # The probe is fixed, but the turn still travels the native tool
            # path so that reasoning continuity, call ids and the result round
            # trip match the treatment exactly. Only the probe differs.
            prompt += f'''本轮的 probe 已由系统锁定，共 {len(locked_probes)} 个：\n```aliencode\n{code}\n```\n请照常调用 {ALIEN_TOOL_NAME} 工具 {len(locked_probes)} 次以取得它们的执行结果。无论你在参数里写什么，系统都只会执行上面逐项列出的锁定 probe，并按顺序返回结果。\n请先记录你当前的规则假设和最不确定的规则，并预测锁定 probe 的结果。'''
        elif controlled:
            prompt += f'''以下 probe 已由系统锁定，本轮只会执行它：\n```aliencode\n{code}\n```\n你不能修改、替换或提出其他 probe。请只记录当前规则假设、最不确定的规则，并预测该固定 probe 的结果。你输出的任何 Code 都不会被执行。'''






        elif r == 0 and TOOL_MODE:
            prompt += f'''现在进入自由探索阶段。\n请调用 {ALIEN_TOOL_NAME} 工具执行 AlienCode 代码，测试你还不确定的操作行为。\n参考手册中列出了所有可用操作，请自行选择需要验证的操作进行测试。\n你可以在本轮内多次调用工具：看到一个结果后再设计下一个探针。''' + test_limit_note
        elif r == 0:
            prompt += '现在进入自由探索阶段。\n请写 AlienCode 代码来测试你还不确定的操作行为。\n参考手册中列出了所有可用操作，请自行选择需要验证的操作进行测试。\n请把代码放在 ```aliencode``` 代码块中。' + test_limit_note
        elif TOOL_MODE:
            prompt += f'''请继续探索 AlienCode 环境。\n请调用 {ALIEN_TOOL_NAME} 工具验证你还不确定的操作。\n你可以在本轮内多次调用工具，根据上一个结果调整下一个探针。''' + test_limit_note
        else:





            prompt += '请继续探索 AlienCode 环境。\n请写 AlienCode 测试代码来验证你还不确定的操作。\n请把代码放在 ```aliencode``` 代码块中。' + test_limit_note





        print(f'''\n{"========================================================================"}\n  Explore {loop_idx}-{r + 1} (Round {state["round"]})\n{"========================================================================"}''')
        t0 = time.time()
        state['conversation'].append({'role': 'user', 'content': prompt})
        usage_before = _target_usage()
        executions = []
        model_calls = 1
        if TOOL_MODE and not controlled:
            # A block is a quarter of the exploration budget, so a round that
            # runs long is re-asked rather than written off.
            round_ = _retry_on_deadline(
                lambda: _run_tool_round(
                    state['session'], prompt,
                    label=label, max_shows=MAX_EXPLORE_TESTS),
                label=label)
            assistant_msg = round_.text
            executions = round_.executions
            model_calls = round_.model_calls
        elif TOOL_MODE and locked_probes:
            round_ = _run_tool_round(
                state['session'], prompt,
                label=f'{label} Control Update',
                max_shows=MAX_EXPLORE_TESTS,
                locked_codes=locked_probes)
            assistant_msg = round_.text
            executions = round_.executions
            model_calls = round_.model_calls
        elif controlled:
            assistant_msg = _chat(
                state['session'], prompt,
                label=f'''{label} Control Update''')
        else:

            assistant_msg = _chat(state['session'], prompt, label=label)
        last_call_meta = state['session'].last_call_meta
        transport_retries = int((last_call_meta.get('transport_retry_count', 0)
                or                                            0))
        if not assistant_msg.strip():
            assistant_msg = '（模型返回了空回复）'
        state['conversation'].append(_assistant_turn(assistant_msg))

        no_code_retries = 0
        if TOOL_MODE and not controlled:
            # A round with no tool call produced no evidence, so nudge the
            # same way the code-block protocol does for a missing code block.
            while not executions and no_code_retries < _MAX_NO_CODE_RETRIES:
                no_code_retries += 1
                print(
                    f'  [Explore] 模型未调用工具，重试 '
                    f'{no_code_retries}/{_MAX_NO_CODE_RETRIES}')
                retry_prompt = _EXPLORE_NO_TOOL_RETRY + test_limit_note
                state['conversation'].append({
                    'role': 'user',
                    'content': retry_prompt})
                retry = _run_tool_round(
                    state['session'], retry_prompt,
                    label=f'''{label} (retry {no_code_retries})''',
                    max_shows=MAX_EXPLORE_TESTS)
                executions = retry.executions
                model_calls += retry.model_calls
                retry_call_meta = state['session'].last_call_meta
                transport_retries += int((
                    retry_call_meta.get('transport_retry_count', 0) or 0))
                last_call_meta = retry_call_meta
                assistant_msg = retry.text.strip() or '（模型返回了空回复）'
                state['conversation'].append(_assistant_turn(assistant_msg))

            raw_code = '\n\n'.join(
                record['raw_code'] for record in executions)
            code = '\n\n'.join(
                f'''# 第 {idx} 次调用\n{record["code"]}'''
                for idx, record in enumerate(executions, 1))
            orig_shows = sum(
                record['show_count_original'] for record in executions)
            kept_shows = sum(
                record['show_count_executed'] for record in executions)
            warns = []
        elif TOOL_MODE and locked_probes:
            # The locked round already retried on its own and executed inside
            # the tool loop, so only the per-call accounting is rebuilt here.
            raw_code = '\n\n'.join(
                record['raw_code'] for record in executions)
            code = '\n\n'.join(
                f'''# 第 {idx} 次调用\n{record["code"]}'''
                for idx, record in enumerate(executions, 1))
            orig_shows = sum(
                record['show_count_original'] for record in executions)
            kept_shows = sum(
                record['show_count_executed'] for record in executions)
            warns = []
            if not executions:
                raise RuntimeError(
                    f'{_CONTROL_MODE} delivered no locked probe for {label}')
        elif not controlled:
            raw_code = extract_code(assistant_msg)
            while not raw_code and no_code_retries < _MAX_NO_CODE_RETRIES:
                no_code_retries += 1
                print(f'''  [Explore] 未检测到代码，重试 {no_code_retries}/{_MAX_NO_CODE_RETRIES}''')

                retry_prompt = _EXPLORE_NO_CODE_RETRY + test_limit_note
                state['conversation'].append({
                    'role': 'user',
                    'content': retry_prompt})

                assistant_msg = _chat(
                    state['session'],
                    retry_prompt,
                    label=f'''{label} (retry {no_code_retries})''')

                retry_call_meta = state['session'].last_call_meta
                transport_retries += int((retry_call_meta.get('transport_retry_count', 0)
                        or                                             0))
                last_call_meta = retry_call_meta
                if not assistant_msg.strip():
                    assistant_msg = '（模型返回了空回复）'
                state['conversation'].append(_assistant_turn(assistant_msg))
                raw_code = extract_code(assistant_msg)
            code, orig_shows, kept_shows = _truncate_code_by_show(raw_code, MAX_EXPLORE_TESTS)

            code, warns = _enforce_code_limits(code)
        usage_after = _target_usage()
        call_usage = _usage_delta(usage_before, usage_after)

        if orig_shows > kept_shows:
            print(f'''  [EMIT Limit] 原始 {orig_shows} 条(加权) → 截断为 {kept_shows} 条''')

        for w in warns:
            print(f'''  [Code Limit] Explore: {w}''')

        if _CONTROL_MODE == 'think':
            exec_output = None
            truncated_output = None
        elif TOOL_MODE:
            # Already executed inside the tool loop; re-running would double
            # the side effects and could disagree with what the model saw.
            exec_output = '\n\n'.join(
                f'''# 第 {idx} 次调用\n{record["exec_output"]}'''
                for idx, record in enumerate(executions, 1)
            ) or '(未调用工具)'
            truncated_output = '\n\n'.join(
                f'''# 第 {idx} 次调用\n{_truncate_output(record["exec_output"])}'''
                for idx, record in enumerate(executions, 1)
            ) or '(未调用工具)'
        else:
            exec_output = alien_exec(code) if code else '(无代码)'
            truncated_output = _truncate_output(exec_output)
        elapsed = time.time() - t0
        probe_hash = hashlib.sha256(code.encode()).hexdigest() if code else None

        call_meta = {
                                'model_call_count': model_calls,
                            'input_tokens': call_usage['prompt_tokens'],
                             'output_tokens': call_usage['completion_tokens'],
                                                                  'reasoning_tokens': (call_usage['reasoning_tokens']
                or None),
                                     'configured_max_tokens': _configured_explore_max_tokens(),
                                    'effective_max_tokens': last_call_meta.get('effective_max_tokens'),
                           'retry_count': no_code_retries,
                                     'transport_retry_count': transport_retries,
                               'response_length': len(assistant_msg),
                               'reasoning_chars': len(_reasoning_text(state['session'])),
                                'cot_replay_turns': last_call_meta.get('cot_replay_turns')}

        results.append({
            'round': state['round'], 'phase': 'explore', 'loop_idx': loop_idx,
            'round_in_loop': r + 1, 'code': code, 'raw_code': raw_code,
            'show_count_original': orig_shows, 'show_count_executed': kept_shows,
            'exec_output': _cap_exec_output(exec_output) if exec_output is not None else None,
            'time_seconds': elapsed,
            'no_code_retries': no_code_retries,
            'control_mode': _CONTROL_MODE, 'control_source': control_source,
            'model_update': assistant_msg if controlled else None,
            'probe_hash': probe_hash,
            'donor_response_hash': donor_response_hash,
            'tool_mode': TOOL_MODE,
            'tool_calls': _tool_call_records(executions),
            # A control that delivered fewer probes than its donor saw less
            # evidence than the run it is matched against, so the shortfall
            # has to travel with the record rather than only into the log.
            'locked_probes_expected': (
                len(locked_probes) if locked_probes else None),
            'locked_probes_delivered': (
                len(executions) if locked_probes else None),
            'call_metrics': call_meta})




        print(f'''[Model Update]\n{assistant_msg[:800]}''' if controlled else f'''[Model]\n{assistant_msg[:800]}''')

        if _CONTROL_MODE == 'think':
            print('[Control think] no probe executed; no new feedback')
            state['pending_feedback'] = None
        else:
            print(f'''[AlienCode Exec]\n{truncated_output}''')
        if controlled and _CONTROL_MODE != 'think':
            if TOOL_MODE and locked_probes:
                # Delivered as tool results already; repeating it as a user
                # message would show the control the evidence twice.
                state['pending_feedback'] = None
            else:
                state['pending_feedback'] = f'''上一轮系统锁定 probe 的真实执行结果：\n{truncated_output}'''
            continue


        if TOOL_MODE:
            # Tool results are already in the conversation.
            state['pending_feedback'] = None
        elif not   controlled:
            state['pending_feedback'] = f'''上一轮代码的执行结果：\n{truncated_output}'''







def _build_test_prompt(task, prompt_variant=None):
    '''Build one held-out task prompt under a versioned protocol.'''
    variant = prompt_variant or _PROMPT_VARIANT
    if task['question_type'] == 'predict':
        return f'''【测试 {task["id"]}】\n基于你总结的规则，预测以下代码的输出。\n请明确写出"预测输出：<你的预测>"。\n\n{task["prompt"]}'''



    generalization_note = ''
    task_description = task['prompt']
    if task.get('test_cases'):
        if variant == 'explicit_generalization':
            generalization_note = '【评分协议说明】\n评测器会提取你实现的函数，并在多个未公开的合法输入上独立运行。\n下方调用仅用于展示函数接口和输入格式，不代表完整测试集。\n你必须实现对任意符合题意的合法输入均成立的通用算法。\n不得硬编码展示调用、示例答案、输入长度、固定元素位置，也不得编写仅覆盖展示样例的特殊分支。\n评测输入可能包含不同的值、长度以及合法边界情况。\n\n'








        elif variant == 'generalization':
            generalization_note = '【评分方式】\n本题要求实现一个函数。评测器会从你的回答中提取该函数定义，并使用多组未公开的合法输入分别调用它。\n同一个函数实现必须通过所有测试输入，才判定本题正确。\n\n下方调用仅用于说明函数接口和输入格式，不是唯一测试输入。\n请实现对所有合法输入均成立的通用解法，不要固定返回展示样例的结果，也不要只处理展示样例。\n\n任务：\n'









            task_description = re.sub('测试\\s*:\\s*', '\n\n示例调用：\n',
                task_description, count=1)

    return f'''【测试 {task["id"]}】\n基于你总结的规则，完成以下任务。请给出 AlienCode 代码。\n\n{generalization_note}{task_description}'''





def _with_pending_feedback(prompt: str, pending: str | None) -> str:
    if not pending:
        return prompt
    return pending + '\n\n---\n\n' + prompt


# How many extra sweeps a milestone spends chasing answers the provider never
# delivered, and the seconds between them. Throttling clears in minutes, so a
# few patient passes recover a run that would otherwise be scored as a wipeout.
# Tests set the backoff to zero rather than waiting out a real one.
_MAX_TEST_REPAIR_PASSES = int(os.environ.get('EVAL_TEST_REPAIR_PASSES', '3'))
_TEST_REPAIR_BACKOFF = float(os.environ.get('EVAL_TEST_REPAIR_BACKOFF', '30'))


def _unanswered_tasks(completed_map):
    '''Held-out tasks whose answer the provider never delivered.

    ``error_type`` is only set when the call itself failed, so it separates a
    question the API ate from one the model genuinely got wrong.
    '''

    return [
        task for task in TEST_TASKS
        if (completed_map.get(task['id']) or {}).get('error_type')
    ]


def _run_single_test(
        task,
        base_session: AgentSession,
        pending_feedback: str | None,
        milestone_idx):
    '''Run a single test task with its own forked conversation.'''
    session = _closed_book_fork(base_session)
    task_prompt = _with_pending_feedback(
        _build_test_prompt(task),
        pending_feedback,
    )
    t0 = time.time()
    label = f'''M{milestone_idx} Test {task["id"]}'''
    # The one caller that keeps the run's own deadline: spending it is this
    # question's wrong answer, and a longer wait only delays recording it.
    turn = _chat_turn(session, task_prompt, label=label, timeout=None)
    response = turn.text or ''
    retries = 0
    while retries < _MAX_NO_CODE_RETRIES and _needs_another_ask(
            turn, response, task):
        retries += 1
        silent = not response.strip() and not turn.tool_calls
        leaked = bool(
            turn.tool_calls or _LEAKED_TOOL_CALL.search(response)
        )
        if silent:
            kind, nudge = 'silent', _SILENT_ANSWER_RETRY
        elif leaked:
            kind, nudge = 'no-tool', _CLOSED_BOOK_RETRY
        else:
            kind, nudge = 'unsubmitted', _UNSUBMITTED_ANSWER_RETRY
        retry_label = f'{label} ({kind} retry)'
        if turn.tool_calls:
            turn = _chat_turn(
                session,
                '',
                label=retry_label,
                tool_results=[
                    ToolResult(
                        call_id=call.call_id,
                        output=_CLOSED_BOOK_TOOL_DENIAL,
                        name=call.name,
                        is_error=True,
                    )
                    for call in turn.tool_calls
                ],
            )
        elif leaked:
            # A marker printed as text, not a real tool call: nudging in place
            # would carry that turn into every later request, and a provider
            # handed back its own special token answers by dropping the
            # connection without a reply. One run spent three repair passes and
            # over an hour failing the same question that way, then ended
            # damaged over it. Ask again from the clean base instead.
            session = _closed_book_fork(base_session)
            turn = _chat_turn(
                session, task_prompt + '\n\n' + nudge, label=retry_label)
        else:
            turn = _chat_turn(session, nudge, label=retry_label)
        candidate = turn.text or ''
        # An extra ask may only add an answer, never remove one: a retry that
        # again submits nothing must not overwrite a reply that did.
        if not _unanswered(candidate, task) or _unanswered(response, task):
            response = candidate
    if not response.strip():
        response = '（模型返回了空回复）'
    task_time = time.time() - t0

    if task['question_type'] == 'predict':
        prediction = extract_prediction(response, tool_protocol=TOOL_MODE)
        actual_output = predict_answer(task)
        correct = values_equal(prediction, actual_output) if prediction else False
        exec_output = prediction
    else:
        code = extract_code(response)
        exec_output = alien_exec(code) if code else '(无代码)'

        if task.get('test_cases'):
            tc_passed, tc_detail = _run_test_cases(code, task)
            if task['expected'] is not None:
                orig_correct = values_equal(exec_output, task['expected'])
                correct = orig_correct and tc_passed
            else:
                orig_correct = ('[AlienError]' not in exec_output
                        and exec_output != '(无代码)')

                correct = orig_correct and tc_passed
            exec_output += f'''\n[TestCases {"PASS" if tc_passed else "FAIL"}]\n{tc_detail}'''
        elif task['expected'] is not None:
            correct = values_equal(exec_output, task['expected'])
        else:
            correct = '[AlienError]' not in exec_output and exec_output != '(无代码)'

    return {
                   'task_id': task['id'], 'difficulty': task['difficulty'],
                         'question_type': task['question_type'],
                       'exec_output': _cap_exec_output(exec_output),
                    'expected': task.get('expected'), 'correct': correct,
                        'rules_tested': task['rules_tested'], 'time_seconds': task_time}


_DIFF_CATEGORIES = (
    'apply', 'interact', 'scope', 'engineer',
    'algorithm_1', 'algorithm_2', 'algorithm_3',
)


def _build_code_milestone_snapshot(
        *,
        milestone_idx,
        round_idx,
        milestone_base_snapshot,
        summary,
        rule_scores,
        test_results,
        summary_time,
        milestone_time):
    found_count = sum(1 for v in rule_scores.values() if v)
    total = len(GROUND_TRUTH_SEXPR)
    n_correct = sum(1 for r in test_results if r['correct'])
    by_diff = {}
    for diff in _DIFF_CATEGORIES:
        rows = [r for r in test_results if r['difficulty'] == diff]
        correct = sum(1 for r in rows if r['correct'])
        by_diff[diff] = {
            'correct': correct,
            'total': len(rows),
            'accuracy': correct / len(rows) if rows else 0,
        }
    snapshot = {
        'milestone_idx': milestone_idx,
        'round': round_idx,
        'session_snapshot_path': (
            str(milestone_base_snapshot) if milestone_base_snapshot else None),
        'summary': summary,
        'rule_scores': rule_scores,
        'found': found_count,
        'total': total,
        'test_accuracy': n_correct / len(TEST_TASKS) if TEST_TASKS else 0,
        'test_correct': n_correct,
        'test_total': len(TEST_TASKS),
        **{f'{d}_accuracy': by_diff[d]['accuracy'] for d in by_diff},
        **{f'{d}_correct': by_diff[d]['correct'] for d in by_diff},
        **{f'{d}_total': by_diff[d]['total'] for d in by_diff},
        'test_results': test_results,
        'summary_time': summary_time,
        'milestone_time': milestone_time,
    }
    record = {
        'milestone_idx': milestone_idx,
        'found': found_count,
        'total': total,
        'rule_scores': rule_scores,
        'test_correct': n_correct,
        'test_total': len(TEST_TASKS),
        **{f'{d}_accuracy': by_diff[d]['accuracy'] for d in by_diff},
        'milestone_time': milestone_time,
    }
    return snapshot, record, by_diff


def _print_code_milestone_summary(milestone_idx, snapshot, by_diff):
    print(
        f'''\n[Milestone {milestone_idx}] 规则: {snapshot["found"]}/{snapshot["total"]}  '''
        f'''测试: {snapshot["test_correct"]}/{snapshot["test_total"]}  '''
        f'''用时: {snapshot["milestone_time"]:.1f}s''')
    for diff in _DIFF_CATEGORIES:
        bd = by_diff[diff]
        if bd['total'] > 0:
            print(
                f'''  {diff:12s}: {bd["correct"]}/{bd["total"]} '''
                f'''({bd["accuracy"]:.0%})''')


def _defer_workers() -> int:
    '''How many graded questions this run may have in flight at once.

    The per-model number is the one the provider was sized for, so it is what
    the run uses. A blanket value across a mixed cohort is wrong in both
    directions at once: it leaves the models that can take 72 sitting at a
    third of their capacity, and it drives the ones budgeted for 12 to more
    than twice theirs, where the gateway starts timing requests out.

    ALIENCODE_DEFER_WORKERS_CAP stays available for holding a whole cohort
    under a shared ceiling, but it can only lower a model, never raise it.
    '''
    workers = MAX_PARALLEL_TESTS
    value = os.environ.get('ALIENCODE_DEFER_WORKERS')
    if value:
        try:
            workers = max(1, int(value))
        except ValueError:
            pass
    cap = os.environ.get('ALIENCODE_DEFER_WORKERS_CAP')
    if cap:
        try:
            workers = min(workers, max(1, int(cap)))
        except ValueError:
            pass
    return workers


def _heldout_ledger_path(kind: str = 'heldout') -> str:
    os.makedirs(_CHECKPOINT_DIR, exist_ok=True)
    short = MODEL_SHORT or MODEL.rsplit('_', 1)[-1]
    suffix = f'_{RUN_ID}' if RUN_ID else ''
    return os.path.join(
        _CHECKPOINT_DIR, f'{kind}_alien_code_{short}{suffix}.jsonl')


def _ledger_job_names(kind: str = 'heldout') -> set[str]:
    path = _heldout_ledger_path(kind)
    if not os.path.exists(path):
        return set()
    names = set()
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row.get('result'), dict) and row.get('job'):
                names.add(str(row['job']))
    return names


def _load_code_ledger(names: dict, *, kind: str = 'heldout') -> dict:
    path = _heldout_ledger_path(kind)
    if not os.path.exists(path):
        return {}
    by_name = {name: key for key, name in names.items()}
    restored = {}
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            key = by_name.get(row.get('job'))
            if key is not None and isinstance(row.get('result'), dict):
                restored[key] = row['result']
    if restored:
        print(
            f'\n[heldout-resume] 从账本恢复 {len(restored)} 道已答题目 '
            f'({os.path.basename(path)})')
    return restored


_STREAM_POOL: ThreadPoolExecutor | None = None
_STREAM_FUTURES: list = []
_LEDGER_LOCK = threading.Lock()


def _append_ledger(kind: str, name: str, result: dict) -> None:
    with _LEDGER_LOCK:
        with open(_heldout_ledger_path(kind), 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(
                {'job': name, 'result': result}, ensure_ascii=False) + '\n')


def _stream_milestone_scoring(row: dict) -> None:
    '''Start scoring this milestone while exploration carries on.

    Holding every milestone's questions until the end means M0 is answered from
    a snapshot taken hours earlier. Providers that keep reasoning state on their
    side -- a stored response id, a signed thinking block -- are the ones that
    can refuse to continue from a base that old, so their milestones are scored
    as soon as the base exists. Results land in the same ledger the end-of-run
    flush reads, which is what keeps this a scheduling change and not a scoring
    one: each question is still answered from its own milestone's frozen base.
    '''
    global _STREAM_POOL
    if _STREAM_POOL is None:
        _STREAM_POOL = ThreadPoolExecutor(
            max_workers=_defer_workers(), thread_name_prefix='heldout')
    already = _ledger_job_names('heldout')
    milestone = row['milestone_idx']
    queued = 0
    for task in TEST_TASKS:
        name = f'''M{milestone}|{task["id"]}'''
        if name in already:
            continue
        job = {
            'task': task,
            'base_session': row['base_session'],
            'pending_feedback': row.get('pending_feedback'),
            'milestone_idx': milestone,
        }

        def run(job=job, name=name):
            _append_ledger('heldout', name, _safe_single_test(job))

        _STREAM_FUTURES.append(_STREAM_POOL.submit(run))
        queued += 1
    print(
        f'[heldout-stream] M{milestone}: {queued} questions scoring in the '
        f'background while exploration continues')


def _drain_streamed_scoring() -> None:
    if not _STREAM_FUTURES:
        return
    print(
        f'\n[heldout-stream] waiting for {len(_STREAM_FUTURES)} background '
        'questions to land')
    for future in _STREAM_FUTURES:
        future.result()
    _STREAM_FUTURES.clear()
    if _STREAM_POOL is not None:
        _STREAM_POOL.shutdown(wait=True)


def _request_deadline() -> float:
    # The same fallback the client's own timeout takes, so a run launched
    # without the variable judges a timeout against the deadline it was
    # actually given rather than a stale 600s.
    try:
        return float(os.environ.get(
            'EVAL_HTTP_TIMEOUT', _LOCAL_SETTINGS.get('timeout', 600)))
    except ValueError:
        return 600.0


def _trajectory_deadline() -> float | None:
    '''The budget for turns that build the trajectory rather than answer it.

    A graded question that runs out of time is an answer, the wrong one, and
    waiting longer only delays scoring it. A milestone summary that runs out
    of time is not an answer at all: the milestone loses the rule report it
    exists to collect, and no amount of re-asking the held-out set puts it
    back. One model spent thirty-three minutes on its first summary, failed,
    and started the same wait again -- an hour gone before the run had reached
    its first question. So these turns get their own deadline, capped by the
    route either way; the graded budget is untouched.
    '''
    raw = os.environ.get('EVAL_TRAJECTORY_TIMEOUT', '').strip()
    if not raw:
        return None
    try:
        return max(_request_deadline(), float(raw))
    except ValueError:
        return None


#: Below this fraction of the deadline, a timeout did not come from the model
#: still thinking -- nothing can hold a socket open that long and then claim it
#: ran out of time. Well under the deadline means the gateway gave up on its
#: own, which is the server's problem and not an answer.
_DEADLINE_CREDIT = 0.9


def _answer_deadline_expired(exc: Exception, elapsed: float) -> bool:
    '''True when the request actually spent the thinking budget it was given.

    Every system gets the same wall clock per question, and spending all of it
    without producing an answer is an answer -- the wrong one -- in the same
    way a wrong number is. Treating that as an infrastructure fault is what let
    one question burn hours of retries and then mark a finished run damaged.

    But a 504 alone does not mean the model was thinking. The gateway returns
    one just as readily when it is overloaded and drops the request in seconds,
    and scoring that as a wrong answer would quietly charge a model for the
    provider having a bad afternoon. So the clock decides: only a failure that
    consumed the deadline counts against the model. Anything faster stays a
    fault, and stays retryable.

    A connection the server drops after holding it past the deadline is the
    same outcome under another name: one system's six hardest questions each
    ran fifty-odd minutes and ended in "Server disconnected" on every one of
    six attempts, and filing that as a fault kept a finished run open all
    evening. An explicit refusal such as a 429 is still the provider talking,
    however late it arrives.
    '''
    looks_like_timeout = (
        (isinstance(exc, AgentHTTPError)
         and getattr(exc, 'status_code', None) == 504)
        or isinstance(exc, (TimeoutError, socket.timeout,
                            ConnectionResetError))
        or type(exc).__name__ in {
            'ReadTimeout', 'ConnectTimeout', 'PoolTimeout', 'TimeoutException',
            'RemoteProtocolError', 'RemoteDisconnected', 'ReadError'}
    )
    if not looks_like_timeout:
        return False
    return elapsed >= _DEADLINE_CREDIT * _request_deadline()


def _safe_single_test(job):
    task = job['task']
    started = time.time()
    try:
        return _run_single_test(
            task,
            job['base_session'],
            job.get('pending_feedback'),
            job['milestone_idx'],
        )
    except Exception as exc:
        elapsed = time.time() - started
        timed_out = _answer_deadline_expired(exc, elapsed)
        label = 'TIMEOUT' if timed_out else 'ERR'
        print(f'''  {label} M{job["milestone_idx"]} {task["id"]} '''
              f'''({elapsed:.0f}s): {exc}''')
        record = {
            'task_id': task['id'],
            'difficulty': task['difficulty'],
            'question_type': task['question_type'],
            'exec_output': f'''[TestError] {type(exc).__name__}: {exc}''',
            'expected': task.get('expected'),
            'correct': False,
            'rules_tested': task['rules_tested'],
            'time_seconds': round(elapsed, 1),
            'failure_seconds': round(elapsed, 1),
        }
        if timed_out:
            # Scored, not re-queued: no error_type, so the repair passes leave
            # it alone and the run can finish. Flagged, with the clock that
            # justified the call, so an audit can separate a model that ran out
            # of time from a wrong answer it actually wrote.
            record['timed_out'] = True
            # And with the budget it was refused under. Re-asking these is
            # worth doing once the allowance goes up -- the cohort lost 229
            # questions to a 600s deadline when the routes allow 1200 -- but
            # without this an automatic pass cannot tell a verdict it should
            # revisit from one it just produced, and would ask forever.
            record['budget_seconds'] = int(_request_deadline())
        else:
            record['error_type'] = type(exc).__name__
        return record


def _dispatch_code_jobs(jobs: dict, names: dict, *, kind: str,
        n_workers: int, banner: str) -> dict:
    results = _load_code_ledger(names, kind=kind)
    ledger = open(_heldout_ledger_path(kind), 'a', encoding='utf-8')

    def record(key, result):
        results[key] = result
        ledger.write(json.dumps(
            {'job': names[key], 'result': result}, ensure_ascii=False) + '\n')
        ledger.flush()

    # A question that spent its whole budget is an answer, so the ledger keeps
    # it and it is not asked again -- that is the protocol. The repair tool
    # sets this when the timeouts came from load rather than from the question
    # itself, which is the one case where re-asking is measuring the same
    # thing rather than granting a second attempt.
    reask_timeouts = os.environ.get(
        'ALIENCODE_REASK_TIMEOUTS', '0').strip().lower() in ('1', 'true', 'yes')

    def unanswered(key) -> bool:
        if key not in results:
            return True
        row = results.get(key) or {}
        return bool(row.get('error_type')
                    or (reask_timeouts and row.get('timed_out')))

    todo = [key for key in jobs if unanswered(key)]
    for attempt in range(_MAX_TEST_REPAIR_PASSES + 1):
        if not todo:
            break
        workers = max(1, min(n_workers, len(todo)) >> attempt)
        if attempt:
            print(
                f'\n[heldout-repair] {banner}: {len(todo)} unanswered, '
                f'pass {attempt}/{_MAX_TEST_REPAIR_PASSES} '
                f'(workers={workers})')
            time.sleep(_TEST_REPAIR_BACKOFF * attempt)
        if workers <= 1:
            for key in todo:
                record(key, _safe_single_test(jobs[key]))
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_safe_single_test, jobs[key]): key
                    for key in todo
                }
                for future in as_completed(futures):
                    record(futures[future], future.result())
        todo = [
            key for key in jobs
            if (results.get(key) or {}).get('error_type')
        ]
    if todo:
        print(
            f'\n[heldout-damaged] {banner}: {len(todo)} tasks never answered; '
            'these scores are not model performance')
    ledger.close()
    return results


def _serializable_deferred(deferred):
    rows = []
    for row in deferred or []:
        item = {k: v for k, v in row.items() if k != 'base_session'}
        base = row.get('base_session')
        if base is not None and 'base_session_state' not in item:
            item['base_session_state'] = base.state().to_dict()
        rows.append(item)
    return rows


def _hydrate_deferred(rows):
    hydrated = []
    for row in rows or []:
        item = dict(row)
        state = item.get('base_session_state')
        if isinstance(state, dict) and 'base_session' not in item:
            item['base_session'] = AgentSession.from_state(
                _require_agent_client(), SessionState.from_dict(state))
        hydrated.append(item)
    return hydrated


def _flush_deferred_milestones(deferred, snapshots, milestone_records,
        *, ckpt_data_fn=None):
    if not deferred:
        return
    n_workers = _defer_workers()
    jobs = {}
    names = {}
    for mi, row in enumerate(deferred):
        for ti, task in enumerate(TEST_TASKS):
            key = (mi, ti)
            jobs[key] = {
                'task': task,
                'base_session': row['base_session'],
                'pending_feedback': row.get('pending_feedback'),
                'milestone_idx': row['milestone_idx'],
            }
            names[key] = f'''M{row["milestone_idx"]}|{task["id"]}'''
    labels = ', '.join(f'''M{row["milestone_idx"]}''' for row in deferred)
    print(
        f'\n[heldout-deferred] flushing {len(deferred)} milestones '
        f'[{labels}] × {len(TEST_TASKS)} tasks = {len(jobs)} jobs '
        f'across {n_workers} workers')
    flat = _dispatch_code_jobs(
        jobs, names, kind='heldout', n_workers=n_workers, banner='deferred')

    existing = {snap.get('milestone_idx') for snap in snapshots}
    for mi, row in enumerate(deferred):
        if row['milestone_idx'] in existing:
            print(
                f'''[heldout-deferred] M{row["milestone_idx"]} already '''
                'materialized; skipping duplicate append')
            continue
        results = [flat[(mi, ti)] for ti in range(len(TEST_TASKS))]
        milestone_time = time.time() - row['milestone_start_time']
        snapshot, record, by_diff = _build_code_milestone_snapshot(
            milestone_idx=row['milestone_idx'],
            round_idx=row['round'],
            milestone_base_snapshot=row.get('base_snapshot'),
            summary=row['summary'],
            rule_scores=row['rule_scores'],
            test_results=results,
            summary_time=row['summary_time'],
            milestone_time=milestone_time,
        )
        snapshots.append(snapshot)
        milestone_records.append(record)
        _print_code_milestone_summary(row['milestone_idx'], snapshot, by_diff)
        if ckpt_data_fn is not None:
            _save_checkpoint(ckpt_data_fn())


def _run_v2_oracle_diagnostics(deferred) -> list[dict]:
    if not (PROTOCOL_V2 and V2_ORACLE_DIAGNOSTICS and deferred):
        return []
    by_milestone = {row['milestone_idx']: row for row in deferred}
    selected = [('O@M0', by_milestone.get(0)), ('A4+O', by_milestone.get(4))]
    selected = [(label, row) for label, row in selected if row is not None]
    if not selected:
        return []

    from oracle_rules import build_oracle_message

    oracle_msg = build_oracle_message()

    # The two disclosures are independent -- each forks its own closed-book
    # base from a different milestone -- so they are acknowledged together and
    # then graded out of a single pool. Run label by label instead and the run
    # waits out two separate drains, each ending on whichever one question is
    # slowest, with the workers idle behind it.
    acks: dict[str, tuple] = {}
    with ThreadPoolExecutor(max_workers=len(selected)) as executor:
        futures = {}
        for label, row in selected:
            print(f'\n[Oracle diagnostic {label}] injecting rule disclosure')

            def acknowledge(label=label, row=row):
                base = _closed_book_fork(row['base_session'])
                return base, _chat(
                    base, oracle_msg, label=f'{label} Oracle Ack')

            futures[executor.submit(acknowledge)] = label
        for future in as_completed(futures):
            label = futures[future]
            acks[label] = future.result()
    for label, (_, ack) in acks.items():
        print(f'''[Oracle diagnostic {label} ack]\n{(ack or '')[:800]}''')

    jobs = {}
    names = {}
    for label, _ in selected:
        base, _ack = acks[label]
        for ti, task in enumerate(TEST_TASKS):
            key = (label, ti)
            jobs[key] = {
                'task': task,
                'base_session': base,
                'pending_feedback': None,
                'milestone_idx': label,
            }
            names[key] = f'''{label}|{task["id"]}'''
    labels = ', '.join(label for label, _ in selected)
    flat = _dispatch_code_jobs(
        jobs, names, kind='oracle', n_workers=_defer_workers(),
        banner=f'oracle [{labels}]')

    diagnostics = []
    for label, row in selected:
        results = [flat[(label, ti)] for ti in range(len(TEST_TASKS))]
        correct = sum(1 for item in results if item['correct'])
        diagnostics.append({
            'label': label,
            'source_milestone': row['milestone_idx'],
            'oracle_ack': acks[label][1],
            'test_correct': correct,
            'test_total': len(TEST_TASKS),
            'test_accuracy': correct / len(TEST_TASKS) if TEST_TASKS else 0,
            'test_results': results,
        })
        print(
            f'''[Oracle diagnostic {label}] {correct}/{len(TEST_TASKS)} '''
            f'''({correct / len(TEST_TASKS):.0%})''')
    return diagnostics



def _run_milestone(state, milestone_idx, snapshots, milestone_records, *,
        ckpt_data_fn=None, resume_data=None, deferred=None):

    print(f'''\n{"########################################################################"}\n  Milestone {milestone_idx}\n{"########################################################################"}''')
    milestone_start = time.time()

    if resume_data and resume_data.get('summary_done'):
        summary = resume_data['summary']
        summary_time = resume_data['summary_time']
        rule_scores = resume_data['rule_scores']
        test_results = resume_data['test_results']
        completed_ids = {r['task_id'] for r in test_results}
        found_count = sum(1 for v in rule_scores.values() if v)
        total = len(GROUND_TRUTH_SEXPR)
        print(f'''\n  [Resume] 摘要和规则评分已完成 ({found_count}/{total})''')
        print(f'''  [Resume] 已完成 {len(completed_ids)}/{len(TEST_TASKS)} 道测试''')
    else:
        MAX_RETRY = 2

        if resume_data and resume_data.get('summary_retrying'):
            summary = resume_data['summary']
            summary_time = resume_data['summary_time']
            retry_count = resume_data['retry_count']
            extracted = _extract_sexpr_rules(summary)
            summary_state = resume_data.get('summary_session_state')
            if isinstance(summary_state, dict):
                summary_session = AgentSession.from_state(
                    _require_agent_client(),
                    SessionState.from_dict(summary_state),
                )
                resume_summary_context = ''
            else:
                summary_session = _closed_book_fork(state['session'])
                resume_summary_context = (
                    '迁移前的总结分支没有原生会话状态。以下是上一条总结，'
                    '请据此按重试要求重写：\n\n' + summary + '\n\n---\n\n')
            print(f'''\n  [Resume] 规则总结重试阶段 (已重试 {retry_count}/{MAX_RETRY}，已提取 {len(extracted)} 条)''')
        else:

            state['round'] += 1
            summary_prompt = _with_pending_feedback(
                SEXPR_MILESTONE_PROMPT,
                state.get('pending_feedback'),
            )
            summary_session = _closed_book_fork(state['session'])
            resume_summary_context = ''
            t0 = time.time()
            summary = _chat_within_budget(
                summary_session,
                summary_prompt,
                label=f'''Milestone {milestone_idx} Summary''')

            if not summary.strip():
                summary = '（模型返回了空回复）'
            summary_time = time.time() - t0
            print(f'''\n[Milestone {milestone_idx} Summary]\n{summary[:2000]}''')

            extracted = _extract_sexpr_rules(summary)
            retry_count = 0

            if ckpt_data_fn is not None and len(extracted) < 3:
                _save_checkpoint(ckpt_data_fn(ms_state={
                                     'milestone_idx': milestone_idx,
                                        'summary_retrying': True,
                               'summary': summary, 'summary_time': summary_time,
                                   'retry_count': retry_count,
                         'summary_session_state':
                             summary_session.state().to_dict()}))


        while len(extracted) < 3 and retry_count < MAX_RETRY:
            retry_count += 1
            print(f'''\n[Milestone {milestone_idx}] S-expression 规则不足 ({len(extracted)} 条)，第 {retry_count} 次重试...''')

            retry_prompt = world_data.SEXPR_RETRY_PROMPT









            state['round'] += 1
            t_retry = time.time()
            summary = _chat_within_budget(
                summary_session,
                resume_summary_context + retry_prompt,
                label=f'''Milestone {milestone_idx} Summary (retry {retry_count})''')
            resume_summary_context = ''

            if not summary.strip():
                summary = '（模型返回了空回复）'
            summary_time += time.time() - t_retry
            print(f'''\n[Milestone {milestone_idx} Retry {retry_count}]\n{summary[:2000]}''')
            extracted = _extract_sexpr_rules(summary)

            if ckpt_data_fn is not None:
                _save_checkpoint(ckpt_data_fn(ms_state={
                                     'milestone_idx': milestone_idx,
                                        'summary_retrying': True,
                               'summary': summary, 'summary_time': summary_time,
                                   'retry_count': retry_count,
                         'summary_session_state':
                             summary_session.state().to_dict()}))


        rule_scores = _score_rule_precision(summary)
        found_count = sum(1 for v in rule_scores.values() if v)
        total = len(GROUND_TRUTH_SEXPR)
        print(f'''\n[Rule Discovery]: {found_count}/{total}''')

        test_results = []
        completed_ids = set()

        if ckpt_data_fn is not None:
            _save_checkpoint(ckpt_data_fn(ms_state={
                                 'milestone_idx': milestone_idx, 'summary_done': True,
                           'summary': summary, 'summary_time': summary_time,
                               'rule_scores': rule_scores,
                           'test_results': []}))



    base_session = _closed_book_fork(state['session'])
    # Keep the session these questions are answered from. Exploration continues
    # after this milestone, so the end-of-run snapshot holds knowledge the model
    # did not have here -- redoing a question the API ate needs *this* state to
    # stay an honest measurement. It has to be the fork rather than the live
    # session: snapshots are filed under the session id, so snapshotting the
    # live one every milestone would just overwrite one file with the latest
    # state. The fork is created here and never advances, so its file keeps
    # exactly what this milestone knew.
    milestone_base_snapshot = None
    try:
        milestone_base_snapshot = base_session.snapshot()
    except Exception as exc:
        print(f'  [warn] milestone base snapshot failed: {exc}')
    pending_feedback = state.get('pending_feedback')
    if deferred is not None:
        deferred.append({
            'milestone_idx': milestone_idx,
            'round': state['round'],
            'summary': summary,
            'summary_time': summary_time,
            'rule_scores': rule_scores,
            'found_count': found_count,
            'total': total,
            'pending_feedback': pending_feedback,
            'base_snapshot': (
                str(milestone_base_snapshot)
                if milestone_base_snapshot else None),
            'base_session': base_session,
            'base_session_state': base_session.state().to_dict(),
            'milestone_start_time': milestone_start,
        })
        if STREAM_HELDOUT:
            _stream_milestone_scoring(deferred[-1])
        else:
            print(
                f'[Milestone {milestone_idx}] held-out tests deferred to '
                'end-of-run merged pool')
        return

    pending_tasks = [t for t in TEST_TASKS if t['id'] not in completed_ids]
    completed_map = {r['task_id']: r for r in test_results}
    ckpt_lock = threading.Lock()

    print(f'''\n{"────────────────────────────────────────────────────────────────────────"}\n  Milestone {milestone_idx}: 测试集 A01-A80 (并行 workers={MAX_PARALLEL_TESTS}, 待测={len(pending_tasks)})\n{"────────────────────────────────────────────────────────────────────────"}''')


    print_lock = threading.Lock()

    def _test_worker(task):
        result = _run_single_test(
            task,
            base_session,
            pending_feedback,
            milestone_idx)
        status = 'OK' if result['correct'] else 'XX'
        exp_display = task['expected'] or '(engine-verify)'
        raw_output = str(result['exec_output']).split('\n')[0]
        tc_lines = ''
        if '\n[TestCases ' in str(result['exec_output']):
            tc_part = str(result['exec_output']).split('\n[TestCases ')[1]
            tc_status = tc_part.split(']')[0]
            tc_detail_lines = tc_part.split(']\n', 1)[1] if ']\n' in tc_part else ''
            tc_lines = f'''\n[TestCases {tc_status}]\n{tc_detail_lines}''' if tc_detail_lines else f'''\n[TestCases {tc_status}]'''
        with print_lock:
            print(f'''  {status} {task["id"]} [{task["difficulty"]:12s}] got={raw_output[:40]:40s} exp={exp_display}{tc_lines}''')




        with ckpt_lock:
            completed_map[task['id']] = result
            if ckpt_data_fn is not None:
                ordered = [completed_map[t['id']] for t in TEST_TASKS if t['id'] in completed_map]

                _save_checkpoint(ckpt_data_fn(ms_state={
                                     'milestone_idx': milestone_idx, 'summary_done': True,
                               'summary': summary, 'summary_time': summary_time,
                                   'rule_scores': rule_scores,
                                 'test_results': ordered}))

        return result

    # A worker only raises when the provider never answered, which is not the
    # same thing as the model answering wrongly. Counting those as misses is
    # how a throttled hour became a 0/90 that read like a real score, so the
    # questions the API ate get asked again from this same milestone state.
    # Later passes use fewer workers, since load is usually what caused it.
    sweep_tasks = pending_tasks or _unanswered_tasks(completed_map)
    for attempt in range(_MAX_TEST_REPAIR_PASSES + 1):
        if not sweep_tasks:
            break
        workers = max(1, MAX_PARALLEL_TESTS >> attempt)
        if attempt or not pending_tasks:
            print(
                f'\n  [Repair] Milestone {milestone_idx}: '
                f'{len(sweep_tasks)} 题未获应答，第 {attempt + 1}/'
                f'{_MAX_TEST_REPAIR_PASSES + 1} 轮补答 (workers={workers})')
            time.sleep(_TEST_REPAIR_BACKOFF * max(1, attempt))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_test_worker, task): task
                for task in sweep_tasks
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    future.result()
                    continue
                except Exception as e:
                    print(f'''  ERR {task["id"]}: {e}''')

                    failure = {
                        'task_id': task['id'],
                        'difficulty': task['difficulty'],
                        'question_type': task['question_type'],
                        'exec_output': f'''[TestError] {type(e).__name__}: {e}''',
                        'expected': task.get('expected'),
                        'correct': False,
                        'rules_tested': task['rules_tested'],
                        'time_seconds': 0.0,
                        'error_type': type(e).__name__ }

                    with ckpt_lock:
                        completed_map[task['id']] = failure
                        if ckpt_data_fn is not None:
                            ordered = [completed_map[t['id']] for t in TEST_TASKS if t['id'] in completed_map]

                            _save_checkpoint(ckpt_data_fn(ms_state={
                                                 'milestone_idx': milestone_idx,
                                                'summary_done': True,
                                           'summary': summary,
                                                'summary_time': summary_time,
                                               'rule_scores': rule_scores,
                                                'test_results': ordered}))
                    continue
        sweep_tasks = _unanswered_tasks(completed_map)

    test_results = [completed_map[t['id']] for t in TEST_TASKS if t['id'] in completed_map]
    unanswered = [r for r in test_results if r.get('error_type')]
    if unanswered:
        print(
            f'\n  [Damaged] Milestone {milestone_idx}: {len(unanswered)} '
            f'题始终未获应答，本里程碑的分数不能当作模型表现')

    milestone_time = time.time() - milestone_start
    n_correct = sum(1 for r in test_results if r['correct'])
    diff_categories = ('apply', 'interact', 'scope', 'engineer', 'algorithm_1', 'algorithm_2', 'algorithm_3')
    by_diff = {}
    for d in diff_categories:
        rs = [r for r in test_results if r['difficulty'] == d]
        c = sum(1 for r in rs if r['correct'])
        by_diff[d] = {'correct': c, 'total': len(rs), 'accuracy': c / len(rs) if rs else 0}


    snapshot = {
        'milestone_idx': milestone_idx,
        'round':                                 state['round'],
        'session_snapshot_path': (
            str(milestone_base_snapshot) if milestone_base_snapshot else None),
        'summary': summary,
        'rule_scores':                     rule_scores,
        'found': found_count,
        'total':                       total,
        'test_accuracy': n_correct / len(TEST_TASKS) if TEST_TASKS else 0,
        'test_correct': n_correct,
        'test_total':                            len(TEST_TASKS),
        **{f'''{d}_accuracy''': by_diff[d]['accuracy'] for d in by_diff},
        **{f'''{d}_correct''': by_diff[d]['correct'] for d in by_diff},
        **{f'''{d}_total''': by_diff[d]['total'] for d in by_diff},
        'test_results': test_results,
        'summary_time': summary_time,
        'milestone_time':                               milestone_time }
    snapshots.append(snapshot)
    milestone_records.append({
        'milestone_idx': milestone_idx,
        'found':                                 found_count,
        'total':                                                       total,
        'rule_scores': rule_scores,
        'test_correct': n_correct,
        'test_total':                            len(TEST_TASKS),
        **{f'''{d}_accuracy''': by_diff[d]['accuracy'] for d in by_diff},
        'milestone_time': milestone_time })
    print(f'''\n[Milestone {milestone_idx}] 规则: {found_count}/{total}  测试: {n_correct}/{len(TEST_TASKS)}  用时: {milestone_time:.1f}s''')
    for d in diff_categories:
        bd = by_diff[d]
        if bd['total'] > 0:
            print(f'''  {d:12s}: {bd["correct"]}/{bd["total"]} ({bd["accuracy"]:.0%})''')

def _print_summary(seed_records, explore_records, snapshots, milestone_records,
                   token_usage=None):
    short = MODEL_SHORT or MODEL.rsplit('_', 1)[-1]
    print(f'''\n{"========================================================================"}''')
    print(f'''  EvoBench AlienCode Summary [{short}]''')
    print(f'''  {len(GROUND_TRUTH_SEXPR)} Discovery Targets | {len(TEST_TASKS)} Questions''')
    print(f'''{"========================================================================"}''')

    # v2 files its calibration demos as seed records with their own
    # classification, and the old summary counted into a fixed four-key dict --
    # an unknown label crashed the print after every score was already paid for.
    counts = collections.Counter(
        r.get('classification', 'CONFUSED') for r in seed_records)
    if counts.get('FIXED_DEMO'):
        print(f'''\n  Seed: {counts["FIXED_DEMO"]} fixed calibration demos''')
    else:
        print(f'''\n  Seed: MISLED={counts["MISLED"]} CORRECT={counts["CORRECT"]} EXPLORING={counts["EXPLORING"]} CONFUSED={counts["CONFUSED"]}''')

        t_aware = next((r['task_id'] for r in seed_records if r.get('classification') in ('EXPLORING', 'CORRECT')), '-')

        t_correct = next((r['task_id'] for r in seed_records if r.get('classification') == 'CORRECT'), '-')

        print(f'''  T_aware={t_aware}  T_correct={t_correct}''')

    print('\n  Rule Discovery Trajectory:')
    for s in snapshots:
        bar = '#' * s['found'] + '.' * (s['total'] - s['found'])
        print(f'''  M{s["milestone_idx"]}: {bar} {s["found"]}/{s["total"]}''')

    print('\n  Test Accuracy Trajectory:')
    for s in snapshots:
        print(f'''  M{s["milestone_idx"]}: {s["test_correct"]}/{s["test_total"]} ({s["test_accuracy"]:.2%})''')

    if snapshots:
        final = snapshots[-1]
        baseline = snapshots[0]
        gain = final['test_accuracy'] - baseline['test_accuracy']
        print(f'''\n  Evolution Gain: {baseline["test_accuracy"]:.0%} -> {final["test_accuracy"]:.0%} ({gain:+.0%})''')
        print(f'''  Final Rules: {final["found"]}/{final["total"]}''')

    if token_usage:
        print('\n  Token Usage:')
        print(f'''   Prompt tokens:     {token_usage["prompt_tokens"]:>10,}''')
        print(f'''   Completion tokens: {token_usage["completion_tokens"]:>10,}''')
        print(f'''   Total tokens:      {token_usage["total_tokens"]:>10,}''')






_ORACLE_ONLY = os.environ.get('EVAL_ORACLE_ONLY', '0').lower() in ('1', 'true', 'yes')


def _restore_protocol_config(saved: dict) -> None:
    """Re-apply the protocol flags the checkpoint was created under.

    These come from the environment at import time, so resuming without the
    original exports silently switched protocol mid-run -- an oracle or
    control run could finish as a plain one while the result JSON reported
    only the second half's settings.
    """
    global _ORACLE_ONLY, _CONTROL_MODE, _PROMPT_VARIANT, _HIDDEN_TESTS_ONLY, _EXPLICIT_GENERALIZATION, _CONTROL_BANK_PATH
    global PROTOCOL_V2, DEFER_HELDOUT, V2_ORACLE_DIAGNOSTICS
    global N_EXPLORE_ROUNDS, MAX_TOOL_CALLS_PER_ROUND
    global TASK_SET, TEST_TASKS

    # The task set is part of the protocol: a run that answered 90 questions
    # cannot be finished against 60. Checkpoints written before v2 existed have
    # no record of it, and the only set they could have used is the legacy one.
    was = saved.get('task_set', 'legacy')
    if was != TASK_SET:
        if _TASK_SET_FORCED:
            print(f'  [Checkpoint] task_set: checkpoint={was!r} overridden by '
                  f'env={TASK_SET!r}; questions both sets share are already '
                  'in the ledger and will not be re-asked')
        else:
            print(f'  [Checkpoint] task_set: env={TASK_SET!r} -> '
                  f'checkpoint={was!r}')
            if was == 'legacy':
                TEST_TASKS = _LEGACY_TASKS
            elif os.path.exists(_v2_path):
                with open(_v2_path, encoding='utf-8') as handle:
                    TEST_TASKS = json.load(handle)
            TASK_SET = was


    restored = {
                       'oracle_only': (_ORACLE_ONLY, saved.get('oracle_only', _ORACLE_ONLY)),
                        'control_mode': (_CONTROL_MODE, saved.get('control_mode', _CONTROL_MODE)),
                          'prompt_variant': (_PROMPT_VARIANT, saved.get('prompt_variant', _PROMPT_VARIANT)),
                             'hidden_tests_only': (_HIDDEN_TESTS_ONLY,
                              saved.get('hidden_tests_only', _HIDDEN_TESTS_ONLY)),
                       'protocol_v2': (PROTOCOL_V2,
                                      saved.get('protocol_v2', PROTOCOL_V2)),
                       'defer_heldout': (DEFER_HELDOUT,
                                         saved.get('defer_heldout', DEFER_HELDOUT)),
                       'v2_oracle_diagnostics': (
                           V2_ORACLE_DIAGNOSTICS,
                           saved.get('v2_oracle_diagnostics',
                                     V2_ORACLE_DIAGNOSTICS))}

    for name, (current, want) in restored.items():
        if current != want:
            print(f'''  [Checkpoint] {name}: env={current!r} -> checkpoint={want!r}''')

    _ORACLE_ONLY = restored['oracle_only'][1]
    _CONTROL_MODE = restored['control_mode'][1]
    _PROMPT_VARIANT = restored['prompt_variant'][1]
    _HIDDEN_TESTS_ONLY = restored['hidden_tests_only'][1]
    PROTOCOL_V2 = restored['protocol_v2'][1]
    DEFER_HELDOUT = restored['defer_heldout'][1]
    V2_ORACLE_DIAGNOSTICS = restored['v2_oracle_diagnostics'][1]
    if PROTOCOL_V2:
        N_EXPLORE_ROUNDS = int(saved.get('n_explore_rounds') or 1)
        MAX_TOOL_CALLS_PER_ROUND = int(
            saved.get('max_tool_calls_per_round')
            or saved.get('max_tool_calls')
            or protocol_v2.DEFAULT_PROBES_PER_BLOCK)
    _EXPLICIT_GENERALIZATION = _PROMPT_VARIANT != 'original'
    if saved.get('control_bank'):
        _CONTROL_BANK_PATH = saved['control_bank']
    for config_key, env_key in (
        ('reasoning_effort', 'EVAL_REASONING_EFFORT'),
        ('reasoning_mode', 'EVAL_REASONING_MODE'),
    ):
        value = saved.get(config_key)
        if value not in (None, '', 'default'):
            current = os.environ.get(env_key)
            if current != str(value):
                print(
                    f'  [Checkpoint] {env_key}: '
                    f'env={current!r} -> checkpoint={value!r}')
            os.environ[env_key] = str(value)


def _run_eval_impl(checkpoint=None):
    if checkpoint:
        _restore_protocol_config(checkpoint.get('config') or {})
    root_session = _init_agent_runtime(checkpoint)

    if checkpoint:
        state = {
                            'conversation': checkpoint['conversation'],
                     'round': checkpoint['round'],
                                'pending_feedback': checkpoint.get('pending_feedback'),
                            'session': root_session}

        seed_records = checkpoint.get('seed_records', [])
        explore_records = checkpoint.get('explore_records', [])
        snapshots = checkpoint.get('snapshots', [])
        milestone_records = checkpoint.get('milestone_records', [])
        deferred_milestones = _hydrate_deferred(
            checkpoint.get('deferred_milestones', []))
        oracle_diagnostics = checkpoint.get('oracle_diagnostics', [])
        token_log = checkpoint.get('token_log', [])
        done_phases = set(checkpoint.get('done_phases', []))
        ms_resume = checkpoint.get('milestone_resume')
        print('\n  From checkpoint')
        print(f'''  Done: {", ".join(sorted(done_phases)) or "none"}''')
    else:
        state = {
            'conversation': [
                {'role': 'system', 'content': _initial_system_prompt()}],
            'round': 0,
            'pending_feedback': None,
            'session': root_session}
        seed_records, explore_records, snapshots, milestone_records = [], [], [], []
        deferred_milestones = []
        oracle_diagnostics = []
        token_log = []
        done_phases = set()
        ms_resume = None

    def _snapshot_tokens(phase: str):
        usage = _target_usage()
        token_log.append({'phase': phase, **usage})
        print(f'''  [Token Usage after {phase}] prompt={usage["prompt_tokens"]}  completion={usage["completion_tokens"]}  total={usage["total_tokens"]}''')


    def _make_ckpt(extra_done=None, ms_state=None):
        phases = list(done_phases)
        if extra_done:
            phases.append(extra_done)
        return {
                     'model': MODEL, 'model_short': MODEL_SHORT,
                      'run_id': RUN_ID,
                      'framework': FRAMEWORK,
                      'track': EVAL_TRACK,
                      'budget_profile': BUDGET_PROFILE,
                            'conversation': state['conversation'],
                     'agent_session_state':
                         state['session'].state().to_dict(),
                     'agent_runtime_state':
                         _require_agent_runtime().snapshot(),
                     'agent_trace_path': _AGENT_TRACE_PATH,
                     'agent_snapshot_dir': _AGENT_SNAPSHOT_DIR,
                     'usage_snapshot': _target_usage(),
                     'round': state['round'],
                                'pending_feedback': state.get('pending_feedback'),
                            'seed_records': seed_records,
                               'explore_records': explore_records,
                         'snapshots': snapshots,
                                 'milestone_records': milestone_records,
                         'deferred_milestones':
                             _serializable_deferred(deferred_milestones),
                         'oracle_diagnostics': oracle_diagnostics,
                         'token_log': token_log,
                           'done_phases': phases,
                                'milestone_resume': ms_state,
                      'config': {
                'judge_model': JUDGE_MODEL,
                'n_explore_rounds': N_EXPLORE_ROUNDS,
                'n_explore_loops': N_EXPLORE_LOOPS,
                'max_context_turns': MAX_CONTEXT_TURNS,
                'max_explore_tests': MAX_EXPLORE_TESTS,
                'max_tool_calls_per_round': MAX_TOOL_CALLS_PER_ROUND,
                'max_parallel_tests': MAX_PARALLEL_TESTS,
                'max_seed_shows': MAX_SEED_SHOWS,
                'locked_seed': LOCKED_SEED,
                'protocol_v2': PROTOCOL_V2,
                'protocol_version': (
                    protocol_v2.PROTOCOL_VERSION if PROTOCOL_V2 else None),
                'defer_heldout': DEFER_HELDOUT,
                'v2_oracle_diagnostics': V2_ORACLE_DIAGNOSTICS,
                'max_output_lines': MAX_OUTPUT_LINES,
                'max_output_chars': MAX_OUTPUT_CHARS,
                'max_code_lines': MAX_CODE_LINES,
                'loop_weight': _LOOP_WEIGHT,
                'base_max_tokens': BASE_MAX_TOKENS,
                'oracle_only': _ORACLE_ONLY,
                'explicit_generalization': _EXPLICIT_GENERALIZATION,
                'hidden_tests_only': _HIDDEN_TESTS_ONLY,
                'task_set': TASK_SET,
                'task_count': len(TEST_TASKS),
                'prompt_variant': _PROMPT_VARIANT,
                'api_protocol': _require_agent_client().config.provider.value,
                'framework': FRAMEWORK,
                'track': EVAL_TRACK,
                'budget_profile': BUDGET_PROFILE,
                'reasoning_effort': _requested_effort(), 'reasoning_effort_asked_for': os.environ.get('EVAL_REASONING_EFFORT'), 'reasoning_mode': os.environ.get('EVAL_REASONING_MODE', 'standard'), 'save_raw_responses': True, 'raw_responses_path': (_AGENT_TRACE_PATH
                or None), 'control_mode': _CONTROL_MODE, 'control_bank': (_CONTROL_BANK_PATH
                or None), 'control_preflight': _CONTROL_PREFLIGHT}}

















    if 'seed' not in done_phases:
        _require_agent_runtime().begin_phase(PhaseContext(
            sandbox='code', phase='seed', label='Seed',
            track=EVAL_TRACK, framework=FRAMEWORK))
        _run_seed_phase(state, seed_records)
        _snapshot_tokens('seed')
        done_phases.add('seed')
        _save_checkpoint(_make_ckpt())

    if _ORACLE_ONLY:




        if 'oracle_inject' not in done_phases:
            print('\n========================================================================')
            print('  ORACLE INTERVENTION — injecting GT alien rules')
            print('========================================================================')
            from oracle_rules import build_oracle_message
            oracle_msg = build_oracle_message()
            state['conversation'].append({'role': 'user', 'content': oracle_msg})

            ack = _chat(state['session'], oracle_msg, label='Oracle Ack')
            state['conversation'].append(_assistant_turn(ack))
            print(f'''[Oracle Ack (first 500 chars)]\n{ack or ''[:500]}''')
            done_phases.add('oracle_inject')
            _save_checkpoint(_make_ckpt())
        if 'milestone_0' not in done_phases:
            resume = ms_resume if  ms_resume and ms_resume.get('milestone_idx') == 0 else None
            _run_milestone(state, 0, snapshots, milestone_records,
                           ckpt_data_fn=_make_ckpt, resume_data=resume)
            _snapshot_tokens('milestone_0')
            done_phases.add('milestone_0')
            ms_resume = None
            _save_checkpoint(_make_ckpt())


    elif   'milestone_0' not in done_phases:
        _require_agent_runtime().begin_phase(PhaseContext(
            sandbox='code', phase='milestone', label='M0',
            track=EVAL_TRACK, framework=FRAMEWORK))
        resume =     ms_resume if  ms_resume and ms_resume.get('milestone_idx') == 0 else None
        _run_milestone(    state, 0, snapshots, milestone_records,
                           ckpt_data_fn=_make_ckpt, resume_data=resume,
                           deferred=(
                               deferred_milestones if DEFER_HELDOUT else None))
        _snapshot_tokens(    'milestone_0')
        done_phases.add(    'milestone_0')
        ms_resume = None
        _save_checkpoint(    _make_ckpt())


    if PROTOCOL_V2 and STOP_AFTER_MILESTONE == 0:
        print(
            '\n[Protocol pause] ALIENCODE_STOP_AFTER_MILESTONE=0; '
            'M0 checkpoint is durable. Exiting before exploration.')
        _drain_streamed_scoring()
        _save_checkpoint(_make_ckpt())
        return

    for loop_i in range(
            1, (0 if _ORACLE_ONLY else N_EXPLORE_LOOPS) + 1):
        if f'''explore_{  loop_i}''' not in done_phases:
            if     _CONTROL_MODE == 'none':
                print(f'''\n[Control none] skipping explore loop {  loop_i}; conversation remains unchanged''')
            else:

                _require_agent_runtime().begin_phase(PhaseContext(
                    sandbox='code', phase='explore',
                    label=f'Explore {loop_i}',
                    track=EVAL_TRACK, framework=FRAMEWORK))
                _run_explore_phase(    state, loop_i, explore_records)
            _snapshot_tokens(f'''explore_{  loop_i}''')
            done_phases.add(f'''explore_{  loop_i}''')
            _save_checkpoint(    _make_ckpt())

        if f'''milestone_{  loop_i}''' not in done_phases:
            _require_agent_runtime().begin_phase(PhaseContext(
                sandbox='code', phase='milestone',
                label=f'M{loop_i}', track=EVAL_TRACK,
                framework=FRAMEWORK))
            resume =      ms_resume if  ms_resume and ms_resume.get('milestone_idx') == loop_i else None

            _run_milestone(    state, loop_i, snapshots, milestone_records,
                               ckpt_data_fn=_make_ckpt, resume_data=resume,
                               deferred=(
                                   deferred_milestones
                                   if DEFER_HELDOUT else None))
            _snapshot_tokens(f'''milestone_{  loop_i}''')
            done_phases.add(f'''milestone_{  loop_i}''')
            ms_resume = None
            _save_checkpoint(    _make_ckpt())

        if PROTOCOL_V2 and STOP_AFTER_MILESTONE == loop_i:
            print(
                f'\n[Protocol pause] '
                f'ALIENCODE_STOP_AFTER_MILESTONE={loop_i}; '
                f'M{loop_i} checkpoint is durable. '
                f'Exiting before M{loop_i + 1}.')
            _drain_streamed_scoring()
            _save_checkpoint(_make_ckpt())
            return

    _drain_streamed_scoring()

    if DEFER_HELDOUT and 'heldout_flushed' not in done_phases:
        _flush_deferred_milestones(
            deferred_milestones, snapshots, milestone_records,
            ckpt_data_fn=_make_ckpt)
        _snapshot_tokens('heldout_flushed')
        done_phases.add('heldout_flushed')
        _save_checkpoint(_make_ckpt())

    if (PROTOCOL_V2 and V2_ORACLE_DIAGNOSTICS
            and _CONTROL_MODE == 'self'
            and 'v2_oracle_diagnostics' not in done_phases):
        oracle_diagnostics[:] = _run_v2_oracle_diagnostics(
            deferred_milestones)
        _snapshot_tokens('v2_oracle_diagnostics')
        done_phases.add('v2_oracle_diagnostics')
        _save_checkpoint(_make_ckpt())

    final_usage = _target_usage()
    _print_summary(seed_records, explore_records, snapshots, milestone_records,
                   final_usage)
    root_snapshot_path = state['session'].snapshot()

    # The harness stamps any shortfall at all; the analysis decides later how
    # much it is willing to tolerate.
    validity = derive_validity(
        {'snapshots': snapshots, 'explore_records': explore_records},
        tolerance=0)
    if not validity['ok']:
        print(
            '\n' + '!' * 72
            + f'\n  [Damaged run] {RUN_ID or MODEL_SHORT}: '
            + '; '.join(validity['reasons'])
            + '\n  结果仍会写盘以便修复，但已标记为不可用，分析脚本会跳过它。\n'
            + '!' * 72)

    short = MODEL_SHORT or MODEL.rsplit('_', 1)[-1]
    suffix = f'''_{RUN_ID}''' if RUN_ID else ''
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(_RESULTS_DIR, f'''eval_results_alien_code_{short}{suffix}.json''')
    snapshots_slim = []
    for snap in snapshots:
        slim = {k: v for k, v in snap.items() if k != 'summary'}
        if 'test_results' in slim:
            slim['test_results'] = [{k: v for k, v in tr.items() if k != 'model_response'} for tr in slim['test_results']]



        snapshots_slim.append(slim)

    results_tmp = results_path + '.tmp'
    with open(results_tmp, 'w', encoding='utf-8') as f:
        json.dump({
            'model': MODEL,
            'model_short':                 MODEL_SHORT,
            'run_id': RUN_ID,
            'framework': FRAMEWORK,
            'track': EVAL_TRACK,
            'framework_version': _require_agent_runtime().snapshot().get(
                'framework_spec', {}
            ),
            'budget_profile': _require_agent_runtime().snapshot().get(
                'budget_profile', {}
            ),
            'artifact_hash': _require_agent_runtime().snapshot().get(
                'artifact_digest'
            ),
            'fork_mode': 'closed_book_tools_empty',
            'timestamp': datetime.now().isoformat(),
            'world': {'name': 'AlienCode', 'rule_count': 30, 'discovery_targets': 31, 'layers': 8,
                                         'identity_rules': 8},
            'config': {
                'seed_tasks': len(SEED_TASKS),
                'test_tasks':                                len(TEST_TASKS),
                'n_explore_rounds': N_EXPLORE_ROUNDS,
                'n_explore_loops':                                       N_EXPLORE_LOOPS,
                'max_explore_tests': MAX_EXPLORE_TESTS,
                'max_tool_calls_per_round': MAX_TOOL_CALLS_PER_ROUND,
                'max_parallel_tests': MAX_PARALLEL_TESTS,
                'max_seed_shows': MAX_SEED_SHOWS,
                'locked_seed': LOCKED_SEED,
                'protocol_v2': PROTOCOL_V2,
                'protocol_version': (
                    protocol_v2.PROTOCOL_VERSION if PROTOCOL_V2 else None),
                'defer_heldout': DEFER_HELDOUT,
                'v2_oracle_diagnostics': V2_ORACLE_DIAGNOSTICS,
                'max_output_lines': MAX_OUTPUT_LINES,
                'max_output_chars': MAX_OUTPUT_CHARS,
                'max_code_lines': MAX_CODE_LINES,
                'loop_weight': _LOOP_WEIGHT,
                'base_max_tokens': BASE_MAX_TOKENS,
                'oracle_only': _ORACLE_ONLY,
                'explicit_generalization': _EXPLICIT_GENERALIZATION,
                'hidden_tests_only': _HIDDEN_TESTS_ONLY,
                'task_set': TASK_SET,
                'task_count': len(TEST_TASKS),
                'prompt_variant': _PROMPT_VARIANT,
                'framework': FRAMEWORK,
                'track': EVAL_TRACK,
                'budget_profile': BUDGET_PROFILE,
                'api_protocol': _require_agent_client().config.provider.value,
                'reasoning_effort': _requested_effort(),
                'reasoning_effort_asked_for': os.environ.get('EVAL_REASONING_EFFORT'),
                'reasoning_mode': os.environ.get('EVAL_REASONING_MODE', 'standard'),
                'reasoning_continuation': 'provider_native',
                'send_reasoning_back': True,
                'reasoning_back_turns': None,
                'reasoning_back_max_chars': None,
                'save_raw_responses': True,
                'raw_responses_path': _AGENT_TRACE_PATH,
                'control_mode': _CONTROL_MODE,
                'control_bank': (_CONTROL_BANK_PATH
                        or None),
                'control_preflight': _CONTROL_PREFLIGHT },
            'agent_trace_path': _AGENT_TRACE_PATH,
            'root_session_id': state['session'].session_id,
            'root_session_snapshot_path': (
                str(root_snapshot_path) if root_snapshot_path else None),
            'root_session_history_event_count': len(
                state['session'].history),
            'root_session_provider_history_item_count': len(
                state['session'].provider_history),
            'root_session_last_response_id':
                state['session'].last_response_id,
            'agent_runtime_state': _require_agent_runtime().snapshot(),
            'validity': validity,
            'protocol_manifest': (
                protocol_v2.manifest(alien_exec) if PROTOCOL_V2 else None),
            'token_usage': final_usage,
            'token_log': token_log,
            'seed_records': [{k: v for k, v in r.items() if k != 'model_response'} for r in seed_records],
            'explore_records': [{k: v for k, v in r.items() if k != 'model_response'} for r in explore_records],
            'oracle_diagnostics': oracle_diagnostics,
            'milestones': milestone_records,
            'snapshots':                                  snapshots_slim }, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(results_tmp, results_path)
    _clear_checkpoint()









    print(f'''\n结果已保存到 {results_path}''')

    if _SAVE_TRANSCRIPT:


        tpath = _TRANSCRIPT_PATH or _transcript_path()
        n_turns = n_cot = 0
        try:
            with open(tpath, 'r', encoding='utf-8') as tf:
                for line in tf:
                    line = line.strip()
                    if not line:
                        continue
                    n_turns += 1
                    try:
                        if json.loads(line).get('reasoning_len', 0) > 0:
                            n_cot += 1
                        continue
                    except Exception:
                        continue

            size_mb = os.path.getsize(tpath) / 1e+06
            print(f'''轨迹已保存到 {tpath}  ({n_turns} 轮, {n_cot} 含CoT, {size_mb:.1f} MB)''')
        except FileNotFoundError:
            print(f'''[TRANSCRIPT] 未找到轨迹文件 {tpath}''')
            return None

def run_eval(checkpoint=None):
    os.makedirs(_LOG_DIR, exist_ok=True)
    retired = None if checkpoint else _retire_run_artifacts()
    log_path = _log_path()
    # One log per run id, appended to, so a run picked up from a checkpoint
    # keeps writing where it left off instead of scattering across files. The
    # banner marks where each attempt begins.
    log_file = open(log_path, 'a', encoding='utf-8')
    log_file.write(
        f'\n{"=" * 72}\n'
        f'[{"resume" if checkpoint else "start"}] '
        f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  '
        f'model={MODEL}  run={RUN_ID or "-"}\n'
        f'{"=" * 72}\n'
    )
    log_file.flush()
    old_stdout = sys.stdout
    sys.stdout = _Tee(old_stdout, log_file, max_log_bytes=MAX_LOG_BYTES)
    if retired:
        print(f'[Superseded] 同名 run 的旧产物已移至 {retired}')
    _init_transcript(resuming=checkpoint is not None)
    try:
        _run_eval_impl(checkpoint=checkpoint)
        export_run_artifacts(_AGENT_TRACE_PATH, _DATA_DIR)
    except LogSizeLimitExceeded as exc:
        sys.stdout = old_stdout
        print(f'''[ABORT] {exc}''')
        raise RuntimeError(str(exc)) from exc
    finally:
        sys.stdout = old_stdout
        log_file.close()
        print(f'''评测日志: {log_path}''')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='AlienCode Evaluation')
    parser.add_argument('--model', type=str, default='', help='Model name')
    parser.add_argument('--short', type=str, default='', help='Short name')
    parser.add_argument('--run-id', type=str, default='', help='Run ID for isolating checkpoints/results')
    parser.add_argument('--checkpoint', type=str, default='', help='Checkpoint path')
    parser.add_argument('--fresh', action='store_true',
             help="Start over even if this run id has a checkpoint (the old "
                  "files are set aside under _superseded/, not deleted)")
    parser.add_argument('--resume', action='store_true',
             help="Pick up this run id's checkpoint. Already the default when "
                  "--fresh is absent; accepted so that a caller driving both "
                  "sandboxes can say so the same way in each")
    parser.add_argument('--framework', type=str,
             default=os.environ.get('EVAL_FRAMEWORK', FRAMEWORK),
             help='Framework adapter: baseline, reflexion, ace, evotest, gepa, agent_factory')
    parser.add_argument('--track', choices=('controlled', 'native', 'open'),
             default=os.environ.get('EVAL_TRACK', EVAL_TRACK),
             help='Framework evaluation track')
    parser.add_argument('--budget-profile', type=str,
             default=os.environ.get('EVAL_BUDGET_PROFILE', BUDGET_PROFILE),
             help='Framework budget profile: c1/1x/2x/4x')
    parser.add_argument('--control-mode', choices=sorted(_VALID_CONTROL_MODES),
        default=_CONTROL_MODE,
             help='Causal control: self, none, passive donor replay, or random probes')
    parser.add_argument('--control-bank', default=_CONTROL_BANK_PATH,
             help='Standard-run transcript JSONL used by --control-mode passive')


    parser.add_argument('--control-manifest', default=_CONTROL_MANIFEST_PATH,
             help='Optional donor provenance manifest; mismatches fail preflight')


    args = parser.parse_args()

    RUN_ID = args.run_id
    _RUN_LOCK_HANDLE = acquire_run_lock(RUN_ID)
    FRAMEWORK = args.framework
    EVAL_TRACK = args.track
    BUDGET_PROFILE = args.budget_profile
    os.environ['EVAL_FRAMEWORK'] = FRAMEWORK
    os.environ['EVAL_TRACK'] = EVAL_TRACK
    os.environ['EVAL_BUDGET_PROFILE'] = BUDGET_PROFILE
    _CONTROL_MODE = args.control_mode
    _CONTROL_BANK_PATH = args.control_bank
    _CONTROL_MANIFEST_PATH = args.control_manifest
    # The gate that used to stand here refused passive and random under v2
    # until the autonomous leaderboard was frozen, so that a control could
    # not be built against a protocol still moving. That has happened: the
    # eleven-system cohort is complete and its task set is fixed.
    #
    # The replay machinery needed nothing for v2. A donor is keyed by round
    # label, and v2's labels are `Explore 1-1`..`4-1` with up to twelve
    # probes each, which is what the builder now writes; the manifest still
    # matches loop and round structure, so a bank recorded under the old
    # four-loops-by-three shape fails preflight rather than replaying into
    # the wrong ladder.
    if _CONTROL_MODE in {'random', 'passive'} and not _CONTROL_BANK_PATH:
        parser.error(f'''--control-mode {_CONTROL_MODE} requires --control-bank to pair probe volume with a self run''')


    if args.model or _LOCAL_SETTINGS.get('model'):
        MODEL = args.model or str(_LOCAL_SETTINGS.get('model'))
        MODEL_SHORT = args.short or MODEL.rsplit('_', 1)[-1]
    else:
        MODEL = input('模型名称 MODEL (例如 api_azure_openai_gpt-5.1): ').strip()
        if not MODEL:
            MODEL = 'api_azure_openai_gpt-5.1'
        MODEL_SHORT = input('模型简称 MODEL_SHORT (例如 gpt-5.1): ').strip()
        if not MODEL_SHORT:
            MODEL_SHORT = 'gpt-5.1'

    _preflight_control_bank()

    # A run id whose checkpoint is still on disk is an interrupted run, so pick
    # it up rather than starting over: that is what the stable, appended-to
    # artifacts are for, and a fresh start under the same id sets hours of
    # exploration aside. `--fresh` is how you ask for the old behaviour.
    ckpt = None
    ckpt_path = args.checkpoint or _ckpt_path()
    wants_fresh = args.fresh and not args.checkpoint and not args.resume
    # A finished run clears its checkpoint, so "resume" with a result already on
    # disk is not a resume: it would retire that result and score the model
    # again. A damaged result has to stay damaged unless someone asks for a
    # replacement in as many words.
    if not wants_fresh and not os.path.exists(ckpt_path):
        short = MODEL_SHORT or MODEL.rsplit('_', 1)[-1]
        suffix = f'_{RUN_ID}' if RUN_ID else ''
        existing = os.path.join(
            _RESULTS_DIR, f'eval_results_alien_code_{short}{suffix}.json')
        if os.path.exists(existing):
            print(
                f'  [Terminal] {RUN_ID or short} already has a result and no '
                f'checkpoint: {existing}\n'
                '  Nothing to resume. Pass --fresh to archive it and re-run.')
            sys.exit(0)
    if not wants_fresh and os.path.exists(ckpt_path):
        with open(ckpt_path, 'r', encoding='utf-8') as f:
            ckpt = json.load(f)
        print(f'  [Checkpoint] 从 {ckpt_path} 续跑')
    run_eval(checkpoint=ckpt)