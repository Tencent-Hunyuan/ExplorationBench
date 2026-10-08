#!/usr/bin/env python3
"""Reproduce the closed-book turn that follows a tool-using conversation.

A graded turn offers no tool, but the history it forks from is full of tool
calls. Some models answer out of that habit -- with a real call against a tool
the request never offered, or with nothing at all -- and a grader reading only
the text scores that zero. This walks the exact shape: explore with the tool,
fork without it, ask for an answer, and report what came back.

    python3 scripts/check_closed_book.py --model api_doubao_doubao-seed-2-1-pro-260628
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import local_config  # noqa: E402
from common.agent_client import (  # noqa: E402
    AgentClient,
    AgentClientConfig,
    ToolResult,
)
from common.agent_client.routing import (  # noqa: E402
    chat_route_from_model,
    responses_route_from_model,
)

TOOL_NAME = "run_aliencode"

SYSTEM = (
    "你在学习一门外星编程语言 AlienCode。它的语法与地球语言不同：\n"
    "EMIT(x) 输出 x；SET a AS b 赋值；STRAND() 创建列表；ANNEX(lst, x) 追加。\n"
    "你可以调用工具在真实解释器里做实验，然后总结规则。"
)

TOOL = {
    "name": TOOL_NAME,
    "description": "在真实的 AlienCode 解释器中执行一段代码，返回真实执行输出。",
    "input_schema": {
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"],
    },
}

EXPLORE = (
    "请调用工具做一个实验：执行 EMIT(\"abc\") 并观察输出，"
    "然后告诉我你看到了什么。"
)

# The interpreter reverses strings on output. The model has to carry that rule
# into the closed-book turn, which is what makes a silent turn cost a point.
TOOL_OUTPUT = "cba"

TEST = (
    "现在是闭卷作答，没有任何工具可用。\n"
    "请写一段 AlienCode，让它输出字符串 hello。\n"
    "把代码放在 ```aliencode 代码块里。"
)

DENIAL = (
    f"[ToolError] 本轮为闭卷作答，{TOOL_NAME} 不可用，本次调用没有被执行。\n"
    "请直接把最终答案写出来：代码放在 ```aliencode 代码块里。"
)


def call_overrides(model: str) -> dict[str, object]:
    """The per-call knobs this route needs, matching what the harness sends."""

    if responses_route_from_model(model) == "doubao":
        return {
            "max_tokens": 32768,
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
            "system_in_input": True,
        }
    if chat_route_from_model(model) == "moonshot":
        # max_tokens is retired here, and the tier has to be asked for.
        return {"max_completion_tokens": 32768, "reasoning_effort": "max"}
    if chat_route_from_model(model) == "xai":
        # Same retired field, but the tiers stop at 'high'.
        return {"max_completion_tokens": 32768, "reasoning_effort": "high"}
    return {"max_tokens": 32768}


def replay(args: argparse.Namespace) -> int:
    """Pick up a real closed-book fork and see whether it can be salvaged."""

    # Loading exports the credentials the snapshot deliberately left out.
    local_config.load(required=True)
    client, session = AgentClient.from_snapshot(args.from_snapshot)
    overrides = call_overrides(args.model)

    pending = session.pending_tool_calls
    history = session.history
    print(f"snapshot   {Path(args.from_snapshot).name}")
    print(f"model      {client.config.model}")
    print(f"tools      {[t.name for t in session.tools] or '无（闭卷）'}")
    print(f"history    {len(history)} 条事件, "
          f"{sum(1 for e in history if e.get('kind') == 'tool_call')} 次工具调用")
    print(f"pending    {[c.name for c in pending] or '无'}\n")

    if not pending:
        print("这个快照没有挂起的工具调用，换一个再试")
        return 1

    print(f"复现：闭卷轮里模型调用了 {pending[0].name}，参数就是它的答案")
    print(f"  {json.dumps(pending[0].arguments, ensure_ascii=False)[:160]}\n")
    turn = session.submit_tool_results(
        [
            ToolResult(call_id=c.call_id, output=DENIAL, name=c.name,
                       is_error=True)
            for c in pending
        ],
        label="Replay (no-tool retry)",
        request_overrides=overrides,
    )
    text = (turn.text or "").strip()
    meta = session.last_call_meta
    print(f"拒绝后重问 -> 文本 {len(text)} 字 | 仍在调用工具 "
          f"{len(turn.tool_calls)} | 传输重试 {meta.get('transport_retry_count')}")
    if text:
        print("\n模型答案:")
        print("\n".join(text.splitlines()[:12]))
    return 0 if text else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--rounds", type=int, default=3,
                        help="how many closed-book questions to sample")
    parser.add_argument(
        "--from-snapshot",
        help=(
            "replay a session snapshot from a real run instead of building a "
            "fresh conversation; a short synthetic history rarely provokes the "
            "habit, a milestone's worth of tool calls does"
        ),
    )
    args = parser.parse_args()

    if args.from_snapshot:
        return replay(args)

    settings = local_config.load(required=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = args.model.split("_")[-1].split("/")[-1]
    out_dir = settings.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    trace = out_dir / f"closedbook_{short}_{stamp}.jsonl"

    config = AgentClientConfig(
        model=args.model,
        timeout=float(settings.get("timeout", 600)),
        max_retries=int(settings.get("max_retries", 6)),
        trace_path=str(trace),
    )
    overrides = call_overrides(args.model)
    client = AgentClient(config)

    print(f"model    {args.model}")
    print(f"endpoint {config.endpoint}")
    print(f"trace    {trace}\n")

    explorer = client.create_session(system=SYSTEM, tools=[TOOL])
    turn = explorer.send_user(EXPLORE, label="Explore", request_overrides=overrides)
    print(f"[explore] 工具调用 {len(turn.tool_calls)} 次  "
          f"文本 {len(turn.text or '')} 字")
    if turn.tool_calls:
        turn = explorer.submit_tool_results(
            [
                ToolResult(call_id=c.call_id, output=TOOL_OUTPUT, name=c.name)
                for c in turn.tool_calls
            ],
            label="Explore tool-results",
            request_overrides=overrides,
        )
        print(f"[explore] 收到结果后文本 {len(turn.text or '')} 字")

    stats = {"silent": 0, "tool_call": 0, "answered_first": 0, "recovered": 0}
    for i in range(1, args.rounds + 1):
        graded = explorer.fork(tools=[])
        label = f"Closed-book {i}"
        turn = graded.send_user(TEST, label=label, request_overrides=overrides)
        text = (turn.text or "").strip()
        calls = turn.tool_calls
        meta = graded.last_call_meta
        print(f"\n[{label}] 文本 {len(text)} 字 | 工具调用 {len(calls)} "
              f"| 传输重试 {meta.get('transport_retry_count')} "
              f"| 缓存命中 {meta.get('cache_hit_ratio')}")
        if calls:
            stats["tool_call"] += 1
            print(f"  模型调用了 {calls[0].name}，参数: "
                  f"{json.dumps(calls[0].arguments, ensure_ascii=False)[:120]}")
            turn = graded.submit_tool_results(
                [
                    ToolResult(call_id=c.call_id, output=DENIAL, name=c.name,
                               is_error=True)
                    for c in calls
                ],
                label=f"{label} (no-tool retry)",
                request_overrides=overrides,
            )
            text = (turn.text or "").strip()
            print(f"  拒绝后重问 -> 文本 {len(text)} 字 | "
                  f"仍在调用工具 {len(turn.tool_calls)}")
            if text:
                stats["recovered"] += 1
        elif not text:
            stats["silent"] += 1
        else:
            stats["answered_first"] += 1
        if text:
            print(f"  答案首行: {text.splitlines()[0][:100]}")

    print("\n=== 汇总 ===")
    print(f"直接作答      {stats['answered_first']}/{args.rounds}")
    print(f"发工具调用    {stats['tool_call']}/{args.rounds}"
          f"（其中拒绝后救回 {stats['recovered']}）")
    print(f"完全无输出    {stats['silent']}/{args.rounds}")
    usable = stats["answered_first"] + stats["recovered"]
    print(f"最终可判分    {usable}/{args.rounds}")
    return 0 if usable == args.rounds else 1


if __name__ == "__main__":
    raise SystemExit(main())
