#!/usr/bin/env python3
"""Check that a model actually speaks its native agent protocol.

Runs a short real conversation against whatever model `dev/eval.local.toml`
points at, then reports whether the pieces the evaluation depends on are
present: reasoning that survives across turns, a tool call/result round trip,
per-call usage, and continuation state (Claude/Gemini signatures, OpenAI
`previous_response_id`).

    cp dev/eval.local.example.toml dev/eval.local.toml   # then fill in keys
    python3 dev/scripts/check_protocol.py

Writes the raw trace, session snapshots and an HTML replay under the
configured output directory.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


DEV = Path(__file__).resolve().parents[1]
if str(DEV) not in sys.path:
    sys.path.insert(0, str(DEV))

from common import local_config  # noqa: E402
from common.agent_client import (  # noqa: E402
    AgentClient,
    AgentClientConfig,
    Provider,
    ToolResult,
)
from scripts.audit_agent_trace import _load, audit  # noqa: E402
from scripts.render_interaction import render_file  # noqa: E402


SYSTEM = (
    "You are being checked for protocol compliance. Think before you answer. "
    "When a tool is available and the user asks for it, call it exactly once "
    "with the requested arguments."
)

TOOLS = [{
    "name": "echo",
    "description": "Return the supplied text unchanged.",
    "input_schema": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    },
}]

# ((37*41)+19) % 97 == 81
_ANSWER = "81"
_ECHO = f"signature-roundtrip-{_ANSWER}"


def resolve_model(
    settings: local_config.LocalConfig,
    args: argparse.Namespace,
) -> str:
    model = args.model or settings.get("model")
    if not model:
        raise local_config.LocalConfigError(
            "no model configured; set [run].model in "
            f"{settings.path or local_config.default_config_path()} "
            "or pass --model"
        )
    return str(model)


def build_config(
    settings: local_config.LocalConfig,
    args: argparse.Namespace,
    model: str,
    trace: Path,
) -> AgentClientConfig:
    effort = args.reasoning_effort or settings.get("reasoning_effort", "high")
    config = AgentClientConfig(
        model=model,
        provider=args.provider or settings.get("provider") or None,
        timeout=float(settings.get("timeout", 600)),
        max_retries=int(settings.get("max_retries", 4)),
        trace_path=str(trace),
        snapshot_dir=f"{trace.with_suffix('')}_snapshots",
        settings={
            "max_tokens": int(
                args.max_tokens or settings.get("max_tokens", 4096)
            ),
            "reasoning_effort": str(effort),
        },
    )
    if config.chat_route in ("moonshot", "xai"):
        # Both retired max_tokens for max_completion_tokens.
        config.settings["max_completion_tokens"] = max(
            int(config.settings.pop("max_tokens")), 8192
        )
    if config.chat_route == "xai":
        # Reasoning cannot be turned off here and the tiers stop at 'high', so
        # the 'max' other vendors take would come back a 400.
        if config.settings["reasoning_effort"] not in ("low", "medium", "high"):
            config.settings["reasoning_effort"] = "high"
    if config.provider is Provider.ANTHROPIC:
        # Adaptive thinking keeps the signed block while making a summary
        # visible, which is what the evaluation replays between turns. It is
        # an Anthropic feature though: a vendor borrowing the protocol only
        # knows enabled/disabled with a budget.
        config.settings["thinking"] = (
            {"type": "enabled", "budget_tokens": 60000}
            if config.standard_messages
            else {
                "type": "adaptive",
                "display": str(
                    settings.get("thinking_display", "summarized")
                ),
            }
        )
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", help="override [run].model")
    parser.add_argument("--provider", help="force a protocol")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "one round only: prompt plus the tool round trip, skipping the "
            "forked continuation turn"
        ),
    )
    parser.add_argument(
        "--no-html", action="store_true", help="skip the HTML replay"
    )
    args = parser.parse_args()

    try:
        settings = local_config.load(args.config, required=True)
    except local_config.LocalConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out_dir = args.output_dir or settings.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        model = resolve_model(settings, args)
        # Splitting on the dot would cut model ids apart at their version,
        # naming grok-4.5's artifacts "protocol_5"; the route prefix is what
        # this drops.
        slug = model.split("_")[-1].replace("/", "-")
        trace = out_dir / f"protocol_{slug}_{stamp}.jsonl"
        config = build_config(settings, args, model, trace)
    except local_config.LocalConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"config     {settings.path}")
    print(f"model      {config.model}")
    print(f"protocol   {config.provider.value}")
    print(f"endpoint   {config.endpoint}")
    print(f"trace      {trace}\n", flush=True)

    client = AgentClient(config)
    findings: dict[str, Any] = {}
    try:
        findings = run_probe(client, with_fork=not args.quick)
    except Exception as exc:  # surfaced in the report rather than a traceback
        print(f"call failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        findings = {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        client.close()

    history = findings.get("history") or []
    history_path = out_dir / f"protocol_{slug}_{stamp}.messages_history.json"
    if history:
        history_path.write_text(
            json.dumps(
                {
                    "model": config.model,
                    "provider": config.provider.value,
                    "session_id": findings.get("session_id"),
                    "usage": findings.get("usage"),
                    "messages_history": history,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    records, parse_issues = _load(trace) if trace.exists() else ([], [])
    trace_audit = audit(records) if records else {"issues": [], "ok": False}
    trace_audit["issues"] = parse_issues + list(trace_audit.get("issues", []))
    trace_audit["ok"] = not trace_audit["issues"] and bool(records)

    checks = build_checks(config, findings, trace_audit)
    report = {
        "ok": all(item["ok"] for item in checks),
        "model": config.model,
        "provider": config.provider.value,
        "endpoint": config.endpoint,
        "checks": checks,
        "usage": findings.get("usage"),
        "history_kind_counts": _kind_counts(history),
        "trace": str(trace),
        "messages_history": str(history_path) if history else None,
        "audit_issues": trace_audit["issues"],
    }

    report_path = out_dir / f"protocol_{slug}_{stamp}.report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for item in checks:
        mark = "PASS" if item["ok"] else "FAIL"
        print(f"[{mark}] {item['name']}: {item['detail']}")

    usage = findings.get("usage") or {}
    if usage:
        print(
            f"\ntokens     in={usage.get('prompt_tokens', 0):,} "
            f"out={usage.get('completion_tokens', 0):,} "
            f"reasoning={usage.get('reasoning_tokens', 0):,} "
            f"cache_hit={usage.get('cache_hit_ratio', 0) * 100:.1f}%"
        )
    if history:
        counts = _kind_counts(history)
        print(
            "history    "
            + ", ".join(f"{kind}={count}" for kind, count in counts.items())
        )
        print(f"messages   {history_path}")

    if not args.no_html and trace.exists():
        page = render_file(
            trace,
            trace.with_suffix(".html"),
            title=f"{config.model} — protocol check",
        )
        print(f"replay     {page}")
    print(f"report     {report_path}")

    return 0 if report["ok"] else 3


def run_probe(
    client: AgentClient,
    *,
    with_fork: bool = True,
) -> dict[str, Any]:
    """Reason, call a tool, hand the result back, then recall the context."""

    session = client.create_session(system=SYSTEM, tools=TOOLS)

    first = session.send_user(
        "Compute ((37 * 41) + 19) modulo 97. Think it through, then call the "
        "echo tool exactly once with the text 'signature-roundtrip-' followed "
        "immediately by the decimal result.",
        label="turn 1 · reason and call tool",
    )

    tool_calls = list(first.tool_calls)
    final = None
    if tool_calls:
        call = tool_calls[0]
        final = session.submit_tool_results(
            [
                ToolResult(
                    call_id=item.call_id,
                    name=item.name,
                    output={"text": item.arguments.get("text", "")},
                )
                for item in tool_calls
            ],
            label="turn 2 · tool result",
        )
    else:
        call = None

    recall = None
    if with_fork:
        recall = session.fork().send_user(
            "Without recomputing, what number did you send to the echo tool?",
            label="turn 3 · continuation check (forked)",
        )

    # The full history is the audit artifact: user and assistant messages,
    # reasoning, tool calls, tool results and per-exchange usage.
    history = session.history
    return {
        "first": first,
        "final": final,
        "call": call,
        "recall": recall,
        "checked_fork": with_fork,
        "usage": client.get_usage(),
        "history": history,
        "history_kinds": [event["kind"] for event in history],
        "session_id": session.session_id,
    }


def _kind_counts(history: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in history:
        kind = str(event.get("kind", "?"))
        counts[kind] = counts.get(kind, 0) + 1
    return dict(sorted(counts.items()))


def build_checks(
    config: AgentClientConfig,
    findings: dict[str, Any],
    trace_audit: dict[str, Any],
) -> list[dict[str, Any]]:
    if "error" in findings:
        return [{
            "name": "provider call",
            "ok": False,
            "detail": findings["error"],
        }]

    first = findings.get("first")
    final = findings.get("final")
    call = findings.get("call")
    recall = findings.get("recall")
    kinds = set(findings.get("history_kinds") or [])
    usage = findings.get("usage") or {}
    signed = config.provider in {Provider.ANTHROPIC, Provider.GEMINI}

    checks: list[dict[str, Any]] = []

    reasoning = list(getattr(first, "reasoning", []) or [])
    checks.append({
        "name": "reasoning returned",
        "ok": bool(reasoning),
        "detail": (
            f"{len(reasoning)} artifact(s), "
            f"{sum(len(item.text or '') for item in reasoning)} chars"
            if reasoning
            else "provider returned no reasoning artifacts"
        ),
    })

    if signed:
        signatures = [item for item in reasoning if item.signature]
        checks.append({
            "name": "reasoning signature",
            "ok": bool(signatures),
            "detail": (
                f"{len(signatures)} signed block(s) captured for replay"
                if signatures
                else "no signature returned; context would be lost on replay"
            ),
        })
    else:
        checks.append({
            "name": "response id continuation",
            "ok": bool(getattr(first, "response_id", None)),
            "detail": (
                f"response_id {getattr(first, 'response_id', None)}"
                if getattr(first, "response_id", None)
                else "no response id; cannot chain previous_response_id"
            ),
        })

    checks.append({
        "name": "tool call",
        "ok": call is not None and call.name == "echo",
        "detail": (
            f"{call.name}({json.dumps(call.arguments, ensure_ascii=False)})"
            if call is not None
            else "model never called the tool"
        ),
    })

    answer = getattr(final, "text", "") or ""
    # What is being checked is that the tool's output reached the model, not
    # how it chose to word the reply: some models quote the echoed string back
    # verbatim while others just state the value it carried.
    verbatim = _ECHO in answer
    value = _ECHO.rsplit("-", 1)[-1]
    used = verbatim or re.search(rf"\b{re.escape(value)}\b", answer) is not None
    checks.append({
        "name": "tool result round trip",
        "ok": final is not None and used,
        "detail": (
            f"final answer contains {_ECHO!r}"
            if verbatim
            else f"final answer carries the tool's value {value!r}"
            if used
            else "model did not use the tool result in its answer"
        ),
    })

    required = {"message", "reasoning", "tool_call", "tool_result", "usage"}
    missing = sorted(required - kinds)
    checks.append({
        "name": "messages_history completeness",
        "ok": not missing,
        "detail": (
            "kept " + ", ".join(sorted(kinds))
            if not missing
            else f"missing {', '.join(missing)}; kept {', '.join(sorted(kinds))}"
        ),
    })

    checks.append({
        "name": "usage recorded",
        "ok": bool(usage.get("total_tokens")),
        "detail": (
            f"{usage.get('call_count', 0)} calls, "
            f"{usage.get('total_tokens', 0):,} tokens, "
            f"cache hit {usage.get('cache_hit_ratio', 0) * 100:.1f}%"
        ),
    })

    if findings.get("checked_fork"):
        recall_text = getattr(recall, "text", "") or ""
        checks.append({
            "name": "context carried across turns",
            "ok": _ANSWER in recall_text,
            "detail": (
                f"forked session recalled {_ANSWER}"
                if _ANSWER in recall_text
                else f"expected {_ANSWER} in: {recall_text[:120]!r}"
            ),
        })

    checks.append({
        "name": "trace audit",
        "ok": bool(trace_audit.get("ok")),
        "detail": (
            "no issues"
            if trace_audit.get("ok")
            else "; ".join(trace_audit.get("issues", [])[:3]) or "empty trace"
        ),
    })
    return checks


if __name__ == "__main__":
    raise SystemExit(main())
