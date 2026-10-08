#!/usr/bin/env python3
"""Audit provider-native reasoning continuity and cache counters in a trace."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.agent_client.session import fingerprint  # noqa: E402


def _load(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON: {exc}")
                continue
            if not isinstance(value, dict):
                errors.append(f"line {line_number}: record is not an object")
                continue
            value["_line_number"] = line_number
            records.append(value)
    return records, errors


def audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    issues: list[str] = []
    exchanges = [
        record for record in records
        if record.get("record_type") == "api_exchange"
    ]
    successful = [
        record for record in exchanges
        if record.get("status") == "success"
    ]
    trajectories = [
        record for record in records
        if record.get("record_type") == "trajectory"
    ]
    by_provider: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_write_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
    )
    sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    session_timelines: dict[str, list[dict[str, Any]]] = defaultdict(list)
    turn_of = {
        str(record.get("exchange_id", "")): record
        for record in trajectories
        if record.get("exchange_id")
    }
    response_ids = {
        str((record.get("response") or {}).get("id"))
        for record in successful
        if (record.get("response") or {}).get("id")
    }
    for record in records:
        record_type = record.get("record_type")
        if (
            record_type == "session_restore"
            or (
                record_type == "api_exchange"
                and record.get("status") == "success"
            )
        ):
            session_timelines[str(record.get("session_id", ""))].append(
                record
            )

    for record in successful:
        provider = str(record.get("provider", "unknown"))
        bucket = by_provider[provider]
        bucket["calls"] += 1
        usage = record.get("usage") or {}
        for key in (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cache_read_input_tokens",
            "cache_write_input_tokens",
            "cache_creation_input_tokens",
        ):
            bucket[key] += int(usage.get(key, 0) or 0)
        session_id = str(record.get("session_id", ""))
        sessions[session_id].append(record)

        request = record.get("request") or {}
        response = record.get("response") or {}
        # The turn this call produced is filed once, on its `trajectory`
        # record; the exchange record keeps the wire traffic and the metering.
        turn = turn_of.get(str(record.get("exchange_id", "")), {})
        history_delta = turn.get("history_delta") or []
        if not any(
            isinstance(event, dict) and event.get("kind") == "usage"
            for event in history_delta
        ):
            issues.append(
                f"line {record['_line_number']}: successful exchange is "
                "missing its usage history event"
            )
        if provider == "anthropic":
            for block in _all_dicts(request.get("messages", [])):
                if block.get("type") == "thinking" and not block.get(
                    "signature"
                ):
                    issues.append(
                        f"line {record['_line_number']}: replayed Claude "
                        "thinking block is missing signature"
                    )
                if (
                    block.get("type") == "redacted_thinking"
                    and not block.get("data")
                ):
                    issues.append(
                        f"line {record['_line_number']}: replayed Claude "
                        "redacted_thinking block is missing data"
                    )
            for block in _all_dicts(response.get("content", [])):
                if (
                    block.get("type") == "thinking"
                    and not block.get("signature")
                ):
                    issues.append(
                        f"line {record['_line_number']}: Claude response "
                        "thinking block is missing signature"
                    )
                if (
                    block.get("type") == "redacted_thinking"
                    and not block.get("data")
                ):
                    issues.append(
                        f"line {record['_line_number']}: Claude response "
                        "redacted_thinking block is missing data"
                    )
        elif provider == "gemini":
            for part in _all_dicts(request.get("contents", [])):
                if part.get("thought") is True and not part.get(
                    "thoughtSignature"
                ):
                    issues.append(
                        f"line {record['_line_number']}: replayed Gemini "
                        "thought Part is missing thoughtSignature"
                    )
            candidates = response.get("candidates")
            candidate = (
                candidates[0]
                if isinstance(candidates, list)
                and candidates
                and isinstance(candidates[0], dict)
                else {}
            )
            content = candidate.get("content") or {}
            parts = content.get("parts") or []
            function_parts = [
                part for part in parts
                if isinstance(part, dict)
                and isinstance(part.get("functionCall"), dict)
            ]
            if (
                function_parts
                and not function_parts[0].get("thoughtSignature")
            ):
                issues.append(
                    f"line {record['_line_number']}: Gemini response first "
                    "functionCall Part is missing thoughtSignature"
                )
        elif provider == "openai_responses":
            response_id = response.get("id")
            if response_id and record.get("response_id") != response_id:
                issues.append(
                    f"line {record['_line_number']}: recorded response_id "
                    "does not match raw response"
                )
            previous = record.get("previous_response_id")
            request_previous = request.get("previous_response_id")
            if previous != request_previous:
                issues.append(
                    f"line {record['_line_number']}: previous_response_id "
                    "was not sent exactly as recorded"
                )
            if previous and str(previous) not in response_ids:
                issues.append(
                    f"line {record['_line_number']}: previous_response_id "
                    "does not refer to a response in this trace"
                )
            if request.get("store") is not True:
                issues.append(
                    f"line {record['_line_number']}: Responses request did "
                    "not set store=true"
                )
            if response.get("status") != "completed":
                issues.append(
                    f"line {record['_line_number']}: Responses status is "
                    f"{response.get('status')!r}, not 'completed'"
                )

    # Verify exact same-session handoff while honoring explicit rollback
    # records emitted by AgentSession.restore_in_place().
    for session_id, timeline in session_timelines.items():
        timeline.sort(key=lambda item: int(item.get("sequence", 0)))
        anchor_initialized = False
        expected_provider = ""
        expected_response_id: str | None = None
        expected_signatures: list[str] = []
        expected_tool_calls: list[str] = []
        for record in timeline:
            if record.get("record_type") == "session_restore":
                anchor_initialized = True
                expected_provider = str(record.get("provider", ""))
                restored_id = record.get("restored_response_id")
                expected_response_id = (
                    str(restored_id) if restored_id else None
                )
                expected_signatures = [
                    str(value)
                    for value in (
                        record.get("continuation_signatures") or []
                    )
                    if value
                ]
                expected_tool_calls = _snapshot_tool_call_keys(
                    record.get("pending_tool_calls")
                )
                continue

            provider = str(record.get("provider", ""))
            request = record.get("request") or {}
            if anchor_initialized:
                if expected_provider and provider != expected_provider:
                    issues.append(
                        f"session {session_id}: provider changed from "
                        f"{expected_provider} to {provider}"
                    )
                if provider == "openai_responses":
                    if not record.get("response_id_continuation", True):
                        # This route carries no state server-side, so the
                        # guarantee to check is that the turn it cannot
                        # reference was replayed inline instead.
                        replayed = _responses_item_ids(request)
                        missing = _missing_values(
                            expected_tool_calls,
                            _provider_tool_result_keys(request, provider),
                        )
                        if expected_response_id and not replayed:
                            issues.append(
                                f"session {session_id}: Responses history was "
                                "neither continued nor replayed"
                            )
                        if missing:
                            issues.append(
                                f"session {session_id}: {len(missing)} tool "
                                "result(s) were not returned"
                            )
                    elif request.get(
                        "previous_response_id"
                    ) != expected_response_id:
                        issues.append(
                            f"session {session_id}: Responses continuation "
                            "chain is broken"
                        )
                elif provider in {"anthropic", "gemini"}:
                    history = (
                        request.get("messages", [])
                        if provider == "anthropic"
                        else request.get("contents", [])
                    )
                    replayed = _provider_signatures(history, provider)
                    missing = _missing_values(
                        expected_signatures, replayed
                    )
                    if missing:
                        label = (
                            "Claude thinking signature"
                            if provider == "anthropic"
                            else "Gemini thoughtSignature value"
                        )
                        issues.append(
                            f"session {session_id}: {len(missing)} "
                            f"{label}(s) were not replayed"
                        )

                if expected_tool_calls:
                    replayed_results = _provider_tool_result_keys(
                        request, provider
                    )
                    missing_results = _missing_values(
                        expected_tool_calls, replayed_results
                    )
                    if missing_results:
                        issues.append(
                            f"session {session_id}: {len(missing_results)} "
                            "tool result(s) were not returned"
                        )

            response = record.get("response") or {}
            anchor_initialized = True
            expected_provider = provider
            response_id = response.get(
                "id", response.get("responseId")
            )
            expected_response_id = (
                str(response_id) if response_id else None
            )
            expected_signatures = _provider_signatures(
                response, provider
            )
            expected_tool_calls = _provider_tool_call_keys(
                response, provider
            )

    for record in records:
        if record.get("record_type") != "session_fork":
            continue
        child_calls = sessions.get(str(record.get("session_id", "")), [])
        if not child_calls:
            continue
        first = min(
            child_calls, key=lambda item: int(item.get("sequence", 0))
        )
        provider = str(first.get("provider", ""))
        request = first.get("request") or {}
        if provider == "openai_responses":
            if request.get("previous_response_id") != record.get(
                "previous_response_id"
            ):
                issues.append(
                    f"session {record.get('session_id')}: fork lost its "
                    "Responses continuation ID"
                )
            continue
        expected = list(record.get("continuation_signatures") or [])
        if not expected:
            continue
        history = (
            request.get("messages", [])
            if provider == "anthropic"
            else request.get("contents", [])
        )
        replayed = _provider_signatures(history, provider)
        missing = _missing_values(expected, replayed)
        if missing:
            issues.append(
                f"session {record.get('session_id')}: fork lost "
                f"{len(missing)} continuation signature(s)"
            )

    for provider, bucket in by_provider.items():
        effective = bucket["input_tokens"]
        if provider == "anthropic":
            effective += (
                bucket["cache_read_input_tokens"]
                + bucket["cache_creation_input_tokens"]
            )
        bucket["effective_input_tokens"] = effective
        bucket["cache_hit_ratio"] = (
            bucket["cache_read_input_tokens"] / effective
            if effective
            else 0.0
        )

    return {
        "schema": "agent-trace-audit",
        "records": len(records),
        "api_exchanges": len(exchanges),
        "successful_exchanges": len(successful),
        "failed_attempts": len(exchanges) - len(successful),
        "trajectories": len(trajectories),
        "sessions": len(sessions),
        "providers": dict(by_provider),
        "issues": issues,
        "ok": not issues,
    }


def _all_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _all_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_dicts(child)


def _provider_signatures(value: Any, provider: str) -> list[str]:
    values: list[str] = []
    for item in _all_dicts(value):
        candidate = None
        if provider == "anthropic" and item.get("type") in {
            "thinking",
            "redacted_thinking",
        }:
            candidate = item.get("signature", item.get("data"))
        elif provider == "gemini":
            candidate = item.get("thoughtSignature")
        if isinstance(candidate, str) and candidate:
            values.append(candidate)
    return values


def _missing_values(
    expected: list[str],
    actual: list[str],
) -> list[str]:
    # Fork and restore records name the signatures they expect by fingerprint;
    # traces written before that carry the blobs themselves. Compare in whatever
    # form the expectation was written in.
    if any(str(value).startswith("sha256:") for value in expected):
        actual = [fingerprint(value) for value in actual]
    missing = Counter(expected) - Counter(actual)
    return list(missing.elements())


def _tool_key(call_id: Any, name: Any) -> str:
    if call_id:
        return f"id:{call_id}"
    return f"name:{name}"


def _snapshot_tool_call_keys(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        _tool_key(item.get("call_id"), item.get("name"))
        for item in value
        if isinstance(item, dict)
    ]


def _provider_tool_call_keys(value: Any, provider: str) -> list[str]:
    calls: list[str] = []
    for item in _all_dicts(value):
        if provider == "anthropic" and item.get("type") == "tool_use":
            calls.append(_tool_key(item.get("id"), item.get("name")))
        elif provider == "gemini":
            function_call = item.get("functionCall")
            if isinstance(function_call, dict):
                calls.append(_tool_key(
                    function_call.get("id") or item.get("id"),
                    function_call.get("name"),
                ))
        elif (
            provider == "openai_responses"
            and item.get("type") == "function_call"
        ):
            calls.append(_tool_key(
                item.get("call_id") or item.get("id"),
                item.get("name"),
            ))
    return calls


def _responses_item_ids(request: dict[str, Any]) -> list[str]:
    """Prior-turn items a Responses request replays in its own input.

    A route that cannot continue server-side has to carry the conversation
    itself, so the reasoning and function_call items it is answering travel
    inline. Their presence is what makes the turn self-contained.
    """

    return [
        str(item.get("id") or item.get("call_id") or item.get("type"))
        for item in _all_dicts(request.get("input", []))
        if item.get("type") in {"reasoning", "function_call"}
    ]


def _provider_tool_result_keys(
    request: dict[str, Any],
    provider: str,
) -> list[str]:
    results: list[str] = []
    if provider == "anthropic":
        values = request.get("messages", [])
    elif provider == "gemini":
        values = request.get("contents", [])
    else:
        values = request.get("input", [])

    for item in _all_dicts(values):
        if provider == "anthropic" and item.get("type") == "tool_result":
            results.append(_tool_key(item.get("tool_use_id"), None))
        elif provider == "gemini":
            function_response = item.get("functionResponse")
            if isinstance(function_response, dict):
                results.append(_tool_key(
                    function_response.get("id"),
                    function_response.get("name"),
                ))
        elif (
            provider == "openai_responses"
            and item.get("type") == "function_call_output"
        ):
            results.append(_tool_key(item.get("call_id"), None))
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when integrity issues are found",
    )
    args = parser.parse_args()
    records, parse_issues = _load(args.trace)
    report = audit(records)
    report["issues"] = parse_issues + report["issues"]
    report["ok"] = not report["issues"]
    json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
    print()
    return 1 if args.strict and not report["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
