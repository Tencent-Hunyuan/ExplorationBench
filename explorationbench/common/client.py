"""Backward-compatible facade over the provider-native AgentClient.

New evaluation code should keep an explicit AgentSession.  ``chat()`` remains
for one-shot judges, probes and legacy harnesses; because a plain messages list
does not carry provider continuation state, it intentionally performs a full
history replay and cannot provide ``previous_response_id`` continuity.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from .agent_client import (
    AgentClient,
    AgentClientConfig,
    HistoryEvent,
    JsonlTraceStore,
    Provider,
    ToolDefinition,
    UsageLedger,
    is_retryable_exception,
)
from .agent_client.adapters.base import PreparedInput
from .agent_client.types import json_copy


REQUEST_TIMEOUT = int(os.environ.get("EVAL_REQUEST_TIMEOUT", "3600"))
MAX_RETRIES = int(os.environ.get("EVAL_MAX_RETRIES", "8"))
REASONING_BACK_TURNS = max(
    0, int(os.environ.get("EVAL_REASONING_BACK_TURNS", "2"))
)
REASONING_BACK_MAX_CHARS = max(
    0, int(os.environ.get("EVAL_REASONING_BACK_MAX_CHARS", "12000"))
)
_SEND_REASONING_BACK = os.environ.get(
    "EVAL_SEND_REASONING_BACK", ""
).lower() not in {"", "0", "false", "no"}

_local = threading.local()
_usage_ledger = UsageLedger()
_trace_lock = threading.Lock()
_trace_stores: dict[tuple[str | None, str | None], JsonlTraceStore] = {}


def _trace_path() -> str | None:
    configured = (
        os.environ.get("EVAL_AGENT_TRACE_PATH")
        or os.environ.get("EVAL_RAW_RESPONSES_PATH")
    )
    enabled = os.environ.get(
        "EVAL_SAVE_AGENT_TRACE",
        os.environ.get("EVAL_SAVE_RAW_RESPONSES", ""),
    ).lower() not in {"", "0", "false", "no"}
    if configured:
        return str(Path(configured).resolve())
    if enabled:
        return str(Path("agent_trace.jsonl").resolve())
    return None


def _snapshot_dir() -> str | None:
    value = os.environ.get("EVAL_AGENT_SNAPSHOT_DIR", "").strip()
    return str(Path(value).resolve()) if value else None


def _shared_trace_store() -> JsonlTraceStore:
    path = _trace_path()
    snapshots = _snapshot_dir()
    key = (path, snapshots)
    with _trace_lock:
        store = _trace_stores.get(key)
        if store is None:
            store = JsonlTraceStore(
                path,
                snapshot_dir=snapshots,
                fsync=os.environ.get(
                    "EVAL_AGENT_TRACE_FSYNC", ""
                ).lower() in {"1", "true", "yes"},
            )
            _trace_stores[key] = store
        return store


def create_agent_client(
    *,
    model: str,
    provider: Provider | str | None = None,
    tools: list[ToolDefinition | dict[str, Any]] | None = None,
    system: str | list[dict[str, Any]] | None = None,
    **kwargs: Any,
):
    """Convenience constructor returning ``(client, session)``."""

    config = AgentClientConfig(model=model, provider=provider, **kwargs)
    trace_store = (
        JsonlTraceStore(
            config.trace_path,
            snapshot_dir=config.snapshot_dir,
            fsync=config.trace_fsync,
        )
        if config.trace_path or config.snapshot_dir
        else _shared_trace_store()
    )
    client = AgentClient(
        config,
        trace_store=trace_store,
        usage_ledger=_usage_ledger,
    )
    return client, client.create_session(system=system, tools=tools)


def chat(
    messages: list[dict[str, Any]] | str,
    model: str = "api_azure_openai_gpt-5.2",
    timeout: float = REQUEST_TIMEOUT,
    max_retries: int = MAX_RETRIES,
    response_format: dict[str, Any] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    thinking: dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
    extra_body: dict[str, Any] | None = None,
    tools: list[ToolDefinition | dict[str, Any]] | None = None,
) -> str:
    """One-shot compatibility API.

    The complete plain-text history is replayed once. Provider-native
    signatures cannot be reconstructed from plain messages, so stateful agent
    evaluations must use AgentSession instead.
    """

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if not messages:
        raise ValueError("messages cannot be empty")
    _local.last_response = None
    _local.last_meta = {}
    _local.last_reasoning = ""

    system_parts = [
        message.get("content", "")
        for message in messages
        if message.get("role") in {"system", "developer"}
    ]
    system: str | None = "\n\n".join(
        str(value) for value in system_parts if value not in (None, "")
    ) or None
    conversation = [
        message for message in messages
        if message.get("role") not in {"system", "developer"}
    ]
    settings: dict[str, Any] = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "thinking": json_copy(thinking),
        "reasoning_effort": reasoning_effort,
        "response_format": json_copy(response_format),
        "extra_body": json_copy(extra_body or {}),
    }
    config = AgentClientConfig(
        model=model,
        timeout=timeout,
        max_retries=max_retries,
        settings={k: v for k, v in settings.items() if v is not None},
    )
    client = AgentClient(
        config,
        trace_store=_shared_trace_store(),
        usage_ledger=_usage_ledger,
    )
    session = client.create_session(system=system, tools=tools)
    provider_items, events = _plain_history(
        conversation, config.provider
    )
    replayed_reasoning = _reasoning_replay_indexes(
        conversation, config.provider
    )
    response = session._exchange(  # compatibility bridge by design
        PreparedInput(provider_items=provider_items, history_events=events),
        label="compat_chat",
    )
    _local.last_response = response
    _local.last_meta = session.last_call_meta
    _local.last_meta["cot_replay_turns"] = len(replayed_reasoning)
    _local.last_reasoning = "\n".join(
        artifact.text
        for artifact in response.reasoning
        if artifact.text
    )
    return response.text


def _plain_history(
    messages: list[dict[str, Any]],
    provider: Provider,
) -> tuple[list[dict[str, Any]], list[HistoryEvent]]:
    items: list[dict[str, Any]] = []
    events: list[HistoryEvent] = []
    replayed_reasoning = _reasoning_replay_indexes(messages, provider)
    for index, message in enumerate(messages):
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        if provider is Provider.ANTHROPIC:
            blocks = (
                json_copy(content)
                if isinstance(content, list)
                else [{"type": "text", "text": str(content)}]
            )
            item = {
                "role": "assistant" if role == "assistant" else "user",
                "content": blocks,
            }
        elif provider is Provider.GEMINI:
            parts = (
                json_copy(content)
                if isinstance(content, list)
                else [{"text": str(content)}]
            )
            item = {
                "role": "model" if role == "assistant" else "user",
                "parts": parts,
            }
        elif provider is Provider.LEGACY_CHAT:
            item = {
                key: json_copy(value)
                for key, value in message.items()
                if key in {
                    "role",
                    "content",
                    "name",
                    "tool_calls",
                    "tool_call_id",
                    "reasoning",
                    "reasoning_content",
                }
            }
            if index in replayed_reasoning:
                reasoning = str(message.get("reasoning_content", ""))
                if REASONING_BACK_MAX_CHARS:
                    reasoning = reasoning[-REASONING_BACK_MAX_CHARS:]
                item["content"] = (
                    f"<think>{reasoning}</think>\n{content}"
                )
            item.pop("reasoning_content", None)
        else:
            item = {"role": role, "content": json_copy(content)}
        items.append(item)
        events.append(HistoryEvent(
            kind="message",
            role=role,
            payload={"content": json_copy(content)},
            provider_payload=json_copy(item),
        ))
    return items, events


def _reasoning_replay_indexes(
    messages: list[dict[str, Any]],
    provider: Provider,
) -> set[int]:
    """Select bounded CoT turns for compatible stateless proxy replay."""

    if (
        not _SEND_REASONING_BACK
        or provider is not Provider.LEGACY_CHAT
        or REASONING_BACK_TURNS <= 0
    ):
        return set()
    indexes = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "assistant"
        and message.get("reasoning_content")
    ]
    return set(indexes[-REASONING_BACK_TURNS:])


def get_last_reasoning() -> str:
    return getattr(_local, "last_reasoning", "") or ""


def get_last_call_meta() -> dict[str, Any]:
    return dict(getattr(_local, "last_meta", {}) or {})


def get_token_usage() -> dict[str, Any]:
    return _usage_ledger.snapshot()


def reset_token_usage() -> None:
    _usage_ledger.reset()


def restore_token_usage(snapshot: dict[str, Any]) -> None:
    _usage_ledger.restore_legacy(snapshot)


def is_retryable_error(error: BaseException) -> bool:
    return is_retryable_exception(error)
