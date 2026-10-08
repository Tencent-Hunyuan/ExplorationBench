from __future__ import annotations

import json
from typing import Any

from .base import PreparedInput, ProviderAdapter, text_history_event
from ..types import (
    HistoryEvent,
    Provider,
    ProviderResult,
    ReasoningArtifact,
    ToolCall,
    ToolDefinition,
    ToolResult,
    UsageRecord,
    json_copy,
)


_DEFAULT_MAX_TOKENS = 4096
# Routing/auth knobs that must never reach the provider body.
_NON_BODY_SETTINGS = {
    "anthropic_version",
    "gateway_provider",
    "gateway_max_retry",
    "passthrough_route",
    "responses_route",
    "cache_breakpoints",
    "cache_task_id",
    "account_id",
}
_MISSING = object()


def _as_blocks(value: Any) -> list[dict[str, Any]] | None:
    """Content as a block list, or None if it cannot carry cache_control."""

    if isinstance(value, str):
        return [{"type": "text", "text": value}] if value else None
    if isinstance(value, list):
        blocks = [item for item in value if isinstance(item, dict)]
        return blocks or None
    return None


# Thinking blocks are signed verbatim and reject the extra key; the rest of the
# block types the protocol defines accept a breakpoint.
_CACHEABLE_BLOCKS = frozenset({
    "text", "tool_use", "tool_result", "image", "document", "search_result",
})


def _cacheable_block(blocks: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The last block in a turn that may carry a cache breakpoint."""

    for block in reversed(blocks):
        if block.get("type") in _CACHEABLE_BLOCKS:
            return block
    return None


def _mark_cache_prefix(
    request: dict[str, Any], *, history_length: int, ttl: str
) -> None:
    """Mark the reusable prefix of a request so the provider caches it.

    Two breakpoints earn their keep here. The system prompt carries the
    reference manual and never changes, and the conversation so far is shared
    by every later call -- including the hundreds of closed-book questions
    that all branch from the same explored history. Marking the end of the
    history means each call reads that prefix instead of re-paying for it.

    A ttl of "default" marks the prefix without naming one: the named TTLs are
    an Anthropic extension, and vendors that merely borrow this protocol take
    the bare ephemeral block.
    """

    control = {"type": "ephemeral"}
    if ttl and ttl != "default":
        control["ttl"] = ttl

    system_blocks = _as_blocks(request.get("system"))
    if system_blocks is not None:
        system_blocks[-1]["cache_control"] = dict(control)
        request["system"] = system_blocks

    if history_length:
        # The last turn the model has already seen; anything after it is new
        # input and would only invalidate the entry. Walk back from there to
        # the nearest block that may carry a breakpoint at all: a turn whose
        # content ends in thinking is a 400 if marked, and since a rejected
        # request is rejected again on every later call -- that turn stays in
        # the history -- one such turn would otherwise end the run.
        for index in range(history_length - 1, -1, -1):
            message = request["messages"][index]
            blocks = _as_blocks(message.get("content"))
            if blocks is None:
                continue
            anchor = _cacheable_block(blocks)
            if anchor is None:
                continue
            anchor["cache_control"] = dict(control)
            message["content"] = blocks
            break


def _token_count(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _content_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list) and all(
        isinstance(block, dict) for block in content
    ):
        return json_copy(content)
    raise TypeError("Anthropic user content must be str or a list of content blocks")


def _tool_result_content(output: Any) -> str | list[dict[str, Any]]:
    if isinstance(output, str):
        return output
    if isinstance(output, list) and all(
        isinstance(block, dict) for block in output
    ):
        return json_copy(output)
    try:
        return json.dumps(output, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(output)


def _tool_definition(tool: ToolDefinition) -> dict[str, Any]:
    value: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description,
        "input_schema": json_copy(tool.input_schema),
    }
    if tool.strict is not None:
        value["strict"] = tool.strict
    return value


def _request_settings(settings_value: dict[str, Any]) -> dict[str, Any]:
    settings = json_copy(settings_value)
    extra_body = settings.pop("extra_body", {})
    if extra_body is None:
        extra_body = {}
    if not isinstance(extra_body, dict):
        raise TypeError("Anthropic extra_body setting must be an object")
    merged = json_copy(extra_body)
    merged.update(settings)
    settings = merged

    output_config_value = settings.pop("output_config", {})
    if output_config_value is None:
        output_config_value = {}
    if not isinstance(output_config_value, dict):
        raise TypeError("Anthropic output_config setting must be an object")
    output_config: dict[str, Any] = json_copy(output_config_value)

    effort: Any = _MISSING
    for key in ("reasoning_effort", "effort"):
        if key in settings:
            effort = settings.pop(key)
            break
    if effort is not _MISSING:
        if effort is None:
            output_config.pop("effort", None)
        else:
            output_config["effort"] = json_copy(effort)

    response_format = settings.pop("response_format", _MISSING)
    if response_format is not _MISSING:
        if response_format is None:
            output_config.pop("format", None)
        else:
            output_config["format"] = json_copy(response_format)
    if output_config:
        settings["output_config"] = output_config

    tool_choice = settings.get("tool_choice")
    if isinstance(tool_choice, str):
        mapped = {
            "required": "any",
            "any": "any",
            "auto": "auto",
            "none": "none",
        }.get(tool_choice.lower(), tool_choice.lower())
        settings["tool_choice"] = {"type": mapped}

    for key in _NON_BODY_SETTINGS:
        settings.pop(key, None)
    return {
        key: json_copy(value)
        for key, value in settings.items()
        if value is not None
    }


class AnthropicMessagesAdapter(ProviderAdapter):
    """Adapter for Anthropic's native Messages REST protocol."""

    def prepare_user(self, content: Any) -> PreparedInput:
        blocks = _content_blocks(content)
        message = {"role": "user", "content": blocks}
        return PreparedInput(
            provider_items=[json_copy(message)],
            history_events=[
                text_history_event(
                    "user",
                    json_copy(content),
                    provider_payload=json_copy(message),
                )
            ],
        )

    def prepare_tool_results(
        self,
        results: list[ToolResult],
        *,
        call_names: dict[str, str],
    ) -> PreparedInput:
        blocks: list[dict[str, Any]] = []
        events: list[HistoryEvent] = []
        for result in results:
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": result.call_id,
                "content": _tool_result_content(result.output),
            }
            if result.is_error:
                block["is_error"] = True
            blocks.append(block)
            payload = result.to_dict()
            resolved_name = result.name or call_names.get(result.call_id)
            if resolved_name:
                payload["name"] = resolved_name
            events.append(
                HistoryEvent(
                    kind="tool_result",
                    role="tool",
                    payload=payload,
                    provider_payload=json_copy(block),
                )
            )

        message = {"role": "user", "content": blocks}
        return PreparedInput(
            provider_items=[json_copy(message)],
            history_events=events,
        )

    def build_request(
        self,
        *,
        system: str | list[dict[str, Any]] | None,
        tools: list[ToolDefinition],
        provider_history: list[dict[str, Any]],
        input_items: list[dict[str, Any]],
        last_response_id: str | None,
        request_overrides: dict[str, Any],
    ) -> dict[str, Any]:
        del last_response_id  # Messages continuation is the explicit history.

        combined_settings = json_copy(self.config.settings)
        combined_settings.update(json_copy(request_overrides))
        settings = _request_settings(combined_settings)

        # These fields are owned by the session and cannot be overridden by
        # arbitrary settings.
        settings.pop("model", None)
        settings.pop("messages", None)
        settings.pop("system", None)
        settings.pop("tools", None)

        max_tokens = settings.get("max_tokens")
        if max_tokens is None:
            settings["max_tokens"] = _DEFAULT_MAX_TOKENS

        request: dict[str, Any] = settings
        if self.config.anthropic_url_addressed:
            # These routes take the model from the URL path and reject a body
            # model; they require the anthropic_version there instead, and its
            # value differs per upstream. They also reject `stream`, which is
            # negotiated by the route rather than the body.
            request.pop("stream", None)
            request["anthropic_version"] = (
                self.config.passthrough_anthropic_version
            )
        else:
            request["model"] = str(
                self.config.wire_model or self.config.model
            )
        history = json_copy(provider_history)
        request["messages"] = history + json_copy(input_items)
        if system is not None:
            request["system"] = json_copy(system)
        if tools:
            request["tools"] = [_tool_definition(tool) for tool in tools]
        ttl = self._cache_breakpoints()
        if ttl:
            _mark_cache_prefix(
                request, history_length=len(history), ttl=ttl
            )
        return request

    def _cache_breakpoints(self) -> str | None:
        """The TTL to cache under, or None to leave caching off.

        Anthropic only reuses a prefix that was explicitly marked, so without
        this every call re-reads the whole conversation at full price.
        """

        value = self.config.settings.get("cache_breakpoints")
        if value in (None, False, ""):
            return None
        return "5m" if value is True else str(value)

    def parse_response(
        self,
        raw: dict[str, Any],
        *,
        previous_response_id: str | None,
    ) -> ProviderResult:
        del previous_response_id

        raw_content = raw.get("content")
        if not isinstance(raw_content, list):
            raise ValueError("Anthropic Messages response content must be a list")
        content = json_copy(raw_content)
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        reasoning: list[ReasoningArtifact] = []

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                value = block.get("text")
                if isinstance(value, str):
                    text_parts.append(value)
                continue
            if block_type == "tool_use":
                name = block.get("name")
                call_id = block.get("id")
                arguments = block.get("input", {})
                if not isinstance(arguments, dict):
                    raise ValueError("Anthropic tool_use input must be an object")
                tool_calls.append(
                    ToolCall(
                        call_id=str(call_id or ""),
                        name=str(name or ""),
                        arguments=json_copy(arguments),
                        raw=json_copy(block),
                    )
                )
                continue
            if block_type in {"thinking", "redacted_thinking"}:
                thought = block.get("thinking")
                signature = block.get("signature")
                if signature is None and block_type == "redacted_thinking":
                    signature = block.get("data")
                reasoning.append(
                    ReasoningArtifact(
                        kind=str(block_type),
                        raw=json_copy(block),
                        text=thought if isinstance(thought, str) else None,
                        signature=(
                            signature if isinstance(signature, str) else None
                        ),
                    )
                )

        assistant_message = {
            "role": "assistant",
            "content": json_copy(content),
        }
        text = "".join(text_parts)
        events: list[HistoryEvent] = [
            text_history_event(
                "assistant",
                json_copy(content),
                provider_payload=json_copy(assistant_message),
            )
        ]
        for artifact in reasoning:
            events.append(
                HistoryEvent(
                    kind="reasoning",
                    role="assistant",
                    payload=artifact.to_dict(),
                    provider_payload=json_copy(artifact.raw),
                )
            )
        for call in tool_calls:
            events.append(
                HistoryEvent(
                    kind="tool_call",
                    role="assistant",
                    payload=call.to_dict(),
                    provider_payload=json_copy(call.raw),
                )
            )
        usage_raw = raw.get("usage", {})
        usage_data = usage_raw if isinstance(usage_raw, dict) else {}
        input_tokens = _token_count(usage_data.get("input_tokens"))
        cache_creation = _token_count(
            usage_data.get("cache_creation_input_tokens")
        )
        cache_read = _token_count(
            usage_data.get("cache_read_input_tokens")
        )
        output_tokens = _token_count(usage_data.get("output_tokens"))
        usage = UsageRecord(
            provider=Provider.ANTHROPIC.value,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=(
                input_tokens + cache_creation + cache_read + output_tokens
            ),
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_creation,
            raw=json_copy(usage_data),
        )

        response_id = raw.get("id")
        stop_reason = raw.get("stop_reason")
        return ProviderResult(
            text=text,
            tool_calls=tool_calls,
            reasoning=reasoning,
            usage=usage,
            raw_response=json_copy(raw),
            provider_history_delta=[assistant_message],
            history_events=events,
            response_id=str(response_id) if response_id is not None else None,
            stop_reason=(
                str(stop_reason) if stop_reason is not None else None
            ),
        )


__all__ = ["AnthropicMessagesAdapter"]
