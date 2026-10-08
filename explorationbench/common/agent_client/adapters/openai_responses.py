from __future__ import annotations

import json
import sys
from typing import Any

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
from .base import PreparedInput, ProviderAdapter, text_history_event


_MISSING = object()

_RESPONSE_SETTING_KEYS = (
    "background",
    # Vendor extensions to this protocol: an explicit prompt cache and a
    # thinking switch that is separate from the reasoning tier.
    "caching",
    "conversation",
    "include",
    "max_output_tokens",
    "max_tool_calls",
    "metadata",
    "modalities",
    "parallel_tool_calls",
    "prompt",
    "prompt_cache_key",
    "prompt_cache_retention",
    "safety_identifier",
    "service_tier",
    "temperature",
    "text",
    "thinking",
    "tool_choice",
    "top_logprobs",
    "top_p",
    "truncation",
    "user",
)


class OpenAIResponsesAdapter(ProviderAdapter):
    """Adapter for OpenAI's stateful Responses wire protocol."""

    def prepare_user(self, content: Any) -> PreparedInput:
        item = {
            "type": "message",
            "role": "user",
            "content": _input_content(content),
        }
        return PreparedInput(
            provider_items=[item],
            history_events=[
                text_history_event(
                    "user",
                    json_copy(content),
                    provider_payload=json_copy(item),
                )
            ],
        )

    def prepare_tool_results(
        self,
        results: list[ToolResult],
        *,
        call_names: dict[str, str],
    ) -> PreparedInput:
        items: list[dict[str, Any]] = []
        events: list[HistoryEvent] = []
        for result in results:
            item = {
                "type": "function_call_output",
                "call_id": result.call_id,
                "output": _tool_output(result.output),
            }
            items.append(item)

            payload = result.to_dict()
            resolved_name = result.name or call_names.get(result.call_id)
            if resolved_name:
                payload["name"] = resolved_name
            events.append(
                HistoryEvent(
                    kind="tool_result",
                    role="tool",
                    payload=payload,
                    provider_payload=json_copy(item),
                )
            )
        return PreparedInput(provider_items=items, history_events=events)

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
        # Not every upstream that speaks Responses honours the continuation.
        # One accepts previous_response_id and then rejects the tool output it
        # implies, answering "No tool call found for tool output with call_id",
        # because the turn holding that call was never retained. Replaying the
        # history inline is the same conversation, so routes that cannot
        # continue server-side opt out here rather than losing tool use.
        if not self.config.settings.get("response_id_continuation", True):
            last_response_id = None

        # Responses continuations are linked by previous_response_id, so
        # replaying provider_history alongside one would duplicate every
        # preceding turn. Without an id the history is the only context there
        # is: either this is the session's first call and the history is
        # empty, or a continuation was dropped because the gateway routed us
        # to a different Azure resource than the one holding the response.
        if last_response_id:
            conversation = json_copy(input_items)
        else:
            conversation = [
                _detached(item)
                for item in json_copy(provider_history) + json_copy(input_items)
            ]

        settings = json_copy(self.config.settings)
        settings.update(json_copy(request_overrides))
        extra_body = settings.pop("extra_body", {})
        if extra_body is None:
            extra_body = {}
        if not isinstance(extra_body, dict):
            raise TypeError("Responses extra_body setting must be an object")
        merged_settings = json_copy(extra_body)
        merged_settings.update(settings)
        settings = merged_settings

        if "max_tokens" in settings:
            settings.setdefault(
                "max_output_tokens", settings.pop("max_tokens")
            )
        if "max_completion_tokens" in settings:
            settings.setdefault(
                "max_output_tokens",
                settings.pop("max_completion_tokens"),
            )
        response_format = settings.pop("response_format", _MISSING)
        if response_format is not _MISSING and response_format is not None:
            text_config = settings.get("text")
            if text_config is None:
                text_config = {}
            if not isinstance(text_config, dict):
                raise TypeError("Responses text setting must be an object")
            text_config = json_copy(text_config)
            text_config["format"] = _response_format(response_format)
            settings["text"] = text_config
        include_encrypted = bool(
            settings.pop(
                "include_encrypted_reasoning",
                _is_reasoning_model(
                    str(self.config.wire_model or self.config.model)
                ),
            )
        )
        if include_encrypted:
            include = settings.get("include")
            if include is None:
                include = []
            if not isinstance(include, list):
                raise TypeError("Responses include setting must be a list")
            include = list(include)
            if "reasoning.encrypted_content" not in include:
                include.append("reasoning.encrypted_content")
            settings["include"] = include
        if settings.pop("stream", False):
            raise ValueError("streaming is not supported by AgentClient")
        settings.pop("stream_options", None)

        # Some upstreams read and write no prompt cache at all once
        # instructions are set, and the system prompt is usually the whole
        # reusable prefix. Lead the conversation with it instead. A
        # continuation inherits it along with the rest of the turn, so it only
        # belongs here when the history is being sent in full.
        system_in_input = bool(settings.pop("system_in_input", False))
        if system_in_input and system is not None and not last_response_id:
            conversation = [
                {"role": "system", "content": _instructions(system)},
                *conversation,
            ]

        request: dict[str, Any] = {
            "model": self.config.wire_model,
            "input": json_copy(conversation),
            "store": True,
        }
        if system is not None and not system_in_input:
            request["instructions"] = _instructions(system)
        if tools:
            request["tools"] = [_response_tool(tool) for tool in tools]
        if last_response_id:
            request["previous_response_id"] = last_response_id

        reasoning = _reasoning_settings(
            settings,
            model=str(self.config.wire_model or self.config.model),
        )
        if reasoning:
            request["reasoning"] = reasoning

        for key in _RESPONSE_SETTING_KEYS:
            if key in settings:
                request[key] = json_copy(settings[key])

        # These are protocol invariants, not provider-tunable settings.
        request["input"] = json_copy(conversation)
        request["store"] = True
        request["model"] = self.config.wire_model
        if last_response_id:
            request["previous_response_id"] = last_response_id
        else:
            request.pop("previous_response_id", None)
        return request

    def parse_response(
        self,
        raw: dict[str, Any],
        *,
        previous_response_id: str | None,
    ) -> ProviderResult:
        del previous_response_id

        raw_output = raw.get("output", [])
        if not isinstance(raw_output, list):
            raise ValueError("Responses output must be a list")

        output_items = [
            json_copy(item) for item in raw_output if isinstance(item, dict)
        ]
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        reasoning: list[ReasoningArtifact] = []
        events: list[HistoryEvent] = []

        for item in output_items:
            item_type = item.get("type")
            if item_type == "message":
                item_text = _message_output_text(item)
                text_parts.append(item_text)
                events.append(
                    text_history_event(
                        "assistant",
                        item_text,
                        provider_payload=json_copy(item),
                    )
                )
                continue

            if item_type == "reasoning":
                summary = _reasoning_summary(item.get("summary"))
                encrypted = item.get("encrypted_content")
                signature = (
                    encrypted
                    if isinstance(encrypted, str)
                    else None if encrypted is None else str(encrypted)
                )
                artifact = ReasoningArtifact(
                    kind="reasoning",
                    raw=json_copy(item),
                    text=summary or None,
                    signature=signature,
                )
                reasoning.append(artifact)
                events.append(
                    HistoryEvent(
                        kind="reasoning",
                        role="assistant",
                        payload=artifact.to_dict(),
                        provider_payload=json_copy(item),
                    )
                )
                continue

            if item_type == "function_call":
                call_id = str(item.get("call_id") or item.get("id") or "")
                name = str(item.get("name") or "")
                call = ToolCall(
                    call_id=call_id,
                    name=name,
                    arguments=_tool_arguments(
                        item.get("arguments"), call_id=call_id
                    ),
                    raw=json_copy(item),
                )
                tool_calls.append(call)
                events.append(
                    HistoryEvent(
                        kind="tool_call",
                        role="assistant",
                        payload=call.to_dict(),
                        provider_payload=json_copy(item),
                    )
                )

        status = raw.get("status")
        detail = raw.get("error") or raw.get("incomplete_details")
        # `incomplete` means the model ran out of output budget, usually with
        # the reasoning having spent it. Whatever it did manage to say is worth
        # keeping -- a truncated answer can still be right, and a turn with
        # nothing in it falls to the empty-completion retry, which asks again.
        # Treating it as a parse failure instead forfeits the question outright,
        # and on a graded run that is a wrong score rather than a slow one.
        if status not in ("completed", "incomplete"):
            raise ValueError(
                "Responses payload did not complete "
                f"(status={status!r}, detail={detail!r})"
            )
        usage = _usage_record(raw.get("usage"))
        response_id = raw.get("id")
        if not isinstance(response_id, str) or not response_id:
            raise ValueError("Responses payload is missing response id")
        raw_copy = json_copy(raw)
        if status == "incomplete":
            raw_copy["stop_details"] = json_copy(detail) if detail else {}
        return ProviderResult(
            text="".join(text_parts),
            tool_calls=tool_calls,
            reasoning=reasoning,
            usage=usage,
            raw_response=raw_copy,
            provider_history_delta=output_items,
            history_events=events,
            response_id=response_id,
            stop_reason=str(status) if status is not None else None,
        )


def _input_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return [_input_content_part(part) for part in content]
    return [_input_content_part(content)]


def _input_content_part(part: Any) -> dict[str, Any]:
    if isinstance(part, str):
        return {"type": "input_text", "text": part}
    if isinstance(part, dict):
        block = json_copy(part)
        block_type = block.get("type")
        if block_type == "text":
            block["type"] = "input_text"
        elif block_type is None and isinstance(block.get("text"), str):
            block["type"] = "input_text"
        return block
    return {"type": "input_text", "text": _json_text(part)}


# Server-side object ids on a stored item point at the Azure resource that
# created it, so replaying an item that still carries one fails the same way
# the dropped continuation did. Dropping the id turns the item into inline
# content; encrypted_content keeps the reasoning intact, and call_id is a
# pairing key between a call and its output rather than a stored object.
_STORED_OBJECT_KEYS = ("id",)


def _detached(item: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(item, dict):
        return item
    for key in _STORED_OBJECT_KEYS:
        item.pop(key, None)
    return item


def _instructions(system: str | list[dict[str, Any]]) -> str:
    if isinstance(system, str):
        return system
    parts: list[str] = []
    for block in system:
        if isinstance(block, str):
            parts.append(block)
            continue
        if isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
                continue
            content = block.get("content")
            if isinstance(content, str):
                parts.append(content)
    return "\n".join(parts) if parts else _json_text(system)


def _response_tool(tool: ToolDefinition) -> dict[str, Any]:
    value: dict[str, Any] = {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": json_copy(tool.input_schema),
    }
    if tool.strict is not None:
        value["strict"] = tool.strict
    return value


def _response_format(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("Responses response_format must be an object")
    result = json_copy(value)
    if result.get("type") == "json_schema":
        nested = result.pop("json_schema", None)
        if isinstance(nested, dict):
            result = {"type": "json_schema", **json_copy(nested)}
    return result


def _reasoning_settings(
    settings: dict[str, Any],
    *,
    model: str,
) -> dict[str, Any]:
    configured = settings.get("reasoning")
    if configured is None:
        reasoning: dict[str, Any] = {}
    elif isinstance(configured, dict):
        reasoning = json_copy(configured)
    else:
        raise TypeError("settings.reasoning must be an object or null")

    aliases = (
        ("effort", ("reasoning_effort", "effort")),
        ("summary", ("reasoning_summary", "summary")),
        ("mode", ("reasoning_mode", "mode")),
        ("context", ("reasoning_context",)),
    )
    for wire_key, setting_keys in aliases:
        value: Any = _MISSING
        for setting_key in setting_keys:
            if setting_key in settings:
                value = settings[setting_key]
                break
        if value is not _MISSING:
            if value is None:
                reasoning.pop(wire_key, None)
            else:
                reasoning[wire_key] = json_copy(value)

    reasoning_model = _is_reasoning_model(model)
    if (
        reasoning_model
        and "summary" not in reasoning
        and "reasoning_summary" not in settings
        and "summary" not in settings
    ):
        reasoning["summary"] = "auto"
    return reasoning


def _is_reasoning_model(model: str) -> bool:
    name = model.lower()
    return "gpt-5" in name or name.startswith(("o1", "o3", "o4"))


def _tool_output(output: Any) -> Any:
    if isinstance(output, str):
        return output
    if isinstance(output, list) and all(
        isinstance(part, dict)
        and part.get("type") in {"input_text", "input_image", "input_file"}
        for part in output
    ):
        return json_copy(output)
    return _json_text(output)


def _json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _message_output_text(item: dict[str, Any]) -> str:
    content = item.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "output_text":
            continue
        text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _reasoning_summary(summary: Any) -> str:
    if isinstance(summary, str):
        return summary
    if not isinstance(summary, list):
        return ""

    parts: list[str] = []
    for block in summary:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def _tool_arguments(value: Any, *, call_id: str) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return json_copy(value)
    if not isinstance(value, str):
        raise ValueError(
            f"tool arguments for {call_id or '<unknown>'} must be JSON"
        )
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        # A tool argument here usually carries a multi-line program, and some
        # upstreams emit the newlines inside the JSON string literally instead
        # of escaping them. That is malformed JSON but unambiguous, and the
        # strict parser is the only thing rejecting it, so retry permissively
        # before giving up. Structure is still enforced below: the result must
        # be an object, so a genuinely broken payload still raises.
        try:
            parsed = json.loads(value, strict=False)
        except json.JSONDecodeError:
            # Neither parser can recover an argument the upstream truncated
            # mid-string, which is how a long program arrives when the model
            # runs out of output budget composing it. Raising here killed
            # multi-hour runs over a single malformed call, so hand back no
            # arguments instead: the harness already treats a call carrying no
            # code as a tool error, reports that to the model and lets it try
            # again, and a locked round runs its own probe regardless of what
            # the model asked for.
            sys.stderr.write(
                f"[ToolArgs] {call_id or '<unknown>'}: upstream returned "
                f"unparseable arguments ({len(value)} chars); "
                "delivering an empty call\n"
            )
            return {}
    if not isinstance(parsed, dict):
        raise ValueError(
            f"tool arguments for {call_id or '<unknown>'} must be an object"
        )
    return parsed


def _usage_record(value: Any) -> UsageRecord:
    raw = value if isinstance(value, dict) else {}
    input_details = raw.get("input_tokens_details")
    if not isinstance(input_details, dict):
        input_details = {}
    output_details = raw.get("output_tokens_details")
    if not isinstance(output_details, dict):
        output_details = {}

    input_tokens = _token_count(raw.get("input_tokens"))
    output_tokens = _token_count(raw.get("output_tokens"))
    total_value = raw.get("total_tokens")
    total_tokens = (
        input_tokens + output_tokens
        if total_value is None
        else _token_count(total_value)
    )
    cache_write = input_details.get(
        "cache_write_tokens", raw.get("cache_write_tokens")
    )
    return UsageRecord(
        provider=Provider.OPENAI_RESPONSES.value,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=_token_count(
            output_details.get("reasoning_tokens")
        ),
        total_tokens=total_tokens,
        cache_read_input_tokens=_token_count(
            input_details.get("cached_tokens")
        ),
        cache_write_input_tokens=_token_count(cache_write),
        raw=json_copy(raw),
    )


def _token_count(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


__all__ = ["OpenAIResponsesAdapter"]
