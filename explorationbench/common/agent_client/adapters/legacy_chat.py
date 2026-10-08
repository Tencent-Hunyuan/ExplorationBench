from __future__ import annotations

import json
import re
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
from .base import (
    MalformedTurnError,
    PreparedInput,
    ProviderAdapter,
    text_history_event,
)


class LegacyChatAdapter(ProviderAdapter):
    """Adapter for OpenAI-compatible Chat Completions endpoints."""

    def prepare_user(self, content: Any) -> PreparedInput:
        message = {
            "role": "user",
            "content": _chat_content(content),
        }
        return PreparedInput(
            provider_items=[message],
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
        messages: list[dict[str, Any]] = []
        events: list[HistoryEvent] = []
        for result in results:
            message = {
                "role": "tool",
                "tool_call_id": result.call_id,
                "content": _tool_content(result.output),
            }
            messages.append(message)

            payload = result.to_dict()
            resolved_name = result.name or call_names.get(result.call_id)
            if resolved_name:
                payload["name"] = resolved_name
            events.append(
                HistoryEvent(
                    kind="tool_result",
                    role="tool",
                    payload=payload,
                    provider_payload=json_copy(message),
                )
            )
        return PreparedInput(provider_items=messages, history_events=events)

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
        del last_response_id

        messages: list[dict[str, Any]] = []
        if system is not None:
            messages.append(
                {
                    "role": "system",
                    "content": json_copy(system),
                }
            )
        messages.extend(json_copy(provider_history))
        messages.extend(json_copy(input_items))

        settings = json_copy(self.config.settings)
        settings.update(json_copy(request_overrides))
        extra_body = settings.pop("extra_body", {})
        if extra_body is None:
            extra_body = {}
        if not isinstance(extra_body, dict):
            raise TypeError("Chat extra_body setting must be an object")
        merged_settings = json_copy(extra_body)
        merged_settings.update(settings)
        settings = merged_settings
        if settings.pop("stream", False):
            raise ValueError("streaming is not supported by AgentClient")
        settings.pop("stream_options", None)
        request: dict[str, Any] = {
            "model": self.config.wire_model,
            "messages": messages,
        }
        for key, value in settings.items():
            if key not in {"model", "messages", "tools"} and value is not None:
                request[key] = json_copy(value)
        if tools:
            request["tools"] = [_chat_tool(tool) for tool in tools]

        # History and tool definitions come from the session, never settings.
        request["model"] = self.config.wire_model
        request["messages"] = messages
        return request

    def parse_response(
        self,
        raw: dict[str, Any],
        *,
        previous_response_id: str | None,
    ) -> ProviderResult:
        del previous_response_id

        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("Chat Completions response has no choices")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise ValueError("Chat Completions choice must be an object")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("Chat Completions choice has no assistant message")

        assistant = json_copy(message)
        content = assistant.get("content")
        text = _content_text(content)
        reasoning = _reasoning_artifacts(assistant)
        text, inline = _split_inline_reasoning(text)
        if inline is not None:
            reasoning.append(
                ReasoningArtifact(
                    kind="inline_reasoning",
                    raw={"text": inline},
                    text=inline,
                )
            )
        tool_calls = _parse_tool_calls(assistant.get("tool_calls"))

        events: list[HistoryEvent] = [
            text_history_event(
                "assistant",
                json_copy(content),
                provider_payload=json_copy(assistant),
            )
        ]
        for artifact in reasoning:
            events.append(
                HistoryEvent(
                    kind="reasoning",
                    role="assistant",
                    payload=artifact.to_dict(),
                    provider_payload=json_copy(assistant),
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

        response_id = raw.get("id")
        finish_reason = choice.get("finish_reason")
        return ProviderResult(
            text=text,
            tool_calls=tool_calls,
            reasoning=reasoning,
            usage=_usage_record(raw.get("usage")),
            raw_response=json_copy(raw),
            provider_history_delta=[assistant],
            history_events=events,
            response_id=(
                str(response_id) if response_id is not None else None
            ),
            stop_reason=(
                str(finish_reason) if finish_reason is not None else None
            ),
        )


def _chat_content(content: Any) -> Any:
    if isinstance(content, (str, list)) or content is None:
        return json_copy(content)
    return _json_text(content)


def _tool_content(output: Any) -> Any:
    if isinstance(output, str):
        return output
    if isinstance(output, list) and all(
        isinstance(part, dict) and part.get("type") == "text"
        for part in output
    ):
        return json_copy(output)
    return _json_text(output)


def _chat_tool(tool: ToolDefinition) -> dict[str, Any]:
    function: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description,
        "parameters": json_copy(tool.input_schema),
    }
    if tool.strict is not None:
        function["strict"] = tool.strict
    return {"type": "function", "function": function}


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        for key in ("text", "content"):
            value = content.get(key)
            if isinstance(value, str):
                return value
        return ""
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
            elif isinstance(block.get("content"), str):
                parts.append(block["content"])
    return "".join(parts)


# One self-hosted deployment ends its reasoning with a tagged marker and
# never opens it, so the thinking arrives inside `content` ahead of the
# answer. Splitting it out is not cosmetic: a graded answer is read from the
# first code block, and the thinking contains drafts of that code, so leaving
# it joined marks a run wrong on the model's scratch work.
_INLINE_REASONING_END = re.compile(r"</think(?::[0-9a-f]+)?>")


def _split_inline_reasoning(text: str) -> tuple[str, str | None]:
    """Answer text and the reasoning that preceded it, if any was inline.

    The last marker is the one that closes the thinking; an opening tag is
    honoured when present but is not required.
    """

    end = None
    for end in _INLINE_REASONING_END.finditer(text):
        pass
    if end is None:
        return text, None
    thinking = text[: end.start()]
    opened = thinking.rfind("<think")
    if opened != -1:
        thinking = thinking[opened:]
    return text[end.end():].lstrip(), thinking.strip() or None


def _reasoning_artifacts(
    message: dict[str, Any],
) -> list[ReasoningArtifact]:
    artifacts: list[ReasoningArtifact] = []
    for field_name in ("reasoning_content", "reasoning"):
        if field_name not in message or message[field_name] is None:
            continue
        value = message[field_name]
        artifacts.append(
            ReasoningArtifact(
                kind=field_name,
                raw={field_name: json_copy(value)},
                text=_content_text(value) or None,
                signature=_reasoning_signature(value),
            )
        )
    return artifacts


def _reasoning_signature(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in ("signature", "encrypted_content"):
            signature = value.get(key)
            if isinstance(signature, str):
                return signature
    if isinstance(value, list):
        for part in value:
            signature = _reasoning_signature(part)
            if signature is not None:
                return signature
    return None


def _parse_tool_calls(value: Any) -> list[ToolCall]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("assistant tool_calls must be a list")

    calls: list[ToolCall] = []
    for raw_call in value:
        if not isinstance(raw_call, dict):
            raise ValueError("assistant tool call must be an object")
        function = raw_call.get("function")
        if not isinstance(function, dict):
            raise ValueError("assistant tool call has no function object")
        call_id = str(raw_call.get("id") or raw_call.get("call_id") or "")
        name = str(function.get("name") or "")
        calls.append(
            ToolCall(
                call_id=call_id,
                name=name,
                arguments=_tool_arguments(
                    function.get("arguments"), call_id=call_id
                ),
                raw=json_copy(raw_call),
            )
        )
    return calls


def _tool_arguments(value: Any, *, call_id: str) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return json_copy(value)
    if not isinstance(value, str):
        raise MalformedTurnError(
            f"tool arguments for {call_id or '<unknown>'} must be JSON"
        )
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise MalformedTurnError(
            f"invalid tool arguments for {call_id or '<unknown>'}"
        ) from error
    if not isinstance(parsed, dict):
        raise MalformedTurnError(
            f"tool arguments for {call_id or '<unknown>'} must be an object"
        )
    return parsed


def _usage_record(value: Any) -> UsageRecord:
    raw = value if isinstance(value, dict) else {}
    prompt_details = raw.get("prompt_tokens_details")
    if not isinstance(prompt_details, dict):
        prompt_details = {}
    completion_details = raw.get("completion_tokens_details")
    if not isinstance(completion_details, dict):
        completion_details = {}

    input_tokens = _token_count(raw.get("prompt_tokens"))
    output_tokens = _token_count(raw.get("completion_tokens"))
    total_value = raw.get("total_tokens")
    total_tokens = (
        input_tokens + output_tokens
        if total_value is None
        else _token_count(total_value)
    )
    cached = prompt_details.get(
        "cached_tokens",
        raw.get("cached_tokens", raw.get("cache_read_input_tokens")),
    )
    cache_write = prompt_details.get(
        "cache_write_tokens", raw.get("cache_write_tokens")
    )
    reasoning = completion_details.get(
        "reasoning_tokens", raw.get("reasoning_tokens")
    )
    return UsageRecord(
        provider=Provider.LEGACY_CHAT.value,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=_token_count(reasoning),
        total_tokens=total_tokens,
        cache_read_input_tokens=_token_count(cached),
        cache_write_input_tokens=_token_count(cache_write),
        raw=json_copy(raw),
    )


def _json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _token_count(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


__all__ = ["LegacyChatAdapter"]
