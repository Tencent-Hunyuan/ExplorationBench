from __future__ import annotations

import hashlib
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


_GENERATION_KEY_ALIASES = {
    "candidate_count": "candidateCount",
    "frequency_penalty": "frequencyPenalty",
    "image_config": "imageConfig",
    "max_output_tokens": "maxOutputTokens",
    "max_tokens": "maxOutputTokens",
    "media_resolution": "mediaResolution",
    "presence_penalty": "presencePenalty",
    "response_logprobs": "responseLogprobs",
    "response_mime_type": "responseMimeType",
    "response_modalities": "responseModalities",
    "response_schema": "responseSchema",
    "speech_config": "speechConfig",
    "stop_sequences": "stopSequences",
    "thinking_config": "thinkingConfig",
    "top_k": "topK",
    "top_p": "topP",
}
_TOP_LEVEL_KEY_ALIASES = {
    "cached_content": "cachedContent",
    "safety_settings": "safetySettings",
    "tool_config": "toolConfig",
}
_TOP_LEVEL_KEYS = {
    "cachedContent",
    "labels",
    "safetySettings",
    "toolConfig",
}
_RESERVED_SETTINGS = {
    "cache_task_id",
    "contents",
    "gateway_provider",
    "gemini_api",
    "model",
    "passthrough_route",
    "responses_route",
    "systemInstruction",
    "system_instruction",
    "tools",
}
_MISSING = object()


def _token_count(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}]
    if isinstance(content, list) and all(
        isinstance(part, dict) for part in content
    ):
        return json_copy(content)
    raise TypeError("Gemini content must be str or a list of Parts")


def _function_declaration(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": json_copy(tool.input_schema),
    }


def _function_response_payload(result: ToolResult) -> dict[str, Any]:
    output = json_copy(result.output)
    try:
        json.dumps(output, ensure_ascii=False)
    except (TypeError, ValueError):
        output = str(output)
    if result.is_error:
        return {"error": output}
    if isinstance(output, dict):
        return output
    return {"result": output}


def _stable_call_id(
    raw: dict[str, Any],
    part: dict[str, Any],
    *,
    part_index: int,
    previous_response_id: str | None,
) -> str:
    seed = {
        "response_id": raw.get("responseId", raw.get("response_id")),
        "previous_response_id": previous_response_id,
        "part_index": part_index,
        "part": part,
    }
    try:
        encoded = json.dumps(
            seed,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        encoded = repr(seed)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]
    return f"gemini_call_{digest}"


def _request_settings(
    settings_value: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    settings = json_copy(settings_value)
    extra_body = settings.pop("extra_body", {})
    if extra_body is None:
        extra_body = {}
    if not isinstance(extra_body, dict):
        raise TypeError("Gemini extra_body setting must be an object")
    merged = json_copy(extra_body)
    merged.update(settings)
    settings = merged

    explicit_generation = settings.pop(
        "generationConfig",
        settings.pop("generation_config", {}),
    )
    if explicit_generation is None:
        explicit_generation = {}
    if not isinstance(explicit_generation, dict):
        raise TypeError("Gemini generationConfig setting must be an object")
    generation_config: dict[str, Any] = json_copy(explicit_generation)
    top_level: dict[str, Any] = {}

    for source, target in _TOP_LEVEL_KEY_ALIASES.items():
        if source in settings:
            top_level[target] = settings.pop(source)
    for key in tuple(_TOP_LEVEL_KEYS):
        if key in settings:
            top_level[key] = settings.pop(key)
    for key in _RESERVED_SETTINGS:
        settings.pop(key, None)

    effort = settings.pop(
        "reasoning_effort",
        settings.pop("effort", None),
    )
    if effort is not None:
        thinking_config = generation_config.get("thinkingConfig")
        if thinking_config is None:
            thinking_config = {}
            generation_config["thinkingConfig"] = thinking_config
        if not isinstance(thinking_config, dict):
            raise TypeError("Gemini thinkingConfig setting must be an object")
        thinking_config.setdefault("thinkingLevel", effort)

    thinking = settings.pop("thinking", None)
    if thinking is not None:
        generation_config["thinkingConfig"] = json_copy(thinking)

    response_format = settings.pop("response_format", _MISSING)
    if response_format is not _MISSING and response_format is not None:
        if not isinstance(response_format, dict):
            raise TypeError("Gemini response_format setting must be an object")
        response_type = response_format.get("type")
        if response_type in {"json_object", "json_schema"}:
            generation_config["responseMimeType"] = "application/json"
        json_schema = response_format.get("json_schema")
        if isinstance(json_schema, dict):
            schema = json_schema.get("schema")
            if isinstance(schema, dict):
                generation_config["responseJsonSchema"] = json_copy(schema)

    tool_choice = settings.pop("tool_choice", _MISSING)
    if tool_choice is not _MISSING and tool_choice is not None:
        existing = top_level.get("toolConfig", {})
        if not isinstance(existing, dict):
            raise TypeError("Gemini toolConfig setting must be an object")
        tool_config = json_copy(existing)
        if isinstance(tool_choice, str):
            mode = {
                "required": "ANY",
                "any": "ANY",
                "auto": "AUTO",
                "none": "NONE",
            }.get(tool_choice.lower(), tool_choice.upper())
            function_config = tool_config.get(
                "functionCallingConfig", {}
            )
            if not isinstance(function_config, dict):
                raise TypeError(
                    "Gemini functionCallingConfig must be an object"
                )
            function_config = json_copy(function_config)
            function_config["mode"] = mode
            tool_config["functionCallingConfig"] = function_config
        elif isinstance(tool_choice, dict):
            tool_config.update(json_copy(tool_choice))
        else:
            raise TypeError("Gemini tool_choice must be a string or object")
        top_level["toolConfig"] = tool_config

    for key, value in settings.items():
        if value is not None:
            generation_config[_GENERATION_KEY_ALIASES.get(key, key)] = value
    return generation_config, top_level


class GeminiGenerateContentAdapter(ProviderAdapter):
    """Adapter for Gemini's native generateContent REST protocol."""

    def prepare_user(self, content: Any) -> PreparedInput:
        parts = _parts(content)
        message = {"role": "user", "parts": parts}
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
        parts: list[dict[str, Any]] = []
        events: list[HistoryEvent] = []
        for result in results:
            name = result.name or call_names.get(result.call_id)
            if not name:
                raise ValueError(
                    f"missing function name for Gemini call {result.call_id!r}"
                )
            function_response = {
                "name": name,
                "response": _function_response_payload(result),
            }
            if not result.call_id.startswith("gemini_call_"):
                function_response["id"] = result.call_id
            part = {"functionResponse": function_response}
            parts.append(part)
            payload = result.to_dict()
            payload["name"] = name
            events.append(
                HistoryEvent(
                    kind="tool_result",
                    role="tool",
                    payload=payload,
                    provider_payload=json_copy(part),
                )
            )

        message = {"role": "user", "parts": parts}
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
        del last_response_id  # generateContent continuation is explicit history.

        combined_settings = json_copy(self.config.settings)
        combined_settings.update(json_copy(request_overrides))
        generation_config, top_level = _request_settings(combined_settings)
        request: dict[str, Any] = {
            **top_level,
            "contents": json_copy(provider_history) + json_copy(input_items),
            "generationConfig": generation_config,
        }
        if system is not None:
            request["systemInstruction"] = {"parts": _parts(system)}
        if tools:
            request["tools"] = [{
                "functionDeclarations": [
                    _function_declaration(tool) for tool in tools
                ]
            }]
        return request

    def parse_response(
        self,
        raw: dict[str, Any],
        *,
        previous_response_id: str | None,
    ) -> ProviderResult:
        candidates_value = raw.get("candidates", [])
        candidates = (
            candidates_value if isinstance(candidates_value, list) else []
        )
        if not candidates or not isinstance(candidates[0], dict):
            raise ValueError("Gemini generateContent response has no candidate")
        candidate = candidates[0]
        raw_content = candidate.get("content")
        # A candidate that spent its budget thinking comes back with a
        # finishReason and nothing to say -- no content, or content with no
        # parts. That is an answer we did not get, not a malformed response, so
        # it is handed on as an empty turn for the caller to ask again rather
        # than raised as a permanent parse failure that forfeits the question.
        content = json_copy(raw_content) if isinstance(raw_content, dict) else {}
        content.setdefault("role", "model")
        parts_value = content.get("parts")
        parts = parts_value if isinstance(parts_value, list) else []
        content["parts"] = parts

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        reasoning: list[ReasoningArtifact] = []
        for part_index, part in enumerate(parts):
            if not isinstance(part, dict):
                continue

            is_thought = part.get("thought") is True
            part_text = part.get("text")
            signature = part.get("thoughtSignature")
            if is_thought:
                reasoning.append(
                    ReasoningArtifact(
                        kind="thought",
                        raw=json_copy(part),
                        text=part_text if isinstance(part_text, str) else None,
                        signature=(
                            signature if isinstance(signature, str) else None
                        ),
                    )
                )
            else:
                if isinstance(part_text, str):
                    text_parts.append(part_text)
                if isinstance(signature, str):
                    # Gemini 3 may attach reasoning state to any Part,
                    # including visible text and functionCall Parts.
                    reasoning.append(
                        ReasoningArtifact(
                            kind="thought_signature",
                            raw=json_copy(part),
                            text=None,
                            signature=signature,
                        )
                    )

            function_call = part.get("functionCall")
            if not isinstance(function_call, dict):
                continue
            name = function_call.get("name")
            native_call_id = function_call.get("id") or part.get("id")
            call_id = (
                str(native_call_id)
                if native_call_id
                else _stable_call_id(
                    raw,
                    part,
                    part_index=part_index,
                    previous_response_id=previous_response_id,
                )
            )
            arguments = function_call.get("args", {})
            if not isinstance(arguments, dict):
                raise ValueError("Gemini functionCall args must be an object")
            tool_calls.append(
                ToolCall(
                    call_id=call_id,
                    name=str(name or ""),
                    arguments=json_copy(arguments),
                    raw=json_copy(part),
                )
            )

        usage_value = raw.get("usageMetadata", {})
        usage_data = usage_value if isinstance(usage_value, dict) else {}
        input_tokens = _token_count(usage_data.get("promptTokenCount"))
        output_tokens = _token_count(usage_data.get("candidatesTokenCount"))
        reasoning_tokens = _token_count(
            usage_data.get("thoughtsTokenCount")
        )
        reported_total = usage_data.get("totalTokenCount")
        total_tokens = (
            _token_count(reported_total)
            if reported_total is not None
            else input_tokens + output_tokens + reasoning_tokens
        )
        usage = UsageRecord(
            provider=Provider.GEMINI.value,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens,
            cache_read_input_tokens=_token_count(
                usage_data.get("cachedContentTokenCount")
            ),
            tool_prompt_tokens=_token_count(
                usage_data.get("toolUsePromptTokenCount")
            ),
            raw=json_copy(usage_data),
        )

        text = "".join(text_parts)
        events: list[HistoryEvent] = [
            text_history_event(
                "assistant",
                json_copy(parts),
                provider_payload=json_copy(content),
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
        response_id = raw.get("responseId", raw.get("response_id"))
        finish_reason = candidate.get("finishReason")
        return ProviderResult(
            text=text,
            tool_calls=tool_calls,
            reasoning=reasoning,
            usage=usage,
            raw_response=json_copy(raw),
            provider_history_delta=[content],
            history_events=events,
            response_id=str(response_id) if response_id is not None else None,
            stop_reason=(
                str(finish_reason) if finish_reason is not None else None
            ),
        )


__all__ = ["GeminiGenerateContentAdapter"]
