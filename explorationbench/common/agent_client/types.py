from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable
from uuid import uuid4


JsonValue = (
    bool
    | int
    | float
    | str
    | list["JsonValue"]
    | dict[str, "JsonValue"]
    | None
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def json_copy(value: Any) -> Any:
    """Return a detached, JSON-compatible copy.

    Provider payloads must never be mutated after they have been recorded:
    Claude and Gemini signatures are bound to the exact content block/Part in
    which they were returned.
    """

    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError):
        return copy.deepcopy(value)


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OPENAI_RESPONSES = "openai_responses"
    LEGACY_CHAT = "legacy_chat"


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    strict: bool | None = None

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("tool name must be a non-empty string")
        if not isinstance(self.input_schema, dict):
            raise TypeError("tool input_schema must be a JSON-schema object")

    @classmethod
    def from_value(cls, value: "ToolDefinition | dict[str, Any]") -> "ToolDefinition":
        if isinstance(value, cls):
            return cls(
                name=value.name,
                description=value.description,
                input_schema=json_copy(value.input_schema),
                strict=value.strict,
            )
        if not isinstance(value, dict):
            raise TypeError("tools must contain ToolDefinition or dict values")

        # Accept the canonical form and the common OpenAI function wrapper.
        nested = value.get("function")
        raw = (
            nested
            if value.get("type") == "function"
            and isinstance(nested, dict)
            else value
        )
        if not isinstance(raw, dict):
            raise TypeError("invalid function tool definition")
        schema = raw.get("input_schema", raw.get("parameters"))
        return cls(
            name=(
                raw.get("name")
                if isinstance(raw.get("name"), str)
                else ""
            ),
            description=(
                raw.get("description")
                if isinstance(raw.get("description"), str)
                else ""
            ),
            input_schema=json_copy(
                schema or {"type": "object", "properties": {}}
            ),
            strict=raw.get("strict"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_tools(
    tools: Iterable[ToolDefinition | dict[str, Any]] | None,
) -> list[ToolDefinition]:
    seen: set[str] = set()
    result: list[ToolDefinition] = []
    for value in tools or ():
        tool = ToolDefinition.from_value(value)
        if tool.name in seen:
            raise ValueError(f"duplicate tool name: {tool.name}")
        seen.add(tool.name)
        result.append(tool)
    return result


@dataclass(slots=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.call_id:
            raise ValueError("tool call_id must be non-empty")
        if not self.name:
            raise ValueError("tool name must be non-empty")
        if not isinstance(self.arguments, dict):
            raise TypeError("tool arguments must be an object")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolResult:
    call_id: str
    output: Any
    name: str | None = None
    is_error: bool = False

    @classmethod
    def from_value(cls, value: "ToolResult | dict[str, Any]") -> "ToolResult":
        if isinstance(value, cls):
            return cls(
                call_id=value.call_id,
                output=json_copy(value.output),
                name=value.name,
                is_error=value.is_error,
            )
        if not isinstance(value, dict):
            raise TypeError("tool results must contain ToolResult or dict values")
        raw_call_id = value.get(
            "call_id", value.get("tool_use_id")
        )
        return cls(
            call_id=(
                str(raw_call_id) if raw_call_id is not None else ""
            ),
            output=json_copy(value.get("output", value.get("content"))),
            name=value.get("name"),
            is_error=bool(value.get("is_error", False)),
        )

    def __post_init__(self) -> None:
        if not self.call_id:
            raise ValueError("tool result call_id must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReasoningArtifact:
    kind: str
    raw: dict[str, Any]
    text: str | None = None
    signature: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class UsageRecord:
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    tool_prompt_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def effective_input_tokens(self) -> int:
        if self.provider == Provider.ANTHROPIC.value:
            # Anthropic reports uncached, cache-create and cache-read input as
            # disjoint counters.
            return (
                self.input_tokens
                + self.cache_creation_input_tokens
                + self.cache_read_input_tokens
            )
        # OpenAI and Gemini include cached tokens in the reported input total.
        return self.input_tokens

    @property
    def cache_hit_ratio(self) -> float:
        denom = self.effective_input_tokens
        return self.cache_read_input_tokens / denom if denom else 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["effective_input_tokens"] = self.effective_input_tokens
        data["cache_hit_ratio"] = self.cache_hit_ratio
        return data

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "UsageRecord":
        fields = {
            name
            for name in cls.__dataclass_fields__
        }
        return cls(**{k: json_copy(v) for k, v in value.items() if k in fields})

    def add(self, other: "UsageRecord") -> None:
        for name in (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "total_tokens",
            "cache_read_input_tokens",
            "cache_write_input_tokens",
            "cache_creation_input_tokens",
            "tool_prompt_tokens",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))


@dataclass(slots=True)
class HistoryEvent:
    kind: str
    payload: dict[str, Any]
    role: str | None = None
    provider_payload: Any = None
    exchange_id: str | None = None
    event_id: str = field(default_factory=lambda: new_id("evt"))
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "HistoryEvent":
        return cls(
            kind=str(value["kind"]),
            payload=json_copy(value.get("payload", {})),
            role=value.get("role"),
            provider_payload=json_copy(value.get("provider_payload")),
            exchange_id=value.get("exchange_id"),
            event_id=str(value.get("event_id") or new_id("evt")),
            created_at=str(value.get("created_at") or utc_now()),
        )


@dataclass(slots=True)
class ProviderResult:
    text: str
    tool_calls: list[ToolCall]
    reasoning: list[ReasoningArtifact]
    usage: UsageRecord
    raw_response: dict[str, Any]
    provider_history_delta: list[dict[str, Any]]
    history_events: list[HistoryEvent]
    response_id: str | None = None
    stop_reason: str | None = None


@dataclass(slots=True)
class AgentResponse:
    text: str
    tool_calls: list[ToolCall]
    reasoning: list[ReasoningArtifact]
    usage: UsageRecord
    raw_response: dict[str, Any]
    response_id: str | None
    previous_response_id: str | None
    exchange_id: str
    stop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "reasoning": [item.to_dict() for item in self.reasoning],
            "usage": self.usage.to_dict(),
            "raw_response": json_copy(self.raw_response),
            "response_id": self.response_id,
            "previous_response_id": self.previous_response_id,
            "exchange_id": self.exchange_id,
            "stop_reason": self.stop_reason,
        }


@dataclass(slots=True)
class SessionState:
    provider: str
    model: str
    system: str | list[dict[str, Any]] | None
    tools: list[dict[str, Any]]
    history: list[dict[str, Any]]
    provider_history: list[dict[str, Any]]
    last_response_id: str | None
    pending_tool_calls: list[dict[str, Any]]
    usage: dict[str, Any]
    config: dict[str, Any]
    session_id: str = field(default_factory=lambda: new_id("session"))
    parent_session_id: str | None = None
    forked_from_event_id: str | None = None
    schema_version: int = 2

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["messages_history"] = data.pop("history")
        return data

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SessionState":
        version = int(value.get("schema_version", 1))
        if version > 2:
            raise ValueError(
                f"unsupported session snapshot schema_version={version}"
            )
        return cls(
            provider=str(value["provider"]),
            model=str(value["model"]),
            system=json_copy(value.get("system")),
            tools=json_copy(value.get("tools", [])),
            history=json_copy(
                value.get(
                    "messages_history",
                    value.get("history", []),
                )
            ),
            provider_history=json_copy(value.get("provider_history", [])),
            last_response_id=value.get("last_response_id"),
            pending_tool_calls=json_copy(
                value.get("pending_tool_calls", [])
            ),
            usage=json_copy(value.get("usage", {})),
            config=json_copy(value.get("config", {})),
            session_id=str(value.get("session_id") or new_id("session")),
            parent_session_id=value.get("parent_session_id"),
            forked_from_event_id=value.get("forked_from_event_id"),
            schema_version=version,
        )
