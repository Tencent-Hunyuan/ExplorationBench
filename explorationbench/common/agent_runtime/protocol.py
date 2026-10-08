from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, runtime_checkable

from common.agent_client import AgentResponse, SessionState, ToolDefinition, ToolResult


@dataclass(slots=True)
class PhaseContext:
    """Audit context for one benchmark phase."""

    sandbox: str
    phase: str
    label: str = ""
    track: str = "controlled"
    framework: str = "baseline"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sandbox": self.sandbox,
            "phase": self.phase,
            "label": self.label,
            "track": self.track,
            "framework": self.framework,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ArtifactSnapshot:
    """Frozen framework state that may be read by closed-book branches."""

    artifact: dict[str, Any]
    digest: str
    mutable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact,
            "digest": self.digest,
            "mutable": self.mutable,
        }


@runtime_checkable
class AgentSessionLike(Protocol):
    session_id: str
    system: str | list[dict[str, Any]] | None
    tools: list[ToolDefinition]

    @property
    def history(self) -> list[dict[str, Any]]:
        ...

    @property
    def last_response(self) -> AgentResponse | None:
        ...

    @property
    def last_call_meta(self) -> dict[str, Any]:
        ...

    def send_user(
        self,
        content: Any,
        *,
        label: str = "",
        request_overrides: dict[str, Any] | None = None,
    ) -> AgentResponse:
        ...

    def submit_tool_results(
        self,
        results: Iterable[ToolResult | dict[str, Any]],
        *,
        label: str = "",
        request_overrides: dict[str, Any] | None = None,
    ) -> AgentResponse:
        ...

    def using_tools(
        self,
        tools: Iterable[ToolDefinition | dict[str, Any]] | None,
    ) -> AbstractContextManager["AgentSessionLike"]:
        ...

    def fork(
        self,
        *,
        session_id: str | None = None,
        tools: Iterable[ToolDefinition | dict[str, Any]] | None = None,
    ) -> "AgentSessionLike":
        ...

    def state(self) -> SessionState:
        ...

    def snapshot(self) -> Any:
        ...

    def restore_in_place(
        self,
        state: SessionState | dict[str, Any],
        *,
        preserve_usage: Any = None,
    ) -> None:
        ...


@runtime_checkable
class AgentRuntime(Protocol):
    """Phase-aware boundary between benchmark harnesses and agent systems."""

    framework: str
    track: str

    def create_session(
        self,
        *,
        system: str | list[dict[str, Any]] | None = None,
        tools: Iterable[ToolDefinition | dict[str, Any]] | None = None,
        session_id: str | None = None,
    ) -> AgentSessionLike:
        ...

    def restore_session(
        self,
        state: SessionState | dict[str, Any] | str,
    ) -> AgentSessionLike:
        ...

    def begin_phase(self, context: PhaseContext) -> None:
        ...

    def after_environment_feedback(
        self,
        context: PhaseContext,
        *,
        probe: dict[str, Any],
        feedback: Any,
    ) -> None:
        ...

    def freeze_artifact(self, context: PhaseContext) -> ArtifactSnapshot:
        ...

    def fork_closed_book(
        self,
        session: AgentSessionLike,
        *,
        context: PhaseContext,
    ) -> AgentSessionLike:
        ...

    def get_usage(self) -> dict[str, Any]:
        ...

    def snapshot(self) -> dict[str, Any]:
        ...

    def restore_runtime_state(self, state: dict[str, Any] | None) -> None:
        ...

    def close(self) -> None:
        ...
