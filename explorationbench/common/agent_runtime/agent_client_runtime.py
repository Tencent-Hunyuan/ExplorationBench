from __future__ import annotations

from typing import Any, Iterable

from common.agent_client import (
    AgentClient,
    AgentClientConfig,
    AgentSession,
    JsonlTraceStore,
    SessionState,
    ToolDefinition,
)
from common.agent_client.types import json_copy

from .closed_book import closed_book_fork, stable_digest
from .protocol import ArtifactSnapshot, PhaseContext
from .trace import append_runtime_event


class AgentClientRuntime:
    """Phase-aware wrapper around the existing provider-native AgentClient."""

    framework = "baseline"
    track = "controlled"

    def __init__(
        self,
        *,
        client: AgentClient,
        trace_store: JsonlTraceStore | None = None,
        framework: str | None = None,
        track: str | None = None,
        budget_profile: dict[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.trace_store = trace_store or client.trace_store
        if framework:
            self.framework = framework
        if track:
            self.track = track
        self.budget_profile = json_copy(budget_profile or {})
        self._phase: PhaseContext | None = None
        self._artifact: dict[str, Any] = {}

    @classmethod
    def from_config(
        cls,
        *,
        config: AgentClientConfig,
        trace_store: JsonlTraceStore,
        framework: str = "baseline",
        track: str = "controlled",
        budget_profile: dict[str, Any] | None = None,
    ) -> "AgentClientRuntime":
        return cls(
            client=AgentClient(config=config, trace_store=trace_store),
            trace_store=trace_store,
            framework=framework,
            track=track,
            budget_profile=budget_profile,
        )

    def create_session(
        self,
        *,
        system: str | list[dict[str, Any]] | None = None,
        tools: Iterable[ToolDefinition | dict[str, Any]] | None = None,
        session_id: str | None = None,
    ) -> AgentSession:
        session = self.client.create_session(
            system=system,
            tools=tools,
            session_id=session_id,
        )
        append_runtime_event(
            self.trace_store,
            event="runtime_session_created",
            framework=self.framework,
            track=self.track,
            session_id=session.session_id,
            payload={"budget_profile": self.budget_profile},
        )
        return session

    def restore_session(
        self,
        state: SessionState | dict[str, Any] | str,
    ) -> AgentSession:
        return self.client.restore_session(state)

    def begin_phase(self, context: PhaseContext) -> None:
        self._phase = context
        append_runtime_event(
            self.trace_store,
            event="phase_begin",
            framework=self.framework,
            track=self.track,
            payload=context.to_dict(),
        )

    def after_environment_feedback(
        self,
        context: PhaseContext,
        *,
        probe: dict[str, Any],
        feedback: Any,
    ) -> None:
        append_runtime_event(
            self.trace_store,
            event="environment_feedback_observed",
            framework=self.framework,
            track=self.track,
            payload={
                "context": context.to_dict(),
                "probe_digest": stable_digest(probe),
                "feedback_digest": stable_digest(feedback),
            },
        )

    def freeze_artifact(self, context: PhaseContext) -> ArtifactSnapshot:
        snapshot = ArtifactSnapshot(
            artifact=json_copy(self._artifact),
            digest=stable_digest(self._artifact),
            mutable=False,
        )
        append_runtime_event(
            self.trace_store,
            event="artifact_frozen",
            framework=self.framework,
            track=self.track,
            payload={"context": context.to_dict(), **snapshot.to_dict()},
        )
        return snapshot

    def fork_closed_book(
        self,
        session: AgentSession,
        *,
        context: PhaseContext,
    ) -> AgentSession:
        artifact = self.freeze_artifact(context)
        parent_digest = stable_digest(session.state().to_dict())
        child = closed_book_fork(session, artifact=artifact, context=context)
        append_runtime_event(
            self.trace_store,
            event="closed_book_fork",
            framework=self.framework,
            track=self.track,
            session_id=child.session_id,
            payload={
                "context": context.to_dict(),
                "parent_session_id": session.session_id,
                "parent_digest": parent_digest,
                "artifact_digest": artifact.digest,
            },
        )
        return child

    def get_usage(self) -> dict[str, Any]:
        return self.client.get_usage()

    def snapshot(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "track": self.track,
            "budget_profile": json_copy(self.budget_profile),
            "artifact": json_copy(self._artifact),
            "artifact_digest": stable_digest(self._artifact),
        }

    def restore_runtime_state(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        artifact = state.get("artifact")
        if isinstance(artifact, dict):
            self._artifact = json_copy(artifact)

    def close(self) -> None:
        self.client.close()
