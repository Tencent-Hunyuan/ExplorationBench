from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from common.agent_client import (
    AgentClient,
    AgentClientConfig,
    AgentSession,
    JsonlTraceStore,
    ToolDefinition,
)
from common.agent_client.types import json_copy
from common.agent_runtime import AgentClientRuntime, ArtifactSnapshot, PhaseContext
from common.agent_runtime.closed_book import stable_digest
from common.agent_runtime.trace import append_runtime_event


@dataclass(frozen=True, slots=True)
class FrameworkSpec:
    key: str
    display_name: str
    mechanism: str
    official_url: str
    default_track: str = "controlled"
    official_compatibility: str = "method-style"
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "display_name": self.display_name,
            "mechanism": self.mechanism,
            "official_url": self.official_url,
            "default_track": self.default_track,
            "official_compatibility": self.official_compatibility,
            "notes": list(self.notes),
        }


class ControlledFrameworkRuntime(AgentClientRuntime):
    """A controlled, episode-local scaffold adapter.

    The adapter records only evidence exposed by the benchmark harness and
    injects a compact frozen playbook into subsequent calls. It deliberately
    does not import official framework packages by default, because controlled
    runs must keep the benchmark phase machine, tools, and metering intact.
    """

    def __init__(
        self,
        *,
        spec: FrameworkSpec,
        client: AgentClient,
        trace_store: JsonlTraceStore,
        track: str,
        budget_profile: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            client=client,
            trace_store=trace_store,
            framework=spec.key,
            track=track,
            budget_profile=budget_profile,
        )
        self.spec = spec
        self._artifact = {
            "spec": spec.to_dict(),
            "lessons": [],
            "phase_count": 0,
        }

    def create_session(
        self,
        *,
        system: str | list[dict[str, Any]] | None = None,
        tools: Iterable[ToolDefinition | dict[str, Any]] | None = None,
        session_id: str | None = None,
    ) -> AgentSession:
        return super().create_session(
            system=self._augment_system(system),
            tools=tools,
            session_id=session_id,
        )

    def begin_phase(self, context: PhaseContext) -> None:
        self._artifact["phase_count"] = int(
            self._artifact.get("phase_count", 0) or 0
        ) + 1
        super().begin_phase(context)

    def after_environment_feedback(
        self,
        context: PhaseContext,
        *,
        probe: dict[str, Any],
        feedback: Any,
    ) -> None:
        lessons = self._artifact.setdefault("lessons", [])
        if isinstance(lessons, list):
            lessons.append({
                "context": context.to_dict(),
                "probe_digest": stable_digest(probe),
                "feedback_digest": stable_digest(feedback),
                "summary": self._summarize_feedback(probe, feedback),
            })
            del lessons[:-48]
        super().after_environment_feedback(context, probe=probe, feedback=feedback)

    def freeze_artifact(self, context: PhaseContext) -> ArtifactSnapshot:
        snapshot = super().freeze_artifact(context)
        append_runtime_event(
            self.trace_store,
            event="framework_artifact_snapshot",
            framework=self.framework,
            track=self.track,
            payload={
                "context": context.to_dict(),
                "framework_spec": self.spec.to_dict(),
                "artifact_digest": snapshot.digest,
                "lesson_count": len(self._artifact.get("lessons") or []),
            },
        )
        return snapshot

    def snapshot(self) -> dict[str, Any]:
        state = super().snapshot()
        state["framework_spec"] = self.spec.to_dict()
        return state

    def restore_runtime_state(self, state: dict[str, Any] | None) -> None:
        super().restore_runtime_state(state)
        if state and "framework_spec" in state:
            self._artifact.setdefault("spec", json_copy(state["framework_spec"]))

    def _augment_system(
        self,
        system: str | list[dict[str, Any]] | None,
    ) -> str | list[dict[str, Any]] | None:
        if isinstance(system, list):
            system = json_copy(system)
            system.append({"type": "text", "text": self._scaffold_instructions()})
            return system
        return (system or "") + "\n\n" + self._scaffold_instructions()

    def _scaffold_instructions(self) -> str:
        method_note = {
            "reflexion": (
                "After each visible feedback item, write a concise reflection: "
                "what hypothesis was tested, what failed, and one next probe."
            ),
            "ace": (
                "Maintain a structured playbook with sections for stable rules, "
                "counterexamples, unresolved uncertainties, and reusable solving "
                "recipes. Prefer incremental updates over rewriting everything."
            ),
            "evotest": (
                "Treat each explore loop as an Act-Evolve micro-iteration: the "
                "actor probes; the evolver revises the next-loop configuration, "
                "including probe priorities and tool-use routines."
            ),
            "gepa": (
                "Treat your rule table and probing policy as textual artifacts. "
                "Use trace-level feedback to diagnose failure causes and mutate "
                "only the component implicated by the feedback."
            ),
            "agent_factory": (
                "When a reusable procedure emerges, describe it as a named, "
                "auditable subagent with inputs, outputs, invariants, and failure "
                "cases. In the controlled track this is text-only and episode-local."
            ),
        }.get(self.spec.key, "Use the visible feedback to keep a compact hypothesis state.")
        return (
            f"[Framework Track: {self.spec.display_name}]\n"
            f"Mechanism: {self.spec.mechanism}.\n"
            f"{method_note}\n"
            "You may maintain an episode-local playbook of hypotheses, failed "
            "probes, and reusable strategies. Update it only from seed/explore "
            "environment feedback visible in this conversation. During milestone "
            "summaries and held-out tests, treat the playbook as frozen and do "
            "not ask for unavailable tools. No cross-run memory is available in "
            "the controlled track."
        )

    def _summarize_feedback(self, probe: dict[str, Any], feedback: Any) -> str:
        text = str(feedback)
        probe_keys = ",".join(sorted(str(k) for k in probe.keys()))[:80]
        return f"probe_keys={probe_keys}; feedback={text[:240]}"
