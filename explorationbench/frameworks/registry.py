from __future__ import annotations

from typing import Any

from common.agent_client import AgentClient, AgentClientConfig, JsonlTraceStore
from common.agent_runtime import AgentClientRuntime, BudgetProfile

from .base import ControlledFrameworkRuntime, FrameworkSpec


FRAMEWORK_SPECS: dict[str, FrameworkSpec] = {
    "baseline": FrameworkSpec(
        key="baseline",
        display_name="AW-Base",
        mechanism="single agent with provider-native context",
        official_url="local:dev/common/agent_client",
        official_compatibility="native",
        notes=("Anchor system; no scaffold beyond the benchmark prompt.",),
    ),
    "reflexion": FrameworkSpec(
        key="reflexion",
        display_name="Reflexion",
        mechanism="verbal reflection over visible feedback",
        official_url="https://github.com/noahshinn/reflexion",
        notes=("Classic non-frontier control, not counted among the four systems.",),
    ),
    "ace": FrameworkSpec(
        key="ace",
        display_name="ACE",
        mechanism="structured evolving playbook with generator/reflector/curator roles",
        official_url="https://github.com/ace-agent/ace",
        notes=("Controlled adapter uses ACE-style episode-local playbook updates.",),
    ),
    "evotest": FrameworkSpec(
        key="evotest",
        display_name="EvoTest",
        mechanism="actor-evolver test-time configuration updates",
        official_url="https://github.com/yf-he/EvoTest",
        notes=("Official implementation is Jericho-oriented; benchmark adapter is method-style.",),
    ),
    "gepa": FrameworkSpec(
        key="gepa",
        display_name="GEPA",
        mechanism="reflective text artifact evolution with Pareto-style lineage",
        official_url="https://github.com/gepa-ai/gepa",
        notes=("Controlled adapter forbids hidden-rule or held-out-score metrics.",),
    ),
    "agent_factory": FrameworkSpec(
        key="agent_factory",
        display_name="AgentFactory",
        mechanism="episode-local executable subagent accumulation and reuse",
        official_url="https://github.com/zzatpku/AgentFactory",
        notes=("Native executable skills require an external isolation worker.",),
    ),
}


def list_frameworks() -> list[dict[str, Any]]:
    return [spec.to_dict() for spec in FRAMEWORK_SPECS.values()]


def normalize_framework_key(value: str | None) -> str:
    key = (value or "baseline").strip().lower().replace("-", "_")
    aliases = {
        "aw_base": "baseline",
        "agentclient": "baseline",
        "agent_client": "baseline",
        "agentfactory": "agent_factory",
    }
    return aliases.get(key, key)


def build_runtime(
    *,
    framework: str | None,
    track: str | None,
    config: AgentClientConfig,
    trace_store: JsonlTraceStore,
    budget_profile: BudgetProfile | None = None,
) -> AgentClientRuntime:
    key = normalize_framework_key(framework)
    run_track = (track or "controlled").strip().lower()
    if run_track not in {"controlled", "native", "open"}:
        raise ValueError("track must be one of: controlled, native, open")
    spec = FRAMEWORK_SPECS.get(key)
    if spec is None:
        supported = ", ".join(sorted(FRAMEWORK_SPECS))
        raise ValueError(f"unknown framework {framework!r}; supported: {supported}")
    client = AgentClient(config=config, trace_store=trace_store)
    profile = (budget_profile or BudgetProfile.from_name("c1")).to_dict()
    if key == "baseline":
        return AgentClientRuntime(
            client=client,
            trace_store=trace_store,
            framework=key,
            track=run_track,
            budget_profile=profile,
        )
    return ControlledFrameworkRuntime(
        spec=spec,
        client=client,
        trace_store=trace_store,
        track=run_track,
        budget_profile=profile,
    )
