#!/usr/bin/env python3
"""Turn an agent trace into a single self-contained HTML replay.

Reads the append-only JSONL written by ``JsonlTraceStore`` and renders one
page per run: every turn in order, what the model saw, what it thought, what
it answered, which tools it called and what each call cost.

    python3 dev/scripts/render_interaction.py logs/protocol_tests/foo.jsonl

Reasoning is collapsed by default and signatures are reduced to a short
fingerprint; the raw trace keeps the full values for auditing.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Turn:
    sequence: int
    session_id: str
    parent_session_id: str | None
    exchange_id: str
    label: str
    provider: str
    model: str
    recorded_at: str
    prompt: str
    response: str
    reasoning: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    previous_response_id: str | None = None
    response_id: str | None = None
    returned_signatures: int = 0
    replayed_signatures: int = 0
    status: str = "success"
    error: dict[str, Any] | None = None
    attempt: int = 0
    http_status: int | None = None
    stop_reason: str | None = None
    stop_details: dict[str, Any] | None = None

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"

    @property
    def refusal_note(self) -> str:
        details = self.stop_details or {}
        category = str(details.get("category") or "unspecified")
        explanation = str(details.get("explanation") or "").strip()
        note = f"Blocked by the provider's usage policy (category: {category})."
        return f"{note} {explanation}".strip()


@dataclass(slots=True)
class Phase:
    """A run of consecutive turns that belong to one stage of the eval.

    A full AlienCode run is a few hundred calls, most of them held-out test
    questions. Flattening them into one list buries the interesting part --
    what the model probed and what rules it inferred -- so the replay groups
    turns by stage and folds the graded questions away by default.
    """

    key: str
    title: str
    turns: list["Turn"] = field(default_factory=list)
    collapsed: bool = False

    def usage(self) -> dict[str, int]:
        keys = (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        )
        totals = {key: 0 for key in keys}
        for turn in self.turns:
            for key in keys:
                totals[key] += int(turn.usage.get(key, 0) or 0)
        return totals

    @property
    def refusals(self) -> int:
        return sum(1 for turn in self.turns if turn.refused)


# Both sandboxes are matched here: AlienCode writes "Explore 1-1",
# "Milestone 0 Summary" and "M0 Test 12"; AlienLogic writes "Explore 1.1",
# "M0 Summary" and "Mpre T01" for the pre-exploration baseline.
_SEED = re.compile(r"^Seed\s+(\S+)")
_EXPLORE = re.compile(r"^Explore\s+(\d+)[-.](\d+)")
_SUMMARY = re.compile(r"^(?:Milestone\s+|M)(_?pre|\d+)\s+Summary")
_TEST = re.compile(r"^M(_?pre|\d+)\s+T(?:est\b|\d)")


def _milestone_name(index: str) -> str:
    return "基线（探索前）" if index.lstrip("_") == "pre" else f"Milestone {index}"


def _phase_of(label: str) -> tuple[str, str, bool]:
    """Map a turn label to (phase key, human title, collapsed by default)."""

    label = label.strip()
    if _SEED.match(label):
        return "seed", "Seed 阶段 · 首次接触环境", False
    match = _EXPLORE.match(label)
    if match:
        loop = match.group(1)
        return f"explore-{loop}", f"自由探索 · 第 {loop} 轮", False
    match = _SUMMARY.match(label)
    if match:
        index = match.group(1)
        return (
            f"summary-{index}",
            f"{_milestone_name(index)} · 闭卷总结规则",
            False,
        )
    match = _TEST.match(label)
    if match:
        index = match.group(1)
        return (
            f"tests-{index}",
            f"{_milestone_name(index)} · 闭卷答题",
            True,
        )
    return "other", "其他调用", False


def _phase_order(key: str) -> tuple[int, str]:
    """Canonical eval order, independent of parallel fork completion order."""

    if key == "summary-_pre" or key == "summary-pre":
        return -3, key
    if key == "tests-_pre" or key == "tests-pre":
        return -2, key
    if key == "seed":
        return -1, key
    match = re.match(r"^(explore|summary|tests)-_?(\d+)$", key)
    if match:
        kind, raw = match.groups()
        offset = {"explore": 0, "summary": 1, "tests": 2}[kind]
        return int(raw) * 3 + offset, key
    return 10_000, key


def group_turns(turns: list["Turn"]) -> list[Phase]:
    """Group the timeline by eval stage in canonical milestone order.

    Held-out forks execute concurrently (and Logic can defer every milestone
    into one shared pool), so their completion order is not the pedagogical
    order of the evaluation. Within a phase, JSONL/wire order is preserved.
    """

    phases_by_key: dict[str, Phase] = {}
    for turn in turns:
        key, title, collapsed = _phase_of(turn.label)
        if key not in phases_by_key:
            phases_by_key[key] = Phase(
                key=key, title=title, collapsed=collapsed
            )
        phases_by_key[key].turns.append(turn)
    return sorted(phases_by_key.values(), key=lambda phase: _phase_order(phase.key))


@dataclass(slots=True)
class Interaction:
    source: Path
    turns: list[Turn] = field(default_factory=list)
    failures: list[Turn] = field(default_factory=list)
    sessions: dict[str, dict[str, Any]] = field(default_factory=dict)
    restores: list[dict[str, Any]] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)

    @property
    def provider(self) -> str:
        return self.turns[0].provider if self.turns else "unknown"

    @property
    def model(self) -> str:
        return self.turns[0].model if self.turns else "unknown"

    def totals(self) -> dict[str, int | float]:
        keys = (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "total_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        )
        totals: dict[str, int | float] = {key: 0 for key in keys}
        for turn in self.turns:
            for key in keys:
                totals[key] += int(turn.usage.get(key, 0) or 0)
        if self.provider == "anthropic":
            # Anthropic reports uncached, cache-create and cache-read input as
            # three disjoint counters; the others fold cached into the input.
            billed = (
                totals["input_tokens"]
                + totals["cache_creation_input_tokens"]
                + totals["cache_read_input_tokens"]
            )
        else:
            billed = totals["input_tokens"]
        totals["effective_input_tokens"] = billed
        totals["cache_hit_ratio"] = (
            totals["cache_read_input_tokens"] / billed if billed else 0.0
        )
        totals["calls"] = len(self.turns)
        return totals


def open_trace(path: Path):
    """A trace reader that accepts the archived form as well as the live one.

    Completed runs are kept gzipped, so requiring the plain file would make a
    replay impossible for every run that has already been archived.
    """

    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def trace_stem(path: Path) -> str:
    """Trace basename without either ``.jsonl`` or ``.jsonl.gz``."""

    name = path.name
    if name.endswith(".gz"):
        name = name[:-3]
    if name.endswith(".jsonl"):
        name = name[:-6]
    return name


def load_trace(path: Path) -> Interaction:
    interaction = Interaction(source=path)
    seen_exchanges: set[str] = set()
    # Signatures sent *back* to the provider, which is what proves the
    # reasoning context survived. Only the request payload shows this.
    replayed: dict[str, int] = {}

    # A small number of long provider traces contain isolated non-UTF-8 bytes
    # from a gateway write. Keep the rest of the run reviewable and surface the
    # damaged line explicitly instead of making the whole HTML export fail.
    with open_trace(path) as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            if "\ufffd" in line:
                interaction.parse_errors.append(
                    f"line {line_number}: invalid UTF-8 byte(s) replaced"
                )
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                interaction.parse_errors.append(f"line {line_number}: {exc}")
                continue
            if not isinstance(record, dict):
                interaction.parse_errors.append(
                    f"line {line_number}: record is not an object"
                )
                continue

            kind = record.get("record_type")
            if kind == "session_created":
                interaction.sessions.setdefault(
                    str(record.get("session_id", "")),
                    {"parent": None, "origin": "root"},
                )
            elif kind == "session_fork":
                interaction.sessions[str(record.get("session_id", ""))] = {
                    "parent": record.get("parent_session_id"),
                    "origin": "fork",
                }
            elif kind == "session_restore":
                interaction.restores.append(record)
            elif kind == "api_exchange":
                if record.get("status") == "failed":
                    interaction.failures.append(_failed_turn(record))
                    continue
                exchange_id = str(record.get("exchange_id", ""))
                if exchange_id:
                    replayed[exchange_id] = _count_signatures(
                        record.get("request"),
                        str(record.get("provider", "")),
                    )
            elif kind == "trajectory":
                # `api_exchange` and `trajectory` describe the same call; the
                # trajectory view is the readable one.
                exchange_id = str(record.get("exchange_id", ""))
                if exchange_id and exchange_id in seen_exchanges:
                    continue
                seen_exchanges.add(exchange_id)
                interaction.turns.append(_turn(record))

    for turn in interaction.turns:
        turn.replayed_signatures = replayed.get(turn.exchange_id, 0)

    # The sequence counter is session-local for forked held-out calls. Sorting
    # globally by it interleaves unrelated sessions and shatters one milestone
    # into hundreds of tiny sections. JSONL append order is the wire record.
    return interaction


def _turn(record: dict[str, Any]) -> Turn:
    history = [
        _repair_display_value(event)
        for event in (record.get("history_delta") or [])
        if isinstance(event, dict)
    ]
    tool_calls = [
        event.get("payload", {})
        for event in history
        if event.get("kind") == "tool_call"
    ]
    tool_results = [
        event.get("payload", {})
        for event in history
        if event.get("kind") == "tool_result"
    ]
    provider = str(record.get("provider", "unknown"))
    return Turn(
        sequence=int(record.get("sequence", 0) or 0),
        session_id=str(record.get("session_id", "")),
        parent_session_id=record.get("parent_session_id"),
        exchange_id=str(record.get("exchange_id", "")),
        label=_repair_mojibake(str(record.get("label") or "(unlabelled)")),
        provider=provider,
        model=str(record.get("model", "")),
        recorded_at=str(record.get("recorded_at", "")),
        prompt=_as_text(record.get("prompt")),
        response=_as_text(record.get("response")),
        reasoning=[
            _repair_display_value(item)
            for item in (record.get("reasoning") or [])
            if isinstance(item, dict)
        ],
        tool_calls=tool_calls,
        tool_results=tool_results,
        usage=record.get("usage") or {},
        previous_response_id=record.get("previous_response_id"),
        response_id=record.get("response_id"),
        returned_signatures=_count_signatures(
            record.get("reasoning"), provider
        ) or sum(
            1 for item in (record.get("reasoning") or [])
            if isinstance(item, dict) and item.get("signature")
        ),
        stop_reason=record.get("stop_reason"),
        stop_details=(
            record.get("stop_details")
            if isinstance(record.get("stop_details"), dict)
            else None
        ),
    )


def _failed_turn(record: dict[str, Any]) -> Turn:
    return Turn(
        sequence=int(record.get("sequence", 0) or 0),
        session_id=str(record.get("session_id", "")),
        parent_session_id=record.get("parent_session_id"),
        exchange_id=str(record.get("exchange_id", "")),
        label=str(record.get("label") or "(unlabelled)"),
        provider=str(record.get("provider", "unknown")),
        model=str(record.get("model", "")),
        recorded_at=str(record.get("recorded_at", "")),
        prompt="",
        response="",
        status="failed",
        error=_repair_display_value(record.get("error")),
        attempt=int(record.get("attempt", 0) or 0),
        http_status=(
            int(record["http_status"])
            if isinstance(record.get("http_status"), (int, str))
            and str(record.get("http_status")).isdigit()
            else None
        ),
    )


def _count_signatures(value: Any, provider: str) -> int:
    count = 0
    for item in _walk(value):
        if provider == "anthropic" and item.get("type") in {
            "thinking",
            "redacted_thinking",
        }:
            if item.get("signature") or item.get("data"):
                count += 1
        elif provider == "gemini" and item.get("thoughtSignature"):
            count += 1
    return count


def _walk(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _repair_mojibake(value)
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(_repair_mojibake(item["text"]))
            else:
                parts.append(
                    _repair_mojibake(
                        json.dumps(item, ensure_ascii=False, indent=2)
                    )
                )
        return "\n".join(parts)
    return _repair_mojibake(json.dumps(value, ensure_ascii=False, indent=2))


_MOJIBAKE_MARKERS = frozenset("ÃÂâäåæçèéïð")


def _mojibake_score(value: str) -> int:
    return sum(
        1
        for char in value
        if char in _MOJIBAKE_MARKERS or 0x80 <= ord(char) <= 0x9F
    )


def _repair_mojibake(value: str) -> str:
    """Repair UTF-8 text that a gateway decoded once as Latin-1.

    The raw JSONL stays byte-for-byte untouched. This only improves the human
    replay/archive view, and only when the round trip strictly lowers a
    conservative mojibake score.
    """

    if not value or _mojibake_score(value) == 0:
        return value
    try:
        candidate = value.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value
    return candidate if _mojibake_score(candidate) < _mojibake_score(value) else value


def _repair_display_value(value: Any) -> Any:
    if isinstance(value, str):
        return _repair_mojibake(value)
    if isinstance(value, list):
        return [_repair_display_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _repair_display_value(item) for key, item in value.items()}
    return value


def fingerprint(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{digest} ({len(value)} chars)"


def _short(value: str | None, keep: int = 12) -> str:
    if not value:
        return "—"
    return value if len(value) <= keep * 2 else f"{value[:keep]}…{value[-4:]}"


# ── rendering ──────────────────────────────────────────────────────────

_STYLE = """
:root{--bg:#0f1115;--panel:#171a21;--panel2:#1e222b;--line:#2b303b;
--text:#e6e8ee;--dim:#9aa3b2;--user:#4c8dff;--assist:#3ddc97;
--think:#c08cff;--tool:#ffb454;--err:#ff6b6b}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",
"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif}
.wrap{max-width:1040px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:22px;margin:0 0 4px}
.sub{color:var(--dim);font-size:13px;margin-bottom:24px;word-break:break-all}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
gap:12px;margin-bottom:28px}
.card{background:var(--panel);border:1px solid var(--line);
border-radius:10px;padding:12px 14px}
.card .k{color:var(--dim);font-size:12px;text-transform:uppercase;
letter-spacing:.04em}
.card .v{font-size:19px;font-weight:600;margin-top:4px;word-break:break-all}
.turn{background:var(--panel);border:1px solid var(--line);
border-radius:12px;margin-bottom:18px;overflow:hidden}
.turn.forked{border-left:3px solid var(--think)}
.turn.failed{border-left:3px solid var(--err)}
.thead{display:flex;align-items:center;gap:10px;flex-wrap:wrap;
padding:12px 16px;background:var(--panel2);border-bottom:1px solid var(--line)}
.num{color:var(--dim);font-variant-numeric:tabular-nums;font-size:13px}
.label{font-weight:600}
.tags{margin-left:auto;display:flex;gap:6px;flex-wrap:wrap}
.tag{font-size:11px;padding:2px 8px;border-radius:999px;
border:1px solid var(--line);color:var(--dim);white-space:nowrap}
.tag.ok{color:var(--assist);border-color:#2c5b47}
.tag.warn{color:var(--tool);border-color:#5c4526}
.tag.bad{color:var(--err);border-color:#5c2b2b}
.block{padding:14px 16px;border-bottom:1px solid var(--line)}
.block:last-child{border-bottom:none}
.role{font-size:12px;font-weight:700;letter-spacing:.06em;
text-transform:uppercase;margin-bottom:8px}
.role.user{color:var(--user)}.role.assistant{color:var(--assist)}
.role.think{color:var(--think)}.role.tool{color:var(--tool)}
.role.err{color:var(--err)}.role.refused{color:var(--err)}
.block.refusal{background:rgba(180,60,60,.10)}
pre{margin:0;white-space:pre-wrap;word-wrap:break-word;font:13px/1.65
ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
details summary{cursor:pointer;color:var(--dim);font-size:13px;
list-style:none;user-select:none}
details summary::-webkit-details-marker{display:none}
details summary:before{content:"\\25B8 ";color:var(--think)}
details[open] summary:before{content:"\\25BE "}
details>pre,details>div{margin-top:10px;padding-left:12px;
border-left:2px solid var(--line)}
.meta{display:flex;gap:16px;flex-wrap:wrap;padding:10px 16px;
background:var(--panel2);color:var(--dim);font-size:12px;
font-variant-numeric:tabular-nums}
.controls{display:flex;gap:10px;margin-bottom:18px}
button{background:var(--panel2);color:var(--text);border:1px solid var(--line);
border-radius:8px;padding:7px 14px;font-size:13px;cursor:pointer}
button:hover{border-color:var(--dim)}
.issues{background:#2a1b1b;border:1px solid #5c2b2b;border-radius:10px;
padding:12px 16px;margin-bottom:22px}
.issues li{color:#ffc9c9;font-size:13px}
.sess{font-family:ui-monospace,monospace;font-size:11px}
.outline{background:var(--panel);border:1px solid var(--line);
border-radius:10px;padding:14px 18px;margin-bottom:24px}
.otitle{font-size:12px;text-transform:uppercase;letter-spacing:.04em;
color:var(--dim);margin-bottom:8px}
.outline ol{margin:0;padding-left:22px}
.outline li{margin:3px 0}
.outline a{color:var(--text);text-decoration:none}
.outline a:hover{color:var(--user);text-decoration:underline}
.onote{color:var(--dim);font-size:12px;margin-left:8px;
font-variant-numeric:tabular-nums}
.phase{margin:0 0 22px;border:1px solid var(--line);border-radius:12px;
background:#12151c;scroll-margin-top:16px}
.phase>summary{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;
padding:13px 18px;background:var(--panel2);border-radius:11px}
.phase[open]>summary{border-bottom:1px solid var(--line);
border-radius:11px 11px 0 0}
.phase>summary:before{content:"\\25B8 ";color:var(--user)}
.phase[open]>summary:before{content:"\\25BE "}
.ptitle{font-size:16px;font-weight:600;color:var(--text)}
.pmeta{margin-left:auto;color:var(--dim);font-size:12px;
font-variant-numeric:tabular-nums}
.pmeta .bad{color:var(--err)}
.pbody{padding:18px 18px 2px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;
padding:18px 20px;margin-bottom:24px}
.ptitle2{font-size:15px;font-weight:600;margin-bottom:4px}
.ptitle2.gap{margin-top:26px}
.chint{color:var(--dim);font-size:12px;margin-bottom:10px}
.chint.redo{margin:12px 0 0;padding:9px 12px;border-radius:8px;
background:rgba(255,180,84,.08);border:1px solid #5c4526;color:#ffd9a8}
.chart{width:100%;height:auto;display:block}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin-top:6px;
color:var(--dim);font-size:12px}
.legend .lg{display:inline-flex;align-items:center;gap:6px}
.legend i{width:10px;height:10px;border-radius:2px;display:inline-block}
.mtable{width:100%;border-collapse:collapse;margin-top:22px;font-size:13px;
font-variant-numeric:tabular-nums}
.mtable th{text-align:right;color:var(--dim);font-weight:600;font-size:12px;
padding:6px 10px;border-bottom:1px solid var(--line)}
.mtable th:first-child,.mtable td:first-child{text-align:left}
.mtable td{text-align:right;padding:6px 10px;border-bottom:1px solid #21252e}
"""

_SCRIPT = """
function toggleAll(open){
  document.querySelectorAll('details').forEach(function(d){d.open=open});
}
function togglePhases(open){
  document.querySelectorAll('details.phase').forEach(function(d){d.open=open});
}
"""

# Charts are inline SVG rather than a charting library so the file stays
# openable offline, with no network fetch and nothing to install.
_CHART_COLORS = ("#4c8dff", "#3ddc97", "#ffb454", "#c08cff", "#ff6b6b",
                 "#5ec8d8", "#e6e8ee", "#f78fb3", "#a3d977", "#8f9bff",
                 "#d9a441", "#6fd3b8")
_DIMENSIONS = (
    ("apply", "应用"), ("interact", "交互"), ("scope", "作用域"),
    ("engineer", "工程"), ("algorithm_1", "算法1"),
    ("algorithm_2", "算法2"), ("algorithm_3", "算法3"),
)


def _axes(width: int, height: int, pad: dict[str, int], y_max: float,
          y_suffix: str, categories: list[str]) -> list[str]:
    left, right = pad["l"], width - pad["r"]
    top, bottom = pad["t"], height - pad["b"]
    out = []
    for step in range(6):
        value = y_max * step / 5
        y = bottom - (bottom - top) * step / 5
        out.append(
            f"<line x1='{left}' y1='{y:.1f}' x2='{right}' y2='{y:.1f}' "
            f"stroke='#2b303b' stroke-width='1'/>"
        )
        out.append(
            f"<text x='{left - 8}' y='{y + 4:.1f}' fill='#9aa3b2' "
            f"font-size='11' text-anchor='end'>{value:g}{y_suffix}</text>"
        )
    span = max(len(categories) - 1, 1)
    for index, name in enumerate(categories):
        x = left + (right - left) * index / span
        out.append(
            f"<text x='{x:.1f}' y='{bottom + 18}' fill='#9aa3b2' "
            f"font-size='11' text-anchor='middle'>{html.escape(name)}</text>"
        )
    return out


def _legend(series: list[tuple[str, list[float]]], width: int) -> str:
    chips = []
    for index, (name, _) in enumerate(series):
        color = _CHART_COLORS[index % len(_CHART_COLORS)]
        chips.append(
            f"<span class='lg'><i style='background:{color}'></i>"
            f"{html.escape(name)}</span>"
        )
    return f"<div class='legend'>{''.join(chips)}</div>"


def _line_chart(categories: list[str], series: list[tuple[str, list[float]]],
                *, y_max: float = 100, y_suffix: str = "%",
                height: int = 240, width: int = 900) -> str:
    pad = {"l": 52, "r": 18, "t": 14, "b": 30}
    left, right = pad["l"], width - pad["r"]
    top, bottom = pad["t"], height - pad["b"]
    span = max(len(categories) - 1, 1)
    body = _axes(width, height, pad, y_max, y_suffix, categories)
    for index, (_, values) in enumerate(series):
        color = _CHART_COLORS[index % len(_CHART_COLORS)]
        points = []
        for position, value in enumerate(values):
            x = left + (right - left) * position / span
            y = bottom - (bottom - top) * (min(value, y_max) / y_max)
            points.append((x, y))
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        body.append(
            f"<polyline points='{path}' fill='none' stroke='{color}' "
            f"stroke-width='2' stroke-linejoin='round'/>"
        )
        for (x, y), value in zip(points, values):
            body.append(
                f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3.5' fill='{color}'>"
                f"<title>{value:g}{y_suffix}</title></circle>"
            )
    return (
        f"<svg viewBox='0 0 {width} {height}' class='chart' "
        f"preserveAspectRatio='xMidYMid meet'>{''.join(body)}</svg>"
        + _legend(series, width)
    )


def _bar_chart(categories: list[str], series: list[tuple[str, list[float]]],
               *, y_max: float = 100, y_suffix: str = "%",
               height: int = 240, width: int = 900) -> str:
    pad = {"l": 52, "r": 18, "t": 14, "b": 30}
    left, right = pad["l"], width - pad["r"]
    top, bottom = pad["t"], height - pad["b"]
    body = _axes(width, height, pad, y_max, y_suffix, categories)
    slot = (right - left) / max(len(categories), 1)
    bar = min(slot / (len(series) + 1), 26)
    for group, name in enumerate(categories):
        base = left + slot * group + (slot - bar * len(series)) / 2
        for index, (_, values) in enumerate(series):
            if group >= len(values):
                continue
            value = min(values[group], y_max)
            colour = _CHART_COLORS[index % len(_CHART_COLORS)]
            tall = (bottom - top) * (value / y_max)
            x = base + bar * index
            body.append(
                f"<rect x='{x:.1f}' y='{bottom - tall:.1f}' "
                f"width='{bar:.1f}' height='{tall:.1f}' fill='{colour}' "
                f"rx='2'><title>{html.escape(name)} · "
                f"{html.escape(series[index][0])}: {value:g}{y_suffix}</title>"
                f"</rect>"
            )
    return (
        f"<svg viewBox='0 0 {width} {height}' class='chart' "
        f"preserveAspectRatio='xMidYMid meet'>{''.join(body)}</svg>"
        + _legend(series, width)
    )


_LOGIC_ROLES = [
    ("sanity", "对照题"),
    ("single_rule", "单规则"),
    ("two_rule", "双规则"),
    ("multi_rule", "多规则"),
    ("trap", "陷阱题"),
    ("unprovable", "不可证"),
]


def _render_charts_logic(milestones: list[dict[str, Any]]) -> str:
    """The same charts for AlienLogic, which scores proofs rather than code.

    A logic milestone reports rates directly instead of a rule count, and the
    baseline milestone is labelled 'pre' rather than numbered.
    """

    def label_of(m: dict[str, Any], i: int) -> str:
        value = m.get("milestone", i)
        return "Mpre" if value == "pre" else f"M{value}"

    labels = [label_of(m, i) for i, m in enumerate(milestones)]

    def rate(key: str) -> list[float]:
        return [round(100 * (m.get(key) or 0), 1) for m in milestones]

    total = milestones[0].get("test_total") or 85
    out = [
        "<div class='panel'><div class='ptitle2'>学习曲线</div>",
        "<div class='chint'>纵轴为百分比。定理通过率 = verifier 接受的证明数 ÷ "
        f"{total}（作答时无法调用 verifier）；含隐藏规则题通过率只统计触及 alien "
        "规则的题；不可证识别率 = 正确判定为不可证的比例</div>",
        _line_chart(labels, [
            ("定理通过率", rate("pass_rate")),
            ("含隐藏规则题", rate("alien_aware_pass_rate")),
            ("不可证识别率", rate("unprovable_recognition_rate")),
        ]),
    ]

    roles = [
        (key, name) for key, name in _LOGIC_ROLES
        if any((m.get("pass_by_role") or {}).get(key) for m in milestones)
    ]
    if roles:
        series = []
        for label, m in zip(labels, milestones):
            by_role = m.get("pass_by_role") or {}
            values = []
            for key, _ in roles:
                cell = by_role.get(key) or {}
                den = cell.get("total") or 0
                values.append(
                    round(100 * (cell.get("pass") or 0) / den, 1) if den else 0.0
                )
            series.append((label, values))
        out.extend([
            "<div class='ptitle2 gap'>分难度通过率</div>",
            "<div class='chint'>每个里程碑在各类定理上的通过率："
            "对照题不涉及隐藏规则，规则数越多越难，陷阱题专为常见误解设计</div>",
            _bar_chart([name for _, name in roles], series),
        ])

    rows = ["<table class='mtable'><thead><tr><th>里程碑</th><th>通过</th>"
            "<th>通过率</th><th>含隐藏规则题</th><th>不可证识别</th>"
            "<th>证明简洁度</th></tr></thead><tbody>"]
    for label, m in zip(labels, milestones):
        def pctf(key: str) -> str:
            value = m.get(key)
            return f"{100 * value:.1f}%" if isinstance(value, (int, float)) else "—"
        minimality = m.get("proof_minimality_mean")
        rows.append(
            f"<tr><td>{html.escape(label)}</td>"
            f"<td>{m.get('test_pass')}/{m.get('test_total')}</td>"
            f"<td>{pctf('pass_rate')}</td>"
            f"<td>{pctf('alien_aware_pass_rate')}</td>"
            f"<td>{pctf('unprovable_recognition_rate')}</td>"
            f"<td>{f'{minimality:.2f}' if isinstance(minimality, (int, float)) else '—'}</td>"
            "</tr>"
        )
    rows.append("</tbody></table></div>")
    out.extend(rows)
    return "\n".join(out)


def _render_charts(scores: dict[str, Any]) -> str:
    """Learning curves and per-dimension accuracy from a scored run."""

    milestones = [
        m for m in (scores.get("milestones") or [])
        if isinstance(m, dict)
    ]
    if not milestones:
        return ""
    # The two sandboxes score different things, so tell them apart by the
    # shape of a milestone rather than threading the sandbox down to here.
    if any("pass_rate" in m or "test_pass" in m for m in milestones):
        return _render_charts_logic(milestones)
    labels = [f"M{m.get('milestone_idx', i)}" for i, m in enumerate(milestones)]

    def pct(num, den):
        return round(100 * (num or 0) / den, 1) if den else 0.0

    rules = [pct(m.get("found"), m.get("total") or 0) for m in milestones]
    tests = [pct(m.get("test_correct"), m.get("test_total") or 0)
             for m in milestones]

    out = [
        "<div class='panel'><div class='ptitle2'>学习曲线</div>",
        "<div class='chint'>纵轴为百分比：规则发现率 = 归纳出的规则数 ÷ "
        f"{milestones[0].get('total') or 31}；闭卷正确率 = 答对题数 ÷ "
        f"{milestones[0].get('test_total') or 90}（作答时无法执行代码）</div>",
        _line_chart(labels, [("规则发现率", rules), ("闭卷正确率", tests)]),
    ]

    dims = [(key, name) for key, name in _DIMENSIONS
            if any(m.get(f"{key}_accuracy") is not None for m in milestones)]
    if dims:
        series = [
            (label, [round(100 * (m.get(f"{key}_accuracy") or 0), 1)
                     for key, _ in dims])
            for label, m in zip(labels, milestones)
        ]
        out.extend([
            "<div class='ptitle2 gap'>分维度正确率</div>",
            "<div class='chint'>每个里程碑在七类题目上的正确率，"
            "用来看能力短板落在哪一类</div>",
            _bar_chart([name for _, name in dims], series),
        ])

    rows = ["<table class='mtable'><thead><tr><th>里程碑</th><th>规则</th>"
            "<th>闭卷答对</th><th>正确率</th><th>用时</th><th>备注</th>"
            "</tr></thead><tbody>"]
    redone = False
    for label, m in zip(labels, milestones):
        seconds = m.get("milestone_time")
        spent = f"{seconds / 60:.1f} 分钟" if isinstance(seconds, (int, float)) else "—"
        redo = m.get("redo") if isinstance(m.get("redo"), dict) else None
        note = "—"
        if redo:
            redone = True
            failed = len(redo.get("still_failed") or [])
            note = f"重跑 {len(redo.get('recovered') or [])} 题"
            if failed:
                note += f"，仍失败 {failed} 题"
        rows.append(
            f"<tr><td>{html.escape(label)}</td>"
            f"<td>{m.get('found')}/{m.get('total')}</td>"
            f"<td>{m.get('test_correct')}/{m.get('test_total')}</td>"
            f"<td>{pct(m.get('test_correct'), m.get('test_total') or 0):g}%</td>"
            f"<td>{html.escape(spent)}</td>"
            f"<td>{html.escape(note)}</td></tr>"
        )
    rows.append("</tbody></table>")
    out.extend(rows)
    if redone:
        out.append(
            "<div class='chint redo'>标注为“重跑”的里程碑：原始运行中有题目"
            "因网关报错而未拿到作答，已在同一会话快照上重新提问后计分；"
            "下方的交互回放仍是原始记录，因此会看到那些失败的调用。</div>"
        )
    out.append("</div>")
    return "".join(out)


def render_html(
    interaction: Interaction,
    *,
    title: str | None = None,
    scores: dict[str, Any] | None = None,
) -> str:
    totals = interaction.totals()
    heading = title or f"{interaction.model} — interaction replay"
    root_sessions = {
        key for key, value in interaction.sessions.items()
        if value.get("origin") == "root"
    }

    parts: list[str] = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>{html.escape(heading)}</title>",
        f"<style>{_STYLE}</style></head><body><div class='wrap'>",
        f"<h1>{html.escape(heading)}</h1>",
        f"<div class='sub'>{html.escape(str(interaction.source))}</div>",
    ]

    cards = [
        ("Protocol", interaction.provider),
        ("Model calls", str(totals["calls"])),
        ("Sessions", str(max(len(interaction.sessions), 1))),
        ("Input tokens", f"{totals['effective_input_tokens']:,}"),
        ("Output tokens", f"{totals['output_tokens']:,}"),
        ("Reasoning tokens", f"{totals['reasoning_tokens']:,}"),
        ("Cache hit", f"{totals['cache_hit_ratio'] * 100:.1f}%"),
        ("Failed attempts", str(len(interaction.failures))),
    ]
    parts.append("<div class='cards'>")
    for key, value in cards:
        parts.append(
            f"<div class='card'><div class='k'>{html.escape(key)}</div>"
            f"<div class='v'>{html.escape(value)}</div></div>"
        )
    parts.append("</div>")

    if interaction.parse_errors:
        parts.append("<div class='issues'><strong>Trace parse errors</strong><ul>")
        for issue in interaction.parse_errors:
            parts.append(f"<li>{html.escape(issue)}</li>")
        parts.append("</ul></div>")

    if scores:
        parts.append(_render_charts(scores))

    parts.append(
        "<div class='controls'>"
        "<button onclick='toggleAll(true)'>展开全部</button>"
        "<button onclick='toggleAll(false)'>收起全部</button>"
        "<button onclick='togglePhases(true)'>展开所有阶段</button>"
        "<button onclick='togglePhases(false)'>只看阶段目录</button></div>"
    )

    phases = group_turns(interaction.turns)
    if phases:
        parts.append(_render_outline(phases))

    index = 0
    for phase_number, phase in enumerate(phases, 1):
        parts.append(_render_phase_head(phase_number, phase))
        for turn in phase.turns:
            index += 1
            parts.append(_render_turn(index, turn, root_sessions))
        parts.append("</div></details>")

    for turn in interaction.failures:
        parts.append(_render_failure(turn))

    for restore in interaction.restores:
        parts.append(
            "<div class='turn failed'><div class='thead'>"
            "<span class='label'>Session rolled back</span>"
            f"<span class='num sess'>{html.escape(_short(str(restore.get('session_id'))))}"
            "</span></div><div class='block'><pre>"
            f"discarded response: {html.escape(str(restore.get('discarded_response_id') or '—'))}\n"
            f"restored response: {html.escape(str(restore.get('restored_response_id') or '—'))}"
            "</pre></div></div>"
        )

    parts.append(f"</div><script>{_SCRIPT}</script></body></html>")
    return "\n".join(parts)


def _render_outline(phases: list[Phase]) -> str:
    """A table of contents, so the shape of the run is visible at a glance."""

    rows = ["<div class='outline'><div class='otitle'>评测流程</div><ol>"]
    for number, phase in enumerate(phases, 1):
        usage = phase.usage()
        note = f"{len(phase.turns)} 次调用"
        if phase.refusals:
            note += f" · {phase.refusals} 次被拒"
        rows.append(
            f"<li><a href='#phase-{number}'>{html.escape(phase.title)}</a>"
            f"<span class='onote'>{html.escape(note)} · "
            f"出 {usage['output_tokens']:,} tokens</span></li>"
        )
    rows.append("</ol></div>")
    return "".join(rows)


def _render_phase_head(number: int, phase: Phase) -> str:
    usage = phase.usage()
    billed = (
        usage["input_tokens"]
        + usage["cache_creation_input_tokens"]
        + usage["cache_read_input_tokens"]
    )
    hit = usage["cache_read_input_tokens"] / billed if billed else 0.0
    bits = [
        f"{len(phase.turns)} 次调用",
        f"入 {billed:,}",
        f"出 {usage['output_tokens']:,}",
        f"缓存命中 {hit * 100:.0f}%",
    ]
    if phase.refusals:
        bits.append(f"<span class='bad'>{phase.refusals} 次被拒</span>")
    open_attr = "" if phase.collapsed else " open"
    return (
        f"<details class='phase' id='phase-{number}'{open_attr}>"
        f"<summary><span class='ptitle'>{html.escape(phase.title)}</span>"
        f"<span class='pmeta'>{' · '.join(bits)}</span></summary>"
        "<div class='pbody'>"
    )


def _render_turn(index: int, turn: Turn, roots: set[str]) -> str:
    forked = bool(turn.parent_session_id) or turn.session_id not in roots
    classes = "turn forked" if forked else "turn"
    out = [f"<div class='{classes}'>"]

    tags = [f"<span class='tag'>{html.escape(turn.provider)}</span>"]
    if forked:
        tags.append("<span class='tag warn'>forked session</span>")
    if turn.replayed_signatures:
        tags.append(
            f"<span class='tag ok'>replayed {turn.replayed_signatures} "
            "signed block(s)</span>"
        )
    if turn.returned_signatures:
        tags.append(
            f"<span class='tag ok'>signed x{turn.returned_signatures}</span>"
        )
    if turn.previous_response_id:
        tags.append(
            "<span class='tag ok'>continues "
            f"{html.escape(_short(turn.previous_response_id, 8))}</span>"
        )
    if turn.tool_calls:
        tags.append(
            f"<span class='tag warn'>{len(turn.tool_calls)} tool call(s)</span>"
        )
    if turn.refused:
        category = str((turn.stop_details or {}).get("category") or "policy")
        tags.append(
            f"<span class='tag bad'>refused · {html.escape(category)}</span>"
        )

    out.append(
        f"<div class='thead'><span class='num'>#{index}</span>"
        f"<span class='label'>{html.escape(turn.label)}</span>"
        f"<span class='tags'>{''.join(tags)}</span></div>"
    )

    if turn.tool_results:
        out.append("<div class='block'><div class='role tool'>Tool results sent back</div>")
        for result in turn.tool_results:
            out.append(
                "<pre>"
                f"{html.escape(str(result.get('name') or result.get('call_id') or 'tool'))}"
                f" → {html.escape(_as_text(result.get('output', result.get('content'))))}"
                "</pre>"
            )
        out.append("</div>")

    if turn.prompt:
        out.append(
            "<div class='block'><div class='role user'>User</div>"
            f"<pre>{html.escape(turn.prompt)}</pre></div>"
        )

    if turn.reasoning:
        out.append("<div class='block'><div class='role think'>Reasoning</div>")
        for artifact in turn.reasoning:
            text = artifact.get("text") or ""
            signature = artifact.get("signature") or ""
            summary = (
                f"{len(text)} chars"
                if text
                else f"{artifact.get('kind', 'reasoning')} (no visible text)"
            )
            if signature:
                summary += f" · signature {fingerprint(signature)}"
            out.append(
                "<details><summary>"
                f"{html.escape(summary)}</summary>"
                f"<pre>{html.escape(text) or '(the provider returned only a signed opaque block)'}</pre>"
                "</details>"
            )
        out.append("</div>")

    if turn.tool_calls:
        out.append("<div class='block'><div class='role tool'>Tool calls</div>")
        for call in turn.tool_calls:
            arguments = json.dumps(
                call.get("arguments", {}), ensure_ascii=False
            )
            out.append(
                f"<pre>{html.escape(str(call.get('name', '?')))}"
                f"({html.escape(arguments)})</pre>"
            )
        out.append("</div>")

    if turn.response:
        out.append(
            "<div class='block'><div class='role assistant'>Assistant</div>"
            f"<pre>{html.escape(turn.response)}</pre></div>"
        )

    if turn.refused:
        out.append(
            "<div class='block refusal'>"
            "<div class='role refused'>Refused</div>"
            f"<pre>{html.escape(turn.refusal_note)}</pre></div>"
        )

    usage = turn.usage
    ratio = float(usage.get("cache_hit_ratio", 0) or 0)
    out.append(
        "<div class='meta'>"
        f"<span>in {int(usage.get('input_tokens', 0) or 0):,}</span>"
        f"<span>out {int(usage.get('output_tokens', 0) or 0):,}</span>"
        f"<span>reasoning {int(usage.get('reasoning_tokens', 0) or 0):,}</span>"
        f"<span>cached {int(usage.get('cache_read_input_tokens', 0) or 0):,}"
        f" ({ratio * 100:.0f}%)</span>"
        f"<span class='sess'>session {html.escape(_short(turn.session_id, 8))}</span>"
        f"<span>{html.escape(turn.recorded_at)}</span>"
        "</div></div>"
    )
    return "".join(out)


def _render_failure(turn: Turn) -> str:
    error = turn.error or {}
    return (
        "<div class='turn failed'><div class='thead'>"
        f"<span class='label'>{html.escape(turn.label)}</span>"
        f"<span class='tags'><span class='tag bad'>attempt "
        f"{turn.attempt} failed</span></span></div>"
        "<div class='block'><div class='role err'>Error</div><pre>"
        f"{html.escape(str(error.get('type', 'error')))}: "
        f"{html.escape(str(error.get('message', ''))[:2000])}</pre></div></div>"
    )


def render_file(
    trace: Path,
    output: Path | None = None,
    *,
    title: str | None = None,
    scores: dict[str, Any] | None = None,
) -> Path:
    interaction = load_trace(trace)
    target = output or trace.with_suffix(".html")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        render_html(interaction, title=title, scores=scores),
        encoding="utf-8",
    )
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="agent trace JSONL")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--title")
    args = parser.parse_args()

    if not args.trace.is_file():
        print(f"no such trace: {args.trace}", file=sys.stderr)
        return 1

    target = render_file(args.trace, args.output, title=args.title)
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
