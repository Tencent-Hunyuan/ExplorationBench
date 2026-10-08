#!/usr/bin/env python3
"""Turn one evaluation run into a readable pair of artefacts.

An AlienCode run leaves a call-by-call trace plus a snapshot per session.
Neither is something a person reads. This produces the two things that are
actually reviewable afterwards:

  * ``<run>.run.json`` -- the whole run in one file: every exchange grouped by
    the stage of the eval it belongs to, with reasoning, tool calls, tool
    results and usage beside the prompt and answer, plus the scored milestones
    and a count of what the API lost. Nothing else needs reading to review a
    run, and ``tools/run_viewer.html`` opens it directly. It keeps the
    provider's continuation signatures verbatim, so treat it as
    credential-adjacent and don't paste it around.
  * ``<run>.html`` -- the same run as a self-contained, redacted replay,
    which is the one that is safe to share.

The graded questions dominate a run by volume (five milestones of ninety
closed-book questions each), so they are summarised rather than inlined
unless asked for; the probing and the inferred rules are the part worth
reading.

    python scripts/export_run.py logs/code/traces/agent_trace_*.jsonl
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import re
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from scripts.render_interaction import (  # noqa: E402
    Interaction,
    Phase,
    Turn,
    _line_chart,
    _STYLE,
    group_turns,
    load_trace,
    render_file,
    trace_stem,
)
from scripts.audit_api_failures import (  # noqa: E402
    GRADED_LABEL as _GRADED_LABEL,
    TRACE_STEM as _TRACE_STEM,
    classify as classify_failure,
    excluded_questions,
    find_scores,
    question_of,
    repaired_questions,
)


def _turn_dict(turn: Turn, *, full: bool) -> dict[str, Any]:
    """One exchange as the model saw it, in wire order."""

    record: dict[str, Any] = {
        "label": turn.label,
        "session_id": turn.session_id,
        "recorded_at": turn.recorded_at,
        "user": turn.prompt,
    }
    if turn.parent_session_id:
        record["parent_session_id"] = turn.parent_session_id
    # On the Responses protocol the reasoning stays server-side and these ids
    # are what carries it between calls, so they are this archive's equivalent
    # of a thinking signature: without them you cannot tell from the file
    # whether a turn continued the previous context or started over.
    if turn.previous_response_id:
        record["previous_response_id"] = turn.previous_response_id
    if turn.response_id:
        record["response_id"] = turn.response_id
    if turn.reasoning:
        # Kept verbatim, continuation signatures included: this file is the
        # archive you audit a run from, and the signature is the evidence the
        # reasoning context actually survived the round trip. The HTML replay
        # is the redacted view meant for sharing.
        record["reasoning"] = turn.reasoning
        record["signed_reasoning_artifacts"] = turn.returned_signatures
    if turn.tool_calls:
        record["tool_calls"] = turn.tool_calls
    if turn.tool_results:
        record["tool_results"] = turn.tool_results
    record["assistant"] = turn.response
    if turn.stop_reason:
        record["stop_reason"] = turn.stop_reason
    if turn.stop_details:
        record["stop_details"] = turn.stop_details
    record["usage"] = turn.usage
    if not full:
        for key in ("reasoning", "tool_calls", "tool_results"):
            record.pop(key, None)
    return record


def _phase_dict(phase: Phase, *, full: bool) -> dict[str, Any]:
    usage = phase.usage()
    billed = (
        usage["input_tokens"]
        + usage["cache_creation_input_tokens"]
        + usage["cache_read_input_tokens"]
    )
    return {
        "phase": phase.key,
        "title": phase.title,
        "calls": len(phase.turns),
        "refusals": phase.refusals,
        "usage": {
            **usage,
            "billed_input_tokens": billed,
            "cache_hit_ratio": (
                round(usage["cache_read_input_tokens"] / billed, 4)
                if billed else 0.0
            ),
        },
        "turns": [
            _turn_dict(turn, full=full) for turn in phase.turns
        ],
    }


# Fields of the results file that only repeat what the turns already hold, or
# that are large enough to make the archive unopenable in a browser.
_SCORE_DROP = frozenset({
    "snapshots", "seed_records", "explore_records", "token_log",
})


def _score_block(scores: dict[str, Any] | None, source: Path | None) -> dict[str, Any]:
    """The scored side of a run, trimmed to what a reader or a chart needs."""

    if not scores:
        return {}
    block = {
        key: value for key, value in scores.items() if key not in _SCORE_DROP
    }
    if source is not None:
        block["results_file"] = source.name
    return block


def build_export(
    interaction: Interaction,
    *,
    full_tests: bool = False,
    scores: dict[str, Any] | None = None,
    scores_path: Path | None = None,
) -> dict[str, Any]:
    """The whole run as one self-contained document.

    Messages, scores and usage in a single file, because the two questions
    asked of a finished run -- what did the model do, and did it get better --
    are otherwise answered from two places that have to be paired up by
    filename. Anything reading this needs no other file.
    """

    phases = group_turns(interaction.turns)
    export = {
        "schema": "self-evolve-run/1",
        "model": interaction.model,
        "provider": interaction.provider,
        "trace": str(interaction.source),
        "totals": interaction.totals(),
        "api_health": api_health(
            interaction, repaired_questions(scores), excluded_questions(scores)
        ),
        "scores": _score_block(scores, scores_path),
        "phases": [
            _phase_dict(
                phase,
                # Held-out answers stay terse unless asked for: there are
                # hundreds of them and each is a single question.
                full=full_tests or not phase.key.startswith("tests-"),
            )
            for phase in phases
        ],
        "failed_attempts": [
            {
                "label": turn.label,
                "attempt": turn.attempt,
                "error": turn.error,
            }
            for turn in interaction.failures
        ],
        "parse_errors": interaction.parse_errors,
    }
    return export


def api_health(
    interaction: Interaction,
    repaired: set[str] | None = None,
    excluded: set[str] | None = None,
) -> dict[str, Any]:
    """How much of this run the API cost us.

    A failed attempt that was retried into an answer only cost time. One whose
    exchange never produced a turn is a question the model never got to answer,
    and a score computed over those is not the model's score -- so they are
    counted apart, and the graded ones are named.

    A question a redo dealt with is no longer missing from the score, so it is
    reported apart from the lost ones -- re-asked if an answer was recovered,
    excluded if the milestone was instead scored over what it did answer.
    Otherwise a run stays flagged as untrustworthy long after the damage was
    accounted for.
    """

    answered = {turn.exchange_id for turn in interaction.turns if turn.exchange_id}
    # A retry that the client performs inside one exchange reuses its id, so
    # matching on the id alone finds it. A retry the harness performs -- it
    # re-dispatches the whole question -- opens a fresh exchange, and the
    # failed one then never produces a turn however well the re-ask went. One
    # run reported four lost graded calls that way while its result file held
    # real answers for all four, so the question has to be checked too.
    answered_questions = {turn.label for turn in interaction.turns if turn.label}
    lost: dict[str, Any] = {}
    kinds: dict[str, int] = {}
    for failure in interaction.failures:
        error = failure.error if isinstance(failure.error, dict) else {}
        message = str(error.get("message", ""))
        kind = classify_failure(
            failure.http_status, str(error.get("type", "")), message
        )
        kinds[kind] = kinds.get(kind, 0) + 1
        exchange = failure.exchange_id
        if exchange and exchange in answered:
            continue
        if failure.label and failure.label in answered_questions:
            continue
        previous = lost.get(exchange or failure.label)
        if previous is None or failure.attempt >= previous["attempt"]:
            lost[exchange or failure.label] = {
                "label": failure.label,
                "attempt": failure.attempt,
                "kind": kind,
                "message": message[:300],
            }

    repaired = repaired or set()
    excluded = excluded or set()
    lost_calls = []
    repaired_calls = []
    excluded_calls = []
    for call in lost.values():
        found = question_of(call["label"])
        if found and found in repaired:
            repaired_calls.append(call)
        elif found and found in excluded:
            excluded_calls.append(call)
        else:
            lost_calls.append(call)
    graded = [call for call in lost_calls if _GRADED_LABEL.search(call["label"])]
    return {
        "attempts": len(interaction.turns) + len(interaction.failures),
        "failed_attempts": len(interaction.failures),
        "retried_and_recovered": len(interaction.failures) - len(lost.values()),
        "lost_calls": len(lost_calls),
        "lost_graded_calls": len(graded),
        "repaired_calls": len(repaired_calls),
        "excluded_calls": len(excluded_calls),
        "failure_kinds": kinds,
        "lost": lost_calls,
        "repaired": repaired_calls,
        "excluded": excluded_calls,
    }


SANDBOX_TITLES = {
    "alien_code": "AlienCode",
    "alien_logic": "AlienLogic",
}


def sandbox_of(trace: Path) -> str:
    match = _TRACE_STEM.match(trace_stem(trace))
    return match.group("sandbox") if match else "alien_code"


def export_one(
    trace: Path,
    out_dir: Path,
    *,
    full_tests: bool,
    title: str | None,
) -> dict[str, Any] | None:
    interaction = load_trace(trace)
    if not interaction.turns:
        print(f"trace has no completed turns: {trace}", file=sys.stderr)
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = trace_stem(trace)

    # The scored milestones drive the charts; a run that crashed before scoring
    # still exports, just without them.
    scores, scores_path = find_scores(trace)
    export = build_export(
        interaction,
        full_tests=full_tests,
        scores=scores,
        scores_path=scores_path,
    )
    history_path = out_dir / f"{stem}.run.json"
    history_path.write_text(
        json.dumps(export, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    sandbox = SANDBOX_TITLES[sandbox_of(trace)]
    html_path = render_file(
        trace,
        out_dir / f"{stem}.html",
        title=title or f"{interaction.model} — {sandbox} 评测回放",
        scores=scores,
    )

    totals = interaction.totals()
    print(f"\nmodel      {interaction.model}  ({stem})")
    print(f"calls      {totals['calls']}  "
          f"in={totals['effective_input_tokens']:,} "
          f"out={totals['output_tokens']:,} "
          f"cache_hit={totals['cache_hit_ratio'] * 100:.1f}%")
    for phase in group_turns(interaction.turns):
        note = f"{len(phase.turns):>4} calls"
        if phase.refusals:
            note += f"  {phase.refusals} refused"
        print(f"  {phase.title:<34s} {note}")
    health = export["api_health"]
    # Two populations, printed apart: every attempt the API refused, and the
    # subset of them that never turned into an answer. Putting the breakdown
    # of the first beside the count of the second read as a contradiction --
    # "4 lost" next to kinds summing to 17.
    if health["failed_attempts"]:
        print(f"  {'API 失败尝试':<34s} {health['failed_attempts']} 次"
              f"（重试后答上 {health['retried_and_recovered']} 次）"
              f" {health['failure_kinds']}")
    if health["lost_calls"]:
        print(f"  {'其中始终没答上':<32s} {health['lost_calls']} 次"
              f"（计分 {health['lost_graded_calls']} 次）")
    source = scores_path.name if scores_path else "_redone"
    if health.get("repaired_calls"):
        print(f"  {'已补做':<34s} {health['repaired_calls']} 次"
              f"（分数取自 {source}）")
    if health.get("excluded_calls"):
        print(f"  {'已从分母剔除':<32s} {health['excluded_calls']} 次"
              f"（无当时会话，无法忠实重问；分数取自 {source}）")
    print(f"run json   {history_path}  (完整存档：message+分数+usage，含签名，勿外传)")
    print(f"replay     {html_path}  (已脱敏，可分享)")

    return {
        "trace": trace,
        "model": interaction.model,
        "totals": totals,
        "scores": scores,
    }


def print_comparison(runs: list[dict[str, Any]]) -> None:
    """Line the runs up on the numbers that decide which config is better."""

    scored = [run for run in runs if run.get("scores")]
    if len(runs) < 2:
        return

    print("\n" + "=" * 78)
    print("跨运行对比")
    print("=" * 78)
    header = (
        f"{'run':<22s} {'calls':>6s} {'in':>12s} {'out':>10s} {'cache':>7s}"
    )
    print(header)
    for run in runs:
        totals = run["totals"]
        name = _TRACE_STEM.match(
            run["trace"].with_suffix("").name
        )
        print(
            f"{(name.group('run') if name else run['trace'].stem):<22.22s} "
            f"{totals['calls']:>6,} "
            f"{totals['effective_input_tokens']:>12,} "
            f"{totals['output_tokens']:>10,} "
            f"{totals['cache_hit_ratio'] * 100:>6.1f}%"
        )

    if not scored:
        print("\n（尚无评分结果，评测可能仍在进行）")
        return

    print("\n每个 milestone 的成绩：")
    for run in scored:
        name = _TRACE_STEM.match(run["trace"].with_suffix("").name)
        label = name.group("run") if name else run["trace"].stem
        cells = []
        for milestone in run["scores"].get("milestones", []):
            # The two sandboxes score different things, so read whichever set
            # of keys this run actually has rather than the code one always.
            if "pass_rate" in milestone or "test_pass" in milestone:
                stage = milestone.get("milestone")
                cells.append(
                    f"M{stage}: 过"
                    f"{milestone.get('test_pass')}/{milestone.get('test_total')}"
                )
            else:
                cells.append(
                    f"M{milestone.get('milestone_idx')}: "
                    f"规则{milestone.get('found')}/{milestone.get('total')} "
                    f"题{milestone.get('test_correct')}/"
                    f"{milestone.get('test_total')}"
                )
        print(f"  {label:<22.22s} {'  |  '.join(cells) or '—'}")


def _run_label(trace: Path) -> str:
    match = _TRACE_STEM.match(trace.with_suffix("").name)
    return match.group("run") if match else trace.stem


def write_index(runs: list[dict[str, Any]], out_dir: Path) -> Path:
    """One page that lists every run and overlays their learning curves."""

    scored = [run for run in runs if run.get("scores")]
    width = max((len(run["scores"].get("milestones") or []) for run in scored),
                default=0)
    labels = [f"M{i}" for i in range(width)]

    def curve(run: dict[str, Any], key: str, total: str) -> list[float]:
        out = []
        for milestone in run["scores"].get("milestones") or []:
            denominator = milestone.get(total) or 0
            out.append(round(100 * (milestone.get(key) or 0) / denominator, 1)
                       if denominator else 0.0)
        return out + [0.0] * (width - len(out))

    body: list[str] = []
    if scored:
        body.extend([
            "<div class='panel'><div class='ptitle2'>闭卷正确率</div>",
            "<div class='chint'>每个里程碑答对的题数占 90 题的比例</div>",
            _line_chart(labels, [
                (_run_label(run["trace"]), curve(run, "test_correct", "test_total"))
                for run in scored
            ]),
            "<div class='ptitle2 gap'>规则发现率</div>",
            "<div class='chint'>归纳出的规则数占 31 条的比例</div>",
            _line_chart(labels, [
                (_run_label(run["trace"]), curve(run, "found", "total"))
                for run in scored
            ]),
            "</div>",
        ])

    rows = ["<div class='panel'><div class='ptitle2'>全部运行</div>",
            "<table class='mtable'><thead><tr><th>运行</th><th>调用</th>"
            "<th>输入</th><th>输出</th><th>缓存命中</th><th>最终规则</th>"
            "<th>最终答对</th><th>存档</th><th>回放</th></tr></thead><tbody>"]
    for run in runs:
        totals = run["totals"]
        label = _run_label(run["trace"])
        stem = run["trace"].with_suffix("").name
        milestones = (run.get("scores") or {}).get("milestones") or []
        last = milestones[-1] if milestones else {}
        rules = (f"{last.get('found')}/{last.get('total')}"
                 if milestones else "—")
        tests = (f"{last.get('test_correct')}/{last.get('test_total')}"
                 if milestones else "—")
        rows.append(
            f"<tr><td>{html_mod.escape(label)}</td>"
            f"<td>{totals['calls']:,}</td>"
            f"<td>{totals['effective_input_tokens']:,}</td>"
            f"<td>{totals['output_tokens']:,}</td>"
            f"<td>{totals['cache_hit_ratio'] * 100:.1f}%</td>"
            f"<td>{html_mod.escape(rules)}</td>"
            f"<td>{html_mod.escape(tests)}</td>"
            f"<td><a href='{stem}.run.json'>JSON</a></td>"
            f"<td><a href='{stem}.html'>打开</a></td></tr>"
        )
    rows.append("</tbody></table></div>")
    body.extend(rows)

    page = (
        "<!doctype html><html lang='zh'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>AlienCode 评测总览</title>"
        f"<style>{_STYLE}</style></head><body><div class='wrap'>"
        "<h1>AlienCode 评测总览</h1>"
        f"<div class='sub'>{len(runs)} 次运行 · 每次运行都有完整 messages 存档"
        "与可分享的行为回放</div>"
        + "".join(body) +
        "</div></body></html>"
    )
    target = out_dir / "index.html"
    target.write_text(page, encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trace", type=Path, nargs="+", help="agent trace JSONL(s)"
    )
    parser.add_argument("-o", "--out-dir", type=Path)
    parser.add_argument(
        "--full-tests",
        action="store_true",
        help="inline reasoning and tool traffic for every graded question",
    )
    parser.add_argument("--title")
    parser.add_argument(
        "--index",
        action="store_true",
        help="also write index.html linking every run, with curves overlaid",
    )
    args = parser.parse_args()

    runs: list[dict[str, Any]] = []
    for trace in args.trace:
        if not trace.is_file():
            print(f"no such trace: {trace}", file=sys.stderr)
            return 1
        run = export_one(
            trace,
            args.out_dir or trace.parent,
            full_tests=args.full_tests,
            title=args.title,
        )
        if run:
            runs.append(run)

    if not runs:
        return 1
    print_comparison(runs)
    if args.index:
        index = write_index(runs, args.out_dir or args.trace[0].parent)
        print(f"\nindex      {index}  (总览：曲线对比 + 两个文件的入口)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
