#!/usr/bin/env python3
"""Tell apart evaluation results the models earned from ones the API spoiled.

Every call attempt lands in the trace as an ``api_exchange`` record, and a call
that eventually produced an answer also lands as a ``trajectory`` record. A
transport retry keeps the exchange id, but a harness-level empty/tool retry is
a new exchange whose label gains a ``retry`` suffix. A failed attempt is
therefore recovered when either its exchange id appears in a trajectory or a
successful trajectory has the same logical label after that suffix is removed.
Anything else is a call that was lost for good.

Traces reach a gigabyte, and the failure records embed whole request bodies, so
this reads line by line and pulls the handful of fields it needs with regexes
rather than parsing every line as JSON.

    python3 scripts/audit_api_failures.py                    # every trace on disk
    python3 scripts/audit_api_failures.py --sandbox code
    python3 scripts/audit_api_failures.py --trace path.jsonl -o report.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEV = Path(__file__).resolve().parent.parent
LOGS = DEV.parent / "logs"
ARCHIVE = Path("logs_archive")

# Where traces have lived across the project's life; later runs keep code and
# logic apart, but the logic harness wrote its traces into results/ for a while.
TRACE_DIRS = {
    "code": [LOGS / "code" / "traces"],
    "logic": [LOGS / "logic" / "traces", LOGS / "logic" / "results"],
}

_EXCHANGE = re.compile(r'"exchange_id":\s*"([^"]+)"')
_LABEL = re.compile(r'"label":\s*"([^"]*)"')
_HTTP = re.compile(r'"http_status":\s*(\d+|null)')
_ATTEMPT = re.compile(r'"attempt":\s*(\d+)')
_MODEL = re.compile(r'"model":\s*"([^"]+)"')
_ERROR_TYPE = re.compile(r'"error":\s*\{\s*"type":\s*"([^"]+)",\s*"message":\s*"(.{0,400})')
_RETRY_SUFFIX = re.compile(
    r"(?:\s+retry(?:\s+\d+)?|\s+\(retry[^)]*\))$",
    re.IGNORECASE,
)
_STOP_REASON = re.compile(r'"stop_reason":\s*"([^"]+)"')
# Traces are written compactly, but hand-made and older ones carry spaces after
# the colons, so the line tests have to tolerate both.
_RECORD_TYPE = re.compile(r'"record_type":\s*"([a-z_]+)"')
_FAILED = re.compile(r'"status":\s*"failed"')

# Which phases a lost call actually costs us. Explore probes are cheap -- the
# model gets more rounds -- but a summary or a held-out test is graded once.
GRADED_LABEL = re.compile(r"(Summary|Test|T\d)", re.IGNORECASE)
_HTTP_IN_MESSAGE = re.compile(r"^HTTP (\d{3})")


def _trace_stem(path: Path) -> str:
    """Trace basename without either ``.jsonl`` or ``.jsonl.gz``."""

    name = path.name
    if name.endswith(".gz"):
        name = name[:-3]
    if name.endswith(".jsonl"):
        name = name[:-6]
    return name


def _open_trace(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, mode="rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def _available_trace(path: Path) -> Path | None:
    if path.exists():
        return path
    name = path.name
    sandbox = (
        "code" if "agent_trace_alien_code_" in name
        else "logic" if "agent_trace_alien_logic_" in name
        else None
    )
    if sandbox is None:
        return None
    archived = ARCHIVE / sandbox / "traces" / (
        name if name.endswith(".gz") else name + ".gz"
    )
    return archived if archived.exists() else None


def logical_label(label: str) -> str:
    """The request label shared by an initial call and its harness retry."""

    previous = label.strip()
    while True:
        current = _RETRY_SUFFIX.sub("", previous).strip()
        if current == previous:
            return current
        previous = current


def classify(
    http_status: int | None, error_type: str, message: str
) -> str:
    """Bucket one failure by what actually went wrong upstream.

    The status is passed in where the caller has it; readers that only kept the
    error text can leave it out, since the message opens with it.
    """

    lowered = message.lower()
    if http_status is None:
        sniffed = _HTTP_IN_MESSAGE.match(message.strip())
        if sniffed:
            http_status = int(sniffed.group(1))
    if error_type == "AgentEmptyCompletionError":
        return "empty_completion"
    if "无可用账号" in message or "no available account" in lowered:
        return "account_pool"
    if http_status in (408, 504) or "timeout" in lowered or "deadline" in lowered:
        return "timeout"
    if http_status == 429 or "rate limit" in lowered:
        return "rate_limit"
    if http_status in (401, 403):
        return "auth"
    if http_status == 400:
        return "bad_request"
    if http_status in (500, 502, 503, 529):
        return "server"
    if http_status is None:
        return "transport"
    return f"http_{http_status}"


def scan(path: Path) -> dict:
    """Walk one trace and summarise what the API did to that run."""

    attempts = 0
    trajectories = 0
    failures: list[dict] = []
    seen_exchanges: set[str] = set()
    successful_labels: set[str] = set()
    model = ""
    refusals = 0

    with _open_trace(path) as handle:
        for line in handle:
            kind = _RECORD_TYPE.search(line[:400])
            if kind is None:
                continue
            is_exchange = kind.group(1) == "api_exchange"
            is_trajectory = kind.group(1) == "trajectory"
            if not is_exchange and not is_trajectory:
                continue
            if is_trajectory:
                trajectories += 1
                found = _EXCHANGE.search(line)
                if found:
                    seen_exchanges.add(found.group(1))
                label = _LABEL.search(line)
                if label and label.group(1):
                    successful_labels.add(logical_label(label.group(1)))
                stop = _STOP_REASON.search(line)
                if stop and stop.group(1) == "refusal":
                    refusals += 1
                continue

            attempts += 1
            if not model:
                found = _MODEL.search(line)
                if found:
                    model = found.group(1)
            if not _FAILED.search(line):
                continue

            http_raw = _HTTP.search(line)
            http_status = (
                None
                if not http_raw or http_raw.group(1) == "null"
                else int(http_raw.group(1))
            )
            error = _ERROR_TYPE.search(line)
            error_type = error.group(1) if error else ""
            message = error.group(2) if error else ""
            exchange = _EXCHANGE.search(line)
            label = _LABEL.search(line)
            attempt = _ATTEMPT.search(line)
            failures.append({
                "exchange_id": exchange.group(1) if exchange else "",
                "label": label.group(1) if label else "",
                "attempt": int(attempt.group(1)) if attempt else -1,
                "http_status": http_status,
                "kind": classify(http_status, error_type, message),
                "message": message[:200],
            })

    lost: dict[str, dict] = {}
    for failure in failures:
        exchange = failure["exchange_id"]
        if exchange and exchange in seen_exchanges:
            continue
        if (
            failure["label"]
            and logical_label(failure["label"]) in successful_labels
        ):
            continue
        # Keep the last attempt for a lost call: that is the error it died on.
        current = lost.get(exchange)
        if current is None or failure["attempt"] >= current["attempt"]:
            lost[exchange] = failure

    # A question a redo dealt with is no longer missing from the score even
    # though this trace only records it failing, so it is reported apart from
    # the ones still missing -- otherwise a repaired run looks damaged forever.
    scores = find_scores(path)[0]
    repaired_ids = repaired_questions(scores)
    excluded_ids = excluded_questions(scores)
    lost_calls = []
    repaired_calls = []
    excluded_calls = []
    for failure in lost.values():
        found = question_of(failure["label"])
        if found and found in repaired_ids:
            repaired_calls.append(failure)
        elif found and found in excluded_ids:
            excluded_calls.append(failure)
        else:
            lost_calls.append(failure)
    graded_lost = [f for f in lost_calls if GRADED_LABEL.search(f["label"])]
    return {
        "trace": relative(path),
        "run": run_of(path),
        "model": model,
        "attempts": attempts,
        "answered_calls": trajectories,
        "failed_attempts": len(failures),
        "retried_and_recovered": len(failures) - len(lost),
        "lost_calls": len(lost_calls),
        "lost_graded_calls": len(graded_lost),
        "repaired_calls": len(repaired_calls),
        "repaired_labels": sorted({f["label"] for f in repaired_calls if f["label"]}),
        "excluded_calls": len(excluded_calls),
        "excluded_labels": sorted({f["label"] for f in excluded_calls if f["label"]}),
        "refusals": refusals,
        "failure_kinds": dict(Counter(f["kind"] for f in failures).most_common()),
        "lost_kinds": dict(Counter(f["kind"] for f in lost_calls).most_common()),
        "lost_labels": sorted({f["label"] for f in lost_calls if f["label"]}),
        "lost_graded_labels": sorted({f["label"] for f in graded_lost if f["label"]}),
        "sample_errors": [
            f"{f['kind']} {f['label']}: {f['message'][:120]}"
            for f in lost_calls[:5]
        ],
    }


# Runs name their artifacts after the run id alone; traces from before that
# carry the launch timestamp as well, and both still have to pair with scores.
TRACE_STEM = re.compile(
    r"^agent_trace_(?P<sandbox>alien_code|alien_logic)"
    r"(?:_(?P<stamp>\d{8}_\d{6}))?_(?P<run>.+)$"
)

# Held-out attempts are labelled "M4 Test A56"; the question id is what a redo
# reports having recovered, so it is the join between the two.
# "M4 Test A56" in AlienCode, "Mpre T13" / "M6 T67 retry" in AlienLogic.
QUESTION_IN_LABEL = re.compile(r"Test (\w+)|\b(T\d+)\b")


def question_of(label: str) -> str | None:
    """The held-out question a call label names, in either sandbox's wording."""

    match = QUESTION_IN_LABEL.search(label or "")
    if not match:
        return None
    return match.group(1) or match.group(2)


def find_scores(trace: Path) -> tuple[dict | None, Path | None]:
    """Locate the scored results that go with a trace, if the run finished.

    The trace and the score file are written by the same run and share a tail,
    so the tail is enough to pair them -- except that the two sandboxes lay
    them out differently: AlienCode drops the timestamp from the score file and
    keeps traces in their own directory, AlienLogic keeps the timestamp and
    puts both side by side. Try both shapes rather than encode which is which.
    A ``_redone`` file wins when present: it is the same run rescored after
    questions that failed on infrastructure errors were retried, so it
    reflects the model rather than the outage.
    """

    match = TRACE_STEM.match(_trace_stem(trace))
    if not match:
        return None, None
    sandbox, stamp, run = (
        match.group("sandbox"), match.group("stamp"), match.group("run"),
    )
    nearby_results = (
        trace.parent if trace.parent.name == "results"
        else trace.parent.parent / "results"
    )
    sandbox_dir = "code" if sandbox == "alien_code" else "logic"
    result_dirs = [nearby_results, LOGS / sandbox_dir / "results"]
    stems = [f"eval_results_{sandbox}_{run}"]
    if stamp:
        stems.append(f"eval_results_{sandbox}_{stamp}_{run}")
    for results in dict.fromkeys(result_dirs):
        for stem in stems:
            for candidate in (
                results / f"{stem}_redone.json", results / f"{stem}.json",
            ):
                if not candidate.is_file():
                    continue
                try:
                    return (
                        json.loads(candidate.read_text(encoding="utf-8")),
                        candidate,
                    )
                except (OSError, json.JSONDecodeError):
                    continue
    return None, None


def resolve_snapshot(recorded: str | None, results_path: Path) -> str | None:
    """A snapshot path that still points at the file, or None.

    Runs record absolute paths, and the logic traces were later moved out of
    ``results/`` into their own ``traces/`` directory, taking their snapshot
    directories along. The recorded path is therefore stale on older runs even
    though the file is still on disk, so fall back to the name.
    """

    if not recorded:
        return None
    if Path(recorded).is_file():
        return recorded
    name = Path(recorded).name
    root = results_path.resolve().parent.parent
    for directory in (root / "traces", root / "results"):
        if not directory.is_dir():
            continue
        for found in directory.glob(f"*_snapshots/{name}"):
            return str(found)
    return None


def _redo_attempts(scores: dict | None):
    for milestone in (scores or {}).get("milestones", []):
        history = milestone.get("redo_history") or (
            [milestone["redo"]] if milestone.get("redo") else []
        )
        yield from history


def repaired_questions(scores: dict | None) -> set[str]:
    """Held-out questions a later redo re-asked and got an answer for."""

    return {
        task_id
        for attempt in _redo_attempts(scores)
        for task_id in attempt.get("recovered", [])
    }


def excluded_questions(scores: dict | None) -> set[str]:
    """Questions dropped from a milestone's denominator instead of re-asked.

    Where the session a milestone answered from is gone, re-asking would hand
    the model what it learned afterwards, so the honest repair is to score the
    milestone over the questions it did answer. Those are no longer dragging the
    score down either, so they are not damage the number still carries.
    """

    return {
        task_id
        for attempt in _redo_attempts(scores)
        if attempt.get("mode") == "exclude"
        for task_id in attempt.get("still_failed", [])
    }


def relative(path: Path) -> str:
    """Path as written in the report, repo-relative when it lives here."""

    resolved = path.resolve()
    root = LOGS.parent
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def run_of(path: Path) -> str:
    """The run id a trace belongs to, from its filename."""

    stem = _trace_stem(path)
    for prefix in ("agent_trace_alien_code_", "agent_trace_alien_logic_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def collect(args) -> list[Path]:
    if args.trace:
        return [Path(p) for p in args.trace]
    sandboxes = [args.sandbox] if args.sandbox else list(TRACE_DIRS)
    found: list[Path] = []
    for sandbox in sandboxes:
        for directory in TRACE_DIRS[sandbox]:
            if directory.is_dir():
                found.extend(sorted(directory.glob("agent_trace_*.jsonl")))
                found.extend(sorted(directory.glob("agent_trace_*.jsonl.gz")))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sandbox", choices=sorted(TRACE_DIRS))
    parser.add_argument("--trace", action="append", help="audit these traces only")
    parser.add_argument("-o", "--output", help="write the full report as JSON")
    parser.add_argument(
        "--min-lost",
        type=int,
        default=1,
        help="only print runs that lost at least this many calls",
    )
    args = parser.parse_args()

    traces = collect(args)
    if not traces:
        print("no traces found", file=sys.stderr)
        return 1

    print(f"scanning {len(traces)} trace(s)\n", flush=True)
    reports = []
    for requested in traces:
        path = _available_trace(requested)
        if path is None:
            print(f"    skipped: trace disappeared: {requested}", file=sys.stderr)
            continue
        try:
            size_mb = path.stat().st_size / 1e6
            print(f"  … {path.name} ({size_mb:,.0f} MB)", flush=True)
            reports.append(scan(path))
        except OSError as exc:
            retry = _available_trace(requested)
            if retry is not None and retry != path:
                try:
                    reports.append(scan(retry))
                    continue
                except OSError as retry_exc:
                    exc = retry_exc
            print(f"    skipped: {exc}", file=sys.stderr)

    reports.sort(key=lambda r: (-r["lost_graded_calls"], -r["lost_calls"]))

    print(f"\n{'run':46s} {'calls':>7s} {'fail':>6s} {'saved':>6s} {'lost':>5s} {'graded':>7s}")
    print("-" * 82)
    for report in reports:
        if report["lost_calls"] < args.min_lost:
            continue
        print(
            f"{report['run'][:46]:46s} {report['attempts']:7,d} "
            f"{report['failed_attempts']:6,d} {report['retried_and_recovered']:6,d} "
            f"{report['lost_calls']:5,d} {report['lost_graded_calls']:7,d}"
        )
        kinds = ", ".join(f"{k}={v}" for k, v in report["lost_kinds"].items())
        if kinds:
            print(f"{'':46s} lost: {kinds}")
        if report["lost_graded_labels"]:
            shown = ", ".join(report["lost_graded_labels"][:8])
            more = len(report["lost_graded_labels"]) - 8
            print(f"{'':46s} graded lost: {shown}{f' (+{more})' if more > 0 else ''}")

    clean = [r for r in reports if r["lost_calls"] == 0]
    spoiled = [r for r in reports if r["lost_graded_calls"] > 0]
    print(
        f"\n{len(reports)} run(s): {len(clean)} lost nothing, "
        f"{len(spoiled)} lost a graded call"
    )
    total_recovered = sum(r["retried_and_recovered"] for r in reports)
    print(f"retries rescued {total_recovered:,} attempt(s) across all runs")

    if args.output:
        Path(args.output).write_text(
            json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"report {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
