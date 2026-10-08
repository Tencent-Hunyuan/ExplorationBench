#!/usr/bin/env python3
"""Build and check the held-out task set of the AlienCode demo world.

This is the construction procedure the evaluation world also follows, run on
public data. Every task carries two programs:

  reference  correct in this world;
  naive      correct under the reference manual, i.e. what a system writes if
             it has not discovered the rules.

Expected outputs are never typed in: they are the output of the reference
program in the world. The build then checks that

  1. the reference prints, on every hidden call, what the task's intended
     function (a Python oracle) returns for the values the call really passes;
  2. the naive program does the same under the manual's semantics but fails in
     the world, so the task separates discovering the rules from not doing so;
  3. every rule a task lists is relevant to it: switching that rule back to the
     manual changes the outcome of the reference or of the naive program
     (targets that already behave as the manual says are exempt);
  4. every discovery target is listed by at least one task;
  5. the worked examples and seed tasks in world_data.py still show the
     outputs stored there.

Only then does it write eval_set_v2.json, which carries no programs, and
reference_solutions.json, which the release ships as worked answers.

    python3 -m sandboxes.code.build_world
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

from sandboxes.code import engine as pack
from sandboxes.code import world_data
from sandboxes.code.execution import alien_exec

HERE = Path(__file__).resolve().parent
ALL_RULES = [spec["key"] for spec in pack.ALIEN_RULE_SPECS]
RED_HERRINGS = {spec["key"] for spec in pack.ALIEN_RULE_SPECS if spec["identity"]}


def _code(prompt: str, call: str) -> str:
    return f"{prompt}\n测试: {call}"


TASKS = [
    {
        "id": "Q01", "difficulty": "engineer", "question_type": "predict",
        "program": "SET xs AS STRAND(4, 8, 15)\nEMIT(PLUCK(xs, -7), GAUGE(xs))",
        "rules_tested": ["int_offset", "pluck_next", "gauge_double", "strand_identity"],
    },
    {
        "id": "Q02", "difficulty": "engineer",
        "spec": lambda xs: len(xs),
        "prompt": _code("写一个函数 CRAFT count(xs)，返回 xs 的真实元素个数。",
                        "EMIT(count(STRAND(4, 4, 4)))"),
        "calls": ["EMIT(count(STRAND()))", "EMIT(count(STRAND(1)))",
                  "EMIT(count(STRAND(2, 7)))", "EMIT(count(STRAND(5, 5, 5, 5, 5)))"],
        "reference": "CRAFT count(xs):\n    DELIVER HALVE(GAUGE(xs), -5)",
        "naive": "CRAFT count(xs):\n    DELIVER GAUGE(xs)",
        "rules_tested": ["gauge_double", "int_offset"],
    },
    {
        "id": "Q03", "difficulty": "engineer",
        "spec": lambda xs: xs[1],
        "prompt": _code("写一个函数 CRAFT second(xs)，返回 xs 中下标为 1 的元素（即第二个元素）。",
                        "EMIT(second(STRAND(4, 9, 2)))"),
        "calls": ["EMIT(second(STRAND(1, 2)))", "EMIT(second(STRAND(8, 3, 6, 1)))",
                  "EMIT(second(STRAND(-7, 0, -7)))", "EMIT(second(STRAND(5, 5, 9)))"],
        "reference": "CRAFT second(xs):\n    DELIVER PLUCK(xs, -7)",
        "naive": "CRAFT second(xs):\n    DELIVER PLUCK(xs, 1)",
        "rules_tested": ["pluck_next", "int_offset"],
    },
    {
        "id": "Q04", "difficulty": "algorithm_1",
        "spec": lambda xs, t: sum(1 for x in xs if x < t),
        "prompt": _code("写一个函数 CRAFT count_below(xs, t)，返回 xs 中严格小于 t 的元素个数。",
                        "EMIT(count_below(STRAND(1, 5, 3, 5), 5))"),
        "calls": ["EMIT(count_below(STRAND(4, 4, 4), 4))", "EMIT(count_below(STRAND(1, 2, 3), 9))",
                  "EMIT(count_below(STRAND(), 3))", "EMIT(count_below(STRAND(6, 2, 6, 1, 7), 6))"],
        "reference": ("CRAFT count_below(xs, t):\n    SET c AS -7\n    SWEEP x IN xs:\n"
                      "        UPON OVER(t, x):\n            SET c AS SHATTER(c, -6)\n    DELIVER c"),
        "naive": ("CRAFT count_below(xs, t):\n    SET c AS 0\n    SWEEP x IN xs:\n"
                  "        UPON UNDER(x, t):\n            SET c AS SHATTER(c, 1)\n    DELIVER c"),
        "rules_tested": ["under_le", "int_offset"],
    },
    {
        "id": "Q05", "difficulty": "algorithm_2",
        "spec": lambda xs: xs[::-1],
        "prompt": _code("写一个函数 CRAFT reverse_list(xs)，返回一个新的 STRAND，元素顺序与 xs 相反。",
                        "EMIT(reverse_list(STRAND(1, 2, 3)))"),
        "calls": ["EMIT(reverse_list(STRAND()))", "EMIT(reverse_list(STRAND(4)))",
                  "EMIT(reverse_list(STRAND(9, 8)))", "EMIT(reverse_list(STRAND(1, 3, 5, 7)))"],
        "reference": ("CRAFT reverse_list(xs):\n    SET out AS STRAND()\n"
                      "    SET n AS HALVE(GAUGE(xs), -5)\n    SWEEP k IN EXTENT(n):\n"
                      "        ANNEX(out, PLUCK(xs, WEAVE(WEAVE(n, k), -5)))\n    DELIVER out"),
        "naive": ("CRAFT reverse_list(xs):\n    SET out AS STRAND()\n    SET n AS GAUGE(xs)\n"
                  "    SWEEP k IN EXTENT(n):\n"
                  "        ANNEX(out, PLUCK(xs, WEAVE(WEAVE(n, k), 1)))\n    DELIVER out"),
        "rules_tested": ["gauge_double", "pluck_next", "int_offset", "strand_identity"],
    },
]


def run(program: str, disabled=()) -> str:
    pack.DISABLED.clear()
    pack.DISABLED.update(disabled)
    try:
        return alien_exec(program)
    finally:
        pack.DISABLED.clear()


def outcome(task: dict, program: str | None, disabled=()) -> tuple[str, ...]:
    if task["question_type"] == "predict":
        return (run(task["program"], disabled),)
    return tuple(run(f"{program}\n{call}", disabled) for call in task["calls"])


def arguments(call: str, disabled=()) -> tuple:
    """The runtime values a hidden call passes, read back through KNOT."""
    inner = call[len("EMIT("):-1]
    args = inner[inner.index("(") + 1:-1]
    return ast.literal_eval(run(f"EMIT(KNOT({args},))", disabled))


def intended(task: dict, disabled=()) -> tuple[str, ...]:
    return tuple(str(task["spec"](*arguments(call, disabled))) for call in task["calls"])


def check_task(task: dict) -> list[str]:
    task.setdefault("question_type", "code")
    problems = []
    unknown = set(task["rules_tested"]) - set(ALL_RULES)
    if unknown:
        problems.append(f"unknown rules {sorted(unknown)}")
    if task["question_type"] == "predict":
        world = outcome(task, None)
        if world == outcome(task, None, ALL_RULES):
            problems.append("prediction is the same under the manual")
        relevant = {r for r in ALL_RULES if outcome(task, None, [r]) != world}
    else:
        world = outcome(task, task["reference"])
        if world != intended(task):
            problems.append(f"reference is wrong in the world: {world} != {intended(task)}")
        naive_world = outcome(task, task["naive"])
        if outcome(task, task["naive"], ALL_RULES) != intended(task, ALL_RULES):
            problems.append("naive program is wrong under the manual")
        if naive_world == world:
            problems.append("naive program passes in the world")
        relevant = {r for r in ALL_RULES
                    if outcome(task, task["reference"], [r]) != world
                    or outcome(task, task["naive"], [r]) != naive_world}
    stated = set(task["rules_tested"]) - RED_HERRINGS
    if stated - relevant:
        problems.append(f"listed but irrelevant: {sorted(stated - relevant)}")
    task["_relevant"] = sorted(relevant)
    task["_expected"] = world
    return problems


def check_world_data() -> list[str]:
    problems = []
    for seed in world_data.SEED_TASKS:
        code = world_data.LOCKED_SEED_CODES[seed["id"]]
        if run(code) != seed["expected"]:
            problems.append(f"seed {seed['id']}: stored expected {seed['expected']!r} != {run(code)!r}")
        if run(code, ALL_RULES) != seed["real_world_answer"]:
            problems.append(f"seed {seed['id']}: stored manual answer is stale")
    for demo in world_data.DEMOS:
        relevant = {r for r in ALL_RULES if run(demo["code"], [r]) != run(demo["code"])}
        stated = set(demo["rules"]) - RED_HERRINGS
        if stated - relevant:
            problems.append(f"demo {demo['id']}: lists irrelevant {sorted(stated - relevant)}")
    ids = {spec["id"] for spec in pack.ALIEN_RULE_SPECS}
    if ids != set(world_data.GROUND_TRUTH_SEXPR):
        problems.append("GROUND_TRUTH_SEXPR ids differ from ALIEN_RULE_SPECS")
    return problems


def main() -> int:
    problems = {}
    for task in TASKS:
        found = check_task(task)
        if found:
            problems[task["id"]] = found
    covered = {r for task in TASKS for r in task["rules_tested"]}
    uncovered = set(ALL_RULES) - covered
    world = check_world_data()
    for task in TASKS:
        print(f"{task['id']} {task['difficulty']:<12} relevant={task['_relevant']}")
    if uncovered:
        print(f"targets no task lists: {sorted(uncovered)}")
    for task_id, found in problems.items():
        print(f"{task_id}: " + "; ".join(found))
    for line in world:
        print(line)
    if problems or uncovered or world:
        return 1
    rows, solutions = [], {}
    for task in TASKS:
        row = {"id": task["id"], "difficulty": task["difficulty"],
               "question_type": task["question_type"], "rules_tested": task["rules_tested"]}
        if task["question_type"] == "predict":
            row["prompt"] = f"预测以下代码的输出：\n```aliencode\n{task['program']}\n```"
            row["expected"] = task["_expected"][0]
            # The grader runs this to get the output a prediction is compared with.
            row["given_code"] = task["program"]
        else:
            row["prompt"] = task["prompt"]
            row["expected"] = None
            row["test_cases"] = [{"call": c, "expected": e}
                                 for c, e in zip(task["calls"], task["_expected"])]
            row["checker"] = "exact"
            solutions[task["id"]] = {"reference": task["reference"], "naive": task["naive"]}
        rows.append(row)
    (HERE / "eval_set_v2.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n")
    (HERE / "reference_solutions.json").write_text(
        json.dumps(solutions, ensure_ascii=False, indent=1) + "\n")
    print(f"{len(rows)} tasks, {sum(len(r.get('test_cases', [])) for r in rows)} hidden calls, "
          f"all {len(ALL_RULES)} discovery targets covered -> eval_set_v2.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
