#!/usr/bin/env python3
"""Write the representative examples from the demo worlds in this package.

Everything below is computed from the released code and data -- outputs by the
AlienCode front end with the demo rule pack, verdicts by the AlienLogic
checker with the demo side conditions -- so the examples cannot disagree with
what a run would do.

    python3 examples/render.py        # writes examples/aliencode.md and alienlogic.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent / "explorationbench"
sys.path.insert(0, str(PACKAGE))

from sandboxes.code import engine as pack  # noqa: E402
from sandboxes.code import world_data  # noqa: E402
from sandboxes.code.execution import alien_exec  # noqa: E402
from sandboxes.logic import certify, episodes  # noqa: E402
from sandboxes.logic.engine import proof_diagnostic  # noqa: E402
from sandboxes.logic.oracle_rules import _ORACLE_SPECS  # noqa: E402
from sandboxes.logic.protocol_data import REJECTED_DEMOS  # noqa: E402

ALL_RULES = [spec["key"] for spec in pack.ALIEN_RULE_SPECS]
WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen")


def count(n: int) -> str:
    return WORDS[n] if n < len(WORDS) else str(n)


def manual(program: str) -> str:
    pack.DISABLED.update(ALL_RULES)
    try:
        return alien_exec(program)
    finally:
        pack.DISABLED.clear()


def fence(text: str, lang: str = "") -> str:
    return f"```{lang}\n{text.rstrip()}\n```"


def aliencode() -> str:
    code = PACKAGE / "sandboxes" / "code"
    tasks = json.loads((code / "eval_set_v2.json").read_text())
    solutions = json.loads((code / "reference_solutions.json").read_text())
    out = ["# AlienCode: the public demo world", "",
           "The surface language and the reference manual (`explorationbench/sandboxes/code/manual.md`) "
           f"are those of the evaluation world. The demo world differs from the manual in the "
           f"{count(len(ALL_RULES))} discovery targets below; the evaluation world has 31 targets of its "
           "own, and no demo target is one of them.", "",
           "## Discovery targets", "",
           "| Id | Operation | Manual | Demo world | Answer key |", "|---|---|---|---|---|"]
    for spec in pack.ALIEN_RULE_SPECS:
        out.append(f"| {spec['id']} | {spec['name']} | {spec['std']} | {spec['actual']} | "
                   f"`{world_data.GROUND_TRUTH_SEXPR[spec['id']]}` |")
    out += ["", "## Fixed worked examples", "",
            f"Every system sees these {count(len(world_data.DEMOS))} programs and their outputs before M0, "
            "and nothing else.", ""]
    for demo in world_data.DEMOS:
        out += [f"**{demo['id']} ({demo['band']}).** {demo['prompt']}", "", fence(demo["code"], "aliencode"),
                f"Output in the demo world: `{alien_exec(demo['code'])}`; under the manual: "
                f"`{manual(demo['code'])}`.", ""]
    out += ["## Held-out tasks", "",
            "Each code task is graded by running the submitted function on hidden calls; the call in "
            "the prompt only shows the interface. Expected outputs are what the reference program "
            "prints in the world. The naive program is what a system that trusts the manual writes: it "
            "is correct under the manual and wrong here.", ""]
    for task in tasks:
        out += [f"### {task['id']} · {task['difficulty']} · {task['question_type']}", "",
                fence(task["prompt"]), f"Rules involved: {', '.join(task['rules_tested'])}.", ""]
        if task["question_type"] == "predict":
            out += [f"Answer: `{task['expected']}` (under the manual: `{manual(task['given_code'])}`).", ""]
            continue
        rows = ["| Hidden call | Expected |", "|---|---|"]
        rows += [f"| `{c['call']}` | `{c['expected']}` |" for c in task["test_cases"]]
        naive = solutions[task["id"]]["naive"]
        miss = next((case, got) for case in task["test_cases"]
                    for got in [alien_exec(f"{naive}\n{case['call']}")] if got != case["expected"])
        last = miss[1].splitlines()[-1] if miss[1] else ""
        out += rows + ["", "Reference program:", "", fence(solutions[task["id"]]["reference"], "aliencode"),
                       f"Naive program. On `{miss[0]['call']}` it prints `{last}` instead of "
                       f"`{miss[0]['expected']}`:", "", fence(naive, "aliencode"), ""]
    out += ["## Rule report", "",
            "At every milestone a tool-less copy of the conversation is asked for this report; each "
            "line is matched structurally against the answer key above.", "",
            fence(world_data.SEXPR_MILESTONE_PROMPT)]
    return "\n".join(out) + "\n"


def alienlogic() -> str:
    episode = episodes.PUBLIC_DEMO
    out = ["# AlienLogic: the public demo world", "",
           "The proof format, the standard rules, and the reference manual are those of the evaluation "
           f"world (`explorationbench/sandboxes/logic/engine.py`). The demo world adds "
           f"{count(len(episode.alien_rules))} side conditions; the evaluation world adds 24 of its own, "
           "and no demo condition is one of them.",
           "", "## Side conditions", "", "| Id | Name | Condition |", "|---|---|---|"]
    for name in episode.alien_rules:
        rid, text = _ORACLE_SPECS[name]
        out.append(f"| {rid} | `{name}` | {text} |")
    out += ["", "## Fixed worked examples", "",
            f"{count(len(episode.seed_examples)).capitalize()} accepted proofs and "
            f"{count(len(REJECTED_DEMOS))} refused ones, with the verdict the checker returns. A refusal "
            "carries only an opaque reason id.", ""]
    for seed in episode.seed_examples:
        verdict = proof_diagnostic(seed.proof_text, episode.alien_rules)
        out += [f"**{seed.id}.** {seed.description} Verdict: "
                f"{'ACCEPTED' if verdict['accepted'] else verdict['reason_id']}.", "",
                fence(seed.proof_text), ""]
    for rid, description, proof in REJECTED_DEMOS:
        verdict = proof_diagnostic(proof, episode.alien_rules)
        out += [f"**{rid}.** {description} Verdict: {verdict['reason_class']} / {verdict['reason_id']}.",
                "", fence(proof), ""]
    out += ["## Held-out theorems", ""]
    for theorem in episode.heldout_theorems:
        out += [f"### {theorem.id} · {theorem.role}", "",
                f"Premises: `{', '.join(theorem.premises)}`. Goal: `{theorem.goal}`. "
                f"Built around: {', '.join(theorem.key_rules) or 'none'}.", ""]
        if theorem.alien_provable:
            out += ["Reference proof:", "", fence(theorem.reference_alien_proof), ""]
            naive = episodes.NAIVE_PROOFS.get(theorem.id)
            if naive:
                verdict = proof_diagnostic(naive, episode.alien_rules, expected_goal=theorem.goal,
                                           expected_premises=theorem.premises)
                out += [f"The standard-calculus proof below is refused here ({verdict['reason_id']}):",
                        "", fence(naive), ""]
        else:
            model = certify.counter_model(theorem.premises, theorem.goal)
            out += ["Unprovable. The correct answer is `ANSWER: UNPROVABLE`. Certificate: with AND_E2 "
                    "gone every remaining rule is sound when AND(φ, ψ) is read as φ, and under that "
                    f"reading the assignment {model} makes the premises true and the goal false.", "",
                    "A naive attempt the checker refuses:", "", fence(theorem.unprovable_naive_proof), ""]
    return "\n".join(out) + "\n"


def main() -> None:
    (HERE / "aliencode.md").write_text(aliencode(), encoding="utf-8")
    (HERE / "alienlogic.md").write_text(alienlogic(), encoding="utf-8")
    print("wrote examples/aliencode.md and examples/alienlogic.md")


if __name__ == "__main__":
    main()
