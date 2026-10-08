"""Rule pack of the public AlienLogic demo world.

Three side conditions on the standard calculus. They were written for this
release, are not among the rules of the evaluation world, and never enter a
reported score. See ``engine.py`` for the hook contract.
"""

from __future__ import annotations


def build_registry(engine) -> dict[str, dict]:
    err = engine._err

    def and_e_left_only(state, idx):
        step = state.steps[idx]
        if step.rule == "AND_E2":
            return err("ALIEN_AXIOM_DISABLED", "X01", "AND_E2 is disabled")
        return None

    def impl_e_major_first(state, idx):
        step = state.steps[idx]
        if step.rule != "IMPL_E":
            return None
        first, second = step.refs
        major = state.steps[first - 1].formula
        minor = state.steps[second - 1].formula
        if isinstance(major, engine.Impl) and major.l == minor:
            return None
        return err("ALIEN_SIDE_CONDITION", "X02",
                   "IMPL_E must cite the implication before its antecedent")

    def bot_e_top_level_only(state, idx):
        step = state.steps[idx]
        if step.rule == "BOT_E" and step.indent > 0:
            return err("ALIEN_SIDE_CONDITION", "X03",
                       "BOT_E is only allowed outside assumption boxes")
        return None

    return {
        "AND_E_LEFT_ONLY": {
            "id": "X01", "family": "disabled", "affects": ["AND_E2"],
            "summary": "only the left conjunct can be extracted",
            "hook": and_e_left_only,
        },
        "IMPL_E_MAJOR_FIRST": {
            "id": "X02", "family": "reference", "affects": ["IMPL_E"],
            "summary": "IMPL_E cites the implication first",
            "hook": impl_e_major_first,
        },
        "BOT_E_TOP_LEVEL_ONLY": {
            "id": "X03", "family": "scope", "affects": ["BOT_E"],
            "summary": "no explosion inside an assumption box",
            "hook": bot_e_top_level_only,
        },
    }
