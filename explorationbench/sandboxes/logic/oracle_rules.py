"""oracle_rules.py — the open-book disclosure for the AlienLogic demo world.

A hand-written statement of every side condition in ``rules.py``, checked
against the hook it describes. ``build_oracle_message`` refuses to run for a
rule that has no entry, so a changed rule pack cannot silently hand the model
a stale rule set.
"""
from __future__ import annotations

from typing import Iterable

_ORACLE_SPECS: dict[str, tuple[str, str]] = {
    "AND_E_LEFT_ONLY": (
        "X01", "AND_E2 is disabled: from AND(φ, ψ) only the left conjunct φ can be "
        "extracted, with AND_E1."),
    "IMPL_E_MAJOR_FIRST": (
        "X02", "IMPL_E must cite the implication line first and its antecedent second: "
        "[IMPL_E i, j] where line i is IMPL(φ, ψ) and line j is φ."),
    "BOT_E_TOP_LEVEL_ONLY": (
        "X03", "BOT_E is refused inside an assumption box; it is only legal at indent 0."),
}


def build_oracle_message(active_rules: Iterable[str]) -> str:
    """Return the user-turn message body listing the world's side conditions."""
    active = list(active_rules)
    missing = [r for r in active if r not in _ORACLE_SPECS]
    if missing:
        raise LookupError(f"oracle_rules._ORACLE_SPECS has no entry for: {missing}")
    lines = [
        "以下是本 sandbox 实际执行的 **alien 规则**（每条已与 verifier "
        "hook 源码逐一对照）。",
        "这些是真实的、绝对正确的规则集合 —— 你可以直接把它们当作 "
        "ground truth 使用，在接下来的 held-out 证明中请严格遵循，不需要"
        "再自行探索或总结。",
        "",
    ]
    for rule in active:
        kid, desc = _ORACLE_SPECS[rule]
        lines.append(f"- **{kid} / {rule}** — {desc}")
    lines.append("")
    lines.append(f"共 {len(active)} 条。请直接开始 held-out 测试，遇到与上述规则"
                 "冲突的构造一律避免。")
    return "\n".join(lines)


if __name__ == "__main__":
    from episodes import ALIEN_RULES
    print(build_oracle_message(ALIEN_RULES))
