"""oracle_rules.py — the open-book disclosure for the AlienCode demo world.

Open-book answering (O@M0 and A4+O) hands the model the complete rule set
right before the held-out tasks. The message is built from the rule pack's
ALIEN_RULE_SPECS and the answer key in world_data, so it cannot drift from
the world that grades the answers.
"""
from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def build_oracle_message() -> str:
    """Return the user-turn message body listing the world's rules."""
    from sandboxes.code.engine import ALIEN_RULE_SPECS
    from sandboxes.code.world_data import GROUND_TRUTH_SEXPR
    lines = [
        "以下是本 sandbox 实际执行的 **alien 规则** ground truth。",
        "manual 中的规范只是名义描述，运行时的真实行为按下表判定 —— 你可以直接把它们当作绝对正确的规则集，",
        "在接下来的 held-out 测试中请严格遵循，不需要再自行探索或总结。",
        "",
        "格式：`<RULE_ID>` (`<key>`) — manual 声称：`<std>` — 实际执行：`<actual>` — 规范 S-expression：`<sexpr>`",
        "",
    ]
    for spec in ALIEN_RULE_SPECS:
        sexpr = GROUND_TRUTH_SEXPR.get(spec["id"], "?")
        lines.append(
            f"- **{spec['id']}** (`{spec['key']}`) — manual: `{spec['std']}` — actual: **{spec['actual']}** "
            f"— S-expression: `{sexpr}`"
        )
    lines.append("")
    lines.append(
        f"共 {len(ALIEN_RULE_SPECS)} 条 discovery target（其中 identity=True 表示与 manual 一致，"
        "无需修正；其余需按 actual 列覆盖 manual 的描述）。请直接进入 held-out 测试。"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    print(build_oracle_message())
