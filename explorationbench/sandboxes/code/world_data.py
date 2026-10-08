"""World data of the public AlienCode demo world.

Everything the harness reads that is specific to a world, apart from the rule
pack (``engine.py``) and the held-out task set (``eval_set_v2.json``): the fixed
worked examples, the legacy seed phase, the answer key of the rule report, the
rule-report template, and the diagnostic phrases the harness looks for in
exploration notes. The evaluation world supplies the same names from private
data. ``build_world.py`` recomputes every stored output from the rule pack and
fails if one has drifted.
"""

from __future__ import annotations

#: Rules the worked examples expose. Recorded for the audit, never shown.
PUBLIC_DEMO_RULES = ("int_offset", "under_le", "strand_identity", "gauge_double")

DEMOS = (
    {"id": "D01", "band": "apply", "prompt": "Output the integer constant 32.",
     "code": "EMIT(32)", "rules": ("int_offset",)},
    {"id": "D02", "band": "interact", "prompt": "Predict UNDER(2, 2) inside a one-item STRAND.",
     "code": "EMIT(STRAND(UNDER(2, 2)))", "rules": ("under_le", "strand_identity")},
    {"id": "D03", "band": "engineer", "prompt": "Implement a function that measures a short STRAND.",
     "code": "CRAFT count3():\n    SET xs AS STRAND(2, 4, 6)\n    DELIVER GAUGE(xs)\nEMIT(count3())",
     "rules": ("gauge_double",)},
    {"id": "D04", "band": "algorithm", "prompt": "Implement a two-step product using PARE.",
     "code": ("CRAFT product3(a, b, c):\n"
              "    SET p AS PARE(a, b)\n"
              "    DELIVER PARE(p, c)\n"
              "EMIT(product3(2, 3, 4))"),
     "rules": ("int_offset",)},
)

#: The pre-v2 seed phase. v2 runs use DEMOS instead; these keep the legacy
#: protocol runnable.
SEED_TASKS = [
    {"id": "S01", "prompt": "请写代码用 EMIT 显示整数 100。先给出预测输出，再给出代码。",
     "real_world_answer": "100", "expected": "107", "rules": ["int_offset"]},
    {"id": "S02", "prompt": '请写代码用 EMIT 显示字符串 "hello"。先给出预测输出，再给出代码。',
     "real_world_answer": "hello", "expected": "hello", "rules": []},
    {"id": "S03", "prompt": "请写代码计算 UNDER(5, 5) 并用 EMIT 显示。先给出预测输出，再给出代码。",
     "real_world_answer": "False", "expected": "True", "rules": ["under_le"]},
    {"id": "S04", "prompt": "请写代码用 EMIT 显示 GAUGE(STRAND(10, 20, 30))。先给出预测输出，再给出代码。",
     "real_world_answer": "3", "expected": "6", "rules": ["gauge_double"]},
]

LOCKED_SEED_CODES = {
    "S01": "EMIT(100)",
    "S02": 'EMIT("hello")',
    "S03": "EMIT(UNDER(5, 5))",
    "S04": "EMIT(GAUGE(STRAND(10, 20, 30)))",
}

GROUND_TRUTH_SEXPR = {
    "R01": "(+ n 7)",
    "R02": "(<= a b)",
    "R03": "(get seq (+ i 1))",
    "R04": "IDENTITY",
    "R05": "(* (len seq) 2)",
}

_RULE_SLOTS = """\
--- L1: 字面量编码 ---
R01   := ??    # 整数字面量编码规则

--- L3: 比较运算 ---
R02   := ??    # UNDER(a, b) 的实际行为

--- L5: 位置与结构 ---
R03   := ??    # PLUCK(seq, i) 的实际索引行为
R04   := ??    # STRAND(...) 的元素顺序

--- L7: 聚合 ---
R05   := ??    # GAUGE(seq) 的实际返回值
"""

SEXPR_MILESTONE_PROMPT = (
    "⚠️ 重要：请暂停探索，不要再编写任何 AlienCode 测试代码。\n\n"
    "现在进入【规则总结阶段】。你的任务是回顾到目前为止的所有实验结果，\n"
    "用 S-Expression（前序表达式）格式总结你发现的所有 AlienCode 规则偏差。\n\n"
    "⚠️ 注意：你的回复中 **只能包含规则总结**，不要包含任何 AlienCode 代码、测试用例或 EMIT 语句。\n\n"
    "格式要求：\n  RULE_ID := S-EXPRESSION\n\n"
    "每条规则描述对应操作的 **实际运行时行为**（不是文档中声称的标准行为）。\n"
    "未发现的规则写 UNKNOWN。\n\n"
    "允许的操作符：\n"
    "  算术:  + - * / // % **\n  位运算: ^\n  比较:  == != > < >= <=\n"
    "  逻辑:  and or not\n  序列:  reverse get slice len range\n"
    "  迭代:  zip enumerate filter sort_asc sort_desc\n  聚合:  max min any all\n"
    "  列表:  prepend pop_left\n  输出:  print\n\n"
    "变量约定：a b = 二元操作数, n x s = 单元操作数,\n"
    "         seq = 序列, i = 索引, lo hi = 边界, pred = 谓词\n\n"
    f"规则清单（共 {len(GROUND_TRUTH_SEXPR)} 条）：\n\n" + _RULE_SLOTS
)

SEXPR_RETRY_PROMPT = (
    "⚠️ 你刚才的回复没有按要求的 S-Expression 格式总结规则。\n"
    "请 **不要** 继续编写测试代码或 EMIT 语句。\n\n"
    f"请严格按以下格式，逐条列出 {min(GROUND_TRUTH_SEXPR)} 到 {max(GROUND_TRUTH_SEXPR)} 的规则总结：\n"
    "  RULE_ID := S-EXPRESSION\n\n例如：\n  R99 := (op a b)\n\n"
    f"未发现的规则写 UNKNOWN。\n请现在输出完整的 {len(GROUND_TRUTH_SEXPR)} 条规则总结。"
)

#: Phrases that mark an exploration note as reasoning about a divergence from
#: the manual. World-neutral on purpose.
DIVERGENCE_PATTERNS = [
    "规则.*?不同", "环境.*?不同", "手册.*?不同", "标准.*?不同", "猜测", "假设", "假说",
    "推测", "编码", "映射", "不是.*?标准", "偏移", "反转", "互换", "取反", "偏差",
    "diverge", "differ", "一致",
]
