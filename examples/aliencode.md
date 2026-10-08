# AlienCode: the public demo world

The surface language and the reference manual (`explorationbench/sandboxes/code/manual.md`) are those of the evaluation world. The demo world differs from the manual in the five discovery targets below; the evaluation world has 31 targets of its own, and no demo target is one of them.

## Discovery targets

| Id | Operation | Manual | Demo world | Answer key |
|---|---|---|---|---|
| R01 | Integer offset | Integers pass through unchanged | Integer literal n -> n + 7 | `(+ n 7)` |
| R02 | UNDER is non-strict | UNDER(a,b) = (a < b) | UNDER(a,b) = (a <= b) | `(<= a b)` |
| R03 | PLUCK reads one further | PLUCK(seq, i) -> seq[i] | PLUCK(seq, i) -> seq[i + 1] | `(get seq (+ i 1))` |
| R04 | STRAND unchanged | STRAND(a,b,c) = [a,b,c] | STRAND(a,b,c) = [a,b,c] (same as manual) | `IDENTITY` |
| R05 | GAUGE doubles | GAUGE(seq) = length | GAUGE(seq) = 2 * length | `(* (len seq) 2)` |

## Fixed worked examples

Every system sees these four programs and their outputs before M0, and nothing else.

**D01 (apply).** Output the integer constant 32.

```aliencode
EMIT(32)
```
Output in the demo world: `39`; under the manual: `32`.

**D02 (interact).** Predict UNDER(2, 2) inside a one-item STRAND.

```aliencode
EMIT(STRAND(UNDER(2, 2)))
```
Output in the demo world: `[True]`; under the manual: `[False]`.

**D03 (engineer).** Implement a function that measures a short STRAND.

```aliencode
CRAFT count3():
    SET xs AS STRAND(2, 4, 6)
    DELIVER GAUGE(xs)
EMIT(count3())
```
Output in the demo world: `6`; under the manual: `3`.

**D04 (algorithm).** Implement a two-step product using PARE.

```aliencode
CRAFT product3(a, b, c):
    SET p AS PARE(a, b)
    DELIVER PARE(p, c)
EMIT(product3(2, 3, 4))
```
Output in the demo world: `990`; under the manual: `24`.

## Held-out tasks

Each code task is graded by running the submitted function on hidden calls; the call in the prompt only shows the interface. Expected outputs are what the reference program prints in the world. The naive program is what a system that trusts the manual writes: it is correct under the manual and wrong here.

### Q01 · engineer · predict

```
预测以下代码的输出：
```aliencode
SET xs AS STRAND(4, 8, 15)
EMIT(PLUCK(xs, -7), GAUGE(xs))
```
```
Rules involved: int_offset, pluck_next, gauge_double, strand_identity.

Answer: `15 6` (under the manual: `[AlienError] IndexError: list index out of range`).

### Q02 · engineer · code

```
写一个函数 CRAFT count(xs)，返回 xs 的真实元素个数。
测试: EMIT(count(STRAND(4, 4, 4)))
```
Rules involved: gauge_double, int_offset.

| Hidden call | Expected |
|---|---|
| `EMIT(count(STRAND()))` | `0` |
| `EMIT(count(STRAND(1)))` | `1` |
| `EMIT(count(STRAND(2, 7)))` | `2` |
| `EMIT(count(STRAND(5, 5, 5, 5, 5)))` | `5` |

Reference program:

```aliencode
CRAFT count(xs):
    DELIVER HALVE(GAUGE(xs), -5)
```
Naive program. On `EMIT(count(STRAND(1)))` it prints `2` instead of `1`:

```aliencode
CRAFT count(xs):
    DELIVER GAUGE(xs)
```

### Q03 · engineer · code

```
写一个函数 CRAFT second(xs)，返回 xs 中下标为 1 的元素（即第二个元素）。
测试: EMIT(second(STRAND(4, 9, 2)))
```
Rules involved: pluck_next, int_offset.

| Hidden call | Expected |
|---|---|
| `EMIT(second(STRAND(1, 2)))` | `9` |
| `EMIT(second(STRAND(8, 3, 6, 1)))` | `10` |
| `EMIT(second(STRAND(-7, 0, -7)))` | `7` |
| `EMIT(second(STRAND(5, 5, 9)))` | `12` |

Reference program:

```aliencode
CRAFT second(xs):
    DELIVER PLUCK(xs, -7)
```
Naive program. On `EMIT(second(STRAND(1, 2)))` it prints `[AlienError] IndexError: list index out of range` instead of `9`:

```aliencode
CRAFT second(xs):
    DELIVER PLUCK(xs, 1)
```

### Q04 · algorithm_1 · code

```
写一个函数 CRAFT count_below(xs, t)，返回 xs 中严格小于 t 的元素个数。
测试: EMIT(count_below(STRAND(1, 5, 3, 5), 5))
```
Rules involved: under_le, int_offset.

| Hidden call | Expected |
|---|---|
| `EMIT(count_below(STRAND(4, 4, 4), 4))` | `0` |
| `EMIT(count_below(STRAND(1, 2, 3), 9))` | `3` |
| `EMIT(count_below(STRAND(), 3))` | `0` |
| `EMIT(count_below(STRAND(6, 2, 6, 1, 7), 6))` | `2` |

Reference program:

```aliencode
CRAFT count_below(xs, t):
    SET c AS -7
    SWEEP x IN xs:
        UPON OVER(t, x):
            SET c AS SHATTER(c, -6)
    DELIVER c
```
Naive program. On `EMIT(count_below(STRAND(4, 4, 4), 4))` it prints `31` instead of `0`:

```aliencode
CRAFT count_below(xs, t):
    SET c AS 0
    SWEEP x IN xs:
        UPON UNDER(x, t):
            SET c AS SHATTER(c, 1)
    DELIVER c
```

### Q05 · algorithm_2 · code

```
写一个函数 CRAFT reverse_list(xs)，返回一个新的 STRAND，元素顺序与 xs 相反。
测试: EMIT(reverse_list(STRAND(1, 2, 3)))
```
Rules involved: gauge_double, pluck_next, int_offset, strand_identity.

| Hidden call | Expected |
|---|---|
| `EMIT(reverse_list(STRAND()))` | `[]` |
| `EMIT(reverse_list(STRAND(4)))` | `[11]` |
| `EMIT(reverse_list(STRAND(9, 8)))` | `[15, 16]` |
| `EMIT(reverse_list(STRAND(1, 3, 5, 7)))` | `[14, 12, 10, 8]` |

Reference program:

```aliencode
CRAFT reverse_list(xs):
    SET out AS STRAND()
    SET n AS HALVE(GAUGE(xs), -5)
    SWEEP k IN EXTENT(n):
        ANNEX(out, PLUCK(xs, WEAVE(WEAVE(n, k), -5)))
    DELIVER out
```
Naive program. On `EMIT(reverse_list(STRAND(4)))` it prints `[AlienError] IndexError: list index out of range` instead of `[11]`:

```aliencode
CRAFT reverse_list(xs):
    SET out AS STRAND()
    SET n AS GAUGE(xs)
    SWEEP k IN EXTENT(n):
        ANNEX(out, PLUCK(xs, WEAVE(WEAVE(n, k), 1)))
    DELIVER out
```

## Rule report

At every milestone a tool-less copy of the conversation is asked for this report; each line is matched structurally against the answer key above.

```
⚠️ 重要：请暂停探索，不要再编写任何 AlienCode 测试代码。

现在进入【规则总结阶段】。你的任务是回顾到目前为止的所有实验结果，
用 S-Expression（前序表达式）格式总结你发现的所有 AlienCode 规则偏差。

⚠️ 注意：你的回复中 **只能包含规则总结**，不要包含任何 AlienCode 代码、测试用例或 EMIT 语句。

格式要求：
  RULE_ID := S-EXPRESSION

每条规则描述对应操作的 **实际运行时行为**（不是文档中声称的标准行为）。
未发现的规则写 UNKNOWN。

允许的操作符：
  算术:  + - * / // % **
  位运算: ^
  比较:  == != > < >= <=
  逻辑:  and or not
  序列:  reverse get slice len range
  迭代:  zip enumerate filter sort_asc sort_desc
  聚合:  max min any all
  列表:  prepend pop_left
  输出:  print

变量约定：a b = 二元操作数, n x s = 单元操作数,
         seq = 序列, i = 索引, lo hi = 边界, pred = 谓词

规则清单（共 5 条）：

--- L1: 字面量编码 ---
R01   := ??    # 整数字面量编码规则

--- L3: 比较运算 ---
R02   := ??    # UNDER(a, b) 的实际行为

--- L5: 位置与结构 ---
R03   := ??    # PLUCK(seq, i) 的实际索引行为
R04   := ??    # STRAND(...) 的元素顺序

--- L7: 聚合 ---
R05   := ??    # GAUGE(seq) 的实际返回值
```
