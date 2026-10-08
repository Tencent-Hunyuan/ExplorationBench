# AlienLogic: the public demo world

The proof format, the standard rules, and the reference manual are those of the evaluation world (`explorationbench/sandboxes/logic/engine.py`). The demo world adds three side conditions; the evaluation world adds 24 of its own, and no demo condition is one of them.

## Side conditions

| Id | Name | Condition |
|---|---|---|
| X01 | `AND_E_LEFT_ONLY` | AND_E2 is disabled: from AND(φ, ψ) only the left conjunct φ can be extracted, with AND_E1. |
| X02 | `IMPL_E_MAJOR_FIRST` | IMPL_E must cite the implication line first and its antecedent second: [IMPL_E i, j] where line i is IMPL(φ, ψ) and line j is φ. |
| X03 | `BOT_E_TOP_LEVEL_ONLY` | BOT_E is refused inside an assumption box; it is only legal at indent 0. |

## Fixed worked examples

Three accepted proofs and two refused ones, with the verdict the checker returns. A refusal carries only an opaque reason id.

**S01.** Modus Ponens, implication cited first. Verdict: ACCEPTED.

```
PROOF
premises: IMPL(p, q), p
goal: q
1. IMPL(p, q)          [premise]
2. p                   [premise]
3. q                   [IMPL_E 1, 2]
END
```

**S02.** Left projection of a conjunction. Verdict: ACCEPTED.

```
PROOF
premises: AND(p, q)
goal: p
1. AND(p, q)           [premise]
2. p                   [AND_E1 1]
END
```

**S03.** Explosion at the top level. Verdict: ACCEPTED.

```
PROOF
premises: p, NOT(p)
goal: q
1. p                   [premise]
2. NOT(p)              [premise]
3. BOT                 [NOT_E 1, 2]
4. q                   [BOT_E 3]
END
```

**R01.** 用 AND_E2 取出右侧合取项。 Verdict: ALIEN_AXIOM_DISABLED / X01.

```
PROOF
premises: AND(p, q)
goal: q
1. AND(p, q)               [premise]
2. q                       [AND_E2 1]
END
```

**R02.** IMPL_E 先引用前件，再引用蕴含式。 Verdict: ALIEN_SIDE_CONDITION / X02.

```
PROOF
premises: p, IMPL(p, q)
goal: q
1. p                       [premise]
2. IMPL(p, q)              [premise]
3. q                       [IMPL_E 1, 2]
END
```

## Held-out theorems

### P01 · sanity

Premises: `a, b`. Goal: `AND(a, b)`. Built around: none.

Reference proof:

```
PROOF
premises: a, b
goal: AND(a, b)
1. a                   [premise]
2. b                   [premise]
3. AND(a, b)           [AND_I 1, 2]
END
```

### P02 · single_rule

Premises: `r, IMPL(r, t)`. Goal: `t`. Built around: IMPL_E_MAJOR_FIRST.

Reference proof:

```
PROOF
premises: r, IMPL(r, t)
goal: t
1. r                   [premise]
2. IMPL(r, t)          [premise]
3. t                   [IMPL_E 2, 1]
END
```

The standard-calculus proof below is refused here (X02):

```
PROOF
premises: r, IMPL(r, t)
goal: t
1. r                   [premise]
2. IMPL(r, t)          [premise]
3. t                   [IMPL_E 1, 2]
END
```

### P03 · single_rule

Premises: `NOT(p)`. Goal: `IMPL(p, q)`. Built around: BOT_E_TOP_LEVEL_ONLY.

Reference proof:

```
PROOF
premises: NOT(p)
goal: IMPL(p, q)
1. NOT(p)              [premise]
2. | p                 [assume]
3. | | NOT(q)          [assume]
4. | | BOT             [NOT_E 2, 1]
5. | NOT(NOT(q))       [NOT_I 3-4]
6. | q                 [DNE 5]
7. IMPL(p, q)          [IMPL_I 2-6]
END
```

The standard-calculus proof below is refused here (X03):

```
PROOF
premises: NOT(p)
goal: IMPL(p, q)
1. NOT(p)              [premise]
2. | p                 [assume]
3. | BOT               [NOT_E 2, 1]
4. | q                 [BOT_E 3]
5. IMPL(p, q)          [IMPL_I 2-4]
END
```

### P04 · two_rule

Premises: `AND(s, t), IMPL(s, t)`. Goal: `t`. Built around: AND_E_LEFT_ONLY, IMPL_E_MAJOR_FIRST.

Reference proof:

```
PROOF
premises: AND(s, t), IMPL(s, t)
goal: t
1. AND(s, t)           [premise]
2. IMPL(s, t)          [premise]
3. s                   [AND_E1 1]
4. t                   [IMPL_E 2, 3]
END
```

The standard-calculus proof below is refused here (X01):

```
PROOF
premises: AND(s, t), IMPL(s, t)
goal: t
1. AND(s, t)           [premise]
2. IMPL(s, t)          [premise]
3. t                   [AND_E2 1]
END
```

### U01 · unprovable

Premises: `AND(p, q)`. Goal: `q`. Built around: AND_E_LEFT_ONLY.

Unprovable. The correct answer is `ANSWER: UNPROVABLE`. Certificate: with AND_E2 gone every remaining rule is sound when AND(φ, ψ) is read as φ, and under that reading the assignment {'p': True, 'q': False} makes the premises true and the goal false.

A naive attempt the checker refuses:

```
PROOF
premises: AND(p, q)
goal: q
1. AND(p, q)           [premise]
2. q                   [AND_E2 1]
END
```

