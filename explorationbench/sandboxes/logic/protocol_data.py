"""World data the AlienLogic v2 protocol reads: the refused worked examples.

The accepted worked examples are the episode's seed examples. These refusals
are shown next to them, with the verifier's opaque diagnostic, because in this
sandbox a refusal is the only evidence a worked example can carry. One side
condition of the demo world is left for exploration to find.
"""

from __future__ import annotations

REJECTED_DEMOS: tuple[tuple[str, str, str], ...] = (
    ("R01", "用 AND_E2 取出右侧合取项。", """PROOF
premises: AND(p, q)
goal: q
1. AND(p, q)               [premise]
2. q                       [AND_E2 1]
END"""),
    ("R02", "IMPL_E 先引用前件，再引用蕴含式。", """PROOF
premises: p, IMPL(p, q)
goal: q
1. p                       [premise]
2. IMPL(p, q)              [premise]
3. q                       [IMPL_E 1, 2]
END"""),
)
