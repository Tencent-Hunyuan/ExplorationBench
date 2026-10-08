"""Certificates that the demo world's unprovable theorems have no proof.

The naive proofs in ``episodes.py`` only show that one attempt fails. The
certificate here covers every proof. With AND_E2 disabled, every remaining rule
of the calculus stays sound when a conjunction AND(phi, psi) is read as phi
alone; the other side conditions only remove steps, so they cannot break
soundness. A theorem whose premises can all be true while its goal is false
under that reading therefore has no proof in the demo world.

For each unprovable theorem the module searches the truth assignments for such
a counter-model. It also tests the soundness claim on every proof the verifier
accepts in this world: each line of an accepted proof must hold, under the
reading, in every assignment that satisfies the premises and the assumptions
open at that line.

    python3 -m sandboxes.logic.certify
"""

from __future__ import annotations

import itertools
import sys

from sandboxes.logic.engine import (And, Atom, Bot, Impl, Not, Or, parse_formula,
                                    parse_proof, proof_diagnostic)
from sandboxes.logic.episodes import NAIVE_PROOFS, PUBLIC_DEMO


def atoms(formula, acc: set[str]) -> set[str]:
    if isinstance(formula, Atom):
        acc.add(formula.name)
    elif isinstance(formula, Not):
        atoms(formula.sub, acc)
    elif isinstance(formula, (And, Or, Impl)):
        atoms(formula.l, acc)
        atoms(formula.r, acc)
    return acc


def holds(formula, valuation: dict[str, bool]) -> bool:
    """Truth under the reading AND(phi, psi) := phi, classical otherwise."""
    if isinstance(formula, Atom):
        return valuation[formula.name]
    if isinstance(formula, Bot):
        return False
    if isinstance(formula, Not):
        return not holds(formula.sub, valuation)
    if isinstance(formula, And):
        return holds(formula.l, valuation)
    if isinstance(formula, Or):
        return holds(formula.l, valuation) or holds(formula.r, valuation)
    if isinstance(formula, Impl):
        return (not holds(formula.l, valuation)) or holds(formula.r, valuation)
    raise TypeError(formula)


def valuations(names: set[str]):
    names = sorted(names)
    for bits in itertools.product((False, True), repeat=len(names)):
        yield dict(zip(names, bits))


def counter_model(premises: list[str], goal: str) -> dict[str, bool] | None:
    parsed = [parse_formula(p) for p in premises]
    target = parse_formula(goal)
    names = set().union(*(atoms(f, set()) for f in parsed + [target]))
    for valuation in valuations(names):
        if all(holds(f, valuation) for f in parsed) and not holds(target, valuation):
            return valuation
    return None


def unsound_lines(proof_text: str) -> list[int]:
    """Lines of a proof that do not follow, under the reading, from what is open."""
    proof = parse_proof(proof_text)
    open_assumptions: list[tuple[int, object]] = []
    names = set().union(*(atoms(f, set()) for f in proof.premises),
                        *(atoms(s.formula, set()) for s in proof.steps))
    bad = []
    for step in proof.steps:
        keep_below = step.indent if step.rule == "assume" else step.indent + 1
        open_assumptions = [(d, f) for d, f in open_assumptions if d < keep_below]
        if step.rule == "assume":
            open_assumptions.append((step.indent, step.formula))
            continue
        context = list(proof.premises) + [f for _, f in open_assumptions]
        if any(all(holds(f, v) for f in context) and not holds(step.formula, v)
               for v in valuations(names)):
            bad.append(step.line)
    return bad


def main() -> int:
    episode = PUBLIC_DEMO
    ok = True
    for theorem in episode.heldout_theorems:
        if theorem.alien_provable:
            continue
        model = counter_model(theorem.premises, theorem.goal)
        verdict = f"counter-model {model}" if model else "NO COUNTER-MODEL"
        ok &= model is not None
        print(f"{theorem.id}: {', '.join(theorem.premises)} |- {theorem.goal}: {verdict}")

    accepted = [s.proof_text for s in episode.seed_examples if s.proof_text]
    accepted += [t.reference_alien_proof for t in episode.heldout_theorems if t.alien_provable]
    naive = [p for t in episode.heldout_theorems if not t.alien_provable
             for p in [t.unprovable_naive_proof, *t.unprovable_naive_proof_alts] if p]
    for proof in accepted:
        assert proof_diagnostic(proof, episode.alien_rules)["accepted"]
        lines = unsound_lines(proof)
        ok &= not lines
        if lines:
            print(f"accepted proof has lines invalid under the reading: {lines}")
    print(f"soundness of the reading: {len(accepted)} accepted proofs, every line valid"
          if ok else "soundness check FAILED")
    flagged = sum(1 for proof in naive
                  if proof_diagnostic(proof, ())["accepted"] and unsound_lines(proof))
    print(f"control: {flagged} of {len(naive)} naive proofs pass the standard calculus "
          "and are caught by the reading")
    aware = [t for t in episode.heldout_theorems if t.alien_aware_required]
    for theorem in aware:
        proof = NAIVE_PROOFS.get(theorem.id)
        standard = proof is not None and proof_diagnostic(
            proof, (), expected_goal=theorem.goal, expected_premises=theorem.premises)["accepted"]
        world = proof is not None and proof_diagnostic(
            proof, episode.alien_rules, expected_goal=theorem.goal,
            expected_premises=theorem.premises)["accepted"]
        if not standard or world:
            ok = False
            print(f"{theorem.id}: the naive proof must pass the standard calculus and fail here")
    print(f"naive proofs: {len(aware)} theorems that need the side conditions are solved by "
          "the standard calculus and refused in this world")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
