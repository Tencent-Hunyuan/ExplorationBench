"""Episode of the public AlienLogic demo world.

The same data structures as the evaluation episode, filled with a small public
world: the three side conditions of ``rules.py``, three accepted worked
examples, and five held-out theorems in the main roles the evaluation set uses
(sanity, single-rule, two-rule, unprovable). The evaluation episode itself
(24 side conditions, 70 held-out theorems) is private.

``validate_episode`` is the check every episode must pass before a run starts:
worked examples get the verdict they declare, every provable theorem has a
reference proof the verifier accepts, and every unprovable theorem has naive
proofs the verifier refuses. That an unprovable theorem has no proof at all is
certified separately, in ``certify.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SeedExample:
    id: str
    description: str
    # Worked-example mode: the proof is shown to the model together with the
    # verifier's verdict on it.
    proof_text: str = ""
    expected_accepted: bool = True
    expected_reason_id: str | None = None
    # Task mode: only premises and goal are shown; `reference_proof` is a hidden
    # witness that the task is solvable.
    premises: list[str] = field(default_factory=list)
    goal: str | None = None
    reference_proof: str | None = None


@dataclass
class HeldoutTheorem:
    id: str
    cluster_id: str
    role: str
    premises: list[str]
    goal: str
    alien_provable: bool = True
    min_alien_steps: int = 0
    min_classical_steps: int = 0
    alien_aware_required: bool = False
    reference_alien_proof: Optional[str] = None
    unprovable_naive_proof: Optional[str] = None
    unprovable_naive_proof_alts: list[str] = field(default_factory=list)
    # The side conditions a theorem is built around; for analysis only and never
    # shown to the model.
    key_rules: tuple[str, ...] = ()


@dataclass
class EpisodeSpec:
    id: str
    level: str
    alien_rules: tuple[str, ...]
    description: str
    seed_examples: list[SeedExample]
    heldout_theorems: list[HeldoutTheorem]
    seed_hide_detail: bool = False


ALIEN_RULES: tuple[str, ...] = (
    "AND_E_LEFT_ONLY",       # X01
    "IMPL_E_MAJOR_FIRST",    # X02
    "BOT_E_TOP_LEVEL_ONLY",  # X03
)


def _proof(premises: str, goal: str, *lines: str) -> str:
    return "\n".join(["PROOF", f"premises: {premises}", f"goal: {goal}", *lines, "END"])


_SEEDS: list[SeedExample] = [
    SeedExample("S01", "Modus Ponens, implication cited first.", _proof(
        "IMPL(p, q), p", "q",
        "1. IMPL(p, q)          [premise]",
        "2. p                   [premise]",
        "3. q                   [IMPL_E 1, 2]")),
    SeedExample("S02", "Left projection of a conjunction.", _proof(
        "AND(p, q)", "p",
        "1. AND(p, q)           [premise]",
        "2. p                   [AND_E1 1]")),
    SeedExample("S03", "Explosion at the top level.", _proof(
        "p, NOT(p)", "q",
        "1. p                   [premise]",
        "2. NOT(p)              [premise]",
        "3. BOT                 [NOT_E 1, 2]",
        "4. q                   [BOT_E 3]")),
]


def _theorem(tid: str, role: str, premises: list[str], goal: str, *,
             proof: str | None = None, naive: list[str] = (),
             key_rules: tuple[str, ...] = (), aware: bool = False) -> HeldoutTheorem:
    provable = proof is not None
    steps = sum(1 for line in (proof or "").splitlines() if line[:1].isdigit())
    return HeldoutTheorem(
        id=tid, cluster_id=f"demo_{role}", role=role, premises=premises, goal=goal,
        alien_provable=provable, min_alien_steps=steps, alien_aware_required=aware,
        reference_alien_proof=proof,
        unprovable_naive_proof=None if provable else naive[0],
        unprovable_naive_proof_alts=[] if provable else list(naive[1:]),
        key_rules=key_rules)


_HELDOUT: list[HeldoutTheorem] = [
    _theorem("P01", "sanity", ["a", "b"], "AND(a, b)", proof=_proof(
        "a, b", "AND(a, b)",
        "1. a                   [premise]",
        "2. b                   [premise]",
        "3. AND(a, b)           [AND_I 1, 2]")),
    _theorem("P02", "single_rule", ["r", "IMPL(r, t)"], "t", aware=True,
             key_rules=("IMPL_E_MAJOR_FIRST",), proof=_proof(
        "r, IMPL(r, t)", "t",
        "1. r                   [premise]",
        "2. IMPL(r, t)          [premise]",
        "3. t                   [IMPL_E 2, 1]")),
    _theorem("P03", "single_rule", ["NOT(p)"], "IMPL(p, q)", aware=True,
             key_rules=("BOT_E_TOP_LEVEL_ONLY",), proof=_proof(
        "NOT(p)", "IMPL(p, q)",
        "1. NOT(p)              [premise]",
        "2. | p                 [assume]",
        "3. | | NOT(q)          [assume]",
        "4. | | BOT             [NOT_E 2, 1]",
        "5. | NOT(NOT(q))       [NOT_I 3-4]",
        "6. | q                 [DNE 5]",
        "7. IMPL(p, q)          [IMPL_I 2-6]")),
    _theorem("P04", "two_rule", ["AND(s, t)", "IMPL(s, t)"], "t", aware=True,
             key_rules=("AND_E_LEFT_ONLY", "IMPL_E_MAJOR_FIRST"), proof=_proof(
        "AND(s, t), IMPL(s, t)", "t",
        "1. AND(s, t)           [premise]",
        "2. IMPL(s, t)          [premise]",
        "3. s                   [AND_E1 1]",
        "4. t                   [IMPL_E 2, 3]")),
    _theorem("U01", "unprovable", ["AND(p, q)"], "q",
             key_rules=("AND_E_LEFT_ONLY",), naive=[_proof(
        "AND(p, q)", "q",
        "1. AND(p, q)           [premise]",
        "2. q                   [AND_E2 1]")]),
]


#: What a solver who trusts the standard calculus writes for the theorems that
#: need the side conditions. Each is accepted without the rule pack and refused
#: with it (checked in certify.py); the smoke-test model answers with them.
NAIVE_PROOFS: dict[str, str] = {
    "P02": _proof("r, IMPL(r, t)", "t",
                  "1. r                   [premise]",
                  "2. IMPL(r, t)          [premise]",
                  "3. t                   [IMPL_E 1, 2]"),
    "P03": _proof("NOT(p)", "IMPL(p, q)",
                  "1. NOT(p)              [premise]",
                  "2. | p                 [assume]",
                  "3. | BOT               [NOT_E 2, 1]",
                  "4. | q                 [BOT_E 3]",
                  "5. IMPL(p, q)          [IMPL_I 2-4]"),
    "P04": _proof("AND(s, t), IMPL(s, t)", "t",
                  "1. AND(s, t)           [premise]",
                  "2. IMPL(s, t)          [premise]",
                  "3. t                   [AND_E2 1]"),
}


PUBLIC_DEMO = EpisodeSpec(
    id="public_demo",
    level="demo",
    alien_rules=ALIEN_RULES,
    description="Public demo world: three side conditions, five held-out theorems.",
    seed_examples=_SEEDS,
    heldout_theorems=_HELDOUT,
)

EPISODES: dict[str, EpisodeSpec] = {PUBLIC_DEMO.id: PUBLIC_DEMO}


def get_episode(episode_id: str) -> EpisodeSpec:
    if episode_id not in EPISODES:
        raise KeyError(f"unknown episode: {episode_id}")
    return EPISODES[episode_id]


def validate_episode(episode: EpisodeSpec) -> dict[str, Any]:
    """Check an episode against the loaded rule pack before any run uses it."""
    from sandboxes.logic.engine import proof_diagnostic
    report: dict[str, Any] = {"episode": episode.id, "seeds": [], "heldouts": [],
                              "unprovable_checks": []}
    for s in episode.seed_examples:
        if s.goal is not None:
            assert s.reference_proof is not None, f"task-mode seed {s.id}: no witness"
            r = proof_diagnostic(s.reference_proof, episode.alien_rules,
                                 hide_reason_detail=False,
                                 expected_goal=s.goal, expected_premises=s.premises)
            ok = bool(r["accepted"])
            report["seeds"].append({"id": s.id, "ok": ok, "diag": r})
            assert ok, f"task-mode seed {s.id}: reference witness rejected: {r}"
            continue
        r = proof_diagnostic(s.proof_text, episode.alien_rules, hide_reason_detail=False)
        ok = bool(r["accepted"]) == s.expected_accepted
        if s.expected_reason_id is not None:
            ok = ok and r.get("reason_id") == s.expected_reason_id
        report["seeds"].append({"id": s.id, "ok": ok, "diag": r})
        assert ok, (f"seed {s.id} mismatch: expected accepted={s.expected_accepted} "
                    f"reason_id={s.expected_reason_id}, got {r}")
    for t in episode.heldout_theorems:
        if t.alien_provable:
            assert t.reference_alien_proof is not None, f"held-out {t.id}: no reference proof"
            r = proof_diagnostic(t.reference_alien_proof, episode.alien_rules,
                                 hide_reason_detail=False, expected_goal=t.goal,
                                 expected_premises=t.premises)
            ok = bool(r["accepted"])
            report["heldouts"].append({"id": t.id, "ok": ok, "diag": r})
            assert ok, f"held-out {t.id} reference proof rejected: {r}"
        else:
            naive_proofs = [p for p in [t.unprovable_naive_proof, *t.unprovable_naive_proof_alts] if p]
            assert naive_proofs, f"unprovable held-out {t.id}: no naive proof"
            for i, naive in enumerate(naive_proofs):
                r = proof_diagnostic(naive, episode.alien_rules, hide_reason_detail=False,
                                     expected_goal=t.goal, expected_premises=t.premises)
                ok = not bool(r["accepted"])
                report["unprovable_checks"].append({"id": t.id, "alt_idx": i,
                                                    "rejected": ok, "diag": r})
                assert ok, f"unprovable held-out {t.id} alt#{i}: naive proof accepted: {r}"
            report["heldouts"].append({"id": t.id, "ok": True, "skip_reason": "unprovable"})
    return report


if __name__ == "__main__":
    rep = validate_episode(PUBLIC_DEMO)
    print(f"{PUBLIC_DEMO.id}: {len(PUBLIC_DEMO.alien_rules)} rules, "
          f"{len(rep['seeds'])} worked examples, {len(rep['heldouts'])} held-out theorems, "
          f"{len(rep['unprovable_checks'])} naive proofs refused -- all checks pass")
