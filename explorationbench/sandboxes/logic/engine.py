"""
AlienLogic proof checker
========================

A Fitch-style natural deduction checker whose inference rules can carry extra
side conditions. The standard calculus, the proof-script format, the verifier,
and the reference manual shown to models are defined here. The side conditions
that make a particular world are not: they come from a rule pack, the module
``sandboxes.logic.rules``, which exposes

    build_registry(engine) -> dict[str, dict]

mapping a rule name to ``{"id": <opaque reason id>, "family": ..., "affects":
[...], "summary": ..., "hook": callable}``. A hook has the signature
``hook(state, idx) -> Optional[RuleErr]``: it runs after the standard rule for
``state.steps[idx]`` has accepted the step and returns ``None`` or
``engine._err(reason_class, reason_id, detail)``. Hooks run in registry order
and the first violation is the one reported, so verdicts are deterministic.

The released pack is the public demo world. The two evaluation worlds use
private packs with the same interface and the same checker.

Public API:
  REFERENCE_MANUAL                : human-readable spec shown to model
  STANDARD_RULES                  : list of canonical rule names
  ALIEN_RULE_REGISTRY             : registry of the loaded rule pack
  parse_formula(text)             : Formula text -> Formula
  parse_proof(text)               : full proof block -> Proof
  verify_proof(proof, alien_rules): -> VerifierResult
  proof_diagnostic(text, alien_rules): -> dict (parses + verifies + reports)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Union
import re


# ═══════════════════════════════════════════════════════════════════════
#  Formula AST
# ═══════════════════════════════════════════════════════════════════════

class Formula:
    pass


@dataclass(frozen=True)
class Atom(Formula):
    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Bot(Formula):
    def __repr__(self) -> str:
        return "BOT"


@dataclass(frozen=True)
class Not(Formula):
    sub: Formula

    def __repr__(self) -> str:
        return f"NOT({self.sub})"


@dataclass(frozen=True)
class And(Formula):
    l: Formula
    r: Formula

    def __repr__(self) -> str:
        return f"AND({self.l}, {self.r})"


@dataclass(frozen=True)
class Or(Formula):
    l: Formula
    r: Formula

    def __repr__(self) -> str:
        return f"OR({self.l}, {self.r})"


@dataclass(frozen=True)
class Impl(Formula):
    l: Formula
    r: Formula

    def __repr__(self) -> str:
        return f"IMPL({self.l}, {self.r})"


def is_atomic(f: Formula) -> bool:
    return isinstance(f, Atom)


def is_compound(f: Formula) -> bool:
    return not isinstance(f, (Atom, Bot))


# ═══════════════════════════════════════════════════════════════════════
#  Formula parser
# ═══════════════════════════════════════════════════════════════════════

class ParseError(Exception):
    pass


_KEYWORDS = {"NOT", "AND", "OR", "IMPL", "BOT"}
_TOKEN_RE = re.compile(r"\s*(NOT|AND|OR|IMPL|BOT|[A-Za-z][A-Za-z0-9_]*|[\(\),])")


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    pos = 0
    while pos < len(text):
        if text[pos].isspace():
            pos += 1
            continue
        m = _TOKEN_RE.match(text, pos)
        if not m or m.start() != pos:
            raise ParseError(f"unexpected character at {pos}: {text[pos]!r}")
        tokens.append(m.group(1))
        pos = m.end()
    return tokens


def parse_formula(text: str) -> Formula:
    tokens = _tokenize(text)
    if not tokens:
        raise ParseError("empty formula")
    tree, pos = _parse_formula(tokens, 0)
    if pos != len(tokens):
        raise ParseError(f"trailing tokens after formula: {tokens[pos:]}")
    return tree


def _parse_formula(tokens: list[str], pos: int) -> tuple[Formula, int]:
    if pos >= len(tokens):
        raise ParseError("unexpected end of input")
    tok = tokens[pos]
    if tok == "BOT":
        return Bot(), pos + 1
    if tok == "NOT":
        if pos + 1 >= len(tokens) or tokens[pos + 1] != "(":
            raise ParseError(f"expected '(' after NOT")
        sub, p = _parse_formula(tokens, pos + 2)
        if p >= len(tokens) or tokens[p] != ")":
            raise ParseError("expected ')' after NOT subformula")
        return Not(sub), p + 1
    if tok in {"AND", "OR", "IMPL"}:
        if pos + 1 >= len(tokens) or tokens[pos + 1] != "(":
            raise ParseError(f"expected '(' after {tok}")
        left, p = _parse_formula(tokens, pos + 2)
        if p >= len(tokens) or tokens[p] != ",":
            raise ParseError(f"expected ',' inside {tok}")
        right, p = _parse_formula(tokens, p + 1)
        if p >= len(tokens) or tokens[p] != ")":
            raise ParseError(f"expected ')' to close {tok}")
        cls = {"AND": And, "OR": Or, "IMPL": Impl}[tok]
        return cls(left, right), p + 1
    if tok in _KEYWORDS:
        raise ParseError(f"unexpected keyword: {tok}")
    if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", tok):
        raise ParseError(f"unexpected token: {tok}")
    return Atom(tok), pos + 1


def _parse_formula_list(text: str) -> list[Formula]:
    out: list[Formula] = []
    depth = 0
    buf: list[str] = []
    for ch in text:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            piece = "".join(buf).strip()
            if piece:
                out.append(parse_formula(piece))
            buf = []
        else:
            buf.append(ch)
    last = "".join(buf).strip()
    if last:
        out.append(parse_formula(last))
    return out


# ═══════════════════════════════════════════════════════════════════════
#  Proof script structure and parser
# ═══════════════════════════════════════════════════════════════════════

LineRef = Union[int, tuple[int, int]]


@dataclass
class Step:
    line: int               # 1-indexed
    indent: int             # 0 = top level; +1 per assumption box
    formula: Formula
    rule: str               # canonical rule name
    refs: list[LineRef]     # int line refs and/or (a, b) sub-proof ranges
    raw: str = ""           # original text for diagnostics


@dataclass
class Proof:
    premises: list[Formula]
    goal: Formula
    steps: list[Step]


# Rule alias map: accept multiple notations for the same canonical rule
_RULE_ALIASES: dict[str, str] = {}
def _add_aliases(canonical: str, *names: str) -> None:
    for n in names:
        _RULE_ALIASES[n] = canonical
        _RULE_ALIASES[n.upper()] = canonical
        _RULE_ALIASES[n.lower()] = canonical

_add_aliases("premise",  "premise", "prem")
_add_aliases("assume",   "assume", "ass", "hyp", "assumption")
_add_aliases("reit",     "reit", "reiterate", "copy")
_add_aliases("AND_I",    "AND_I", "&I", "∧I", "andI")
_add_aliases("AND_E1",   "AND_E1", "&E1", "∧E1", "andE1")
_add_aliases("AND_E2",   "AND_E2", "&E2", "∧E2", "andE2")
_add_aliases("OR_I1",    "OR_I1", "|I1", "∨I1", "orI1")
_add_aliases("OR_I2",    "OR_I2", "|I2", "∨I2", "orI2")
_add_aliases("OR_E",     "OR_E", "|E", "∨E", "orE")
_add_aliases("IMPL_I",   "IMPL_I", "->I", "→I", "implI", "implI.")
_add_aliases("IMPL_E",   "IMPL_E", "->E", "→E", "implE", "MP", "ModusPonens")
_add_aliases("NOT_I",    "NOT_I", "~I", "¬I", "notI")
_add_aliases("NOT_E",    "NOT_E", "~E", "¬E", "notE")
_add_aliases("BOT_E",    "BOT_E", "⊥E", "EFQ", "explosion", "exFalso")
_add_aliases("DNE",      "DNE", "DN", "doubleNeg")
_add_aliases("LEM",      "LEM", "EM", "excludedMiddle")


def _normalize_rule_name(name: str) -> str:
    if name in _RULE_ALIASES:
        return _RULE_ALIASES[name]
    if name.upper() in _RULE_ALIASES:
        return _RULE_ALIASES[name.upper()]
    return name


# Proof line: "  3. | | P -> Q   [IMPL_I 1-2]"
_PROOF_LINE_RE = re.compile(
    r"^\s*(\d+)\.\s*((?:\|\s*)*)([^\[\]]+?)\s*\[\s*([^\]]+?)\s*\]\s*$"
)


def parse_proof(text: str) -> Proof:
    lines = text.splitlines()
    start_idx = end_idx = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.upper() == "PROOF":
            start_idx = i + 1
        elif s.upper() == "END":
            end_idx = i
            break
    if start_idx is None:
        raise ParseError("missing PROOF header")
    if end_idx is None:
        raise ParseError("missing END terminator")

    body = lines[start_idx:end_idx]
    premises: list[Formula] = []
    goal: Optional[Formula] = None
    steps: list[Step] = []

    for raw_ln in body:
        ln = raw_ln.rstrip()
        s = ln.strip()
        if not s:
            continue
        low = s.lower()
        if low.startswith("premises:"):
            payload = s.split(":", 1)[1].strip()
            premises = _parse_formula_list(payload) if payload else []
            continue
        if low.startswith("goal:"):
            payload = s.split(":", 1)[1].strip()
            goal = parse_formula(payload)
            continue
        m = _PROOF_LINE_RE.match(ln)
        if not m:
            raise ParseError(f"malformed proof line: {ln!r}")
        line_no = int(m.group(1))
        bars = m.group(2) or ""
        indent = bars.count("|")
        formula = parse_formula(m.group(3).strip())
        rule_text = m.group(4).strip()
        rule, refs = _parse_rule_and_refs(rule_text)
        steps.append(Step(
            line=line_no, indent=indent, formula=formula,
            rule=rule, refs=refs, raw=ln,
        ))

    if goal is None:
        raise ParseError("missing 'goal:' line")
    return Proof(premises=premises, goal=goal, steps=steps)


def _parse_rule_and_refs(text: str) -> tuple[str, list[LineRef]]:
    parts = text.split(None, 1)
    if not parts:
        raise ParseError("empty rule annotation")
    rule = _normalize_rule_name(parts[0])
    refs: list[LineRef] = []
    if len(parts) > 1:
        for tok in parts[1].split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                if "-" in tok:
                    a, b = tok.split("-", 1)
                    refs.append((int(a.strip()), int(b.strip())))
                else:
                    refs.append(int(tok))
            except ValueError as exc:
                raise ParseError(
                    f"reference token {tok!r} in rule {rule!r} is not a "
                    f"line number or range: {exc}"
                ) from exc
    return rule, refs


# ═══════════════════════════════════════════════════════════════════════
#  Verifier — scope and rule semantics
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class _State:
    steps: list[Step]
    premises: list[Formula]
    alien_rules: set[str]


@dataclass
class VerifierResult:
    accepted: bool
    rejected_step: Optional[int] = None
    rejected_rule: Optional[str] = None
    reason_class: Optional[str] = None
    reason_id: Optional[str] = None
    detail: Optional[str] = None
    goal_match: bool = False
    n_steps: int = 0

    def to_dict(self, *, hide_rule_name: bool = False) -> dict[str, Any]:
        return {
            "accepted": bool(self.accepted and self.goal_match),
            "verifier_passed": bool(self.accepted),
            "goal_match": bool(self.goal_match),
            "rejected_step": self.rejected_step,
            "rejected_rule": None if hide_rule_name else self.rejected_rule,
            "reason_class": self.reason_class,
            "reason_id": self.reason_id,
            "detail": self.detail,
            "n_steps": self.n_steps,
        }


def _accessible(steps: list[Step], from_idx: int, to_line: int) -> bool:
    """Is `to_line` (1-indexed) accessible from steps[from_idx]?

    A line at indent k is in scope of a later line iff no intermediate line
    has indent < k (i.e., we never exited the box that contained line k).
    """
    if to_line < 1 or to_line - 1 >= from_idx:
        return False
    target_idx = to_line - 1
    target_indent = steps[target_idx].indent
    if steps[from_idx].indent < target_indent:
        return False
    for k in range(target_idx + 1, from_idx):
        if steps[k].indent < target_indent:
            return False
    return True


def _is_complete_subproof(steps: list[Step], a: int, b: int, from_idx: int) -> bool:
    """Is steps[a-1..b-1] a complete sub-proof referenced from from_idx?

    Requirements:
    - 1 <= a <= b < from_idx + 1 (range strictly before citing line)
    - line a uses 'assume' at indent k > 0
    - all lines in [a, b] have indent >= k
    - line b has indent exactly k (the conclusion of the sub-proof sits at
      the box's own level, not inside a deeper nested box)
    - the citing line is at indent k - 1

    We deliberately do NOT require the box to have been closed in
    [b+1, from_idx-1]; sibling sub-proofs (e.g. the second branch of an
    OR_E) live at indent k between the first sub-proof's end and the OR_E
    line itself, and forbidding indent >= k there would break standard
    Fitch case analysis. Cross-box leakage is still prevented by the
    per-line accessibility check.
    """
    if a < 1 or b < a or b >= from_idx + 1:
        return False
    a_idx, b_idx = a - 1, b - 1
    if b_idx >= from_idx:
        return False
    box_indent = steps[a_idx].indent
    if box_indent == 0 or steps[a_idx].rule != "assume":
        return False
    for k in range(a_idx, b_idx + 1):
        if steps[k].indent < box_indent:
            return False
    if steps[b_idx].indent != box_indent:
        return False
    if steps[from_idx].indent != box_indent - 1:
        return False
    return True


# ─── individual rule handlers ──────────────────────────────────────────

RuleErr = tuple[str, str, str]


def _err(cls: str, rid: str, detail: str) -> RuleErr:
    return (cls, rid, detail)


def _r_premise(state: _State, idx: int) -> Optional[RuleErr]:
    step = state.steps[idx]
    if step.refs:
        return _err("STRUCTURE", "EXTRANEOUS_REFS", "premise takes no refs")
    if step.indent != 0:
        return _err("STRUCTURE", "PREMISE_INSIDE_BOX",
                    "premise must be at top indent")
    if step.formula not in state.premises:
        return _err("RULE_VIOLATION", "FORMULA_NOT_IN_PREMISES",
                    f"{step.formula} is not declared as a premise")
    # Core well-formedness: a declared premise may be asserted at most as many
    # times as it appears in the premise list (normally once). Re-asserting a
    # premise on a fresh line would otherwise let a model mint duplicate
    # premise lines and dodge side conditions that count citations per line.
    # Always active, independent of the alien rule set.
    allowed = state.premises.count(step.formula)
    prior = sum(
        1 for k in range(idx)
        if state.steps[k].rule == "premise"
        and state.steps[k].formula == step.formula
    )
    if prior >= allowed:
        return _err("STRUCTURE", "DUPLICATE_PREMISE",
                    f"premise {step.formula} already asserted "
                    f"{prior} time(s); a declared premise may be asserted "
                    f"at most {allowed} time(s)")
    return None


def _r_assume(state: _State, idx: int) -> Optional[RuleErr]:
    """An 'assume' line opens a fresh assumption box.

    The new box may either:
      - nest deeper than the previous step (indent = prev.indent + 1), or
      - start a sibling sub-proof at the same depth as a previously-closed
        box (1 <= indent <= prev.indent + 1).

    The latter case happens in classic Fitch-style OR_E case analysis, where
    two parallel sub-proofs both live at indent k right after the OR-line at
    indent k - 1. We never allow skipping levels (indent > prev.indent + 1)
    or opening a top-level box (indent = 0).
    """
    step = state.steps[idx]
    if step.refs:
        return _err("STRUCTURE", "EXTRANEOUS_REFS", "assume takes no refs")
    if step.indent < 1:
        return _err("STRUCTURE", "BAD_ASSUME_POSITION",
                    "'assume' must be inside an assumption box (indent >= 1)")
    if idx == 0:
        if step.indent != 1:
            return _err("STRUCTURE", "BAD_ASSUME_POSITION",
                        "first 'assume' must be at indent 1")
        return None
    prev = state.steps[idx - 1]
    if step.indent > prev.indent + 1:
        return _err("STRUCTURE", "BAD_ASSUME_POSITION",
                    f"'assume' may open at most one new level "
                    f"(prev indent {prev.indent}, current {step.indent})")
    return None


def _r_reit(state: _State, idx: int) -> Optional[RuleErr]:
    step = state.steps[idx]
    if len(step.refs) != 1 or not isinstance(step.refs[0], int):
        return _err("STRUCTURE", "BAD_REFS", "reit takes exactly 1 line ref")
    a = step.refs[0]
    if not _accessible(state.steps, idx, a):
        return _err("RULE_VIOLATION", "REF_NOT_ACCESSIBLE",
                    f"line {a} is not in scope")
    if state.steps[a - 1].formula != step.formula:
        return _err("RULE_VIOLATION", "FORMULA_MISMATCH",
                    f"reit'd formula doesn't match line {a}")
    return None


def _r_and_intro(state: _State, idx: int) -> Optional[RuleErr]:
    step = state.steps[idx]
    if len(step.refs) != 2 or not all(isinstance(r, int) for r in step.refs):
        return _err("STRUCTURE", "BAD_REFS", "AND_I takes 2 line refs")
    a, b = step.refs  # type: ignore[misc]
    if not _accessible(state.steps, idx, a) or not _accessible(state.steps, idx, b):
        return _err("RULE_VIOLATION", "REF_NOT_ACCESSIBLE", "ref not in scope")
    fa, fb = state.steps[a - 1].formula, state.steps[b - 1].formula
    if not isinstance(step.formula, And):
        return _err("RULE_VIOLATION", "WRONG_CONCLUSION_SHAPE",
                    "AND_I must yield an AND formula")
    if step.formula.l != fa or step.formula.r != fb:
        return _err("RULE_VIOLATION", "FORMULA_MISMATCH",
                    f"AND_I should yield AND({fa}, {fb})")
    return None


def _r_and_elim1(state: _State, idx: int) -> Optional[RuleErr]:
    step = state.steps[idx]
    if len(step.refs) != 1 or not isinstance(step.refs[0], int):
        return _err("STRUCTURE", "BAD_REFS", "AND_E1 takes 1 line ref")
    a = step.refs[0]
    if not _accessible(state.steps, idx, a):
        return _err("RULE_VIOLATION", "REF_NOT_ACCESSIBLE", "ref not in scope")
    src = state.steps[a - 1].formula
    if not isinstance(src, And):
        return _err("RULE_VIOLATION", "WRONG_PREMISE_SHAPE",
                    f"AND_E1 expects AND, got {src}")
    if step.formula != src.l:
        return _err("RULE_VIOLATION", "FORMULA_MISMATCH",
                    f"AND_E1 should yield {src.l}")
    return None


def _r_and_elim2(state: _State, idx: int) -> Optional[RuleErr]:
    step = state.steps[idx]
    if len(step.refs) != 1 or not isinstance(step.refs[0], int):
        return _err("STRUCTURE", "BAD_REFS", "AND_E2 takes 1 line ref")
    a = step.refs[0]
    if not _accessible(state.steps, idx, a):
        return _err("RULE_VIOLATION", "REF_NOT_ACCESSIBLE", "ref not in scope")
    src = state.steps[a - 1].formula
    if not isinstance(src, And):
        return _err("RULE_VIOLATION", "WRONG_PREMISE_SHAPE",
                    f"AND_E2 expects AND, got {src}")
    if step.formula != src.r:
        return _err("RULE_VIOLATION", "FORMULA_MISMATCH",
                    f"AND_E2 should yield {src.r}")
    return None


def _r_or_intro1(state: _State, idx: int) -> Optional[RuleErr]:
    step = state.steps[idx]
    if len(step.refs) != 1 or not isinstance(step.refs[0], int):
        return _err("STRUCTURE", "BAD_REFS", "OR_I1 takes 1 line ref")
    a = step.refs[0]
    if not _accessible(state.steps, idx, a):
        return _err("RULE_VIOLATION", "REF_NOT_ACCESSIBLE", "ref not in scope")
    if not isinstance(step.formula, Or):
        return _err("RULE_VIOLATION", "WRONG_CONCLUSION_SHAPE",
                    "OR_I1 must yield OR")
    src = state.steps[a - 1].formula
    if step.formula.l != src:
        return _err("RULE_VIOLATION", "FORMULA_MISMATCH",
                    "OR_I1 left disjunct must equal cited formula")
    return None


def _r_or_intro2(state: _State, idx: int) -> Optional[RuleErr]:
    step = state.steps[idx]
    if len(step.refs) != 1 or not isinstance(step.refs[0], int):
        return _err("STRUCTURE", "BAD_REFS", "OR_I2 takes 1 line ref")
    a = step.refs[0]
    if not _accessible(state.steps, idx, a):
        return _err("RULE_VIOLATION", "REF_NOT_ACCESSIBLE", "ref not in scope")
    if not isinstance(step.formula, Or):
        return _err("RULE_VIOLATION", "WRONG_CONCLUSION_SHAPE",
                    "OR_I2 must yield OR")
    src = state.steps[a - 1].formula
    if step.formula.r != src:
        return _err("RULE_VIOLATION", "FORMULA_MISMATCH",
                    "OR_I2 right disjunct must equal cited formula")
    return None


def _r_or_elim(state: _State, idx: int) -> Optional[RuleErr]:
    step = state.steps[idx]
    if len(step.refs) != 3:
        return _err("STRUCTURE", "BAD_REFS",
                    "OR_E takes 1 line ref + 2 sub-proof ranges")
    a, r1, r2 = step.refs
    if not isinstance(a, int) or not isinstance(r1, tuple) or not isinstance(r2, tuple):
        return _err("STRUCTURE", "BAD_REFS",
                    "OR_E refs must be: int, range, range")
    if not _accessible(state.steps, idx, a):
        return _err("RULE_VIOLATION", "REF_NOT_ACCESSIBLE", "ref not in scope")
    src = state.steps[a - 1].formula
    if not isinstance(src, Or):
        return _err("RULE_VIOLATION", "WRONG_PREMISE_SHAPE",
                    f"OR_E expects OR, got {src}")
    if not _is_complete_subproof(state.steps, r1[0], r1[1], idx):
        return _err("RULE_VIOLATION", "INVALID_SUBPROOF_RANGE",
                    f"first sub-proof range {r1} is invalid")
    if not _is_complete_subproof(state.steps, r2[0], r2[1], idx):
        return _err("RULE_VIOLATION", "INVALID_SUBPROOF_RANGE",
                    f"second sub-proof range {r2} is invalid")
    if state.steps[r1[0] - 1].formula != src.l:
        return _err("RULE_VIOLATION", "WRONG_ASSUMPTION",
                    f"first sub-proof must assume {src.l}")
    if state.steps[r2[0] - 1].formula != src.r:
        return _err("RULE_VIOLATION", "WRONG_ASSUMPTION",
                    f"second sub-proof must assume {src.r}")
    if state.steps[r1[1] - 1].formula != step.formula:
        return _err("RULE_VIOLATION", "BRANCH_CONCLUSION_MISMATCH",
                    f"first branch must end with {step.formula}")
    if state.steps[r2[1] - 1].formula != step.formula:
        return _err("RULE_VIOLATION", "BRANCH_CONCLUSION_MISMATCH",
                    f"second branch must end with {step.formula}")
    return None


def _r_impl_intro(state: _State, idx: int) -> Optional[RuleErr]:
    step = state.steps[idx]
    if len(step.refs) != 1 or not isinstance(step.refs[0], tuple):
        return _err("STRUCTURE", "BAD_REFS", "IMPL_I takes 1 sub-proof range")
    rng = step.refs[0]
    if not _is_complete_subproof(state.steps, rng[0], rng[1], idx):
        return _err("RULE_VIOLATION", "INVALID_SUBPROOF_RANGE",
                    f"range {rng} is not a valid sub-proof")
    if not isinstance(step.formula, Impl):
        return _err("RULE_VIOLATION", "WRONG_CONCLUSION_SHAPE",
                    "IMPL_I must yield IMPL")
    assumed = state.steps[rng[0] - 1].formula
    derived = state.steps[rng[1] - 1].formula
    if step.formula.l != assumed:
        return _err("RULE_VIOLATION", "ANTECEDENT_MISMATCH",
                    f"IMPL_I antecedent must equal assumption {assumed}")
    if step.formula.r != derived:
        return _err("RULE_VIOLATION", "CONSEQUENT_MISMATCH",
                    f"IMPL_I consequent must equal {derived}")
    return None


def _r_impl_elim(state: _State, idx: int) -> Optional[RuleErr]:
    step = state.steps[idx]
    if len(step.refs) != 2 or not all(isinstance(r, int) for r in step.refs):
        return _err("STRUCTURE", "BAD_REFS",
                    "IMPL_E (MP) takes 2 line refs")
    a, b = step.refs  # type: ignore[misc]
    if not _accessible(state.steps, idx, a) or not _accessible(state.steps, idx, b):
        return _err("RULE_VIOLATION", "REF_NOT_ACCESSIBLE", "ref not in scope")
    fa, fb = state.steps[a - 1].formula, state.steps[b - 1].formula
    impl: Optional[Impl] = None
    if isinstance(fa, Impl) and fa.l == fb:
        impl = fa
    elif isinstance(fb, Impl) and fb.l == fa:
        impl = fb
    if impl is None:
        return _err("RULE_VIOLATION", "MP_NO_MATCH",
                    f"no IMPL/antecedent pairing among {fa}, {fb}")
    if step.formula != impl.r:
        return _err("RULE_VIOLATION", "CONSEQUENT_MISMATCH",
                    f"IMPL_E should yield {impl.r}")
    return None


def _r_not_intro(state: _State, idx: int) -> Optional[RuleErr]:
    step = state.steps[idx]
    if len(step.refs) != 1 or not isinstance(step.refs[0], tuple):
        return _err("STRUCTURE", "BAD_REFS", "NOT_I takes 1 sub-proof range")
    rng = step.refs[0]
    if not _is_complete_subproof(state.steps, rng[0], rng[1], idx):
        return _err("RULE_VIOLATION", "INVALID_SUBPROOF_RANGE",
                    f"range {rng} is not a valid sub-proof")
    if not isinstance(step.formula, Not):
        return _err("RULE_VIOLATION", "WRONG_CONCLUSION_SHAPE",
                    "NOT_I must yield NOT")
    assumed = state.steps[rng[0] - 1].formula
    derived = state.steps[rng[1] - 1].formula
    if step.formula.sub != assumed:
        return _err("RULE_VIOLATION", "ASSUMPTION_MISMATCH",
                    f"NOT_I assumption must equal {step.formula.sub}")
    if not isinstance(derived, Bot):
        return _err("RULE_VIOLATION", "EXPECTED_BOT",
                    f"NOT_I sub-proof must end with BOT, got {derived}")
    return None


def _r_not_elim(state: _State, idx: int) -> Optional[RuleErr]:
    step = state.steps[idx]
    if len(step.refs) != 2 or not all(isinstance(r, int) for r in step.refs):
        return _err("STRUCTURE", "BAD_REFS", "NOT_E takes 2 line refs")
    a, b = step.refs  # type: ignore[misc]
    if not _accessible(state.steps, idx, a) or not _accessible(state.steps, idx, b):
        return _err("RULE_VIOLATION", "REF_NOT_ACCESSIBLE", "ref not in scope")
    fa, fb = state.steps[a - 1].formula, state.steps[b - 1].formula
    contradicts = (
        (isinstance(fb, Not) and fb.sub == fa)
        or (isinstance(fa, Not) and fa.sub == fb)
    )
    if not contradicts:
        return _err("RULE_VIOLATION", "NO_CONTRADICTION",
                    f"{fa} and {fb} are not contradictory")
    if not isinstance(step.formula, Bot):
        return _err("RULE_VIOLATION", "EXPECTED_BOT",
                    "NOT_E must yield BOT")
    return None


def _r_bot_elim(state: _State, idx: int) -> Optional[RuleErr]:
    step = state.steps[idx]
    if len(step.refs) != 1 or not isinstance(step.refs[0], int):
        return _err("STRUCTURE", "BAD_REFS", "BOT_E takes 1 line ref")
    a = step.refs[0]
    if not _accessible(state.steps, idx, a):
        return _err("RULE_VIOLATION", "REF_NOT_ACCESSIBLE", "ref not in scope")
    src = state.steps[a - 1].formula
    if not isinstance(src, Bot):
        return _err("RULE_VIOLATION", "WRONG_PREMISE_SHAPE",
                    f"BOT_E expects BOT, got {src}")
    return None


def _r_dne(state: _State, idx: int) -> Optional[RuleErr]:
    step = state.steps[idx]
    if len(step.refs) != 1 or not isinstance(step.refs[0], int):
        return _err("STRUCTURE", "BAD_REFS", "DNE takes 1 line ref")
    a = step.refs[0]
    if not _accessible(state.steps, idx, a):
        return _err("RULE_VIOLATION", "REF_NOT_ACCESSIBLE", "ref not in scope")
    src = state.steps[a - 1].formula
    if not (isinstance(src, Not) and isinstance(src.sub, Not)):
        return _err("RULE_VIOLATION", "WRONG_PREMISE_SHAPE",
                    f"DNE expects NOT(NOT(_)), got {src}")
    if step.formula != src.sub.sub:
        return _err("RULE_VIOLATION", "FORMULA_MISMATCH",
                    f"DNE should yield {src.sub.sub}")
    return None


def _r_lem(state: _State, idx: int) -> Optional[RuleErr]:
    step = state.steps[idx]
    if step.refs:
        return _err("STRUCTURE", "EXTRANEOUS_REFS", "LEM takes no refs")
    if not isinstance(step.formula, Or):
        return _err("RULE_VIOLATION", "WRONG_CONCLUSION_SHAPE",
                    "LEM yields OR(φ, NOT(φ))")
    r = step.formula.r
    if not (isinstance(r, Not) and step.formula.l == r.sub):
        return _err("RULE_VIOLATION", "WRONG_LEM_SHAPE",
                    "LEM must be OR(φ, NOT(φ))")
    return None


_RULE_HANDLERS: dict[str, Callable[[_State, int], Optional[RuleErr]]] = {
    "premise":  _r_premise,
    "assume":   _r_assume,
    "reit":     _r_reit,
    "AND_I":    _r_and_intro,
    "AND_E1":   _r_and_elim1,
    "AND_E2":   _r_and_elim2,
    "OR_I1":    _r_or_intro1,
    "OR_I2":    _r_or_intro2,
    "OR_E":     _r_or_elim,
    "IMPL_I":   _r_impl_intro,
    "IMPL_E":   _r_impl_elim,
    "NOT_I":    _r_not_intro,
    "NOT_E":    _r_not_elim,
    "BOT_E":    _r_bot_elim,
    "DNE":      _r_dne,
    "LEM":      _r_lem,
}

STANDARD_RULES: list[str] = list(_RULE_HANDLERS.keys())


# ═══════════════════════════════════════════════════════════════════════
#  Rule pack
# ═══════════════════════════════════════════════════════════════════════

# The registry is filled at the end of this module, once every name a hook may
# use is defined.
ALIEN_RULE_REGISTRY: dict[str, dict[str, Any]] = {}


def _run_alien_hooks(state: _State, idx: int) -> Optional[RuleErr]:
    """Run every active alien hook on step `idx` and report violations.

    Iteration follows ``ALIEN_RULE_REGISTRY`` insertion order, NOT
    ``set`` order. This is required for reproducibility: when a single step
    violates more than one alien rule, the opaque ``reason_id`` returned to the
    model must be stable across processes. Iterating ``state.alien_rules``
    (a set) made the reported code depend on ``PYTHONHASHSEED``, so the same
    probe could return one code in one worker and another code elsewhere.

    Determinism contract:
      - the *primary* reported code is the lowest-id (earliest-registered)
        violated rule;
      - when several rules fire, all violated codes are appended to `detail`
        (visible only when hide_reason_detail=False) for transparency.
    """
    errs: list[RuleErr] = []
    for name, spec in ALIEN_RULE_REGISTRY.items():
        if name not in state.alien_rules:
            continue
        err = spec["hook"](state, idx)
        if err is not None:
            errs.append(err)
    if not errs:
        return None
    if len(errs) == 1:
        return errs[0]
    cls, rid, detail = errs[0]
    all_ids = ",".join(e[1] for e in errs)
    return _err(cls, rid, f"{detail} [also violates: {all_ids}]")




# ═══════════════════════════════════════════════════════════════════════
#  Top-level verifier
# ═══════════════════════════════════════════════════════════════════════

def verify_proof(
    proof: Proof,
    alien_rules: Iterable[str] = (),
    *,
    hide_reason_detail: bool = True,
) -> VerifierResult:
    """Verify a proof under the given alien rule set.

    `hide_reason_detail` controls whether the human-readable detail is
    returned. By default the verifier returns only `reason_class` and
    `reason_id` (opaque codes), forcing the model to triangulate.
    """
    rules = set(alien_rules)
    n = len(proof.steps)

    if n == 0:
        return VerifierResult(False, rejected_step=0, rejected_rule="(none)",
                              reason_class="STRUCTURE",
                              reason_id="EMPTY_PROOF",
                              detail=None if hide_reason_detail else "no steps",
                              n_steps=0)

    for i, step in enumerate(proof.steps):
        if step.line != i + 1:
            return VerifierResult(
                False, rejected_step=step.line, rejected_rule=step.rule,
                reason_class="STRUCTURE",
                reason_id="NONSEQUENTIAL_LINE_NUMBER",
                detail=None if hide_reason_detail else
                    f"expected line {i+1}, got {step.line}",
                n_steps=n,
            )

    state = _State(steps=proof.steps, premises=proof.premises, alien_rules=rules)

    for i, step in enumerate(proof.steps):
        handler = _RULE_HANDLERS.get(step.rule)
        if handler is None:
            return VerifierResult(
                False, rejected_step=step.line, rejected_rule=step.rule,
                reason_class="UNKNOWN_RULE",
                reason_id="RULE_NOT_REGISTERED",
                detail=None if hide_reason_detail else
                    f"rule {step.rule!r} is not part of the proof system",
                n_steps=n,
            )
        err = handler(state, i)
        if err is None:
            err = _run_alien_hooks(state, i)
        if err is not None:
            cls, rid, detail = err
            return VerifierResult(
                False, rejected_step=step.line, rejected_rule=step.rule,
                reason_class=cls, reason_id=rid,
                detail=None if hide_reason_detail else detail,
                n_steps=n,
            )

    last = proof.steps[-1]
    if last.indent != 0:
        return VerifierResult(
            True, goal_match=False, n_steps=n,
            reason_class="STRUCTURE", reason_id="ENDED_INSIDE_BOX",
            detail=None if hide_reason_detail else
                "proof ended inside an open sub-proof",
        )
    goal_match = (last.formula == proof.goal)
    return VerifierResult(True, goal_match=goal_match, n_steps=n)


def proof_diagnostic(
    text: str,
    alien_rules: Iterable[str] = (),
    *,
    hide_reason_detail: bool = True,
    hide_rule_name: Optional[bool] = None,
    expected_goal: Optional[str] = None,
    expected_premises: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    """Parse + verify a proof block. Returns a JSON-serialisable diagnostic.

    This is the primary entry-point used by the eval harness for both
    explore-time CHECK_PROOF queries and milestone test scoring.

    `hide_reason_detail` controls whether the human-readable detail string is
    returned (default True, opaque codes only).
    `hide_rule_name` controls whether `rejected_rule` is returned. If left as
    None it defaults to `hide_reason_detail`, i.e. the rule name is shown only
    when the detail is shown (seed phase). This avoids leaking the canonical
    rule that an alien side condition is attached to.
    `expected_goal` (optional, textual formula): if given, the proof block's
    declared `goal:` header MUST parse to the same Formula or the diagnostic
    is rejected with reason_id=GOAL_HEADER_MISMATCH. This prevents a model
    from "switching" the goal of an unprovable theorem to a different
    provable tautology and slipping past the verifier.
    `expected_premises` (optional, list of textual formulas): if given, the
    proof block may only declare premises drawn from this assigned set;
    introducing any *foreign* premise (one not in the assignment) is rejected
    with reason_id=PREMISE_HEADER_MISMATCH. This is the symmetric counterpart
    to the goal guard: it stops a model from "switching" the premises of a
    theorem to an easier set (e.g. replacing `OR(AND(p,q),r), IMPL(r,p)` with
    just `AND(p,q)`) and proving the goal trivially. A *subset* of the assigned
    premises is allowed: dropping a given premise only weakens the proof, so it
    can never turn an unprovable theorem provable.
    """
    if hide_rule_name is None:
        hide_rule_name = hide_reason_detail
    try:
        proof = parse_proof(text)
    except ParseError as exc:
        return {
            "accepted": False,
            "verifier_passed": False,
            "goal_match": False,
            "rejected_step": None,
            "rejected_rule": None,
            "reason_class": "PARSE_ERROR",
            "reason_id": "MALFORMED_PROOF",
            "detail": None if hide_reason_detail else str(exc),
            "n_steps": 0,
        }
    if expected_goal is not None:
        try:
            expected = parse_formula(expected_goal)
        except ParseError as exc:
            return {
                "accepted": False,
                "verifier_passed": False,
                "goal_match": False,
                "rejected_step": None,
                "rejected_rule": None,
                "reason_class": "INTERNAL",
                "reason_id": "EXPECTED_GOAL_PARSE_ERROR",
                "detail": None if hide_reason_detail else str(exc),
                "n_steps": 0,
            }
        if proof.goal != expected:
            return {
                "accepted": False,
                "verifier_passed": False,
                "goal_match": False,
                "rejected_step": None,
                "rejected_rule": None,
                "reason_class": "ANSWER_MISMATCH",
                "reason_id": "GOAL_HEADER_MISMATCH",
                "detail": None if hide_reason_detail else
                    f"submitted goal {proof.goal!r} != theorem goal {expected!r}",
                "n_steps": 0,
            }
    if expected_premises is not None:
        try:
            allowed = [parse_formula(p) for p in expected_premises]
        except ParseError as exc:
            return {
                "accepted": False,
                "verifier_passed": False,
                "goal_match": False,
                "rejected_step": None,
                "rejected_rule": None,
                "reason_class": "INTERNAL",
                "reason_id": "EXPECTED_PREMISE_PARSE_ERROR",
                "detail": None if hide_reason_detail else str(exc),
                "n_steps": 0,
            }
        foreign = [p for p in proof.premises if p not in allowed]
        if foreign:
            return {
                "accepted": False,
                "verifier_passed": False,
                "goal_match": False,
                "rejected_step": None,
                "rejected_rule": None,
                "reason_class": "ANSWER_MISMATCH",
                "reason_id": "PREMISE_HEADER_MISMATCH",
                "detail": None if hide_reason_detail else
                    f"proof declares premise(s) {foreign!r} not in the "
                    f"assigned set {allowed!r}",
                "n_steps": 0,
            }
    result = verify_proof(proof, alien_rules, hide_reason_detail=hide_reason_detail)
    return result.to_dict(hide_rule_name=hide_rule_name)


# ═══════════════════════════════════════════════════════════════════════
#  Reference manual (shown to the model)
# ═══════════════════════════════════════════════════════════════════════

REFERENCE_MANUAL = """\
AlienLogic Proof-Script — Proof Reference Manual
=======================================

You write proofs in a Fitch-style natural deduction calculus. Each proof
is a numbered sequence of lines. Indentation marks assumption boxes:
  - depth 0 means top level
  - each '|' prefix opens one further depth of assumption box
  - 'assume' must open a fresh box (indent goes from k to k+1)
  - a sub-proof closes when the next line drops indent by at least 1
  - lines inside a closed sub-proof are not directly accessible; they can
    only be cited through a range like [IMPL_I 3-5]

────────── Formula syntax ──────────
  Atoms       p, q, r, s, t, u, ...
  Constant    BOT
  Operators   NOT(phi), AND(phi, psi), OR(phi, psi), IMPL(phi, psi)

────────── Inference rules ──────────
  premise               assert a declared premise (top indent only)
  assume                open a new assumption box (no refs)
  reit n                copy line n into current scope
  AND_I a, b            from φ and ψ derive AND(φ, ψ)
  AND_E1 a              from AND(φ, ψ) derive φ
  AND_E2 a              from AND(φ, ψ) derive ψ
  OR_I1 a               from φ derive OR(φ, ψ)        (right disjunct chosen freely)
  OR_I2 a               from ψ derive OR(φ, ψ)        (left disjunct chosen freely)
  OR_E a, b-c, d-e      OR-elim by case analysis on line a
  IMPL_I a-b            discharge sub-proof a..b (assume φ, derive ψ) ⇒ IMPL(φ, ψ)
  IMPL_E a, b           Modus Ponens: from IMPL(φ, ψ) and φ derive ψ
  NOT_I a-b             discharge sub-proof a..b (assume φ, derive BOT) ⇒ NOT(φ)
  NOT_E a, b            from φ and NOT(φ) derive BOT
  BOT_E a               from BOT derive any formula (Explosion)
  DNE a                 from NOT(NOT(φ)) derive φ
  LEM                   axiom: derive OR(φ, NOT(φ)) for any φ

────────── Proof script format ──────────
PROOF
premises: <comma-separated list of formulas>
goal: <formula>
1. <formula>           [<rule> <refs>]
2. | <formula>         [<rule> <refs>]
...
END

Reference annotation conventions:
  - line refs are integers (e.g.  3, 7)
  - sub-proof ranges are 'a-b' (e.g.  3-5)
  - separate multiple refs with commas (e.g.  1, 2)

────────── Important notice ──────────
The verifier may enforce hidden side conditions on top of the rules listed
above. A rejected step returns a structured diagnostic with:
  reason_class : broad category (RULE_VIOLATION / ALIEN_SIDE_CONDITION /
                 ALIEN_AXIOM_DISABLED / STRUCTURE / PARSE_ERROR / ...)
  reason_id    : opaque code identifying the specific failure mode
                 (e.g. "K3", "K7"); rule names are NOT revealed
You should treat the manual as the *standard* proof system. The actual
verifier may differ in subtle ways. Submit small test proofs to discover
which rules carry hidden constraints, then apply that knowledge to the
held-out theorems.

────────── Worked example A (Modus Ponens) ──────────
PROOF
premises: IMPL(p, q), p
goal: q
1. IMPL(p, q)          [premise]
2. p                   [premise]
3. q                   [IMPL_E 1, 2]
END

────────── Worked example B (case analysis with OR_E) ──────────
Two parallel sub-proofs both at indent 1 are written one after the other;
each begins with its own 'assume' line. Both must close with the same
formula. The OR_E line then sits at indent 0 and cites the OR-line plus
the two sub-proof ranges.

PROOF
premises: OR(p, q), IMPL(p, r), IMPL(q, r)
goal: r
1. OR(p, q)            [premise]
2. IMPL(p, r)          [premise]
3. IMPL(q, r)          [premise]
4. | p                 [assume]
5. | r                 [IMPL_E 2, 4]
6. | q                 [assume]
7. | r                 [IMPL_E 3, 6]
8. r                   [OR_E 1, 4-5, 6-7]
END
"""


# ═══════════════════════════════════════════════════════════════════════
#  Load the rule pack
# ═══════════════════════════════════════════════════════════════════════

import sys as _sys  # noqa: E402

from sandboxes.logic import rules as _rule_pack  # noqa: E402

ALIEN_RULE_REGISTRY.update(_rule_pack.build_registry(_sys.modules[__name__]))
