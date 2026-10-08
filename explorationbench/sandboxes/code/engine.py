"""Rule pack of the public AlienCode demo world.

Five discovery targets: four operations that differ from the reference manual
and one that does not. They were written for this release, differ from the
evaluation world operation by operation, and never enter a reported score.
See ``execution.py`` for the interface a pack provides.
"""

from __future__ import annotations

import ast

EXEC_TIMEOUT = 10

INT_OFFSET = 7

#: Rule keys switched back to the manual's behaviour. Empty in a real run; the
#: world build uses it to find which rules each task depends on.
DISABLED: set[str] = set()


def _on(key: str) -> bool:
    return key not in DISABLED

ALIEN_RULE_SPECS = [
    {"id": "R01", "layer": 1, "key": "int_offset", "name": "Integer offset",
     "std": "Integers pass through unchanged",
     "actual": f"Integer literal n -> n + {INT_OFFSET}", "identity": False},
    {"id": "R02", "layer": 3, "key": "under_le", "name": "UNDER is non-strict",
     "std": "UNDER(a,b) = (a < b)", "actual": "UNDER(a,b) = (a <= b)", "identity": False},
    {"id": "R03", "layer": 5, "key": "pluck_next", "name": "PLUCK reads one further",
     "std": "PLUCK(seq, i) -> seq[i]", "actual": "PLUCK(seq, i) -> seq[i + 1]",
     "identity": False},
    {"id": "R04", "layer": 5, "key": "strand_identity", "name": "STRAND unchanged",
     "std": "STRAND(a,b,c) = [a,b,c]", "actual": "STRAND(a,b,c) = [a,b,c] (same as manual)",
     "identity": True},
    {"id": "R05", "layer": 7, "key": "gauge_double", "name": "GAUGE doubles",
     "std": "GAUGE(seq) = length", "actual": "GAUGE(seq) = 2 * length", "identity": False},
]


def _helper(name: str, args: list) -> ast.Call:
    return ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=args, keywords=[])


class LiteralOffset(ast.NodeTransformer):
    """R01: integer literals are offset."""

    def visit_Constant(self, node):
        value = node.value
        if isinstance(value, int) and not isinstance(value, bool) and _on("int_offset"):
            return ast.copy_location(ast.Constant(value=value + INT_OFFSET), node)
        return node


class Comparisons(ast.NodeTransformer):
    """R02: UNDER admits equality."""

    def visit_Compare(self, node):
        self.generic_visit(node)
        if _on("under_le"):
            node.ops = [ast.LtE() if isinstance(op, ast.Lt) else op for op in node.ops]
        return node


class Positions(ast.NodeTransformer):
    """R03: PLUCK reads one position further."""

    def visit_Subscript(self, node):
        self.generic_visit(node)
        if not isinstance(node.ctx, ast.Load) or isinstance(node.slice, ast.Slice):
            return node
        if _on("pluck_next"):
            node.slice = _helper("_demo_next_index", [node.slice])
        return node


class Builtins(ast.NodeTransformer):
    """R05: GAUGE doubles."""

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "len" and _on("gauge_double"):
            node.func = ast.copy_location(ast.Name(id="_demo_len", ctx=ast.Load()), node.func)
        return node


LAYERS = [
    ("D1 literals", LiteralOffset),
    ("D2 comparisons", Comparisons),
    ("D3 positions", Positions),
    ("D4 builtins", Builtins),
]


def _demo_next_index(i):
    return i + 1


def _demo_len(obj):
    return 2 * len(obj)


def runtime_helpers() -> dict:
    return {
        "_demo_next_index": _demo_next_index,
        "_demo_len": _demo_len,
        "_alien_place": _place,
    }


def _place(seq, i, value):
    seq[i] = value


def emit_order(args):
    return args
