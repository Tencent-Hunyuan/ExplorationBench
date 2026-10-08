"""
AlienCode front end
===================

Everything about AlienCode that the reference manual defines: the surface
syntax, its translation to Python with the manual's semantics, and sandboxed
execution. A world is this front end plus a rule pack, the module
``sandboxes.code.engine``, which rewrites the translated program before it
runs. The pack exposes

    LAYERS            ordered (name, ast.NodeTransformer subclass) pairs
    runtime_helpers() names the rewritten program may call
    emit_order(args)  the order in which EMIT prints its arguments
    ALIEN_RULE_SPECS  the discovery targets: id, key, name, std, actual, identity
    EXEC_TIMEOUT      seconds a program may run

Pipeline:
  1. Text pre-process   -- AlienCode keywords -> Python tokens
  2. Python parse        -- source -> AST (hardened mode: refuse Python internals)
  3. AlienCode desugar   -- operation calls -> Python operators (manual semantics)
  4. Constant folding    -- fold  -n  into Constant(-n)
  5. Rule pack           -- the world's LAYERS, in order
  6. Output routing      -- EMIT writes to the captured buffer
  7. Sandboxed execution -- capture stdout

The released pack is the public demo world. The evaluation world is a private
pack with the same interface run by this front end.
"""

from __future__ import annotations

import ast
import re
import os
import sys
import signal
import threading
import builtins
import ctypes
import tokenize
from io import StringIO

# ══════════════════════════════════════════════════════════════════
#  Rule pack
# ══════════════════════════════════════════════════════════════════

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sandboxes.code.engine import (  # noqa: E402
    ALIEN_RULE_SPECS,
    EXEC_TIMEOUT,
    LAYERS as ALIEN_LAYERS,
    emit_order,
    runtime_helpers,
)


class FoldNegativeConstants(ast.NodeTransformer):
    """Fold  -n  into Constant(-n), so a rule pack sees one literal, not two nodes."""

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if (
            isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float))
        ):
            return ast.copy_location(ast.Constant(value=-node.operand.value), node)
        return node


class RouteOutput(ast.NodeTransformer):
    """Send every print the program still makes to the captured EMIT buffer."""

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            node.func = ast.copy_location(ast.Name(id="_alien_print", ctx=ast.Load()), node.func)
        return node


def _blocked_import(name, *args, **kwargs):
    raise ImportError(f"AlienCode has no module system (tried to import {name})")


#: A hosted evaluation runs programs from models it does not trust, in a process
#: that can read the private task data. ALIENCODE_HARDENED=1, which the
#: submission service sets, confines a program to the language the manual
#: describes: whitelisted builtins only, and no name or attribute that reaches
#: Python internals. It is off by default, the mode every reported
#: result was scored in, and no program in the experiment logs changes its
#: output under it.
HARDENED = os.environ.get("ALIENCODE_HARDENED", "0").strip().lower() in {"1", "true", "yes", "on"}

SAFE_BUILTINS = (
    "abs", "all", "any", "ascii", "bin", "bool", "bytes", "callable", "chr", "complex",
    "dict", "divmod", "enumerate", "filter", "float", "frozenset", "hash", "hex", "int",
    "isinstance", "issubclass", "iter", "len", "list", "map", "max", "min", "next", "oct",
    "ord", "pow", "print", "range", "repr", "reversed", "round", "set", "slice", "sorted",
    "str", "sum", "tuple", "type", "zip", "True", "False", "None", "NotImplemented", "Ellipsis",
    "ArithmeticError", "AssertionError", "AttributeError", "Exception", "IndexError",
    "KeyError", "LookupError", "OverflowError", "RecursionError", "RuntimeError",
    "StopIteration", "TypeError", "ValueError", "ZeroDivisionError",
    # A class statement needs these two to build the class and name its module.
    "__build_class__", "__name__",
)
#: Attributes that lead from a value to frames, code, or globals, and the string
#: methods that perform attribute lookups named inside a format string.
BLOCKED_ATTRIBUTES = frozenset({
    "gi_frame", "gi_code", "gi_yieldfrom", "cr_frame", "cr_code", "cr_await", "ag_frame",
    "ag_code", "ag_await", "f_back", "f_globals", "f_locals", "f_builtins", "f_code",
    "tb_frame", "tb_next", "co_code", "format", "format_map", "mro",
})
#: The one name the keyword table produces that would otherwise be refused (FUSE).
_KEYWORD_TARGETS = frozenset({"__alien_merge__"})
_BLOCKED_NODES = (ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith)


class SandboxViolation(Exception):
    pass


def _dunder(name: str) -> bool:
    return name.startswith("__") and name not in _KEYWORD_TARGETS


def check_program(tree: ast.AST) -> None:
    """Refuse, in hardened mode, a program that reaches for Python internals.

    Imports need no rule here: the import hook refuses them when they run, as
    it always has. With getattr and str.format gone, a string cannot name an
    attribute either, so only names and attribute nodes need checking.
    """
    for node in ast.walk(tree):
        if isinstance(node, _BLOCKED_NODES):
            raise SandboxViolation(f"{type(node).__name__} is not part of AlienCode")
        if isinstance(node, ast.Name) and _dunder(node.id):
            raise SandboxViolation(f"name {node.id!r} is not available")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("__") or node.attr in BLOCKED_ATTRIBUTES):
            raise SandboxViolation(f"attribute {node.attr!r} is not available")
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.arg)):
            name = node.arg if isinstance(node, ast.arg) else node.name
            if _dunder(name):
                raise SandboxViolation(f"name {name!r} is not available")


def _execution_env() -> dict:
    # Private builtins per execution: programs run on several threads in one
    # process, and sharing the live builtins mapping would let one program
    # redefine a builtin for all the others. Imports are blocked for the same
    # reason.
    if HARDENED:
        safe_builtins = {name: getattr(builtins, name) for name in SAFE_BUILTINS}
    else:
        safe_builtins = dict(vars(builtins))
    safe_builtins["__import__"] = _blocked_import
    env = {"__builtins__": safe_builtins}
    env.update(runtime_helpers())
    return env


# ══════════════════════════════════════════════════════════════════
#  Constants & keyword tables
# ══════════════════════════════════════════════════════════════════

FUNC_KEYWORD_MAP = {
    "EMIT":     "print",
    "EXTENT":   "range",
    "INDEX":    "enumerate",
    "PAIR":     "zip",
    "STRAIN":   "filter",
    "GAUGE":    "len",
    "NADIR":    "min",
    "APEX":     "max",
    "SOME_OF":  "any",
    "EVERY_OF": "all",
    "ORDER":    "sorted",
    "GATHER":   "list",
    "FUSE":     "__alien_merge__",
    "IMPRINT":  "_alien_place",
}

# ══════════════════════════════════════════════════════════════════
#  Reference manual  (the text shown to models in the prompt)
#  Single source of truth: manual.md, which sits next to this file.
#  README advertises manual.md as "the only thing the model sees", so we
#  load it here instead of duplicating the spec as a string literal.
# ══════════════════════════════════════════════════════════════════

_MANUAL_PATH = os.path.join(_THIS_DIR, "manual.md")
try:
    with open(_MANUAL_PATH, encoding="utf-8") as _mf:
        REFERENCE_MANUAL = _mf.read().strip()
except OSError as _e:  # pragma: no cover - manual.md ships alongside this file
    raise RuntimeError(
        f"AlienCode reference manual not found at {_MANUAL_PATH!r}; "
        "manual.md must sit next to execution.py."
    ) from _e


# ══════════════════════════════════════════════════════════════════
#  Stage 1 — Text pre-processor  (AlienCode → Python source)
# ══════════════════════════════════════════════════════════════════

_LINE_PATTERNS: list[tuple[re.Pattern, str | None]] = [
    (re.compile(r"SET\s+([\w,\s]+?)\s+AS\s+(.*)"),   r"\1 = \2"),
    (re.compile(r"SWEEP\s+(.+?)\s+IN\s+(.*)"),       r"for \1 in \2"),
    (re.compile(r"UPON\s+(.*)"),                      r"if \1"),
    (re.compile(r"LEST\s+(.*)"),                      r"elif \1"),
    (re.compile(r"WHILE\s+(.*)"),                     r"while \1"),
    (re.compile(r"CRAFT\s+(.*)"),                     r"def \1"),
]

_DELIVER_RE = re.compile(r"DELIVER\b\s*(.*)")
_CEASE_RE   = re.compile(r"CEASE\b")
_BYPASS_RE  = re.compile(r"BYPASS\b")
_IDLE_RE    = re.compile(r"IDLE\b")
_DEFAULT_RE = re.compile(r"DEFAULT\s*:")

_FUNC_RE_CACHE: dict[str, re.Pattern] = {}

def _func_re(name: str) -> re.Pattern:
    if name not in _FUNC_RE_CACHE:
        _FUNC_RE_CACHE[name] = re.compile(r"\b" + re.escape(name) + r"\s*\(")
    return _FUNC_RE_CACHE[name]


def _literal_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of string literals and comments in `text`.

    Keyword rewriting must skip these: `EMIT("YES")` is a request to print the
    three characters Y-E-S, not a boolean. Partial results are kept when
    tokenizing fails part-way (unbalanced delimiters raise at EOF, by which
    point the earlier spans are already correct).
    """
    line_starts = [0]
    for line in text.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))
    spans: list[tuple[int, int]] = []
    try:
        for tok in tokenize.generate_tokens(StringIO(text).readline):
            if tok.type in (tokenize.STRING, tokenize.COMMENT):
                (r1, c1), (r2, c2) = tok.start, tok.end
                spans.append((line_starts[r1 - 1] + c1,
                              line_starts[r2 - 1] + c2))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return spans


def _sub_outside_literals(text: str, apply) -> str:
    """Run `apply` on the code parts of `text`, leaving literals untouched.

    Safe to apply piecewise: every pattern involved (`\\bYES\\b`,
    `\\bNAME\\s*\\(`) lies wholly outside quotes, so no match can straddle a
    literal boundary.
    """
    spans = _literal_spans(text)
    if not spans:
        return apply(text)
    out: list[str] = []
    pos = 0
    for start, end in spans:
        out.append(apply(text[pos:start]))
        out.append(text[start:end])
        pos = end
    out.append(apply(text[pos:]))
    return "".join(out)


def alien_preprocess(source: str) -> str:
    """Convert AlienCode source to valid Python source via text transforms."""
    lines = source.split("\n")
    result: list[str] = []

    for line in lines:
        if not line.strip():
            result.append(line)
            continue

        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]

        if stripped.startswith("#"):
            result.append(line)
            continue

        if _DEFAULT_RE.match(stripped):
            result.append(f"{indent}else:")
            continue

        if _CEASE_RE.match(stripped):
            result.append(f"{indent}break")
            continue
        if _BYPASS_RE.match(stripped):
            result.append(f"{indent}continue")
            continue
        if _IDLE_RE.match(stripped):
            result.append(f"{indent}pass")
            continue

        m = _DELIVER_RE.match(stripped)
        if m:
            expr = m.group(1).strip()
            result.append(f"{indent}return {expr}" if expr else f"{indent}return")
            continue

        matched = False
        for pat, repl in _LINE_PATTERNS:
            m = pat.match(stripped)
            if m:
                expanded = m.expand(repl).strip()
                result.append(f"{indent}{expanded}")
                matched = True
                break
        if matched:
            continue

        result.append(line)

    text = "\n".join(result)

    def rewrite_keywords(code: str) -> str:
        code = re.sub(r"\bYES\b", "True", code)
        code = re.sub(r"\bNO\b", "False", code)
        for alien_name, py_name in FUNC_KEYWORD_MAP.items():
            code = _func_re(alien_name).sub(py_name + "(", code)
        return code

    return _sub_outside_literals(text, rewrite_keywords)


# ══════════════════════════════════════════════════════════════════
#  Stage 3 — AST desugarer  (AlienCode calls → Python operators)
# ══════════════════════════════════════════════════════════════════

class AlienDesugarer(ast.NodeTransformer):

    BINOP = {
        "SHATTER":  ast.Add,
        "WEAVE":    ast.Sub,
        "PARE":     ast.Mult,
        "FRACTURE": ast.Div,
        "COIL":     ast.Pow,
        "RESIDUE":  ast.Mod,
        "HALVE":    ast.FloorDiv,
    }

    CMPOP = {
        "AKIN":    ast.Eq,
        "APART":   ast.NotEq,
        "OVER":    ast.Gt,
        "UNDER":   ast.Lt,
        "ATOP":    ast.GtE,
        "BENEATH": ast.LtE,
    }

    BOOLOP = {
        "BOND": ast.And,
        "RIFT": ast.Or,
    }

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)

        if not isinstance(node.func, ast.Name):
            return node

        name = node.func.id
        args = node.args

        if name in self.BINOP and len(args) == 2:
            return ast.copy_location(
                ast.BinOp(left=args[0], op=self.BINOP[name](), right=args[1]),
                node,
            )

        if name in self.CMPOP and len(args) == 2:
            return ast.copy_location(
                ast.Compare(left=args[0],
                            ops=[self.CMPOP[name]()],
                            comparators=[args[1]]),
                node,
            )

        if name in self.BOOLOP and len(args) >= 2:
            return ast.copy_location(
                ast.BoolOp(op=self.BOOLOP[name](), values=list(args)),
                node,
            )

        if name == "NEGATE" and len(args) == 1:
            return ast.copy_location(
                ast.UnaryOp(op=ast.Not(), operand=args[0]),
                node,
            )

        if name == "STRAND":
            return ast.copy_location(
                ast.List(elts=list(args), ctx=ast.Load()),
                node,
            )

        if name == "KNOT":
            return ast.copy_location(
                ast.Tuple(elts=list(args), ctx=ast.Load()),
                node,
            )

        if name == "PLUCK" and len(args) == 2:
            return ast.copy_location(
                ast.Subscript(value=args[0], slice=args[1], ctx=ast.Load()),
                node,
            )

        if name == "CARVE" and 2 <= len(args) <= 4:
            lo   = args[1] if len(args) > 1 else None
            hi   = args[2] if len(args) > 2 else None
            step = args[3] if len(args) > 3 else None
            return ast.copy_location(
                ast.Subscript(
                    value=args[0],
                    slice=ast.Slice(lower=lo, upper=hi, step=step),
                    ctx=ast.Load(),
                ),
                node,
            )

        if name == "ANNEX" and len(args) == 2:
            return ast.copy_location(
                ast.Call(
                    func=ast.Attribute(value=args[0], attr="append",
                                       ctx=ast.Load()),
                    args=[args[1]],
                    keywords=[],
                ),
                node,
            )

        if name == "EXPEL" and len(args) >= 1:
            return ast.copy_location(
                ast.Call(
                    func=ast.Attribute(value=args[0], attr="pop",
                                       ctx=ast.Load()),
                    args=list(args[1:]),
                    keywords=[],
                ),
                node,
            )

        return node


# ══════════════════════════════════════════════════════════════════
#  Full pipeline:  AlienCode source  →  transformed AST
# ══════════════════════════════════════════════════════════════════

def alien_transform(code: str, *, debug: bool = False) -> ast.Module:
    py_source = alien_preprocess(code)
    if debug:
        print("─── Stage 1  Text pre-process ───")
        print(py_source)

    tree = ast.parse(py_source)
    if HARDENED:
        check_program(tree)

    tree = AlienDesugarer().visit(tree)
    ast.fix_missing_locations(tree)
    if debug:
        print(f"─── Stage 3  Desugared ───\n{ast.unparse(tree)}")

    tree = FoldNegativeConstants().visit(tree)
    ast.fix_missing_locations(tree)

    if debug:
        print("─── Stage 5  Rule transforms ───")
    for layer_name, layer_cls in ALIEN_LAYERS:
        tree = layer_cls().visit(tree)
        ast.fix_missing_locations(tree)
        if debug:
            print(f"  {layer_name}: {ast.unparse(tree)}")

    tree = RouteOutput().visit(tree)
    ast.fix_missing_locations(tree)
    return tree


# ══════════════════════════════════════════════════════════════════
#  Execution sandbox  (thread-safe)
# ══════════════════════════════════════════════════════════════════

class _AlienTimeout(Exception):
    pass


def _alien_timeout_handler(signum, frame):
    raise _AlienTimeout(f"execution exceeded {EXEC_TIMEOUT}s limit")


def _is_main_thread() -> bool:
    return threading.current_thread() is threading.main_thread()


def _raise_in_thread(tid: int, exc_type: type):
    """Inject an exception into a running thread (CPython-only)."""
    ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(tid), ctypes.py_object(exc_type))
    if ret > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(tid), None)


_MAX_CAPTURE_CHARS = 262_144


class _CappedStringIO(StringIO):
    """Bound captured program output before a pathological loop can exhaust RAM."""

    def __init__(self, limit: int = _MAX_CAPTURE_CHARS):
        super().__init__()
        self._limit = limit
        self._truncated = False

    def write(self, text: str) -> int:
        requested = len(text)
        remaining = self._limit - self.tell()
        if remaining > 0:
            super().write(text[:remaining])
        if requested > max(remaining, 0):
            self._truncated = True
        # ``print`` expects the number accepted. Reporting the original length
        # lets execution continue until its ordinary timeout without retaining
        # the discarded suffix.
        return requested

    def getvalue(self) -> str:
        value = super().getvalue()
        if self._truncated:
            return value + "\n[AlienOutputTruncated]"
        return value


def _make_captured_alien_print(buf: StringIO):
    """Create an _alien_print that writes to buf instead of sys.stdout."""
    import builtins
    _real_print = builtins.print

    def _captured_alien_print(*args, **kwargs):
        kwargs["file"] = buf
        _real_print(*emit_order(args), **kwargs)

    return _captured_alien_print


def alien_exec(code: str, *, debug: bool = False,
               timeout: int = EXEC_TIMEOUT) -> str:
    captured = _CappedStringIO()

    try:
        tree = alien_transform(code, debug=debug)
        compiled = compile(tree, "<aliencode>", "exec")
    except Exception as exc:
        return f"[AlienError] {type(exc).__name__}: {exc}"

    env = _execution_env()
    env["__alien_merge__"] = lambda *args: "".join(str(a) for a in args)
    env["_alien_print"] = _make_captured_alien_print(captured)

    if _is_main_thread():
        return _exec_main_thread(compiled, env, captured, timeout)
    else:
        return _exec_worker_thread(compiled, env, captured, timeout)


def _exec_main_thread(compiled, env, captured, timeout):
    """Execute in main thread using signal.SIGALRM for timeout."""
    old_handler = None
    try:
        if timeout and hasattr(signal, "SIGALRM"):
            old_handler = signal.signal(signal.SIGALRM, _alien_timeout_handler)
            signal.alarm(timeout)

        exec(compiled, env, env)

        if timeout and hasattr(signal, "SIGALRM"):
            signal.alarm(0)

        return captured.getvalue().strip()

    except _AlienTimeout:
        partial = captured.getvalue().strip()
        err = f"[AlienError] TimeoutError: code exceeded {timeout}s limit"
        return (partial + "\n" + err).strip()
    except Exception as exc:
        partial = captured.getvalue().strip()
        err = f"[AlienError] {type(exc).__name__}: {exc}"
        return (partial + "\n" + err).strip()
    finally:
        if timeout and hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)


def _exec_worker_thread(compiled, env, captured, timeout):
    """Execute in worker thread using a daemon timer for timeout."""
    result_box: list[str | None] = [None]
    error_box: list[str | None] = [None]
    exec_tid: list[int | None] = [None]
    done_event = threading.Event()

    def _run():
        exec_tid[0] = threading.get_ident()
        try:
            exec(compiled, env, env)
            result_box[0] = captured.getvalue().strip()
        except _AlienTimeout:
            partial = captured.getvalue().strip()
            error_box[0] = (partial + f"\n[AlienError] TimeoutError: "
                            f"code exceeded {timeout}s limit").strip()
        except Exception as exc:
            partial = captured.getvalue().strip()
            error_box[0] = (partial + f"\n[AlienError] "
                            f"{type(exc).__name__}: {exc}").strip()
        finally:
            done_event.set()

    runner = threading.Thread(target=_run, daemon=True)
    runner.start()

    if done_event.wait(timeout=timeout if timeout else None):
        if error_box[0] is not None:
            return error_box[0]
        return result_box[0] or ""
    else:
        if exec_tid[0] is not None:
            _raise_in_thread(exec_tid[0], _AlienTimeout)
        runner.join(timeout=2)
        partial = captured.getvalue().strip()
        err = f"[AlienError] TimeoutError: code exceeded {timeout}s limit"
        return (partial + "\n" + err).strip()
