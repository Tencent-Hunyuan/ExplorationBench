"""Figures and the main table for the eight v2 findings.

From the repository root::

    python3 paper/figures/findings_v2.py            # every figure and the table
    python3 paper/figures/findings_v2.py --only f4_keystone

Every number is read from the run ledgers: the three-sample files under
``logs/<sandbox>/variance``, the oracle files under ``logs/<sandbox>/oracle``,
and, for the AlienCode rule reports, the result artefacts' milestone
snapshots. A system's autonomous values come from its Best@3 run, the run with
the highest three-sample $M_4$ (lowest index on a tie); control conditions
carry one trajectory per system, and a cell whose sampling is not complete is
left out rather than scored.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, HPacker, OffsetImage, TextArea  # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO / "analysis" / "logic"))
sys.path.insert(0, str(_REPO / "dev"))
from common.arm_names import ARM_EN  # noqa: E402
from house_style import (BRANDICON, SYSTEM_COLOR, _BADGE_EM, _art_zoom,  # noqa: E402
                         badge_ticks, read_brandicon, smallcaps_title)
from plot_style import GREY, INK, RED, BLUE, SUBTLE, W_FULL, ZONE, set_style  # noqa: E402

LOGS = _REPO / "logs"
ICONS = _HERE / "brandicon"
MARKS = _HERE / "mainfig"
OUT = _HERE / "findings"
ARXIV = _REPO / "arxiv"

KEYS = [("opus5", "Claude Opus 5"), ("gpt56", "GPT-5.6 Sol"), ("gemini38", "Gemini 3.8 Flash"),
        ("kimik3", "Kimi K3"), ("grok46", "Grok 4.6"), ("qwen38", "Qwen3.8-Max"),
        ("hy4", "Hy4 preview"), ("dsflash41", "DeepSeek-V4.1-Flash"), ("doubao21", "Seed2.1 Pro"),
        ("dspro", "DeepSeek-V4-Pro")]
NAME = dict(KEYS)
BOXES = {"code": "AlienCode", "logic": "AlienLogic"}
#: The Gemini route takes a thinking level rather than an effort, and every
#: request sends `thinkingLevel: high`; the run config records the effort
#: field, which that route never reads.
EFFORT = {"gemini38": "high"}
# ------------------------------------------------------------------ data
def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def variance(box, run):
    return _json(LOGS / box / "variance" / f"{run}.json")


def _trials(v, milestone=4):
    cols = list(zip(*[x[:3] for k, x in v["verdicts"].items()
                      if k.startswith(f"M{milestone}|") and len(x) >= 3]))
    return [100 * sum(c) / len(c) for c in cols]


def _result(box, key, n):
    paths = sorted((LOGS / box / "results").glob(f"eval_results_alien_{box}_*_v2_{key}_n{n}.json"),
                   key=lambda p: p.stat().st_mtime)
    return _json(paths[-1]) if paths else None


def load():
    data = {}
    for box in BOXES:
        for key, _ in KEYS:
            runs = []
            for n in (1, 2, 3):
                v = variance(box, f"v2_{key}_n{n}")
                accs = _trials(v)
                runs.append({"n": n, "curve": [v["mean_score"][f"M{m}"] for m in range(5)],
                             "accs": accs, "noise": statistics.stdev(accs),
                             "unprov": (v.get("mean_unprovable") or {}).get("M4")})
            best = max(range(3), key=lambda i: (runs[i]["curve"][4], -i))
            oracle = (_json(LOGS / box / "oracle" / f"v2_{key}_n{best + 1}.json") or {}).get("mean_score") or {}
            config = (_result(box, key, best + 1) or {}).get("config") or {}
            data[box, key] = {
                "runs": runs, "best": best, "best_m4": runs[best]["curve"][4],
                "mean": statistics.mean(r["curve"][4] for r in runs),
                "worst": min(r["curve"][4] for r in runs),
                "o_m0": oracle.get("O@M0"), "a4o": oracle.get("A4+O"),
                "effort": EFFORT.get(key) or config.get("reasoning_effort_asked_for")
                or config.get("reasoning_effort"),
            }
    ctrl = {}
    for box in BOXES:
        for key, _ in KEYS:
            for arm in ("none", "think", "random", "passive"):
                ctrl[box, key, arm] = None
                # One run per condition: n1, or the lowest index that exists.
                # A missing answer is scored as wrong ("complete" only flags it).
                for n in (1, 2, 3):
                    v = variance(box, f"ctrl_{arm}_v2_{key}_{box}_n{n}")
                    if v and "M4" in (v.get("mean_score") or {}):
                        ctrl[box, key, arm] = v["mean_score"]["M4"]
                        break
    key2id = {k: r for r, k in re.findall(r'\{"id": "(R\d+\+?)", "key": "([a-z_0-9]+)"',
                                           (_REPO / "dev/sandboxes/code/engine.py").read_text())}
    body_rule = re.search(r'"id": "R01\+",.*?"key": "(\w+)"',
                          (_REPO / "dev/sandboxes/code/execution.py").read_text(), re.S)
    key2id.setdefault(body_rule.group(1), "R01+")
    # The rules each task exercises come from the 70-task held-out set itself;
    # the in-run test records cover a different, larger task list.
    needs = {t["id"]: {key2id.get(k, k) for k in t.get("rules_tested") or []}
             for t in _json(_REPO / "dev/sandboxes/code/eval_set_v2.json")}
    runs = []
    for key, _ in KEYS:
        for n in (1, 2, 3):
            snaps = {s["milestone_idx"]: s for s in _result("code", key, n)["snapshots"]}
            v = variance("code", f"v2_{key}_n{n}")
            s4 = snaps[4]
            scores = s4.get("rule_scores") or {}
            known, missing = [], []
            for task_id, rules in needs.items():
                votes = v["verdicts"].get(f"M4|{task_id}")
                if not votes:
                    continue
                flags = [bool(scores.get(r)) for r in rules if r in scores]
                (known if flags and all(flags) else missing).append(100 * sum(votes) / len(votes))
            runs.append({
                "key": key, "n": n, "found": s4.get("found"),
                "curve": [v["mean_score"][f"M{m}"] for m in range(5)],
                "first": {r: next((m for m in range(5)
                                   if (snaps.get(m, {}).get("rule_scores") or {}).get(r)), None)
                          for r in ("R14", "R15")},
                "keystones": [sum(bool((snaps.get(m, {}).get("rule_scores") or {}).get(r))
                                  for r in ("R14", "R15")) for m in range(5)],
                "known": known, "missing": missing,
            })
    return data, ctrl, runs


DATA, CTRL, RUNS = load()


# ------------------------------------------------------------------ helpers
def order(box):
    return sorted((k for k, _ in KEYS), key=lambda k: -DATA[box, k]["best_m4"])


def _frame(ax, grid="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GREY)
    if grid:
        ax.grid(axis=grid, color=ZONE, lw=0.8, zorder=0)


def _rows(ax, keys, size=7.2):
    y = {k: len(keys) - 1 - i for i, k in enumerate(keys)}
    ax.set_yticks([y[k] for k in keys], [NAME[k] for k in keys], fontsize=size)
    ax.set_ylim(-0.6, len(keys) - 0.4)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    return y


def _letter(ax, text, x=-0.02):
    ax.text(x, 1.05, rf"\textbf{{{text}}}", transform=ax.transAxes, fontsize=10.5,
            ha="right", va="bottom", color=INK)


def _ties(values, rank, gap=2.8, step=0.08):
    """Offsets that set near-equal values side by side in one flat row.

    Values closer than `gap` chain into a row; within a row, systems keep the
    legend order, so a row never leans with its values.
    """
    idx = sorted(range(len(values)), key=lambda i: values[i])
    rows, row = [], idx[:1]
    for i in idx[1:]:
        if values[i] - values[row[-1]] < gap:
            row.append(i)
        else:
            rows.append(row)
            row = [i]
    rows.append(row)
    offsets = [0.0] * len(values)
    for row in rows:
        for j, i in enumerate(sorted(row, key=lambda i: rank[i])):
            offsets[i] = (j - (len(row) - 1) / 2) * step
    return offsets


# ------------------------------------------------------------------ Finding 1
def f1_conditions():
    # Short tick labels: the full condition names ran into each other.
    conds = list(zip(("none", "think", "random", "passive", "auto"),
                     ("Direct", "Without tool", "Fixed probe", "Hindsight", "Autonomous")))
    legend = order("code")
    rank = {k: i for i, k in enumerate(legend)}
    # The wider step after the second column keeps "Without-tool" clear of
    # "Fixed-probe" and sets the two no-feedback conditions apart.
    xs = (0, 1, 2.14, 3.22, 4.46)

    def value(box, k, arm):
        return DATA[box, k]["best_m4"] if arm == "auto" else CTRL[box, k, arm]

    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.85), sharey=True)
    for ax, box, lt in zip(axes, BOXES, "ab"):
        ax.axvspan(-0.5, (xs[1] + xs[2]) / 2, color=ZONE, zorder=0, lw=0)
        ax.text(0.5, 96, "no environment\nfeedback", ha="center", va="top", fontsize=7.4, color=SUBTLE)
        for i, (arm, _) in enumerate(conds):
            pts = [(value(box, k, arm), k) for k, _ in KEYS if value(box, k, arm) is not None]
            for (v, k), dx in zip(pts, _ties([v for v, _ in pts], [rank[k] for _, k in pts])):
                ax.scatter(xs[i] + dx, v, s=16, color=SYSTEM_COLOR[NAME[k]], edgecolor="white", lw=0.4, zorder=3)
            if pts:
                med = statistics.median(v for v, _ in pts)
                ax.plot([xs[i] - 0.4, xs[i] + 0.4], [med, med], color=INK, lw=1.1, zorder=4)
        # Rotated first, then centred by the bounding box, so each label sits
        # under its column and stays below the axis.
        ax.set_xticks(xs, [c[1] for c in conds], fontsize=8.1, rotation=22, ha="center",
                      rotation_mode="default")
        ax.tick_params(axis="y", labelsize=8.1)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.48, xs[-1] + 0.48)
        ax.set_ylim(-3, 100)
        _frame(ax, grid="y")
        if box == "code":
            ax.set_ylabel(r"$M_4$ (\%)", fontsize=8.6)
        smallcaps_title(ax, BOXES[box], MARKS)
        _letter(ax, lt, x=-0.04)
    fig.tight_layout(rect=(0, 0, 0.83, 1), w_pad=0.4)
    handles = [Line2D([], [], marker="o", ls="", ms=5.0, mfc=SYSTEM_COLOR[NAME[k]], mec="white", mew=0.4,
                      label=NAME[k]) for k in legend]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.834, 0.5), frameon=False, fontsize=7.6,
               handletextpad=0.3, labelspacing=0.62, borderaxespad=0)
    return fig, []


# ------------------------------------------------------------------ Finding 2
def f2_sandbox_ranks():
    keys = order("code")
    logic_rank = {k: i + 1 for i, k in enumerate(order("logic"))}
    fig, ax = plt.subplots(figsize=(W_FULL * 0.7, 3.5))
    y = _rows(ax, keys)
    for i, k in enumerate(keys):
        xc, xl = DATA["code", k]["best_m4"], DATA["logic", k]["best_m4"]
        ax.plot([xc, xl], [y[k]] * 2, color=ZONE, lw=4.5, solid_capstyle="round", zorder=1)
        ax.scatter(xc, y[k], s=34, color="#CC785C", zorder=3, edgecolor="white", lw=0.6)
        ax.scatter(xl, y[k], s=34, color="#4F4234", zorder=3, edgecolor="white", lw=0.6)
        move = logic_rank[k] - (i + 1)
        ax.text(103, y[k], f"{i + 1} $\\rightarrow$ {logic_rank[k]}", va="center", fontsize=6.8,
                color=RED if move >= 3 else (BLUE if move <= -3 else SUBTLE))
    ax.text(103, len(keys) - 0.45, "rank", fontsize=6.6, color=SUBTLE)
    ax.set_xlim(0, 100)
    _frame(ax)
    ax.set_xlabel(r"Best@3 $M_4$ (\%)")
    handles = [Line2D([], [], marker="o", ls="", color=c, ms=5, label=b)
               for b, c in (("AlienCode", "#CC785C"), ("AlienLogic", "#4F4234"))]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=6.8)
    fig.tight_layout(rect=(0.05, 0, 0.94, 1))
    return fig, [(ax, keys, "y")]


# ------------------------------------------------------------------ Finding 3
def f3_disclosure():
    fig, axes = plt.subplots(2, 1, figsize=(W_FULL, 4.6))
    badges = []
    for ax, box, lt in zip(axes, BOXES, "ab"):
        keys = order(box)
        x = np.arange(len(keys))
        width = 0.27
        colors = [SYSTEM_COLOR[NAME[k]] for k in keys]
        styles = [("o_m0", dict(color="white", edgecolor=colors, hatch="////", lw=0.8)),
                  ("best_m4", dict(color=colors, edgecolor="white", lw=0.6)),
                  ("a4o", dict(color="white", edgecolor=colors, lw=1.1))]
        for j, (field, style) in enumerate(styles):
            vals = [DATA[box, k][field] for k in keys]
            ax.bar(x + (j - 1) * width, vals, width * 0.9, zorder=3, **style)
            for xi, v in zip(x + (j - 1) * width, vals):
                ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=5.4, color=SUBTLE)
        ax.set_xticks(x, [NAME[k] for k in keys], fontsize=6.7, rotation=28, ha="center", rotation_mode="default")
        ax.tick_params(axis="x", length=2.5, color=GREY)
        ax.set_ylim(0, 108)
        _frame(ax, grid="y")
        ax.set_ylabel(r"$M_4$ (\%)")
        smallcaps_title(ax, BOXES[box], MARKS)
        _letter(ax, lt, x=-0.06)
        badges.append((ax, keys, "x"))
    handles = [Patch(facecolor="white", edgecolor=SUBTLE, hatch="////", label=ARM_EN["oracle"] + r" at $M_0$ (O@$M_0$)"),
               Patch(facecolor=GREY, edgecolor="white", label=ARM_EN["self"] + r" (Best@3 $M_4$)"),
               Patch(facecolor="white", edgecolor=SUBTLE, lw=1.1, label=ARM_EN["oracle"] + " after exploration (A4+O)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=7.0, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.6)
    return fig, badges


# ------------------------------------------------------------------ Finding 4
#: The board is drawn in units of W_FULL * 0.8 / 748 inches, shared by both
#: axes, so type and rules keep one size whether it has one column or two. A
#: trajectory row is 25 units tall; a milestone column takes the width left.
_UNIT = W_FULL * 0.8 * 72 / 748
_ROW, _TOP = 25, 52
_RULE, _RULE_BLOCK, _INDEX, _TEXT = "#B9B2A4", "#8C8474", "#8A8477", "#2B2A28"
#: R15 sits left of the value and points up; R14 sits right and points down.
_KEYSTONES = (("R15", "#C0392B", -1, "R15 first reported correctly"),
              ("R14", "#1F6F8B", 1, "R14 first reported correctly"))


def _triangle(x, y, side):
    """A keystone mark centred at (x, y) in board units, where y grows downwards."""
    return [(x - 4, y - 3 * side), (x + 4, y - 3 * side), (x, y + 4 * side)]


def _text_width(fig, text, size):
    """Rendered width of `text` in points."""
    probe = fig.text(0, 0, text, fontsize=size)
    width = probe.get_window_extent(fig.canvas.get_renderer()).width * 72 / fig.dpi
    probe.remove()
    return width


def _board_column(ax, x0, col, keys):
    """One column of the milestone board whose first milestone rule is at x0."""
    bottom = _TOP + 3 * _ROW * len(keys)
    right = x0 + 5 * col
    offset = min(24, col / 4 + 3.6)
    for m in range(6):
        ax.plot([x0 + m * col] * 2, [_TOP - 9, bottom + 9], color=_RULE, lw=0.5, solid_capstyle="butt")
    for i in range(3 * len(keys) + 1):
        block = i % 3 == 0
        ax.plot([x0 - 9, right + 9], [_TOP + i * _ROW] * 2, color=_RULE_BLOCK if block else _RULE,
                lw=0.72 if block else 0.5, solid_capstyle="butt")
    for m in range(5):
        ax.text(x0 + (m + 0.5) * col, _TOP - 16, rf"$M_{m}$", ha="center", va="baseline",
                fontsize=16 * _UNIT, color=_TEXT)
    for g, key in enumerate(keys):
        y0 = _TOP + 3 * g * _ROW
        mid = y0 + 1.5 * _ROW
        ax.plot([x0 - 32, x0 - 37, x0 - 37, x0 - 32], [y0 + 4.5, y0 + 4.5, y0 + 3 * _ROW - 4.5, y0 + 3 * _ROW - 4.5],
                color=_RULE_BLOCK, lw=0.62, solid_capstyle="butt", solid_joinstyle="miter")
        ax.text(x0 - 61, mid, NAME[key], ha="right", va="center_baseline", fontsize=14.5 * _UNIT, color=_TEXT)
        image, scale = read_brandicon(ICONS, NAME[key])
        if image is not None:
            ax.add_artist(AnnotationBbox(OffsetImage(image, zoom=_art_zoom(14.5 * _UNIT, None, _BADGE_EM) / scale),
                                         (x0 - 49.5, mid), box_alignment=(0.5, 0.5), frameon=False, pad=0,
                                         annotation_clip=False))
        rows = sorted((r for r in RUNS if r["key"] == key), key=lambda r: r["n"])
        for i, r in enumerate(rows):
            y = y0 + (i + 0.5) * _ROW
            ax.text(x0 - 21, y, str(r["n"]), ha="center", va="center_baseline", fontsize=11 * _UNIT,
                    color=_INDEX)
            for m, value in enumerate(r["curve"]):
                ax.text(x0 + (m + 0.5) * col, y, f"{value:.0f}", ha="center", va="center_baseline",
                        fontsize=13 * _UNIT, color=_TEXT)
            for rule, color, side, _ in _KEYSTONES:
                m = r["first"][rule]
                if m is not None:
                    ax.add_patch(Polygon(_triangle(x0 + (m + 0.5) * col + offset * side, y, side),
                                         closed=True, facecolor=color, lw=0))


def f4_keystone(columns=2):
    keys = order("code")
    per = -(-len(keys) // columns)
    groups = [keys[i:i + per] for i in range(0, len(keys), per)]
    width = W_FULL * (0.8 if columns == 1 else 1.0) * 72 / _UNIT
    bottom = _TOP + 3 * _ROW * per
    height = bottom + 56
    fig = plt.figure(figsize=(width * _UNIT / 72, height * _UNIT / 72))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    # Name start to first milestone rule, per column; the columns then share
    # what is left of the width equally among their milestone columns.
    labels = [61 + max(_text_width(fig, NAME[k], 14.5 * _UNIT) for k in group) / _UNIT for group in groups]
    gap = 30
    col = (width - 4 - sum(labels) - 9 * len(groups) - gap * (len(groups) - 1)) / (5 * len(groups))
    x0, starts = 2 + labels[0], []
    for i, group in enumerate(groups):
        _board_column(ax, x0, col, group)
        starts.append(x0)
        if i + 1 < len(groups):
            x0 += 5 * col + 9 + gap + labels[i + 1]
    items = []
    for _, color, side, label in _KEYSTONES:
        mark = DrawingArea(8 * _UNIT, 8 * _UNIT)
        mark.add_artist(Polygon([(x * _UNIT, (8 - y) * _UNIT) for x, y in _triangle(4, 4, side)],
                                closed=True, facecolor=color, lw=0))
        items.append(HPacker(children=[mark, TextArea(label, textprops={"fontsize": 12.5 * _UNIT,
                                                                        "color": _TEXT})],
                             align="center", pad=0, sep=6 * _UNIT))
    ax.add_artist(AnnotationBbox(HPacker(children=items, align="center", pad=0, sep=34 * _UNIT),
                                 (width / 2, bottom + 34),
                                 box_alignment=(0.5, 0.5), frameon=False, pad=0, annotation_clip=False))
    return fig, []


# ------------------------------------------------------------------ Finding 5
def f5_knowing_doing():
    per = {}
    for k, _ in KEYS:
        known = [a for r in RUNS if r["key"] == k for a in r["known"]]
        missing = [a for r in RUNS if r["key"] == k for a in r["missing"]]
        if known and missing:
            per[k] = (statistics.mean(known), statistics.mean(missing), len(known))
    keys = sorted(per, key=lambda k: -per[k][0])
    fig, ax = plt.subplots(figsize=(W_FULL * 0.72, 3.0))
    for i, k in enumerate(keys):
        known, missing, n = per[k]
        color = SYSTEM_COLOR[NAME[k]]
        ax.plot([i, i], [missing, known], color=ZONE, lw=5, solid_capstyle="round", zorder=1)
        ax.plot([i, i], [known, 100], color="#F1D9D2", lw=1.2, ls=(0, (2, 2)), zorder=1)
        ax.scatter(i, missing, s=26, facecolor="white", edgecolor=color, lw=1.0, zorder=3)
        ax.scatter(i, known, s=34, color=color, edgecolor="white", lw=0.5, zorder=4)
        ax.text(i, 104, f"n={n}", ha="center", fontsize=5.8, color=SUBTLE)
    ax.axhline(100, color=GREY, lw=0.7, ls=(0, (3, 3)))
    ax.set_xticks(range(len(keys)), [NAME[k] for k in keys], fontsize=6.8, rotation=45, ha="center",
                  rotation_mode="default")
    ax.tick_params(axis="x", length=2.5, color=GREY)
    ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.set_ylim(0, 110)
    _frame(ax, grid="y")
    ax.set_ylabel(r"held-out accuracy at $M_4$ (\%)")
    handles = [Line2D([], [], marker="o", ls="", color=INK, ms=5, label="all required rules reported correctly"),
               Line2D([], [], marker="o", ls="", mfc="white", mec=INK, mew=1.0, ms=5,
                      label="a required rule missing or wrong")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, frameon=False, fontsize=6.6)
    fig.tight_layout()
    return fig, [(ax, keys, "x")]


# ------------------------------------------------------------------ Finding 6
def f6_run_spread():
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 3.5))
    badges = []
    for ax, box, lt in zip(axes, BOXES, "ab"):
        keys = order(box)
        y = _rows(ax, keys, 7.0)
        for k in keys:
            d = DATA[box, k]
            xs = [r["curve"][4] for r in d["runs"]]
            color = SYSTEM_COLOR[NAME[k]]
            ax.plot([min(xs), max(xs)], [y[k]] * 2, color=color, alpha=0.28, lw=5, solid_capstyle="round", zorder=1)
            for i, r in enumerate(d["runs"]):
                m4 = r["curve"][4]
                ax.plot([m4 - r["noise"], m4 + r["noise"]], [y[k]] * 2, color=INK, alpha=0.13, lw=9,
                        solid_capstyle="round", zorder=2)
                ax.scatter(m4, y[k], s=30 if i == d["best"] else 22, zorder=4,
                           facecolor=color if i == d["best"] else "white", edgecolor=color, lw=1.0)
            ax.plot([d["mean"]] * 2, [y[k] - 0.26, y[k] + 0.26], color=INK, lw=0.9, zorder=3)
        ax.set_xlim(-2, 102)
        _frame(ax)
        ax.set_xlabel(r"$M_4$ of each trajectory (\%)")
        smallcaps_title(ax, BOXES[box], MARKS)
        _letter(ax, lt)
        badges.append((ax, keys, "y"))
    fig.subplots_adjust(left=0.2, right=0.985, top=0.9, bottom=0.14, wspace=0.62)
    return fig, badges


# ------------------------------------------------------------------ Finding 8
def f8_leap_timing():
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 3.6))
    badges = []
    for ax, box, lt in zip(axes, BOXES, "ab"):
        keys = order(box)
        y = _rows(ax, keys)
        for k in keys:
            d = DATA[box, k]
            curve = d["runs"][d["best"]]["curve"]
            steps = [curve[t] - curve[t - 1] for t in range(1, 5)]
            lead = max(range(4), key=lambda t: steps[t])
            color = SYSTEM_COLOR[NAME[k]]
            ax.plot([1, 4], [y[k]] * 2, color=ZONE, lw=1.0, zorder=1)
            for t, step in enumerate(steps, start=1):
                size = 6 + 3.4 * abs(step)
                if step >= 0:
                    ax.scatter(t, y[k], s=size, color=color, alpha=1.0 if t - 1 == lead else 0.3,
                               edgecolor="white", lw=0.6, zorder=3)
                else:
                    ax.scatter(t, y[k], s=size, facecolor="white", edgecolor=RED, lw=1.0, zorder=3)
                if t - 1 == lead and step >= 8:
                    ax.annotate(f"+{step:.0f}", (t, y[k]), xytext=((size / 3.1416) ** 0.5 + 1.5, 0),
                                textcoords="offset points", fontsize=6.3, va="center", color=INK, zorder=4)
            ax.text(4.72, y[k], f"{curve[4]:.1f}", fontsize=6.8, va="center", color=SUBTLE)
        ax.text(4.72, len(keys) - 0.45, r"$M_4$", fontsize=6.8, color=SUBTLE)
        ax.set_xticks([1, 2, 3, 4], [rf"$M_{t}$" for t in range(1, 5)], fontsize=7.6)
        ax.set_xlabel("change from the previous milestone", fontsize=7.2)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(0.55, 5.05)
        _frame(ax, grid=None)
        smallcaps_title(ax, BOXES[box], MARKS)
        _letter(ax, lt)
        badges.append((ax, keys, "y"))
    fig.subplots_adjust(left=0.2, right=0.985, top=0.9, bottom=0.1, wspace=0.62)
    return fig, badges


FIGURES = {f.__name__: f for f in (f1_conditions, f2_sandbox_ranks, f3_disclosure, f4_keystone,
                                   f5_knowing_doing, f6_run_spread, f8_leap_timing)}


# ------------------------------------------------------------------ table
def _num(value):
    return "{--}" if value is None else f"{value:.1f}"


def main_table_rows() -> str:
    lines = ["% Auto-generated by paper/figures/findings_v2.py -- do not edit."]
    for box, macro in (("code", "MainCodeRows"), ("logic", "MainLogicRows")):
        rows = []
        for k in order(box):
            d = DATA[box, k]
            best = d["runs"][d["best"]]
            if box == "code":
                found = next(r["found"] for r in RUNS if r["key"] == k and r["n"] == best["n"])
                aux = str(found)
            else:
                aux = _num(best["unprov"])
            rows.append(
                f"\\modelname{{{BRANDICON[NAME[k]]}}}{{{NAME[k]}}} & {d['effort']} & "
                f"{_num(best['curve'][0])} & {_num(d['best_m4'])} & {_num(d['mean'])} & {_num(d['worst'])} & "
                f"{_num(d['o_m0'])} & {_num(d['a4o'])} & {aux} \\\\")
        lines.append(f"\\newcommand{{\\{macro}}}{{%")
        lines.extend(rows)
        lines.append("}")
    return "\n".join(lines) + "\n"


def exploration_budget(box, key, n):
    """Tool calls C, probe units P, and exploration tokens T over the four rounds.

    T is API-reported input plus output tokens of the model requests made
    while exploring; closed-book testing is not included."""
    records = (_result(box, key, n) or {}).get("explore_records") or []
    if box == "code":
        calls = sum(len(r.get("tool_calls") or []) for r in records)
        units = sum(tc.get("show_count_executed") or 0 for r in records for tc in r.get("tool_calls") or [])
    else:
        calls = units = sum(r.get("n_probes_accepted_for_eval") or 0 for r in records)
    tokens = sum((r.get("call_metrics") or {}).get("input_tokens", 0)
                 + (r.get("call_metrics") or {}).get("output_tokens", 0) for r in records)
    return calls, units, tokens


def budget_table_rows() -> str:
    lines = ["% Auto-generated by paper/figures/findings_v2.py -- do not edit.",
             "\\newcommand{\\BudgetRows}{%"]
    for k in order("code"):
        cells = []
        for box in BOXES:
            calls, units, tokens = exploration_budget(box, k, DATA[box, k]["best"] + 1)
            cells += [str(calls), str(units)] if box == "code" else [str(calls)]
            cells.append(f"{tokens / 1e6:.2f}")
        lines.append(f"\\modelname{{{BRANDICON[NAME[k]]}}}{{{NAME[k]}}} & " + " & ".join(cells) + " \\\\")
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="figure names, or 'table'")
    args = ap.parse_args()
    wanted = args.only or [*FIGURES, "table"]
    set_style()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in wanted:
        if name == "table":
            target = ARXIV / "tables" / "v2_main_rows.tex"
            target.write_text(main_table_rows(), encoding="utf-8")
            print("wrote", target.relative_to(_REPO))
            budget = ARXIV / "tables" / "budget_rows.tex"
            budget.write_text(budget_table_rows(), encoding="utf-8")
            print("wrote", budget.relative_to(_REPO))
            continue
        fig, badges = FIGURES[name]()
        for ax, keys, axis in badges:
            badge_ticks(ax, ICONS, [NAME[k] for k in keys], axis=axis)
        stem = OUT / name
        fig.savefig(stem.with_suffix(".pdf"))
        fig.savefig(stem.with_suffix(".png"), dpi=220)
        plt.close(fig)
        target = ARXIV / "figures" / "findings"
        if target.parent.is_dir():
            target.mkdir(exist_ok=True)
            shutil.copy2(stem.with_suffix(".pdf"), target / f"{name}.pdf")
        print("wrote", stem.with_suffix(".pdf").relative_to(_REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
