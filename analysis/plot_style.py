"""
analysis/logic/plot_style.py
============================

Shared publication-quality matplotlib style for the paper figures, tuned for a
restrained top-tier-ML-conference look:

  * XCharter via usetex, the same face main.cls sets for the body,
  * a restrained, colourblind-aware palette with fixed semantic roles,
  * de-spined axes with a single faint horizontal grid,
  * soft tinted confidence bands instead of busy error-bar whiskers,
  * helpers for the recurring "design language" of the paper figures:
        - shaded baseline / no-learning zones,
        - rounded callout annotation boxes,
        - direct end-of-line series labels,
        - colour lightening for fills.

Usage:
    from plot_style import set_style, figure, panel_title, despine, finalize
    set_style()
    fig, axes = figure(ncols=2, height=H_PAIR)
    ...
    despine(ax)
    panel_title(ax, "a", r"\textsc{AlienLogic}")
    finalize(fig, "figures/foo")   # writes .pdf and .png at the design width

All text passed to matplotlib is rendered by LaTeX, so literal '%' must be
written as '\\%' and other TeX specials escaped by the caller.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
from matplotlib.patheffects import withStroke  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

# The visual language is deliberately narrow: blue identifies the primary
# result, teal is a second positive condition, red marks an intervention or
# failure, and muted slate tones carry the remaining cohort. Marker shape still
# distinguishes individual systems in all-series trajectory plots.
PALETTE = [
    "#2F6FAF",  # primary result
    "#3C9273",  # secondary positive result
    "#D65362",  # intervention / failure
    "#72869A",  # muted cohort
    "#8494A3",
    "#96A3AE",
    "#A7B1BA",
    "#B5BDC5",
    "#C1C8CE",
    "#CCD1D6",
]
# Ten visually distinct filled marker shapes, paired 1:1 with PALETTE.
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "p"]

# Semantic colours for two/three-condition comparisons.
BLUE = "#2F6FAF"
RED = "#D65362"
GREEN = "#3C9273"
PURPLE = "#8166A8"
ORANGE = "#C98A23"
TEAL = "#3C9273"
GREY = "#8E99A5"

# Neutral ink / fill tones for text, dividers, zones.
INK = "#30343B"
SUBTLE = "#6E7883"
ZONE = "#F1F3F5"

# The project website re-renders some of these figures in its own ink-on-white
# palette. The switch lives in paper/figures/eb_theme.py, which is only
# imported when the environment asks for it, so the paper build never sees it.
_EB_THEME = None
if os.environ.get("EB_FIGURE_THEME") == "hunyuan":
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "paper" / "figures"))
    import eb_theme as _EB_THEME  # noqa: E402

    INK = _EB_THEME.INK
    SUBTLE = _EB_THEME.SUBTLE
    ZONE = _EB_THEME.ZONE
    GREY = _EB_THEME.STONE[400]
    BLUE = _EB_THEME.BLUE
    # The cohort tail of PALETTE is already a grey ramp; the three semantic
    # slots in front of it lose their hues. Series that must stay apart are
    # separated by MARKERS, which is untouched.
    PALETTE = [_EB_THEME.BLUE, _EB_THEME.STONE[600], _EB_THEME.STONE[900],
               _EB_THEME.STONE[500], _EB_THEME.STONE[450],
               _EB_THEME.STONE[400], "#b5b0ab", _EB_THEME.STONE[300],
               "#dedbd8", _EB_THEME.STONE[200]]

# ---------------------------------------------------------------------------
# Canonical geometry
# ---------------------------------------------------------------------------
# Every figure is drawn on a canvas of width W_FULL and included in LaTeX at
# one \linewidth (5.5in in the ICLR style), so each figure is reduced by the
# same factor and in-figure type lands at the same physical size throughout the
# paper. Scripts that need a narrower figure must scale W_FULL by the same
# fraction they pass to \includegraphics, never pick an unrelated width.
W_FULL = 7.2

# Heights are drawn from this set so that axes rectangles match across figures.
# The _LEGEND variants add exactly the strip a below-axes legend consumes, which
# keeps the plotting area itself the same height as the plain variant.
H_SINGLE = 3.05          # one wide panel
H_PAIR = 3.05            # two panels side by side
H_PAIR_LEGEND = 3.45     # ... plus a legend strip under the axes
H_TALL = 4.10            # horizontal-bar panels with one row per system
H_TALL_LEGEND = 4.35

# Milestone tick labels. Written as math so subscripts match the body text.
MS_TICKS = [rf"$M_{i}$" for i in range(9)]


def figure(height=H_PAIR, *, ncols=1, nrows=1, width=W_FULL, **kwargs):
    """Create a canonical-width figure. Returns whatever plt.subplots returns."""
    return plt.subplots(nrows, ncols, figsize=(width, height), **kwargs)


def panel_title(ax, letter, text, *, loc="left", pad=None):
    """Set a panel title in the paper's one format: bold '(a)' + plain text."""
    ax.set_title(rf"\textbf{{({letter})}}~{text}", loc=loc,
                 **({} if pad is None else {"pad": pad}))


def figure_legend(fig, handles, labels, *, ncol, y=0.012, fontsize=8.0):
    """One legend for the whole figure, centred beneath the panels.

    Multi-panel figures share a single key rather than repeating one legend per
    panel; the `_LEGEND` heights above reserve the strip this occupies.
    """
    return fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, y),
        ncol=ncol, frameon=False, fontsize=fontsize,
        handlelength=1.4, handletextpad=0.5, columnspacing=1.4,
        borderaxespad=0.0,
    )


def set_style() -> None:
    plt.rcParams.update({
        "text.usetex": True,
        # Same stack the class loads, so a figure label and the sentence that
        # refers to it are set in one typeface.
        "text.latex.preamble": (
            r"\usepackage{XCharter}"
            r"\usepackage[xcharter,bigdelims,vvarbb]{newtxmath}"
            r"\usepackage{amsmath}"
        ),
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "axes.titlepad": 7,
        "axes.labelpad": 4,
        "axes.edgecolor": "#68717B",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#D9DEE5",
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "grid.alpha": 0.45,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.8,
        "lines.markeredgewidth": 0.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.38,
        "legend.columnspacing": 1.0,
        "legend.borderaxespad": 0.3,
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })
    if _EB_THEME is not None:
        plt.rcParams.update(_EB_THEME.rcparams())


def lighten(color: str, amount: float = 0.5) -> tuple:
    """Blend `color` toward white by `amount` in [0, 1] (1 == white)."""
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def darken(color: str, amount: float = 0.25) -> tuple:
    """Blend `color` toward black by `amount` in [0, 1] (1 == black)."""
    r, g, b = mcolors.to_rgb(color)
    return (r * (1 - amount), g * (1 - amount), b * (1 - amount))


def despine(ax, *, left: bool = True, bottom: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.tick_params(length=3, width=0.8)


def band(ax, x, y, err, color, *, alpha: float = 0.12, zorder: int = 2,
         feather: float = 0.25, layers: int = 20, deepen: float = 0.0):
    """Draw a soft mean+-err confidence band (no whiskers).

    A single flat fill ends at a hard edge, which reads as a boundary the data
    does not have: one standard deviation is not where the uncertainty stops.

    The two knobs are independent. ``alpha`` is how dark the band is, held flat
    across its solid core. ``feather`` is the outer share of ``err`` over which
    that opacity ramps to nothing, so a small value keeps a legible band with a
    softened rim while a large one fades the whole width. Enough layers are
    used that the steps between them fall below a printed hairline.

    ``deepen`` darkens the fill before it is made transparent, which keeps a
    pale series from washing out where a dark one still registers.
    """

    if deepen:
        color = darken(color, deepen)
    feather = min(max(feather, 0.0), 1.0)
    core = 1.0 - feather
    step = 1.0 - (1.0 - alpha) ** (1.0 / layers)
    for index in range(layers, 0, -1):
        frac = core + feather * index / layers
        lo = [a - frac * b for a, b in zip(y, err)]
        hi = [a + frac * b for a, b in zip(y, err)]
        ax.fill_between(x, lo, hi, color=color, alpha=step, linewidth=0,
                        zorder=zorder)


def baseline_zone(ax, x_split, *, label=None, label_y=None, text_color=SUBTLE):
    """Shade the region left of `x_split` as a 'no-learning' zone with a divider."""
    x0 = ax.get_xlim()[0]
    ax.axvspan(x0, x_split, color=ZONE, zorder=0)
    ax.axvline(x_split, color="#AAB0B8", linestyle=(0, (3, 2)),
               linewidth=0.9, zorder=1)
    if label is not None:
        y0, y1 = ax.get_ylim()
        ly = label_y if label_y is not None else y0 + 0.06 * (y1 - y0)
        ax.text(x_split - 0.12, ly, label, color=text_color, fontsize=7.6,
                ha="right", va="bottom", style="italic", zorder=2)


def callout(ax, text, xy, xytext, *, color=INK, fc="white", ec="#C9CDD2",
            arrow=True, fontsize=7.8, ha="left", va="center"):
    """Rounded annotation box, optionally with a thin curved leader arrow."""
    arrowprops = dict(arrowstyle="-", color="#9AA0A6", linewidth=0.7,
                      connectionstyle="arc3,rad=0.12") if arrow else None
    ax.annotate(
        text, xy=xy, xytext=xytext, fontsize=fontsize, color=color,
        ha=ha, va=va, zorder=6,
        bbox=dict(boxstyle="round,pad=0.32,rounding_size=0.5",
                  fc=fc, ec=ec, linewidth=0.7),
        arrowprops=arrowprops,
    )


def end_label(ax, x, y, text, color, *, dx=0.10, fontsize=8.0, va="center"):
    """Place a direct, colour-matched label just past the end of a series."""
    ax.text(x + dx, y, text, color=color, fontsize=fontsize, va=va, ha="left",
            zorder=5, fontweight="bold",
            path_effects=[withStroke(linewidth=1.6, foreground="white")])


# Offsets tried in order when placing a point label, nearest ring first.
_COMPASS = [
    (0.0, 1.0, "center", "bottom"),
    (1.0, 0.55, "left", "center"),
    (-1.0, 0.55, "right", "center"),
    (0.0, -1.0, "center", "top"),
    (1.0, 1.0, "left", "bottom"),
    (-1.0, 1.0, "right", "bottom"),
    (1.0, -1.0, "left", "top"),
    (-1.0, -1.0, "right", "top"),
]


def place_point_labels(ax, items, *, fontsize=6.6, pad_pt=3.0, avoid=(),
                       colors=None):
    """Label scatter points, choosing the first offset that collides with nothing.

    `items` is a sequence of (x, y, text) in data coordinates. Candidate
    positions are searched in display space against the markers, the labels
    already placed, the axes frame, and any `avoid` artists, so a dense cluster
    degrades into leader lines instead of overlapping text. Hand-tuned offset
    tables do the same job until the axes limits change, then silently rot.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)

    placed = []
    for x, y, _text in items:
        px, py = ax.transData.transform((x, y))
        placed.append((px - 4, py - 4, px + 4, py + 4))
    for artist in avoid:
        bb = artist.get_tightbbox(renderer)
        placed.append((bb.x0, bb.y0, bb.x1, bb.y1))

    def _hits(box):
        bx0, by0, bx1, by1 = box
        # Running past the axes frame is as bad as an overlap: the label lands
        # on the tick labels or gets cropped when the figure is saved.
        if bx0 < frame.x0 or bx1 > frame.x1 \
           or by0 < frame.y0 or by1 > frame.y1:
            return True
        return any(bx0 < ox1 + pad_pt and ox0 < bx1 + pad_pt
                   and by0 < oy1 + pad_pt and oy0 < by1 + pad_pt
                   for ox0, oy0, ox1, oy1 in placed)

    def _annotate(x, y, text, dx, dy, ha, va, leader, color):
        return ax.annotate(
            text, (x, y), xytext=(dx, dy), textcoords="offset points",
            fontsize=fontsize, color=color, ha=ha, va=va, zorder=6,
            # The search avoids other text and markers but not trend lines or
            # gridlines, so labels carry a halo to stay readable over them.
            path_effects=[withStroke(linewidth=1.9, foreground="white")],
            arrowprops=dict(arrowstyle="-", color="#B4BBC2", linewidth=0.5,
                            shrinkA=1, shrinkB=2) if leader else None,
        )

    for x, y, text in items:
        color = (colors or {}).get(text, INK)
        choice = None
        for radius in (7.0, 12.0, 18.0, 26.0):
            for dx, dy, ha, va in _COMPASS:
                probe = _annotate(x, y, text, dx * radius, dy * radius,
                                  ha, va, False, color)
                bbox = probe.get_window_extent(renderer)
                probe.remove()
                if not _hits((bbox.x0, bbox.y0, bbox.x1, bbox.y1)):
                    choice = (dx * radius, dy * radius, ha, va, radius,
                              (bbox.x0, bbox.y0, bbox.x1, bbox.y1))
                    break
            if choice:
                break
        if choice is None:
            probe = _annotate(x, y, text, 0, 30, "center", "bottom", False,
                              color)
            bbox = probe.get_window_extent(renderer)
            probe.remove()
            choice = (0, 30, "center", "bottom", 30.0,
                      (bbox.x0, bbox.y0, bbox.x1, bbox.y1))

        dx, dy, ha, va, radius, box = choice
        # Only labels pushed well clear of their marker need a leader line.
        _annotate(x, y, text, dx, dy, ha, va, radius >= 12.0, color)
        placed.append(box)


def finalize(fig, path_noext: str, *, also_png: bool = True,
             width: float | None = W_FULL, pad: float = 0.02) -> None:
    """Save pdf (+png) for a finished figure at an exact output width.

    A plain ``bbox_inches='tight'`` save crops to the content, so two figures
    built on the same canvas can come out at different widths and then be
    rescaled by different factors when LaTeX fits them to \\linewidth. Instead
    we take the tight box for the vertical extent -- there is no reason to ship
    blank rows -- and force the horizontal extent to `width`, centred on the
    content. Pass ``width=None`` for a plain tight save.
    """
    d = os.path.dirname(path_noext)
    if d:
        os.makedirs(d, exist_ok=True)

    save_kwargs: dict = {}
    if width is not None:
        fig.canvas.draw()
        box = fig.get_tightbbox(fig.canvas.get_renderer()).padded(pad)
        if box.width > width:
            # Content overflows the canvas (usually a legend anchored outside
            # the axes). Cropping it would lose information, so keep the tight
            # box and report the drift instead of silently shipping a figure
            # at the wrong scale.
            print(f"[warn] {os.path.basename(path_noext)}: content is "
                  f"{box.width:.2f}in wide, exceeds the {width:.2f}in design "
                  f"width; reserve space for artists drawn outside the axes")
        else:
            centre = 0.5 * (box.x0 + box.x1)
            box = Bbox.from_extents(centre - width / 2, box.y0,
                                    centre + width / 2, box.y1)
        save_kwargs["bbox_inches"] = box

    # Vector content ignores dpi, but the sandbox and brand marks are raster
    # and the PDF backend embeds them at this density. At the sheet default
    # of 400 a mark lands as a 75 px thumbnail and prints soft.
    fig.savefig(f"{path_noext}.pdf", **{**save_kwargs, "dpi": 1200})
    if also_png:
        fig.savefig(f"{path_noext}.png", **save_kwargs)
    plt.close(fig)
