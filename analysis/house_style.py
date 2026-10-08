"""Shared conventions for the paper's figures.

Colour carries one of two meanings and never a third:

  ``ARM`` identifies an exploration mode. These are the values the milestone
  trajectories and the oracle comparison already use, so a reader who learned
  the code in one figure keeps it in the next.

  ``VALENCE`` marks a quantity as favourable or not, matching the endpoints of
  the diverging colormap used for the task-family heatmaps.

Sandboxes are never distinguished by colour. They get their own panel, titled
and marked, which is what the earlier figures do.
"""

from __future__ import annotations

import io
import math
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.patheffects as path_effects
from matplotlib.offsetbox import (
    AnchoredOffsetbox,
    AnnotationBbox,
    DrawingArea,
    HPacker,
    OffsetImage,
    TextArea,
    VPacker,
)

ARM = {
    "self": "#3A5F8A",
    "passive": "#E4985D",
    "random": "#63A69F",
    "think": "#79A7C8",
    "none": "#B8BDC6",
}
VALENCE = {"good": "#3A5F8A", "bad": "#B8614A"}
# Sandbox identity, matching \AlienCodeClay and \AlienLogicWalnut in main.tex.
# Used for table tints, marks, and the appendix figures. Body figures separate
# sandboxes by panel instead, so these rarely appear there.
BRAND = {"AlienCode": "#CC785C", "AlienLogic": "#4F4234"}
PAIR = ("#79A7C8", "#E4985D")
HATCH = ("///", "\\\\\\")

# Probes the protocol allows across the four scored rounds. A cell above this
# ran extra rounds: the matrix truncates its milestones back to M4 but keeps the
# full budget, so its gain and its cost would be measured over different spans.
PROBE_CEILING = {"AlienLogic": 32, "AlienCode": 360}


def within_protocol(cell: dict) -> bool:
    ceiling = PROBE_CEILING.get(cell["sandbox"])
    return ceiling is None or (cell.get("probe_units") or 0) <= ceiling


AXIS = "#68717B"
GRID = "#E4E7E9"
MUTED = "#7A8087"
INK = "#30343B"

# The website embeds these figures beside its own ink-on-white chrome, so it
# re-renders them from the same scripts with the palette swapped. Unset in the
# paper build, where nothing below runs. See eb_theme.
import eb_theme  # noqa: E402

if eb_theme.enabled():
    AXIS = eb_theme.AXIS
    GRID = eb_theme.GRID
    MUTED = eb_theme.MUTED
    INK = eb_theme.INK
    ARM = dict(eb_theme.ARM)
    VALENCE = dict(eb_theme.VALENCE)
    BRAND = dict(eb_theme.BRAND)
    PAIR = tuple(eb_theme.PAIR)
    # SYSTEM_COLOR and SYSTEM_COLOR_MUTED below are deliberately not touched:
    # they are vendor identity, and the muted set's spacing was solved under a
    # colourblindness simulation. Flattening ten curves to grey would undo it.

# Badges keyed on the display names the matrix carries, matching the icons
# \modelname sets in the tables so a system looks the same wherever it appears.
BRANDICON = {
    "GPT-5.6-sol": "openai",
    "GPT-5.6 Sol": "openai",
    "Qwen 3.8 Max": "qwen-color",
    "Qwen3.8-Max": "qwen-color",
    "Grok 4.5": "grok",
    "Grok 4.6": "grok",
    "DeepSeek V4 Pro": "deepseek-color",
    "DeepSeek V4 Pro (0813)": "deepseek-color",
    "DeepSeek V4 Flash (0731)": "deepseek-color",
    "DeepSeek Flash": "deepseek-color",
    "DS V4.1 Flash": "deepseek-color",
    "DS V4 Pro": "deepseek-color",
    "DeepSeek-V4.1-Flash": "deepseek-color",
    "DeepSeek-V4-Pro": "deepseek-color",
    "Kimi K3": "kimi",
    "Gemini 3.6 Flash": "gemini-color",
    "Gemini 3.8 Flash": "gemini-color",
    "Claude Opus 4.8": "claude-ai",
    "Claude Opus 5": "claude-ai",
    "Hy4 preview": "hunyuan-color",
    "Doubao Seed 2.1 Pro": "bytedance-light",
    "Seed 2.1": "bytedance-light",
    "Seed2.1 Pro": "bytedance-light",
}

# One curve colour per system, shared by every figure that separates systems,
# so a reader who learns a colour in one figure keeps it in the next. Each is
# taken from its vendor's own mark, then pulled apart where two marks collide:
# four of the ten logos are pure black and four are a similar blue, which no
# amount of good faith renders as ten distinguishable curves.
#
#   black    OpenAI keeps it; xAI drops to grey and Kimi to a blue-black,
#            since all three marks are monochrome and only rank can separate
#            them.
#   blue     DeepSeek keeps #4D6BFE and its Flash sibling takes a tint of it,
#            which is right: they share a badge. Hunyuan moves to the cyan in
#            its own gradient, and Gemini to the green in its four-colour
#            mark, because both are otherwise the same blue as DeepSeek.
#   red      ByteDance's mark is black, so Doubao borrows the house red.
SYSTEM_COLOR = {
    "GPT-5.6-sol": "#0D0D0D",
    "GPT-5.6 Sol": "#0D0D0D",
    "Claude Opus 4.8": "#D97757",
    "Claude Opus 5": "#B8614A",
    "Qwen 3.8 Max": "#6336E7",
    "Qwen3.8-Max": "#6336E7",
    "Grok 4.5": "#7C838C",
    "Grok 4.6": "#7C838C",
    "DeepSeek V4 Pro": "#4D6BFE",
    "DeepSeek V4 Pro (0813)": "#4D6BFE",
    "DeepSeek V4 Flash (0731)": "#7C93F2",
    # The gateway's own call name for the current fast DeepSeek. It is not
    # the 0731 flash and must not borrow that entry: they are different
    # models and the v2 cohort contains only this one.
    "DeepSeek Flash": "#7C93F2",
    "DS V4.1 Flash": "#7C93F2",
    "DS V4 Pro": "#4D6BFE",
    "DeepSeek-V4.1-Flash": "#7C93F2",
    "DeepSeek-V4-Pro": "#4D6BFE",
    "Kimi K3": "#2B3A55",
    "Gemini 3.6 Flash": "#1BA672",
    "Gemini 3.8 Flash": "#1BA672",
    "Hy4 preview": "#00A0C4",
    "Doubao Seed 2.1 Pro": "#D0343F",
    "Seed 2.1": "#D0343F",
    "Seed2.1 Pro": "#D0343F",
}

# The same ten systems, pulled into the register the rest of the paper is drawn
# in. Vendor colour is right where a system is a point in a cloud, but ten
# vendor curves crossing in one panel put five full-chroma hues beside the
# muted slate, clay and teal every other figure uses, and the panel reads as a
# different paper. The chromas here are the ones the house palette already
# spends -- roughly 25 to 44 against vendor's 35 to 103 -- and each hue still
# points at its own mark: OpenAI stays monochrome, Claude keeps the clay, the
# DeepSeek pair keeps one blue in two values, Doubao keeps a red.
#
# Within those bounds the ten were placed to maximise the smallest CIE Lab
# distance between any two, measured both in normal vision and through a
# deuteranope simulation, which is what the vendor set cannot do: its four
# blues and three blacks leave neighbouring curves 11 units apart to a
# red-green colourblind reader, against 19 here. Every value also sits between
# L* 13 and 71, so no curve fades into the page and none reads as ink.
SYSTEM_COLOR_MUTED = {
    "GPT-5.6-sol": "#1F2126",             # charcoal; the mark is monochrome
    "GPT-5.6 Sol": "#1F2126",
    "Claude Opus 4.8": "#D98A63",         # the house clay
    "Claude Opus 5": "#B8614A",           # same clay, one value darker
    "Qwen 3.8 Max": "#8671BA",            # violet at a quarter of #6336E7's chroma
    "Qwen3.8-Max": "#8671BA",
    "Grok 4.5": "#A6A8AC",                # neutral grey, a value above the grid
    "Grok 4.6": "#A6A8AC",
    "DeepSeek V4 Pro": "#3A527C",         # the house slate
    "DeepSeek V4 Pro (0813)": "#3A527C",
    "DeepSeek V4 Flash (0731)": "#92AEDF",  # the same blue, two values lighter
    "DeepSeek Flash": "#92AEDF",
    "DS V4.1 Flash": "#92AEDF",
    "DS V4 Pro": "#3A527C",
    "DeepSeek-V4.1-Flash": "#92AEDF",
    "DeepSeek-V4-Pro": "#3A527C",
    "Kimi K3": "#6E6560",                 # warm graphite, off the blue axis
    "Gemini 3.6 Flash": "#5D9E6D",        # moss, not emerald
    "Gemini 3.8 Flash": "#5D9E6D",
    "Hy4 preview": "#2F8B98",             # teal, the cool end of the house pair
    "Doubao Seed 2.1 Pro": "#9F443C",     # brick
    "Seed 2.1": "#9F443C",
    "Seed2.1 Pro": "#9F443C",
}

# Shape per system, matching the key the cost-of-a-point figure prints, so a
# reader who learns a system's mark there keeps it here. Colour alone cannot
# separate ten crossing curves at print size however well it is chosen, and
# shape survives both a greyscale print and a colourblind reader.
SYSTEM_MARKER = {
    "GPT-5.6-sol": "o",
    "GPT-5.6 Sol": "o",
    "Claude Opus 4.8": "*",
    "Claude Opus 5": "*",
    "Qwen 3.8 Max": "v",
    "Qwen3.8-Max": "v",
    "Grok 4.5": "P",
    "Grok 4.6": "P",
    "DeepSeek V4 Pro": "s",
    "DeepSeek V4 Pro (0813)": "s",
    "DeepSeek V4 Flash (0731)": "D",
    "DeepSeek Flash": "D",
    "DS V4.1 Flash": "D",
    "DS V4 Pro": "s",
    "DeepSeek-V4.1-Flash": "D",
    "DeepSeek-V4-Pro": "s",
    "Kimi K3": "X",
    "Gemini 3.6 Flash": "^",
    "Gemini 3.8 Flash": "^",
    "Doubao Seed 2.1 Pro": "<",
    "Seed 2.1": "<",
    "Seed2.1 Pro": "<",
    "Hy4 preview": ">",
}
# A marker's drawn area, not its bounding box, is what a reader sees, and a
# star or a thin diamond spends much less of that box than a disc.
MARKER_SCALE = {"*": 1.45, "P": 1.15, "X": 1.15, "D": 1.10, "s": 0.95}

# What to call a system in a slot too narrow for its full name. A badge already
# carries the vendor, so these carry the version, which is the part a reader
# cannot recover from the logo: the two DeepSeek entries share one badge and are
# told apart only here. Kept short enough to set on one line over a cell of a
# ten-column grid, the tightest slot in the paper.
SYSTEM_SHORT = {
    "GPT-5.6-sol": "GPT-5.6-sol",
    "GPT-5.6 Sol": "GPT-5.6 Sol",
    "Claude Opus 4.8": "Claude Opus 4.8",
    "Claude Opus 5": "Claude Opus 5",
    "Qwen 3.8 Max": "Qwen 3.8 Max",
    "Qwen3.8-Max": "Qwen3.8-Max",
    "Grok 4.5": "Grok 4.5",
    "Grok 4.6": "Grok 4.6",
    "DeepSeek V4 Pro": "V4 Pro",
    "DeepSeek V4 Pro (0813)": "V4 Pro",
    "DeepSeek V4 Flash (0731)": "V4 Flash",
    "DeepSeek Flash": "DS Flash",
    "DS V4.1 Flash": "DS V4.1 Flash",
    "DS V4 Pro": "DS V4 Pro",
    "DeepSeek-V4.1-Flash": "DeepSeek-V4.1-Flash",
    "DeepSeek-V4-Pro": "DeepSeek-V4-Pro",
    "Kimi K3": "Kimi K3",
    "Gemini 3.6 Flash": "Gemini 3.6",
    "Gemini 3.8 Flash": "Gemini 3.8 Flash",
    "Doubao Seed 2.1 Pro": "Doubao 2.1",
    "Seed 2.1": "Seed 2.1",
    "Seed2.1 Pro": "Seed2.1 Pro",
    "Hy4 preview": "Hy4 preview",
}


def label_ink(color: str, floor: float = 0.20) -> tuple:
    """Darken a curve colour just enough for small bold type to hold up.

    A value printed beside its own curve is set in the curve's colour, which
    keeps the two associated without a leader. That works down to the middle of
    the value range and then stops: the pale blue and the grey that read as
    curves against white do not read as text. Blending toward the colour's own
    hue at full chroma is wrong too, since it would make the label louder than
    the line, so the blend runs toward black and stops as soon as the label
    clears the same contrast the body text has.
    """
    r, g, b = mpl.colors.to_rgb(color)
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if luma <= floor:
        return (r, g, b)
    amount = min(0.55, 1.0 - floor / luma)
    return (r * (1 - amount), g * (1 - amount), b * (1 - amount))

# Matches the gap the oracle-comparison figure renders at.
_TITLE_GAP_PT = 6.6
_RASTER_PX = 512
_RASTER_BASE = 128

# Small capitals run at 0.800 of the cap height, which is what \textsc gives
# in the milestone-trajectory figure. Scaling a capital down also thins its
# stems by the same factor, so the reduced run is stroked back to the weight
# of the full capitals; without this the faked small caps read as a lighter
# second font beside the real ones in the body.
_SMALLCAP_RATIO = 0.800
_TIMES_STEM_EM = 0.09

# Art beside type is sized the way the body macros size it: \aliencodebrand
# sets the sandbox mark at 1.08em and \modelname the system badge at 0.92em.
# An OffsetImage renders at ``_RASTER_BASE * zoom`` points whatever the raster
# resolution, so the zoom follows from the type size.
_MARK_EM = 1.08
_BADGE_EM = 0.92
# \aliencodebrand and \modelname both set this much air between art and name.
_BRAND_GAP_EM = 0.28
# Centring art on the name centres it on the text box, and that box keeps a
# little of the font's depth while a name in capitals sits on the baseline,
# so the art lands low. Measured against \aliencodebrand as the body renders
# it -- mark 1.61x the cap height, overhanging evenly -- the lift is 0.106 em.
_BRAND_LIFT_EM = 0.106


def _on_capitals(art, size: float):
    """Lift art onto the band of capitals, where the body macro puts it."""
    spacer = DrawingArea(0, 2 * abs(_BRAND_LIFT_EM) * size, 0, 0)
    order = [art, spacer] if _BRAND_LIFT_EM > 0 else [spacer, art]
    return VPacker(children=order, pad=0, sep=0)


def _art_zoom(size: float, zoom: float | None, em: float) -> float:
    return em * size / _RASTER_BASE if zoom is None else zoom


def _smallcap_stroke(size: float) -> float:
    """Stem width a capital loses on the way down to small-cap size."""
    return size * (1.0 - _SMALLCAP_RATIO) * _TIMES_STEM_EM


def _read_art(svg: Path, png: Path):
    """Prefer the vector source, rasterised well above its rendered size.

    A badge is drawn a few points tall. Embedding the shipped PNG at that size
    is what made the earlier figures look soft, so the SVG is rasterised large
    and the caller divides its zoom by the returned scale.
    """
    if svg.is_file():
        try:
            done = subprocess.run(
                ["rsvg-convert", "--format=png", f"--width={_RASTER_PX}",
                 f"--height={_RASTER_PX}", str(svg)],
                check=True, stdout=subprocess.PIPE,
            )
            return (mpimg.imread(io.BytesIO(done.stdout), format="png"),
                    _RASTER_PX / _RASTER_BASE)
        except (OSError, subprocess.CalledProcessError):
            pass
    if png.is_file():
        return mpimg.imread(str(png)), 1.0
    return None, 1.0


def read_mark(marks: Path, sandbox: str):
    """Load a sandbox mark, rasterising the vector source when available."""
    slug = sandbox.lower().replace("alien", "")
    marks = eb_theme.marks(marks)
    return _read_art(marks / f"alien{slug}_mark.svg",
                     marks / f"alien{slug}_mark.png")


def read_brandicon(icons: Path, model: str):
    """Load a system badge, rasterising the vector source when available."""
    slug = BRANDICON.get(model)
    if slug is None:
        return None, 1.0
    return _read_art(icons / f"{slug}.svg", icons / f"{slug}.png")


def _smallcaps_runs(text: str, size: float, weight: str):
    """Set a sandbox name in small capitals.

    Under the LaTeX renderer this is a single ``\\textsc`` run, the same call
    the milestone-trajectory figure makes, so the reduced letters are the
    font's own small capitals and carry the stem weight of the full ones.
    """
    if mpl.rcParams.get("text.usetex"):
        body = rf"\textsc{{{text}}}"
        if weight == "bold":
            body = rf"\textbf{{{body}}}"
        return [TextArea(body, textprops={"size": size, "color": INK})]
    return _faux_smallcaps_runs(text, size, weight)


def _faux_smallcaps_runs(text: str, size: float, weight: str):
    """Split a name into full-size capitals and reduced-size uppercase runs.

    The milestone-trajectory figure gets true small caps from ``\\textsc`` under
    a LaTeX text renderer. These figures do not use that renderer, so the same
    look is built from two text sizes, which keeps the sandbox names matching
    the body without pulling a second font stack into every figure.
    """
    runs: list[tuple[str, float, bool]] = []
    for char in text:
        reduced = char.islower()
        glyph = char.upper()
        scale = size * _SMALLCAP_RATIO if reduced else size
        if runs and runs[-1][1] == scale:
            runs[-1] = (runs[-1][0] + glyph, scale, reduced)
        else:
            runs.append((glyph, scale, reduced))
    stroke = _smallcap_stroke(size)
    areas = []
    for glyph, scale, reduced in runs:
        props = {"size": scale, "weight": weight, "color": INK}
        if reduced and stroke > 0:
            props["path_effects"] = [
                path_effects.withStroke(linewidth=stroke, foreground=INK)]
        areas.append(TextArea(glyph, textprops=props))
    return areas


def smallcaps_title(ax, sandbox: str, marks: Path | None = None, *,
                    letter: str | None = None, size: float = 9.6,
                    weight: str = "normal",
                    zoom: float | None = None) -> None:
    """Draw a sandbox mark and its small-caps name as the panel title.

    Laid out the way \\aliencodebrand lays it out in the body: mark first, one
    ``_BRAND_GAP_EM`` of air, then the name. Centring the pack aligns the mark
    on the band of capitals, which is what the macro's \\raisebox does.
    """
    # Letter runs butt against each other; only the mark gets the gap.
    word = HPacker(children=_smallcaps_runs(sandbox, size, weight),
                   pad=0, sep=0, align="baseline")
    children = []
    if letter is not None:
        children.append(TextArea(f"({letter})",
                                 textprops={"size": size, "weight": weight,
                                            "color": INK}))
    if marks is not None:
        image, scale = read_mark(marks, sandbox)
        if image is not None:
            children.append(_on_capitals(OffsetImage(
                image, zoom=_art_zoom(size, zoom, _MARK_EM) / scale), size))
    children.append(word)
    packed = HPacker(children=children, pad=0, sep=_BRAND_GAP_EM * size,
                     align="center")
    ax.add_artist(AnchoredOffsetbox(
        loc="lower left", child=packed, pad=0, borderpad=0, frameon=False,
        bbox_to_anchor=(0.0, 1.012), bbox_transform=ax.transAxes,
    ))


def badge_title(ax, icons: Path, model: str, *, size: float = 9.6,
                weight: str = "normal", zoom: float | None = None,
                y: float = 1.012) -> None:
    """Set a system's badge and name as the panel title.

    Badge and name are packed into one box, so the gap is fixed in points and
    the badge stays in proportion to the type at any size.
    """
    children = []
    image, scale = read_brandicon(icons, model)
    if image is not None:
        children.append(OffsetImage(
            image, zoom=_art_zoom(size, zoom, _BADGE_EM) / scale))
    children.append(TextArea(model, textprops={"size": size, "weight": weight,
                                               "color": INK}))
    packed = HPacker(children=children, pad=0, sep=_BRAND_GAP_EM * size,
                     align="center")
    ax.add_artist(AnchoredOffsetbox(
        loc="lower left", child=packed, pad=0, borderpad=0, frameon=False,
        bbox_to_anchor=(0.0, y), bbox_transform=ax.transAxes,
    ))


def smallcaps_at(ax, sandbox: str, xy, marks: Path | None = None, *,
                 size: float = 8.2, weight: str = "normal",
                 zoom: float | None = None, transform=None) -> None:
    """Draw a small-caps sandbox name anchored at an arbitrary point."""
    word = HPacker(children=_smallcaps_runs(sandbox, size, weight),
                   pad=0, sep=0, align="baseline")
    children = []
    if marks is not None:
        image, scale = read_mark(marks, sandbox)
        if image is not None:
            children.append(_on_capitals(OffsetImage(
                image, zoom=_art_zoom(size, zoom, _MARK_EM) / scale), size))
    children.append(word)
    packed = HPacker(children=children, pad=0, sep=_BRAND_GAP_EM * size,
                     align="center")
    ax.add_artist(AnnotationBbox(
        packed, xy, xycoords=transform or ax.transAxes,
        box_alignment=(0.5, 1.0), frameon=False, pad=0,
        annotation_clip=False,
    ))


def smallcaps_figure_text(figure, sandbox: str, xy, marks: Path | None = None,
                          *, size: float = 10.5, weight: str = "normal",
                          zoom: float | None = None) -> None:
    """Draw a small-caps sandbox name, with its mark, in figure coordinates."""
    word = HPacker(children=_smallcaps_runs(sandbox, size, weight),
                   pad=0, sep=0, align="baseline")
    children = []
    if marks is not None:
        image, scale = read_mark(marks, sandbox)
        if image is not None:
            children.append(_on_capitals(OffsetImage(
                image, zoom=_art_zoom(size, zoom, _MARK_EM) / scale), size))
    children.append(word)
    packed = HPacker(children=children, pad=0, sep=_BRAND_GAP_EM * size,
                     align="center")
    figure.add_artist(AnnotationBbox(
        packed, xy, xycoords=figure.transFigure, box_alignment=(0.5, 0.0),
        frameon=False, pad=0, annotation_clip=False,
    ))


def mark_after_title(ax, marks: Path, sandbox: str, zoom: float = 0.115) -> None:
    """Set the sandbox mark beside a left-aligned panel title.

    Call this once the layout is final. The anchor is measured from the
    rendered title, and any later resize of the axes invalidates it. The gap is
    in points rather than axes fractions so it does not shrink with the panel.
    """
    image, scale = read_mark(marks, sandbox)
    if image is None:
        return
    figure = ax.figure
    figure.canvas.draw()
    # A left-aligned title lives on the private artist; ax.title stays empty
    # and would measure zero width.
    title = next(
        (artist for artist in (getattr(ax, "_left_title", None), ax.title)
         if artist is not None and artist.get_text()),
        ax.title,
    )
    extent = title.get_window_extent(renderer=figure.canvas.get_renderer())
    pad = _TITLE_GAP_PT * figure.dpi / 72.0
    x, y = ax.transAxes.inverted().transform(
        (extent.x1 + pad, (extent.y0 + extent.y1) / 2.0))
    ax.add_artist(AnnotationBbox(
        OffsetImage(image, zoom=zoom / scale), (x, y), xycoords=ax.transAxes,
        box_alignment=(0.0, 0.5), frameon=False, pad=0, zorder=6,
        annotation_clip=False,
    ))


def badge_ticks(ax, icons: Path, models: list[str], *, axis: str = "y",
                zoom: float | None = None, gap: float = 3.2) -> None:
    """Set each system's badge beside its categorical tick label.

    Call this once the layout is final: the anchor is measured from the
    rendered labels. Reserve a little room for the badge column in whatever
    ``tight_layout`` rect the caller passes, since the labels themselves are
    all the layout can see.
    """
    figure = ax.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    labels = (ax.get_yticklabels() if axis == "y" else ax.get_xticklabels())
    pad = gap * figure.dpi / 72.0
    inverse = ax.transAxes.inverted()
    for label, model in zip(labels, models):
        image, scale = read_brandicon(icons, model)
        if image is None:
            continue
        extent = label.get_window_extent(renderer=renderer)
        if axis == "y":
            anchor = (extent.x0 - pad, (extent.y0 + extent.y1) / 2.0)
            alignment = (1.0, 0.5)
        elif label.get_rotation() % 180:
            # A rotated label runs down-left from its tick, so the badge
            # continues the line of text at the far end rather than sitting
            # over the middle of a slanted bounding box.
            anchor = (extent.x0 - pad * 0.5, extent.y0 - pad * 0.2)
            alignment = (1.0, 0.25)
        else:
            anchor = ((extent.x0 + extent.x1) / 2.0, extent.y1 + pad)
            alignment = (0.5, 0.0)
        ax.add_artist(AnnotationBbox(
            OffsetImage(image, zoom=_art_zoom(label.get_fontsize(), zoom,
                                              _BADGE_EM) / scale),
            inverse.transform(anchor),
            xycoords=ax.transAxes, box_alignment=alignment, frameon=False,
            pad=0, zorder=6, annotation_clip=False,
        ))


def _axis_fraction(value: float, limits: tuple[float, float],
                   scale: str) -> float:
    low, high = limits
    if scale == "log":
        value, low, high = (math.log10(v) for v in (value, low, high))
    return 0.5 if high == low else (value - low) / (high - low)


def badge_point(ax, icons: Path, model: str, xy, *, size: float = 6.6,
                zoom: float = 0.062, gap: float = 2.4,
                offset: float = 5.0, label: str | None = None) -> None:
    """Label a plotted point with the system's badge and its name.

    The label sits above the point and is pulled inboard near a panel edge,
    which is where a centred label would otherwise run off or land on top of
    its neighbour. Pass ``label`` to print something shorter than ``model``
    while still resolving the badge from the full name, which is what a panel
    labelling every system needs to keep neighbouring labels apart.
    """
    image, scale = read_brandicon(icons, model)
    # An empty label is a caller asking for the badge alone, so test for None
    # rather than falsiness.
    text = TextArea(model if label is None else label,
                    textprops={"size": size, "color": INK})
    box = text if image is None else HPacker(
        children=[OffsetImage(image, zoom=_art_zoom(size, zoom, _BADGE_EM)
                              / scale), text],
        align="center", pad=0, sep=gap,
    )
    # Read the limits rather than the transform: callers place badges once the
    # panel is configured, and the transform still holds the autoscale default
    # until the first draw.
    x_frac = _axis_fraction(xy[0], ax.get_xlim(), ax.get_xscale())
    y_frac = _axis_fraction(xy[1], ax.get_ylim(), ax.get_yscale())
    if x_frac > 0.84:
        horizontal = 1.0
    elif x_frac < 0.14:
        horizontal = 0.0
    else:
        horizontal = 0.5
    below = y_frac > 0.86
    ax.add_artist(AnnotationBbox(
        box, xy, xybox=(0, -offset if below else offset),
        xycoords="data", boxcoords="offset points",
        box_alignment=(horizontal, 1.0 if below else 0.0),
        frameon=False, pad=0, zorder=7, annotation_clip=False,
    ))


def mark_left_of_tick(ax, marks: Path, sandbox: str, index: int,
                      zoom: float = 0.105) -> None:
    """Set the sandbox mark immediately left of a categorical tick label.

    Call this once the layout is final. Like the title variant, the anchor is
    measured from rendered text and the gap is held in points.
    """
    image, scale = read_mark(marks, sandbox)
    if image is None:
        return
    figure = ax.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    label = ax.get_xticklabels()[index]
    extent = label.get_window_extent(renderer=renderer)
    pad = _TITLE_GAP_PT * figure.dpi / 72.0
    inverse = ax.transAxes.inverted()
    x, y = inverse.transform(
        (extent.x0 - pad, (extent.y0 + extent.y1) / 2.0))
    ax.add_artist(AnnotationBbox(
        OffsetImage(image, zoom=zoom / scale), (x, y), xycoords=ax.transAxes,
        box_alignment=(1.0, 0.5), frameon=False, pad=0, zorder=6,
        annotation_clip=False,
    ))
