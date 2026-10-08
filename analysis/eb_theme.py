"""Alternate figure palette for the project website. Never used by the paper.

The website (``website/``) is drawn in the Tencent Hunyuan system: ink on
white, one warm grey ramp, and a single blue reserved for the item under
discussion. The paper's figures are drawn in the house palette, which is a
cool slate with clay and walnut sandbox accents. Embedding the paper's PNGs
in the website therefore puts two colour temperatures on one page.

Rather than fork the plotting scripts, every module that owns a colour
constant ends its constant block with

    if eb_theme.enabled():
        ...  # rebind the constants from eb_theme

so the figures can be re-rendered in the website's palette by setting one
environment variable:

    EB_FIGURE_THEME=hunyuan python3 paper/figures/task_composition.py \
        --outputs website/assets/img/figures/task_composition

With the variable unset -- which is how the paper is built -- nothing in this
module is imported and no constant changes. ``scripts/render_website_figures.py``
drives the whole set and checks that the paper tree is untouched afterwards.

Values are taken from hunyuan.tencent.com's build output; see
website/assets/css/site.css, which is the same palette in CSS.
"""

from __future__ import annotations

import os
from pathlib import Path

_VAR = "EB_FIGURE_THEME"
_NAME = "hunyuan"

# The website keeps its own monochrome cut of the two sandbox marks, the same
# glyphs on the same 64-unit grid. The paper's marks are a clay tile and a
# beige one, which is the loudest thing left in an otherwise neutral figure.
MARKS = Path(__file__).resolve().parents[2] / "website/assets/img/mark"


def enabled() -> bool:
    return os.environ.get(_VAR) == _NAME


def marks(default):
    """The sandbox-mark directory a figure should draw from."""
    return MARKS if enabled() else default


# ---------------------------------------------------------------------------
# Chrome. Axes, ticks and grid cover more of a figure than the data does, so
# these carry most of the colour temperature.
# ---------------------------------------------------------------------------
INK = "#1a1a1a"        # titles, value labels, anything that must read as text
MUTED = "#00000066"    # rgba(0,0,0,.4): tick labels, captions, footnotes
SUBTLE = "#00000099"   # rgba(0,0,0,.6): secondary prose
AXIS = "#e6e6e6"       # spines and reference rules -- lines only, never type
GRID = "#f1f1f1"       # hairline grid
ZONE = "#f3f3f3"       # shaded regions
PAPER = "#f8f1e7"

# The warm grey ramp (Tailwind stone). Warm, not slate: a cool grey next to
# the website's beige band reads as a different page.
STONE = {
    900: "#1c1917", 800: "#292524", 600: "#57534d", 500: "#79716b",
    450: "#8c8884", 400: "#a6a09b", 300: "#d6d3d1", 200: "#e7e5e4",
}
BLUE = "#2d68ff"       # the one colour, and only for the item being discussed

# ---------------------------------------------------------------------------
# Semantic roles, mapped onto that ramp
# ---------------------------------------------------------------------------
# Exploration modes. Five hues collapse to blue plus four values, so anything
# drawing all five at once has to separate them by width, dash or hatch as
# well -- which the trajectory panels already do.
ARM = {
    "self": BLUE,
    "passive": STONE[500],
    "random": STONE[900],
    # A panel drawing all five puts these two on white as thin lines. #d6d3d1
    # and #e7e5e4 disappear there, so the two quietest arms are pulled one and
    # two steps up the ramp; they stay the two quietest, which is the point.
    "think": STONE[400],
    "none": STONE[300],
}
# The website never says good and bad in red and green: blue is the subject,
# grey is everything else.
VALENCE = {"good": BLUE, "bad": STONE[500]}
PAIR = (STONE[900], STONE[400])
# Sandbox identity. The figures separate sandboxes by panel, so this is a
# redundant encoding and can safely lose its hue.
BRAND = {"AlienCode": STONE[900], "AlienLogic": STONE[500]}

# A diverging scale still has to diverge. Grey through white to blue keeps the
# two directions apart without spending a second hue on it.
DIVERGING = (STONE[500], "#f5f5f4", BLUE)
DIVERGING_BAD = "#f3f3f3"

BASELINE = STONE[400]


# ---------------------------------------------------------------------------
# rcParams. Tick and text colour are set inside the style helpers rather than
# from a module constant, so they are overridden as a block.
# ---------------------------------------------------------------------------
def rcparams() -> dict:
    return {
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": AXIS,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        # The house grid is a mid grey at 45% opacity. #f1f1f1 is already the
        # finished weight, so it is drawn at full strength.
        "grid.alpha": 1.0,
    }
