"""Render both figures.

    python docs/workflow/build.py

Writes ``workflow.svg`` (the filtering waterfall and what each pool feeds) and
``criteria.svg`` (one diagram per classification rule). Either panel script also
runs standalone and writes the same file.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import panel_criteria
import panel_sankey
from svgkit import write

FIGURES = [
    ("workflow.svg", panel_sankey.draw, "LibraONT workflow"),
    ("criteria.svg", panel_criteria.draw, "LibraONT classification rules"),
]


if __name__ == "__main__":
    for name, draw, title in FIGURES:
        svg, w, h = draw()
        print(write(Path(__file__).with_name(name), w, h, svg, title), f"({w}x{h})")
