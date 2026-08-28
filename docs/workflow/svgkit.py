"""Minimal SVG primitives shared by the workflow panels.

Every helper returns an SVG fragment as a string, so a panel is just a list of
fragments and a size. Colours come from the app itself, keeping the schematic in
step with what the plots actually draw.
"""

from __future__ import annotations

import sys
from pathlib import Path
import math
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from libraont.plots import FATE_COLORS                      # noqa: E402
from libraont.theme import PALETTE                          # noqa: E402

FONT = "Inter, Segoe UI, Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

INK = PALETTE["text"]
MUTED = PALETTE["muted"]
GRID = PALETTE["grid"]
SURFACE = PALETTE["surface"]
PRIMARY = PALETTE["primary"]
DARK = PALETTE["primary_dark"]
DROP = "#C44E5A"            # reads leaving the workflow
KEEP = FATE_COLORS["Variant"]


def rect(x, y, w, h, fill=SURFACE, stroke=GRID, rx=0, sw=1, opacity=1.0):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>')


def text(x, y, s, size=12, fill=INK, anchor="start", weight="400", family=FONT,
         opacity=1.0):
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" '
            f'opacity="{opacity}">{escape(str(s))}</text>')


def line(x1, y1, x2, y2, stroke=GRID, sw=1, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{sw}"{d}/>')


def group(*parts, transform=""):
    t = f' transform="{transform}"' if transform else ""
    return f"<g{t}>" + "".join(parts) + "</g>"


def panel(title, subtitle, w, h, body, tag="", title_size=16):
    """A titled card: soft frame, rounded corners."""
    head = [rect(0, 0, w, h, fill="#FFFFFF", stroke=GRID, rx=10),
            text(22, 40, title, size=title_size, weight="700", fill=DARK)]
    if tag:
        head.append(text(w - 22, 40, tag, size=12, weight="700", fill=MUTED,
                         anchor="end"))
    if subtitle:
        head.append(text(22, 58, subtitle, size=10.5, fill=MUTED))
    return group(*head, body)


def card(x, y, w, h, label, accent, size=9, rx=7):
    """A small output chip, mirroring the app's metric cards. ``label`` may be a
    string or a list of lines, centred as a block."""
    lines = [label] if isinstance(label, str) else list(label)
    lh, top = size * 1.15, y + h / 2 + size * 0.39
    return "".join([
        rect(x, y, w, h, fill=SURFACE, stroke=GRID, rx=rx),
        rect(x + 1, y + 0.5, w - 2, 3, fill=accent, stroke="none", rx=1.5),
    ] + [text(x + 11, top + (k - (len(lines) - 1) / 2) * lh, part,
              size=size, weight="600", fill=DARK)
         for k, part in enumerate(lines)])


def document(w, h, body, title="LibraONT workflow"):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}">'
            f"<title>{escape(title)}</title>"
            '<defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{MUTED}"/></marker></defs>'
            f'<rect width="{w}" height="{h}" fill="#FFFFFF"/>'
            f"{body}</svg>")


def write(path, w, h, body, title="LibraONT workflow"):
    Path(path).write_text(document(w, h, body, title), encoding="utf-8")
    return path

def ribbon(x0, top0, bot0, x1, top1, bot1, fill, opacity=0.55):
    """A Sankey link: a band curving from one node edge to the next."""
    cx = (x0 + x1) / 2
    return (f'<path d="M {x0},{top0} C {cx},{top0} {cx},{top1} {x1},{top1} '
            f'L {x1},{bot1} C {cx},{bot1} {cx},{bot0} {x0},{bot0} Z" '
            f'fill="{fill}" opacity="{opacity}"/>')

def arc(cx, cy, r, a0, a1, stroke, sw=2, dash=None, cap="round"):
    """Circular arc from ``a0`` to ``a1`` degrees, clockwise, 0 = three o'clock."""
    x0, y0 = cx + r * math.cos(math.radians(a0)), cy + r * math.sin(math.radians(a0))
    x1, y1 = cx + r * math.cos(math.radians(a1)), cy + r * math.sin(math.radians(a1))
    large = 1 if abs(a1 - a0) > 180 else 0
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="M {x0:.2f},{y0:.2f} A {r},{r} 0 {large},1 {x1:.2f},{y1:.2f}" '
            f'fill="none" stroke="{stroke}" stroke-width="{sw}" '
            f'stroke-linecap="{cap}"{d}/>')


def circle(cx, cy, r, stroke, sw=2, fill="none", dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}"{d}/>')
