"""The workflow as a Sankey: reads flowing left to right, losses peeling away.

The trunk runs under a flat top edge and narrows by exactly the branch that
drops away at each node, so what leaves and what carries on always add up. Every
branch is the same thickness and lands on one baseline - each says *that* reads
leave, not how many. Outputs sit above the pool they are built from.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from steps import PLOT_ICONS, STEPS
from svgkit import (DARK, DROP, KEEP, PRIMARY, arc, card, circle, line, panel,
                    rect, ribbon, text, write)

W, H = 1080, 505
NODE_W, LINK_W, LEFT = 11, 148, 30
CHIP_W, CHIP_H, CHIP_TOP, CHIP_GAP = 148, 40, 64, 8
ICON_H, ICON_GAP = 18, 4            # the wordless picture sitting on each chip
ROW = ICON_H + ICON_GAP + CHIP_H
TRUNK_TOP, TRUNK_H = 206, 165
STUB = 140          # the flow entering from the left, so pool one has a band too
LOSS_Y, BRANCH_H = 404, 21          # every branch lands here, at one thickness
GLYPH_Y, GLYPH_R = 458, 17          # the little plasmid under each branch
TITLE_PT, POOL_PT, CHIP_PT, LOSS_PT = 24, 14.25, 13.5, 12.75
BACKBONE, INSERT, CODON = "#B4BAC1", "#7FB3A4", "#E8913C"


def _wrap(s, width):
    out, cur = [], ""
    for word in s.split():
        if len(cur) + len(word) + 1 > width:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    return out + ([cur] if cur else [])


def _tick(cx, cy, deg, colour, reach=5, sw=1.4, r=None):
    """A short radial mark, on the plasmid ring by default or on any radius."""
    r = GLYPH_R if r is None else r
    a = math.radians(deg)
    return line(cx + (r - reach) * math.cos(a), cy + (r - reach) * math.sin(a),
                cx + (r + reach) * math.cos(a), cy + (r + reach) * math.sin(a),
                stroke=colour, sw=sw)


def _plasmid(cx, cy, insert=True, gap=False, junctions=False):
    """The reference: a circular plasmid with its insert arc, optionally with the
    two vector-insert junctions ticked."""
    parts = [arc(cx, cy, GLYPH_R, 50, 310, BACKBONE, 2.4)]
    if insert:
        parts.append(arc(cx, cy, GLYPH_R, -50, 50, INSERT, 3.4))
    elif gap:                                   # insert missing: an open circle
        parts.append(arc(cx, cy, GLYPH_R, -50, 50, INSERT, 2, dash="2,3"))
    parts += [_tick(cx, cy, deg, DARK) for deg in ((-50, 50) if junctions else ())]
    return "".join(parts)


def _glyph(kind: str, cx: int, cy: int) -> str:
    """One small picture of why a read failed. Minimal by design: a plasmid, plus
    the single mark that makes this filter different from the others."""
    if kind == "length":
        # No plasmid: the filter is on raw read length. Three reads sharing a
        # left edge, so their lengths compare directly.
        x0 = cx - 26
        return "".join([
            line(x0, cy - 9, cx - 15, cy - 9, stroke=DROP, sw=2.8),   # too short
            line(x0, cy, cx + 2, cy, stroke=KEEP, sw=2.8),            # accepted
            line(x0, cy + 9, cx + 26, cy + 9, stroke=DROP, sw=2.8),   # too long
        ])
    if kind == "contaminant":
        # A read that is simply not this molecule: a straight length of DNA
        # sitting beside the plasmid, touching nothing.
        return (_plasmid(cx - 6, cy)
                + line(cx + 24, cy - 15, cx + 24, cy + 15, stroke=DROP, sw=3))
    if kind == "junction":                      # read sits clear of both junctions
        return (_plasmid(cx, cy, junctions=True)
                + arc(cx, cy, GLYPH_R + 7, 125, 225, DROP, 3))
    if kind == "structural":                    # the insert is not there
        return _plasmid(cx, cy, insert=False, gap=True) + arc(
            cx, cy, GLYPH_R + 6, -70, 70, DROP, 3)
    if kind == "typed":
        # The read that survives everything: it spans both junctions, so the whole
        # insert is covered, and both diversified codons can be called.
        return (_plasmid(cx, cy, junctions=True)
                + arc(cx, cy, GLYPH_R + 7, -72, 72, PRIMARY, 3)
                # the codons are called *in the read*, so mark them on it
                + _tick(cx, cy, -18, CODON, 3.5, 2.6, r=GLYPH_R + 7)
                + _tick(cx, cy, 18, CODON, 3.5, 2.6, r=GLYPH_R + 7))
    if kind == "unreadable":
        # The read is there and aligned - it is one of the diversified codons
        # that cannot be called, so the read cannot be typed.
        a = math.radians(20)
        return (_plasmid(cx, cy, junctions=True)
                + arc(cx, cy, GLYPH_R + 7, -72, 72, DROP, 3)
                + _tick(cx, cy, -20, CODON, 3.5, 2.6, r=GLYPH_R + 7)
                + text(cx + (GLYPH_R + 17) * math.cos(a),
                       cy + (GLYPH_R + 17) * math.sin(a) + 5, "?",
                       size=19.5, weight="700", fill=DROP, anchor="middle"))
    return ""


def _icon(kind: str, cx: float, top: float) -> str:
    """A wordless thumbnail of one output, drawn in an ICON_H-tall box."""
    if kind == "histogram":                     # read lengths: a few bars
        base, hs = top + ICON_H, (4, 9, 16, 11, 6)
        return "".join(rect(cx - 14 + i * 6, base - h, 4, h, fill=PRIMARY,
                            stroke="none", rx=1, opacity=0.55)
                       for i, h in enumerate(hs))
    if kind == "tracks":                        # per-read rows, some mismatching
        rows = ((0, None), (1, 6), (2, None), (3, 17))
        out = []
        for r, bad in rows:
            y = top + 3 + r * 4.4
            out.append(line(cx - 15, y, cx + 15, y, stroke=BACKBONE, sw=2.6))
            if bad is not None:
                out.append(line(cx - 15 + bad, y, cx - 9 + bad, y, stroke=DROP,
                                sw=2.6))
        return "".join(out)
    if kind == "waffle":                        # the icon array, in miniature
        fills = (KEEP,) * 11 + (CODON,) * 3 + (DROP,) * 2 + (BACKBONE,) * 2
        return "".join(rect(cx - 15.4 + (i % 6) * 5.2, top + 1.5 + (i // 6) * 5.2,
                            4, 4, fill=f, stroke="none", rx=0.8)
                       for i, f in enumerate(fills))
    if kind == "trace":                         # match % along the insert
        pts = ((0, 3), (7, 4), (13, 12), (19, 4), (24, 5), (30, 14), (36, 4))
        d = " ".join(f"{'M' if i == 0 else 'L'} {cx - 18 + x},{top + y}"
                     for i, (x, y) in enumerate(pts))
        return (line(cx - 18, top + ICON_H, cx + 18, top + ICON_H, stroke=BACKBONE,
                     sw=1)
                + f'<path d="{d}" fill="none" stroke="{PRIMARY}" stroke-width="1.8" '
                  'stroke-linejoin="round"/>')
    if kind == "pie":                           # AA frequencies at one codon
        cy, r = top + ICON_H / 2, 7
        return (circle(cx, cy, r, BACKBONE, sw=4.5)
                + arc(cx, cy, r, -90, 20, CODON, 4.5, cap="butt")
                + arc(cx, cy, r, 20, 95, INSERT, 4.5, cap="butt"))
    if kind == "treemap":                       # variants, boxed by abundance
        x0, y0, h = cx - 16, top + 1, ICON_H - 2
        return "".join([
            rect(x0, y0, 17, h, fill=PRIMARY, stroke="none", opacity=0.55),
            rect(x0 + 18, y0, 14, 9, fill=PRIMARY, stroke="none", opacity=0.34),
            rect(x0 + 18, y0 + 10, 8, 6, fill=PRIMARY, stroke="none", opacity=0.22),
            rect(x0 + 27, y0 + 10, 5, 6, fill=PRIMARY, stroke="none", opacity=0.22),
        ])
    return ""


def body(steps=STEPS) -> str:
    parts = []
    xs = [LEFT + STUB + i * (NODE_W + LINK_W) for i in range(len(steps))]
    hs = [TRUNK_H - i * BRANCH_H for i in range(len(steps))]

    # The flow entering the first node, so pool one sits in a band like the rest.
    parts.append(ribbon(LEFT, TRUNK_TOP, TRUNK_TOP + hs[0],
                        xs[0], TRUNK_TOP, TRUNK_TOP + hs[0], PRIMARY, 0.30))
    bands = [(LEFT, xs[0], TRUNK_TOP + hs[0] / 2, hs[0])]

    for i, s in enumerate(steps):
        x, h = xs[i], hs[i]
        if i + 1 < len(steps):
            nxt, nh = xs[i + 1], hs[i + 1]
            # What continues leaves above the branch and settles onto the next
            # node's full depth - the same height the whole way.
            parts.append(ribbon(x + NODE_W, TRUNK_TOP, TRUNK_TOP + nh,
                                nxt, TRUNK_TOP, TRUNK_TOP + nh, PRIMARY, 0.30))
            bands.append((x + NODE_W, nxt, TRUNK_TOP + nh / 2, nh))
            # The branch takes the bottom slice, from the same depth every time,
            # so all of them are the same shape.
            lx = nxt - 30
            parts += [
                ribbon(x + NODE_W, TRUNK_TOP + nh, TRUNK_TOP + h,
                       lx, LOSS_Y, LOSS_Y + BRANCH_H, DROP, 0.42),
                rect(lx, LOSS_Y, 4, BRANCH_H, fill=DROP, stroke="none", rx=2),
            ]
            # Set level with the cap the branch lands on, not stacked beneath it.
            note = _wrap(steps[i + 1]["lost"], 17)
            tx = min(lx + 10, W - 26 - max(len(p) for p in note) * 6.4)
            top = LOSS_Y + BRANCH_H / 2 + 5.25 - (len(note) - 1) * 7.5
            for k, part in enumerate(note):
                parts.append(text(tx, top + k * 15, part, size=LOSS_PT, fill=DROP))
            parts.append(_glyph(steps[i + 1]["glyph"], lx + GLYPH_R + 4, GLYPH_Y))
        parts.append(rect(x, TRUNK_TOP, NODE_W, h, fill=PRIMARY,
                          stroke="none", rx=3))

    # Pool names ride inside the band that carries that pool.
    for i, s in enumerate(steps):
        bx0, bx1, cy, bh = bands[i]
        lines = _wrap(s["pool"], 17)
        cx = (bx0 + bx1) / 2
        for k, part in enumerate(lines):
            parts.append(text(cx, cy + 6 - (len(lines) - 1 - 2 * k) * 8.25, part,
                              size=POOL_PT, weight="700", fill=DARK, anchor="middle"))
        # Chips top-align across pools and stack downward, so the first output
        # of every pool sits on one line.
        for j, name in enumerate(s["plots"]):
            chip_x = min(max(cx - CHIP_W / 2, LEFT), W - CHIP_W - 26)
            row_y = CHIP_TOP + j * (ROW + CHIP_GAP)
            parts.append(_icon(PLOT_ICONS.get(name, ""), chip_x + CHIP_W / 2, row_y))
            parts.append(card(chip_x, row_y + ICON_H + ICON_GAP, CHIP_W, CHIP_H,
                              _wrap(name, 17), PRIMARY, size=CHIP_PT))
        if s["plots"]:
            lowest = CHIP_TOP + (len(s["plots"]) - 1) * (ROW + CHIP_GAP) + ROW
            parts.append(line(cx, lowest, cx, TRUNK_TOP - 4,
                              stroke=PRIMARY, sw=0.9, dash="2,3"))

    # The read that clears every filter, drawn where the trunk ends.
    parts.append(_glyph("typed", xs[-1] + NODE_W + 10 + GLYPH_R + 7,
                        TRUNK_TOP + hs[-1] / 2))

    return "".join(parts)


def draw(steps=STEPS) -> tuple[str, int, int]:
    return panel("Library FASTQ Read Processing", "", W, H, body(steps),
                 title_size=TITLE_PT), W, H


if __name__ == "__main__":
    svg, w, h = draw()
    print(write(Path(__file__).with_name("workflow.svg"), w, h, svg))
