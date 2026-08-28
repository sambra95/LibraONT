"""Panel B - one small diagram per classification rule.

Each row draws the reference (vector | insert | vector) with the read beneath
it, so the rule can be read off the picture rather than the prose.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from steps import THRESHOLDS
from svgkit import (DARK, DROP, GRID, MONO, MUTED, SURFACE, line, panel, rect,
                    text, write)
from libraont.plots import FATE_COLORS

W, H = 560, 880
LEFT, TOP, ROW = 18, 104, 94
BAR, REF_W = 13, 236            # read/reference bar height, and their width
INS_A, INS_B = 0.36, 0.68       # insert occupies this span of the reference bar

VECTOR = "#9AA3AB"
INSERT = "#7FB3A4"
READ = "#B4BAC1"


def _reference(x, y):
    """The reference: vector, insert, vector, with the two junctions ticked."""
    a, b = x + REF_W * INS_A, x + REF_W * INS_B
    return "".join([
        rect(x, y, REF_W, BAR, fill=VECTOR, stroke="none"),
        rect(a, y, b - a, BAR, fill=INSERT, stroke="none"),
        line(a, y - 4, a, y + BAR + 4, stroke=DARK, sw=1.2),
        line(b, y - 4, b, y + BAR + 4, stroke=DARK, sw=1.2),
        text(x + REF_W * (INS_A + INS_B) / 2, y - 6, "insert", size=8.5,
             fill=DARK, anchor="middle"),
    ])


def _read(x, y, segments, marks=()):
    """The read as one or more aligned segments, plus any event markers."""
    out = []
    for f0, f1, colour in segments:
        out.append(rect(x + REF_W * f0, y, REF_W * (f1 - f0), BAR,
                        fill=colour, stroke="none"))
    for f, colour, label in marks:
        cx = x + REF_W * f
        out.append(rect(cx - 3, y - 3, 6, BAR + 6, fill=colour, stroke="none"))
        if label:
            out.append(text(cx, y + BAR + 13, label, size=8, fill=colour,
                            anchor="middle", family=MONO))
    return "".join(out)


def _cases(t=THRESHOLDS):
    """(verdict, colour, read segments, markers, note) per rule, in funnel order."""
    ins, dele = t["insertion_bp"], t["deletion_bp"]
    return [
        ("Contaminant - discarded", DROP,
         [], [], "no alignment to the plasmid at all"),
        ("Uninformative - excluded", MUTED,
         [(0.72, 1.0, READ)], [], "aligns clear of both junctions: says nothing about the insert"),
        ("Does not contain insert region", FATE_COLORS["Does not contain insert region"],
         [(0.06, INS_A, READ), (INS_B, 0.96, READ)],
         [(INS_A, DROP, ""), (INS_B, DROP, "")],
         "crosses a junction, insert absent: religated vector"),
        ("Insertion", FATE_COLORS["Insertion"],
         [(0.06, 0.96, READ)], [(0.52, FATE_COLORS["Insertion"], f"≥{ins} bp")],
         "extra DNA the reference does not have"),
        ("Deletion", FATE_COLORS["Deletion"],
         [(0.06, 0.46, READ), (0.58, 0.96, READ)],
         [(0.52, FATE_COLORS["Deletion"], f"≥{dele} bp")],
         "insert bases missing"),
        ("Wild type", FATE_COLORS["Wild type"],
         [(0.06, 0.96, READ)],
         [(0.44, FATE_COLORS["Wild type"], ""), (0.52, FATE_COLORS["Wild type"], ""),
          (0.60, FATE_COLORS["Wild type"], "")],
         "every diversified codon read, none changed"),
        ("Variant", FATE_COLORS["Variant"],
         [(0.06, 0.96, READ)],
         [(0.44, FATE_COLORS["Wild type"], ""), (0.52, FATE_COLORS["Variant"], "≠"),
          (0.60, FATE_COLORS["Wild type"], "")],
         "at least one diversified codon differs: a library member"),
        ("Ambiguous diversity", FATE_COLORS["Ambiguous diversity"],
         [(0.06, 0.56, READ)],
         [(0.44, FATE_COLORS["Wild type"], ""), (0.52, MUTED, "?")],
         "a diversified codon unreadable: could be either"),
    ]


def body(t=THRESHOLDS) -> str:
    parts = []
    for i, (verdict, colour, segs, marks, note) in enumerate(_cases(t)):
        y = TOP + i * ROW
        parts += [
            rect(LEFT - 6, y - 32, W - 24, ROW - 10, fill="#FFFFFF", stroke=GRID, rx=7),
            rect(LEFT - 6, y - 32, 4, ROW - 10, fill=colour, stroke="none", rx=2),
            text(LEFT + 6, y - 17, verdict, size=11, weight="700", fill=colour),
            _reference(LEFT + 6, y),
            _read(LEFT + 6, y + BAR + 6, segs, marks),
            text(LEFT + REF_W + 24, y + 5, "reference", size=8, fill=MUTED),
            text(LEFT + REF_W + 24, y + BAR + 12, "read", size=8, fill=MUTED),
        ]
        for k, part in enumerate(_wrap(note, 34)):
            parts.append(text(LEFT + REF_W + 24, y + BAR + 26 + k * 10, part,
                              size=8, fill=MUTED))
    return "".join(parts)


def _wrap(s: str, width: int) -> list[str]:
    out, cur = [], ""
    for word in s.split():
        if len(cur) + len(word) + 1 > width:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    return out + ([cur] if cur else [])


def draw(t=THRESHOLDS) -> tuple[str, int, int]:
    return panel("What each rule looks like",
                 "the read is compared with the reference it was aligned to",
                 W, H, body(t)), W, H


if __name__ == "__main__":
    svg, w, h = draw()
    print(write(Path(__file__).with_name("criteria.svg"), w, h, svg,
                title="LibraONT classification rules"))
