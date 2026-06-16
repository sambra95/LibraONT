"""Central colour scheme and Plotly theming.

Importing this module registers (and activates) the ``libraont`` Plotly template
so every figure shares one consistent, calm palette. The same colours are mirrored
in ``.streamlit/config.toml`` for the surrounding app chrome.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from .constants import AA_ORDER

# --- Core palette -----------------------------------------------------------
PALETTE: dict[str, str] = {
    "bg": "#FFFFFF",
    "surface": "#F4F6F8",
    "primary": "#1F6F8B",       # deep teal-blue (lines, primary series)
    "primary_dark": "#16505F",  # titles
    "secondary": "#99AEBB",     # axes / muted accents
    "accent": "#E8743B",        # warm coral (gaps, highlights)
    "accent_soft": "rgba(232, 116, 59, 0.22)",
    "text": "#26323A",
    "grid": "#E6EBEF",
    "muted": "#6B7B86",
}

# Harmonious qualitative palette for treemaps / general categories.
CATEGORICAL: list[str] = [
    "#1F6F8B", "#E8743B", "#5BA199", "#C44E5A", "#E0B252",
    "#7D6FB0", "#3C9DBA", "#A0563B", "#6FB07D", "#B07D9B",
]

# Gap vs reference-match line colours.
GAP_COLOR: str = PALETTE["accent"]
MATCH_COLOR: str = PALETTE["primary"]

# Amino-acid colours by biochemical group, consistent across every pie.
AA_GROUPS: dict[str, str] = {
    "A": "Hydrophobic", "V": "Hydrophobic", "I": "Hydrophobic", "L": "Hydrophobic",
    "M": "Hydrophobic", "F": "Hydrophobic", "W": "Hydrophobic", "Y": "Hydrophobic",
    "S": "Hydrophilic", "T": "Hydrophilic", "N": "Hydrophilic", "Q": "Hydrophilic",
    "D": "Acidic", "E": "Acidic",
    "K": "Basic", "R": "Basic", "H": "Basic",
    "C": "Special", "G": "Special", "P": "Special",
}
AA_GROUP_COLORS: dict[str, str] = {
    "Hydrophobic": "#1F6F8B",
    "Hydrophilic": "#5BA199",
    "Acidic": "#C44E5A",
    "Basic": "#7D6FB0",
    "Special": "#E0B252",
}
AA_COLORS: dict[str, str] = {
    aa: AA_GROUP_COLORS[AA_GROUPS[aa]]
    for aa in AA_ORDER
    if aa in AA_GROUPS
}
AA_COLORS["*"] = "#5A5A5A"

TEMPLATE_NAME = "libraont"


def _build_template() -> go.layout.Template:
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family="Inter, Segoe UI, Helvetica, sans-serif",
                  color=PALETTE["text"], size=13),
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["bg"],
        colorway=CATEGORICAL,
        title=dict(font=dict(size=17, color=PALETTE["primary_dark"]), x=0.02, xanchor="left"),
        xaxis=dict(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"],
                   linecolor=PALETTE["secondary"], ticks="outside",
                   tickcolor=PALETTE["secondary"]),
        yaxis=dict(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"],
                   linecolor=PALETTE["secondary"], ticks="outside",
                   tickcolor=PALETTE["secondary"]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=60, r=24, t=60, b=48),
    )
    return template


def activate() -> None:
    """Register the template and make it the Plotly default (idempotent)."""
    pio.templates[TEMPLATE_NAME] = _build_template()
    pio.templates.default = TEMPLATE_NAME


def aa_color_sequence(labels: list[str]) -> list[str]:
    """Colours for amino-acid labels; 'Other'/'(no data)' get a neutral grey."""
    return [AA_COLORS.get(lbl, PALETTE["muted"]) for lbl in labels]


# Activate on import so any figure built afterwards is themed by default.
activate()

__all__ = [
    "PALETTE", "CATEGORICAL", "GAP_COLOR", "MATCH_COLOR",
    "AA_GROUPS", "AA_GROUP_COLORS", "AA_COLORS", "TEMPLATE_NAME", "activate",
    "aa_color_sequence", "AA_ORDER",
]
