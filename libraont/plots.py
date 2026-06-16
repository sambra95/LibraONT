"""Themed Plotly figures. Pure builders: each returns a ``go.Figure`` (no I/O).

Importing :mod:`libraont.theme` activates the shared template, so colours and
fonts here stay consistent with the rest of the app.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import theme
from .alignment import CoverageResult
from .analysis import haplotype_gini, reference_match_percent

_T = theme.TEMPLATE_NAME

_AA_NAMES: dict[str, str] = {
    "A": "Alanine",
    "R": "Arginine",
    "N": "Asparagine",
    "D": "Aspartic acid",
    "C": "Cysteine",
    "Q": "Glutamine",
    "E": "Glutamic acid",
    "G": "Glycine",
    "H": "Histidine",
    "I": "Isoleucine",
    "L": "Leucine",
    "K": "Lysine",
    "M": "Methionine",
    "F": "Phenylalanine",
    "P": "Proline",
    "S": "Serine",
    "T": "Threonine",
    "W": "Tryptophan",
    "Y": "Tyrosine",
    "V": "Valine",
    "*": "Stop codon",
}


def read_length_figure(length_counts: Counter) -> go.Figure:
    """Bar chart of read-length frequencies."""
    if not length_counts:
        return _empty("No reads found")
    x, y = zip(*sorted(length_counts.items()))
    fig = go.Figure(go.Bar(
        x=x, y=y, marker_color=theme.PALETTE["primary"],
        hovertemplate="Length %{x} bp<br>%{y} reads<extra></extra>",
    ))
    fig.update_layout(
        template=_T, title="Read length distribution",
        xaxis_title="Read length (bp)", yaxis_title="Count", bargap=0.1,
        meta={"description": "Shows how many reads fall at each read length. Use it to spot "
                             "the dominant library size and any unexpected short or long reads."},
    )
    return fig


def coverage_figure(cov: CoverageResult) -> go.Figure:
    """Per-base coverage as a fraction of max depth, with optional insert highlight."""
    cards = [
        {"label": "Mapped reads", "value": f"{cov.mapped_reads:,}",
         "accent": theme.PALETTE["primary"]},
    ]
    fig = go.Figure(go.Scatter(
        x=cov.positions, y=cov.fractions, mode="lines",
        line=dict(color=theme.PALETTE["primary"], width=1.6), name="Fraction",
        fill="tozeroy", fillcolor="rgba(31, 111, 139, 0.12)",
        hovertemplate="Pos %{x}<br>Frac %{y:.3f}<extra></extra>",
    ))
    if cov.region:
        fig.add_vrect(x0=cov.region["start"], x1=cov.region["end"],
                      fillcolor=theme.PALETTE["accent_soft"], line_width=0, layer="below")
        cards.append({"label": "Insert", "value": f"{cov.region['start']}-{cov.region['end']}",
                      "accent": theme.PALETTE["accent"]})
    fig.update_layout(
        template=_T, title="Coverage fraction", xaxis_title="Position (bp)",
        yaxis_title="Fraction of overlapping reads",
        yaxis=dict(range=[0, 1]), height=360,
        meta={
            "description": "Shows normalized read coverage across the plasmid. Regions near 1 "
                           "have the strongest support; drops indicate that a fraction of the "
                           "variants do not align to the reference in the region.",
            "metric_cards": cards,
        },
    )
    return fig


def gap_match_figure(df_counts: pd.DataFrame, ref_seq: str, *, gap_char: str = "-",
                     shade_codons=None, frame_offset: int = 0,
                     min_identity: float | None = None,
                     auto_match_threshold: float | None = None) -> go.Figure:
    """Per-position gap % (counts['-'] / row total) and reference-match %
    (gaps excluded from the denominator), optionally shading full codons and
    showing the auto-detect reference-match cutoff."""
    if gap_char not in df_counts.columns:
        raise ValueError(f"Gap column '{gap_char}' not in df_counts: {list(df_counts.columns)}")
    if frame_offset not in (0, 1, 2):
        raise ValueError("frame_offset must be 0, 1, or 2")

    L = len(df_counts)
    totals = df_counts.sum(axis=1).astype(float).replace(0, np.nan)
    gap_perc = (100.0 * df_counts[gap_char].astype(float) / totals).fillna(0.0)
    match_perc = reference_match_percent(df_counts, ref_seq, gap_char)

    x = df_counts.index.astype(int)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=gap_perc.values, mode="lines+markers",
                             name="Gap %", line=dict(color=theme.GAP_COLOR),
                             marker=dict(size=4),
                             hovertemplate="Pos %{x}<br>Gap %{y:.2f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=match_perc, mode="lines+markers",
                             name="Reference match %", line=dict(color=theme.MATCH_COLOR),
                             marker=dict(size=4),
                             hovertemplate="Pos %{x}<br>Ref match %{y:.2f}%<extra></extra>"))

    if auto_match_threshold is not None:
        fig.add_trace(go.Scatter(
            x=[x.min(), x.max()],
            y=[auto_match_threshold, auto_match_threshold],
            mode="lines",
            name="Detection threshold",
            line=dict(color=theme.PALETTE["muted"], width=1.5, dash="3px,3px"),
            hovertemplate="Detection threshold %{y:g}%<extra></extra>",
        ))

    if shade_codons:
        for cpos in shade_codons:
            nt_start = (cpos - 1) * 3 + 1 + frame_offset
            nt_end = nt_start + 2
            if nt_end < 1 or nt_start > L:
                continue
            fig.add_vrect(x0=max(1, nt_start) - 0.5, x1=min(L, nt_end) + 0.5,
                          fillcolor=theme.PALETTE["accent_soft"], line_width=0, layer="below")

    n_seqs = int((df_counts.sum(axis=1) - df_counts[gap_char]).max())
    pct = f"{min_identity * 100:g}%" if min_identity is not None else "-"
    cards = [
        {"label": "Aligned reads", "value": f"{n_seqs:,}",
         "accent": theme.PALETTE["primary"]},
        {"label": "Min identity", "value": pct,
         "accent": theme.PALETTE["accent"]},
    ]
    if auto_match_threshold is not None:
        cards.append({"label": "Detection threshold",
                      "value": f"{auto_match_threshold:g}%",
                      "accent": theme.PALETTE["muted"]})
    fig.update_layout(
        template=_T, hovermode="x unified",
        title="Alignment to reference insert",
        xaxis_title="Position in reference (1-based, nucleotides)",
        yaxis_title="Percentage (%)",
        margin=dict(t=90),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        meta={
            "description": "Compares gap frequency with reference-match frequency at each "
                           "insert base. Low reference-match positions indicate likely "
                           "variable or mutated sites.",
            "metric_cards": cards,
        },
    )
    return fig


def aa_pies_figure(df_aa_counts: pd.DataFrame, positions, *, min_frac: float = 0.01,
                   pie_subtitles=None,
                   title: str = "Amino Acid distribution at selected codon positions",
                   hole: float = 0.5, height_per_row: int = 300,
                   width_per_col: int = 300) -> go.Figure | None:
    """Grid of donut charts showing the AA distribution at each codon position.
    Slices below ``min_frac`` are folded into 'Other'."""
    pos_ok = [p for p in positions if p in df_aa_counts.index]
    if not pos_ok:
        return None
    if pie_subtitles is None:
        pie_subtitles = pos_ok

    n = len(pos_ok)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    height = rows * height_per_row
    # Plotly expresses subplot spacing as a fraction of total figure height.
    # Keep the actual row gap near-constant so large auto-detected sets do not
    # violate Plotly's max spacing limit or waste vertical space.
    vertical_spacing = 0.0 if rows == 1 else min(0.08, 28 / height)
    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{"type": "domain"} for _ in range(cols)] for _ in range(rows)],
        subplot_titles=[f"Codon {p}" for p in pie_subtitles],
        horizontal_spacing=0.035, vertical_spacing=vertical_spacing,
    )

    k = 0
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            if k >= n:
                # Invisible pie keeps empty cells the same size as filled ones.
                fig.add_trace(go.Pie(labels=[""], values=[1], hole=hole, opacity=0,
                                     showlegend=False, hoverinfo="skip"), row=r, col=c)
                continue
            pos = pos_ok[k]; k += 1
            counts = df_aa_counts.loc[pos].astype(float)
            total = counts.sum()
            if total > 0:
                fracs = counts / total
                keep = fracs[fracs >= min_frac]
                small = fracs[fracs < min_frac]
                labels = list(keep.index)
                values = list((keep * total).to_numpy())
                if small.sum() > 0:
                    labels.append("Other")
                    values.append(float((small * total).sum()))
            else:
                labels, values = ["(no data)"], [1]
            fig.add_trace(go.Pie(
                labels=labels, values=values, hole=hole, sort=False,
                customdata=[_AA_NAMES.get(label, label) for label in labels],
                marker=dict(
                    colors=theme.aa_color_sequence(labels),
                    line=dict(color="#000000", width=1),
                ),
                textinfo="label+percent", textposition="inside",
                hovertemplate="%{customdata}<br>Count %{value:.0f}<br>%{percent:.2%}<extra></extra>",
                showlegend=False,
            ), row=r, col=c)

    # Centred n=… annotation inside each real donut.
    for trace in fig.data:
        if isinstance(trace, go.Pie) and trace.opacity != 0 and trace.labels != ("",):
            xd, yd = trace.domain["x"], trace.domain["y"]
            fig.add_annotation(x=(xd[0] + xd[1]) / 2, y=(yd[0] + yd[1]) / 2,
                               xref="paper", yref="paper", showarrow=False,
                               xanchor="center", yanchor="middle", align="center",
                               text=f"n={int(sum(trace.values))}",
                               font=dict(size=13, color=theme.PALETTE["muted"]))

    fig.update_layout(
        template=_T, title=title, showlegend=False,
        height=height, width=cols * width_per_col,
        uniformtext_minsize=10, uniformtext_mode="hide",
        meta={"description": "Each donut summarizes amino-acid frequencies at one selected "
                             "codon. Larger slices are more common variants; the center count "
                             "is the number of reads contributing to that codon."},
    )
    return fig


def haplotype_treemap_figure(hap_df: pd.DataFrame, *, title: str = "Treemap of unique variants",
                             top_n: int | None = None) -> go.Figure:
    """Treemap of haplotypes sized by count; if ``top_n`` is set, fold the rest
    into 'Other'. The subtitle reports the Gini coefficient of the distribution."""
    if hap_df.empty:
        return _empty("No haplotypes to display", title)

    df = hap_df[["combo_label", "count"]].copy()
    df["is_reference"] = hap_df.get("is_reference", False)
    df["aa_hamming_distance"] = hap_df.get("aa_hamming_distance", pd.NA)
    df = df.sort_values("count", ascending=False).reset_index(drop=True)
    total = int(df["count"].sum())
    gini = haplotype_gini(hap_df)

    if top_n is not None and len(df) > top_n:
        other = pd.DataFrame([{"combo_label": "Other",
                               "count": int(df.iloc[top_n:]["count"].sum()),
                               "is_reference": False,
                               "aa_hamming_distance": pd.NA}])
        df_plot = pd.concat([df.iloc[:top_n], other], ignore_index=True)
    else:
        df_plot = df

    colors = [theme.CATEGORICAL[i % len(theme.CATEGORICAL)] for i in range(len(df_plot))]
    colors = [theme.PALETTE["muted"] if lbl == "Other" else col
              for lbl, col in zip(df_plot["combo_label"], colors)]
    is_reference = df_plot["is_reference"].fillna(False).astype(bool).to_list()
    line_colors = ["#000000" if flag else "#FFFFFF" for flag in is_reference]
    line_widths = [4 if flag else 1 for flag in is_reference]
    status = ["Reference match" if flag else "Variant" for flag in is_reference]
    distances = [
        "n/a" if pd.isna(distance) else str(int(distance))
        for distance in df_plot["aa_hamming_distance"]
    ]

    fig = go.Figure(go.Treemap(
        labels=df_plot["combo_label"], parents=[""] * len(df_plot),
        values=df_plot["count"], branchvalues="total",
        customdata=list(zip(status, distances)),
        marker=dict(colors=colors, line=dict(color=line_colors, width=line_widths)),
        textinfo="label+percent entry",
        hovertemplate=("%{label}<br>Hamming distance to WT: %{customdata[1]}"
                       "<br>Reads %{value}"
                       "<br>%{percentRoot:.1%}<extra></extra>"),
    ))
    fig.update_layout(
        template=_T, height=500, title=title,
        meta={
            "description": "Shows unique amino-acid haplotypes across the selected codons. "
                           "The reference-matching combination is outlined in black. "
                           "Larger rectangles represent more abundant variant combinations. "
                           "Only identified codons are considered when defining variants and "
                           "calculating amino-acid Hamming distances. "
                           "Gini summarizes unevenness: 0 means variants are evenly represented, "
                           "while values closer to 1 mean a few variants dominate.",
            "metric_cards": [
                {"label": "Unique sequences", "value": f"{len(df):,}",
                 "accent": theme.PALETTE["primary"]},
                {"label": "Reads", "value": f"{total:,}",
                 "accent": theme.PALETTE["accent"]},
                {"label": "Gini", "value": f"{gini:.3f}",
                 "accent": theme.PALETTE["secondary"]},
            ],
        },
    )
    return fig


def _empty(message: str, title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper",
                       text=message, showarrow=False, font=dict(color=theme.PALETTE["muted"]))
    fig.update_layout(template=_T, title=title, height=360)
    return fig
