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
from .constants import GENETIC_CODE

_T = theme.TEMPLATE_NAME

# Theme red, used to flag length-excluded reads and to mark the insert.
_RED = "#C44E5A"

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


def read_length_figure(length_counts: Counter, min_read_len: int | None = None,
                       max_read_len: int | None = None,
                       plasmid_len: int | None = None) -> go.Figure:
    """Bar chart of read-length frequencies binned every 10 bp. This plot always
    covers *all* reads; dashed lines mark the min/max read-length cutoffs that
    filter reads out of every other plot, plus the plasmid length when one was
    given - whole-plasmid reads pile up there."""
    if not length_counts:
        return _empty("No reads found")
    bin_counts: Counter = Counter()
    for length, count in length_counts.items():
        bin_counts[(int(length) // 10) * 10] += count
    x, y = zip(*sorted(bin_counts.items()))
    labels = [f"{start}-{start + 9} bp" for start in x]
    # Blue when the bin lies within the kept length window, red when it falls
    # entirely outside it (those reads are dropped from every downstream plot).
    bar_colors = [
        _RED if ((min_read_len is not None and start + 9 < min_read_len)
                 or (max_read_len is not None and start > max_read_len))
        else theme.PALETTE["primary"]
        for start in x
    ]
    fig = go.Figure(go.Bar(
        x=x, y=y, customdata=labels, width=9,
        marker_color=bar_colors,
        hovertemplate="Length %{customdata}<br>%{y} reads<extra></extra>",
    ))
    for value, label in ((min_read_len, "min"), (max_read_len, "max")):
        if value is not None:
            fig.add_vline(x=value, line=dict(color=theme.PALETTE["muted"], width=1, dash="dash"),
                          annotation_text=label, annotation_position="top",
                          annotation=dict(font=dict(size=11, color=theme.PALETTE["muted"])))
    if plasmid_len:
        # Labelled in the margin above the plot: the max cutoff usually sits close
        # by, and "top" would put the two annotations on top of each other.
        accent = theme.PALETTE["accent"]
        fig.add_vline(x=plasmid_len, line=dict(color=accent, width=1.4, dash="dash"))
        fig.add_annotation(xref="x", yref="paper", x=plasmid_len, y=1.02, yanchor="bottom",
                           # Kept inside the plot when the line is its right-hand edge.
                           xanchor="right" if plasmid_len >= x[-1] else "center",
                           text=f"plasmid ({plasmid_len:,} bp)", showarrow=False,
                           font=dict(size=11, color=accent))
    description = ("Shows how many reads fall into each 10 bp length bin (all reads). "
                   "Dashed lines mark the min/max read-length cutoffs; reads outside "
                   "them are excluded from every other plot and the analysis.")
    if plasmid_len:
        description += (" The coral line marks the plasmid length: whole-plasmid reads "
                        "pile up around it, amplicon reads well below it.")
    fig.update_layout(
        template=_T, title="Read length distribution",
        xaxis_title="Read length bin (bp)", yaxis_title="Count", bargap=0.05,
        meta={"description": description},
    )
    return fig


def _insert_marker(fig: go.Figure, start: float, end: float) -> None:
    """Mark the insert with a red bar (labelled "insert") in the margin just
    above the plot, spanning ``[start, end]`` on the x-axis."""
    fig.add_shape(type="rect", xref="x", yref="paper",
                  x0=start, x1=end, y0=1.02, y1=1.05,
                  fillcolor=_RED, line_width=0, layer="above")
    fig.add_annotation(xref="x", yref="paper", x=(start + end) / 2, y=1.06,
                       text="insert", showarrow=False, yanchor="bottom",
                       font=dict(size=11, color=_RED))


def coverage_figure(cov: CoverageResult) -> go.Figure:
    """Per-base coverage as a fraction of max depth, with optional insert highlight."""
    cards = [{"label": "Reads mapped to plasmid", "value": f"{cov.mapped_reads:,}",
              "sub": "aligned to whole plasmid (minimap2)"}]
    fig = go.Figure(go.Scatter(
        x=cov.positions, y=cov.fractions, mode="lines",
        line=dict(color=theme.PALETTE["primary"], width=1.6), name="Fraction",
        fill="tozeroy", fillcolor="rgba(31, 111, 139, 0.12)",
        hovertemplate="Pos %{x}<br>Frac %{y:.3f}<extra></extra>",
    ))
    if cov.region:
        _insert_marker(fig, cov.region["start"], cov.region["end"])
        cards.append({"label": "Insert",
                      "value": f"{cov.region['start']}-{cov.region['end']}"})
    fig.update_layout(
        template=_T, title="Coverage fraction", xaxis_title="Position (bp)",
        yaxis_title="Fraction of overlapping reads",
        yaxis=dict(range=[0, 1]), height=360, margin=dict(t=72),
        meta={
            "description": "Shows normalized read coverage across the plasmid. Regions near 1 "
                           "have the strongest support; drops indicate that a fraction of the "
                           "variants do not align to the reference in the region.",
            "metric_cards": cards,
        },
    )
    return fig


# Sequential colour scale for read-to-reference identity: low = warm red,
# mid = amber, high = teal. Always applied over a fixed 0-100% range.
_IDENTITY_SCALE = [[0.0, "#C44E5A"], [0.5, "#E0B252"], [1.0, theme.PALETTE["primary"]]]


def read_alignment_figure(cov: CoverageResult) -> go.Figure:
    """Map of where each read aligns on the plasmid: one horizontal bar per read
    (its own row, ordered by start position), coloured by the read's percent
    identity to its aligned section, with the insert region shaded. Reads are
    *not* stacked onto shared rows, so every read stays individually visible."""
    starts, ends, idents = cov.read_starts, cov.read_ends, cov.read_identities
    n_total = int(starts.shape[0])
    if n_total == 0:
        return _empty("No reads aligned to the plasmid")

    # Order by start (then end) so the pileup reads as a staircase. Every read
    # gets its own row (no subsampling); for deep libraries the rows compress to
    # sub-pixel bars, reading as a dense pileup.
    order = np.lexsort((ends, starts))
    starts, ends, idents = starts[order], ends[order], idents[order]
    n_shown = n_total
    rows = np.arange(n_shown)

    # Compact: ~1 px per read row, clamped so the panel never dominates.
    _T_MARGIN, _B_MARGIN = 60, 48
    height = int(min(700, max(160, n_shown + _T_MARGIN + _B_MARGIN + 20)))

    ident_pct = idents * 100.0
    finite = np.isfinite(ident_pct)
    # Fixed 0-100% colour range; reads with no identity (missing NM) fall back
    # to the low end so they still render.
    color = np.where(finite, ident_pct, 0.0)

    fig = go.Figure(go.Bar(
        y=rows, x=(ends - starts + 1), base=starts, orientation="h", width=1.0,
        marker=dict(color=color, colorscale=_IDENTITY_SCALE, cmin=0.0, cmax=100.0,
                    line_width=0,
                    colorbar=dict(title=dict(text="Identity to<br>reference (%)",
                                             side="right"),
                                  thickness=12, len=0.9, ticksuffix="%")),
        customdata=np.column_stack([starts, ends, ident_pct]),
        hovertemplate="%{customdata[0]}–%{customdata[1]} bp<br>"
                      "Identity %{customdata[2]:.1f}%<extra></extra>",
    ))

    cards = [{"label": "Reads on map", "value": f"{n_shown:,}", "sub": "one bar per read"}]
    if finite.any():
        cards.append({"label": "Median identity",
                      "value": f"{np.median(ident_pct[finite]):.1f}%"})
    if cov.region:
        _insert_marker(fig, cov.region["start"], cov.region["end"])
        cards.append({"label": "Insert",
                      "value": f"{cov.region['start']}-{cov.region['end']}"})

    x_max = cov.contig_length or int(ends.max())
    fig.update_layout(
        template=_T, title="Read alignment map", height=height, showlegend=False,
        bargap=0,
        xaxis=dict(title="Plasmid position (bp)", range=[0, x_max], zeroline=False,
                   showline=True, showticklabels=True, ticks="outside"),
        yaxis=dict(range=[-1, n_shown], showgrid=False, zeroline=False,
                   showticklabels=False, title=""),
        margin=dict(l=48, r=88, t=_T_MARGIN + 12, b=_B_MARGIN),
        meta={
            "description": "Each horizontal bar is a single read, drawn on its own row, "
                           "positioned where it aligns to the plasmid and coloured by its "
                           "percent identity to that aligned section. Rows are ordered by start "
                           "position; a red bar above the plot marks the insert.",
            "metric_cards": cards,
        },
    )
    return fig


def _aa_track(ref_seq: str, length: int, frame_offset: int) -> go.Bar | None:
    """Reference amino acid of each codon as a coloured band for ``yaxis2``,
    using the shared group colours (``theme.AA_COLORS``). Letters are drawn
    inside each block whenever they fit."""
    ref_up = (ref_seq or "").upper()
    centres, letters, colors = [], [], []
    for start in range(frame_offset, length - 2, 3):
        aa = GENETIC_CODE.get(ref_up[start:start + 3]) or "?"
        centres.append(start + 2)  # 1-based centre of the three nucleotides
        letters.append(aa)
        colors.append(theme.AA_COLORS.get(aa, theme.PALETTE["muted"]))
    if not centres:
        return None
    return go.Bar(x=centres, y=[1] * len(centres), width=3, yaxis="y2",
                  marker=dict(color=colors, line=dict(width=0)),
                  text=letters, textposition="inside", insidetextanchor="middle",
                  textangle=0,  # never let Plotly rotate letters in narrow blocks
                  textfont=dict(family="monospace", size=11,
                                color=[theme.contrast_text(c) for c in colors]),
                  showlegend=False, hoverinfo="skip")


def gap_match_figure(df_counts: pd.DataFrame, ref_seq: str, *, gap_char: str = "-",
                     shade_codons=None, frame_offset: int = 0,
                     min_identity: float | None = None,
                     auto_match_threshold: float | None = None,
                     aa_counts: pd.DataFrame | None = None,
                     reads_passing: int | None = None,
                     reads_total: int | None = None) -> go.Figure:
    """Per-position gap % (counts['-'] / row total) and reference-match %
    (gaps excluded from the denominator), optionally shading full codons and
    showing the auto-detect reference-match cutoff.

    A band above the plot shows the reference amino acid of each codon, coloured
    by biochemical group like every other amino-acid view in the app.

    When ``aa_counts`` (per-codon amino-acid counts, 1-based codon index) is
    given, the hover for each position also lists the 20 amino acids in two
    columns with their relative frequency at that codon."""
    if gap_char not in df_counts.columns:
        raise ValueError(f"Gap column '{gap_char}' not in df_counts: {list(df_counts.columns)}")
    if frame_offset not in (0, 1, 2):
        raise ValueError("frame_offset must be 0, 1, or 2")

    L = len(df_counts)
    totals = df_counts.sum(axis=1).astype(float).replace(0, np.nan)
    gap_perc = (100.0 * df_counts[gap_char].astype(float) / totals).fillna(0.0)
    match_perc = reference_match_percent(df_counts, ref_seq, gap_char)

    x = df_counts.index.astype(int)
    codon_heads, aa_blocks = _codon_hover_labels(ref_seq, L, frame_offset, aa_counts)
    fig = go.Figure()
    # Invisible trace drawn first carries the codon number so, in the unified
    # hover, it renders at the very top (just under the position header) with no
    # colour swatch. A trailing <br> on each element leaves a blank separator line.
    fig.add_trace(go.Scatter(x=x, y=gap_perc.values, mode="lines",
                             line=dict(color="rgba(0,0,0,0)"), showlegend=False,
                             customdata=codon_heads,
                             hovertemplate="%{customdata}<br><extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=gap_perc.values, mode="lines+markers",
                             name="Gap %", line=dict(color=theme.GAP_COLOR),
                             marker=dict(size=4),
                             hovertemplate="<b>Gap:</b> %{y:.2f}%<br><extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=match_perc, mode="lines+markers",
                             name="Reference match %", line=dict(color=theme.MATCH_COLOR),
                             marker=dict(size=4),
                             hovertemplate="<b>Reference match:</b> %{y:.2f}%<br><extra></extra>"))
    if aa_blocks is not None:
        # AA table on an invisible trace drawn last so it renders as its own
        # block underneath both coloured line entries (no swatch beside it).
        fig.add_trace(go.Scatter(x=x, y=match_perc, mode="lines",
                                 line=dict(color="rgba(0,0,0,0)"), showlegend=False,
                                 customdata=aa_blocks,
                                 hovertemplate="%{customdata}<extra></extra>"))

    aa_track = _aa_track(ref_seq, L, frame_offset)
    if aa_track is not None:
        fig.add_trace(aa_track)

    if auto_match_threshold is not None:
        fig.add_trace(go.Scatter(
            x=[x.min(), x.max()],
            y=[auto_match_threshold, auto_match_threshold],
            mode="lines",
            name="Detection threshold",
            line=dict(color=theme.PALETTE["muted"], width=1.5, dash="3px,3px"),
            hoverinfo="skip",
        ))

    if shade_codons:
        for cpos in shade_codons:
            nt_start = (cpos - 1) * 3 + 1 + frame_offset
            nt_end = nt_start + 2
            if nt_end < 1 or nt_start > L:
                continue
            fig.add_vrect(x0=max(1, nt_start) - 0.5, x1=min(L, nt_end) + 0.5,
                          fillcolor=theme.PALETTE["accent_soft"], line_width=0, layer="below")

    pct = f"{min_identity * 100:g}%" if min_identity is not None else "-"
    # Reads passing / failing the insert filter (fall back to the aligned-read
    # count from the MSA when the report-level totals are not supplied).
    if reads_passing is not None and reads_total is not None:
        failing = max(reads_total - reads_passing, 0)
        pass_sub = f"{reads_passing / reads_total * 100:.1f}% of {reads_total:,}" if reads_total else ""
        fail_sub = f"{failing / reads_total * 100:.1f}% of {reads_total:,}" if reads_total else ""
        read_cards = [
            {"label": "Passed insert filter", "value": f"{reads_passing:,}", "sub": pass_sub},
            {"label": "Failed insert filter", "value": f"{failing:,}", "sub": fail_sub},
        ]
    else:
        n_seqs = int((df_counts.sum(axis=1) - df_counts[gap_char]).max())
        read_cards = [{"label": "Aligned reads", "value": f"{n_seqs:,}"}]
    cards = [
        *read_cards,
        {"label": "Min identity", "value": pct},
        {"label": "Detected positions", "value": f"{len(shade_codons or []):,}"},
    ]
    if auto_match_threshold is not None:
        cards.append({"label": "Detection threshold",
                      "value": f"{auto_match_threshold:g}%"})
    fig.update_layout(
        template=_T, hovermode="x unified",
        title="Alignment to reference insert",
        xaxis_title="Position in reference (1-based, nucleotides)",
        yaxis=dict(title="Percentage (%)", domain=[0, 0.90] if aa_track else [0, 1]),
        yaxis2=dict(domain=[0.94, 1.0], range=[0, 1], fixedrange=True,
                    showgrid=False, zeroline=False, showticklabels=False, ticks=""),
        # Letters shrink to fit their codon block and drop out only once they
        # would be unreadable; zooming in brings them back.
        uniformtext=dict(minsize=5, mode="hide"),
        margin=dict(t=90),
        hoverlabel=dict(font=dict(family="monospace", size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        meta={
            "description": "Compares gap frequency with reference-match frequency at each "
                           "insert base. Low reference-match positions indicate likely "
                           "variable or mutated sites. The band above the plot gives the "
                           "reference amino acid at each codon, coloured by biochemical group.",
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


def _rare_aa_by_position(aa_counts: pd.DataFrame | None, positions, min_frac: float) -> dict:
    """Per codon position, the set of amino acids whose frequency is below
    ``min_frac`` - i.e. those folded into the 'Other' slice in the AA pies."""
    rare: dict[int, set[str]] = {}
    if aa_counts is None or not positions or min_frac <= 0:
        return rare
    for pos in positions:
        if pos not in aa_counts.index:
            continue
        row = aa_counts.loc[pos].astype(float)
        total = row.sum()
        if total > 0:
            rare[pos] = {aa for aa, count in row.items() if count / total < min_frac}
    return rare


def haplotype_treemap_figure(hap_df: pd.DataFrame, *, title: str = "Treemap of unique variants",
                             top_n: int | None = None, aa_counts: pd.DataFrame | None = None,
                             positions=None, min_frac: float = 0.0,
                             include_rare: bool = True) -> go.Figure:
    """Treemap of haplotypes sized by count; if ``top_n`` is set, fold the rest
    into 'Other'. The subtitle reports the Gini coefficient of the distribution.

    When ``include_rare`` is False, variants containing an amino acid below
    ``min_frac`` at its codon position (a rare/'Other' residue in the AA pies)
    are excluded, and the summary stats reflect the remaining subset."""
    if hap_df.empty:
        return _empty("No haplotypes to display", title)

    if not include_rare:
        rare_by_pos = _rare_aa_by_position(aa_counts, positions, min_frac)
        if rare_by_pos:
            def _has_rare(combo_tuple) -> bool:
                return isinstance(combo_tuple, tuple) and any(
                    aa in rare_by_pos.get(pos, ()) for pos, aa in zip(positions, combo_tuple))
            hap_df = hap_df[~hap_df["combo_tuple"].map(_has_rare)]
        if hap_df.empty:
            return _empty("No variants remain after excluding rare amino acids", title)

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
                {"label": "Unique sequences", "value": f"{len(df):,}"},
                {"label": "Reads", "value": f"{total:,}"},
                {"label": "Gini", "value": f"{gini:.3f}"},
            ],
        },
    )
    return fig


# 20 standard amino acids ordered by biochemical group so same-group residues
# sit together in the hover table; colours match the AA pie charts
# (``theme.AA_COLORS``). Laid out column-major (first 10 left, last 10 right);
# the group order makes the left column exactly Hydrophobic(8)+Acidic(2) and the
# right Hydrophilic(4)+Basic(3)+Special(3), so no group straddles the columns.
_GROUP_ORDER = ("Hydrophobic", "Acidic", "Hydrophilic", "Basic", "Special")
_AA_BY_GROUP = [aa for group in _GROUP_ORDER
                for aa in theme.AA_ORDER if theme.AA_GROUPS.get(aa) == group]


def _aa_freq_block(counts_row: pd.Series, ref_aa: str | None = None) -> str:
    """Two-column list of the 20 amino acids with their relative frequency (%)
    at one codon. Each cell is coloured by biochemical group (see
    ``theme.AA_COLORS``); observed residues (and the reference) are bold, and
    cells reading 0.0% are faded. Relies on a monospace hover font for column
    alignment."""
    total = float(counts_row.sum())
    half = len(_AA_BY_GROUP) // 2

    def cell(aa: str) -> str:
        count = float(counts_row.get(aa, 0.0))
        pct = 100.0 * count / total if total > 0 else 0.0
        text = f"{aa} {pct:5.1f}%"
        if count > 0 or aa == ref_aa:
            text = f"<b>{text}</b>"
        color = theme.AA_COLORS.get(aa, theme.PALETTE["muted"])
        if round(pct, 1) == 0.0:  # displays as 0% – stay group-coloured but recede
            color = theme.fade(color)
        return f"<span style='color:{color}'>{text}</span>"

    rows = [f"{cell(_AA_BY_GROUP[i])}   {cell(_AA_BY_GROUP[i + half])}"
            for i in range(half)]
    return "<b>AA freq (%):</b><br>" + "<br>".join(rows)


def _codon_hover_labels(ref_seq: str, length: int, frame_offset: int,
                        aa_counts: pd.DataFrame | None = None
                        ) -> tuple[list[str], list[str] | None]:
    """Per-nucleotide hover pieces, frame-aligned to
    :func:`libraont.analysis.detect_variable_codons`.

    Returns ``(codon_heads, aa_blocks)`` as parallel lists. ``codon_heads`` gives
    the gene codon number for each position (positions before ``frame_offset``
    report no codon). ``aa_blocks`` is the per-codon two-column AA frequency
    table (reference amino acid bold), or ``None`` when ``aa_counts`` is not
    supplied.
    """
    ref_up = (ref_seq or "").upper()
    heads: list[str] = []
    blocks: list[str] | None = [] if aa_counts is not None else None
    for j in range(length):
        if j < frame_offset:
            heads.append("Codon n/a (outside reading frame)")
            if blocks is not None:
                blocks.append("")
            continue
        codon = (j - frame_offset) // 3 + 1
        start = frame_offset + 3 * (codon - 1)
        ref_aa = GENETIC_CODE.get(ref_up[start:start + 3])
        label = f"{ref_aa}{codon}" if ref_aa else f"{codon}"
        heads.append(f"<b>Codon:</b> {label}")
        if blocks is not None:
            if codon in aa_counts.index:
                blocks.append(_aa_freq_block(aa_counts.loc[codon], ref_aa))
            else:
                blocks.append("")
    return heads, blocks


def _empty(message: str, title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper",
                       text=message, showarrow=False, font=dict(color=theme.PALETTE["muted"]))
    fig.update_layout(template=_T, title=title, height=360)
    return fig
