"""Themed Plotly figures. Pure builders: each returns a ``go.Figure`` (no I/O).

Importing :mod:`libraont.theme` activates the shared template.
"""

from __future__ import annotations

import itertools
import math
import textwrap
from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import theme
from .alignment import ReadMap
from .analysis import haplotype_gini, reference_match_percent
from .constants import GENETIC_CODE

_T = theme.TEMPLATE_NAME

# Theme red, used to flag length-excluded reads and to mark the insert.
_RED = theme.PALETTE["danger"]
GHOST = "rgba(0,0,0,0)"

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
    """Read-length frequencies in 10 bp bins, over *all* reads. Dashed lines mark
    the length cutoffs that filter every other plot, and the plasmid length."""
    if not length_counts:
        return _empty("No reads found")
    bin_counts: Counter = Counter()
    for length, count in length_counts.items():
        bin_counts[(int(length) // 10) * 10] += count
    x, y = zip(*sorted(bin_counts.items()))
    labels = [f"{start}-{start + 9} bp" for start in x]
    # Red where the bin falls entirely outside the kept length window.
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
        # Labelled in the margin: the max cutoff usually sits close by.
        accent = theme.PALETTE["accent"]
        fig.add_vline(x=plasmid_len, line=dict(color=accent, width=1.4, dash="dash"))
        fig.add_annotation(xref="x", yref="paper", x=plasmid_len, y=1.02, yanchor="bottom",
                           # Kept inside the plot when the line is its right-hand edge.
                           xanchor="right" if plasmid_len >= x[-1] else "center",
                           text=f"plasmid ({plasmid_len:,} bp)", showarrow=False,
                           font=dict(size=11, color=accent))
    description = ("Read lengths in 10 bp bins, over every read in the FASTQ. "
                   "Dashed lines mark the length cutoffs, and reads outside them are "
                   "excluded from every other plot")
    description += (" - the coral line is the plasmid length, where whole-plasmid "
                    "reads pile up." if plasmid_len else ".")
    fig.update_layout(
        template=_T, title="Read length distribution",
        xaxis_title="Read length bin (bp)", yaxis_title="Count", bargap=0.05,
        meta={"description": description},
    )
    return fig


def read_quality_figure(phred_counts: Counter, min_phred: int | None = None) -> go.Figure:
    """Reads surviving each quality cutoff: for every whole Q, how many reads
    average at least that. The curve below the cutoff in force is drawn red, as
    the excluded bars are on the read-length plot."""
    if not phred_counts:
        return _empty("No quality scores found")
    # The cutoff joins the whole-Q points, so the two coloured stretches meet on it.
    steps = sorted(set(phred_counts) | ({min_phred} if min_phred is not None else set()))
    # Reverse-cumulative: reads at or above each step, read straight off as the
    # yield of that cutoff.
    survive = {b: sum(n for k, n in phred_counts.items() if k >= b) for b in steps}
    total = sum(phred_counts.values())

    def curve(points: list[int], colour: str, fill: str | None = None) -> go.Scatter:
        return go.Scatter(
            x=[survive[b] for b in points], y=points, mode="lines",
            customdata=[f"{survive[b] / total:.1%}" for b in points],
            # Splined, with modest smoothing: enough to read as a curve without
            # bowing far off the whole-Q points it interpolates.
            line=dict(color=colour, width=2.4, shape="spline", smoothing=0.8),
            fill=fill, fillcolor=_rgba(colour, 0.13), showlegend=False,
            hovertemplate="Mean Q%{y} or better<br>%{x} reads "
                          "(%{customdata})<extra></extra>")

    cut = min_phred if min_phred is not None else steps[0]
    below = [b for b in steps if b <= cut]
    # Both shaded areas bottom out on the lowest Q present, which is the axis.
    kept, floor_q = survive[cut], steps[0]
    fig = go.Figure()
    if len(below) > 1:
        # Discarded reads: between the cutoff's drop line and the curve below it.
        fig.add_trace(go.Scatter(x=[kept] * len(below), y=below, mode="lines",
                                 line_width=0, hoverinfo="skip", showlegend=False))
        fig.add_trace(curve(below, _RED, fill="tonextx"))
    # Kept reads: under the curve above the cutoff, squared off below it.
    fig.add_trace(curve([b for b in steps if b >= cut], theme.PALETTE["primary"],
                        fill="tozerox"))
    fig.add_shape(type="rect", x0=0, x1=kept, y0=floor_q, y1=cut, layer="below",
                  line_width=0, fillcolor=_rgba(theme.PALETTE["primary"], 0.13))
    # Crosshair on hover: the spike down to the Reads axis answers "how many
    # reads at this Q" without reading the tooltip.
    fig.update_layout(hovermode="closest")
    spike = dict(showspikes=True, spikemode="across", spikesnap="hovered data",
                 spikecolor=theme.PALETTE["muted"], spikethickness=1, spikedash="dot")
    fig.update_xaxes(**spike)
    fig.update_yaxes(**spike)
    if min_phred is not None:
        # Cutoff stops at the curve, then drops to the axis: where that foot lands
        # is how many reads the cutoff keeps.
        for x0, y0, x1, y1 in ((0, min_phred, kept, min_phred),
                               (kept, floor_q, kept, min_phred)):
            fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1, layer="above",
                          line=dict(color=_RED, width=1, dash="dash"))
        fig.add_trace(go.Scatter(
            x=[kept], y=[min_phred], mode="markers", showlegend=False,
            marker=dict(size=7, color=_RED),
            hovertemplate=f"Cutoff Q{min_phred}<br>{kept:,} reads kept "
                          f"({kept / total:.1%})<extra></extra>"))
        fig.add_annotation(x=kept / 2, y=min_phred, yshift=9, xanchor="center",
                           text=f"Q{min_phred} - {kept:,} reads kept", showarrow=False,
                           font=dict(size=11, color=_RED))
    description = ("Reads whose mean Phred is at least each whole Q, over every read in "
                   "the FASTQ - the yield left by any quality cutoff. Q10 is 90% "
                   "base-call accuracy, Q20 99%, Q30 99.9%")
    description += (". The dashed line is the cutoff in force - its foot on the axis "
                    "is the reads kept, and the red stretch below it those excluded "
                    "from every other plot."
                    if min_phred is not None else ".")
    fig.update_layout(
        template=_T, title="Read quality distribution",
        xaxis_title="Reads", yaxis_title="Mean Phred score per read (Q or better)",
        xaxis_range=[0, total * 1.04], yaxis_range=[floor_q, steps[-1] + 0.5],
        showlegend=False, meta={"description": description},
    )
    return fig


def _insert_marker(fig: go.Figure, start: float, end: float) -> None:
    """Red bar labelled "insert" in the margin above the plot."""
    fig.add_shape(type="rect", xref="x", yref="paper",
                  x0=start, x1=end, y0=1.02, y1=1.05,
                  fillcolor=_RED, line_width=0, layer="above")
    fig.add_annotation(xref="x", yref="paper", x=(start + end) / 2, y=1.06,
                       text="insert", showarrow=False, yanchor="bottom",
                       font=dict(size=11, color=_RED))


# Per-base agreement with the reference, as flat bands so one changed base is
# as visible as a long stretch. The three ways of differing are split because
# they mean different things: a substitution swaps a residue, a deletion removes
# bases, an insertion adds them.
_MATCH_GREY = "#B4BAC1"
_SUBSTITUTION_AMBER = "#E0B252"
_DELETION_RED = _RED
_INSERTION_BLUE = "#3A6FB0"
# Four flat bands over zmin=0.5..zmax=4.5, so codes 1-4 land mid-band.
_AGREEMENT_SCALE = [[0.00, _MATCH_GREY], [0.25, _MATCH_GREY],
                    [0.25, _SUBSTITUTION_AMBER], [0.50, _SUBSTITUTION_AMBER],
                    [0.50, _DELETION_RED], [0.75, _DELETION_RED],
                    [0.75, _INSERTION_BLUE], [1.00, _INSERTION_BLUE]]
# A base is one pixel over a whole plasmid, so every departure also gets a
# square marker; otherwise the map reads as uniform grey.
_EVENT_MARKER = dict(size=3, symbol="square", line=dict(width=0))


def read_alignment_figure(cov: ReadMap, insert_seq: str | None = None,
                          insertion_bp: int = 0, deletion_bp: int = 0,
                          auto_match_threshold: float | None = None) -> go.Figure:
    """One row per read, coloured base by base against the reference: grey
    match, amber substitution, red deletion, blue on the base an insertion
    begins before, blank where the read does not reach. Each departure also gets
    a square marker so it survives being one pixel wide, and indels below their
    direction's threshold are dropped as basecall noise.

    Rows sort by start position, so the pileup reads as a staircase and a shared
    change lines up as a vertical stripe. Above the map: the percentage of
    covering reads matching at each base (over every read, not just the rows
    drawn), and the insert as its reference amino acids.
    """
    ends, idents = cov.read_ends, cov.read_identities
    matrix = cov.match_matrix
    # mapped_reads counts every read; the arrays may hold an even sample of them.
    n_total = max(int(cov.mapped_reads), int(cov.read_starts.shape[0]))
    if n_total == 0 or matrix.size == 0:
        return _empty("No reads aligned to the plasmid", "Read alignment map")

    order = np.lexsort((ends, cov.read_starts))
    ends, idents, matrix = ends[order], idents[order], matrix[order]
    n_shown = matrix.shape[0]
    rank = np.empty_like(order)          # old row index -> new, for the sparse records
    rank[order] = np.arange(order.size)

    z = matrix.astype(float)
    z[z == 0] = np.nan                   # uncovered renders as background
    # Insertions own no reference base, so the cell they begin before is painted
    # here, after thresholding; sub-threshold deletions are blanked rather than
    # recoloured, the read having no base there to show.
    ins_keep = ((cov.insertion_rows < order.size)
                & (cov.insertion_lengths >= max(insertion_bp, 1)))
    if ins_keep.any():
        z[rank[cov.insertion_rows[ins_keep]],
          cov.insertion_positions[ins_keep] - 1] = 4
    del_keep = ((cov.deletion_rows < order.size)
                & (cov.deletion_lengths >= max(deletion_bp, 1)))
    for read, at, length in zip(cov.deletion_rows[~del_keep],
                                cov.deletion_positions[~del_keep],
                                cov.deletion_lengths[~del_keep]):
        if read < order.size:
            z[rank[read], at - 1:at - 1 + length] = np.nan

    x_max = cov.contig_length or int(ends.max())
    _T_MARGIN, _B_MARGIN = 60, 48
    height = int(min(880, max(340, n_shown + _T_MARGIN + _B_MARGIN + 230)))
    positions = np.arange(1, matrix.shape[1] + 1)

    depth = cov.depth_counts.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = np.where(depth > 0, 100.0 * cov.match_counts / depth, np.nan)

    # The insert's amino acids, then its bases, laid over the plasmid coordinate.
    tracks = []
    if insert_seq and cov.region:
        reverse = cov.region.get("strand") == "-"
        placing = dict(origin=cov.region["end"] if reverse else cov.region["start"],
                       step=-1 if reverse else 1)
        tracks = [t for t in (
            _aa_track(insert_seq, len(insert_seq), 0, yaxis=None,
                      number_labels=True, **placing),
            _base_track(insert_seq, len(insert_seq), yaxis=None, **placing),
        ) if t is not None]

    rows = len(tracks) + 2
    heights = {2: [0.06, 0.05, 0.30, 0.59], 1: [0.07, 0.30, 0.63]}.get(
        len(tracks), [0.30, 0.70])
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=heights,
                        vertical_spacing=0.025)
    trace_row, map_row = len(tracks) + 1, len(tracks) + 2
    for i, track in enumerate(tracks, start=1):
        fig.add_trace(track, row=i, col=1)
    fig.add_trace(go.Scatter(
        x=positions, y=pct, mode="lines", name="Match",
        line=dict(color=theme.PALETTE["primary"], width=1.4),
        fill="tozeroy", fillcolor="rgba(31, 111, 139, 0.12)",
        customdata=depth,
        hovertemplate="Position %{x}<br>%{y:.1f}% of %{customdata:.0f} reads match"
                      "<extra></extra>"), row=trace_row, col=1)
    if auto_match_threshold is not None:
        fig.add_hline(y=auto_match_threshold, row=trace_row, col=1,
                      line=dict(color=theme.PALETTE["muted"], width=1.5,
                                dash="3px,3px"),
                      annotation_text="codon detection cutoff",
                      annotation_position="top left",
                      annotation_font=dict(size=10,
                                           color=theme.PALETTE["muted"]))
    fig.add_trace(go.Heatmap(
        z=z, x=positions, showscale=False, zmin=0.5, zmax=4.5,
        colorscale=_AGREEMENT_SCALE,
        hovertemplate="Position %{x}<br>read row %{y}<extra></extra>"),
        row=map_row, col=1)

    # A marker per coloured cell, so no departure is lost to being one pixel
    # wide. Taken from the rendered z, not the raw matrix: an insertion repaints
    # the cell it begins before, which must not also claim a substitution.
    sub_rows, sub_cols = np.nonzero(z == 2)
    if sub_rows.size:
        fig.add_trace(go.Scatter(
            x=sub_cols + 1, y=sub_rows, mode="markers", showlegend=False,
            marker=dict(color=_SUBSTITUTION_AMBER, **_EVENT_MARKER),
            hovertemplate="Substitution at position %{x}<extra></extra>"),
            row=map_row, col=1)

    del_rows, del_cols = np.nonzero(z == 3)
    if del_rows.size:
        # Each deleted base carries the length of the deletion it belongs to.
        spans = {}
        for read, at, length in zip(cov.deletion_rows[del_keep],
                                    cov.deletion_positions[del_keep],
                                    cov.deletion_lengths[del_keep]):
            for offset in range(length):
                spans[(rank[read], at - 1 + offset)] = length
        fig.add_trace(go.Scatter(
            x=del_cols + 1, y=del_rows, mode="markers", showlegend=False,
            marker=dict(color=_DELETION_RED, **_EVENT_MARKER),
            customdata=[spans.get((r, c), 0) for r, c in zip(del_rows, del_cols)],
            hovertemplate="Deletion of %{customdata:,} bp<br>"
                          "at position %{x}<extra></extra>"),
            row=map_row, col=1)

    if ins_keep.any():
        fig.add_trace(go.Scatter(
            x=cov.insertion_positions[ins_keep],
            y=rank[cov.insertion_rows[ins_keep]], mode="markers", showlegend=False,
            marker=dict(color=_INSERTION_BLUE, **_EVENT_MARKER),
            customdata=cov.insertion_lengths[ins_keep],
            hovertemplate="Insertion of %{customdata:,} bp<br>"
                          "before position %{x}<extra></extra>"),
            row=map_row, col=1)

    cards = [{"label": "Reads on map", "value": f"{n_shown:,}",
              "sub": "one row per read" if n_shown == n_total
                     else f"sampled from {n_total:,} reads"}]
    finite = np.isfinite(idents)
    if finite.any():
        cards.append({"label": "Median identity",
                      "value": f"{np.median(idents[finite]) * 100:.1f}%"})
    if cov.region:
        if not tracks:
            _insert_marker(fig, cov.region["start"], cov.region["end"])
        cards.append({"label": "Insert",
                      "value": f"{cov.region['start']}-{cov.region['end']}"})

    fig.update_layout(
        template=_T, title="Read alignment map", height=height, showlegend=False,
        bargap=0, uniformtext=dict(minsize=5, mode="hide"),
        margin=dict(l=60, r=48, t=_T_MARGIN + 12, b=_B_MARGIN),
        meta={
            "description": "Each row is one read across the plasmid, coloured base "
                           "by base: grey matches the reference, amber a substitution, "
                           "red a deletion, blue where an insertion begins, blank "
                           "where the read does not reach. Above it, the percentage "
                           "of covering reads matching at each base carries the "
                           "codon-detection cutoff, over bands giving the insert's "
                           "amino acids and its reference bases.",
            "metric_cards": cards,
        })
    for i, title in enumerate(["AA", "bp"][:len(tracks)], start=1):
        fig.update_yaxes(range=[0, 1], showgrid=False, zeroline=False,
                         showticklabels=False, title_text=title, row=i, col=1)
        fig.update_xaxes(range=[0, x_max], row=i, col=1)
    fig.update_yaxes(title_text="Match (%)", range=[0, 101], row=trace_row, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, title="",
                     autorange="reversed", row=map_row, col=1)
    fig.update_xaxes(range=[0, x_max], row=trace_row, col=1)
    fig.update_xaxes(title_text="Plasmid position (bp)", range=[0, x_max],
                     zeroline=False, showline=True, ticks="outside", row=map_row, col=1)
    return fig


# --- Read fates --------------------------------------------------------------
# One square per percent of the library, coloured by what became of those reads.
# Proportions read more accurately off an icon array than off bar length, which
# is the whole job here: "what fraction of my library can I actually use?"
# Variants are the library's product and read saturated; wild type is parental
# background and recedes; warm colours are assembly defects.
FATE_COLORS: dict[str, str] = {
    "Variant": "#1F6F5C",
    "Ambiguous diversity": "#9FB6C4",
    "Wild type": "#E8913C",
    "Deletion": "#D8C13F",
    "Insertion": _RED,
    # Neutral greys, darkening with severity.
    "Does not contain insert region": "#646C73",
}
def fate_descriptors(insertion_bp: int, deletion_bp: int) -> dict[str, str]:
    """A few words per fate, quoting its thresholds - the metric cards are the
    plot's only key, so each has to state its own rule."""
    return {
        "Variant": "≥1 diversified codon changed",
        "Wild type": "no diversified codon changed",
        "Ambiguous diversity": "≥1 diversified codon unreadable or uncovered",
        "Deletion": f"deletion ≥{deletion_bp} bp",
        "Insertion": f"insertion ≥{insertion_bp} bp",
        "Does not contain insert region": "aligned, but clear of the insert",
    }


def _aa_track(ref_seq: str, length: int, frame_offset: int, *,
              yaxis: str | None = "y2", origin: int = 1, step: int = 1,
              number_labels: bool = False) -> go.Bar | None:
    """Reference amino acid of each codon as a coloured band; letters show
    wherever they fit.

    ``origin``/``step`` place the band on a different axis: the first codon's
    first base sits at ``origin``, each base advancing by ``step`` (-1 for an
    insert on the reverse strand). ``number_labels`` appends the codon number.
    """
    ref_up = (ref_seq or "").upper()
    centres, letters, colors, hovers = [], [], [], []
    for codon, start in enumerate(range(frame_offset, length - 2, 3), start=1):
        aa = GENETIC_CODE.get(ref_up[start:start + 3]) or "?"
        centres.append(origin + step * (start + 1))
        letters.append(f"{aa}{codon}" if number_labels else aa)
        colors.append(theme.AA_COLORS.get(aa, theme.PALETTE["muted"]))
        hovers.append(f"Codon {codon}: {aa} ({_AA_NAMES.get(aa, aa)})")
    if not centres:
        return None
    bar = go.Bar(x=centres, y=[1] * len(centres), width=3,
                 marker=dict(color=colors, line=dict(width=0)),
                 text=letters, textposition="inside", insidetextanchor="middle",
                 textangle=0,  # never let Plotly rotate letters in narrow blocks
                 textfont=dict(family="monospace", size=11,
                               color=[theme.contrast_text(c) for c in colors]),
                 showlegend=False,
                 hovertext=hovers, hoverinfo="text" if number_labels else "skip")
    if yaxis:
        bar.update(yaxis=yaxis)
    return bar


def _base_track(ref_seq: str, length: int, yaxis: str | None = "y3", *,
                origin: int = 1, step: int = 1) -> go.Bar | None:
    """Reference base at each position as a coloured band; letters show wherever
    they fit. ``origin``/``step`` place the band elsewhere on the x axis, as for
    :func:`_aa_track`."""
    bases = (ref_seq or "").upper()[:length]
    if not bases:
        return None
    colors = [theme.BASE_COLORS.get(b, theme.PALETTE["muted"]) for b in bases]
    bar = go.Bar(x=[origin + step * i for i in range(len(bases))],
                 y=[1] * len(bases), width=1,
                 marker=dict(color=colors, line=dict(width=0)),
                 text=list(bases), textposition="inside",
                 insidetextanchor="middle", textangle=0,
                 textfont=dict(family="monospace", size=9,
                               color=[theme.contrast_text(c) for c in colors]),
                 showlegend=False, hoverinfo="skip")
    if yaxis:
        bar.update(yaxis=yaxis)
    return bar


def gap_match_figure(df_counts: pd.DataFrame, ref_seq: str, *, gap_char: str = "-",
                     shade_codons=None, frame_offset: int = 0,
                     auto_match_threshold: float | None = None,
                     aa_counts: pd.DataFrame | None = None,
                     reads_passing: int | None = None,
                     reads_total: int | None = None) -> go.Figure:
    """Per-position reference-match % (gaps excluded from the denominator),
    optionally shading codons and marking the auto-detect cutoff. A band above
    gives each codon's reference amino acid; with ``aa_counts``, the hover also
    lists AA frequencies at that codon."""
    if gap_char not in df_counts.columns:
        raise ValueError(f"Gap column '{gap_char}' not in df_counts: {list(df_counts.columns)}")
    if frame_offset not in (0, 1, 2):
        raise ValueError("frame_offset must be 0, 1, or 2")

    L = len(df_counts)
    match_perc = reference_match_percent(df_counts, ref_seq, gap_char)

    x = df_counts.index.astype(int)
    codon_heads, aa_blocks = _codon_hover_labels(ref_seq, L, frame_offset, aa_counts)
    fig = go.Figure()
    # Invisible, drawn first so the codon number heads the unified hover with no
    # colour swatch; the trailing <br> leaves a blank separator line.
    fig.add_trace(go.Scatter(x=x, y=match_perc, mode="lines",
                             line=dict(color="rgba(0,0,0,0)"), showlegend=False,
                             customdata=codon_heads,
                             hovertemplate="%{customdata}<br><extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=match_perc, mode="lines+markers",
                             name="Reference match %", line=dict(color=theme.MATCH_COLOR),
                             marker=dict(size=4),
                             hovertemplate="<b>Reference match:</b> %{y:.2f}%<br><extra></extra>"))
    if aa_blocks is not None:
        # Invisible, drawn last so the AA table sits under the line entry.
        fig.add_trace(go.Scatter(x=x, y=match_perc, mode="lines",
                                 line=dict(color="rgba(0,0,0,0)"), showlegend=False,
                                 customdata=aa_blocks,
                                 hovertemplate="%{customdata}<extra></extra>"))

    aa_track = _aa_track(ref_seq, L, frame_offset)
    if aa_track is not None:
        fig.add_trace(aa_track)
    base_track = _base_track(ref_seq, L)
    if base_track is not None:
        fig.add_trace(base_track)

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

    # Reads behind these curves, falling back to the per-column maximum when the
    # report-level totals are absent.
    if reads_passing is not None and reads_total is not None:
        excluded = max(reads_total - reads_passing, 0)
        pass_sub = f"{reads_passing / reads_total * 100:.1f}% of {reads_total:,}" if reads_total else ""
        fail_sub = f"{excluded / reads_total * 100:.1f}% of {reads_total:,}" if reads_total else ""
        read_cards = [
            {"label": "Reads used", "value": f"{reads_passing:,}", "sub": pass_sub},
            {"label": "Excluded, structural defect", "value": f"{excluded:,}",
             "sub": fail_sub},
        ]
    else:
        n_seqs = int((df_counts.sum(axis=1) - df_counts[gap_char]).max())
        read_cards = [{"label": "Aligned reads", "value": f"{n_seqs:,}"}]
    cards = [
        *read_cards,
        {"label": "Detected positions", "value": f"{len(shade_codons or []):,}"},
    ]
    if auto_match_threshold is not None:
        cards.append({"label": "Detection threshold",
                      "value": f"{auto_match_threshold:g}%"})
    fig.update_layout(
        template=_T, hovermode="x unified",
        title="Alignment to reference insert",
        xaxis_title="Position in reference (1-based, nucleotides)",
        # Pinned to the full range: a match track that autoscaled would make a
        # 99%-vs-97% wobble look like a cliff.
        yaxis=dict(title="Percentage (%)", range=[0, 100],
                   domain=[0, 0.83] if base_track else
                          [0, 0.90] if aa_track else [0, 1]),
        yaxis2=dict(domain=[0.95, 1.0], range=[0, 1], fixedrange=True,
                    showgrid=False, zeroline=False, showticklabels=False, ticks=""),
        # The base band sits just under the codons it spells out, and gets the
        # same depth so its letters survive the uniform-text minimum.
        yaxis3=dict(domain=[0.885, 0.935], range=[0, 1], fixedrange=True,
                    showgrid=False, zeroline=False, showticklabels=False, ticks=""),
        barmode="overlay",
        # Letters shrink to fit, dropping out only when unreadable.
        uniformtext=dict(minsize=5, mode="hide"),
        margin=dict(t=90),
        hoverlabel=dict(font=dict(family="monospace", size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        meta={
            "description": "Reference-match frequency at each insert base, gaps "
                           "excluded from the denominator; low positions mark likely "
                           "variable or mutated sites. The bands above give the "
                           "reference amino acid at each codon, coloured by "
                           "biochemical group, and the reference base at each "
                           "position.",
            "metric_cards": cards,
        },
    )
    return fig


def aa_pies_figure(df_aa_counts: pd.DataFrame, positions, *, min_frac: float = 0.01,
                   pie_subtitles=None,
                   title: str = "Amino Acid distribution at selected codon positions",
                   hole: float = 0.5, height_per_row: int = 300,
                   width_per_col: int = 300) -> go.Figure | None:
    """Donut per codon position; slices below ``min_frac`` fold into 'Other'."""
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
    """Per codon, the amino acids below ``min_frac`` - the pies' 'Other' slice."""
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
                             positions=None, min_frac: float = 0.0) -> go.Figure:
    """Treemap of haplotypes sized by count; ``top_n`` folds the rest into
    'Other'. Variants carrying an amino acid below ``min_frac`` at its codon are
    excluded (0 keeps everything), and the stats follow the remaining subset.

    A read missing more than the tolerated number of diversified codons has no
    combination to place, so it is absent from the tiles and the denominator;
    within the tolerance, the codons it misses read '?'."""
    if hap_df.empty:
        return _empty("No haplotypes to display", title)

    rare_by_pos = _rare_aa_by_position(aa_counts, positions, min_frac)
    if rare_by_pos:
        def _has_rare(combo_tuple) -> bool:
            return isinstance(combo_tuple, tuple) and any(
                aa in rare_by_pos.get(pos, ()) for pos, aa in zip(positions, combo_tuple))
        hap_df = hap_df[~hap_df["combo_tuple"].map(_has_rare)]
    if hap_df.empty:
        return _empty("No variants remain above the grouping threshold", title)

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
            "description": "Unique amino-acid haplotypes across the selected codons, "
                           "sized by abundance, with the reference combination "
                           "outlined in black. Gini summarises unevenness: 0 is evenly "
                           "represented, closer to 1 means a few variants dominate.",
            "metric_cards": [
                {"label": "Unique sequences", "value": f"{len(df):,}"},
                {"label": "Reads", "value": f"{total:,}"},
                {"label": "Gini", "value": f"{gini:.3f}"},
            ],
        },
    )
    return fig


# Amino acids grouped so same-group residues sit together in the hover table.
# Laid out column-major (first 10 left, last 10 right); this group order splits
# 10/10 exactly, so no group straddles the two columns.
_GROUP_ORDER = ("Hydrophobic", "Acidic", "Hydrophilic", "Basic", "Special")
_AA_BY_GROUP = [aa for group in _GROUP_ORDER
                for aa in theme.AA_ORDER if theme.AA_GROUPS.get(aa) == group]


def _aa_freq_block(counts_row: pd.Series, ref_aa: str | None = None) -> str:
    """Two-column list of all amino acids and their frequency (%) at one codon,
    coloured by group. Observed residues and the reference are bold, 0.0% cells
    faded; needs a monospace hover font to line up."""
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

    ``(codon_heads, aa_blocks)``: the codon number at each position, and the
    two-column AA frequency table for it (``None`` without ``aa_counts``).
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


def _label(text: str, count: int, width: int = 17) -> str:
    """Node caption: the name wrapped to ``width``, then the count in bold."""
    return "<br>".join(textwrap.wrap(text, width) + [f"<b>{count:,}</b>"])


def _rgba(hex_colour: str, alpha: float) -> str:
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


_DIVERSITY_HOVER = {
    "Variant": "carry a change at one or more diversified codons",
    "Wild type": "cover every diversified codon and carry no change at any of them",
    "Ambiguous diversity": "carry no detected change, but do not cover every "
                           "diversified codon",
}
_EMPTY = "No insert (empty vector)"


def read_funnel_sankey_figure(funnel, fates=(), insertion_bp: int = 0,
                              deletion_bp: int = 0) -> go.Figure:
    """Reads flowing through each filter: a teal trunk narrowing along the top,
    with each step's losses curving away below it. Early filters drop their reads
    where they leave; the verdicts on a read that reached the insert - empty
    vector, structural indel, and the diversity call - all run to the right-hand
    edge so they can be read off together. Empty branches keep a hairline and a
    label, so nothing is ever hidden."""
    stages = [s for s in funnel if s.count is not None]
    if len(stages) < 2:
        return _empty("No filtering steps to draw")
    pal = theme.PALETTE
    last = next((i for i, s in enumerate(stages) if s.label == "Correctly assembled"),
                len(stages) - 1)
    stages = stages[:last + 1]
    counts = {f.label: f.count for f in fates}
    describe = fate_descriptors(insertion_bp, deletion_bp)
    # Losses from these steps are verdicts on a read that reached the insert, so
    # they run to the edge instead of stopping under their own column - the
    # structural step splitting into the two kinds of defect.
    to_edge = {
        "Contains the insert": [
            (_EMPTY, "Does not contain insert region",
             FATE_COLORS["Does not contain insert region"])],
        "Correctly assembled": [
            (k, k, FATE_COLORS[k]) for k in ("Deletion", "Insertion")],
    }
    hover_for = dict(_DIVERSITY_HOVER,
                     Deletion=f"carry a deletion ≥ {deletion_bp} bp",
                     Insertion=f"carry an insertion ≥ {insertion_bp} bp")
    call = [(label, counts[label], FATE_COLORS[label], hover_for[label], label)
            for label in ("Variant", "Wild type", "Ambiguous diversity")
            if label in counts]

    n, start = len(stages), stages[0].count or 1
    # Zero-width links vanish in Plotly, so give them a sliver of the total.
    hair = start * 0.005
    kept = [max(s.count, hair) for s in stages]
    lost = [max(a.count - b.count, hair) for a, b in zip(stages, stages[1:])]
    # A transparent flow parked at the right edge widens the scale Plotly draws
    # against, freeing the lower half for the branches to curve into. Plotly
    # sizes a node as value/depth of the plot, with depth the total flow, so
    # every top edge lands on ``crest`` and the trunk stays level.
    crest, floor, gap = 0.01, 0.97, 0.07
    depth = max(kept[0] / 0.36, (kept[0] + max(lost)) / (floor - crest - gap))
    spacer = depth - kept[0]

    right = 0.80
    step = right / (n - 1)
    xs = [0.02 + i * step for i in range(n)]
    ys = [crest + 0.5 * k / depth for k in kept]
    tints = [pal["primary_dark"]] * n
    caps = [f"<b>{s.label}</b><br>{s.count:,} reads"
            + (f" - {s.passed}" if s.passed else "") for s in stages]
    src, tgt, value, hover, colors = [], [], [], [], []
    drops, tail = [], []                     # short branches, and edge-bound ones
    for i, (a, b) in enumerate(zip(stages, stages[1:])):
        gone = a.count - b.count
        note = f"<b>{gone:,} reads</b> {b.failed or 'were removed here'}"
        src.append(i)
        tgt.append(i + 1)
        value.append(kept[i + 1])
        hover.append(f"<b>{b.count:,} reads</b> {b.passed or 'carried on'}")
        colors.append(_rgba(pal["primary"], 0.32))
        if b.label in to_edge:
            tail.append([(label, counts.get(fate, gone), tint,
                          f"<b>{counts.get(fate, gone):,} reads</b> "
                          f"{hover_for.get(fate) or b.failed or 'were removed'}",
                          i, fate)
                         for label, fate, tint in to_edge[b.label]])
            continue
        src.append(i)
        tgt.append(len(xs))
        value.append(lost[i])
        hover.append(note)
        colors.append(_rgba(_RED, 0.42))
        drops.append((xs[i + 1], b.lost or "removed", gone))
        xs.append(xs[i + 1])
        ys.append(floor - 0.5 * lost[i] / depth)
        tints.append(_RED)
        caps.append(note)

    # Everything that ends the story lands in one column, the trunk's own
    # continuation on top and each verdict stacked below it in the order it left.
    edge = [(label, count, tint, f"<b>{count:,} reads</b> {note}", n - 1, fate)
            for label, count, tint, note, fate in call]
    edge += [item for group in reversed(tail) for item in group]
    ends = [max(count, hair) for _, count, _, _, _, _ in edge]
    splay = min(0.13, (0.95 - crest - sum(ends) / depth) / max(1, len(edge) - 1))
    fan, top = len(xs), crest
    for j, (label, count, tint, note, source, _) in enumerate(edge):
        src.append(source)
        tgt.append(len(xs))
        value.append(ends[j])
        hover.append(note)
        colors.append(_rgba(tint, 0.42))
        xs.append(0.02 + right + 0.05)
        ys.append(top + 0.5 * ends[j] / depth)
        tints.append(tint)
        caps.append(f"<b>{label}</b><br>{count:,} reads")
        top += ends[j] / depth + splay

    src.append(len(xs))                      # the invisible scale-setter
    tgt.append(len(xs) + 1)
    value.append(spacer)
    hover.append("")
    colors.append(GHOST)
    xs += [0.999, 0.9999]                    # tucked into the right-hand edge
    ys += [0.5, 0.5]
    tints += [GHOST, GHOST]
    caps += ["", ""]
    # Plotly would pop an empty hover box over the scale-setter, so its label is
    # made transparent rather than merely blank.
    def ghost_label(total, ghosts):
        real = total - ghosts
        return dict(bgcolor=[pal["surface"]] * real + [GHOST] * ghosts,
                    bordercolor=[pal["grid"]] * real + [GHOST] * ghosts,
                    font=dict(color=[pal["text"]] * real + [GHOST] * ghosts))

    # Captions ride outside the ribbon - Plotly's own node labels sit on top of
    # it in a halo box, and collide once the columns are close.
    lo, hi = 0.13, 0.80
    notes = [dict(x=xs[i], y=hi + 0.015, yanchor="bottom",
                  xanchor="left" if i == 0 else "right" if i == n - 1 else "center",
                  text=f"<b>{'<br>'.join(textwrap.wrap(s.label, 16))}</b><br>"
                       f"<span style='color:{pal['muted']}'>{s.count:,} · "
                       f"{s.count / start:.0%}</span>",
                  font=dict(size=11, color=pal["primary_dark"]))
             for i, s in enumerate(stages)]
    notes += [dict(x=x, y=lo - 0.015, yanchor="top", xanchor="center",
                   text=f"{'<br>'.join(textwrap.wrap(label, 18))}<br><b>{gone:,}</b>",
                   font=dict(size=10.5, color=_RED))
              for x, label, gone in drops]
    notes += [dict(x=xs[fan + j] + 0.012, y=hi - (hi - lo) * ys[fan + j],
                   xanchor="left", yanchor="middle",
                   text=f"<b>{'<br>'.join(textwrap.wrap(label, 14))}</b><br>"
                        f"{count:,} · {count / start:.0%}",
                   font=dict(size=10.5, color=tint))
              for j, (label, count, tint, _, _, _) in enumerate(edge)]

    fig = go.Figure(go.Sankey(
        arrangement="fixed", domain=dict(x=[0, 1], y=[lo, hi]),
        node=dict(label=[""] * len(xs), x=xs, y=ys, pad=0, thickness=11,
                  color=tints, line=dict(color="white", width=0.5),
                  customdata=caps, hoverlabel=ghost_label(len(xs), 2),
                  hovertemplate="%{customdata}<extra></extra>"),
        link=dict(source=src, target=tgt, value=value, color=colors,
                  customdata=hover, hoverlabel=ghost_label(len(value), 1),
                  hovertemplate="%{customdata}<extra></extra>"),
    ))
    fig.update_layout(
        template=_T, title="Library FASTQ read processing", height=600,
        margin=dict(l=10, r=10, t=54, b=10),
        annotations=[dict(xref="paper", yref="paper", showarrow=False,
                          align="center", **a) for a in notes],
        meta={"description":
              "Every read in the FASTQ followed through each filter: the teal "
              "trunk is what carries on, each red branch what that step removes. "
              f"{stages[-1].count:,} of {start:,} reads "
              f"({stages[-1].count / start:.1%}) are correctly assembled, and the "
              "column on the right is every way a read can end up.",
              # The key for that column, moved off the icon array so it sits
              # with the branches it names.
              "metric_cards": [
                  {"label": label, "value": f"{count / start:.0%}",
                   "sub": f"{count:,} read{'' if count == 1 else 's'} - "
                          f"{describe.get(fate, '')}", "color": tint}
                  for label, count, tint, _, _, fate in edge]},
    )
    return fig
