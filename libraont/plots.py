"""Themed Plotly figures. Pure builders: each returns a ``go.Figure`` (no I/O).

Importing :mod:`libraont.theme` activates the shared template.
"""

from __future__ import annotations

import math
import textwrap
from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import theme
from .alignment import SUBSTITUTION_CODES, ReadMap
from .analysis import reference_match_percent
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


def _length_bars(length_counts: Counter, min_read_len: int | None,
                 max_read_len: int | None) -> tuple[go.Bar, list[int]]:
    """Read lengths in 10 bp bins, red where the bin falls entirely outside the
    kept window."""
    bin_counts: Counter = Counter()
    for length, count in length_counts.items():
        bin_counts[(int(length) // 10) * 10] += count
    x, y = zip(*sorted(bin_counts.items()))
    colors = [
        _RED if ((min_read_len is not None and start + 9 < min_read_len)
                 or (max_read_len is not None and start > max_read_len))
        else theme.PALETTE["primary"]
        for start in x
    ]
    return go.Bar(x=x, y=y, width=9, marker_color=colors, showlegend=False,
                  customdata=[f"{start}-{start + 9} bp" for start in x],
                  hovertemplate="Length %{customdata}<br>%{y:,} reads"
                                "<extra></extra>"), list(x)


def _quality_bars(phred_counts: Counter, min_phred: int | None) -> go.Bar:
    """One bar per whole Q of mean read quality, red below the cutoff - the same
    colouring as the length bins above."""
    x, y = zip(*sorted(phred_counts.items()))
    colors = [_RED if (min_phred is not None and q < min_phred)
              else theme.PALETTE["primary"] for q in x]
    return go.Bar(x=x, y=y, width=0.9, marker_color=colors, showlegend=False,
                  hovertemplate="Mean Q%{x}<br>%{y:,} reads<extra></extra>")


def _kept_pie(n_input: int, n_length_kept: int, n_quality_kept: int,
              min_phred: int | None) -> go.Pie:
    """Included against excluded, the count in the hole and the hover naming
    which filter dropped what."""
    lost_length, lost_quality = n_input - n_length_kept, n_length_kept - n_quality_kept
    quality = f"below Q{min_phred}" if min_phred is not None else "below the cutoff"
    return go.Pie(
        title=dict(text=f"<b>{n_quality_kept:,}</b><br>"
                        f"<span style='font-size:11px'>of {n_input:,} reads</span>",
                   position="middle center",
                   font=dict(size=14, color=theme.PALETTE["muted"])),
        labels=["Included", "Excluded"], values=[n_quality_kept, lost_length + lost_quality],
        marker=dict(colors=[theme.PALETTE["primary"], _RED],
                    line=dict(color=theme.PALETTE["bg"], width=2)),
        hole=0.58, sort=False, direction="clockwise", showlegend=False,
        automargin=True, textinfo="label+percent", textposition="outside",
        textfont=dict(size=12, color=theme.PALETTE["text"]),
        customdata=["carried into every other plot",
                    f"{lost_length:,} outside the length window, "
                    f"{lost_quality:,} {quality}"],
        hovertemplate="%{label}<br>%{value:,} reads (%{percent})<br>"
                      "%{customdata}<extra></extra>")


def _mean(counts: Counter) -> float:
    """Mean of a value -> count mapping."""
    n = sum(counts.values())
    return sum(v * c for v, c in counts.items()) / n if n else 0.0


def read_summary_figure(length_counts: Counter, phred_counts: Counter, funnel=(),
                        min_read_len: int | None = None,
                        max_read_len: int | None = None,
                        min_phred: int | None = None,
                        plasmid_len: int | None = None,
                        insert_len: int | None = None,
                        mean_phred: float | None = None) -> go.Figure:
    """The read set as it arrives: read length, mean read quality, and what the
    two cutoffs keep. Bars a cutoff excludes are red, as are the cutoff lines
    and the slice they account for. The funnel below carries on from here."""
    if not length_counts and not phred_counts:
        return _empty("No reads found", "Read Dataset and Filtering")
    # The funnel already counts what the filters keep, stage by stage.
    counts = {s.label: s.count for s in funnel if s.count is not None}
    n_input = counts.get("Reads in FASTQ", sum(length_counts.values()))
    n_length_kept = counts.get("Within read-length window", n_input)
    n_quality_kept = counts.get("Above the quality cutoff", n_length_kept)

    fig = go.Figure()
    annotations = [_panel_title(0.31, "Read length distribution"),
                   _panel_title(0.31, "Read quality distribution", y=0.43),
                   _panel_title(0.86, "Reads kept")]
    if length_counts:
        bars, bins = _length_bars(length_counts, min_read_len, max_read_len)
        fig.add_trace(bars)
        annotations += [_cutoff_line(fig, "x", "y", value, label, _RED)
                        for value, label in ((min_read_len, "min"), (max_read_len, "max"))
                        if value is not None]
        if plasmid_len:
            label = _cutoff_line(fig, "x", "y", plasmid_len,
                                 f"plasmid ({plasmid_len:,} bp)",
                                 theme.PALETTE["accent"], width=1.4)
            # Level with min and max; anchored inward at the right-hand edge.
            if plasmid_len >= bins[-1]:
                label.update(xanchor="right")
            annotations.append(label)
    if phred_counts:
        fig.add_trace(_quality_bars(phred_counts, min_phred).update(xaxis="x2",
                                                                   yaxis="y2"))
        if min_phred is not None:
            annotations.append(_cutoff_line(fig, "x2", "y2", min_phred - 0.5,
                                            f"Q{min_phred}", _RED))
    if n_input:
        fig.add_trace(_kept_pie(n_input, n_length_kept, n_quality_kept, min_phred)
                      .update(domain=dict(x=[0.72, 1.0], y=[0.08, 0.92])))

    description = (
        "Above, every read in the FASTQ by length and by mean Phred, red where a "
        "dashed cutoff excludes it, and the ring is what the two cutoffs keep. "
        "Below, those reads followed through each filter: the teal trunk carries "
        "on, each red branch is what that step removes, and the right-hand column "
        "is every way a read can end up.")
    fig.update_layout(
        template=_T, title="Read Dataset and Filtering", height=560, bargap=0.05,
        showlegend=False, margin=dict(l=56, r=24, t=72, b=44),
        annotations=annotations,
        # The two histograms stack in the left column; the ring sits beside them.
        xaxis=dict(domain=[0.0, 0.62], anchor="y", title_text="Read length bin (bp)"),
        yaxis=dict(domain=[0.62, 1.0], anchor="x", title_text="Count"),
        xaxis2=dict(domain=[0.0, 0.62], anchor="y2",
                    title_text="Mean Phred per read (Q)"),
        yaxis2=dict(domain=[0.0, 0.36], anchor="x2", title_text="Count"),
        meta={"description": description,
              "metrics": [
                  ("Total reads", f"{n_input:,}"),
                  ("Plasmid size", f"{plasmid_len:,} bp" if plasmid_len else "-"),
                  ("Insert size", f"{insert_len:,} bp" if insert_len else "-"),
                  ("Mean Phred", f"Q{mean_phred:.1f}" if mean_phred is not None else "-"),
                  ("Mean read length", f"{_mean(length_counts):,.0f} bp"
                                       if length_counts else "-"),
              ]})
    return fig


def read_funnel_sankey_figure(funnel, fates=(), insertion_bp: int = 0,
                              deletion_bp: int = 0) -> go.Figure:
    """The funnel on its own, titleless: it reads as the lower half of the read
    summary above it."""
    drawn = _funnel_sankey(funnel, fates, insertion_bp, deletion_bp)
    if drawn is None:
        return _empty("No filtering steps to draw")
    trace, annotations = drawn
    fig = go.Figure(trace)
    fig.update_layout(template=_T, title="", height=600,
                      margin=dict(l=10, r=10, t=16, b=10), annotations=annotations)
    return fig


def _cutoff_line(fig: go.Figure, xref: str, yref: str, at: float, label: str,
                 colour: str, dash: str = "dash", width: float = 1) -> dict:
    """Draw a dashed cutoff across its panel; return the label to go with it."""
    fig.add_shape(type="line", xref=xref, yref=f"{yref} domain", x0=at, x1=at,
                  y0=0, y1=1, line=dict(color=colour, width=width, dash=dash))
    # Just above the bars, in the gap under the panel's own heading.
    return dict(xref=xref, yref=f"{yref} domain", x=at, y=1.01, yanchor="bottom",
                xanchor="right" if label == "max" else "left", text=label,
                showarrow=False, font=dict(size=11, color=colour))


def _panel_title(x: float, text: str, y: float = 1.05) -> dict:
    """Heading over one of the summary's panels."""
    return dict(x=x, y=y, xref="paper", yref="paper", xanchor="center",
                yanchor="bottom", showarrow=False, text=text,
                font=dict(size=13, color=theme.PALETTE["primary_dark"]))




def _insert_marker(fig: go.Figure, start: float, end: float) -> None:
    """Red bar labelled "insert" in the margin above the plot."""
    fig.add_shape(type="rect", xref="x", yref="paper",
                  x0=start, x1=end, y0=1.02, y1=1.05,
                  fillcolor=_RED, line_width=0, layer="above")
    fig.add_annotation(xref="x", yref="paper", x=(start + end) / 2, y=1.06,
                       text="insert", showarrow=False, yanchor="bottom",
                       font=dict(size=11, color=_RED))


# Per-base agreement, as flat bands: match, substitution, deletion, insertion.
_MATCH_GREY = "#B4BAC1"
_SUBSTITUTION_AMBER = "#E0B252"
_DELETION_RED = _RED
_INSERTION_BLUE = "#3A6FB0"
# Flat bands over zmin=0.5..zmax=8.5, so every agreement code lands mid-band;
# the substituted-base codes 5-8 all read amber.
_AGREEMENT_SCALE = [[0.000, _MATCH_GREY], [0.125, _MATCH_GREY],
                    [0.125, _SUBSTITUTION_AMBER], [0.250, _SUBSTITUTION_AMBER],
                    [0.250, _DELETION_RED], [0.375, _DELETION_RED],
                    [0.375, _INSERTION_BLUE], [0.500, _INSERTION_BLUE],
                    [0.500, _SUBSTITUTION_AMBER], [1.000, _SUBSTITUTION_AMBER]]
# Agreement code -> the read's base, and byte -> letter for the reference.
_SUB_LETTER = np.array(["?"] * 9)
for _base, _code in SUBSTITUTION_CODES.items():
    _SUB_LETTER[_code] = _base
_BASE_LETTER = np.full(256, "?", dtype="<U1")
for _base in "ACGTN":
    _BASE_LETTER[ord(_base)] = _base
# A base is one pixel over a whole plasmid, so every departure also gets a
# square marker; otherwise the map reads as uniform grey.
_EVENT_MARKER = dict(size=3, symbol="square", line=dict(width=0))
# A deletion under this width gets a marker per base, so it cannot vanish
# between pixels; a wider one is a visible block already and carries this many
# markers, purely to keep its size reachable on hover.
_DENSE_DELETION, _MARKS_PER_DELETION = 200, 6


def match_percent(cov: ReadMap) -> np.ndarray:
    """Percentage of covering reads matching the reference at each base, ``nan``
    where nothing covers it. The read alignment map's trace and the plasmid
    map's match panel are the same numbers, taken from here."""
    depth = cov.depth_counts.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(depth > 0, 100.0 * cov.match_counts / depth, np.nan)


def read_alignment_figure(cov: ReadMap, insert_seq: str | None = None,
                          insertion_bp: int = 0, deletion_bp: int = 0) -> go.Figure:
    """One row per read, coloured base by base against the reference: grey
    match, amber substitution, red deletion, blue on the base an insertion
    begins before, blank where the read does not reach. Each departure also gets
    a square marker so it survives being one pixel wide, and indels below their
    direction's threshold are dropped as basecall noise. Rows sort by start
    position, so the pileup reads as a staircase and a shared change lines up as
    a vertical stripe; above it runs the match trace and the insert's bands."""
    ends, matrix = cov.read_ends, cov.match_matrix
    # mapped_reads counts every read; the arrays may hold an even sample of them.
    if max(int(cov.mapped_reads), int(cov.read_starts.shape[0])) == 0 or matrix.size == 0:
        return _empty("No reads aligned to the plasmid")

    order = np.lexsort((ends, cov.read_starts))
    ends, matrix = ends[order], matrix[order]
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
    width = matrix.shape[1]
    for read, at, length in zip(cov.deletion_rows[~del_keep],
                                cov.deletion_positions[~del_keep],
                                cov.deletion_lengths[~del_keep]):
        if read < order.size:                # a deletion may run past the origin
            z[rank[read], (np.arange(at - 1, at - 1 + length) % width)] = np.nan

    x_max = cov.contig_length or int(ends.max())
    _T_MARGIN, _B_MARGIN = 60, 48
    height = int(min(880, max(340, n_shown + _T_MARGIN + _B_MARGIN + 230)))
    positions = np.arange(1, matrix.shape[1] + 1)

    depth = cov.depth_counts.astype(float)
    pct = match_percent(cov)

    # The insert's amino acids over its own span, then the whole reference's
    # bases, both on the plasmid coordinate.
    tracks: list[tuple[str, go.Bar]] = []
    if insert_seq and cov.region:
        reverse = cov.region.get("strand") == "-"
        aa = _aa_track(insert_seq, len(insert_seq), 0, yaxis=None, number_labels=True,
                       origin=cov.region["end"] if reverse else cov.region["start"],
                       step=-1 if reverse else 1)
        if aa is not None:
            tracks.append(("AA", aa))
    bases = _base_track(cov.contig_seq, len(cov.contig_seq), yaxis=None)
    if bases is not None:
        tracks.append(("bp", bases))

    rows = len(tracks) + 2
    heights = {2: [0.045, 0.04, 0.10, 0.815], 1: [0.05, 0.10, 0.85]}.get(
        len(tracks), [0.10, 0.90])
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=heights,
                        vertical_spacing=0.025)
    trace_row, map_row = len(tracks) + 1, len(tracks) + 2
    for i, (_, track) in enumerate(tracks, start=1):
        fig.add_trace(track, row=i, col=1)
    fig.add_trace(go.Scatter(
        x=positions, y=pct, mode="lines", name="Match",
        line=dict(color=theme.PALETTE["primary"], width=1.4),
        fill="tozeroy", fillcolor="rgba(31, 111, 139, 0.12)",
        customdata=depth,
        hovertemplate="Position %{x}<br>%{y:.1f}% of %{customdata:.0f} reads match"
                      "<extra></extra>"), row=trace_row, col=1)
    fig.add_trace(go.Heatmap(
        z=z, x=positions, showscale=False, zmin=0.5, zmax=8.5,
        colorscale=_AGREEMENT_SCALE,
        hovertemplate="Position %{x}<br>read row %{y}<extra></extra>"),
        row=map_row, col=1)

    # A marker per event, so no departure is lost to being one pixel wide.
    def events(x, y, colour: str, customdata, hover: str) -> None:
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers", showlegend=False, customdata=customdata,
            marker=dict(color=colour, **_EVENT_MARKER), hovertemplate=hover),
            row=map_row, col=1)

    # Substitutions come from the rendered z, not the raw matrix: an insertion
    # repaints the cell it begins before, which must not also claim one.
    sub_rows, sub_cols = np.nonzero((z == 2) | (z >= 5))
    if sub_rows.size:
        # Each marker names the change: reference base -> the read's own base.
        alt = _SUB_LETTER[z[sub_rows, sub_cols].astype(int)]
        ref_bases = np.frombuffer(cov.contig_seq.upper().encode(), dtype=np.uint8)
        events(sub_cols + 1, sub_rows, _SUBSTITUTION_AMBER,
               np.char.add(np.char.add(_BASE_LETTER[ref_bases[sub_cols]], "→"), alt)
               if ref_bases.size >= matrix.shape[1] else alt,
               "Substitution %{customdata} at position %{x}<extra></extra>")

    if del_keep.any():
        # Sampled along each deletion, and kept where the cell still reads as
        # deleted, so every marker carries its own deletion's length.
        at, length = cov.deletion_positions[del_keep], cov.deletion_lengths[del_keep]
        stride = np.where(length <= _DENSE_DELETION, 1,
                          np.maximum(1, length // _MARKS_PER_DELETION))
        cols = np.concatenate([np.arange(a, a + n, s)
                               for a, n, s in zip(at, length, stride)])
        cols = (cols - 1) % width + 1        # the plasmid is circular
        repeats = -(-length // stride)
        del_rows = np.repeat(rank[cov.deletion_rows[del_keep]], repeats)
        del_lens = np.repeat(length, repeats)
        keep = z[del_rows, cols - 1] == 3
        if keep.any():
            events(cols[keep], del_rows[keep], _DELETION_RED, del_lens[keep],
                   "Deletion of %{customdata:,} bp<br>at position %{x}<extra></extra>")

    if ins_keep.any():
        events(cov.insertion_positions[ins_keep], rank[cov.insertion_rows[ins_keep]],
               _INSERTION_BLUE, cov.insertion_lengths[ins_keep],
               "Insertion of %{customdata:,} bp<br>before position %{x}<extra></extra>")

    if cov.region and not tracks:
        _insert_marker(fig, cov.region["start"], cov.region["end"])

    fig.update_layout(
        template=_T, title="", height=height, showlegend=False,
        bargap=0, uniformtext=dict(minsize=5, mode="hide"),
        margin=dict(l=60, r=48, t=_T_MARGIN + 12, b=_B_MARGIN),
        meta={"subtitle": "Every read, base by base",
              "description":
              "One row per read - an even sample of them where the pileup is deep "
              "- coloured against the reference: grey matches, amber a "
              "substitution, red a deletion, blue the base an insertion begins "
              "before, and blank where the read does not reach. Rows sort by start "
              "position, so a change many reads share lines up as a vertical "
              "stripe. Above the map run the match trace and the reference's "
              "amino acids and bases."})
    for i, (title, _) in enumerate(tracks, start=1):
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


# --- Circular plasmid map ----------------------------------------------------
_RING_R = 1.0                        # backbone radius
_TRACE_R0, _TRACE_R1 = 1.20, 1.95    # 0% and 100% on the outer trace


def _theta(positions, contig_length: int) -> np.ndarray:
    """Plasmid coordinates (1-based) as degrees; the axis puts 1 at the top."""
    return (np.asarray(positions, dtype=float) - 1.0) * 360.0 / contig_length


def _arc(start: float, end: float, contig_length: int) -> np.ndarray:
    """Degrees along the arc start->end, clockwise through the origin if it wraps."""
    span = (end - start) % contig_length * 360.0 / contig_length
    return _theta(start, contig_length) + np.linspace(0, span, max(int(span * 2), 2))


def _ring_panel(fig: go.Figure, cov: ReadMap, numerator: np.ndarray,
                denominator: np.ndarray, *, subplot: str, colour: str, hover: str,
                title: str, value_scale: float = 100.0, scale_max: float = 100.0,
                cutoff: float | None = None,
                cutoff_label: str = "", max_points: int = 20000) -> None:
    """One circular panel: backbone ring, the insert arc on it, and
    ``value_scale * numerator/denominator`` traced round the outside, with
    ``scale_max`` at the outermost ring - 100 and 100 for a percentage. Only a
    reference past ``max_points`` is binned, and binned before dividing, so the
    values stay true to the counts. ``hover`` templates customdata: position,
    value, numerator, denominator, each per base."""
    num, den = numerator.astype(float), denominator.astype(float)
    n = int(cov.contig_length or num.size)
    pos = np.arange(1, num.size + 1, dtype=float)
    per_base = np.ones(num.size)
    if num.size > max_points:
        bins = (np.arange(num.size) * max_points) // num.size
        per_base = np.bincount(bins).astype(float)
        pos = np.bincount(bins, pos) / per_base
        num, den = np.bincount(bins, num), np.bincount(bins, den)
    with np.errstate(invalid="ignore", divide="ignore"):
        value = np.where(den > 0, value_scale * num / den, np.nan)

    # Close the ring by repeating the first sample one turn on.
    wrap = lambda a: np.append(a, a[:1])
    theta = np.append(_theta(pos, n), _theta(pos[:1], n) + 360.0)
    value, pos, num, den, per_base = map(wrap, (value, pos, num, den, per_base))
    radius = _TRACE_R0 + (_TRACE_R1 - _TRACE_R0) * np.nan_to_num(value) / scale_max

    # The panel's name sits at r=0, the middle of its circle.
    fig.add_trace(go.Scatterpolar(
        theta=[0], r=[0], mode="text", subplot=subplot, hoverinfo="skip",
        text=[f"<b>{title}</b>"], textfont=dict(size=13, color=colour)))
    fig.add_trace(go.Scatterpolar(                       # backbone
        theta=np.linspace(0, 360, 361), r=np.full(361, _RING_R), mode="lines",
        subplot=subplot, line=dict(color=_MATCH_GREY, width=8), hoverinfo="skip"))
    if cov.region:
        start, end = cov.region["start"], cov.region["end"]
        arc = _arc(start, end, n)
        fig.add_trace(go.Scatterpolar(
            theta=arc, r=np.full(arc.size, _RING_R), mode="lines", subplot=subplot,
            line=dict(color=_RED, width=12),
            hovertemplate=f"Insert {start:,}-{end:,} bp "
                          f"({cov.region.get('strand', '+')} strand)<extra></extra>"))

    # The fill drops to the baseline where there is nothing to divide, so the
    # band stays one polygon; the line above it keeps the gap.
    fig.add_trace(go.Scatterpolar(
        theta=np.concatenate([theta, theta[::-1]]),
        r=np.concatenate([radius, np.full(theta.size, _TRACE_R0)]),
        mode="lines", subplot=subplot, fill="toself", fillcolor=_rgba(colour, 0.12),
        line=dict(width=0), hoverinfo="skip"))
    fig.add_trace(go.Scatterpolar(
        theta=theta, r=np.where(np.isfinite(value), radius, np.nan), mode="lines",
        subplot=subplot, line=dict(color=colour, width=1.4),
        customdata=np.stack([pos, value, num / per_base, den / per_base], axis=1),
        hovertemplate=hover + "<extra></extra>"))
    if cutoff is not None:
        r_cut = _TRACE_R0 + (_TRACE_R1 - _TRACE_R0) * cutoff / scale_max
        fig.add_trace(go.Scatterpolar(
            theta=np.linspace(0, 360, 181), r=np.full(181, r_cut), mode="lines",
            subplot=subplot, line=dict(color=theme.PALETTE["muted"], width=1.2,
                                       dash="3px,3px"),
            hovertemplate=f"{cutoff_label} ({cutoff:g}%)<extra></extra>"))


def _polar_axes(contig_length: int, domain: list[float],
                radial_labels: tuple[str, str, str] = ("0%", "50%", "100%")) -> dict:
    """Polar axes for a panel: bp clockwise from the top, the panel's own scale
    outwards."""
    raw = contig_length / 4                    # roughly four round-number bp labels
    mag = 10 ** int(math.floor(math.log10(raw)))
    step = max(next(m * mag for m in (1, 2, 5, 10) if m * mag >= raw), 1)
    ticks = np.arange(0, contig_length, step, dtype=int)
    muted = dict(size=10, color=theme.PALETTE["muted"])
    return dict(
        domain=dict(x=domain), bgcolor=theme.PALETTE["bg"],
        radialaxis=dict(range=[0, _TRACE_R1 + 0.10], angle=45, tickangle=45,
                        tickmode="array", ticks="",
                        tickvals=[_TRACE_R0, (_TRACE_R0 + _TRACE_R1) / 2, _TRACE_R1],
                        ticktext=list(radial_labels), tickfont=muted,
                        gridcolor=theme.PALETTE["grid"], linecolor=GHOST),
        angularaxis=dict(direction="clockwise", rotation=90, tickmode="array",
                         tickvals=_theta(ticks + 1, contig_length),
                         ticktext=[f"{t:,}" for t in ticks], tickfont=muted,
                         gridcolor=GHOST, linecolor=GHOST))


def plasmid_map_figure(cov: ReadMap, auto_match_threshold: float | None = None
                       ) -> go.Figure:
    """The plasmid drawn three times as the circle it is: how many reads reach
    each base, how well basecalled they are there, and how well they match the
    reference - the read alignment map's trace wrapped round. All three carry
    the backbone ring with the insert marked."""
    depth, covered = cov.depth_counts, cov.covered_counts
    n = int(cov.contig_length or depth.size)
    if n == 0 or depth.size == 0:
        return _empty("No reads aligned to the plasmid", "Read Alignment to the Plasmid")
    reads = int(cov.mapped_reads)
    # Without per-base quality (a BAM with no quality strings) the panel simply
    # has nothing to draw.
    sums = (cov.quality_sums.astype(float) if cov.quality_sums.size == covered.size
            else np.full(covered.size, np.nan))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_q = np.where(covered > 0, sums / np.maximum(covered, 1), np.nan)
    # The quality ring carries its own scale, rounded up to a whole 10 Q.
    q_top = (max(10.0 * math.ceil(np.nanmax(mean_q) / 10.0), 10.0)
             if np.isfinite(mean_q).any() else 10.0)

    match_colour, cover_colour = theme.PALETTE["primary"], theme.PALETTE["accent"]
    qual_colour = "#7D6FB0"
    fig = go.Figure()
    # Depth, not held bases: a read whose alignment deletes a base still spans
    # it, and the deletion is the read's verdict on that base, not a blind spot.
    _ring_panel(fig, cov, depth, np.full(depth.size, reads),
                subplot="polar", colour=cover_colour, title="Read<br>coverage",
                hover="Position %{customdata[0]:,.0f} bp<br>"
                      "%{customdata[1]:.1f}% of reads cover it "
                      "(%{customdata[2]:,.0f} of %{customdata[3]:,.0f})")
    _ring_panel(fig, cov, sums, covered, subplot="polar2",
                colour=qual_colour, title="Mean<br>quality", value_scale=1.0,
                scale_max=q_top,
                hover="Position %{customdata[0]:,.0f} bp<br>"
                      "mean Q%{customdata[1]:.1f} over %{customdata[3]:,.0f} reads")
    _ring_panel(fig, cov, cov.match_counts, depth, subplot="polar3",
                colour=match_colour, title="Match to<br>reference",
                cutoff=auto_match_threshold, cutoff_label="Codon detection cutoff",
                hover="Position %{customdata[0]:,.0f} bp<br>"
                      "%{customdata[1]:.1f}% of %{customdata[3]:,.0f} reads match")

    fig.update_layout(
        template=_T, title="Read Alignment to the Plasmid", height=400, showlegend=False,
        margin=dict(l=52, r=52, t=60, b=36),   # room for the bp labels either side
        polar=_polar_axes(n, [0.0, 0.28]),
        polar2=_polar_axes(n, [0.36, 0.64], ("Q0", f"Q{q_top / 2:g}", f"Q{q_top:g}")),
        polar3=_polar_axes(n, [0.72, 1.0]),
        annotations=[
            dict(x=0.5, y=-0.04, xref="paper", yref="paper", showarrow=False,
                 text=f"{n:,} bp plasmid",
                 font=dict(size=11, color=theme.PALETTE["muted"]))],
        meta={"subtitle": "Coverage, quality and match around the plasmid",
              "description":
              f"The plasmid as the circle it is, three times over {reads:,} mapped "
              "reads: how many of them reach each base (a base a read deletes "
              "counts as reached), how well basecalled they are there, and how "
              "many of the covering reads match the reference. "
              "On all three the grey ring is the backbone, the red arc the insert, "
              "and position runs clockwise from the top."})
    return fig


# The funnel's end states: the library's product saturated, defects warm.
FATE_COLORS: dict[str, str] = {
    "Variant": "#1F6F5C",
    "Ambiguous diversity": "#9FB6C4",
    "Wild type": "#E8913C",
    # The same red and blue the read-by-read map uses for the two defects.
    "Deletion": _DELETION_RED,
    "Insertion": _INSERTION_BLUE,
}


def _letter_track(x, letters, colors, *, width: float, size: int,
                  yaxis: str | None, hovers: list[str] | None = None) -> go.Bar:
    """A band of coloured blocks, one per position, lettered where one fits."""
    bar = go.Bar(x=x, y=[1] * len(x), width=width, showlegend=False,
                 marker=dict(color=colors, line=dict(width=0)),
                 text=letters, textposition="inside", insidetextanchor="middle",
                 textangle=0,   # never let Plotly rotate letters in narrow blocks
                 textfont=dict(family="monospace", size=size,
                               color=[theme.contrast_text(c) for c in colors]),
                 hovertext=hovers, hoverinfo="text" if hovers else "skip")
    return bar.update(yaxis=yaxis) if yaxis else bar


def _aa_track(ref_seq: str, length: int, frame_offset: int, *,
              yaxis: str | None = "y2", origin: int = 1, step: int = 1,
              number_labels: bool = False) -> go.Bar | None:
    """Reference amino acid of each codon as a coloured band; letters show
    wherever they fit. ``origin``/``step`` place the first codon's first base
    (``step=-1`` for a reverse-strand insert); ``number_labels`` numbers them."""
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
    return _letter_track(centres, letters, colors, width=3, size=11, yaxis=yaxis,
                         hovers=hovers if number_labels else None)


def _base_track(ref_seq: str, length: int, yaxis: str | None = "y3", *,
                origin: int = 1, step: int = 1) -> go.Bar | None:
    """Reference base at each position as a coloured band; letters show wherever
    they fit. ``origin``/``step`` place the band elsewhere on the x axis, as for
    :func:`_aa_track`."""
    bases = (ref_seq or "").upper()[:length]
    if not bases:
        return None
    colors = [theme.BASE_COLORS.get(b, theme.PALETTE["muted"]) for b in bases]
    return _letter_track([origin + step * i for i in range(len(bases))],
                         list(bases), colors, width=1, size=9, yaxis=yaxis)


def _gap_match_metrics(n_reads, threshold, user_positions, shade_codons):
    """Cards for the match track: reads behind it, how its codons were picked,
    and how many came out."""
    named = list(user_positions or [])
    if threshold is not None:
        picked = (f"< {threshold:g}% match"
                  + (f" + codons {_codon_list(named)}" if named else ""))
    else:
        picked = f"Codons {_codon_list(named)}" if named else "Off"
    return [("Reads used", f"{n_reads:,}" if n_reads is not None else "-"),
            ("Codon detection", picked),
            ("Variant positions", f"{len(shade_codons or ()):,}")]


def _codon_list(positions, limit: int = 6) -> str:
    """User-named codons, trimmed once the list stops fitting on a card."""
    shown = ", ".join(str(p) for p in positions[:limit])
    return shown + (f" +{len(positions) - limit} more" if len(positions) > limit else "")


def gap_match_figure(df_counts: pd.DataFrame, ref_seq: str, *, gap_char: str = "-",
                     shade_codons=None, frame_offset: int = 0,
                     auto_match_threshold: float | None = None,
                     aa_counts: pd.DataFrame | None = None,
                     n_reads: int | None = None,
                     user_positions=None) -> go.Figure:
    """Per-position reference-match % (gaps excluded from the denominator),
    optionally shading codons and marking the auto-detect cutoff. A band above
    gives each codon's reference amino acid; with ``aa_counts``, the hover also
    lists AA frequencies at that codon. ``n_reads`` and ``user_positions`` (the
    codons the user named) only feed the metric cards."""
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

    fig.update_layout(
        template=_T, hovermode="x unified",
        title="Library Diversity and Coverage",
        # Pinned to the bases themselves, so the trace and the bands below it
        # run to both edges instead of floating inside a padded axis.
        xaxis=dict(title_text="Position in reference (1-based, nucleotides)",
                   range=[0.5, L + 0.5]),
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
        meta={"subtitle": "Reference match along the insert",
              "metrics": _gap_match_metrics(n_reads, auto_match_threshold,
                                            user_positions, shade_codons),
              "description":
              "How often each insert base matches the reference, gaps excluded "
              "from the denominator: a position the library diversifies drops "
              "away from 100%, and anything under the dashed cutoff is taken as "
              "a variable codon. The bands above give each codon's reference "
              "amino acid, coloured by biochemical group, and its reference "
              "base."},
    )
    return fig


def aa_pies_figure(df_aa_counts: pd.DataFrame, positions, *,
                   ref_seq: str | None = None,
                   min_frac: float = 0.01) -> go.Figure | None:
    """Donut per codon position; slices below ``min_frac`` fold into 'Other'.
    The middle carries the reference residue and the reads behind the donut."""
    pos_ok = [p for p in positions if p in df_aa_counts.index]
    if not pos_ok:
        return None
    n = len(pos_ok)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    height = rows * 300
    # Plotly expresses subplot spacing as a fraction of total figure height.
    # Keep the actual row gap near-constant so large auto-detected sets do not
    # violate Plotly's max spacing limit or waste vertical space.
    vertical_spacing = 0.0 if rows == 1 else min(0.08, 28 / height)
    up = (ref_seq or "").upper()
    wild_type = {p: GENETIC_CODE.get(up[(p - 1) * 3:(p - 1) * 3 + 3]) for p in pos_ok}
    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{"type": "domain"} for _ in range(cols)] for _ in range(rows)],
        subplot_titles=[f"Codon {p}" for p in pos_ok],
        horizontal_spacing=0.035, vertical_spacing=vertical_spacing,
    )

    k = 0
    middles: list[str] = []
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            if k >= n:
                # Invisible pie keeps empty cells the same size as filled ones.
                fig.add_trace(go.Pie(labels=[""], values=[1], hole=0.5, opacity=0,
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
            aa = wild_type.get(pos)
            middles.append(
                (f"<span style='color:{theme.AA_COLORS.get(aa, theme.PALETTE['muted'])}'>"
                 f"<b>WT {aa}</b></span><br>" if aa else "") + f"n={int(total):,}")
            fig.add_trace(go.Pie(
                labels=labels, values=values, hole=0.5, sort=False,
                customdata=[_AA_NAMES.get(label, label) for label in labels],
                marker=dict(
                    colors=theme.aa_color_sequence(labels),
                    line=dict(color="#000000", width=1),
                ),
                textinfo="label+percent", textposition="inside",
                hovertemplate="%{customdata}<br>Count %{value:.0f}<br>"
                              "%{percent:.2%}<extra></extra>",
                showlegend=False,
            ), row=r, col=c)

    # Reference residue and read count in the middle of each real donut.
    drawn = (t for t in fig.data
             if isinstance(t, go.Pie) and t.opacity != 0 and t.labels != ("",))
    for trace, middle in zip(drawn, middles):
        xd, yd = trace.domain["x"], trace.domain["y"]
        fig.add_annotation(x=(xd[0] + xd[1]) / 2, y=(yd[0] + yd[1]) / 2,
                           xref="paper", yref="paper", showarrow=False,
                           xanchor="center", yanchor="middle", align="center",
                           text=middle, font=dict(size=12,
                                                  color=theme.PALETTE["muted"]))

    fig.update_layout(
        template=_T, title="", showlegend=False,
        height=height, width=cols * 300,
        uniformtext_minsize=10, uniformtext_mode="hide",
        meta={"subtitle": "Amino acids at each variable codon",
              "description":
              "One donut per codon the analysis calls variable: each slice is an "
              "amino acid's share of the reads there, with residues below the "
              "grouping threshold folded into 'Other'. The middle names the "
              "reference residue and the reads behind that codon."},
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


def _drop_rare_variants(hap_df: pd.DataFrame, aa_counts: pd.DataFrame | None,
                        positions, min_frac: float) -> pd.DataFrame:
    """Variants carrying an amino acid below ``min_frac`` at its codon leave the
    library view entirely, so every variant plot describes the same subset."""
    rare = _rare_aa_by_position(aa_counts, positions, min_frac)
    if not rare:
        return hap_df
    def has_rare(combo) -> bool:
        return isinstance(combo, tuple) and any(
            aa in rare.get(pos, ()) for pos, aa in zip(positions, combo))
    return hap_df[~hap_df["combo_tuple"].map(has_rare)]


def _gini(counts) -> float:
    """Gini coefficient of a count distribution: 0 even, 1 dominated by one variant."""
    x = np.sort(np.asarray(counts, dtype=float))
    total = x.sum()
    if len(x) < 2 or total <= 0:
        return 0.0
    i = np.arange(1, len(x) + 1)
    return float((2 * (i * x).sum()) / (len(x) * total) - (len(x) + 1) / len(x))


def haplotype_treemap_figure(hap_df: pd.DataFrame, *,
                             top_n: int | None = None, aa_counts: pd.DataFrame | None = None,
                             positions=None, min_frac: float = 0.0) -> go.Figure:
    """Treemap of haplotypes sized by count; ``top_n`` folds the rest into
    'Other'. Variants carrying an amino acid below ``min_frac`` at its codon are
    excluded (0 keeps everything), as are reads not called at every codon."""
    if hap_df.empty:
        return _empty("No haplotypes to display")

    hap_df = _drop_rare_variants(hap_df, aa_counts, positions, min_frac)
    if hap_df.empty:
        return _empty("No variants remain above the grouping threshold")

    df = hap_df[["combo_label", "count"]].copy()
    df["mutations"] = hap_df.get("mutations", "")
    df["is_reference"] = hap_df.get("is_reference", False)
    df["aa_hamming_distance"] = hap_df.get("aa_hamming_distance", pd.NA)
    df = df.sort_values("count", ascending=False).reset_index(drop=True)

    if top_n is not None and len(df) > top_n:
        other = pd.DataFrame([{"combo_label": "Other", "mutations": "several",
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

    # Tiles name the mutations, as the hover does; "Other" and uncalled
    # references keep the positional label.
    tiles = df_plot["mutations"].fillna("").astype(str)
    tiles = tiles.where(tiles.ne("") & df_plot["combo_label"].ne("Other"),
                        df_plot["combo_label"])

    fig = go.Figure(go.Treemap(
        ids=df_plot["combo_label"], labels=tiles, parents=[""] * len(df_plot),
        values=df_plot["count"], branchvalues="total",
        customdata=list(zip(status, distances,
                            df_plot["mutations"].fillna("").astype(str))),
        marker=dict(colors=colors, line=dict(color=line_colors, width=line_widths)),
        textinfo="label+percent entry",
        hovertemplate=("<b>%{customdata[2]}</b><br>Hamming distance to WT: "
                       "%{customdata[1]}<br>Reads %{value}"
                       "<br>%{percentRoot:.1%}<extra></extra>"),
    ))
    reads = int(df["count"].sum())
    by_distance = (df.dropna(subset=["aa_hamming_distance"])
                     .groupby("aa_hamming_distance")["count"].sum())
    fig.update_layout(
        template=_T, height=500, title="",
        meta={"subtitle": "Variant combinations by abundance",
              "metrics": [("Unique variants", f"{len(df):,}"),
                          ("Fully called reads", f"{reads:,}"),
                          ("Gini coefficient", f"{_gini(df['count']):.2f}"),
                          ("Most common mutation count",
                           f"{int(by_distance.idxmax())}" if not by_distance.empty
                           else "-")],
              "description":
              "Every unique combination of amino acids across the variable "
              "codons, each tile sized by the reads carrying it, so a library "
              "dominated by a few clones shows a few large tiles. The reference "
              "combination is outlined in black."})
    return fig


# Column-major over two columns; this order splits 10/10, so no group straddles.
_GROUP_ORDER = ("Hydrophobic", "Acidic", "Hydrophilic", "Basic", "Special")
_AA_BY_GROUP = [aa for group in _GROUP_ORDER
                for aa in theme.AA_ORDER if theme.AA_GROUPS.get(aa) == group]


def variant_panels_figure(hap_df: pd.DataFrame, codon_matrix: np.ndarray | None,
                          positions, *, aa_counts: pd.DataFrame | None = None,
                          min_frac: float = 0.0, replicates: int = 8,
                          points: int = 120) -> go.Figure | None:
    """The library's variants two ways, in one row: how far each sits from the
    reference, and whether the sequencing has seen them all. A panel is left out
    when its data cannot say anything."""
    df = _drop_rare_variants(hap_df, aa_counts, positions, min_frac)
    distances = pd.to_numeric(df["aa_hamming_distance"], errors="coerce") \
        if not df.empty else None
    graded = distances is not None and distances.notna().any()
    sampled = codon_matrix is not None and codon_matrix.shape[0] >= 2
    titles = [t for t, ok in (("Frequency of Hamming Distance to WT", graded),
                              ("Unique variants seen", sampled)) if ok]
    if not titles:
        return None
    at = {title: i + 1 for i, title in enumerate(titles)}
    fig = make_subplots(rows=1, cols=len(titles), subplot_titles=titles,
                        horizontal_spacing=0.07)

    if graded:
        col = at["Frequency of Hamming Distance to WT"]
        counts = df["count"].to_numpy(dtype=float)
        per_step = (pd.DataFrame({"changes": distances, "count": counts}).dropna()
                    .groupby("changes").agg(variants=("count", "size"),
                                            reads=("count", "sum")))
        fig.add_trace(go.Bar(
            x=per_step.index.astype(int), y=per_step["variants"], showlegend=False,
            # The section's own teal, with wild type in the funnel's amber.
            marker_color=[FATE_COLORS["Wild type"] if d == 0
                          else theme.PALETTE["primary"] for d in per_step.index],
            customdata=100.0 * per_step["reads"].to_numpy() / counts.sum(),
            hovertemplate="%{x} change(s) from the reference<br>%{y:,} variants, "
                          "%{customdata:.1f}% of reads<extra></extra>"), row=1, col=col)
        fig.update_xaxes(title_text="Amino-acid changes", dtick=1, row=1, col=col)
        fig.update_yaxes(title_text="Number of Unique Variants", row=1, col=col)

    if sampled:
        col = at["Unique variants seen"]
        n = codon_matrix.shape[0]
        ids = np.unique(codon_matrix, axis=0, return_inverse=True)[1]
        xs = np.unique(np.linspace(1, n, points).astype(int))
        # A variant is new at the first read carrying it, so the curve is a
        # lookup into the sorted first-occurrence positions of one shuffle.
        rng = np.random.default_rng(0)
        curves = np.array([
            np.searchsorted(
                np.sort(np.unique(ids[rng.permutation(n)], return_index=True)[1]),
                xs, side="left")
            for _ in range(replicates)], dtype=float)
        mean = curves.mean(axis=0)
        # Upper edge first, then the lower one filling back to it.
        for edge, fill in ((curves.max(axis=0), None), (curves.min(axis=0), "tonexty")):
            fig.add_trace(go.Scatter(
                x=xs, y=edge, mode="lines", line=dict(width=0), hoverinfo="skip",
                showlegend=False, fill=fill,
                fillcolor=_rgba(theme.PALETTE["primary"], 0.14)), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=[0, n], y=[0, n], mode="lines", showlegend=False,
            line=dict(color=theme.PALETTE["muted"], width=1, dash="3px,3px"),
            hovertemplate="Every read a new variant<extra></extra>"), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=xs, y=mean, mode="lines", showlegend=False,
            line=dict(color=theme.PALETTE["primary"], width=1.8),
            customdata=100.0 * mean / max(mean[-1], 1.0),
            hovertemplate="%{x:,} reads<br>%{y:,.0f} variants "
                          "(%{customdata:.1f}% of all seen)<extra></extra>"),
            row=1, col=col)
        # Both axes run 0..n, so the diagonal spans the panel corner to corner
        # however wide it is drawn.
        fig.update_xaxes(title_text="Reads sampled", range=[0, n], row=1, col=col)
        fig.update_yaxes(title_text="Unique variants", range=[0, n], row=1, col=col)

    fig.update_layout(
        template=_T, title="", height=400, bargap=0.25,
        margin=dict(l=58, r=24, t=62, b=48),
        meta={"subtitle": "Variant spread and sampling",
              "description":
              "Unique variants by how many amino-acid changes they carry from the "
              "reference, wild type in amber; the variants found as reads are "
              "sampled, over eight shuffles, against the diagonal where every "
              "read is new."})
    return fig


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
    """``(codon_heads, aa_blocks)`` per nucleotide, frame-aligned to
    :func:`libraont.analysis.detect_variable_codons`: the codon number, and its
    two-column AA frequency table (``None`` without ``aa_counts``)."""
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


def _rgba(hex_colour: str, alpha: float) -> str:
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


_DIVERSITY_HOVER = {
    "Variant": "carry a change at one or more diversified codons",
    "Wild type": "cover every diversified codon and carry no change at any of them",
    "Ambiguous diversity": "carry no detected change, but do not cover every "
                           "diversified codon",
}
def _funnel_sankey(funnel, fates=(), insertion_bp: int = 0, deletion_bp: int = 0,
                   band: tuple[float, float] = (0.13, 0.80)) -> tuple | None:
    """The funnel as (trace, annotations) in the paper band ``band``; ``None``
    when there is nothing to draw. A teal trunk narrows along the top with each
    step's losses curving away below it, except the verdicts on a read that
    reached the insert, which all run to the right-hand edge to be read off
    together. Empty branches keep a hairline and a label."""
    stages = [s for s in funnel if s.count is not None]
    if len(stages) < 2:
        return None
    pal = theme.PALETTE
    last = next((i for i, s in enumerate(stages) if s.label == "Correctly assembled"),
                len(stages) - 1)
    stages = stages[:last + 1]
    counts = {f.label: f.count for f in fates}
    hover_for = dict(_DIVERSITY_HOVER,
                     Deletion=f"carry a deletion ≥ {deletion_bp} bp, or no "
                              "insert at all",
                     Insertion=f"carry an insertion ≥ {insertion_bp} bp")
    # Losses from this step are verdicts on a read that reached the insert, so
    # they run to the edge instead of stopping under their own column.
    to_edge = {"Correctly assembled": [(k, counts.get(k, 0), hover_for[k])
                                       for k in ("Deletion", "Insertion")]}

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
            tail.append([(fate, count, note, i) for fate, count, note in to_edge[b.label]])
            continue
        src.append(i)
        tgt.append(len(xs))
        value.append(lost[i])
        hover.append(note)
        # Grey, not red: red is a defect in the library (see ``FATE_COLORS``),
        # while these reads are simply set aside before it can be judged.
        colors.append(_rgba(pal["muted"], 0.36))
        drops.append((xs[i + 1], b.lost or "removed", gone))
        xs.append(xs[i + 1])
        ys.append(floor - 0.5 * lost[i] / depth)
        tints.append(pal["muted"])
        caps.append(note)

    # Everything that ends the story lands in one column, the trunk's own
    # continuation on top and each verdict stacked below it in the order it left.
    # One node per fate, however many steps feed it.
    edge: dict[str, list] = {label: [(n - 1, counts[label], hover_for[label])]
                             for label in ("Variant", "Wild type", "Ambiguous diversity")
                             if label in counts}
    for group in reversed(tail):
        for fate, count, note, source in group:
            edge.setdefault(fate, []).append((source, count, note))
    totals = [sum(c for _, c, _ in links) for links in edge.values()]
    ends = [max(t, hair) for t in totals]
    splay = min(0.13, (0.95 - crest - sum(ends) / depth) / max(1, len(edge) - 1))
    fan, top = len(xs), crest
    for j, (fate, links) in enumerate(edge.items()):
        tint = FATE_COLORS[fate]
        for source, count, note in links:
            src.append(source)
            tgt.append(fan + j)
            value.append(max(count, hair))
            hover.append(f"<b>{count:,} reads</b> {note}")
            colors.append(_rgba(tint, 0.42))
        xs.append(0.02 + right + 0.05)
        ys.append(top + 0.5 * ends[j] / depth)
        tints.append(tint)
        caps.append(f"<b>{fate}</b><br>{totals[j]:,} reads")
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
    lo, hi = band
    notes = [dict(x=xs[i], y=hi + 0.015, yanchor="bottom",
                  xanchor="left" if i == 0 else "right" if i == n - 1 else "center",
                  text=f"<b>{'<br>'.join(textwrap.wrap(s.label, 16))}</b><br>"
                       f"<span style='color:{pal['muted']}'>{s.count:,} · "
                       f"{s.count / start:.0%}</span>",
                  font=dict(size=11, color=pal["primary_dark"]))
             for i, s in enumerate(stages)]
    notes += [dict(x=x, y=lo - 0.015, yanchor="top", xanchor="center",
                   text=f"{'<br>'.join(textwrap.wrap(label, 18))}<br><b>{gone:,}</b>",
                   font=dict(size=10.5, color=pal["muted"]))
              for x, label, gone in drops]
    notes += [dict(x=xs[fan + j] + 0.012, y=hi - (hi - lo) * ys[fan + j],
                   xanchor="left", yanchor="middle",
                   text=f"<b>{'<br>'.join(textwrap.wrap(fate, 14))}</b><br>"
                        f"{totals[j]:,} · {totals[j] / start:.0%}",
                   font=dict(size=10.5, color=FATE_COLORS[fate]))
              for j, fate in enumerate(edge)]

    trace = go.Sankey(
        arrangement="fixed", domain=dict(x=[0, 1], y=[lo, hi]),
        node=dict(label=[""] * len(xs), x=xs, y=ys, pad=0, thickness=11,
                  color=tints, line=dict(color="white", width=0.5),
                  customdata=caps, hoverlabel=ghost_label(len(xs), 2),
                  hovertemplate="%{customdata}<extra></extra>"),
        link=dict(source=src, target=tgt, value=value, color=colors,
                  customdata=hover, hoverlabel=ghost_label(len(value), 1),
                  hovertemplate="%{customdata}<extra></extra>"),
    )
    annotations = [dict(xref="paper", yref="paper", showarrow=False, align="center",
                        **a) for a in notes]
    return trace, annotations
