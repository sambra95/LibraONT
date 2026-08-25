"""Render a computed :class:`Report`: figures, summary metrics, tables, downloads."""

from __future__ import annotations

import io
import base64
import math
import html
import os
import zipfile

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from libraont import plots, theme
from libraont.alignment import tool_versions
from libraont.analysis import haplotype_gini
from libraont.pipeline import Report
from libraont.sequences import clean_sequence


def _build_figures(report: Report) -> list[tuple[str, object]]:
    """Ordered (title, figure) pairs: raw reads, then their alignment, then the
    composition that follows from it. Everything below the composition describes
    the correctly assembled reads only."""
    p = report.params
    figs: list[tuple[str, object]] = [
        ("Read processing", plots.read_funnel_sankey_figure(report.funnel)),
        ("Read lengths", plots.read_length_figure(
            report.length_counts, p.min_read_len, p.max_read_len,
            plasmid_len=len(clean_sequence(p.plasmid_seq)) if p.plasmid_seq else None)),
    ]
    if report.read_map is not None:
        figs.append(("Read alignment map",
                     plots.read_alignment_figure(report.read_map, report.target,
                                                 p.structural_insertion_bp,
                                                 p.structural_deletion_bp)))
    figs.append(("Library composition", plots.read_fate_waffle_figure(
        report.fates, p.structural_insertion_bp, p.structural_deletion_bp)))
    figs.append(("Gap & match %", plots.gap_match_figure(
        report.df_counts, ref_seq=report.target[:len(report.df_counts)],
        shade_codons=report.valid_positions or None, frame_offset=0,
        auto_match_threshold=p.auto_codon_match_pct,
        aa_counts=report.df_aa_counts,
        reads_passing=report.n_reads_kept,
        reads_total=report.projection.n_mapped)))

    if report.valid_positions:
        aa_fig = plots.aa_pies_figure(
            report.df_aa_counts, report.valid_positions, min_frac=p.pie_min_frac)
        if aa_fig is not None:
            figs.append(("AA distribution", aa_fig))
        if report.hap_df is not None:
            figs.append(("Variant treemap", plots.haplotype_treemap_figure(
                report.hap_df, aa_counts=report.df_aa_counts,
                positions=report.valid_positions, min_frac=p.pie_min_frac)))
    return figs


# Card geometry: the big summary row, and the compact per-plot key.
_CARD_LARGE = dict(box="flex:1 1 0;min-width:130px;", radius=12, pad="14px 16px",
                   shadow="box-shadow:0 1px 2px rgba(0,0,0,0.05)", label="0.72rem",
                   value="1.7rem", line="1.15", gap=4, sub="0.72rem", sub_gap=3)
_CARD_COMPACT = dict(box="flex:0 1 170px;", radius=8, pad="9px 12px", shadow="",
                     label="0.68rem", value="1.15rem", line="1.2", gap=2,
                     sub="0.7rem", sub_gap=2)


def _card(label: str, value: str, sub: str, accent: str, s: dict) -> str:
    """One statistic as a styled HTML card, laid out to the preset ``s``."""
    pal = theme.PALETTE
    sub_html = (f"<div style='font-size:{s['sub']};color:{pal['muted']};"
                f"margin-top:{s['sub_gap']}px'>{html.escape(sub)}</div>") if sub else ""
    return (
        f"<div style='{s['box']}background:{pal['surface']};"
        f"border:1px solid {pal['grid']};border-top:3px solid {accent};"
        f"border-radius:{s['radius']}px;padding:{s['pad']}"
        f"{';' + s['shadow'] if s['shadow'] else ''}'>"
        f"<div style='font-size:{s['label']};font-weight:600;letter-spacing:0.04em;"
        f"text-transform:uppercase;color:{pal['muted']}'>{html.escape(label)}</div>"
        f"<div style='font-size:{s['value']};font-weight:700;line-height:{s['line']};"
        f"margin-top:{s['gap']}px;color:{pal['primary_dark']}'>{html.escape(value)}</div>"
        f"{sub_html}</div>")


def _stat_card(label: str, value: str, *, sub: str = "") -> str:
    """One summary statistic, for the big flex row at the top of the report."""
    return _card(str(label), str(value), str(sub), theme.PALETTE["primary"], _CARD_LARGE)


def _metric_cards_html(cards: list[dict], rows: int = 0) -> str:
    """Metric cards as a wrapping flex row, or a grid of exactly ``rows`` rows.

    ``color`` on a card overrides the accent, since these double as the colour
    key for plots that carry no legend.
    """
    if not cards:
        return ""
    preset = _CARD_COMPACT if not rows else {**_CARD_COMPACT, "box": ""}
    html_cards = [
        _card(str(c.get("label", "")), str(c.get("value", "")), str(c.get("sub", "")),
              c.get("color") or theme.PALETTE["primary"], preset)
        for c in cards]
    if rows:
        cols = max(1, math.ceil(len(html_cards) / rows))
        style = (f"display:grid;grid-template-columns:repeat({cols},minmax(0,1fr));"
                 "gap:8px;margin:0")
    else:
        style = "display:flex;gap:10px;flex-wrap:wrap;margin:-4px 0 10px"
    return f"<div class='metric-cards' style='{style}'>" + "".join(html_cards) + "</div>"


def _figure_meta(fig: go.Figure) -> dict:
    """Return figure metadata as a dict."""
    return fig.layout.meta if isinstance(fig.layout.meta, dict) else {}


def _figure_metric_cards(fig: go.Figure) -> None:
    """Compact cards for the metrics in ``fig.layout.meta``."""
    if cards := _metric_cards_html(_figure_meta(fig).get("metric_cards") or []):
        st.markdown(cards, unsafe_allow_html=True)


def _figure_description_text(fig: go.Figure) -> str:
    """The plot's interpretation note, if it carries one."""
    return str(_figure_meta(fig).get("description") or "")


def _figure_description(fig: go.Figure) -> None:
    if description := _figure_description_text(fig):
        st.caption(description)


def _summary_cards(report: Report) -> list[str]:
    """Summary card HTML shared by Streamlit rendering and HTML export."""
    if report.hap_df is not None and not report.hap_df.empty:
        variants, var_sub = f"{len(report.hap_df):,}", f"Gini {haplotype_gini(report.hap_df):.3f}"
    else:
        variants, var_sub = "-", "no codon positions"

    spanning, assembled_n = report.n_spanning, report.n_assembled
    assembled = f"{assembled_n / spanning:.1%}" if spanning else "-"
    total = sum(report.length_counts.values())
    unaligned = report.n_discarded_unaligned
    unaligned_sub = (f"{unaligned / total:.1%} of reads — likely contaminant DNA"
                     if total else "")
    return [
        _stat_card("Total reads", f"{total:,}",
                   sub=f"{report.projection.n_informative:,} informative about the insert"),
        _stat_card("Discarded, no alignment", f"{unaligned:,}", sub=unaligned_sub),
        _stat_card("Correctly assembled", assembled,
                   sub=f"{assembled_n:,} of {spanning:,} spanning reads"),
        _stat_card("Mean Phred", f"{report.mean_phred:.1f}" if report.mean_phred is not None else "-",
                   sub="all reads"),
        _stat_card("Insert length", f"{len(report.target):,} bp"),
        _stat_card("Codons", f"{report.df_aa_counts.shape[0]:,}"),
        _stat_card("Unique variants", variants, sub=var_sub),
    ]


def _summary(report: Report) -> None:
    st.markdown(
        "<div style='display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px'>"
        + "".join(_summary_cards(report)) + "</div>",
        unsafe_allow_html=True)


def _parameter_rows(report: Report) -> list[tuple[str, str]]:
    """Readable analysis parameters for the downloaded HTML report."""
    p = report.params
    tools = tool_versions()
    if p.min_read_len is not None or p.max_read_len is not None:
        lo = f"{p.min_read_len:,}" if p.min_read_len is not None else "0"
        hi = f"{p.max_read_len:,}" if p.max_read_len is not None else "∞"
        read_len_range = f"{lo}-{hi} bp"
    else:
        read_len_range = "No filtering"
    return [
        ("Input FASTQ", _fastq_download_name(report)),
        ("Insert Region", f"{p.start_pos}-{p.stop_pos} (length {len(report.target):,} bp)"),
        ("Plasmid reference", "Yes" if p.plasmid_seq else "No"),
        ("Read length range", read_len_range),
        ("Structural insertion threshold", f"≥ {p.structural_insertion_bp} bp"),
        ("Structural deletion threshold", f"≥ {p.structural_deletion_bp} bp"),
        ("Discarded, no alignment to reference", f"{report.n_discarded_unaligned:,}"),
        ("Correctly assembled reads",
         f"{report.n_assembled:,} of {report.n_spanning:,} spanning the insert"),
        ("Reads used for composition", f"{report.n_intact:,}"),
        ("Identified Variable Codons", ", ".join(map(str, report.valid_positions)) or "None"),
        ("Variable Codon Detection Threshold",
         f"< {p.auto_codon_match_pct:g}% match" if p.auto_codon_match_pct is not None else "Off"),
        ("Grouping threshold", f"{p.pie_min_frac:.3f}"),
        ("minimap2 version", tools.get("minimap2") or "-"),
        ("samtools version", tools.get("samtools") or "-"),
    ]


def _funnel_html(report: Report) -> str:
    """Stage-by-stage read accounting for the downloaded report."""
    rows = "".join(
        f"<tr><th>{html.escape(s.label)}</th><td>{s.count:,}</td>"
        f"<td>{html.escape(s.detail)}</td>"
        f"<td>{html.escape(', '.join(s.used_by) or '-')}</td></tr>"
        for s in report.funnel)
    return (
        "<section class='parameter-summary'><h2>Read filtering</h2>"
        "<p class='caption'>How many reads survive each step, and which outputs "
        "are built from each.</p>"
        "<table class='parameter-table'><tbody>"
        "<tr><th>Step</th><td><b>Reads</b></td><td><b>Rule</b></td>"
        "<td><b>Used by</b></td></tr>" + rows + "</tbody></table></section>")


def _parameters_html(report: Report) -> str:
    """HTML table summarising analysis parameters."""
    rows = []
    for label, value in _parameter_rows(report):
        rows.append(
            "<tr>"
            f"<th>{html.escape(str(label))}</th>"
            f"<td>{html.escape(str(value))}</td>"
            "</tr>")
    return (
        "<section class='parameter-summary'>"
        "<h2>Analysis parameters</h2>"
        "<p class='caption'>Settings and inputs used to generate this report.</p>"
        "<table class='parameter-table'><tbody>"
        + "".join(rows)
        + "</tbody></table></section>")


def _sequence_block(label: str, sequence: str | None) -> str:
    """HTML block for long nucleotide sequences."""
    if not sequence:
        return ""
    return (
        "<section class='sequence-section'>"
        f"<h3>{html.escape(label)}</h3>"
        f"<pre>{html.escape(sequence)}</pre>"
        "</section>")


def _sequences_html(report: Report) -> str:
    """HTML section containing the analysed insert and optional plasmid sequence."""
    return (
        "<section class='sequence-summary'>"
        "<h2>Sequences</h2>"
        "<p class='caption'>Reference sequences used for this analysis.</p>"
        + _sequence_block("Insert sequence", report.target)
        + _sequence_block("Plasmid sequence", report.params.plasmid_seq)
        + "</section>")


def _funnel_detail(report: Report) -> None:
    """Stage-by-stage read accounting, and which plots consume each stage."""
    with st.expander("Read filtering, step by step"):
        st.caption("The same journey as the library-composition squares above, "
                   "read as a sequence of filters.")
        st.dataframe(
            pd.DataFrame([{"Step": s.label, "Reads": s.count,
                           "Rule": s.detail,
                           "Used by": ", ".join(s.used_by) or "-"}
                          for s in report.funnel]),
            hide_index=True, use_container_width=True)


def _tables(report: Report) -> None:
    st.subheader("Data tables")
    st.caption("Inspect the tabulated base counts, amino-acid frequencies, and haplotype calls "
               "used to build the plots.")
    with st.expander("Data tables"):
        tabs = st.tabs(["Base counts", "Base freq", "AA counts", "AA freq", "Haplotypes"])
        tabs[0].dataframe(report.df_counts, use_container_width=True)
        tabs[1].dataframe(report.df_freq.round(4), use_container_width=True)
        tabs[2].dataframe(report.df_aa_counts, use_container_width=True)
        tabs[3].dataframe(report.df_aa_freq.round(4), use_container_width=True)
        if report.hap_df is not None and not report.hap_df.empty:
            tabs[4].dataframe(report.hap_df[["combo_label", "count"]], use_container_width=True)
        else:
            tabs[4].info("No haplotypes (provide codon positions).")


def _fastq_download_name(report: Report) -> str:
    """Original FASTQ filename for downloads, falling back to the readable path."""
    name = report.params.fastq_name or os.path.basename(report.params.fastq_path)
    return os.path.basename(name).replace("/", "_").replace("\\", "_") or "report"


def _figure_png_data_uri(fig: go.Figure) -> str:
    """Render a Plotly figure to an embedded PNG data URI."""
    image = go.Figure(fig).update_layout(title_text="").to_image(format="png", scale=3)
    return "data:image/png;base64," + base64.b64encode(image).decode("ascii")


def _html_report(report: Report, figs: list[tuple[str, object]]) -> str:
    """Standalone HTML rendering of the main results section with static plot images."""
    pal = theme.PALETTE
    warning = ""
    if report.params.plasmid_seq and report.read_map is None:
        warning = (
            "<div class='warning'>Read alignment map skipped - samtools was not "
            "found on PATH.</div>")

    sections = []
    for label, fig in figs:
        title = html.escape(str(fig.layout.title.text or label))
        description = html.escape(_figure_description_text(fig))
        caption = f"<p class='caption'>{description}</p>" if description else ""
        meta = _figure_meta(fig)
        cards = meta.get("metric_cards") or []
        image_uri = _figure_png_data_uri(fig)
        plot_html = f"<div class='plot'><img src='{image_uri}' alt='{title}'></div>"
        if meta.get("cards_inline") and cards:
            body = (f"<div class='plot-inline'>{plot_html}<div class='plot-key'>"
                    f"{_metric_cards_html(cards, rows=2)}</div></div>")
        else:
            body = _metric_cards_html(cards) + plot_html
        sections.append(
            f"<section class='plot-section'><h2>{title}</h2>{caption}{body}</section>")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LibraONT report</title>
  <style>
    body {{
      margin: 0;
      background: {pal['bg']};
      color: {pal['text']};
      font-family: Inter, Segoe UI, Helvetica, Arial, sans-serif;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 28px 44px;
    }}
    h1 {{
      color: {pal['primary_dark']};
      margin: 0 0 4px;
      font-size: 2.2rem;
      line-height: 1.15;
    }}
    .subtitle {{
      color: {pal['muted']};
      margin: 0 0 18px;
    }}
    h2 {{
      color: {pal['primary_dark']};
      margin: 30px 0 6px;
      font-size: 1.35rem;
      line-height: 1.25;
    }}
    .caption {{
      color: {pal['muted']};
      margin: 0 0 12px;
      line-height: 1.45;
    }}
    .summary-cards {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 18px;
    }}
    .warning {{
      background: #FFF7E6;
      border: 1px solid #F1D08A;
      border-radius: 8px;
      color: {pal['text']};
      padding: 10px 12px;
      margin: 12px 0 18px;
    }}
    .parameter-summary {{
      border-top: 1px solid {pal['grid']};
      padding-top: 2px;
      margin-bottom: 22px;
    }}
    .sequence-summary {{
      border-top: 1px solid {pal['grid']};
      padding-top: 2px;
      margin-bottom: 22px;
    }}
    .sequence-section h3 {{
      color: {pal['primary_dark']};
      font-size: 1rem;
      margin: 14px 0 6px;
    }}
    .sequence-section pre {{
      margin: 0;
      padding: 12px;
      background: {pal['surface']};
      border: 1px solid {pal['grid']};
      border-radius: 8px;
      color: {pal['text']};
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.82rem;
      line-height: 1.45;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }}
    .parameter-table {{
      width: 100%;
      border-collapse: collapse;
      background: {pal['bg']};
      border: 1px solid {pal['grid']};
      border-radius: 8px;
      overflow: hidden;
    }}
    .parameter-table th,
    .parameter-table td {{
      padding: 9px 12px;
      border-bottom: 1px solid {pal['grid']};
      text-align: left;
      vertical-align: top;
    }}
    .parameter-table tr:last-child th,
    .parameter-table tr:last-child td {{
      border-bottom: 0;
    }}
    .parameter-table th {{
      width: 210px;
      background: {pal['surface']};
      color: {pal['muted']};
      font-size: 0.78rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .plot-section {{
      border-top: 1px solid {pal['grid']};
      padding-top: 2px;
    }}
    .plot {{
      margin-top: 4px;
      flex: 3 1 0;
      min-width: 0;
    }}
    .plot-inline {{
      display: flex;
      gap: 16px;
      align-items: center;
      flex-wrap: wrap;
      margin-top: 4px;
    }}
    .plot-key {{
      flex: 2 1 320px;
      min-width: 0;
    }}
    .plot img {{
      display: block;
      width: 100%;
      height: auto;
    }}
  </style>
</head>
<body>
  <main>
    <h1>LibraONT</h1>
    <p class="subtitle">Nanopore mutagenesis-library analysis - orientation, alignment, base/AA composition and variant diversity.</p>
    <div class="summary-cards">{''.join(_summary_cards(report))}</div>
    {warning}
    {_parameters_html(report)}
    {_funnel_html(report)}
    {_sequences_html(report)}
    {''.join(sections)}
  </main>
</body>
</html>"""


def _build_html(report: Report,
                figs: list[tuple[str, object]]) -> tuple[str, bytes]:
    """The results section as a standalone HTML file: (name, bytes)."""
    return (f"ANALYSIS_{_fastq_download_name(report)}.html",
            _html_report(report, figs).encode("utf-8"))


def _build_tables_zip(report: Report) -> bytes:
    """Bundle the tabulated datasets (one CSV per table) into a zip."""
    tables = {
        "base_counts.csv": report.df_counts.to_csv(),
        "base_frequencies.csv": report.df_freq.round(6).to_csv(),
        "aa_counts.csv": report.df_aa_counts.to_csv(),
        "aa_frequencies.csv": report.df_aa_freq.round(6).to_csv(),
    }
    if report.hap_df is not None and not report.hap_df.empty:
        tables["haplotypes.csv"] = report.hap_df[["combo_label", "count"]].to_csv(index=False)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, csv in tables.items():
            zf.writestr(name, csv)
    return buf.getvalue()


def _downloads(report: Report, figs: list[tuple[str, object]]) -> None:
    """Build the HTML report and table ZIP fresh, and offer both."""
    st.subheader("Downloads")
    st.caption("Download a static HTML report for sharing, or export the underlying data tables "
               "as CSV files in a ZIP archive.")
    stem = os.path.splitext(_fastq_download_name(report))[0]
    col_html, col_zip = st.columns(2)

    with col_html:
        try:
            fname, data = _build_html(report, figs)
            st.download_button("⬇ Download HTML report", data, file_name=fname,
                               mime="text/html", type="primary", width="stretch")
        except Exception as exc:
            st.error(f"HTML export failed: {exc}")

    with col_zip:
        st.download_button("⬇ Download data tables (ZIP)",
                           _build_tables_zip(report),
                           file_name=f"{stem}_tables.zip", mime="application/zip",
                           type="primary", width="stretch")


def render(report: Report) -> None:
    """Top-level results renderer."""
    _summary(report)

    if report.params.plasmid_seq and report.read_map is None:
        st.warning("Read alignment map skipped - samtools was not found on PATH "
                   "(activate the `libraont` conda environment).")
    spanning, assembled = report.n_spanning, report.n_assembled

    figs = _build_figures(report)
    for label, fig in figs:
        # Title as the heading, and a title-less copy below it so it is not shown
        # twice. Use "" not None: a null title renders as literal "undefined".
        st.subheader(fig.layout.title.text or label)
        _figure_description(fig)
        chart = go.Figure(fig).update_layout(title_text="")
        config = {"toImageButtonOptions": {"format": "svg"}}
        cards = _figure_meta(fig).get("metric_cards") or []
        if _figure_meta(fig).get("cards_inline") and cards:
            col_plot, col_key = st.columns([3, 2], vertical_alignment="center")
            col_plot.plotly_chart(chart, use_container_width=True, config=config)
            col_key.markdown(_metric_cards_html(cards, rows=2),
                             unsafe_allow_html=True)
        else:
            _figure_metric_cards(fig)
            st.plotly_chart(chart, use_container_width=True, config=config)

    _funnel_detail(report)
    _tables(report)
    _downloads(report, figs)
