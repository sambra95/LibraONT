"""Render a computed :class:`Report`: figures, summary metrics, tables, downloads."""

from __future__ import annotations

import io
import base64
import html
import os
import zipfile

import plotly.graph_objects as go
import streamlit as st

from libraont import plots, theme
from libraont.alignment import tool_versions
from libraont.pipeline import Report
from libraont.sequences import clean_sequence


def _build_figures(report: Report) -> list[tuple[str, object]]:
    """Ordered (heading, figure) pairs: reads, their alignment, then the
    composition. An empty heading runs the figure on under the one above it."""
    p = report.params
    figs: list[tuple[str, object]] = [
        ("Read Dataset and Filtering", plots.read_summary_figure(
            report.length_counts, report.phred_counts, report.funnel,
            p.min_read_len, p.max_read_len, p.min_phred,
            plasmid_len=len(clean_sequence(p.plasmid_seq)) if p.plasmid_seq else None,
            insert_len=len(report.target), mean_phred=report.mean_phred)),
        ("", plots.read_funnel_sankey_figure(
            report.funnel, report.fates,
            p.structural_insertion_bp, p.structural_deletion_bp)),
    ]
    if report.read_map is not None:
        figs.append(("Read Alignment to the Plasmid",
                     plots.plasmid_map_figure(report.read_map, p.auto_codon_match_pct)))
        figs.append(("",
                     plots.read_alignment_figure(report.read_map, report.target,
                                                 p.structural_insertion_bp,
                                                 p.structural_deletion_bp)))
    figs.append(("Library Diversity and Coverage", plots.gap_match_figure(
        report.df_counts, ref_seq=report.target[:len(report.df_counts)],
        shade_codons=report.valid_positions or None, frame_offset=0,
        auto_match_threshold=p.auto_codon_match_pct,
        aa_counts=report.df_aa_counts)))

    if report.valid_positions:
        aa_fig = plots.aa_pies_figure(
            report.df_aa_counts, report.valid_positions, ref_seq=report.target,
            min_frac=p.pie_min_frac)
        if aa_fig is not None:
            figs.append(("", aa_fig))
        if report.hap_df is not None:
            figs.append(("", plots.haplotype_treemap_figure(
                report.hap_df, aa_counts=report.df_aa_counts,
                positions=report.valid_positions, min_frac=p.pie_min_frac)))
            panels = plots.variant_panels_figure(
                report.hap_df, report.codon_matrix, report.valid_positions,
                aa_counts=report.df_aa_counts, min_frac=p.pie_min_frac)
            if panels is not None:
                figs.append(("", panels))
    return figs


def _figure_meta(fig: go.Figure) -> dict:
    """The figure's own metadata: its subtitle and interpretation note."""
    return fig.layout.meta if isinstance(fig.layout.meta, dict) else {}


def _insert_note(report: Report) -> str | None:
    """Caveat for an insert that is not a whole number of codons; the trailing
    bases never form one, so they are dropped."""
    codons, spare = divmod(len(report.target), 3)
    if not spare:
        return None
    return (f"Insert is {len(report.target):,} bp - not a whole number of codons "
            f"({codons:,} codons + {spare} spare base{'' if spare == 1 else 's'}). "
            f"The spare base{' is' if spare == 1 else 's are'} dropped from the "
            "codon and amino-acid analysis; check the start and stop positions.")


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
    plasmid = clean_sequence(p.plasmid_seq or "")
    return [
        ("Input FASTQ", _fastq_download_name(report)),
        ("Reads in FASTQ", f"{sum(report.length_counts.values()):,}"),
        ("Mean Phred", f"{report.mean_phred:.1f}" if report.mean_phred is not None
                       else "-"),
        ("Insert length", f"{len(report.target):,} bp"),
        ("Plasmid reference", f"{len(plasmid):,} bp" if plasmid else "No"),
        ("Read length range", read_len_range),
        ("Minimum read quality",
         f"Q{p.min_phred}" if p.min_phred is not None else "No filtering"),
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
    ]


def _table_html(title: str, caption: str, rows: list[tuple[str, ...]]) -> str:
    """A titled section holding one table; the first cell of each row is its
    heading, and a row of ``<b>`` cells reads as a header."""
    body = "".join("<tr><th>" + html.escape(row[0]) + "</th>"
                   + "".join(f"<td>{cell}</td>" for cell in row[1:]) + "</tr>"
                   for row in rows)
    return (f"<section><h2>{title}</h2><p class='caption'>{caption}</p>"
            f"<table class='parameter-table'><tbody>{body}</tbody></table></section>")


def _funnel_html(report: Report) -> str:
    """Stage-by-stage read accounting for the downloaded report."""
    return _table_html(
        "Read filtering",
        "How many reads survive each step, and which outputs are built from each.",
        [("Step", "<b>Reads</b>", "<b>Rule</b>", "<b>Used by</b>")]
        + [(s.label, f"{s.count:,}", html.escape(s.detail),
            html.escape(", ".join(s.used_by) or "-")) for s in report.funnel])


def _parameters_html(report: Report) -> str:
    """HTML table summarising analysis parameters."""
    return _table_html(
        "Analysis parameters", "Settings and inputs used to generate this report.",
        [(label, html.escape(value)) for label, value in _parameter_rows(report)])


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
        "<section>"
        "<h2>Sequences</h2>"
        "<p class='caption'>Reference sequences used for this analysis.</p>"
        + _sequence_block("Insert sequence", report.target)
        + _sequence_block("Plasmid sequence", report.params.plasmid_seq)
        + "</section>")


def _tables(report: Report) -> None:
    _section("Data tables")
    st.caption("Inspect the tabulated base counts, amino-acid frequencies, and haplotype calls "
               "used to build the plots.")
    for label, table in (("Base counts", report.df_counts),
                         ("Base freq", report.df_freq.round(4)),
                         ("AA counts", report.df_aa_counts),
                         ("AA freq", report.df_aa_freq.round(4))):
        with st.expander(label):
            st.dataframe(table, width="stretch")
    with st.expander("Haplotypes"):
        if report.hap_df is not None and not report.hap_df.empty:
            st.dataframe(report.hap_df[["combo_label", "count"]], width="stretch")
        else:
            st.info("No haplotypes (provide codon positions).")


def _fastq_download_name(report: Report) -> str:
    """Original FASTQ filename for downloads, falling back to the readable path."""
    name = report.params.fastq_name or os.path.basename(report.params.fastq_path)
    return os.path.basename(name).replace("/", "_").replace("\\", "_") or "report"


_DIVIDER_CSS = ("height:10px;border-radius:10px;margin:38px 0 26px;background:"
                "linear-gradient(90deg,rgba(0,0,0,0) 0%,{grid} 8%,{secondary} 50%,"
                "{grid} 92%,rgba(0,0,0,0) 100%)").format(**theme.PALETTE)
_DIVIDER = f"<div style='{_DIVIDER_CSS}'></div>"


def _figure_png_data_uri(fig: go.Figure) -> str:
    """Render a Plotly figure to an embedded PNG data URI."""
    image = go.Figure(fig).update_layout(title_text="").to_image(format="png", scale=3)
    return "data:image/png;base64," + base64.b64encode(image).decode("ascii")


def _html_report(report: Report, figs: list[tuple[str, object]]) -> str:
    """Standalone HTML rendering of the main results section with static plot images."""
    pal = theme.PALETTE
    note = _insert_note(report)
    warning = f"<div class='warning'>{html.escape(note)}</div>" if note else ""

    # A divider above each heading, as in the app; a figure without one runs on
    # inside the section above it.
    def block(fig: go.Figure) -> str:
        meta = _figure_meta(fig)
        subtitle = html.escape(str(meta.get("subtitle") or ""))
        description = html.escape(str(meta.get("description") or ""))
        metrics = "".join(
            f"<div class='metric'><span>{html.escape(str(label))}</span>"
            f"<b>{html.escape(str(value))}</b></div>"
            for label, value in meta.get("metrics") or [])
        return ((f"<h3>{subtitle}</h3>" if subtitle else "")
                + (f"<p class='caption'>{description}</p>" if description else "")
                + (f"<div class='metrics'>{metrics}</div>" if metrics else "")
                + f"<div class='plot'><img src='{_figure_png_data_uri(fig)}' "
                  f"alt='{subtitle}'></div>")

    sections = []
    for label, fig in figs:
        title = html.escape(str(fig.layout.title.text or label))
        sections.append((f"{_DIVIDER}<h2>{title}</h2>" if title else "") + block(fig))

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
    h2 {{
      color: {pal['primary_dark']};
      margin: 0 0 6px;
      font-size: 1.35rem;
      line-height: 1.25;
    }}
    .caption {{
      color: {pal['muted']};
      margin: 0 0 12px;
      line-height: 1.45;
    }}
    .warning {{
      background: #FFF7E6;
      border: 1px solid #F1D08A;
      border-radius: 8px;
      color: {pal['text']};
      padding: 10px 12px;
      margin: 12px 0 18px;
    }}
    h3 {{
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
    .metrics {{
      display: flex;
      gap: 10px;
      margin: 0 0 12px;
    }}
    .metric {{
      flex: 1 1 0;
      padding: 9px 12px;
      background: {pal['surface']};
      border: 1px solid {pal['grid']};
      border-radius: 8px;
    }}
    .metric span {{
      display: block;
      color: {pal['muted']};
      font-size: 0.78rem;
    }}
    .metric b {{
      font-size: 1.4rem;
      font-weight: 600;
    }}
    .plot {{
      margin-top: 4px;
      flex: 3 1 0;
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
    {warning}
    {_DIVIDER}{_parameters_html(report)}
    {_DIVIDER}{_funnel_html(report)}
    {_DIVIDER}{_sequences_html(report)}
    {''.join(sections)}
  </main>
</body>
</html>"""


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
    _section("Downloads")
    st.caption("Download a static HTML report for sharing, or export the underlying data tables "
               "as CSV files in a ZIP archive.")
    stem = os.path.splitext(_fastq_download_name(report))[0]
    col_html, col_zip = st.columns(2)

    with col_html:
        try:
            st.download_button("⬇ Download HTML report",
                               _html_report(report, figs).encode("utf-8"),
                               file_name=f"ANALYSIS_{_fastq_download_name(report)}.html",
                               mime="text/html", type="primary", width="stretch")
        except Exception as exc:
            st.error(f"HTML export failed: {exc}")

    with col_zip:
        st.download_button("⬇ Download data tables (ZIP)",
                           _build_tables_zip(report),
                           file_name=f"{stem}_tables.zip", mime="application/zip",
                           type="primary", width="stretch")


def _plot(fig: go.Figure) -> None:
    """One plot with its own heading and note above it."""
    meta = _figure_meta(fig)
    if subtitle := meta.get("subtitle"):
        _subsection(str(subtitle))
    if note := meta.get("description"):
        st.caption(str(note))
    if metrics := meta.get("metrics"):
        for col, (label, value) in zip(st.columns(len(metrics)), metrics):
            col.metric(label, value, border=True)
    st.plotly_chart(go.Figure(fig).update_layout(title_text=""),
                    width="stretch",
                    config={"toImageButtonOptions": {"format": "svg"}})


def _subsection(title: str) -> None:
    """Heading for one plot inside a section."""
    st.markdown(
        f"<h3 style='color:{theme.PALETTE['primary_dark']};font-size:1.1rem;"
        f"font-weight:600;margin:0.7rem 0 0.15rem;padding:0'>"
        f"{html.escape(title)}</h3>", unsafe_allow_html=True)


def _section(title: str) -> None:
    """Section heading, at ``st.title`` size - Streamlit styles the h1 for us,
    and its own title would use the body text colour."""
    st.markdown(
        f"<h1 style='color:{theme.PALETTE['primary_dark']};margin:0 0 0.5rem;"
        f"padding:0'>{html.escape(title)}</h1>",
        unsafe_allow_html=True)


def _divider() -> None:
    """Band between sections, fading out at both ends."""
    st.markdown(_DIVIDER, unsafe_allow_html=True)


def render(report: Report) -> None:
    """Top-level results renderer."""
    if note := _insert_note(report):
        st.toast(note, icon="\u26a0\ufe0f")
    figs = _build_figures(report)
    for label, fig in figs:
        # The heading is the figure's own title; a figure without one runs on
        # under the section above. "" not None: null renders as "undefined".
        if heading := fig.layout.title.text or label:
            _divider()
            _section(heading)
        _plot(fig)

    _divider()
    _tables(report)
    _divider()
    _downloads(report, figs)
