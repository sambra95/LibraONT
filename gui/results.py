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
        ("Summary of Dataset and Processing", plots.read_summary_figure(
            report.length_counts, report.phred_counts, report.funnel,
            p.min_read_len, p.max_read_len, p.min_phred,
            plasmid_len=len(clean_sequence(p.plasmid_seq)) if p.plasmid_seq else None)),
        ("", plots.read_funnel_sankey_figure(
            report.funnel, report.fates,
            p.structural_insertion_bp, p.structural_deletion_bp)),
    ]
    if report.read_map is not None:
        figs.append(("Summary of Read Alignments",
                     plots.plasmid_map_figure(report.read_map, p.auto_codon_match_pct)))
        figs.append(("",
                     plots.read_alignment_figure(report.read_map, report.target,
                                                 p.structural_insertion_bp,
                                                 p.structural_deletion_bp,
                                                 p.auto_codon_match_pct)))
    figs.append(("Summary of Library Diversity and Coverage", plots.gap_match_figure(
        report.df_counts, ref_seq=report.target[:len(report.df_counts)],
        shade_codons=report.valid_positions or None, frame_offset=0,
        auto_match_threshold=p.auto_codon_match_pct,
        aa_counts=report.df_aa_counts)))

    if report.valid_positions:
        aa_fig = plots.aa_pies_figure(
            report.df_aa_counts, report.valid_positions, min_frac=p.pie_min_frac)
        if aa_fig is not None:
            figs.append(("", aa_fig))
        if report.hap_df is not None:
            figs.append(("", plots.haplotype_treemap_figure(
                report.hap_df, aa_counts=report.df_aa_counts,
                positions=report.valid_positions, min_frac=p.pie_min_frac)))
    return figs


def _stat_card(label: str, value: str, *, sub: str = "", accent: str = "") -> str:
    """One statistic as a styled HTML card, for the report's summary row."""
    pal = theme.PALETTE
    sub_html = (f"<div style='font-size:0.72rem;color:{pal['muted']};margin-top:3px'>"
                f"{html.escape(sub)}</div>") if sub else ""
    return (
        f"<div style='flex:1 1 0;min-width:130px;background:{pal['surface']};"
        f"border:1px solid {pal['grid']};border-radius:12px;padding:14px 16px;"
        f"border-top:3px solid {accent or pal['primary']};"
        "box-shadow:0 1px 2px rgba(0,0,0,0.05)'>"
        f"<div style='font-size:0.72rem;font-weight:600;letter-spacing:0.04em;"
        f"text-transform:uppercase;color:{pal['muted']}'>{html.escape(label)}</div>"
        f"<div style='font-size:1.7rem;font-weight:700;line-height:1.15;margin-top:4px;"
        f"color:{pal['primary_dark']}'>{html.escape(value)}</div>{sub_html}</div>")


def _figure_description(fig: go.Figure) -> str:
    """The plot's interpretation note, if it carries one."""
    meta = fig.layout.meta if isinstance(fig.layout.meta, dict) else {}
    return str(meta.get("description") or "")


def _summary_cards(report: Report) -> list[str]:
    """Summary card HTML shared by Streamlit rendering and HTML export."""
    insert = len(report.target)
    codons, spare = divmod(insert, 3)
    plasmid = clean_sequence(report.params.plasmid_seq or "")
    return [
        _stat_card("Total reads", f"{sum(report.length_counts.values()):,}"),
        _stat_card("Mean Phred",
                   f"{report.mean_phred:.1f}" if report.mean_phred is not None else "-",
                   sub="all reads"),
        _stat_card("Insert length", f"{insert:,} bp",
                   sub=f"not a whole number of codons - {spare} spare "
                       f"base{'' if spare == 1 else 's'}" if spare else "",
                   accent=theme.PALETTE["danger"] if spare else ""),
        _stat_card("Plasmid length", f"{len(plasmid):,} bp" if plasmid else "-"),
        _stat_card("Codons",
                   f"{codons:,}" + (f" + {spare} bp \u26a0\ufe0f" if spare else ""),
                   accent=theme.PALETTE["danger"] if spare else ""),
    ]


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
    rows = "".join(f"<tr><th>{html.escape(label)}</th>"
                   f"<td>{html.escape(value)}</td></tr>"
                   for label, value in _parameter_rows(report))
    return (
        "<section class='parameter-summary'>"
        "<h2>Analysis parameters</h2>"
        "<p class='caption'>Settings and inputs used to generate this report.</p>"
        "<table class='parameter-table'><tbody>" + rows + "</tbody></table></section>")


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


def _tables(report: Report) -> None:
    _section("Data tables")
    st.caption("Inspect the tabulated base counts, amino-acid frequencies, and haplotype calls "
               "used to build the plots.")
    for label, table in (("Base counts", report.df_counts),
                         ("Base freq", report.df_freq.round(4)),
                         ("AA counts", report.df_aa_counts),
                         ("AA freq", report.df_aa_freq.round(4))):
        with st.expander(label):
            st.dataframe(table, use_container_width=True)
    with st.expander("Haplotypes"):
        if report.hap_df is not None and not report.hap_df.empty:
            st.dataframe(report.hap_df[["combo_label", "count"]], use_container_width=True)
        else:
            st.info("No haplotypes (provide codon positions).")


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
        description = html.escape(_figure_description(fig))
        caption = f"<p class='caption'>{description}</p>" if description else ""
        heading = f"<h2>{title}</h2>" if title else ""
        sections.append(
            f"<section class='plot-section'>{heading}{caption}"
            f"<div class='plot'><img src='{_figure_png_data_uri(fig)}' "
            f"alt='{title}'></div></section>")

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
    .parameter-summary, .sequence-summary {{
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
    <div class="summary-cards">{''.join(_summary_cards(report))}</div>
    {warning}
    {_parameters_html(report)}
    {_funnel_html(report)}
    {_sequences_html(report)}
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


def _section(title: str) -> None:
    """Section heading; ``st.subheader`` would use the body text colour."""
    st.markdown(
        f"<h3 style='color:{theme.PALETTE['primary_dark']};font-size:1.5rem;"
        f"font-weight:600;margin:0 0 0.4rem;padding:0'>{html.escape(title)}</h3>",
        unsafe_allow_html=True)


def _divider() -> None:
    """Band between sections, fading out at both ends."""
    pal = theme.PALETTE
    st.markdown(
        f"<div style='height:10px;border-radius:10px;margin:38px 0 26px;"
        f"background:linear-gradient(90deg,"
        f"rgba(0,0,0,0) 0%,{pal['grid']} 8%,{pal['secondary']} 50%,{pal['grid']} 92%,"
        "rgba(0,0,0,0) 100%)'></div>",
        unsafe_allow_html=True)


def render(report: Report) -> None:
    """Top-level results renderer."""
    codons, spare = divmod(len(report.target), 3)
    if spare:      # trailing bases never form a codon, so they are dropped
        st.warning(
            f"Insert is {len(report.target):,} bp - not a whole number of codons "
            f"({codons:,} codons + {spare} spare base{'' if spare == 1 else 's'}). "
            f"The spare base{' is' if spare == 1 else 's are'} dropped from the "
            "codon and amino-acid analysis; check the start and stop positions.")
    if report.params.plasmid_seq and report.read_map is None:
        st.warning("Read alignment map skipped - samtools was not found on PATH "
                   "(activate the `libraont` conda environment).")
    figs = _build_figures(report)
    for label, fig in figs:
        # The heading is the figure's own title; a figure without one runs on
        # under the section above. "" not None: null renders as "undefined".
        if heading := fig.layout.title.text or label:
            _divider()
            _section(heading)
        if note := _figure_description(fig):
            st.caption(note)
        st.plotly_chart(go.Figure(fig).update_layout(title_text=""),
                        use_container_width=True,
                        config={"toImageButtonOptions": {"format": "svg"}})

    _divider()
    _tables(report)
    _divider()
    _downloads(report, figs)
