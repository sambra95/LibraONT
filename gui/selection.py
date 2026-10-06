"""A selection followed over time: several FASTQs, one per sampled time point,
run through the same pipeline and drawn as variant frequencies over time."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from libraont import plots, timecourse

from . import results, runner


def run(samples, time_unit: str, progress=None) -> timecourse.TimeCourse:
    """Align every sample (memoised), then follow their variants over time. The
    samples share their settings, so the first carries the grouping threshold."""
    aligned = []
    for i, (name, time, params) in enumerate(samples):
        if progress:
            progress(0.05 + 0.45 * i / len(samples), f"Aligning {name}…")
        aligned.append((name, time, params, runner.align(params)))
    return timecourse.build(aligned, min_frac=samples[0][2].pie_min_frac,
                            time_unit=time_unit, progress=progress)


def _composition_tabs(course: timecourse.TimeCourse) -> None:
    """The single-library composition plots, one tab each, every time point drawn
    as a subplot of the same figure."""
    labels = [f"{p.time:g} {course.time_unit}" for p in course.points]
    reports = [p.report for p in course.points]
    first = reports[0]
    min_frac = first.params.pie_min_frac
    figs = [
        ("Amino acids", plots.aa_pies_time_figure(
            list(zip(labels, [r.df_aa_counts for r in reports])), course.positions,
            ref_seq=first.target, min_frac=min_frac)),
        ("Variant combinations", plots.haplotype_treemap_time_figure(
            [(label, r.hap_df, r.df_aa_counts, course.positions)
             for label, r in zip(labels, reports)], min_frac=min_frac)),
        ("Spread and sampling", plots.variant_panels_time_figure(
            [(label, r.hap_df, r.codon_matrix, r.df_aa_counts)
             for label, r in zip(labels, reports)], course.positions,
            min_frac=min_frac)),
    ]
    figs = [(name, fig) for name, fig in figs if fig is not None]
    if not figs:
        st.info("No variable codon could be described for these samples.")
        return
    for tab, (_, fig) in zip(st.tabs([name for name, _ in figs]), figs):
        with tab:
            results.plot(fig)


def render(course: timecourse.TimeCourse) -> None:
    """The time course: the reads behind each sample, the variant trajectories,
    then the single-library composition plots with every time point in them."""
    results.plot(plots.read_depth_figure(course.reads, time_unit=course.time_unit))
    st.dataframe(pd.DataFrame(
        [{"FASTQ": p.label, course.time_unit: p.time,
          "Assembled reads": p.report.n_intact,
          "Variants": 0 if p.report.hap_df is None else len(p.report.hap_df)}
         for p in course.points]), width="stretch", hide_index=True)
    st.caption("Variable codons followed: "
               + ", ".join(map(str, course.positions)))
    results.plot(plots.variant_trajectory_figure(course.freq,
                                                 time_unit=course.time_unit))
    _composition_tabs(course)
    with st.expander("Variant frequencies (%)"):
        st.dataframe(course.freq, width="stretch")
    st.download_button("⬇ Download frequencies (CSV)",
                       course.freq.to_csv().encode("utf-8"),
                       file_name="variant_timecourse.csv", mime="text/csv",
                       type="primary")
