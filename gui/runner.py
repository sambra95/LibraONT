"""Streamlit-cached composition of the analysis pipeline.

Only the expensive read orientation/trimming + MAFFT alignment stage is memoised
with ``st.cache_data`` (keyed on *only* the parameters that affect it), so tweaking
a cheap downstream input - codon positions, pie layout - reuses the cached alignment
instead of re-running MAFFT. Everything after the alignment (coverage, tabulation,
plots, downloads) recomputes fresh each run, so the dashboard never shows stale
results. Caching lives here, not in ``libraont``, so the package stays free of any
Streamlit dependency.

The cached function must stay pure (no Streamlit element calls): ``st.cache_data``
records and replays element side-effects, which breaks if we drive a progress bar
created outside the function. So progress is updated *around* the cached call here,
and the heavy compute shows feedback via the decorator's ``show_spinner``.
"""

from __future__ import annotations

import streamlit as st

from libraont import pipeline
from libraont.pipeline import AnalysisParams, MsaResult, Report


# ``key`` (hashed) carries everything the result depends on; ``_params`` is passed
# through but ignored by Streamlit's cache hasher. No progress callback here - the
# function must not touch Streamlit elements (see module docstring).
@st.cache_data(show_spinner="Creating alignment…", max_entries=4)
def _cached_msa(key: tuple, _params: AnalysisParams) -> MsaResult:
    return pipeline.compute_msa(_params)


def run_analysis(params: AnalysisParams, progress=None) -> Report:
    """Run the pipeline"""
    def step(frac: float, msg: str) -> None:
        if progress:
            progress(frac, msg)

    step(0.10, "Aligning reads")
    msa_key = (params.fastq_path, params.gene_seq, params.start_pos, params.stop_pos,
               params.min_identity, params.pad, params.min_read_len, params.max_read_len)
    msa = _cached_msa(msa_key, params)

    # Everything past the alignment is uncached, so it always re-runs and may drive
    # the progress bar safely - picking up new coverage / codon / pie settings
    # without touching the cached alignment.
    step(0.65, "Computing coverage")
    coverage = pipeline.compute_coverage(params, msa.target, progress=progress)

    return pipeline.tabulate_report(params, msa, coverage, progress=progress)
