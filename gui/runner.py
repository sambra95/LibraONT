"""Streamlit-cached composition of the analysis pipeline.

The expensive stages - read orientation/trimming + MAFFT alignment, and the
optional minimap2/samtools coverage - are memoised with ``st.cache_data`` keyed
on *only* the parameters that affect them. Tweaking a cheap downstream input
(codon positions, pie layout) therefore reuses the cached alignment instead of
re-running MAFFT. Caching lives here, not in ``libraont``, so the package stays
free of any Streamlit dependency.

The cached functions must stay pure (no Streamlit element calls): ``st.cache_data``
records and replays element side-effects, which breaks if we drive a progress bar
created outside the function. So progress is updated *around* the cached calls
here, and the heavy compute shows feedback via the decorator's ``show_spinner``.
"""

from __future__ import annotations

import streamlit as st

from libraont import pipeline
from libraont.alignment import CoverageResult
from libraont.pipeline import AnalysisParams, MsaResult, Report


def _tool_key(params: AnalysisParams) -> tuple:
    """Hashable view of the tool-path overrides (which binary actually runs)."""
    return tuple(sorted(params.tool_paths.items()))


# ``key`` (hashed) carries everything the result depends on; ``_params`` is passed
# through but ignored by Streamlit's cache hasher. No progress callback here - the
# function must not touch Streamlit elements (see module docstring).
@st.cache_data(show_spinner="Orienting reads + MAFFT alignment…", max_entries=4)
def _cached_msa(key: tuple, _params: AnalysisParams) -> MsaResult:
    return pipeline.compute_msa(_params)


@st.cache_data(show_spinner="Computing coverage (minimap2 + samtools)…", max_entries=4)
def _cached_coverage(key: tuple, _params: AnalysisParams,
                     _target: str) -> CoverageResult | None:
    return pipeline.compute_coverage(_params, _target)


def run_analysis(params: AnalysisParams, progress=None) -> Report:
    """Run the pipeline"""
    def step(frac: float, msg: str) -> None:
        if progress:
            progress(frac, msg)

    step(0.10, "Aligning reads")
    msa_key = (params.fastq_path, params.gene_seq, params.start_pos, params.stop_pos,
               params.min_identity, params.pad, params.min_read_len, params.max_read_len,
               _tool_key(params))
    msa = _cached_msa(msa_key, params)

    step(0.65, "Computing coverage")
    cov_key = (params.fastq_path, params.plasmid_seq, msa.target,
               params.min_read_len, params.max_read_len, _tool_key(params))
    coverage = _cached_coverage(cov_key, params, msa.target)

    # Downstream tabulation is cheap and not cached, so it always re-runs and may
    # drive the progress bar safely (picks up new codon positions / pie settings
    # without touching the cached alignment).
    return pipeline.tabulate_report(params, msa, coverage, progress=progress)
