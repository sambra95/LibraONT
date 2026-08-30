"""Streamlit-cached composition of the analysis pipeline.

Only the minimap2 pass is memoised, keyed on the parameters affecting it. The
cached call must stay pure - ``st.cache_data`` replays element side-effects - so
progress is driven around it.
"""

from __future__ import annotations

import streamlit as st

from libraont import pipeline
from libraont.pipeline import AlignmentResult, AnalysisParams, Report


# ``key`` carries everything the result depends on; ``_params`` rides along but is
# ignored by the cache hasher.
@st.cache_data(show_spinner="Aligning reads…", max_entries=4)
def _cached_alignment(key: tuple, _params: AnalysisParams) -> AlignmentResult:
    return pipeline.compute_alignment(_params)


def run_analysis(params: AnalysisParams, progress=None) -> Report:
    """Run the pipeline."""
    if progress:
        progress(0.10, "Aligning reads")
    key = (params.fastq_path, params.gene_seq, params.plasmid_seq,
           params.min_read_len, params.max_read_len, params.min_phred)
    aln = _cached_alignment(key, params)
    # Uncached from here, so it always re-runs and may drive the progress bar.
    return pipeline.tabulate_report(params, aln, progress=progress)
