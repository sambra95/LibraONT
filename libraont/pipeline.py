"""High-level orchestration: FASTQ + reference -> a fully-populated :class:`Report`.

This mirrors the notebook's ``create_report`` but separates *computation* (here)
from *presentation* (``libraont.plots`` / the GUI). Heavy steps report progress
through an optional ``progress(fraction, message)`` callback.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from . import alignment, analysis
from .alignment import CoverageResult
from .constants import (BASE_CATEGORIES, DEFAULT_MIN_IDENTITY, DEFAULT_MSA_BATCH_SIZE,
                        DEFAULT_PAD, DEFAULT_PIE_MIN_FRAC)
from .sequences import extract_target, read_length_counts

Progress = Optional[Callable[[float, str], None]]


@dataclass
class AnalysisParams:
    """Everything needed to run one analysis (mirrors the notebook inputs)."""
    fastq_path: str
    gene_seq: str
    start_pos: int
    stop_pos: int
    fastq_name: Optional[str] = None
    min_identity: float = DEFAULT_MIN_IDENTITY
    pad: int = DEFAULT_PAD
    plasmid_seq: Optional[str] = None
    pie_positions: list[int] = field(default_factory=list)
    pie_min_frac: float = DEFAULT_PIE_MIN_FRAC
    # When set, codons with a position below this reference-match % are auto-added
    # to ``pie_positions`` (variable-codon detection). ``None`` disables it.
    auto_codon_match_pct: Optional[float] = None
    tool_paths: dict[str, str] = field(default_factory=dict)


@dataclass
class Report:
    """All computed results; figures are built from this by ``libraont.plots``."""
    target: str
    params: AnalysisParams
    n_reads_kept: int
    length_counts: Counter
    df_counts: pd.DataFrame
    df_freq: pd.DataFrame
    df_aa_counts: pd.DataFrame
    df_aa_freq: pd.DataFrame
    ref_cols: list[int]
    ref_codons: list[tuple[int, int, int]]
    valid_positions: list[int]
    mean_phred: Optional[float] = None
    hap_df: Optional[pd.DataFrame] = None
    coverage: Optional[CoverageResult] = None


@dataclass
class MsaResult:
    """Output of the expensive alignment stage (the part worth caching)."""
    target: str
    length_counts: Counter
    mean_phred: Optional[float]
    msa: dict[str, str]
    n_reads_kept: int


def _stepper(progress: Progress):
    """Wrap an optional progress callback into a safe ``step(frac, msg)``."""
    def step(frac: float, msg: str) -> None:
        if progress:
            progress(frac, msg)
    return step


def compute_msa(params: AnalysisParams, tools: Optional[dict] = None,
                progress: Progress = None) -> MsaResult:
    """Expensive stage: target extraction, read orientation/trimming, MAFFT MSA.

    Pure and side-effect free given its inputs, so it is safe to memoise on the
    parameters that affect it (see ``gui.runner``).
    """
    step = _stepper(progress)
    tools = tools if tools is not None else alignment.tool_status(params.tool_paths)

    step(0.05, "Parsing reference and reading FASTQ…")
    target = extract_target(params.gene_seq, params.start_pos, params.stop_pos)
    length_counts = read_length_counts(params.fastq_path)

    step(0.15, "Orienting and trimming reads (edlib)…")
    seqs_named, mean_phred = alignment.edlib_orient_trim_and_quality(
        params.fastq_path, target, min_identity=params.min_identity,
        pad=params.pad)
    if len(seqs_named) <= 1:
        raise RuntimeError("No reads passed trimming/orientation. "
                           "Try lowering min_identity or increasing pad.")

    if not tools.get("mafft"):
        raise RuntimeError("MAFFT not found. Set its path in the sidebar or install it.")
    step(0.30, f"Aligning {len(seqs_named) - 1} reads with MAFFT…")
    msa = alignment.run_msa_mafft(
        seqs_named, mafft_bin=tools["mafft"], threads=1, op=3.5, ep=1.0,
        extra_args=["--localpair", "--maxiterate", "1000", "--quiet"],
        batch_size=DEFAULT_MSA_BATCH_SIZE)
    return MsaResult(target=target, length_counts=length_counts, mean_phred=mean_phred, msa=msa,
                     n_reads_kept=len(seqs_named) - 1)


def compute_coverage(params: AnalysisParams, target: str, tools: Optional[dict] = None,
                     progress: Progress = None) -> Optional[CoverageResult]:
    """Optional whole-plasmid coverage (needs a plasmid + minimap2 + samtools).

    Returns ``None`` when no plasmid was given or the tools are unavailable -
    the GUI surfaces the latter as a warning. Also expensive, so cached too.
    """
    if not params.plasmid_seq:
        return None
    tools = tools if tools is not None else alignment.tool_status(params.tool_paths)
    if not (tools.get("minimap2") and tools.get("samtools")):
        return None
    _stepper(progress)(0.92, "Computing coverage (minimap2 + samtools)…")
    return alignment.compute_coverage(
        params.plasmid_seq, params.fastq_path, inner_seq=target,
        minimap2_bin=tools["minimap2"], samtools_bin=tools["samtools"])


def tabulate_report(params: AnalysisParams, msa_result: MsaResult,
                    coverage: Optional[CoverageResult], progress: Progress = None) -> Report:
    """Cheap stage: base/AA counts and haplotypes from a computed MSA."""
    step = _stepper(progress)
    msa = msa_result.msa

    step(0.70, "Tabulating base counts…")
    df_counts, df_freq, ref_cols, _cov = analysis.counts_from_msa_ref_columns(
        msa, alphabet=BASE_CATEGORIES, ignore_terminal_gaps=True)

    step(0.80, "Tabulating amino-acid counts…")
    df_aa_counts, df_aa_freq, ref_codons = analysis.aa_counts_from_msa(msa, frame_offset=0)

    # Manually-specified codons, plus auto-detected variable codons when enabled.
    pie_positions = list(params.pie_positions)
    if params.auto_codon_match_pct is not None:
        ref_seq = msa_result.target[:len(df_counts)]
        pie_positions += analysis.detect_variable_codons(
            df_counts, ref_seq, params.auto_codon_match_pct)
    valid_positions = sorted({p for p in pie_positions if 1 <= p <= df_aa_counts.shape[0]})
    hap_df = None
    if valid_positions:
        step(0.88, "Counting haplotypes…")
        ref_codons = analysis.get_ref_codons(msa)
        hap_df = analysis.haplotype_counts(msa, ref_codons, valid_positions)

    step(1.0, "Done.")
    return Report(
        target=msa_result.target, params=params, n_reads_kept=msa_result.n_reads_kept,
        length_counts=msa_result.length_counts, mean_phred=msa_result.mean_phred,
        df_counts=df_counts, df_freq=df_freq, df_aa_counts=df_aa_counts,
        df_aa_freq=df_aa_freq, ref_cols=ref_cols, ref_codons=ref_codons,
        valid_positions=valid_positions, hap_df=hap_df, coverage=coverage)


def run_analysis(params: AnalysisParams, progress: Progress = None) -> Report:
    """Run the full pipeline and return a :class:`Report` (uncached convenience
    wrapper; the GUI composes the cached stages itself - see ``gui.runner``)."""
    tools = alignment.tool_status(params.tool_paths)
    msa_result = compute_msa(params, tools=tools, progress=progress)
    coverage = compute_coverage(params, msa_result.target, tools=tools, progress=progress)
    return tabulate_report(params, msa_result, coverage, progress=progress)
