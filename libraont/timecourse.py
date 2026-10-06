"""Several FASTQs, one selection: each is analysed on its own (see
``libraont.pipeline``) and their variant frequencies are lined up over time.

The variable codons are the union of what every sample gives, so one variant
means the same amino-acid combination at every time point; samples detecting a
different set are re-tabulated against that union.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from . import plots
from .analysis import variant_labels
from .pipeline import (AlignmentResult, AnalysisParams, Progress, Report,
                       tabulate_report)


@dataclass
class SamplePoint:
    """One FASTQ, the time it was sampled, and what the pipeline made of it."""
    label: str
    time: float
    report: Report


@dataclass
class TimeCourse:
    """A selection followed over time."""
    points: list[SamplePoint]
    positions: list[int]
    # Variants (rows, labelled as the treemap tiles are) by time point (columns),
    # as a percentage of that sample's fully-called reads.
    freq: pd.DataFrame
    # Those reads themselves, per time point - what each column is a share of.
    reads: pd.Series
    time_unit: str = "Time"

    @property
    def times(self) -> list[float]:
        return [p.time for p in self.points]


def _sample_counts(report: Report, min_frac: float) -> tuple[pd.Series, float]:
    """Variant read counts for one sample over the variants the treemap keeps,
    and the reads behind them."""
    hap = report.hap_df
    if hap is None or hap.empty:
        return pd.Series(dtype=float), 0.0
    kept = plots._drop_rare_variants(hap, report.df_aa_counts,
                                     report.valid_positions, min_frac)
    total = float(kept["count"].sum())
    if kept.empty or total <= 0:
        return pd.Series(dtype=float), 0.0
    return kept["count"].astype(float).groupby(variant_labels(kept)).sum(), total


def build(samples: list[tuple[str, float, AnalysisParams, AlignmentResult]], *,
          min_frac: float = 0.0, time_unit: str = "Time",
          progress: Progress = None) -> TimeCourse:
    """Tabulate each aligned sample, unify their variable codons, and line the
    variant frequencies up over time. ``samples`` carries one
    ``(label, time, params, alignment)`` per FASTQ."""
    if not samples:
        raise ValueError("No samples to follow.")

    reports = {}
    for i, (label, _, params, aln) in enumerate(samples):
        if progress:
            progress(0.1 + 0.5 * i / len(samples), f"Tabulating {label}…")
        reports[label] = tabulate_report(params, aln)

    positions = sorted({p for r in reports.values() for p in r.valid_positions})
    if not positions:
        raise RuntimeError(
            "No variable codon was found in any sample, so there are no variants "
            "to follow. Lower the detection threshold or name the codons.")

    for i, (label, _, params, aln) in enumerate(samples):
        if reports[label].valid_positions == positions:
            continue
        if progress:
            progress(0.6 + 0.3 * i / len(samples), f"Re-reading {label} at the shared codons…")
        reports[label] = tabulate_report(
            replace(params, pie_positions=positions, auto_codon_match_pct=None), aln)

    points = sorted((SamplePoint(label, time, reports[label])
                     for label, time, _, _ in samples), key=lambda p: p.time)
    per_sample = {p.time: _sample_counts(p.report, min_frac) for p in points}
    counts = pd.DataFrame({t: c for t, (c, _) in per_sample.items()}).fillna(0.0)
    counts = counts.reindex(columns=sorted(counts.columns))
    reads = pd.Series({t: n for t, (_, n) in per_sample.items()},
                      dtype=float).sort_index()
    freq = counts.div(reads.replace(0, float("nan")), axis=1).fillna(0.0) * 100
    return TimeCourse(points=points, positions=positions, freq=freq, reads=reads,
                      time_unit=time_unit)
