"""High-level orchestration: FASTQ + reference -> a fully-populated :class:`Report`.

Computation only; presentation lives in ``libraont.plots`` and the GUI. Heavy
steps report through an optional ``progress(fraction, message)`` callback.

Reads holding an indel at or above its structural threshold are mis-assembled
and excluded from the composition tables, so the base/AA counts, pies and
treemap describe the correctly assembled fraction only.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from . import alignment, analysis
from .alignment import Projection, ReadMap, ReadStructure
from .constants import (BASE_CATEGORIES, DEFAULT_PIE_MIN_FRAC,
                        DEFAULT_STRUCTURAL_DELETION_BP, DEFAULT_STRUCTURAL_INSERTION_BP)
from .sequences import extract_target, fastq_stats

Progress = Optional[Callable[[float, str], None]]

REF_NAME = "REF"
UNCALLED = "?"          # stands in for a diversified codon a read does not cover


@dataclass
class AnalysisParams:
    """Everything needed to run one analysis."""
    fastq_path: str
    gene_seq: str
    start_pos: int
    stop_pos: int
    fastq_name: Optional[str] = None
    # Raw-read length window: reads outside ``[min_read_len, max_read_len]`` are
    # dropped before alignment. ``None`` on either side disables that bound.
    min_read_len: Optional[int] = None
    max_read_len: Optional[int] = None
    # Reads whose mean Phred falls below this are dropped with them.
    min_phred: Optional[int] = None
    plasmid_seq: Optional[str] = None
    # Smallest indel counted as an assembly defect rather than basecall noise,
    # per direction.
    structural_insertion_bp: int = DEFAULT_STRUCTURAL_INSERTION_BP
    structural_deletion_bp: int = DEFAULT_STRUCTURAL_DELETION_BP
    pie_positions: list[int] = field(default_factory=list)
    # Amino acids below this frequency at their codon are folded into 'Other' in
    # the AA pies, and variants containing one are dropped from the treemap. 0
    # keeps everything.
    pie_min_frac: float = DEFAULT_PIE_MIN_FRAC
    # How many diversified codons a read may miss and still earn a tile in the
    # variant treemap; the ones it misses are written as '?'.
    max_unknown_codons: int = 0
    # When set, codons with a position below this reference-match % are auto-added
    # to ``pie_positions`` (variable-codon detection). ``None`` disables it.
    auto_codon_match_pct: Optional[float] = None


# Read fates, best first. Every aligned read lands in exactly one, so the counts
# sum to the aligned total. Reads dropped earlier (raw length, no alignment) are
# reported as discards, keeping that denominator meaningful.
FATE_ORDER: tuple[str, ...] = (
    "Variant",
    "Ambiguous diversity",
    "Wild type",
    "Deletion",
    "Insertion",
    "Does not contain insert region",
)

# Fates whose reads still feed the composition tables (partial reads contribute
# wherever they cover the insert).
COMPOSITION_FATES: frozenset[str] = frozenset({
    "Variant", "Wild type", "Ambiguous diversity"})


@dataclass
class ReadFate:
    """How many reads ended up with one fate, and whether they are still used."""
    label: str
    count: int
    used_for_composition: bool = False


@dataclass
class FunnelStage:
    """One step of the read-filtering funnel, and which outputs consume it."""
    label: str
    count: int
    detail: str = ""
    used_by: tuple[str, ...] = ()
    lost: str = ""              # short reason reads fail to reach this stage
    passed: str = ""            # what the reads that do reach it just cleared
    failed: str = ""            # and what the reads that do not fell foul of


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
    valid_positions: list[int]
    projection: Projection
    funnel: list[FunnelStage]
    fates: list[ReadFate]
    mean_phred: Optional[float] = None
    phred_counts: Counter = field(default_factory=Counter)
    hap_df: Optional[pd.DataFrame] = None
    read_map: Optional[ReadMap] = None
    # Most diversified codons any single read fails to cover - the ceiling worth
    # offering on the treemap's tolerance.
    max_unknown_codons: int = 0

    @property
    def structures(self) -> dict[str, ReadStructure]:
        return self.projection.structures

    @property
    def n_discarded_unaligned(self) -> int:
        """Reads dropped before analysis for not aligning to the reference."""
        return self.projection.n_unaligned

    @property
    def n_spanning(self) -> int:
        return len(self.projection.spanning_names())

    @property
    def n_assembled(self) -> int:
        """Correctly assembled reads among those spanning the insert."""
        spanning = set(self.projection.spanning_names())
        return sum(1 for n in self.projection.intact_names(
            self.params.structural_insertion_bp,
            self.params.structural_deletion_bp) if n in spanning)

    @property
    def n_intact(self) -> int:
        return len(self.projection.intact_names(self.params.structural_insertion_bp,
                                                self.params.structural_deletion_bp))


@dataclass
class AlignmentResult:
    """Output of the expensive alignment stage (the part worth caching)."""
    target: str
    length_counts: Counter
    mean_phred: Optional[float]
    phred_counts: Counter
    projection: Projection


def _stepper(progress: Progress):
    """Wrap an optional progress callback into a safe ``step(frac, msg)``."""
    def step(frac: float, msg: str) -> None:
        if progress:
            progress(frac, msg)
    return step


def compute_alignment(params: AnalysisParams, tools: Optional[dict] = None,
                      progress: Progress = None) -> AlignmentResult:
    """Expensive stage: target extraction and minimap2 alignment/projection.

    Pure and side-effect free given its inputs, so it is safe to memoise on the
    parameters that affect it (see ``gui.runner``).
    """
    step = _stepper(progress)
    tools = tools if tools is not None else alignment.tool_status()
    if not tools.get("minimap2"):
        raise RuntimeError("minimap2 not found. Activate the `libraont` environment "
                           "or install it - it produces the alignment every table "
                           "and plot is built from.")

    step(0.05, "Parsing reference and reading FASTQ…")
    target = extract_target(params.gene_seq, params.start_pos, params.stop_pos)
    length_counts, mean_phred, phred_counts = fastq_stats(params.fastq_path)

    step(0.25, "Aligning reads (minimap2)…")
    projection = alignment.project_reads(
        target, params.fastq_path, minimap2_bin=tools["minimap2"],
        reference_seq=params.plasmid_seq, min_read_len=params.min_read_len,
        max_read_len=params.max_read_len, min_phred=params.min_phred,
        n_input=sum(length_counts.values()))
    if not projection.rows:
        raise RuntimeError("No reads aligned to the reference insert. Check the gene "
                           "sequence and the read-length window.")
    return AlignmentResult(target=target, length_counts=length_counts,
                           mean_phred=mean_phred, phred_counts=phred_counts,
                           projection=projection)


def compute_read_map(params: AnalysisParams, target: str, tools: Optional[dict] = None,
                     progress: Progress = None) -> Optional[ReadMap]:
    """Optional whole-plasmid read map (needs a plasmid + minimap2 + samtools).

    Returns ``None`` when no plasmid was given or the tools are unavailable -
    the GUI surfaces the latter as a warning. Also expensive, so cached too.
    """
    if not params.plasmid_seq:
        return None
    tools = tools if tools is not None else alignment.tool_status()
    if not (tools.get("minimap2") and tools.get("samtools")):
        return None
    _stepper(progress)(0.92, "Mapping reads to the plasmid (minimap2 + samtools)…")
    return alignment.map_reads_to_reference(
        params.plasmid_seq, params.fastq_path, inner_seq=target,
        minimap2_bin=tools["minimap2"], samtools_bin=tools["samtools"],
        min_read_len=params.min_read_len, max_read_len=params.max_read_len,
        min_phred=params.min_phred)


def _build_funnel(params: AnalysisParams, result: AlignmentResult,
                  intact: list[str], n_haplotype_reads: Optional[int]) -> list[FunnelStage]:
    """Read counts at each filtering step, and which outputs use each step."""
    proj = result.projection
    spanning = set(proj.spanning_names())
    spanning_intact = [n for n in intact if n in spanning]
    n_span = len(spanning)
    spanning_rate = f"{len(spanning_intact) / n_span:.1%}" if n_span else "-"
    window = "no length filter"
    if params.min_read_len is not None or params.max_read_len is not None:
        lo = f"{params.min_read_len:,}" if params.min_read_len is not None else "0"
        hi = f"{params.max_read_len:,}" if params.max_read_len is not None else "∞"
        window = f"{lo}-{hi} bp"
    quality = f"Q{params.min_phred}+" if params.min_phred is not None else "no quality filter"
    stages = [
        FunnelStage("Reads in FASTQ", proj.n_input, "all reads submitted",
                    ("Read length distribution", "Read quality distribution")),
        FunnelStage("Within read-length window", proj.n_length_kept, window, (),
                    "outside the length window",
                    passed=f"are within the read-length window ({window})",
                    failed=f"fall outside the read-length window ({window})"),
        FunnelStage("Above the quality cutoff", proj.n_quality_kept, quality, (),
                    "below the quality cutoff",
                    passed=f"have a mean Phred of {quality}" if params.min_phred is not None
                           else "are not filtered on quality",
                    failed=f"have a mean Phred below Q{params.min_phred}"),
        FunnelStage("Aligned to the reference", proj.n_aligned,
                    f"{proj.n_unaligned:,} discarded for not aligning at all "
                    "(contaminant DNA, not library)",
                    ("Read alignment map",), "no alignment",
                    passed="align somewhere on the reference",
                    failed="do not align to the reference at all"),
        FunnelStage("Informative about the insert", proj.n_informative,
                    f"{proj.n_uninformative:,} excluded for aligning elsewhere on "
                    "the plasmid without crossing a vector-insert junction, so "
                    "saying nothing about the insert either way",
                    (), "no vector-insert junction crossed",
                    passed="cross a vector-insert junction, so they report on "
                           "how the insert was assembled",
                    failed="align elsewhere on the plasmid without crossing a "
                           "vector-insert junction, so say nothing about the insert"),
        FunnelStage("Contains the insert", proj.n_mapped,
                    f"{proj.n_off_target:,} aligned clear of the insert "
                    "(empty vector)", (), "no insert (empty vector)",
                    passed="carry the insert",
                    failed="align clear of the insert - empty vector"),
        FunnelStage(
            "Correctly assembled", len(intact),
            f"no insertion ≥ {params.structural_insertion_bp} bp or deletion ≥ "
            f"{params.structural_deletion_bp} bp. Of the {n_span:,} reads "
            f"spanning the whole insert, {len(spanning_intact):,} "
            f"({spanning_rate}) are correctly assembled - the rate quoted on the "
            "summary card, since only a spanning read can be judged",
            ("Alignment to reference insert", "AA distribution"),
            "structural indel",
            passed=f"carry no insertion ≥ {params.structural_insertion_bp} bp and "
                   f"no deletion ≥ {params.structural_deletion_bp} bp",
            failed=f"carry an insertion ≥ {params.structural_insertion_bp} bp or a "
                   f"deletion ≥ {params.structural_deletion_bp} bp, or no insert "
                   "at all"),
    ]
    if n_haplotype_reads is not None:
        stages.append(FunnelStage(
            "Called at every variable codon", n_haplotype_reads,
            "intact reads with an unambiguous codon at all selected positions",
            ("Variant treemap",), "a diversified codon unreadable",
            passed="give an unambiguous amino acid at every diversified codon",
            failed="have at least one diversified codon that cannot be called"))
    return stages


def _build_fates(params: AnalysisParams, result: AlignmentResult,
                 calls: dict[str, tuple] | None = None,
                 ref_calls: tuple | None = None) -> list[ReadFate]:
    """Partition every aligned read into exactly one fate."""
    proj = result.projection
    spanning = set(proj.spanning_names())
    counts = dict.fromkeys(FATE_ORDER, 0)

    def usable_fate(name: str) -> str:
        """Sort a structurally sound read by what it proves about diversity.

        Asymmetric: one readable non-reference codon proves a variant whatever
        its neighbours do, while wild type claims *no* designed position changed
        and so needs every diversified codon read. Anything else is ambiguous.
        With no diversified codons selected there is nothing that could differ,
        so a read spanning the insert is wild type by definition.
        """
        if calls is None or ref_calls is None:
            return "Wild type" if name in spanning else "Ambiguous diversity"
        read_calls = calls.get(name)
        if read_calls is None:
            return "Ambiguous diversity"
        if any(aa is not None and aa != ref
               for aa, ref in zip(read_calls, ref_calls)):
            return "Variant"
        if all(aa is not None for aa in read_calls):
            return "Wild type"
        return "Ambiguous diversity"

    counts["Does not contain insert region"] = proj.n_off_target   # empty vector
    for name, st in proj.structures.items():
        if not st.is_intact(params.structural_insertion_bp,
                            params.structural_deletion_bp):
            kind = st.classify(params.structural_insertion_bp,
                               params.structural_deletion_bp)
            counts["Deletion" if kind.startswith("deletion") else "Insertion"] += 1
        else:
            counts[usable_fate(name)] += 1
    return [ReadFate(label, counts[label], label in COMPOSITION_FATES)
            for label in FATE_ORDER]


def tabulate_report(params: AnalysisParams, result: AlignmentResult,
                    read_map: Optional[ReadMap], progress: Progress = None) -> Report:
    """Cheap stage: base/AA counts and haplotypes from the correctly assembled reads."""
    step = _stepper(progress)
    proj = result.projection
    intact = proj.intact_names(params.structural_insertion_bp,
                               params.structural_deletion_bp)
    if not intact:
        raise RuntimeError(
            "No read is free of structural indels, so the library has no correctly "
            "assembled fraction to describe. Check the read alignment map, or raise "
            "the insertion/deletion thresholds.")

    # Reference-anchored rows, keyed like an MSA so the tabulation is unchanged.
    msa = {REF_NAME: result.target, **{n: proj.rows[n] for n in intact}}

    step(0.70, "Tabulating base counts…")
    df_counts, df_freq, _, _ = analysis.counts_from_msa_ref_columns(
        msa, ref_name=REF_NAME, alphabet=BASE_CATEGORIES, ignore_terminal_gaps=True)

    step(0.80, "Tabulating amino-acid counts…")
    df_aa_counts, df_aa_freq, _ = analysis.aa_counts_from_msa(
        msa, ref_name=REF_NAME, frame_offset=0)

    # Manually-specified codons, plus auto-detected variable codons when enabled.
    pie_positions = list(params.pie_positions)
    if params.auto_codon_match_pct is not None:
        ref_seq = result.target[:len(df_counts)]
        pie_positions += analysis.detect_variable_codons(
            df_counts, ref_seq, params.auto_codon_match_pct)
    valid_positions = sorted({p for p in pie_positions if 1 <= p <= df_aa_counts.shape[0]})
    hap_df = None
    if valid_positions:
        step(0.88, "Counting haplotypes…")
        hap_df = analysis.haplotype_counts(
            msa, analysis.get_ref_codons(msa, ref_name=REF_NAME), valid_positions,
            ref_name=REF_NAME,
            unknown=UNCALLED, max_unknown=params.max_unknown_codons)

    calls = ref_calls = None
    if valid_positions:
        ref_codons = analysis.get_ref_codons(msa, ref_name=REF_NAME)
        calls = analysis.read_codon_calls(msa, ref_codons, valid_positions,
                                          ref_name=REF_NAME)
        ref_only = {REF_NAME: result.target, "_ref": result.target}
        ref_calls = analysis.read_codon_calls(ref_only, ref_codons, valid_positions,
                                              ref_name=REF_NAME)["_ref"]

    # Independent of the treemap tolerance: this stage always means every codon
    # read. ``worst`` is the other end - the most any one read is missing.
    n_hap = (sum(1 for c in calls.values() if all(aa is not None for aa in c))
             if calls is not None else None)
    worst = max((sum(aa is None for aa in c) for c in (calls or {}).values()),
                default=0)
    step(1.0, "Done.")
    return Report(
        target=result.target, params=params, n_reads_kept=len(intact),
        length_counts=result.length_counts, mean_phred=result.mean_phred,
        phred_counts=result.phred_counts,
        df_counts=df_counts, df_freq=df_freq, df_aa_counts=df_aa_counts,
        df_aa_freq=df_aa_freq, valid_positions=valid_positions,
        projection=proj, funnel=_build_funnel(params, result, intact, n_hap),
        fates=_build_fates(params, result, calls, ref_calls),
        hap_df=hap_df, read_map=read_map, max_unknown_codons=worst)
