"""Tabulation of an MSA into base/amino-acid counts and haplotypes.

Pure NumPy/pandas over an ``msa`` mapping ``name -> aligned sequence``, one
entry of which is the reference row (``ref_name``).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .constants import AA_ORDER, BASE_CATEGORIES, GENETIC_CODE

_AA_IDX = {a: i for i, a in enumerate(AA_ORDER)}


def counts_from_msa_ref_columns(msa: dict[str, str], ref_name: str = "REF",
                                alphabet: tuple[str, ...] = BASE_CATEGORIES,
                                ignore_terminal_gaps: bool = True):
    """Per-position base counts/frequencies on reference columns (REF != '-'),
    as ``(df_counts, df_freq, ref_cols, coverage)``. With
    ``ignore_terminal_gaps``, '-' outside a read's covered span is skipped."""
    alphabet = list(alphabet)
    alpha_idx = {b: i for i, b in enumerate(alphabet)}

    ref = msa[ref_name]
    aln_len = len(ref)
    ref_cols = [i for i, ch in enumerate(ref) if ch != '-']
    cols = np.asarray(ref_cols, dtype=np.intp)
    L = len(ref_cols)

    # Byte -> alphabet slot, so each read is bucketed in one vectorised lookup.
    lut = np.full(256, alpha_idx['other'], dtype=np.intp)
    for base in "ACGTN-":
        if base in alpha_idx:
            lut[ord(base)] = alpha_idx[base]

    counts = np.zeros((L, len(alphabet)), dtype=np.int64)
    coverage = np.zeros(L, dtype=np.int64)
    gap = ord('-')

    for name, row in msa.items():
        if name == ref_name:
            continue
        assert len(row) == aln_len
        # latin-1 keeps one byte per character, so any stray symbol still lines
        # up with its column and falls through to 'other'.
        r = np.frombuffer(row.upper().encode("latin-1", "replace"),
                          dtype=np.uint8)[cols]

        if ignore_terminal_gaps:
            seen = np.flatnonzero(r != gap)
            if not seen.size:
                continue
            left, right = int(seen[0]), int(seen[-1])
        else:
            left, right = 0, L - 1
        if right < left:
            continue

        span = np.arange(left, right + 1)
        counts[span, lut[r[left:right + 1]]] += 1
        coverage[left:right + 1] += 1

    df_counts = pd.DataFrame(counts, columns=alphabet, index=np.arange(1, L + 1))
    df_counts.index.name = "position"
    denom = df_counts.sum(axis=1).replace(0, np.nan)
    df_freq = df_counts.div(denom, axis=0)
    cov = pd.Series(coverage, index=df_counts.index, name="coverage")
    return df_counts, df_freq, ref_cols, cov


def reference_match_percent(df_counts: pd.DataFrame, ref_seq: str,
                            gap_char: str = "-") -> np.ndarray:
    """Percentage of non-gap reads carrying the reference base, per column,
    aligned to ``df_counts`` rows; gaps are out of the denominator."""
    L = len(df_counts)
    ref_seq = (ref_seq or "").upper()[:L]
    totals = df_counts.sum(axis=1).astype(float).to_numpy()
    gap = df_counts[gap_char].astype(float).to_numpy()
    col_map = {c: i for i, c in enumerate(df_counts.columns)}
    arr = df_counts.to_numpy(dtype=float)
    match = np.array([arr[i, col_map[b]] if b in col_map else 0.0
                      for i, b in enumerate(ref_seq)], dtype=float)
    non_gap = (totals - gap)[:len(match)]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(non_gap > 0, 100.0 * match / non_gap, 0.0)


def detect_variable_codons(df_counts: pd.DataFrame, ref_seq: str, min_match_pct: float,
                           gap_char: str = "-", frame_offset: int = 0) -> list[int]:
    """1-based codon positions holding a nucleotide below ``min_match_pct``
    reference match, frame-aligned to :func:`aa_counts_from_msa`."""
    match_perc = reference_match_percent(df_counts, ref_seq, gap_char)
    codons = {(j - frame_offset) // 3 + 1
              for j, mp in enumerate(match_perc)
              if j >= frame_offset and mp < min_match_pct}
    return sorted(codons)


def _aa_from_triplet(b0: str, b1: str, b2: str) -> str | None:
    """Translate one codon; ``None`` if gapped, ambiguous or not in the code."""
    if '-' in (b0, b1, b2) or any(b not in "ACGT" for b in (b0, b1, b2)):
        return None
    return GENETIC_CODE.get(b0 + b1 + b2)


def aa_counts_from_msa(msa: dict[str, str], ref_name: str = "REF", frame_offset: int = 0):
    """Amino-acid counts per codon from reference columns (REF != '-'), as
    ``(df_aa_counts, df_aa_freq, ref_codons)``. ``frame_offset`` (0/1/2) is
    where the ORF starts relative to the first REF base."""
    if frame_offset not in (0, 1, 2):
        raise ValueError("frame_offset must be 0, 1, or 2")

    ref = msa[ref_name]
    ref_cols = [i for i, ch in enumerate(ref) if ch != '-'][frame_offset:]
    num_codons = len(ref_cols) // 3
    ref_codons = [(ref_cols[3 * k], ref_cols[3 * k + 1], ref_cols[3 * k + 2])
                  for k in range(num_codons)]

    counts = np.zeros((num_codons, len(AA_ORDER)), dtype=np.int64)
    for name, row in msa.items():
        if name == ref_name:
            continue
        seq = row.upper()
        for k, (c0, c1, c2) in enumerate(ref_codons):
            aa = _aa_from_triplet(seq[c0], seq[c1], seq[c2])
            if aa is None:
                continue
            counts[k, _AA_IDX[aa]] += 1

    df_counts = pd.DataFrame(counts, columns=AA_ORDER, index=np.arange(1, num_codons + 1))
    df_counts.index.name = "codon_position"
    row_sums = df_counts.sum(axis=1).replace(0, np.nan)
    return df_counts, df_counts.div(row_sums, axis=0), ref_codons


def get_ref_codons(msa: dict[str, str], ref_name: str = "REF") -> list[tuple[int, int, int]]:
    """Alignment-column triplets for each reference codon."""
    ref = msa[ref_name]
    ref_codons, current = [], []
    for col, base in enumerate(ref):
        if base != "-":
            current.append(col)
            if len(current) == 3:
                ref_codons.append(tuple(current))
                current = []
    return ref_codons


def codon_calls(row: str, ref_codons: list[tuple[int, int, int]],
                positions) -> tuple[str | None, ...]:
    """The amino acid at each 1-based codon in ``positions`` for one aligned
    row; ``None`` where the codon cannot be read."""
    up = row.upper()
    return tuple(_aa_from_triplet(up[c0], up[c1], up[c2])
                 for c0, c1, c2 in (ref_codons[p - 1] for p in positions
                                    if 1 <= p <= len(ref_codons)))


def read_codon_calls(msa: dict[str, str], ref_codons: list[tuple[int, int, int]],
                     positions, ref_name: str = "REF"
                     ) -> dict[str, tuple[str | None, ...]]:
    """:func:`codon_calls` per read. Kept position-by-position rather than
    collapsed, so one readable non-reference codon settles a read as a variant."""
    return {name: codon_calls(row, ref_codons, positions)
            for name, row in msa.items() if name != ref_name}


def call_matrix(calls: dict[str, tuple[str | None, ...]]) -> np.ndarray:
    """Reads called at every position, amino acids as integer codes - the
    pattern the sampling and covariation plots need, without the letters."""
    rows = [c for c in calls.values() if c and all(aa is not None for aa in c)]
    if not rows:
        return np.empty((0, 0), dtype=np.int16)
    arr = np.array(rows)
    codes = np.empty(arr.shape, dtype=np.int16)
    for j in range(arr.shape[1]):
        codes[:, j] = np.unique(arr[:, j], return_inverse=True)[1]
    return codes


def haplotype_counts(calls: dict[str, tuple[str | None, ...]],
                     ref_calls: tuple[str | None, ...] | None, positions, *,
                     unknown: str = "?", max_unknown: int = 0) -> pd.DataFrame:
    """Unique AA combinations over ``positions``, from the reads in ``calls``
    (see :func:`read_codon_calls`) missing at most ``max_unknown`` of them; an
    unreadable codon carries ``unknown`` and counts as a residue in its own
    right. Columns: ``combo_label, combo_tuple, count, is_reference,
    aa_hamming_distance``, descending by count."""
    cols = ["combo_label", "combo_tuple", "count", "is_reference", "aa_hamming_distance"]
    ctr: Counter = Counter(
        tuple(c if c is not None else unknown for c in call)
        for call in calls.values() if sum(c is None for c in call) <= max_unknown)
    if not ctr:
        return pd.DataFrame(columns=cols)
    ref_tuple = (tuple(ref_calls) if ref_calls and all(a is not None for a in ref_calls)
                 else None)
    return pd.DataFrame([
        {"combo_label": " | ".join(f"{p}:{aa}" for p, aa in zip(positions, tup)),
         "combo_tuple": tup,
         "count": cnt,
         "is_reference": tup == ref_tuple,
         "aa_hamming_distance": (sum(a != b for a, b in zip(tup, ref_tuple))
                                 if ref_tuple is not None else None)}
        for tup, cnt in sorted(ctr.items(), key=lambda kv: kv[1], reverse=True)])
