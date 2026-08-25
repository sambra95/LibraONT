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
    """Per-position base counts/frequencies on reference columns (REF != '-').

    With ``ignore_terminal_gaps``, '-' outside a read's covered span is skipped.
    Returns ``(df_counts, df_freq, ref_cols, coverage)``.
    """
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
    """Percentage of non-gap reads carrying the reference base, per column.

    Aligned to ``df_counts`` rows; gaps are out of the denominator.
    """
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
    """Amino-acid counts per codon position from reference columns (REF != '-').

    ``frame_offset`` (0/1/2) is where the ORF starts relative to the first REF base.

    Returns ``(df_aa_counts, df_aa_freq, ref_codons)``.
    """
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


def read_codon_calls(msa: dict[str, str], ref_codons: list[tuple[int, int, int]],
                     positions, ref_name: str = "REF"
                     ) -> dict[str, tuple[str | None, ...]]:
    """Per read, the amino acid at each of ``positions``; ``None`` where the
    codon cannot be read. Kept position-by-position rather than collapsed, so a
    single readable non-reference codon can still settle a read as a variant."""
    pos0 = [p - 1 for p in positions if 1 <= p <= len(ref_codons)]
    out: dict[str, tuple[str | None, ...]] = {}
    for name, row in msa.items():
        if name == ref_name:
            continue
        calls = []
        for p0 in pos0:
            c0, c1, c2 = ref_codons[p0]
            calls.append(_aa_from_triplet(row[c0].upper(), row[c1].upper(),
                                          row[c2].upper()))
        out[name] = tuple(calls)
    return out


def read_haplotypes(msa: dict[str, str], ref_codons: list[tuple[int, int, int]],
                    positions, ref_name: str = "REF") -> dict[str, tuple | None]:
    """Per read, the AA tuple across ``positions``, or ``None`` unless every
    requested codon reads cleanly."""
    return {name: (calls if all(c is not None for c in calls) else None)
            for name, calls in read_codon_calls(msa, ref_codons, positions,
                                                ref_name).items()}


def haplotype_counts(msa: dict[str, str], ref_codons: list[tuple[int, int, int]],
                     positions, ref_name: str = "REF") -> pd.DataFrame:
    """Unique AA combinations across the given 1-based codon positions, from
    reads called at *all* of them. Columns: ``combo_label, combo_tuple, count,
    is_reference, aa_hamming_distance``, descending by count."""
    cols = ["combo_label", "combo_tuple", "count", "is_reference", "aa_hamming_distance"]
    pos_ok = [p for p in positions if 1 <= p <= len(ref_codons)]
    if not pos_ok:
        return pd.DataFrame(columns=cols)

    pos0 = [p - 1 for p in pos_ok]
    ref_row = msa[ref_name]
    ref_hap, ref_ok = [], True
    for p0 in pos0:
        c0, c1, c2 = ref_codons[p0]
        aa = _aa_from_triplet(ref_row[c0].upper(), ref_row[c1].upper(), ref_row[c2].upper())
        if aa is None:
            ref_ok = False
            break
        ref_hap.append(aa)
    ref_tuple = tuple(ref_hap) if ref_ok else None

    ctr: Counter = Counter(
        hap for hap in read_haplotypes(msa, ref_codons, pos_ok, ref_name).values()
        if hap is not None)

    if not ctr:
        return pd.DataFrame(columns=cols)
    rows = []
    for tup, cnt in sorted(ctr.items(), key=lambda kv: kv[1], reverse=True):
        hamming = sum(a != b for a, b in zip(tup, ref_tuple)) if ref_tuple is not None else None
        rows.append({
            "combo_label": " | ".join(f"{p}:{aa}" for p, aa in zip(pos_ok, tup)),
            "combo_tuple": tup,
            "count": cnt,
            "is_reference": tup == ref_tuple,
            "aa_hamming_distance": hamming,
        })
    return pd.DataFrame(rows)


def haplotype_gini(hap_df: pd.DataFrame) -> float:
    """Gini of the haplotype counts: ~0 even, ~1 dominated by a few combos."""
    if hap_df.empty:
        return float("nan")
    counts = np.sort(hap_df["count"].values)
    lorenz_y = np.concatenate([[0], counts.cumsum() / counts.sum()])
    lorenz_x = np.linspace(0, 1, len(lorenz_y))
    trapezoid = getattr(np, "trapezoid", None) or np.trapz   # renamed in NumPy 2
    return float(1 - 2 * trapezoid(lorenz_y, lorenz_x))
