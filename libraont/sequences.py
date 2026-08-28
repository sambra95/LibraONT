"""Sequence parsing and lightweight FASTA/FASTQ I/O (no external alignment tools)."""

from __future__ import annotations

import gzip
import re
from collections import Counter
from typing import Iterator

_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(seq: str) -> str:
    """Reverse complement of a DNA string (case preserved)."""
    return seq.translate(_COMPLEMENT)[::-1]


def clean_sequence(seq: str) -> str:
    """Upper-case, keeping only A/C/G/T/N."""
    return ''.join(c for c in seq.upper() if c in "ACGTN")


def collapse_whitespace(seq: str) -> str:
    """Remove all whitespace from a sequence string (preserves case/symbols)."""
    return re.sub(r"\s+", "", str(seq))


def extract_target(gene_seq: str, start_pos: int, stop_pos: int) -> str:
    """Clean ``gene_seq``, then slice 1-based inclusive ``[start_pos, stop_pos]``."""
    return clean_sequence(gene_seq)[start_pos - 1:stop_pos]


def _open_text(path: str):
    """Open a plain or gzipped text file for reading."""
    opener = gzip.open if path.endswith(".gz") else open
    return opener(path, "rt", encoding="utf-8", errors="ignore")


def read_fastq_records(path: str) -> Iterator[tuple[str, str]]:
    """Yield ``(sequence, quality)`` records from a FASTQ file (supports .gz)."""
    with _open_text(path) as fh:
        while True:
            header = fh.readline()
            if not header:
                break
            seq = fh.readline().strip()
            fh.readline()          # '+' separator
            qual = fh.readline().strip()
            if not qual:
                break
            yield seq, qual


def read_fastq(path: str) -> Iterator[str]:
    """Yield each read's sequence from a FASTQ file (supports .gz)."""
    for seq, _qual in read_fastq_records(path):
        yield seq


def mean_phred(qual: str) -> float:
    """Mean Phred score over one read's quality string."""
    return sum(ord(c) - 33 for c in qual) / len(qual)


def fastq_stats(path: str) -> tuple[Counter, float | None, Counter]:
    """Single pass over a FASTQ: (length -> count, mean Phred over all bases,
    per-read mean Phred floored to a whole Q -> count)."""
    counts: Counter = Counter()
    phred_counts: Counter = Counter()
    total_q = total_b = 0
    for seq, qual in read_fastq_records(path):
        counts[len(seq)] += 1
        if qual:
            phred_counts[int(mean_phred(qual))] += 1
            total_q += sum(ord(c) - 33 for c in qual)
            total_b += len(qual)
    return counts, (total_q / total_b if total_b else None), phred_counts


def filter_fastq(in_path: str, out_path: str, min_len: int | None,
                 max_len: int | None, min_phred: int | None = None) -> tuple[int, int]:
    """Copy FASTQ records with length in ``[min_len, max_len]`` (either bound may
    be ``None``) and mean Phred >= ``min_phred`` to plain-text ``out_path``.
    Returns ``(passing the length window, written)``."""
    length_kept = kept = 0
    with _open_text(in_path) as fin, open(out_path, "w", encoding="utf-8") as fout:
        while True:
            header = fin.readline()
            if not header:
                break
            seq = fin.readline()
            plus = fin.readline()
            qual = fin.readline()
            if not qual:
                break
            n = len(seq.strip())
            if min_len is not None and n < min_len:
                continue
            if max_len is not None and n > max_len:
                continue
            length_kept += 1
            q = qual.strip()
            if min_phred is not None and (not q or mean_phred(q) < min_phred):
                continue
            fout.write(header)
            fout.write(seq)
            fout.write(plus)
            fout.write(qual)
            kept += 1
    return length_kept, kept


def fastq_ranges(path: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """One pass: ``((shortest, longest) read length, (lowest, highest) per-read
    mean Phred floored to a whole Q)``, or ``None`` if the FASTQ is empty.

    Flooring keeps both slider bounds inclusive: no read is below the low bound,
    and the best reads still clear the high one."""
    lo = hi = q_lo = q_hi = None
    for seq, qual in read_fastq_records(path):
        n = len(seq)
        lo = n if lo is None or n < lo else lo
        hi = n if hi is None or n > hi else hi
        if qual:
            q = int(mean_phred(qual))
            q_lo = q if q_lo is None or q < q_lo else q_lo
            q_hi = q if q_hi is None or q > q_hi else q_hi
    if lo is None:
        return None
    return (lo, hi), (0, 0) if q_lo is None else (q_lo, q_hi)


def write_fasta(seq: str, out_fa: str, name: str = "ref1", width: int = 60) -> str:
    """Write one sequence to FASTA at ``width`` chars/line; returns the path."""
    s = collapse_whitespace(seq)
    with open(out_fa, "w") as f:
        f.write(f">{name}\n")
        for i in range(0, len(s), width):
            f.write(s[i:i + width] + "\n")
    return out_fa
