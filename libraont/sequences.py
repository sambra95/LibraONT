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
    """Upper-case and keep only A/C/G/T/N (drops whitespace, numbers, etc.)."""
    return ''.join(c for c in seq.upper() if c in "ACGTN")


def collapse_whitespace(seq: str) -> str:
    """Remove all whitespace from a sequence string (preserves case/symbols)."""
    return re.sub(r"\s+", "", str(seq))


def extract_target(gene_seq: str, start_pos: int, stop_pos: int) -> str:
    """Insert/target region: clean ``gene_seq`` then slice 1-based inclusive
    ``[start_pos, stop_pos]``."""
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


def read_length_counts(path: str) -> Counter:
    """Counter mapping read length -> number of reads (supports .gz)."""
    counts: Counter = Counter()
    for seq in read_fastq(path):
        counts[len(seq)] += 1
    return counts


def filter_fastq_by_length(in_path: str, out_path: str, min_len: int | None,
                          max_len: int | None) -> int:
    """Copy FASTQ records whose sequence length is within ``[min_len, max_len]``
    (either bound may be ``None``) from ``in_path`` to ``out_path`` (plain text).
    Preserves headers/qualities; supports gzipped input. Returns the kept count."""
    kept = 0
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
            fout.write(header)
            fout.write(seq)
            fout.write(plus)
            fout.write(qual)
            kept += 1
    return kept


def read_length_range(path: str) -> tuple[int, int] | None:
    """Shortest and longest read length in a FASTQ (``None`` if empty)."""
    lo = hi = None
    for seq in read_fastq(path):
        n = len(seq)
        if lo is None or n < lo:
            lo = n
        if hi is None or n > hi:
            hi = n
    return None if lo is None else (lo, hi)


def write_fasta(seq: str, out_fa: str, name: str = "ref1", width: int = 60) -> str:
    """Write a single sequence to FASTA (``width`` chars/line). Returns the path."""
    s = collapse_whitespace(seq)
    with open(out_fa, "w") as f:
        f.write(f">{name}\n")
        for i in range(0, len(s), width):
            f.write(s[i:i + width] + "\n")
    return out_fa
