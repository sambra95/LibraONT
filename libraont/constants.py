"""Static biological constants and default analysis parameters."""

from __future__ import annotations

# Standard genetic code (DNA codon -> single-letter amino acid; '*' = stop).
GENETIC_CODE: dict[str, str] = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L', 'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*', 'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L', 'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q', 'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M', 'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K', 'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V', 'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E', 'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

# Amino-acid display order.
AA_ORDER: list[str] = list("ARNDCQEGHILKMFPSTWYV") + ["*"]

# Per-position base categories used to tabulate the MSA.
BASE_CATEGORIES: tuple[str, ...] = ('A', 'C', 'G', 'T', 'N', 'other', '-')

# Amino acids rarer than this fold into 'Other'. Set low: in a diverse library
# the individual variant residues *are* rare.
DEFAULT_PIE_MIN_FRAC: float = 0.02
# Smallest indel treated as an assembly defect, per direction. ONT indel errors
# are 1-9 bp and real rearrangements much larger, so ~10-25 bp behaves alike.
DEFAULT_STRUCTURAL_INSERTION_BP: int = 10
DEFAULT_STRUCTURAL_DELETION_BP: int = 10
