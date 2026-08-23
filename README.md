# LibraONT

LibraONT is a Streamlit app for nanopore mutagenesis-library analysis. It orients and
trims ONT reads against a reference insert, builds a reference-anchored multiple sequence
alignment, and summarises nucleotide, amino-acid and variant diversity across selected or
automatically detected codons.

## What it does

1. Orient and trim each read to the reference insert (`edlib`) and filter by identity and length.
2. Build a reference-anchored multiple sequence alignment (MAFFT).
3. Tabulate per-position base and amino-acid composition.
4. Report variant diversity (haplotypes) across the chosen codons.
5. Optionally compute whole-plasmid coverage (minimap2 + samtools).

## Inputs

- **FASTQ file** — nanopore reads (`.fastq`, `.fq`, or gzipped).
- **Gene sequence** — the reference gene/insert (A/C/G/T/N, case-insensitive).
- **Plasmid sequence** *(optional)* — full plasmid; enables the coverage plot.
- **Insert region** — the full gene, or a start/stop sub-region.
- **Minimum identity** — read-to-insert identity cutoff for keeping reads.
- **Read length range / padding** — length window for reads, and bases kept either side when trimming.
- **Codon positions** — typed manually, or auto-detected from a reference-match threshold.
- **Grouping threshold / rare variants** — fold low-frequency amino acids into `Other` and include or exclude rare variants in the treemap.

## Outputs

Displayed in the app:

- Read-length distribution
- Plasmid coverage fraction (when a plasmid and the coverage tools are available)
- Gap and reference-match percentage across the insert
- Amino-acid distributions at the identified codons
- Variant treemap across the identified codons
- Base, amino-acid and haplotype data tables

Downloads:

- Static HTML report, `ANALYSIS_<FASTQ file name>.html`
- ZIP archive of the CSV data tables

## Install

Requires [conda or mamba](https://github.com/conda-forge/miniforge). The environment file
bundles Python, the app's Python dependencies, and the external tools (MAFFT, minimap2,
samtools) — no separate installs needed.

```bash
git clone https://github.com/sambra95/LibraONT.git
cd LibraONT
conda env create -f environment.yml   # or: mamba env create -f environment.yml
conda activate libraont
```

The app looks for MAFFT, minimap2 and samtools on `PATH`, so the environment must be active
when you launch it. **External tools** in the sidebar confirms detection (a green tick per tool).

## Run

```bash
streamlit run app.py
```

The app opens in your browser, usually at <http://localhost:8501>.

## Deploy to Streamlit Cloud

The repo is deployment-ready: [`requirements.txt`](requirements.txt) provides the Python
dependencies and [`packages.txt`](packages.txt) installs MAFFT, minimap2 and samtools on the
Cloud image. Point a new app at `app.py` and select Python 3.12 in the advanced settings.

## Notes

Variant combinations and amino-acid Hamming distances in the treemap are calculated only
across the identified variable codons. The reference-matching variant is outlined in black.
The coverage plot is skipped unless a plasmid sequence is provided and both minimap2 and
samtools are available.
