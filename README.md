# LibraONT

LibraONT is a Streamlit app for nanopore mutagenesis-library analysis. It orients and trims ONT reads against a reference insert, builds a reference-anchored multiple sequence alignment, summarizes nucleotide and amino-acid variation, and reports variant diversity across selected or automatically detected codons.

## What It Does

- Upload a FASTQ/FASTQ.GZ read file.
- Provide a gene sequence and optionally a full plasmid sequence.
- Align reads to the insert with `edlib` and MAFFT.
- Optionally compute plasmid-wide coverage with minimap2 and samtools.
- Plot read-length distribution, insert/plasmid coverage, gap and reference-match percentages, amino-acid distributions, and haplotype treemaps.
- Detect variable codons from a reference-match threshold.
- Export an HTML report with static plots, analysis parameters, reference sequences, and downloadable CSV tables.

## Requirements

- Python 3.10 or newer
- Python dependencies from `pyproject.toml`
- MAFFT for read alignment
- Optional: minimap2 and samtools for plasmid coverage plots

The app can run without minimap2/samtools, but the plasmid coverage plot is skipped unless both are available.

## Install

Clone the repository:

```bash
git clone https://github.com/<your-username>/LibraONT.git
cd LibraONT
```

Install dependencies with `uv`:

```bash
uv sync
```

## External Tools

Install MAFFT:

```bash
brew install mafft
```

Optional coverage tools:

```bash
brew install minimap2 samtools
```

If the app does not find a tool automatically, expand **External tools** in the sidebar, enter the binary path, and press enter. A green tick means the tool has been identified. The app reports detected tool versions in the sidebar and downloaded report.

## Run Locally

```bash
uv run streamlit run app.py
```

The app opens in your browser, usually at:

```text
http://localhost:8501
```

## Inputs

- **FASTQ file**: nanopore reads in `.fastq`, `.fq`, or gzipped FASTQ format.
- **Gene sequence**: reference gene/insert sequence.
- **Plasmid sequence**: optional full plasmid sequence for coverage analysis.
- **Insert region**: use the full gene or choose start/stop positions.
- **Minimum identity**: read/reference identity cutoff for keeping reads.
- **Padding**: extra bases retained around the matched insert when trimming reads.
- **Codon positions or auto-detection**: manually specify codons, or detect variable codons using a reference-match threshold.
- **Grouping threshold**: amino-acid pie slices below this frequency are grouped into `Other`.

## Outputs

The app displays:

- Read length distribution
- Plasmid coverage fraction, when plasmid coverage tools are available
- Gap and reference-match percentage across the insert
- Amino-acid distributions at identified codons
- Variant treemap across identified codons
- Base, amino-acid, and haplotype data tables

Downloads:

- Static HTML report named `ANALYSIS_<FASTQ file name>.html`
- ZIP archive of CSV data tables

## Notes

Variant combinations and amino-acid Hamming distances in the treemap are calculated only across the identified variable codons. The reference-matching variant is outlined in black.
