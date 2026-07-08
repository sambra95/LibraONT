"""Sidebar input widgets -> a validated :class:`AnalysisParams` (or a reason why not)."""

from __future__ import annotations

import os
import tempfile

import streamlit as st

from libraont import alignment
from libraont.constants import (DEFAULT_MIN_IDENTITY, DEFAULT_PAD,
                                DEFAULT_PIE_MIN_FRAC)
from libraont.pipeline import AnalysisParams
from libraont.sequences import clean_sequence, read_length_range


@st.cache_data(show_spinner=False)
def _cached_length_range(path: str, key: tuple) -> tuple[int, int] | None:
    """Min/max raw read length in the FASTQ, cached per upload (``key``)."""
    return read_length_range(path)


_TOOL_LABELS = {"mafft": "MAFFT", "minimap2": "minimap2", "samtools": "samtools"}

_TOOL_DESCRIPTIONS = {
    "mafft": "Reference-anchored multiple alignment of the reads - the backbone for "
             "base/AA composition, variable-codon detection and the variant treemap.",
    "minimap2": "Maps reads to the whole plasmid (ONT preset) for the coverage plot.",
    "samtools": "Sorts and indexes the minimap2 alignments and computes per-base depth "
                "for the coverage plot.",
}


def _resolve_fastq() -> tuple[str | None, str | None]:
    """Return a readable path and original filename for the single uploaded FASTQ.

    Only one file may be uploaded; uploading a new one replaces the previous -
    its temp copy is deleted first so a stale FASTQ never lingers on disk.
    """
    upload = st.file_uploader(
        "FASTQ file", type=["fastq", "fq", "gz"], accept_multiple_files=False,
        help="Upload a single nanopore read set (.fastq or .fastq.gz). "
             "Uploading a new file replaces the current one.")
    if upload is None:
        return None, None

    # Persist the upload to one temp file, reused across reruns. A different
    # upload (by name/size) overwrites it; the old temp file is removed first.
    cached = st.session_state.get("_fastq_upload")
    key = (upload.name, upload.size)
    if not cached or cached["key"] != key:
        if cached and os.path.isfile(cached["path"]):
            os.remove(cached["path"])
        suffix = ".fastq.gz" if upload.name.endswith(".gz") else ".fastq"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(upload.getbuffer())
        tmp.close()
        st.session_state["_fastq_upload"] = {
            "key": key, "path": tmp.name, "name": upload.name}
    cached = st.session_state["_fastq_upload"]
    return cached["path"], cached.get("name", upload.name)


def _parse_positions(text: str) -> list[int]:
    out = []
    for tok in text.replace(";", ",").split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


def _tool_paths() -> dict[str, str]:
    """Optional binary-path overrides; shows detected status for each tool."""
    overrides: dict[str, str] = {}
    with st.expander("External tools", expanded=False):
        st.caption("External tools. Update the paths and press enter to change these. "
                   "Green tick means the tool has been identified")
        status = alignment.tool_status()
        versions = alignment.tool_versions()
        for name, label in _TOOL_LABELS.items():
            found = status[name]
            version = versions.get(name)
            st.markdown(f"{'✅' if found else '⚠️'} **{label}** - "
                        + (f"`{version}`" if version else "not found"))
            st.caption(_TOOL_DESCRIPTIONS[name])
            val = st.text_input(f"{label} path", key=f"tool_{name}", value=found or "",
                                help=f"Path to the {label} binary. Pre-filled with the "
                                     "detected default; change it to use another build.")
            if val.strip():
                overrides[name] = val.strip()
    return overrides


def render_sidebar() -> tuple[AnalysisParams | None, bool, str | None]:
    """Render all inputs. Returns ``(params_or_None, run_clicked, error_message)``."""
    st.sidebar.header("Inputs")
    with st.sidebar:
        fastq_path, fastq_name = _resolve_fastq()

        gene_seq = st.text_area("Gene sequence", height=120,
                                help="Original gene (A/C/G/T/N, case-insensitive).")
        plasmid_seq = st.text_area("Plasmid sequence (optional)", height=80,
                                   help="Full plasmid; enables the coverage plot. Must "
                                        "include the target/gene sequence, which is located "
                                        "within the plasmid to place it on the coverage plot.")

        gene_len = len(clean_sequence(gene_seq)) if gene_seq else 0

        st.subheader("Initial data analysis")
        st.caption("Applied once when aligning and filtering reads. Changing any "
                   "of these re-runs the analysis and affects every plot and table.")
        full = st.checkbox("Use full gene length", value=True,
                           help="Analyse the whole gene. Uncheck to restrict the "
                                "analysis to a sub-region (e.g. a single domain).")
        if full:
            start_pos, stop_pos = 1, max(gene_len, 1)
        else:
            c1, c2 = st.columns(2)
            start_pos = c1.number_input("Start (1-based)", min_value=1,
                                        max_value=max(gene_len, 1), value=1,
                                        help="First base of the region of interest "
                                             "(1-based, inclusive).")
            stop_pos = c2.number_input("Stop (inclusive)", min_value=1,
                                       max_value=max(gene_len, 1), value=max(gene_len, 1),
                                       help="Last base of the region of interest (inclusive).")

        pad = st.number_input("Padding (bp)", min_value=0, value=DEFAULT_PAD, step=10,
                              help="Extra bases kept on each side of the matched region "
                                   "when trimming each read. Increase to retain flanks.")

        # Read-length window, bounded and defaulted to the range actually present
        # in the uploaded FASTQ. Reads outside the selected window are discarded
        # before alignment.
        min_read_len = max_read_len = None
        cached_upload = st.session_state.get("_fastq_upload")
        if fastq_path and cached_upload:
            rng = _cached_length_range(fastq_path, cached_upload["key"])
            if rng and rng[0] < rng[1]:
                lo, hi = rng
                min_read_len, max_read_len = st.slider(
                    "Read length range (bp)", lo, hi, (lo, hi),
                    help="Only reads whose length falls within this window are kept. "
                         "Bounds default to the shortest and longest read in the FASTQ.")
            elif rng:
                st.caption(f"All reads are {rng[0]} bp long; no length filtering applied.")

        st.subheader("Library Analysis settings")
        st.caption("Adjust the analysis and how results are displayed. The identity "
                   "filter re-runs the alignment; codon/pie settings update instantly. "
                   "Codon selection drives the AA-distribution pies and variant treemap "
                   "(and marks a cutoff on the alignment plot).")
        min_identity = st.slider(
            "Minimum identity", 0.0, 1.0, DEFAULT_MIN_IDENTITY, 0.01,
            help="Minimum identity to the reference insert. Reads that don't align to the "
                 "insert at or above this are discarded - this is the filter behind the "
                 "'Alignment to reference insert' plot, and it sets the reads used across "
                 "the analysis (AA distribution and variant treemap too). Lower it if too "
                 "few reads pass.")
        auto_detect = st.toggle(
            "Auto-detect variable codons", value=True,
            help="Use reference-match % to select codons automatically.")
        positions_text = ""
        auto_pct = None
        if auto_detect:
            auto_pct = st.slider(
                "Minimum identity (%)", 0.0, 100.0, 70.0, 1.0,
                help="Codons with a position below this reference-match % are added "
                     "automatically, exactly as if typed into 'Codon positions'. "
                     "Shown as a cutoff line on the gap/match plot.")
        else:
            positions_text = st.text_input(
                "Codon positions", placeholder="e.g. 16, 129, 231",
                help="1-based codon positions for AA pies & haplotypes.")
        pie_min_frac = st.number_input("Grouping threshold", min_value=0.0, max_value=1.0,
                                       value=DEFAULT_PIE_MIN_FRAC, step=0.01,
                                       help="AA-distribution plot only: amino acids below "
                                            "this frequency are folded into a single "
                                            "'Other' slice.")
        include_rare_variants = st.toggle(
            "Include rare variants", value=False,
            help="Variant treemap only: when off, variants containing an amino acid below the "
                 "grouping threshold at its codon position (an 'Other' residue) are excluded, "
                 "and the treemap stats reflect the remaining variants.")

        tool_paths = _tool_paths()
        run = st.button("Run analysis", type="primary", use_container_width=True)

    # Validation.
    if not fastq_path:
        return None, run, "Upload a FASTQ file."
    if gene_len == 0:
        return None, run, "Provide a gene sequence."
    try:
        positions = _parse_positions(positions_text)
    except ValueError:
        return None, run, "Codon positions must be integers (comma-separated)."

    params = AnalysisParams(
        fastq_path=fastq_path, gene_seq=gene_seq,
        start_pos=int(start_pos), stop_pos=int(stop_pos),
        fastq_name=fastq_name,
        min_identity=float(min_identity), pad=int(pad),
        min_read_len=int(min_read_len) if min_read_len is not None else None,
        max_read_len=int(max_read_len) if max_read_len is not None else None,
        plasmid_seq=plasmid_seq.strip() or None,
        pie_positions=positions, pie_min_frac=float(pie_min_frac),
        include_rare_variants=bool(include_rare_variants),
        auto_codon_match_pct=float(auto_pct) if auto_pct is not None else None,
        tool_paths=tool_paths)
    return params, run, None
