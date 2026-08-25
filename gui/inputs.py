"""Sidebar input widgets -> a validated :class:`AnalysisParams` (or a reason why not)."""

from __future__ import annotations

import os
import tempfile

import streamlit as st

from libraont import alignment
from libraont.constants import (DEFAULT_PIE_MIN_FRAC, DEFAULT_STRUCTURAL_DELETION_BP,
                                DEFAULT_STRUCTURAL_INSERTION_BP)
from libraont.pipeline import AnalysisParams
from libraont.sequences import clean_sequence, read_length_range


@st.cache_data(show_spinner=False)
def _cached_length_range(path: str, key: tuple) -> tuple[int, int] | None:
    """Min/max raw read length in the FASTQ, cached per upload (``key``)."""
    return read_length_range(path)


_TOOL_LABELS = {"minimap2": "minimap2", "samtools": "samtools"}

_TOOL_DESCRIPTIONS = {
    "minimap2": "Required. Aligns every read and keeps the insertions it carries - the "
                "basis for read structure, base/AA composition and the variant treemap.",
    "samtools": "Optional. Sorts and indexes the plasmid alignments behind the "
                "read alignment map.",
}


def _resolve_fastq() -> tuple[str | None, str | None]:
    """Path and original filename for the uploaded FASTQ. A new upload replaces
    the previous one, deleting its temp copy first."""
    upload = st.file_uploader(
        "FASTQ file", type=["fastq", "fq", "gz"], accept_multiple_files=False,
        help="Upload a single nanopore read set (.fastq or .fastq.gz). "
             "Uploading a new file replaces the current one.")
    if upload is None:
        return None, None

    # One temp file, reused across reruns; a different upload (by name/size)
    # replaces it.
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


def _tool_status() -> None:
    """Read-only detection status for the external tools (found on PATH or not)."""
    with st.expander("External tools", expanded=False):
        st.caption("Detected on PATH by the active environment. A missing tool means "
                   "the `libraont` conda environment is not active.")
        versions = alignment.tool_versions()
        for name, label in _TOOL_LABELS.items():
            version = versions.get(name)
            st.markdown(f"{'✅' if version else '⚠️'} **{label}** - "
                        + (f"`{version}`" if version else "not found"))
            st.caption(_TOOL_DESCRIPTIONS[name])


def render_sidebar() -> tuple[AnalysisParams | None, bool, str | None]:
    """Render all inputs. Returns ``(params_or_None, run_clicked, error_message)``."""
    st.sidebar.header("Inputs")
    with st.sidebar:
        fastq_path, fastq_name = _resolve_fastq()

        gene_seq = st.text_area("Gene sequence", height=120,
                                help="Original gene (A/C/G/T/N, case-insensitive).")
        plasmid_seq = st.text_area("Plasmid sequence (optional)", height=80,
                                   help="Full plasmid; enables the read alignment map. Must "
                                        "include the target/gene sequence, which is located "
                                        "within the plasmid to place it on the map.")

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

        # Length window, defaulted to the range present in the FASTQ.
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

        c_ins, c_del = st.columns(2)
        structural_insertion_bp = c_ins.number_input(
            "Insertion threshold (bp)", min_value=1, max_value=500,
            value=DEFAULT_STRUCTURAL_INSERTION_BP, step=1,
            help="A read carrying an insertion at least this large is treated as "
                 "mis-assembled and excluded from every composition plot.")
        structural_deletion_bp = c_del.number_input(
            "Deletion threshold (bp)", min_value=1, max_value=500,
            value=DEFAULT_STRUCTURAL_DELETION_BP, step=1,
            help="As above, for deletions - set apart from the insertion "
                 "threshold because a library can fail one way without the other.")
        st.caption("Separate these from basecall noise: ONT indel errors are 1-9 bp "
                   "and real rearrangements are much larger, so anything in ~10-25 bp "
                   "behaves the same. Reads under both are the 'correctly assembled' "
                   "fraction.")

        st.subheader("Library Analysis settings")
        st.caption("Adjust the analysis and how results are displayed. The identity "
                   "filter re-runs the alignment; codon/pie settings update instantly. "
                   "Codon selection drives the AA-distribution pies and variant treemap "
                   "(and marks a cutoff on the alignment plot).")
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
            st.caption("Codon-level cutoff: codons matching the reference less "
                       "often than this are treated as variable.")
        else:
            positions_text = st.text_input(
                "Codon positions", placeholder="e.g. 16, 129, 231",
                help="1-based codon positions for AA pies & haplotypes.")
        pie_min_frac = st.number_input(
            "Grouping threshold", min_value=0.0, max_value=1.0,
            value=DEFAULT_PIE_MIN_FRAC, step=0.01, format="%.3f",
            help="Amino acids below this frequency at their codon are folded into "
                 "one 'Other' slice in the AA distribution, and variants carrying "
                 "such a residue are dropped from the variant treemap. 0 keeps "
                 "everything; ~0.01 suppresses basecall-noise residues.")
        st.caption("Raising this removes real library members as well as noise - "
                   "in a diverse library the individual variant residues are rare.")

        _tool_status()
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
        min_read_len=int(min_read_len) if min_read_len is not None else None,
        max_read_len=int(max_read_len) if max_read_len is not None else None,
        plasmid_seq=plasmid_seq.strip() or None,
        structural_insertion_bp=int(structural_insertion_bp),
        structural_deletion_bp=int(structural_deletion_bp),
        pie_positions=positions, pie_min_frac=float(pie_min_frac),
        auto_codon_match_pct=float(auto_pct) if auto_pct is not None else None)
    return params, run, None
