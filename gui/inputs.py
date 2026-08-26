"""Sidebar input widgets -> a validated :class:`AnalysisParams` (or a reason why not)."""

from __future__ import annotations

import os
import tempfile

import streamlit as st

from libraont import alignment
from libraont.constants import (DEFAULT_PIE_MIN_FRAC, DEFAULT_STRUCTURAL_DELETION_BP,
                                DEFAULT_STRUCTURAL_INSERTION_BP)
from libraont.pipeline import AnalysisParams
from libraont.sequences import clean_sequence, fastq_ranges


@st.cache_data(show_spinner=False)
def _cached_ranges(path: str, key: tuple):
    """Read-length and per-read Phred ranges in the FASTQ, cached per upload."""
    return fastq_ranges(path)


_TOOL_LABELS = {"minimap2": "minimap2", "samtools": "samtools"}

_TOOL_DESCRIPTIONS = {
    "minimap2": "Required. Aligns every read - the basis for each plot and table.",
    "samtools": "Optional. Backs the read alignment map.",
}


def _resolve_fastq() -> tuple[str | None, str | None]:
    """Path and original filename for the uploaded FASTQ. A new upload replaces
    the previous one, deleting its temp copy first."""
    upload = st.file_uploader(
        "FASTQ file", type=["fastq", "fq", "gz"], accept_multiple_files=False,
        help="One nanopore read set (.fastq/.fastq.gz). A new upload replaces it.")
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
        st.caption("Detected on PATH. A missing tool means the `libraont` "
                   "environment is not active.")
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
                                   help="Full plasmid, including the gene above. "
                                        "Enables the read alignment map.")

        gene_len = len(clean_sequence(gene_seq)) if gene_seq else 0

        st.subheader("Initial data analysis")
        st.caption("Applied when reads are filtered and aligned; changing any of "
                   "these re-runs the analysis.")
        full = st.checkbox("Use full gene length", value=True,
                           help="Uncheck to analyse a sub-region (e.g. one domain).")
        # Shown either way, but frozen on the whole gene while that box is
        # ticked - the value has to be written before the widget is built.
        end = max(gene_len, 1)
        for key, default in (("roi_start", 1), ("roi_stop", end)):
            st.session_state[key] = (default if full else
                                     min(max(st.session_state.get(key, default), 1), end))
        c1, c2 = st.columns(2)
        start_pos = c1.number_input("Start (1-based)", min_value=1, max_value=end,
                                    key="roi_start", disabled=full,
                                    help="First base of the region (1-based).")
        stop_pos = c2.number_input("Stop (inclusive)", min_value=1, max_value=end,
                                   key="roi_stop", disabled=full,
                                   help="Last base of the region (inclusive).")

        # Length window, defaulted to the range present in the FASTQ. Shown even
        # before there is one, so the control does not appear and disappear.
        cached_upload = st.session_state.get("_fastq_upload")
        ranges = (_cached_ranges(fastq_path, cached_upload["key"])
                  if fastq_path and cached_upload else None)
        rng, q_rng = ranges if ranges else (None, None)
        spread = bool(rng) and rng[0] < rng[1]
        bounds = rng if spread else (0, 1)
        window = st.slider(
            "Read length range (bp)", *bounds, bounds, disabled=not spread,
            help="Keeps reads within this window. Bounds are the shortest and "
                 "longest read in the FASTQ.")
        min_read_len, max_read_len = window if spread else (None, None)
        if not spread:
            st.caption(f"All reads are {rng[0]:,} bp long; no length filtering "
                       "applies." if rng else
                       "Upload a FASTQ to filter on read length.")

        # Quality cutoff, over the per-read mean Phred range present in the FASTQ.
        q_spread = bool(q_rng) and q_rng[0] < q_rng[1]
        q_bounds = q_rng if q_spread else (0, 1)
        cutoff = st.slider(
            "Minimum read quality (Phred)", *q_bounds, q_bounds[0], disabled=not q_spread,
            help="Drops reads averaging below this Phred score. Bounds span the "
                 "FASTQ, so the bottom keeps every read and the top only the best.")
        min_phred = int(cutoff) if q_spread else None
        if not q_spread:
            st.caption(f"Every read averages Q{q_rng[0]}; no quality "
                       "filtering applies." if q_rng else
                       "Upload a FASTQ to filter on read quality.")

        c_ins, c_del = st.columns(2)
        structural_insertion_bp = c_ins.number_input(
            "Insertion threshold (bp)", min_value=1, max_value=500,
            value=DEFAULT_STRUCTURAL_INSERTION_BP, step=1,
            help="An insertion this large marks a read mis-assembled, excluding "
                 "it from the composition plots.")
        structural_deletion_bp = c_del.number_input(
            "Deletion threshold (bp)", min_value=1, max_value=500,
            value=DEFAULT_STRUCTURAL_DELETION_BP, step=1,
            help="As above, for deletions - a library can fail one way only.")
        st.caption("ONT indel errors run 1-9 bp and real rearrangements are far "
                   "larger, so anything in ~10-25 bp behaves alike. Reads under "
                   "both count as correctly assembled.")

        st.subheader("Library Analysis settings")
        st.caption("Which codons count as variable, and how results are displayed. "
                   "Drives the AA pies and variant treemap.")
        auto_detect = st.toggle(
            "Auto-detect variable codons", value=True,
            help="Pick variable codons by reference-match %.")
        positions_text = ""
        auto_pct = None
        if auto_detect:
            auto_pct = st.slider(
                "Minimum identity (%)", 0.0, 100.0, 70.0, 1.0,
                help="Codons matching the reference less often than this are "
                     "treated as variable. Marked on the gap/match plot.")
        else:
            positions_text = st.text_input(
                "Codon positions", placeholder="e.g. 16, 129, 231",
                help="1-based codon positions for the pies and treemap.")
        pie_min_frac = st.number_input(
            "Grouping threshold", min_value=0.0, max_value=1.0,
            value=DEFAULT_PIE_MIN_FRAC, step=0.01, format="%.3f",
            help="Amino acids rarer than this are folded into one 'Other' slice, "
                 "and variants carrying them leave the treemap. 0 keeps everything; "
                 "~0.01 hides basecall noise.")
        st.caption("Raising it drops real library members too - in a diverse "
                   "library every residue is rare.")
        # Bounded by the worst read of the last run: a higher tolerance than that
        # cannot admit anything more.
        worst = getattr(st.session_state.get("report"), "max_unknown_codons", 0)
        max_unknown_codons = st.slider(
            "Unknown codons tolerated", 0, worst or 1, 0, disabled=not worst,
            help="Diversified codons a read may miss and still earn a treemap "
                 "tile, each written '?' and counted as a residue of its own. "
                 "0 keeps only fully called reads; the top is the worst read.")
        if not worst:
            max_unknown_codons = 0
            st.caption("Nothing to tolerate: every read covers all the "
                       "diversified codons, or has none to cover.")

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
        min_phred=min_phred,
        plasmid_seq=plasmid_seq.strip() or None,
        structural_insertion_bp=int(structural_insertion_bp),
        structural_deletion_bp=int(structural_deletion_bp),
        pie_positions=positions, pie_min_frac=float(pie_min_frac),
        max_unknown_codons=int(max_unknown_codons),
        auto_codon_match_pct=float(auto_pct) if auto_pct is not None else None)
    return params, run, None
