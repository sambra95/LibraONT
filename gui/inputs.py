"""Sidebar input widgets -> the samples to analyse (or a reason why not).

One FASTQ is described on its own; several are treated as time points of one
selection, so each carries a time and they share every other setting.
"""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import streamlit as st

from libraont.constants import (DEFAULT_PIE_MIN_FRAC, DEFAULT_STRUCTURAL_DELETION_BP,
                                DEFAULT_STRUCTURAL_INSERTION_BP)
from libraont.pipeline import AnalysisParams
from libraont.sequences import clean_sequence, fastq_ranges

# One FASTQ: its name, when it was sampled, and how to analyse it.
Sample = tuple[str, float, AnalysisParams]


@st.cache_data(show_spinner="Scanning reads…")
def _cached_ranges(path: str):
    """Read-length and per-read Phred ranges in one FASTQ. A changed upload is
    written to a fresh temp file, so the path alone keys the cache."""
    return fastq_ranges(path)


def save_uploads(uploads) -> dict[str, str]:
    """Temp copy per uploaded FASTQ, keyed by name and reused across reruns;
    copies of files no longer uploaded are deleted."""
    cache = st.session_state.setdefault("_uploads", {})
    keys = {u.name: (u.name, u.size) for u in uploads}
    for name in [n for n, (key, _) in cache.items() if keys.get(n) != key]:
        path = cache.pop(name)[1]
        if os.path.isfile(path):
            os.remove(path)
    for upload in uploads:
        if upload.name not in cache:
            suffix = ".fastq.gz" if upload.name.endswith(".gz") else ".fastq"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(upload.getbuffer())
            tmp.close()
            cache[upload.name] = (keys[upload.name], tmp.name)
    return {u.name: cache[u.name][1] for u in uploads}


def _ranges(paths) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """Read-length and quality ranges spanning every upload, so one filter means
    the same thing for all of them."""
    found = [r for r in map(_cached_ranges, paths) if r]
    if not found:
        return None, None
    return ((min(r[0][0] for r in found), max(r[0][1] for r in found)),
            (min(r[1][0] for r in found), max(r[1][1] for r in found)))


def _time_table(names: list[str]) -> pd.DataFrame:
    """Editable file -> time point table, keeping times already entered.

    The frame handed to the editor is held fixed while the uploads are: the
    editor's identity covers its data, so writing the edits back into it would
    make a new widget and drop the edit in flight."""
    base = st.session_state.get("_time_base")
    if base is None or list(base["FASTQ"]) != names:
        known = st.session_state.get("_sample_times", {})
        base = pd.DataFrame({"FASTQ": names,
                             "Time": [float(known.get(n, i)) for i, n in enumerate(names)]})
        st.session_state["_time_base"] = base
    edited = st.data_editor(base, key="_time_editor:" + "\n".join(names),
                            width="stretch", hide_index=True,
                            num_rows="fixed", disabled=["FASTQ"],
                            column_config={"Time": st.column_config.NumberColumn(
                                "Time", help="When this sample was taken.")})
    st.session_state["_sample_times"] = dict(zip(edited["FASTQ"], edited["Time"]))
    return edited


def parse_positions(text: str) -> list[int]:
    out = []
    for tok in text.replace(";", ",").split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


@st.fragment
def _sidebar() -> None:
    """All sidebar inputs, drawn wherever it is called - ``st.sidebar`` itself
    cannot be used inside a fragment. A fragment, so changing a setting reruns
    the sidebar alone and leaves the results standing; only Run reruns the app.
    What it collects reaches ``render_sidebar`` through session state, since a
    fragment's return value is not passed out."""
    st.header("Inputs")
    uploads = st.file_uploader(
        "FASTQ files", type=["fastq", "fq", "gz"], accept_multiple_files=True,
        help="One nanopore read set to describe a library, or several - one per "
             "time point - to follow a selection over time.")
    paths = save_uploads(uploads or [])
    course = len(paths) > 1
    times = _time_table(list(paths)) if course else None
    time_unit = st.text_input(
        "Time unit", value="Hours",
        help="Names the x axis, e.g. hours or generations.") if course else "Time"

    gene_seq = st.text_area("Gene sequence", height=120,
                            help="Original gene (A/C/G/T/N, case-insensitive).")
    plasmid_seq = st.text_area("Plasmid sequence (optional)", height=80,
                               help="Full plasmid, including the gene above. "
                                    "Enables the read alignment map.")
    gene_len = len(clean_sequence(gene_seq)) if gene_seq else 0

    st.subheader("Initial data analysis")
    st.caption("Applied when reads are filtered and aligned; takes effect on the "
               "next run." + (" Every sample is filtered alike, so the time "
                              "points stay comparable." if course else ""))
    # Length window and quality cutoff, defaulted to the range present in the
    # uploads. Shown even before there is one, so the controls do not appear and
    # disappear.
    rng, q_rng = _ranges(paths.values())
    spread = bool(rng) and rng[0] < rng[1]
    bounds = rng if spread else (0, 1)
    window = st.slider(
        "Read length range (bp)", *bounds, bounds, disabled=not spread,
        help="Keeps reads within this window. Bounds are the shortest and "
             "longest read uploaded.")
    min_read_len, max_read_len = window if spread else (None, None)
    if not spread:
        st.caption(f"All reads are {rng[0]:,} bp long; no length filtering "
                   "applies." if rng else
                   "Upload a FASTQ to filter on read length.")

    q_spread = bool(q_rng) and q_rng[0] < q_rng[1]
    q_bounds = q_rng if q_spread else (0, 1)
    cutoff = st.slider(
        "Minimum read quality (Phred)", *q_bounds, q_bounds[0], disabled=not q_spread,
        help="Drops reads averaging below this Phred score. Bounds span the "
             "uploads, so the bottom keeps every read and the top only the best.")
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
               + ("Every sample is followed at the union of what they detect, so "
                  "a variant means the same thing at every time point."
                  if course else "Drives the AA pies and variant treemap."))
    auto_detect = st.toggle(
        "Auto-detect variable codons", value=True,
        help="Pick variable codons by reference-match %.")
    positions_text = ""
    auto_pct = None
    if auto_detect:
        auto_pct = st.slider(
            "Minimum identity (%)", 0.0, 100.0, 70.0, 1.0,
            help="Codons matching the reference less often than this are "
                 "treated as variable, and marked on the match trace.")
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

    run = st.button("Run analysis", type="primary", width="stretch")

    def build() -> tuple[list[Sample] | None, str | None]:
        if not paths:
            return None, "Upload a FASTQ file."
        if len(paths) < len(uploads):   # same name twice: one sample would vanish
            return None, "Two uploads share a file name."
        if gene_len == 0:
            return None, "Provide a gene sequence."
        try:
            positions = parse_positions(positions_text)
        except ValueError:
            return None, "Codon positions must be integers (comma-separated)."
        if times is not None:
            if times["Time"].isna().any():
                return None, "Give every sample a time point."
            if times["Time"].duplicated().any():
                return None, "Two samples share a time point."
        sampled = (list(zip(times["FASTQ"], times["Time"])) if times is not None
                   else [(name, 0.0) for name in paths])
        return [(name, float(time), AnalysisParams(
            fastq_path=paths[name], gene_seq=gene_seq, fastq_name=name,
            min_read_len=int(min_read_len) if min_read_len is not None else None,
            max_read_len=int(max_read_len) if max_read_len is not None else None,
            min_phred=min_phred,
            plasmid_seq=plasmid_seq.strip() or None,
            structural_insertion_bp=int(structural_insertion_bp),
            structural_deletion_bp=int(structural_deletion_bp),
            pie_positions=positions, pie_min_frac=float(pie_min_frac),
            auto_codon_match_pct=float(auto_pct) if auto_pct is not None else None))
            for name, time in sampled], None

    st.session_state["_inputs"] = build() + (time_unit,)
    if run:                       # the analysis needs the whole app, not just this
        st.session_state["_run"] = True
        st.rerun(scope="app")


def render_sidebar() -> tuple[list[Sample] | None, str | None, str, bool]:
    """Render the inputs. Returns ``(samples, error, time_unit, run_clicked)``."""
    with st.sidebar:
        _sidebar()
    samples, error, time_unit = st.session_state.get("_inputs", (None, None, "Time"))
    return samples, error, time_unit, st.session_state.pop("_run", False)
