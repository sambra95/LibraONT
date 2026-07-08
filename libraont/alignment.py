"""Read orientation/trimming, reference-anchored MSA, and coverage computation.

This module wraps the external tools:
  * ``edlib`` (Python) for orient + trim,
  * ``mafft`` for the reference-anchored multiple alignment,
  * ``minimap2`` + ``samtools`` for whole-plasmid coverage.

External binaries are resolved via :func:`resolve_tool` (explicit path, then the
matching ``*_BIN`` env var, then ``PATH``) so the app degrades gracefully when a
tool is missing rather than crashing.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass, field

import edlib
import numpy as np

from .sequences import (filter_fastq_by_length, read_fastq_records, revcomp,
                        write_fasta, collapse_whitespace)

# Map a logical tool name to the env var that can override its path.
_ENV_VARS = {"mafft": "MAFFT_BIN", "minimap2": "MINIMAP2_BIN", "samtools": "SAMTOOLS_BIN"}


def resolve_tool(name: str, override: str | None = None) -> str | None:
    """Resolve a binary: explicit ``override`` -> ``<NAME>_BIN`` env -> ``PATH``."""
    for candidate in (override,
                      os.environ.get(_ENV_VARS.get(name, ""), None),
                      name):
        if candidate and shutil.which(candidate):
            return shutil.which(candidate)
    return None


def tool_status(overrides: dict[str, str] | None = None) -> dict[str, str | None]:
    """Resolved path (or None) for each external tool."""
    overrides = overrides or {}
    return {name: resolve_tool(name, overrides.get(name)) for name in _ENV_VARS}


def _samtools_banner_version(path: str) -> str | None:
    """Parse the version from samtools' no-argument help banner.

    Some samtools builds print nothing to ``--version`` but still show a
    ``Version: 1.x (using htslib ...)`` line in the banner they print (to
    stderr) when run with no arguments. Used as a fallback for those builds.
    """
    try:
        # Decode as UTF-8 with replacement: hosts with a C/ASCII locale would
        # otherwise raise UnicodeDecodeError on the non-ASCII bytes samtools
        # prints (e.g. the copyright symbol in its banner).
        proc = subprocess.run([path], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              encoding="utf-8", errors="replace", timeout=20)
    except Exception:
        return None
    text = "\n".join(part for part in (proc.stdout, proc.stderr) if part)
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("version:"):
            tokens = line.split(":", 1)[1].split()
            if tokens:
                return tokens[0]
    return None


@functools.lru_cache(maxsize=None)
def _tool_version(name: str, path: str | None) -> str | None:
    """Return a concise version string for a resolved external tool.

    Cached by (name, resolved path): a binary's version is stable within a run,
    so this avoids re-spawning the tool on every Streamlit rerun - important on
    resource-constrained hosts where a cold start can be slow (see the timeout).

    When a version can't be read it returns a short, *distinct* reason
    (timeout / error / no output) rather than a single generic label, so a
    genuine detection failure is visible and diagnosable from the UI.
    """
    if not path:
        return None
    try:
        # Decode as UTF-8 with replacement so a host's C/ASCII locale can't
        # raise UnicodeDecodeError on samtools' non-ASCII banner bytes. Generous
        # timeout for slow cold-starts on constrained hosts (e.g. Streamlit Cloud).
        proc = subprocess.run([path, "--version"], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, encoding="utf-8",
                              errors="replace", timeout=20)
    except subprocess.TimeoutExpired:
        return "found (version timed out)"
    except Exception as exc:  # on PATH but won't execute (e.g. missing shared lib)
        return f"found ({type(exc).__name__})"

    output = "\n".join(part.strip() for part in (proc.stdout, proc.stderr) if part.strip())
    if output:
        first = output.splitlines()[0].strip()
        if name == "samtools":
            return first.removeprefix("samtools ").strip() or first
        return first

    # ``--version`` produced no output. samtools carries its version in the
    # no-argument help banner, so try that before giving up.
    if name == "samtools":
        banner = _samtools_banner_version(path)
        if banner:
            return banner
    return "found (no version output)"


def tool_versions(overrides: dict[str, str] | None = None) -> dict[str, str | None]:
    """Concise version string (or None) for each external tool."""
    status = tool_status(overrides)
    return {name: _tool_version(name, path) for name, path in status.items()}


# --- Orientation + trimming -------------------------------------------------
_EDLIB_CIGAR_RE = re.compile(r"(\d+)([=XID])")


def _overlap_identity(cigar: str) -> float:
    """Identity over the overlapping region of an edlib HW alignment (query = the
    reference insert). Terminal ``I`` runs - insert bases the read does not reach -
    are excluded, so a truncated but accurate read is scored only on the portion it
    covers, not penalised for partial coverage.
    """
    ops = [(int(n), op) for n, op in _EDLIB_CIGAR_RE.findall(cigar)]
    while ops and ops[0][1] == "I":
        ops.pop(0)
    while ops and ops[-1][1] == "I":
        ops.pop()
    matches = sum(n for n, op in ops if op == "=")
    columns = sum(n for n, op in ops)
    return matches / columns if columns else 0.0


def edlib_orient_trim_and_quality(fastq_path: str, target: str, min_identity: float = 0.65,
                                  pad: int = 200, drop_short: bool = True,
                                  min_read_len: int | None = None,
                                  max_read_len: int | None = None,
                                  ) -> tuple[list[tuple[str, str]], float | None]:
    """For each read: find the best strand/placement vs ``target`` (edlib HW),
    trim to ``[start-pad : end+1+pad]`` on that strand, and keep it only if its
    identity over the overlapping region (see :func:`_overlap_identity`) is
    >= ``min_identity`` - partial coverage of the insert is not penalised.

    Reads whose raw length falls outside ``[min_read_len, max_read_len]`` are
    skipped (either bound may be ``None`` to disable it).

    Returns kept sequences plus the mean Phred score across kept full reads.
    """
    L = len(target)
    out: list[tuple[str, str]] = [("REF", target)]
    total_phred = 0
    total_bases = 0
    for i, (seq, qual) in enumerate(read_fastq_records(fastq_path), 1):
        r = seq.strip().upper()
        if not r:
            continue
        if min_read_len is not None and len(r) < min_read_len:
            continue
        if max_read_len is not None and len(r) > max_read_len:
            continue

        rf = edlib.align(target, r, task="path", mode="HW")
        rr = edlib.align(target, revcomp(r), task="path", mode="HW")
        use_rev = rr["editDistance"] < rf["editDistance"]
        res = rr if use_rev else rf
        if not res["locations"]:
            continue
        if _overlap_identity(res["cigar"]) < min_identity:
            continue

        rseq = revcomp(r) if use_rev else r
        start, end = res["locations"][0]  # edlib end is inclusive
        trimmed = rseq[max(0, start - pad):min(len(rseq), end + 1 + pad)]
        if drop_short and len(trimmed) < max(50, L // 4):
            continue

        out.append((f"read_{i}", trimmed))
        total_phred += sum(ord(ch) - 33 for ch in qual)
        total_bases += len(qual)
    mean_phred = total_phred / total_bases if total_bases else None
    return out, mean_phred


# --- Reference-anchored MAFFT alignment -------------------------------------
def _parse_fasta(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    name, buf = None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if name is not None:
                out[name] = "".join(buf)
            name, buf = line[1:].strip(), []
        else:
            buf.append(line.strip())
    if name is not None:
        out[name] = "".join(buf)
    return out


def run_msa_mafft(seqs_named: list[tuple[str, str]], mafft_bin: str = "mafft", threads: int = 1,
                  op: float = 3.5, ep: float = 1.0, extra_args: list[str] | None = None,
                  batch_size: int = 300) -> dict[str, str]:
    """Reference-anchored alignment in batches: build a REF-only backbone, then add
    fragments per batch with ``mafft --addfragments --keeplength`` and merge the
    rows (REF is identical across batches)."""
    if extra_args is None:
        extra_args = ["--localpair", "--maxiterate", "1000", "--quiet"]
    assert seqs_named and seqs_named[0][0] == "REF", "first entry must be ('REF', ...)"
    ref_name, ref_seq = seqs_named[0]
    reads = seqs_named[1:]

    def run_batch(backbone_fa: str, batch_reads: list[tuple[str, str]]) -> dict[str, str]:
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".fa") as rfa:
            for n, s in batch_reads:
                rfa.write(f">{n}\n{textwrap.fill(s, 80)}\n")
            rfa_name = rfa.name
        cmd = [mafft_bin, "--thread", str(threads), "--addfragments", rfa_name,
               "--keeplength", "--anysymbol", "--op", str(op), "--ep", str(ep),
               *extra_args, backbone_fa]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              encoding="utf-8", errors="replace")
        os.unlink(rfa_name)
        if proc.returncode != 0 or not proc.stdout.strip():
            raise RuntimeError(f"MAFFT addfragments failed.\nCMD: {' '.join(cmd)}\n"
                               f"STDERR:\n{proc.stderr}")
        return _parse_fasta(proc.stdout)

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".fa") as bfa:
        bfa.write(f">{ref_name}\n{textwrap.fill(ref_seq, 80)}\n")
        backbone_fa = bfa.name

    msa: dict[str, str] = {}
    L = 0
    try:
        for i in range(0, len(reads), batch_size):
            batch = reads[i:i + batch_size]
            out = run_batch(backbone_fa, batch)
            if not msa:
                msa[ref_name] = out[ref_name]
                L = len(msa[ref_name])
            elif len(out[ref_name]) != L:
                raise ValueError("Backbone length changed across batches.")
            for n, _ in batch:
                # MAFFT can drop too-short / no-hit fragments; treat as all gaps.
                msa[n] = out.get(n, "-" * L)
    finally:
        os.unlink(backbone_fa)

    if len({len(s) for s in msa.values()}) != 1:
        raise ValueError("Merged MSA rows are not equal length.")
    return msa


# --- Whole-plasmid coverage (minimap2 + samtools) ---------------------------
@dataclass
class CoverageResult:
    positions: np.ndarray
    fractions: np.ndarray
    mapped_reads: int
    contig: str
    region: dict | None = None  # {'start','end','strand'} or None
    # Per-read alignment span on the contig (1-based inclusive) and identity, for
    # the read-alignment map. Parallel arrays over primary alignments.
    read_starts: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    read_ends: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    read_identities: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    contig_length: int = 0


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    # encoding/errors (not universal_newlines) so a C/ASCII-locale host can't
    # raise UnicodeDecodeError on non-ASCII bytes in tool output.
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          encoding="utf-8", errors="replace", check=True, **kw)


# CIGAR operations by what they consume: reference span (a read's footprint on
# the contig) and alignment columns (the denominator for BLAST-style identity).
_CIGAR_RE = re.compile(r"(\d+)([MIDNSHP=X])")
_REF_CONSUMING = frozenset("MDN=X")
_ALN_COLUMNS = frozenset("MIDN=X")


def _read_intervals(samtools: str, bam: str, contig: str
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-read (start, end, identity) for primary alignments to ``contig``.

    Excludes unmapped/secondary/supplementary records (``-F 0x904``). Positions
    are 1-based inclusive on the contig; identity is BLAST-style
    ``1 - NM/aligned_columns`` (``nan`` when the ``NM`` tag is absent).
    """
    view = _run([samtools, "view", "-F", "0x904", bam, contig])
    starts, ends, idents = [], [], []
    for line in view.stdout.splitlines():
        f = line.split("\t")
        if len(f) < 6:
            continue
        cigar = f[5]
        if cigar == "*":
            continue
        ref_span = columns = 0
        for n, op in _CIGAR_RE.findall(cigar):
            length = int(n)
            if op in _REF_CONSUMING:
                ref_span += length
            if op in _ALN_COLUMNS:
                columns += length
        if ref_span <= 0:
            continue
        pos = int(f[3])
        starts.append(pos)
        ends.append(pos + ref_span - 1)
        nm = next((int(t[5:]) for t in f[11:] if t.startswith("NM:i:")), None)
        idents.append(1.0 - nm / columns if (nm is not None and columns > 0) else np.nan)
    return (np.asarray(starts, dtype=int), np.asarray(ends, dtype=int),
            np.asarray(idents, dtype=float))


def _fetch_contig_seq(samtools: str, ref_fasta: str, contig: str) -> str:
    _run([samtools, "faidx", ref_fasta])
    p = _run([samtools, "faidx", ref_fasta, contig])
    return "".join(ln.strip() for ln in p.stdout.splitlines() if ln and not ln.startswith(">"))


def compute_coverage(reference_seq: str, fastq_in: str, inner_seq: str | None,
                     minimap2_bin: str, samtools_bin: str, threads: int = 4,
                     allow_revcomp: bool = True, min_read_len: int | None = None,
                     max_read_len: int | None = None) -> CoverageResult:
    """Align reads to ``reference_seq`` (minimap2 -> sorted/indexed BAM via
    samtools), then return per-base coverage as a fraction of max depth, plus an
    optional exact-match highlight for ``inner_seq`` (forward or reverse complement).

    When ``min_read_len``/``max_read_len`` are set, only reads whose length is in
    that window are mapped, so the coverage and per-read map reflect the same
    length-filtered dataset used by the rest of the analysis."""
    workdir = tempfile.mkdtemp(prefix="libraont_cov_")
    ref_fasta = write_fasta(reference_seq, os.path.join(workdir, "reference.fa"), name="ref1")
    bam = os.path.join(workdir, "all_reads.bam")

    reads_in = fastq_in
    if min_read_len is not None or max_read_len is not None:
        reads_in = os.path.join(workdir, "length_filtered.fastq")
        filter_fastq_by_length(fastq_in, reads_in, min_read_len, max_read_len)

    mm2 = subprocess.Popen([minimap2_bin, "-x", "map-ont", "-a", "-t", str(threads),
                            "--secondary=no", ref_fasta, reads_in], stdout=subprocess.PIPE)
    sort = subprocess.Popen([samtools_bin, "sort", "-o", bam], stdin=mm2.stdout)
    mm2.stdout.close()
    if sort.wait() != 0 or mm2.wait() != 0:
        raise RuntimeError("Alignment failed (minimap2 / samtools sort).")
    _run([samtools_bin, "index", bam])

    # Pick the contig with the most mapped reads.
    idx = _run([samtools_bin, "idxstats", bam])
    contigs = []
    for line in idx.stdout.strip().splitlines():
        name, length, mapped, _ = line.split("\t")
        if name != "*" and int(length) > 0:
            contigs.append((name, int(length), int(mapped)))
    if not contigs:
        raise RuntimeError("No valid contigs found in BAM index.")
    contigs.sort(key=lambda x: x[2], reverse=True)
    contig, contig_length, mapped_reads = contigs[0]

    depth = _run([samtools_bin, "depth", "-d", "0", "-r", contig, bam])
    if not depth.stdout.strip():
        raise RuntimeError(f"samtools depth produced no output for contig '{contig}'.")
    pos, dep = [], []
    for line in depth.stdout.strip().splitlines():
        _, p, d = line.split("\t")
        pos.append(int(p))
        dep.append(int(d))
    positions = np.asarray(pos, dtype=float)
    depths = np.asarray(dep, dtype=float)
    max_cov = depths.max() if depths.size else 0
    fractions = depths / max_cov if max_cov > 0 else depths

    region = None
    if inner_seq:
        contig_seq = _fetch_contig_seq(samtools_bin, ref_fasta, contig).upper()
        q = collapse_whitespace(inner_seq).upper()
        hit, strand = contig_seq.find(q), "+"
        if hit == -1 and allow_revcomp:
            hit = contig_seq.find(revcomp(q))
            if hit != -1:
                strand = "-"
        if hit != -1:
            region = {"start": hit + 1, "end": hit + len(q), "strand": strand}

    read_starts, read_ends, read_idents = _read_intervals(samtools_bin, bam, contig)

    shutil.rmtree(workdir, ignore_errors=True)
    return CoverageResult(positions, fractions, mapped_reads, contig, region,
                          read_starts=read_starts, read_ends=read_ends,
                          read_identities=read_idents, contig_length=contig_length)
