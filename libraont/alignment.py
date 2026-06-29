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

import os
import shutil
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass

import edlib
import numpy as np

from .sequences import read_fastq_records, revcomp, write_fasta, collapse_whitespace

# Map a logical tool name to the env var that can override its path.
_ENV_VARS = {"mafft": "MAFFT_BIN", "minimap2": "MINIMAP2_BIN", "samtools": "SAMTOOLS_BIN"}

# Local default install locations (this Mac). Used only as a fallback when the
# tool isn't supplied explicitly, via a *_BIN env var, or found on PATH - so the
# app works out of the box here even when a tool isn't on the launching PATH
# (e.g. minimap2 lives in a conda env). Override any of these from the sidebar's
# "External tools" panel, or point a *_BIN env var elsewhere.
DEFAULT_TOOL_PATHS = {
    "mafft": "/opt/homebrew/bin/mafft",
    "minimap2": "/Users/sambra/miniforge3/envs/speedygenes_env/bin/minimap2",
    "samtools": "/Users/sambra/miniforge3/bin/samtools",
}


def resolve_tool(name: str, override: str | None = None) -> str | None:
    """Resolve a binary: explicit ``override`` -> ``<NAME>_BIN`` env -> local
    default path -> ``PATH``."""
    for candidate in (override,
                      os.environ.get(_ENV_VARS.get(name, ""), None),
                      DEFAULT_TOOL_PATHS.get(name),
                      name):
        if candidate and shutil.which(candidate):
            return shutil.which(candidate)
    return None


def tool_status(overrides: dict[str, str] | None = None) -> dict[str, str | None]:
    """Resolved path (or None) for each external tool."""
    overrides = overrides or {}
    return {name: resolve_tool(name, overrides.get(name)) for name in _ENV_VARS}


def _tool_version(name: str, path: str | None) -> str | None:
    """Return a concise version string for a resolved external tool."""
    if not path:
        return None
    args = [path, "--version"]
    try:
        proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, timeout=5)
    except Exception:
        return "Detected"
    output = "\n".join(part.strip() for part in (proc.stdout, proc.stderr) if part.strip())
    if not output:
        return "Detected"

    first = output.splitlines()[0].strip()
    if name == "samtools":
        return first.removeprefix("samtools ").strip() or first
    return first


def tool_versions(overrides: dict[str, str] | None = None) -> dict[str, str | None]:
    """Concise version string (or None) for each external tool."""
    status = tool_status(overrides)
    return {name: _tool_version(name, path) for name, path in status.items()}


# --- Orientation + trimming -------------------------------------------------
def edlib_orient_trim_and_quality(fastq_path: str, target: str, min_identity: float = 0.65,
                                  pad: int = 200,
                                  drop_short: bool = True) -> tuple[list[tuple[str, str]],
                                                                    float | None]:
    """For each read: find the best strand/placement vs ``target`` (edlib HW),
    trim to ``[start-pad : end+1+pad]`` on that strand, and keep it only if
    identity >= ``min_identity``.

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

        rf = edlib.align(target, r, task="path", mode="HW")
        rr = edlib.align(target, revcomp(r), task="path", mode="HW")
        use_rev = rr["editDistance"] < rf["editDistance"]
        res = rr if use_rev else rf
        if not res["locations"]:
            continue
        if 1.0 - res["editDistance"] / max(1, L) < min_identity:
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
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True, check=True, **kw)


def _fetch_contig_seq(samtools: str, ref_fasta: str, contig: str) -> str:
    _run([samtools, "faidx", ref_fasta])
    p = _run([samtools, "faidx", ref_fasta, contig])
    return "".join(ln.strip() for ln in p.stdout.splitlines() if ln and not ln.startswith(">"))


def compute_coverage(reference_seq: str, fastq_in: str, inner_seq: str | None,
                     minimap2_bin: str, samtools_bin: str, threads: int = 4,
                     allow_revcomp: bool = True) -> CoverageResult:
    """Align all reads to ``reference_seq`` (minimap2 -> sorted/indexed BAM via
    samtools), then return per-base coverage as a fraction of max depth, plus an
    optional exact-match highlight for ``inner_seq`` (forward or reverse complement)."""
    workdir = tempfile.mkdtemp(prefix="libraont_cov_")
    ref_fasta = write_fasta(reference_seq, os.path.join(workdir, "reference.fa"), name="ref1")
    bam = os.path.join(workdir, "all_reads.bam")

    mm2 = subprocess.Popen([minimap2_bin, "-x", "map-ont", "-a", "-t", str(threads),
                            "--secondary=no", ref_fasta, fastq_in], stdout=subprocess.PIPE)
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
            contigs.append((name, int(mapped)))
    if not contigs:
        raise RuntimeError("No valid contigs found in BAM index.")
    contigs.sort(key=lambda x: x[1], reverse=True)
    contig, mapped_reads = contigs[0]

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

    shutil.rmtree(workdir, ignore_errors=True)
    return CoverageResult(positions, fractions, mapped_reads, contig, region)
