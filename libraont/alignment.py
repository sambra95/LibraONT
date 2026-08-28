"""Read alignment onto reference coordinates, and the whole-plasmid read map.

Wraps ``edlib`` (locating the insert), ``minimap2`` (aligning reads) and
``samtools`` (sorting/reading the plasmid alignments). Binaries are looked up on
``PATH`` only, so a missing tool degrades rather than crashes.
"""

from __future__ import annotations

import functools
import os
import re
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

import edlib
import numpy as np

from .sequences import (collapse_whitespace, filter_fastq, revcomp,
                        write_fasta)

TOOLS = ("minimap2", "samtools")

# Fraction of the insert a read must cover before its assembly can be judged.
SPANNING_MIN_COVER = 0.95
# Rows drawn on the read map; beyond this the pileup outruns the panel's pixels.
MAX_MAP_READS = 2000


def tool_status() -> dict[str, str | None]:
    """``PATH`` location (or None) of each external tool."""
    return {name: shutil.which(name) for name in TOOLS}


@functools.lru_cache(maxsize=None)
def _tool_version(name: str, path: str | None) -> str | None:
    """Version string for a resolved tool, cached so Streamlit reruns are free."""
    if not path:
        return None
    try:
        # utf-8/replace: a C-locale host chokes on samtools' non-ASCII banner.
        proc = subprocess.run([path, "--version"], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, encoding="utf-8",
                              errors="replace", timeout=20)
    except Exception:
        return "detected"
    output = "\n".join(p.strip() for p in (proc.stdout, proc.stderr) if p.strip())
    if not output:
        return "detected"
    first = output.splitlines()[0].strip()
    if name == "samtools":
        return first.removeprefix("samtools ").strip() or first
    return first


def tool_versions() -> dict[str, str | None]:
    """Concise version string (or None) for each external tool."""
    return {name: _tool_version(name, path) for name, path in tool_status().items()}


# CIGAR ops by what they consume: reference span, and alignment columns (the
# denominator for BLAST-style identity).
_CIGAR_RE = re.compile(r"(\d+)([MIDNSHP=X])")
_REF_CONSUMING = frozenset("MDN=X")
_ALN_COLUMNS = frozenset("MIDN=X")


# --- Locating the insert inside a larger reference ---------------------------
def locate_insert(reference: str, insert: str) -> dict | None:
    """1-based inclusive span of ``insert`` within ``reference`` (either strand).

    Approximate, not a substring search: a supplied gene routinely differs from
    the plasmid's copy of it by a few bases.
    """
    ref = collapse_whitespace(reference).upper()
    query = collapse_whitespace(insert).upper()
    if not ref or not query:
        return None
    hits = []
    for strand, q in (("+", query), ("-", revcomp(query))):
        res = edlib.align(q, ref, mode="HW", task="locations")
        if res["locations"]:
            hits.append((res["editDistance"], strand, res["locations"][0]))
    if not hits:
        return None
    distance, strand, (start, end) = min(hits)
    if distance > 0.25 * len(query):     # too poor to be this insert at all
        return None
    return {"start": start + 1, "end": end + 1, "strand": strand,
            "mismatches": distance}


# --- Reference-anchored projection of read alignments ------------------------
@dataclass
class ReadStructure:
    """Structural summary of one read's primary alignment over the insert."""
    name: str
    reverse: bool
    ref_start: int              # 1-based inclusive, insert coordinates
    ref_end: int
    identity: float
    insertions: list[tuple[int, int]] = field(default_factory=list)  # (pos, length)
    deletions: list[tuple[int, int]] = field(default_factory=list)

    @property
    def max_insertion(self) -> int:
        return max((n for _, n in self.insertions), default=0)

    @property
    def max_deletion(self) -> int:
        return max((n for _, n in self.deletions), default=0)

    @property
    def net_length_change(self) -> int:
        return sum(n for _, n in self.insertions) - sum(n for _, n in self.deletions)

    @property
    def covered(self) -> int:
        return self.ref_end - self.ref_start + 1

    def is_intact(self, insertion_bp: int, deletion_bp: int) -> bool:
        """True when neither indel direction reaches its structural threshold."""
        return self.max_insertion < insertion_bp and self.max_deletion < deletion_bp

    def classify(self, insertion_bp: int, deletion_bp: int) -> str:
        """intact / insertion / deletion, split by reading-frame effect. A read
        breaching both thresholds is named for its larger indel."""
        if self.is_intact(insertion_bp, deletion_bp):
            return "intact"
        ins = self.max_insertion if self.max_insertion >= insertion_bp else 0
        dele = self.max_deletion if self.max_deletion >= deletion_bp else 0
        kind = "insertion" if ins >= dele else "deletion"
        return f"{kind} ({'in-frame' if self.net_length_change % 3 == 0 else 'frameshift'})"


@dataclass
class Projection:
    """Reads projected onto insert coordinates, plus what was filtered on the way.

    ``rows`` maps read name -> a string of ``len(insert)`` characters, '-' where
    the read does not cover that base.
    """
    insert: str
    rows: dict[str, str]
    structures: dict[str, ReadStructure]
    n_input: int
    n_length_kept: int
    n_quality_kept: int
    n_mapped: int
    n_unaligned: int = 0        # no alignment at all: contaminant, not library
    n_off_target: int = 0       # crosses a junction but carries no insert: empty vector
    n_uninformative: int = 0    # aligned elsewhere on the plasmid: says nothing
    region: dict | None = None          # insert span on the alignment reference

    @property
    def n_aligned(self) -> int:
        """Reads with any alignment to the reference."""
        return self.n_mapped + self.n_off_target + self.n_uninformative

    @property
    def n_informative(self) -> int:
        """Reads that say something about the insert: they carry part of it, or
        cross a vector-insert junction without it. The analysed population."""
        return self.n_mapped + self.n_off_target

    def spanning_names(self, min_cover: float = SPANNING_MIN_COVER) -> list[str]:
        """Reads covering at least ``min_cover`` of the insert - the only ones
        whose assembly can be judged, since a shorter read hides any defect
        outside its span."""
        need = min_cover * len(self.insert)
        return [n for n, st in self.structures.items() if st.covered >= need]

    def intact_names(self, insertion_bp: int, deletion_bp: int) -> list[str]:
        """Reads carrying no structural indel. Not screened on identity: that
        falls as a clone carries more changes, so it would select against the
        most diverse library members."""
        return [n for n, st in self.structures.items()
                if st.is_intact(insertion_bp, deletion_bp)]


def _sam_records(minimap2_bin: str, ref_fasta: str, fastq: str, threads: int,
                 preset: str = "map-ont"):
    """Split SAM fields for every non-secondary record - unmapped and
    supplementary included, for the caller to count and to span-check."""
    proc = subprocess.Popen(
        [minimap2_bin, "-x", preset, "-a", "-t", str(threads), "--secondary=no",
         ref_fasta, fastq],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, encoding="utf-8",
        errors="replace")
    for line in proc.stdout:
        if line.startswith("@"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 11:
            continue
        flag = int(f[1])
        if flag & 0x100:          # secondary
            continue
        yield f, flag
    proc.stdout.close()
    if proc.wait() != 0:
        raise RuntimeError("minimap2 alignment failed.")


def project_reads(insert: str, fastq_path: str, minimap2_bin: str,
                  reference_seq: str | None = None, threads: int = 4,
                  min_read_len: int | None = None, max_read_len: int | None = None,
                  min_phred: int | None = None, n_input: int = 0) -> Projection:
    """Align reads and project each onto insert coordinates, keeping indels.

    Reads map to ``reference_seq`` (the whole plasmid) when given, else to the
    insert. The larger reference matters: aligned to the insert alone, a read
    extending past it is split, hiding the insertions this exists to measure.
    """
    insert = collapse_whitespace(insert).upper()
    workdir = tempfile.mkdtemp(prefix="libraont_aln_")
    try:
        ref_seq = collapse_whitespace(reference_seq).upper() if reference_seq else insert
        region = locate_insert(ref_seq, insert) if reference_seq else {
            "start": 1, "end": len(insert), "strand": "+", "mismatches": 0}
        if region is None:                       # insert not found in the plasmid
            ref_seq, region = insert, {"start": 1, "end": len(insert),
                                       "strand": "+", "mismatches": 0}
        ref_fasta = write_fasta(ref_seq, os.path.join(workdir, "reference.fa"), name="ref1")

        reads_in = fastq_path
        n_length_kept = n_quality_kept = n_input
        if min_read_len is not None or max_read_len is not None or min_phred is not None:
            reads_in = os.path.join(workdir, "filtered.fastq")
            n_length_kept, n_quality_kept = filter_fastq(
                fastq_path, reads_in, min_read_len, max_read_len, min_phred)

        lo, hi = region["start"], region["end"]
        flip = region["strand"] == "-"
        width = hi - lo + 1
        rows: dict[str, str] = {}
        structures: dict[str, ReadStructure] = {}
        unaligned = 0
        # A read carrying no insert bases only means "insert missing" if it
        # actually crosses a vector-insert junction; one aligned elsewhere on the
        # plasmid says nothing either way. Junction-crossing is judged over all
        # of a read's segments, since minimap2 splits a deletion this large.
        segments: dict[str, list[tuple[int, int]]] = {}
        no_insert: list[str] = []

        for f, flag in _sam_records(minimap2_bin, ref_fasta, reads_in, threads):
            if flag & 0x4 or f[5] == "*":     # contaminant: no alignment at all
                unaligned += 1
                continue
            name, pos, seq = f[0], int(f[3]), f[9]
            ops = [(int(n), op) for n, op in _CIGAR_RE.findall(f[5])]
            span = sum(n for n, op in ops if op in _REF_CONSUMING)
            segments.setdefault(name, []).append((pos, pos + span - 1))
            if flag & 0x800:                  # supplementary: span only
                continue
            row = ["-"] * width
            ins: list[tuple[int, int]] = []
            dels: list[tuple[int, int]] = []
            ref, q = pos, 0
            for n, op in ops:
                if op in "M=X":
                    a, b = max(ref, lo), min(ref + n - 1, hi)
                    if a <= b:
                        row[a - lo:b - lo + 1] = seq[q + a - ref:q + b - ref + 1]
                    ref += n
                    q += n
                elif op == "I":
                    # Point event between ref-1 and ref; hi + 1 is the 3' junction.
                    if lo <= ref <= hi + 1:
                        ins.append((min(ref, hi) - lo + 1, n))
                    q += n
                elif op in "DN":
                    # Overlap, not start-inside: a deletion spanning the whole
                    # insert begins one base before it and would be missed.
                    a, b = max(ref, lo), min(ref + n - 1, hi)
                    if a <= b:
                        dels.append((a - lo + 1, b - a + 1))
                    ref += n
                elif op == "S":
                    q += n

            projected = "".join(row).upper()
            start = width - len(projected.lstrip("-"))
            end = len(projected.rstrip("-")) - 1
            if end < start:            # no insert bases; resolved after the loop
                no_insert.append(name)
                continue
            columns = sum(n for n, op in ops if op in _ALN_COLUMNS)
            nm = next((int(t[5:]) for t in f[11:] if t.startswith("NM:i:")), None)
            identity = 1.0 - nm / columns if (nm is not None and columns > 0) else float("nan")
            start, end = start + 1, end + 1
            if flip:                     # insert lies on the reverse strand
                projected = revcomp(projected)
                start, end = width - end + 1, width - start + 1
                ins = [(width - p + 1, n) for p, n in ins]
                dels = [(width - p + 1, n) for p, n in dels]
            rows[name] = projected
            structures[name] = ReadStructure(
                name=name, reverse=bool(flag & 0x10) != flip, ref_start=start,
                ref_end=end, identity=identity, insertions=ins, deletions=dels)

        def crosses_junction(name: str) -> bool:
            return any(s < lo <= e or s <= hi < e
                       for s, e in segments.get(name, ()))

        off_target = sum(1 for name in no_insert if crosses_junction(name))
        return Projection(insert=insert, rows=rows, structures=structures,
                          n_input=n_input or len(rows), n_length_kept=n_length_kept,
                          n_quality_kept=n_quality_kept,
                          n_mapped=len(rows), n_off_target=off_target,
                          n_uninformative=len(no_insert) - off_target,
                          n_unaligned=unaligned, region=region)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# --- Whole-plasmid read map (minimap2 + samtools) ---------------------------
@dataclass
class ReadMap:
    """Per-read alignment of a read set to a reference contig."""
    mapped_reads: int
    contig: str
    region: dict | None = None  # {'start','end','strand'} or None
    # Per-read alignment span on the contig (1-based inclusive) and identity, for
    # the read-alignment map. Parallel arrays over primary alignments.
    read_starts: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    read_ends: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    read_identities: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    # Per-read, per-contig-base agreement with the reference, row-aligned to the
    # arrays above: 0 = not covered, 1 = matches, 2 = substituted, 3 = deleted.
    # uint8 keeps a deep pileup over a whole plasmid cheap to hold and to draw.
    # Insertions are absent here by design - they occupy no reference base, so
    # they live only in the sparse records below.
    match_matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.uint8))
    # Indels as sparse triples - which kept row, where on the contig, how many
    # bases - so their size survives, which a per-base matrix cannot carry, and
    # so the plot can apply a size threshold without losing the underlying data.
    insertion_rows: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    insertion_positions: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    insertion_lengths: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    deletion_rows: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    deletion_positions: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    deletion_lengths: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    # Per-base tallies over *every* mapped read, not just the rows kept for the
    # map, so the agreement trace stays exact however deep the pileup gets.
    match_counts: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    depth_counts: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    contig_length: int = 0


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    # utf-8/replace so a C/ASCII-locale host can't choke on non-ASCII output.
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          encoding="utf-8", errors="replace", check=True, **kw)


def _read_intervals(samtools: str, bam: str, contig: str, contig_seq: str,
                    contig_length: int, max_rows: int = 0
                    ) -> tuple[np.ndarray, ...]:
    """Per-read (start, end, identity, per-base agreement) on ``contig``.

    Positions are 1-based inclusive; identity is ``1 - NM/aligned_columns``
    (``nan`` without an ``NM`` tag). The agreement matrix codes each base 0 (not
    covered), 1 (match), 2 (substituted) or 3 (deleted); indels also come back
    as sparse (row, position, length) triples, since a per-base matrix carries
    no length and an insertion has no base of its own.

    ``max_rows`` caps the rows kept for the map, sampled evenly; the per-base
    tallies still cover every read, so the agreement trace is never a sample.
    """
    view = _run([samtools, "view", "-F", "0x904", bam, contig])
    lines = [ln for ln in view.stdout.splitlines() if ln]
    step = math.ceil(len(lines) / max_rows) if max_rows and len(lines) > max_rows else 1

    ref = np.frombuffer(contig_seq.upper().encode(), dtype="S1")
    match_counts = np.zeros(contig_length, dtype=np.int64)
    depth_counts = np.zeros(contig_length, dtype=np.int64)
    starts, ends, idents, rows = [], [], [], []
    ins_records: tuple[list, list, list] = ([], [], [])
    del_records: tuple[list, list, list] = ([], [], [])
    for index, line in enumerate(lines):
        f = line.split("\t")
        if len(f) < 11:
            continue
        cigar = f[5]
        if cigar == "*":
            continue
        ops = [(int(n), op) for n, op in _CIGAR_RE.findall(cigar)]
        ref_span = sum(n for n, op in ops if op in _REF_CONSUMING)
        columns = sum(n for n, op in ops if op in _ALN_COLUMNS)
        if ref_span <= 0:
            continue
        pos = int(f[3])
        seq = f[9]
        row = np.zeros(contig_length, dtype=np.uint8)
        insertions: list[tuple[int, int]] = []
        deletions: list[tuple[int, int]] = []
        r = pos - 1          # 0-based cursor on the contig
        q = 0                # cursor in the read
        for n, op in ops:
            if op in "M=X":
                lo, hi = r, min(r + n, contig_length)
                if hi > lo:
                    read_seg = np.frombuffer(seq[q:q + (hi - lo)].upper().encode(),
                                             dtype="S1")
                    row[lo:hi] = np.where(read_seg == ref[lo:hi], 1, 2)
                r += n
                q += n
            elif op in "DN":
                if 0 <= r < contig_length:
                    deletions.append((r, n))
                row[r:min(r + n, contig_length)] = 3
                r += n
            elif op == "I":
                if 0 <= r < contig_length:
                    insertions.append((r, n))
                q += n
            elif op == "S":
                q += n
        match_counts += (row == 1)
        depth_counts += (row > 0)
        if index % step:
            continue
        for store, events in ((ins_records, insertions), (del_records, deletions)):
            for at, length in events:
                store[0].append(len(rows))
                store[1].append(at + 1)
                store[2].append(length)
        starts.append(pos)
        ends.append(pos + ref_span - 1)
        nm = next((int(t[5:]) for t in f[11:] if t.startswith("NM:i:")), None)
        idents.append(1.0 - nm / columns if (nm is not None and columns > 0) else np.nan)
        rows.append(row)
    matrix = (np.vstack(rows) if rows
              else np.empty((0, contig_length), dtype=np.uint8))
    return (np.asarray(starts, dtype=int), np.asarray(ends, dtype=int),
            np.asarray(idents, dtype=float), matrix, match_counts, depth_counts,
            *(np.asarray(a, dtype=int) for a in ins_records),
            *(np.asarray(a, dtype=int) for a in del_records))


def _fetch_contig_seq(samtools: str, ref_fasta: str, contig: str) -> str:
    _run([samtools, "faidx", ref_fasta])
    p = _run([samtools, "faidx", ref_fasta, contig])
    return "".join(ln.strip() for ln in p.stdout.splitlines() if ln and not ln.startswith(">"))


def map_reads_to_reference(reference_seq: str, fastq_in: str, inner_seq: str | None,
                           minimap2_bin: str, samtools_bin: str, threads: int = 4,
                           min_read_len: int | None = None,
                           max_read_len: int | None = None,
                           min_phred: int | None = None,
                           max_map_reads: int = MAX_MAP_READS) -> ReadMap:
    """Align reads to ``reference_seq`` (minimap2 -> sorted/indexed BAM via
    samtools) and return where each one sits and how it agrees with the
    reference, plus an optional highlight for ``inner_seq`` (either strand).

    ``min_read_len``/``max_read_len``/``min_phred`` apply the same read filters as
    the rest of the analysis, so the map reflects the same dataset."""
    workdir = tempfile.mkdtemp(prefix="libraont_cov_")
    ref_fasta = write_fasta(reference_seq, os.path.join(workdir, "reference.fa"), name="ref1")
    bam = os.path.join(workdir, "all_reads.bam")

    reads_in = fastq_in
    if min_read_len is not None or max_read_len is not None or min_phred is not None:
        reads_in = os.path.join(workdir, "filtered.fastq")
        filter_fastq(fastq_in, reads_in, min_read_len, max_read_len, min_phred)

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
    contig, contig_length, _ = contigs[0]
    # idxstats counts alignment *records*; supplementary alignments would make a
    # circular plasmid look like it had twice as many reads as the FASTQ holds.
    mapped_reads = int(_run([samtools_bin, "view", "-c", "-F", "0x904", bam, contig])
                       .stdout.strip() or 0)

    contig_seq = ""
    region = None
    if inner_seq:
        # edlib, not an exact search: the supplied gene is routinely a few bases
        # different from the plasmid's copy of it, which would silently lose the
        # insert marker.
        contig_seq = _fetch_contig_seq(samtools_bin, ref_fasta, contig).upper()
        region = locate_insert(contig_seq, inner_seq)

    if not contig_seq:
        contig_seq = _fetch_contig_seq(samtools_bin, ref_fasta, contig).upper()
    (read_starts, read_ends, read_idents, match_matrix, match_counts, depth_counts,
     ins_rows, ins_pos, ins_len, del_rows, del_pos, del_len) = _read_intervals(
        samtools_bin, bam, contig, contig_seq, contig_length, max_rows=max_map_reads)

    shutil.rmtree(workdir, ignore_errors=True)
    return ReadMap(mapped_reads, contig, region,
                          read_starts=read_starts, read_ends=read_ends,
                          read_identities=read_idents, match_matrix=match_matrix,
                          match_counts=match_counts, depth_counts=depth_counts,
                          insertion_rows=ins_rows, insertion_positions=ins_pos,
                          insertion_lengths=ins_len, deletion_rows=del_rows,
                          deletion_positions=del_pos, deletion_lengths=del_len,
                          contig_length=contig_length)
