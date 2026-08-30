"""Read alignment onto reference coordinates, and the whole-plasmid read map.

Wraps ``edlib`` (locating the insert) and ``minimap2`` (aligning reads).
``minimap2`` is looked up on ``PATH`` only, so a missing tool degrades rather
than crashes.
"""

from __future__ import annotations

import functools
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

import edlib
import numpy as np

from .sequences import (collapse_whitespace, filter_fastq, revcomp,
                        write_fasta)

TOOLS = ("minimap2",)

# Agreement codes written into ``ReadMap.match_matrix``: 0 not covered, 1 match,
# 3 deleted, 4 an insertion begins here, and a substitution carrying the read's
# own base - 5/6/7/8 for A/C/G/T, 2 for anything else.
SUBSTITUTION_CODES: dict[str, int] = {"A": 5, "C": 6, "G": 7, "T": 8}
_SUB_LUT = np.full(256, 2, dtype=np.uint8)
for _base, _code in SUBSTITUTION_CODES.items():
    _SUB_LUT[ord(_base)] = _code
_GAP = ord("-")             # a base the read deletes, in the scratch base track

# Fraction of the insert a read must cover before its assembly can be judged.
SPANNING_MIN_COVER = 0.95
# Rows drawn on the read map; beyond this the pileup outruns the panel's pixels.
MAX_MAP_READS = 2000
# Too little insert to say anything about: the aligner spilling over a
# vector-insert junction, or a read stopping just inside one.
MIN_INSERT_BASES = 50
# A read holding the vector either side of the insert shows the whole molecule,
# so holding less of the insert than this means the insert is not in it at all.
INSERT_PRESENT_FRACTION = 0.5


def tool_status() -> dict[str, str | None]:
    """``PATH`` location (or None) of each external tool."""
    return {name: shutil.which(name) for name in TOOLS}


@functools.lru_cache(maxsize=None)
def _tool_version(name: str, path: str | None) -> str | None:
    """Version string for a resolved tool, cached so Streamlit reruns are free."""
    if not path:
        return None
    try:
        # utf-8/replace: a C-locale host chokes on a non-ASCII banner.
        proc = subprocess.run([path, "--version"], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, encoding="utf-8",
                              errors="replace", timeout=20)
    except Exception:
        return "detected"
    output = "\n".join(p.strip() for p in (proc.stdout, proc.stderr) if p.strip())
    return output.splitlines()[0].strip() if output else "detected"


def tool_versions() -> dict[str, str | None]:
    """Concise version string (or None) for each external tool."""
    return {name: _tool_version(name, path) for name, path in tool_status().items()}


_CIGAR_RE = re.compile(r"(\d+)([MIDNSHP=X])")
_REF_CONSUMING = frozenset("MDN=X")     # CIGAR ops consuming reference span


# --- Locating the insert inside a larger reference ---------------------------
def locate_insert(reference: str, insert: str) -> dict | None:
    """1-based inclusive span of ``insert`` within ``reference`` (either strand).
    Approximate, not a substring search: the supplied gene routinely differs
    from the plasmid's copy by a few bases."""
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
    n_off_target: int = 0       # vector either side, no insert: empty vector
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


def _sam_records(minimap2_bin: str, ref_fasta: str, fastq: str, threads: int):
    """Split SAM fields for every non-secondary record - unmapped and
    supplementary included, for the caller to count and to span-check."""
    proc = subprocess.Popen(
        [minimap2_bin, "-x", "map-ont", "-a", "-t", str(threads), "--secondary=no",
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


def _blank(dtype=int, shape: int | tuple = 0):
    """An empty array of its own, as a dataclass default."""
    return field(default_factory=lambda: np.empty(shape, dtype=dtype))


@dataclass
class ReadMap:
    """Per-read alignment of a read set to a reference contig."""
    mapped_reads: int
    contig: str
    contig_seq: str = ""        # the reference itself, for the base track
    region: dict | None = None  # {'start','end','strand'} or None
    # Span on the contig (1-based inclusive) of the read drawn on each row.
    read_starts: np.ndarray = _blank()
    read_ends: np.ndarray = _blank()
    # Per-read, per-base agreement, row-aligned to those; see
    # ``SUBSTITUTION_CODES``. Insertions occupy no reference base, so they live
    # only in the sparse records below.
    match_matrix: np.ndarray = _blank(np.uint8, (0, 0))
    # Indels as sparse (row, contig position, length) triples, so their size
    # survives for the plot to threshold on.
    insertion_rows: np.ndarray = _blank()
    insertion_positions: np.ndarray = _blank()
    insertion_lengths: np.ndarray = _blank()
    deletion_rows: np.ndarray = _blank()
    deletion_positions: np.ndarray = _blank()
    deletion_lengths: np.ndarray = _blank()
    # Per-base tallies over *every* mapped read, not just the rows kept, so the
    # traces stay exact however deep the pileup gets.
    match_counts: np.ndarray = _blank(np.int64)
    depth_counts: np.ndarray = _blank(np.int64)
    # Reads whose alignment actually holds a base here - matches and
    # substitutions, but not deletions, which ``depth_counts`` still counts.
    covered_counts: np.ndarray = _blank(np.int64)
    # Phred scores summed over those same bases, so ``quality_sums /
    # covered_counts`` is the mean basecall quality at each position.
    quality_sums: np.ndarray = _blank(np.int64)
    contig_length: int = 0



# --- One alignment pass: insert projection + whole-plasmid map ---------------
def align_reads(insert: str, fastq_path: str, minimap2_bin: str,
                reference_seq: str | None = None, threads: int = 4,
                min_read_len: int | None = None, max_read_len: int | None = None,
                min_phred: int | None = None, n_input: int = 0,
                max_map_reads: int = MAX_MAP_READS
                ) -> tuple[Projection, ReadMap | None]:
    """Align the reads once and read the alignments once, building both the
    insert projection and the whole-plasmid read map.

    Reads map to ``reference_seq`` (the whole plasmid) written out twice, so one
    crossing the origin aligns in a single piece; coordinates fold back onto the
    plasmid. Without a plasmid the insert is the reference, there is no map to
    draw, and a read running past the insert is split - hiding the insertions
    this exists to measure."""
    insert = collapse_whitespace(insert).upper()
    plasmid = collapse_whitespace(reference_seq).upper() if reference_seq else ""
    # edlib, not an exact search: the supplied gene is routinely a few bases
    # different from the plasmid's copy of it.
    region = locate_insert(plasmid, insert) if plasmid else None
    if region is None:                  # no plasmid, or the insert is not in it
        plasmid = ""
        region = {"start": 1, "end": len(insert), "strand": "+", "mismatches": 0}
    ref_seq = plasmid or insert
    fold = len(ref_seq)
    laps = 2 if plasmid else 1          # only a plasmid is circular

    workdir = tempfile.mkdtemp(prefix="libraont_aln_")
    try:
        ref_fasta = write_fasta(ref_seq * laps,
                                os.path.join(workdir, "reference.fa"), name="ref1")
        reads_in = fastq_path
        n_length_kept = n_quality_kept = n_input
        if min_read_len is not None or max_read_len is not None or min_phred is not None:
            reads_in = os.path.join(workdir, "filtered.fastq")
            n_length_kept, n_quality_kept = filter_fastq(
                fastq_path, reads_in, min_read_len, max_read_len, min_phred)

        lo, hi = region["start"], region["end"]
        flip = region["strand"] == "-"
        width = hi - lo + 1
        least = max(1, min(MIN_INSERT_BASES, width // 10))
        ref = np.frombuffer(ref_seq.encode(), dtype=np.uint8)

        # A read running past the origin arrives as a primary plus supplementary
        # alignments, so each of a read's alignments writes into the same row.
        drafts: dict[str, np.ndarray] = {}           # insert bases, per read
        events: dict[str, tuple[list, list]] = {}    # insert indels, per read
        reversed_read: dict[str, bool] = {}
        # Alignment spans per read, for the empty-vector test below.
        segments: dict[str, list[tuple[int, int]]] = {}
        unaligned = 0

        # Rows drawn on the map, spread evenly over the reads by taking every
        # step-th one; the per-base tallies below still cover every read.
        step = max(1, math.ceil((n_quality_kept or max_map_reads) / max_map_reads))
        drawn: dict[str, int] = {}
        map_rows: list[np.ndarray] = []
        starts: list[int] = []
        ends: list[int] = []
        ins_records: tuple[list, list, list] = ([], [], [])
        del_records: tuple[list, list, list] = ([], [], [])
        match_counts = np.zeros(fold, dtype=np.int64)
        depth_counts = np.zeros(fold, dtype=np.int64)
        covered_counts = np.zeros(fold, dtype=np.int64)
        quality_sums = np.zeros(fold, dtype=np.int64)
        # Scratch over the doubled reference, folded once per alignment.
        track = np.zeros(fold * laps, dtype=np.uint8)
        phred = np.zeros(fold * laps, dtype=np.int64)

        for f, flag in _sam_records(minimap2_bin, ref_fasta, reads_in, threads):
            if flag & 0x4 or f[5] == "*":     # contaminant: no alignment at all
                unaligned += 1
                continue
            name, pos = f[0], int(f[3])
            ops = [(int(n), op) for n, op in _CIGAR_RE.findall(f[5])]
            span = sum(n for n, op in ops if op in _REF_CONSUMING)
            if span <= 0:
                continue
            first = name not in segments
            start = (pos - 1) % fold + 1
            segments.setdefault(name, []).append((start, start + span - 1))
            if not flag & 0x800:              # the primary carries the strand
                reversed_read[name] = bool(flag & 0x10)
            if first and (len(segments) - 1) % step == 0 and len(drawn) < max_map_reads:
                drawn[name] = len(drawn)
                map_rows.append(np.zeros(fold, dtype=np.uint8))
                starts.append(0)
                ends.append(0)

            seq = f[9].upper().encode()
            scores = (np.frombuffer(f[10].encode(), dtype=np.uint8).astype(np.int64) - 33
                      if f[10] != "*" else None)
            track[:] = 0
            phred[:] = 0
            indels: list[tuple[str, int, int]] = []
            r, q = pos - 1, 0                 # 0-based cursors on reference / read
            for n, op in ops:
                if op in "M=X":
                    a, b = r, min(r + n, track.size)
                    if b > a:
                        track[a:b] = np.frombuffer(seq[q:q + b - a], dtype=np.uint8)
                        if scores is not None:
                            phred[a:b] = scores[q:q + b - a]
                    r += n
                    q += n
                elif op in "DN":
                    if r < track.size:
                        indels.append(("D", r, n))
                    track[r:min(r + n, track.size)] = _GAP
                    r += n
                elif op == "I":
                    if r < track.size:
                        indels.append(("I", r, n))
                    q += n
                elif op == "S":
                    q += n

            laps_view = track.reshape(laps, fold)
            bases = laps_view[0].copy()
            for lap in laps_view[1:]:         # the read crossed the origin
                np.copyto(bases, lap, where=bases == 0)
            held = (bases != 0) & (bases != _GAP)     # a deletion holds no base
            depth_counts += bases != 0
            covered_counts += held
            match_counts += held & (bases == ref)
            if scores is not None:
                quality_sums += phred.reshape(laps, fold).sum(axis=0)

            index = drawn.get(name)
            if index is not None:
                code = np.where(bases == ref, 1, _SUB_LUT[bases]).astype(np.uint8)
                code[bases == _GAP] = 3
                code[bases == 0] = 0
                np.copyto(map_rows[index], code, where=map_rows[index] == 0)
                starts[index] = start if not starts[index] else min(starts[index], start)
                ends[index] = max(ends[index], start + span - 1)
                for kind, at, n in indels:
                    store = ins_records if kind == "I" else del_records
                    store[0].append(index)
                    store[1].append(at % fold + 1)
                    store[2].append(n)

            insert_bases = bases[lo - 1:hi]
            if not insert_bases.any():        # nowhere near the insert
                continue
            draft = drafts.get(name)
            if draft is None:
                draft = drafts[name] = np.zeros(width, dtype=np.uint8)
            np.copyto(draft, insert_bases, where=(draft == 0) & (insert_bases != _GAP))
            ins: list[tuple[int, int]] = []
            dels: list[tuple[int, int]] = []
            for kind, at, n in indels:
                p = at % fold + 1
                if kind == "I":
                    # Point event between p-1 and p; hi + 1 is the 3' junction.
                    if lo <= p <= hi + 1:
                        ins.append((min(p, hi) - lo + 1, n))
                else:
                    # Overlap, not start-inside: a deletion spanning the whole
                    # insert begins one base before it and would be missed.
                    a, b = max(p, lo), min(p + n - 1, hi)
                    if a <= b:
                        dels.append((a - lo + 1, b - a + 1))
            if ins or dels:                   # the read's other alignment may add more
                carried = events.setdefault(name, ([], []))
                carried[0].extend(ins)
                carried[1].extend(dels)

        # Empty vector: the read carries the vector on both sides of where the
        # insert belongs. Judged circularly - the whole backbone can align as one
        # piece running through the origin - and over all of a read's alignments,
        # since minimap2 splits a deletion this large.
        before, after = (lo - 2) % fold + 1, hi % fold + 1

        def spans_flanks(name: str) -> bool:
            spans = segments.get(name, ())
            return all(any(s <= p <= e for s, e in spans for p in (at, at + fold))
                       for at in (before, after))

        rows: dict[str, str] = {}
        structures: dict[str, ReadStructure] = {}
        for name, draft in drafts.items():
            covered = np.flatnonzero(draft)
            if covered.size < least or (spans_flanks(name) and
                                        covered.size < INSERT_PRESENT_FRACTION * width):
                continue                      # no insert to speak of
            start, end = int(covered[0]) + 1, int(covered[-1]) + 1
            projected = np.where(draft == 0, _GAP, draft).astype(np.uint8).tobytes().decode()
            ins, dels = events.get(name, ([], []))
            if flip:                          # insert lies on the reverse strand
                projected = revcomp(projected)
                start, end = width - end + 1, width - start + 1
                ins = [(width - p + 1, n) for p, n in ins]
                dels = [(width - p + 1, n) for p, n in dels]
            rows[name] = projected
            structures[name] = ReadStructure(
                name=name, reverse=reversed_read.get(name, False) != flip,
                ref_start=start, ref_end=end, insertions=ins, deletions=dels)

        empty_vector = [name for name in segments
                        if name not in rows and spans_flanks(name)]
        # Such a read has lost the insert, but the aligner writes that as a
        # deletion only when it bridges the gap - otherwise it stops at one
        # junction and picks up at the other, leaving the map blank there. Count
        # and paint the gap as deleted so both kinds of read read alike.
        for name in empty_vector:
            gap = np.ones(width, dtype=bool)
            for s, e in segments[name]:
                for a, b in ((s, e), (s - fold, e - fold)):   # the arc may wrap
                    a, b = max(a, lo), min(b, hi)
                    if a <= b:
                        gap[a - lo:b - lo + 1] = False
            depth_counts[lo - 1:hi] += gap        # the read spans it, deleted
            index = drawn.get(name)
            if index is not None:
                row = map_rows[index][lo - 1:hi]
                paint = gap & (row == 0)
                row[paint] = 3
                # Each painted run is recorded like any other deletion, so the
                # map can report how long it is.
                edges = np.flatnonzero(np.diff(np.r_[0, paint.view(np.int8), 0]))
                for a, b in zip(edges[::2], edges[1::2]):
                    del_records[0].append(index)
                    del_records[1].append(lo + int(a))
                    del_records[2].append(int(b - a))
        no_insert = len(segments) - len(rows)
        projection = Projection(
            insert=insert, rows=rows, structures=structures,
            n_input=n_input or len(rows), n_length_kept=n_length_kept,
            n_quality_kept=n_quality_kept, n_mapped=len(rows),
            n_off_target=len(empty_vector),
            n_uninformative=no_insert - len(empty_vector),
            n_unaligned=unaligned, region=region)
        if not plasmid:
            return projection, None
        as_int = lambda a: np.asarray(a, dtype=int)
        read_map = ReadMap(
            len(segments), "ref1", ref_seq, region,
            read_starts=as_int(starts), read_ends=as_int(ends),
            match_matrix=(np.vstack(map_rows) if map_rows
                          else np.empty((0, fold), dtype=np.uint8)),
            match_counts=match_counts, depth_counts=depth_counts,
            covered_counts=covered_counts, quality_sums=quality_sums,
            insertion_rows=as_int(ins_records[0]),
            insertion_positions=as_int(ins_records[1]),
            insertion_lengths=as_int(ins_records[2]),
            deletion_rows=as_int(del_records[0]),
            deletion_positions=as_int(del_records[1]),
            deletion_lengths=as_int(del_records[2]), contig_length=fold)
        return projection, read_map
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
