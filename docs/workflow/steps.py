"""The workflow as a sequence of filters, with no counts attached.

``lost`` is why reads leave before reaching that pool, ``glyph`` picks the little
picture of that failure, and ``plots`` names the
outputs built from it. There are no proportions here on purpose: the figure
shows the shape of the workflow, not one run's numbers.
"""

STEPS = [
    dict(pool="All reads in FASTQ file", lost="",
         plots=["Read length distribution"]),
    dict(pool="Reads within read-length window",
         lost="outside the length window", glyph="length", plots=[]),
    dict(pool="Reads aligned to the reference",
         lost="no alignment", glyph="contaminant",
         plots=["Read alignment map"]),
    dict(pool="Reads informative about the insert",
         lost="no vector-insert junction crossed", glyph="junction",
         plots=["Library composition"]),
    dict(pool="Reads correctly assembled",
         lost="structural indel, or no insert", glyph="structural",
         plots=["Alignment to reference insert", "AA distribution"]),
    dict(pool="Reads covering all diversified positions",
         lost="a diversified codon unreadable", glyph="unreadable",
         plots=["Variant treemap"]),
]

# The shape of each output, drawn as a tiny wordless icon above its chip.
PLOT_ICONS = {
    "Read length distribution": "histogram",
    "Read alignment map": "tracks",
    "Library composition": "waffle",
    "Alignment to reference insert": "trace",
    "AA distribution": "pie",
    "Variant treemap": "treemap",
}

THRESHOLDS = {"insertion_bp": 10, "deletion_bp": 10, "span_pct": 95}
