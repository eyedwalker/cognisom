"""Annotate the shipped demo callset once, so real data reaches the pipeline.

The problem this solves
-----------------------
``seqc2_demo.vcf`` is a real callset -- HCC1395, the SEQC2 breast line --
and it carries coordinates and nothing else. A protein consequence
cannot be derived from a coordinate, so the profile builder correctly
reports no coding variants, no drivers and no neoantigens, and says why
in ``tmb_caveats``. The consequence is that every neoantigen number this
platform has ever shown came from the SYNTHETIC VCF; the real genome and
the rest of the pipeline had never met.

``VEPAnnotator`` has been able to close that gap all along -- 560 lines,
two backends, its own tests -- and was simply never passed in.

Why a file rather than a default
--------------------------------
Making VEP a silent default would put a network call to Ensembl in the
path of every profile build, on a platform that otherwise refuses to
depend on anything it cannot verify. So the annotation runs ONCE, here,
and its output ships beside the callset: the demo then reaches the
neoantigen path offline, with the provenance of the run in the file's
own header. Live VEP stays an explicit opt-in for new patient data.

Re-run with::

    python -m cognisom.genomics.annotate_demo_callset

which rewrites the annotated VCF and prints what it found.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple

SOURCE_VCF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seqc2_demo.vcf")
ANNOTATED_VCF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "seqc2_demo.vep.vcf")

#: INFO keys this writes, which VCFParser._parse_functional_annotation reads
#: directly (GENE / CONSEQUENCE / AA_CHANGE). Deliberately the same keys the
#: synthetic VCF uses, so both demo callsets parse by one path.
_INFO_KEYS = ("GENE", "CONSEQUENCE", "AA_CHANGE")


def _info_with(existing: str, additions: Dict[str, str]) -> str:
    """Merge annotations into an INFO column without dropping what is there."""
    kept = [field for field in existing.split(";")
            if field and field.split("=")[0] not in _INFO_KEYS]
    kept.extend(f"{key}={value}" for key, value in additions.items() if value)
    return ";".join(kept) if kept else "."


def annotate(source: str = SOURCE_VCF, destination: str = ANNOTATED_VCF,
             assembly: str = "GRCh38") -> Tuple[int, Dict[str, int], List[str]]:
    """Run VEP over ``source`` and write ``destination``.

    Returns ``(n_coding, stats, genes)``. Needs the network: the local
    backend is used instead when a VEP cache is present.
    """
    from cognisom.genomics.vcf_parser import VCFParser
    from cognisom.genomics.vep_annotator import VEPAnnotator

    variants = VCFParser().parse_file(source)
    annotator = VEPAnnotator(assembly=assembly, allow_remote=True)
    backend = annotator.resolve_backend()
    started = time.time()
    annotator.annotate(variants)
    elapsed = time.time() - started
    stats = annotator.stats.to_dict()

    # key by (chrom, pos, ref, alt) so the rewrite cannot mis-pair a row
    by_site = {}
    for variant in variants:
        if getattr(variant, "protein_change", None):
            by_site[(str(variant.chrom), str(variant.pos),
                     str(variant.ref), str(variant.alt))] = variant

    out: List[str] = []
    with open(source) as handle:
        for line in handle:
            if line.startswith("##"):
                out.append(line.rstrip("\n"))
                continue
            if line.startswith("#CHROM"):
                out.extend([
                    '##INFO=<ID=GENE,Number=1,Type=String,Description="Gene symbol">',
                    '##INFO=<ID=CONSEQUENCE,Number=1,Type=String,Description="Variant consequence">',
                    '##INFO=<ID=AA_CHANGE,Number=1,Type=String,Description="Amino acid change">',
                    f"##cognisom_vep_backend={stats.get('backend')}",
                    f"##cognisom_vep_assembly={assembly}",
                    f"##cognisom_vep_date={time.strftime('%Y-%m-%d')}",
                    f"##cognisom_vep_submitted={stats.get('variants_submitted')}",
                    f"##cognisom_vep_annotated={stats.get('variants_annotated')}",
                    f"##cognisom_vep_mane={stats.get('mane_transcript_used')}",
                    f"##cognisom_vep_fallback_transcript={stats.get('fallback_transcript_used')}",
                    "##cognisom_vep_note=Consequences added by cognisom.genomics."
                    "annotate_demo_callset; coordinates and calls are unchanged "
                    "from seqc2_demo.vcf. Most variants are non-coding and carry "
                    "no consequence, which is expected for a real callset.",
                    line.rstrip("\n"),
                ])
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                out.append(line.rstrip("\n"))
                continue
            key = (fields[0], fields[1], fields[3], fields[4])
            variant = by_site.get(key)
            if variant is not None:
                fields[7] = _info_with(fields[7], {
                    "GENE": getattr(variant, "gene", "") or "",
                    "CONSEQUENCE": getattr(variant, "consequence", "") or "",
                    "AA_CHANGE": variant.protein_change or "",
                })
            out.append("\t".join(fields))

    with open(destination, "w") as handle:
        handle.write("\n".join(out) + "\n")

    genes = sorted({v.gene for v in by_site.values() if getattr(v, "gene", None)})
    stats["seconds"] = round(elapsed, 1)
    stats["backend_resolved"] = backend
    return len(by_site), stats, genes


def main() -> int:            # pragma: no cover - a one-shot maintenance entry
    n_coding, stats, genes = annotate()
    print(f"wrote {ANNOTATED_VCF}")
    print(f"  {n_coding} variants carry a protein change, in {len(genes)} genes")
    print(f"  {stats}")
    print(f"  genes: {', '.join(genes)}")
    return 0


if __name__ == "__main__":    # pragma: no cover
    raise SystemExit(main())
