#!/usr/bin/env python3
"""How much of SC_9184_T's tumour carries AR F877L? What the data allows.

The mutation was read at a variant allele fraction of 0.233 (190 of 817
reads). It is tempting to call that "a 23% subclone". It is not, and the
cohort's own files say why: this sample's AR is AMPLIFIED, so a tumour
cell holds several copies of the gene and the allele fraction no longer
counts cells.

Run once against the cohort (``data/`` is gitignored); the numbers it
prints are the ones recorded in ``cognisom/genomics/demo_patients.py``.

    PYTHONPATH=. python scripts/su2c_ar_clonality.py

The arithmetic, for a locus on the single X of a male:

    reads from tumour cells   ~ purity * C        (C = AR copies per tumour cell)
    reads from normal cells   ~ (1 - purity) * 1
    mutant reads              ~ purity * f * m    (f = fraction of tumour cells
                                                   carrying it, m = mutant copies
                                                   in each of those cells)

    VAF = purity * f * m / (purity * C + 1 - purity)

Three numbers are measured (VAF, the copy ratio, and purity via other
mutations); f and m are not separable. All that follows is the product
f * m, and from it a RANGE for f: at most 1, and at least (f * m) / C,
which is the case where every AR copy in a carrier cell is mutant.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

SAMPLE = "SC_9184_T"
AR_POS = 66943551          # hg19, chrX; the F877L call


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cohort", default="data/cbioportal/prad_su2c_2019")
    args = ap.parse_args()
    root = Path(args.cohort)

    with open(root / "data_cna.txt") as fh:
        rows = csv.reader(fh, delimiter="\t")
        col = next(rows).index(SAMPLE)
        cna = {row[0]: row[col] for row in rows}

    with open(root / "data_mutations.txt") as fh:
        muts = [m for m in csv.DictReader(
            (line for line in fh if not line.startswith("#")), delimiter="\t")
            if m["Tumor_Sample_Barcode"] == SAMPLE]

    def vaf(m):
        alt, ref = int(m["t_alt_count"]), int(m["t_ref_count"])
        return alt / (alt + ref), alt + ref

    snvs = [m for m in muts if m["Variant_Type"] == "SNP"
            and m["t_alt_count"] not in ("", "0")]

    # Purity, two independent ways.
    # (a) a clonal SNV on the male X at normal copy number is in every
    #     tumour cell's one copy and no normal cell's, so VAF = purity.
    x_clonal = sorted((vaf(m) + (m["Hugo_Symbol"],) for m in snvs
                       if m["Chromosome"] == "X"
                       and cna.get(m["Hugo_Symbol"]) == "0"
                       and vaf(m)[0] > 0.5), reverse=True)
    purity_x = statistics.median(v for v, _, _ in x_clonal)
    # (b) a clonal heterozygous SNV on a diploid autosome sits at
    #     purity / 2.
    auto = [vaf(m)[0] for m in snvs
            if m["Chromosome"] not in ("X", "Y")
            and cna.get(m["Hugo_Symbol"]) == "0"]
    cluster = [v for v in auto if 0.25 <= v <= 0.40]
    purity_auto = 2 * statistics.median(cluster)

    with open(root / "data_cna_hg19.seg") as fh:
        segs = [r for r in csv.DictReader(fh, delimiter="\t")
                if r["ID"] == SAMPLE and r["chrom"] in ("X", "23")]
    ar_seg = next(r for r in segs
                  if float(r["loc.start"]) <= AR_POS <= float(r["loc.end"]))
    ratio = 2 ** float(ar_seg["seg.mean"])
    flank = [float(r["seg.mean"]) for r in segs
             if float(r["num.mark"]) >= 100]

    ar = next(m for m in muts if m["Hugo_Symbol"] == "AR")
    ar_vaf, depth = vaf(ar)
    se = math.sqrt(ar_vaf * (1 - ar_vaf) / depth)

    print(f"sample                     {SAMPLE}")
    print(f"AR discrete copy call      {cna['AR']}  (2 = amplified)")
    print(f"AR segment                 chrX:{ar_seg['loc.start']}-"
          f"{ar_seg['loc.end']}, {ar_seg['num.mark']} markers, "
          f"log2 {float(ar_seg['seg.mean']):.3f} -> ratio {ratio:.2f}")
    print(f"flanking X segments        median log2 "
          f"{statistics.median(flank):+.3f} over {len(flank)} large "
          f"segments (so the X baseline is one copy)")
    print(f"purity from clonal X SNVs  {purity_x:.3f}   "
          + ", ".join(f"{g} {v:.3f}/{d}" for v, d, g in x_clonal))
    print(f"purity from autosomes      {purity_auto:.3f}   "
          f"(2 x median of {len(cluster)} SNVs at VAF 0.25-0.40)")
    print(f"AR F877L                   VAF {ar_vaf:.3f} "
          f"(+/- {1.96 * se:.3f}, {depth} reads)")
    print()
    print("purity   AR copies/tumour cell   mutant copies/tumour cell (f*m)"
          "   carrier fraction f")
    for purity in sorted({round(purity_auto, 2), round(purity_x, 2)}):
        copies = (ratio - (1 - purity)) / purity
        for v in (ar_vaf - 1.96 * se, ar_vaf, ar_vaf + 1.96 * se):
            fm = v * ratio / purity
            print(f"  {purity:.2f}          {copies:5.2f}                 "
                  f"{fm:5.2f}                       "
                  f"{min(1.0, fm / copies):.2f} to 1.00")


if __name__ == "__main__":
    main()
