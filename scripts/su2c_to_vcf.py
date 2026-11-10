#!/usr/bin/env python
"""Extract one SU2C/PCF prostate patient from the cohort MAF as a VCF.

Why this script exists
----------------------
The demo ran on HCC1395, a **breast** cancer cell line, while every other
part of this platform -- the tissue geometry, the AR pathway, the drug
library -- is prostate. That mismatch was the honest thing to fix, and
fixing it needs a real prostate callset.

``data/`` is gitignored, so the cohort MAF is not in the repository and
never will be: it is 64,325 rows across 444 patients. A demo that read it
directly would work on the machine that downloaded it and nowhere else --
not in CI, not on the server -- which is this codebase's most common
defect shape. So this script runs ONCE against the local download and
writes a small committed VCF for one patient. The script is the record of
how that file was made.

Why this patient
----------------
``SC_9184_T`` carries **AR p.F877L**, the mutation that switches
enzalutamide and apalutamide from antagonists to agonists (Joseph 2013,
PMID 23779130; Balbas 2013, PMID 23580326). And its clinical record says
``ABI_ENZA_EXPOSURE_STATUS = Exposed`` -- this man received the drug class
the mutation defeats. The engine models that switch (see
``cognisom/workflow/parameter_bridge.py`` and
``tests/test_an_agonist_is_applied_as_activation.py``), so this is a
patient whose genome and whose treatment history both bear on a mechanism
the simulator actually implements.

What is faithful here, and what is not
--------------------------------------
**The somatic call is the consortium's, not ours.** The MAF publishes
``Mutation_Status = Somatic`` against a matched normal (``SC_9184_N``,
``SOMATIC_STATUS = Matched``) but publishes NO normal read counts --
``n_ref_count`` and ``n_alt_count`` are empty for all 93 rows. So the
normal column of the VCF carries a genotype (``0/0``) and nothing else.
That genotype is an assertion of the consortium's finding, not read
evidence we hold, and the VCF header says so in those words. It is what
makes the parser classify these as SOMATIC rather than
``UNKNOWN: no_matched_normal``, and anyone reading the file is entitled
to know the difference.

**Indel read counts are missing in the source.** Several frameshift and
in-frame rows carry ``t_alt_count = 0`` beside a large ``t_ref_count``,
which cannot be literally true of a called variant. Those are emitted as
published; nothing is invented to make them look right.

**Coordinates are GRCh37** and are provenance only. Protein changes come
from the MAF's own ``HGVSp_Short``, and the engine checks each against
its MANE-select reference proteome, so a transcript or numbering
disagreement surfaces as an explicit skip rather than a wrong residue.

Attribution
-----------
Source data: Abida et al., PNAS 2019 (PMID 31061129), "Genomic
correlates of clinical outcome in advanced prostate cancer", SU2C/PCF
Dream Team, via cBioPortal study ``prad_su2c_2019``. The cohort is
distributed under the **Open Data Commons Open Database License
(ODbL) 1.0**, which requires attribution and share-alike on derived
databases. The emitted VCF carries that attribution in its own header so
it travels with the data, and NOTICE records it at the repository root.

Usage
-----
    ./venv/bin/python scripts/su2c_to_vcf.py \\
        --study-dir data/cbioportal/prad_su2c_2019 \\
        --sample SC_9184_T \\
        --out cognisom/genomics/su2c_prostate_demo.vcf
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

#: MAF ``Variant_Classification`` to the CONSEQUENCE vocabulary this
#: repository's VCF parser and annotator use. Only "missense" has an
#: engine path today; the rest are carried so the callset stays whole and
#: the profile can report honestly on what it could NOT use.
CONSEQUENCE_OF = {
    "Missense_Mutation": "missense",
    "Nonsense_Mutation": "nonsense",
    "Nonstop_Mutation": "nonstop",
    "Silent": "synonymous",
    "Frame_Shift_Del": "frameshift",
    "Frame_Shift_Ins": "frameshift",
    "In_Frame_Del": "inframe_deletion",
    "In_Frame_Ins": "inframe_insertion",
    "Splice_Site": "splice_site",
    "Splice_Region": "splice_region",
    "Translation_Start_Site": "start_lost",
    "Intron": "intron",
    "5'UTR": "five_prime_utr",
    "3'UTR": "three_prime_utr",
    "5'Flank": "upstream",
    "3'Flank": "downstream",
    "IGR": "intergenic",
    "RNA": "non_coding_transcript",
}

#: An HGVSp_Short we are willing to put in AA_CHANGE. A protein change
#: that does not parse is worse than an absent one: it would reach the
#: engine's reference-residue check as a silent mismatch rather than as a
#: row we chose not to annotate.
WELL_FORMED_HGVSP = re.compile(r"^p\.[A-Z][a-z]{0,2}\d+")

#: ``*447*`` and friends. This is the MAF's notation for a row with no
#: protein consequence, and on an intron or a UTR it is correct, not
#: corrupt -- 20 of this patient's 21 such rows are Intron, 5'UTR, 3'UTR
#: or RNA. The first version of this script labelled all of them
#: "malformed", which would have put a false accusation in a published
#: file. On a CODING classification the same notation IS an anomaly,
#: because a coding change should have a protein consequence, so the two
#: cases are distinguished below rather than lumped together.
NO_PROTEIN_CONSEQUENCE = re.compile(r"^\*\d+\*$")

#: Classifications that alter protein sequence. Used only to decide
#: whether a missing protein consequence is expected or anomalous.
CODING_CLASSES = frozenset({
    "Missense_Mutation", "Nonsense_Mutation", "Nonstop_Mutation",
    "Frame_Shift_Del", "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins",
    "Splice_Site", "Translation_Start_Site",
})


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as fh:
        lines = [line for line in fh if not line.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t"))


def _clean(value: Optional[str]) -> str:
    text = (value or "").strip()
    return "" if text.upper() in ("", "NA", "NULL", ".") else text


def _vcf_alleles(ref: str, alt: str, pos: int) -> Optional[tuple]:
    """MAF allele columns as a VCF REF/ALT/POS triple.

    A MAF writes an insertion as ``- -> A`` and a deletion as
    ``ACC -> -``; VCF needs an anchor base on both sides. We cannot
    supply one without the reference genome, which this repository does
    not ship. Rather than fabricate an anchor, indels keep the MAF's own
    dash convention in a comment-safe form and are emitted as symbolic
    records: nothing downstream reads REF/ALT for a protein change, and
    the honest alternative -- dropping them -- would hide a third of this
    patient's coding events, FOXA1 and APC among them.
    """
    ref, alt = _clean(ref), _clean(alt)
    if not ref and not alt:
        return None
    if ref == "-":                      # insertion, no anchor available
        return pos, "N", f"N{alt}"
    if alt == "-":                      # deletion, no anchor available
        return pos, f"N{ref}", "N"
    return pos, ref, alt


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract one SU2C patient from the cohort MAF as a VCF")
    ap.add_argument("--study-dir", required=True,
                    help="the unpacked cBioPortal prad_su2c_2019 directory")
    ap.add_argument("--sample", default="SC_9184_T")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    study = Path(args.study_dir)
    maf = study / "data_mutations.txt"
    if not maf.exists():
        print(f"error: {maf} not found. data/ is gitignored, so this "
              f"script needs the cBioPortal download present locally.",
              file=sys.stderr)
        return 2

    rows = [r for r in _read_tsv(maf)
            if r.get("Tumor_Sample_Barcode") == args.sample]
    if not rows:
        print(f"error: {args.sample} has no rows in {maf}", file=sys.stderr)
        return 2

    # ── the clinical record, for the header's provenance block ──────
    sample_rec: Dict[str, str] = {}
    patient_rec: Dict[str, str] = {}
    for r in _read_tsv(study / "data_clinical_sample.txt"):
        if r.get("SAMPLE_ID") == args.sample:
            sample_rec = r
            break
    pid = sample_rec.get("PATIENT_ID", "")
    for r in _read_tsv(study / "data_clinical_patient.txt"):
        if r.get("PATIENT_ID") == pid:
            patient_rec = r
            break

    normal = _clean(rows[0].get("Matched_Norm_Sample_Barcode")) or "unknown_N"
    build = _clean(rows[0].get("NCBI_Build")) or "GRCh37"

    non_somatic = sorted({_clean(r.get("Mutation_Status")) for r in rows
                          if _clean(r.get("Mutation_Status")) != "Somatic"})

    # ── records ─────────────────────────────────────────────────────
    records = []
    anomalous_hgvsp = 0
    for r in rows:
        if _clean(r.get("Mutation_Status")) != "Somatic":
            continue
        chrom = _clean(r.get("Chromosome"))
        try:
            pos = int(_clean(r.get("Start_Position")))
        except ValueError:
            continue
        alleles = _vcf_alleles(r.get("Reference_Allele"),
                               r.get("Tumor_Seq_Allele2"), pos)
        if alleles is None or not chrom:
            continue
        pos, ref, alt = alleles

        gene = _clean(r.get("Hugo_Symbol"))
        klass = _clean(r.get("Variant_Classification"))
        info = [f"GENE={gene}"] if gene else []
        info.append(f"CONSEQUENCE={CONSEQUENCE_OF.get(klass, 'unknown')}")

        hgvsp = _clean(r.get("HGVSp_Short"))
        if hgvsp and WELL_FORMED_HGVSP.match(hgvsp):
            info.append(f"AA_CHANGE={hgvsp}")
        elif hgvsp and NO_PROTEIN_CONSEQUENCE.match(hgvsp):
            # Expected on a non-coding row; anomalous on a coding one.
            if klass in CODING_CLASSES:
                anomalous_hgvsp += 1
                info.append(
                    f"MAF_HGVSP_UNUSABLE={hgvsp.replace(';', '_')}")
        elif hgvsp:
            anomalous_hgvsp += 1
            info.append(f"MAF_HGVSP_UNUSABLE={hgvsp.replace(';', '_')}")
        info.append(f"MAF_CLASS={klass}")
        info.append("SU2C_SOMATIC_BY_CONSORTIUM")

        t_ref, t_alt = _clean(r.get("t_ref_count")), _clean(r.get("t_alt_count"))
        if t_ref and t_alt:
            depth = int(t_ref) + int(t_alt)
            tumour = f"0/1:{t_ref},{t_alt}:{depth}"
            fmt = "GT:AD:DP"
        else:
            tumour = "0/1"
            fmt = "GT"
        # The normal carries a GENOTYPE and no depth, because the source
        # publishes no normal read counts. See the header note.
        normal_call = "0/0" if fmt == "GT" else "0/0:.,.:."

        records.append((chrom, pos, ref, alt, ";".join(info), fmt,
                        normal_call, tumour))

    records.sort(key=lambda rec: (rec[0], rec[1]))

    # ── header ──────────────────────────────────────────────────────
    def prov(label: str, rec: Dict[str, str], key: str) -> Optional[str]:
        value = _clean(rec.get(key))
        return f"##cognisom_clinical_{label}={value}" if value else None

    lines: List[str] = [
        "##fileformat=VCFv4.2",
        f"##reference={build}",
        "##source=cBioPortal prad_su2c_2019 data_mutations.txt",
        f"##cognisom_converter=scripts/su2c_to_vcf.py --sample {args.sample}",
        "##cognisom_attribution=Abida et al., PNAS 2019, PMID 31061129, "
        "'Genomic correlates of clinical outcome in advanced prostate "
        "cancer', SU2C/PCF Dream Team; cBioPortal study prad_su2c_2019",
        "##cognisom_data_licence=Open Data Commons Open Database License "
        "(ODbL) 1.0, http://opendatacommons.org/licenses/odbl/1.0/ -- "
        "attribution and share-alike required on derived databases. This "
        "licence governs this file, not the rest of the repository.",
        # The one thing a reader most needs to know about this file.
        "##cognisom_normal_evidence=The normal column carries GT 0/0 and "
        "NO read depth. The source MAF publishes Mutation_Status=Somatic "
        "against matched normal " + normal + " but publishes no normal "
        "read counts (n_ref_count and n_alt_count are empty for every "
        "row). That 0/0 therefore ASSERTS the consortium's somatic "
        "determination; it is not read evidence held in this repository. "
        "It is what makes these calls classify as SOMATIC rather than "
        "UNKNOWN:no_matched_normal.",
        "##cognisom_indel_depth=Several frameshift and in-frame rows "
        "carry t_alt_count=0 beside a large t_ref_count in the source, "
        "which cannot be literally true of a called variant. They are "
        "emitted as published rather than corrected.",
        "##cognisom_indel_alleles=MAF dash alleles (insertion '-' -> A, "
        "deletion ACC -> '-') have no VCF anchor base, and this "
        "repository ships no reference genome to supply one. Indels use a "
        "placeholder N anchor. REF/ALT on those rows is NOT a usable "
        "genomic allele; AA_CHANGE carries the protein consequence and is "
        "what the engine reads.",
        "##INFO=<ID=GENE,Number=1,Type=String,Description=\"Hugo symbol\">",
        "##INFO=<ID=CONSEQUENCE,Number=1,Type=String,Description="
        "\"Functional consequence, mapped from MAF Variant_Classification\">",
        "##INFO=<ID=AA_CHANGE,Number=1,Type=String,Description="
        "\"Protein change, verbatim from MAF HGVSp_Short\">",
        "##INFO=<ID=MAF_CLASS,Number=1,Type=String,Description="
        "\"Original MAF Variant_Classification\">",
        "##INFO=<ID=MAF_HGVSP_UNUSABLE,Number=1,Type=String,Description="
        "\"A CODING row whose source HGVSp_Short carries no protein "
        "consequence (the MAF *N* notation) or did not parse. Flagged "
        "only on coding classifications: on an intron or UTR that same "
        "notation is correct, not corrupt, and those rows simply carry "
        "no AA_CHANGE. Deliberately NOT used as AA_CHANGE.\">",
        "##INFO=<ID=SU2C_SOMATIC_BY_CONSORTIUM,Number=0,Type=Flag,"
        "Description=\"MAF Mutation_Status was Somatic\">",
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">",
        "##FORMAT=<ID=AD,Number=R,Type=Integer,Description=\"Allelic "
        "depths for the ref and alt alleles\">",
        "##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Read depth\">",
        f"##normal_sample={normal}",
        f"##tumor_sample={args.sample}",
    ]
    for label, key in (("patient_id", "PATIENT_ID"),
                       ("cancer_type", "CANCER_TYPE_DETAILED"),
                       ("oncotree", "ONCOTREE_CODE"),
                       ("biopsy_site", "TISSUE_SITE"),
                       ("gleason", "GLEASON_SCORE"),
                       ("tmb_nonsynonymous", "TMB_NONSYNONYMOUS"),
                       ("abi_enza_exposure", "ABI_ENZA_EXPOSURE_STATUS"),
                       ("taxane_exposure", "TAXANE_EXPOSURE_STATUS"),
                       ("neuroendocrine_features", "NEUROENDOCRINE_FEATURES"),
                       ("ar_score", "AR_SCORE"),
                       ("nepc_score", "NEPC_SCORE"),
                       ("ets_fusion", "ETS_FUSION_SEQ"),
                       ("somatic_status", "SOMATIC_STATUS"),
                       ("age_at_procurement", "AGE_AT_PROCUREMENT")):
        line = prov(label, sample_rec, key)
        if line:
            lines.append(line)
    for label, key in (("psa", "PSA"), ("sex", "SEX"),
                       ("age_at_diagnosis", "AGE_AT_DIAGNOSIS"),
                       ("chemo_regimen", "CHEMO_REGIMEN_CATEGORY")):
        line = prov(label, patient_rec, key)
        if line:
            lines.append(line)
    if non_somatic:
        lines.append("##cognisom_excluded_mutation_status="
                     + ",".join(non_somatic))
    lines.append("##cognisom_assay=whole exome (data_gene_panel_matrix.txt "
                 "records WXS for mutations); TMB above is the study's own "
                 "nonsynonymous density, not recomputed here")
    lines.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                 f"{normal}\t{args.sample}")

    for chrom, pos, ref, alt, info, fmt, normal_call, tumour in records:
        prefix = chrom if chrom.startswith("chr") else f"chr{chrom}"
        lines.append(f"{prefix}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t{info}\t"
                     f"{fmt}\t{normal_call}\t{tumour}")

    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n")

    coding = sum(1 for r in records if "CONSEQUENCE=missense" in r[4])
    print(f"wrote {out}")
    print(f"  {len(records)} somatic records ({coding} missense)")
    print(f"  patient {pid}, biopsy site "
          f"{_clean(sample_rec.get('TISSUE_SITE')) or 'unknown'}, "
          f"abi/enza {_clean(sample_rec.get('ABI_ENZA_EXPOSURE_STATUS'))}")
    if anomalous_hgvsp:
        print(f"  {anomalous_hgvsp} CODING row(s) with no usable protein "
              f"consequence in the source, flagged not guessed")
    if non_somatic:
        print(f"  excluded Mutation_Status values: {non_somatic}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
