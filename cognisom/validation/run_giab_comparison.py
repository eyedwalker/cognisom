"""
GIAB Comparison Runner
========================

Standalone script to run bcftools isec comparison against GIAB truth set.
Designed to run on the GPU box after Parabricks completes.

Usage (on GPU box):
    python3 run_giab_comparison.py [vcf_path]

Outputs:
    - /opt/cognisom/benchmark/giab_results.json
    - Console summary of TP/FP/FN/precision/recall/F1
"""

import json
import os
import subprocess
import sys
import time


TRUTH_VCF = "/opt/cognisom/benchmark/giab/truth.vcf.gz"
TRUTH_BED = "/opt/cognisom/benchmark/giab/truth.bed"
COMPARISON_DIR = "/opt/cognisom/benchmark/giab/comparison"
DEFAULT_VCF = "/opt/cognisom/jobs/giab_benchmark/NA12878_deepvariant.vcf"


def run(cmd):
    """Run shell command, return stdout."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {cmd}")
        print(result.stderr)
    return result.stdout.strip()


def count_variants(vcf_path):
    """Count non-header lines in VCF."""
    result = run(f"grep -cv '^#' {vcf_path} 2>/dev/null || echo 0")
    return int(result)


def calculate_metrics(tp, fp, fn):
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(0.0001, precision + recall)
    return precision, recall, f1


def run_comparison(call_vcf):
    """Run full GIAB comparison."""
    os.makedirs(COMPARISON_DIR, exist_ok=True)

    # Compress and index if needed
    if not call_vcf.endswith(".gz"):
        print("Compressing VCF...")
        run(f"bgzip -c {call_vcf} > {call_vcf}.gz")
        run(f"tabix -p vcf {call_vcf}.gz")
        call_vcf_gz = f"{call_vcf}.gz"
    else:
        call_vcf_gz = call_vcf
        if not os.path.exists(f"{call_vcf}.tbi"):
            run(f"tabix -p vcf {call_vcf_gz}")

    # Filter to PASS in high-confidence regions
    filtered = f"{COMPARISON_DIR}/calls_filtered.vcf.gz"
    print("Filtering to PASS variants in high-confidence regions...")
    run(f"bcftools view -f PASS -R {TRUTH_BED} {call_vcf_gz} -Oz -o {filtered}")
    run(f"tabix -p vcf {filtered}")

    results = {}

    # Overall comparison
    print("Running overall comparison...")
    run(f"rm -rf {COMPARISON_DIR}/isec_all")
    run(f"bcftools isec -p {COMPARISON_DIR}/isec_all {TRUTH_VCF} {filtered} -R {TRUTH_BED}")
    fn = count_variants(f"{COMPARISON_DIR}/isec_all/0000.vcf")
    fp = count_variants(f"{COMPARISON_DIR}/isec_all/0001.vcf")
    tp = count_variants(f"{COMPARISON_DIR}/isec_all/0002.vcf")
    precision, recall, f1 = calculate_metrics(tp, fp, fn)
    results["overall"] = {"tp": tp, "fp": fp, "fn": fn,
                          "precision": round(precision, 6),
                          "recall": round(recall, 6),
                          "f1": round(f1, 6)}
    print(f"  Overall: TP={tp:,} FP={fp:,} FN={fn:,} "
          f"P={precision:.4f} R={recall:.4f} F1={f1:.4f}")

    # SNP comparison
    print("Running SNP comparison...")
    run(f"bcftools view -v snps {TRUTH_VCF} -R {TRUTH_BED} -Oz -o {COMPARISON_DIR}/truth_snps.vcf.gz")
    run(f"tabix -p vcf {COMPARISON_DIR}/truth_snps.vcf.gz")
    run(f"bcftools view -v snps {filtered} -Oz -o {COMPARISON_DIR}/calls_snps.vcf.gz")
    run(f"tabix -p vcf {COMPARISON_DIR}/calls_snps.vcf.gz")
    run(f"rm -rf {COMPARISON_DIR}/isec_snps")
    run(f"bcftools isec -p {COMPARISON_DIR}/isec_snps "
        f"{COMPARISON_DIR}/truth_snps.vcf.gz {COMPARISON_DIR}/calls_snps.vcf.gz -R {TRUTH_BED}")
    snp_fn = count_variants(f"{COMPARISON_DIR}/isec_snps/0000.vcf")
    snp_fp = count_variants(f"{COMPARISON_DIR}/isec_snps/0001.vcf")
    snp_tp = count_variants(f"{COMPARISON_DIR}/isec_snps/0002.vcf")
    snp_p, snp_r, snp_f1 = calculate_metrics(snp_tp, snp_fp, snp_fn)
    results["snp"] = {"tp": snp_tp, "fp": snp_fp, "fn": snp_fn,
                      "precision": round(snp_p, 6),
                      "recall": round(snp_r, 6),
                      "f1": round(snp_f1, 6)}
    print(f"  SNP:     TP={snp_tp:,} FP={snp_fp:,} FN={snp_fn:,} "
          f"P={snp_p:.4f} R={snp_r:.4f} F1={snp_f1:.4f}")

    # Indel comparison
    print("Running Indel comparison...")
    run(f"bcftools view -v indels {TRUTH_VCF} -R {TRUTH_BED} -Oz -o {COMPARISON_DIR}/truth_indels.vcf.gz")
    run(f"tabix -p vcf {COMPARISON_DIR}/truth_indels.vcf.gz")
    run(f"bcftools view -v indels {filtered} -Oz -o {COMPARISON_DIR}/calls_indels.vcf.gz")
    run(f"tabix -p vcf {COMPARISON_DIR}/calls_indels.vcf.gz")
    run(f"rm -rf {COMPARISON_DIR}/isec_indels")
    run(f"bcftools isec -p {COMPARISON_DIR}/isec_indels "
        f"{COMPARISON_DIR}/truth_indels.vcf.gz {COMPARISON_DIR}/calls_indels.vcf.gz -R {TRUTH_BED}")
    indel_fn = count_variants(f"{COMPARISON_DIR}/isec_indels/0000.vcf")
    indel_fp = count_variants(f"{COMPARISON_DIR}/isec_indels/0001.vcf")
    indel_tp = count_variants(f"{COMPARISON_DIR}/isec_indels/0002.vcf")
    indel_p, indel_r, indel_f1 = calculate_metrics(indel_tp, indel_fp, indel_fn)
    results["indel"] = {"tp": indel_tp, "fp": indel_fp, "fn": indel_fn,
                        "precision": round(indel_p, 6),
                        "recall": round(indel_r, 6),
                        "f1": round(indel_f1, 6)}
    print(f"  Indel:   TP={indel_tp:,} FP={indel_fp:,} FN={indel_fn:,} "
          f"P={indel_p:.4f} R={indel_r:.4f} F1={indel_f1:.4f}")

    # VCF stats
    total = count_variants(call_vcf_gz)
    pass_count = int(run(f"bcftools view -f PASS {call_vcf_gz} | grep -cv '^#' 2>/dev/null || echo 0"))
    total_snps = int(run(f"bcftools view -v snps -f PASS {call_vcf_gz} | grep -cv '^#' 2>/dev/null || echo 0"))
    total_indels = int(run(f"bcftools view -v indels -f PASS {call_vcf_gz} | grep -cv '^#' 2>/dev/null || echo 0"))
    results["vcf_stats"] = {"total": total, "pass": pass_count,
                            "snps": total_snps, "indels": total_indels}
    print(f"\n  VCF Stats: Total={total:,} PASS={pass_count:,} SNPs={total_snps:,} Indels={total_indels:,}")

    return results


def main():
    vcf_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VCF

    if not os.path.exists(vcf_path):
        print(f"VCF not found: {vcf_path}")
        sys.exit(1)

    if not os.path.exists(TRUTH_VCF):
        print(f"Truth VCF not found: {TRUTH_VCF}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"GIAB NA12878 Benchmark Comparison")
    print(f"{'='*60}")
    print(f"Call VCF:  {vcf_path}")
    print(f"Truth VCF: {TRUTH_VCF}")
    print(f"Truth BED: {TRUTH_BED}")
    print(f"{'='*60}\n")

    t0 = time.time()
    results = run_comparison(vcf_path)
    elapsed = time.time() - t0

    results["comparison_time_seconds"] = round(elapsed, 1)
    results["call_vcf"] = vcf_path
    results["truth_vcf"] = TRUTH_VCF
    results["citation"] = "Zook et al. Nature Biotechnology 2019; GIAB v4.2.1"

    output_path = "/opt/cognisom/benchmark/giab_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    print(f"Comparison took {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
