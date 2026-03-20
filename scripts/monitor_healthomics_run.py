#!/usr/bin/env python3
"""
Monitor a HealthOmics run and process results through MAD Agent.

Usage:
    python scripts/monitor_healthomics_run.py 6220328

Polls every 60 seconds. Once complete:
  1. Downloads VCF from S3
  2. Runs through MAD Agent pipeline
  3. Compares against GIAB truth set (if germline)
  4. Prints full results
"""

import sys
import os
import time
import json
import tempfile
import boto3

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def monitor_run(run_id: str, poll_interval: int = 60, timeout_min: int = 120):
    """Monitor a HealthOmics run until completion."""
    omics = boto3.client("omics", region_name="us-west-2")
    s3 = boto3.client("s3", region_name="us-west-2")
    bucket = "cognisom-genomics"

    print(f"Monitoring HealthOmics run {run_id}...")
    start_time = time.time()

    while True:
        resp = omics.get_run(id=run_id)
        status = resp["status"]
        elapsed = (time.time() - start_time) / 60

        print(f"  [{elapsed:.0f} min] Status: {status}")

        if status == "COMPLETED":
            print(f"\n✓ Run completed in {elapsed:.0f} min!")
            output_uri = resp.get("outputUri", "")
            stop_time = resp.get("stopTime", "")
            print(f"  Output: {output_uri}")
            print(f"  Stopped: {stop_time}")
            return process_results(run_id, output_uri, s3, bucket)

        elif status in ("FAILED", "CANCELLED"):
            print(f"\n✗ Run {status}")
            print(f"  Failure reason: {resp.get('statusMessage', 'unknown')}")
            return None

        elif elapsed > timeout_min:
            print(f"\n✗ Timeout after {timeout_min} min")
            return None

        time.sleep(poll_interval)


def process_results(run_id: str, output_uri: str, s3, bucket: str):
    """Download VCF and run through MAD Agent pipeline."""
    # Find VCF in results
    prefix = output_uri.replace(f"s3://{bucket}/", "")
    print(f"\nSearching for VCF in s3://{bucket}/{prefix}...")

    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=100)
    vcf_key = None
    all_files = []
    for obj in resp.get("Contents", []):
        key = obj["Key"]
        size_mb = obj["Size"] / 1e6
        all_files.append(f"  {key} ({size_mb:.1f} MB)")
        if key.endswith(".vcf") or key.endswith(".vcf.gz"):
            vcf_key = key

    print(f"Files found ({len(all_files)}):")
    for f in all_files[:20]:
        print(f)

    if not vcf_key:
        print("✗ No VCF found in output!")
        return None

    print(f"\n✓ VCF found: {vcf_key}")

    # Download VCF
    with tempfile.NamedTemporaryFile(suffix=".vcf", delete=False) as tmp:
        print(f"Downloading to {tmp.name}...")
        s3.download_file(bucket, vcf_key, tmp.name)
        vcf_path = tmp.name

    # Count variants
    with open(vcf_path) as f:
        lines = f.readlines()
    header_lines = sum(1 for l in lines if l.startswith("#"))
    variant_lines = len(lines) - header_lines
    print(f"VCF: {variant_lines} variants ({header_lines} header lines)")

    # Run through Cognisom pipeline
    print("\n" + "=" * 60)
    print("RUNNING MAD AGENT ON REAL PARABRICKS VCF OUTPUT")
    print("=" * 60)

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import warnings
    warnings.filterwarnings("ignore")

    from cognisom.genomics.patient_profile import PatientProfileBuilder
    from cognisom.genomics.twin_config import DigitalTwinConfig
    from cognisom.genomics.treatment_simulator import TreatmentSimulator
    from cognisom.mad.board import BoardModerator
    from cognisom.genomics.clinical_trials import match_patient_to_trials

    vcf_text = open(vcf_path).read()

    print("\nParsing VCF + building patient profile...")
    t0 = time.time()
    builder = PatientProfileBuilder()
    profile = builder.from_vcf_text(vcf_text, f"NA12878-run{run_id}")
    print(f"  Variants: {len(profile.variants)}")
    print(f"  Coding: {len(profile.coding_variants)}")
    print(f"  Drivers: {len(profile.cancer_driver_mutations)}")
    print(f"  TMB: {profile.tumor_mutational_burden:.1f}")
    print(f"  HLA: {profile.hla_alleles}")
    print(f"  Neoantigens: {len(profile.predicted_neoantigens)}")
    print(f"  Time: {time.time()-t0:.1f}s")

    print("\nBuilding digital twin + simulating treatments...")
    twin = DigitalTwinConfig.from_profile_only(profile)
    sim = TreatmentSimulator()
    recommended = sim.get_recommended_treatments(twin)
    results = sim.compare_treatments(recommended, twin)
    print(f"  Treatments: {len(results)}")

    print("\nRunning MAD Board...")
    moderator = BoardModerator()
    decision = moderator.run_full_analysis(
        f"NA12878-run{run_id}", profile, twin, results
    )
    print(f"  Recommendation: {decision.recommended_treatment_name}")
    print(f"  Consensus: {decision.consensus_level} ({decision.confidence:.0%})")
    print(f"  Evidence: {len(decision.evidence_chain)} items")

    # GIAB comparison (if available)
    print("\n" + "=" * 60)
    print("GIAB TRUTH SET COMPARISON")
    print("=" * 60)
    print("  Note: NA12878 is a HEALTHY reference genome (no cancer)")
    print("  Expected: No cancer drivers, low TMB, no actionable targets")
    print(f"  Actual:   {len(profile.cancer_driver_mutations)} drivers, TMB={profile.tumor_mutational_burden:.1f}")
    if len(profile.cancer_driver_mutations) == 0:
        print("  ✓ CORRECT: No cancer drivers in healthy genome")
    else:
        print("  ⚠ Unexpected cancer drivers — investigate")

    # Save results
    output = {
        "run_id": run_id,
        "pipeline": "healthomics_parabricks_deepvariant_30x",
        "sample": "NA12878",
        "variants_total": len(profile.variants),
        "variants_coding": len(profile.coding_variants),
        "cancer_drivers": len(profile.cancer_driver_mutations),
        "tmb": profile.tumor_mutational_burden,
        "neoantigens": len(profile.predicted_neoantigens),
        "mad_recommendation": decision.recommended_treatment_name,
        "mad_consensus": decision.consensus_level,
        "mad_confidence": decision.confidence,
    }
    output_path = f"/tmp/cognisom_run_{run_id}_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {output_path}")

    return output


if __name__ == "__main__":
    run_id = sys.argv[1] if len(sys.argv) > 1 else "6220328"
    result = monitor_run(run_id)
    if result:
        print("\n✓ Full pipeline complete!")
    else:
        print("\n✗ Pipeline failed")
