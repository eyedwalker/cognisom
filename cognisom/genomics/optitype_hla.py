"""
OptiType HLA Typing Integration
==================================

Production-grade HLA-I typing from sequencing reads using OptiType.
Replaces population-frequency guessing with real patient-specific
allele determination.

OptiType:
  - >97% concordance with serological typing
  - Works on WGS, WES, or RNA-seq BAM/FASTQ
  - Identifies 6 HLA-I alleles (A, B, C × 2) at 4-digit resolution
  - CPU-only. Runtime is dominated by razers3 read mapping and scales
    with input size: minutes for a targeted HLA-region extraction,
    but hours for whole-exome FASTQ on a few cores.

Falls back to population-frequency assignment if OptiType is not installed.

Requirements (for production):
  - conda install -c bioconda optitype
  - OR docker pull quay.io/biocontainers/optitype:1.3.5--hdfd78af_3

References:
  Szolek et al., Bioinformatics 2014
  https://github.com/FRED-2/OptiType
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Wall-clock ceiling for a single OptiType run.
#:
#: razers3 read mapping dominates OptiType's runtime and scales with input
#: size, so this has to accommodate whole-exome FASTQ rather than a small
#: HLA-region extraction. Six hours is generous for WES on a few cores;
#: override with COGNISOM_OPTITYPE_TIMEOUT (seconds) for larger inputs.
DEFAULT_OPTITYPE_TIMEOUT = 6 * 60 * 60

#: OptiType container image.
#:
#: This was `fred2/optitype:1.3.5`, a tag that does not exist: that
#: repository stops at `release-v1.3.1`, so every containerised run failed
#: on `manifest unknown`. The bioconda build is where 1.3.5 actually ships.
OPTITYPE_IMAGE = "quay.io/biocontainers/optitype:1.3.5--hdfd78af_3"

#: OptiType refuses to start without a config file naming the razers3 binary.
OPTITYPE_CONFIG = "/usr/local/bin/config.ini"


def optitype_timeout() -> int:
    """Resolve the OptiType wall-clock ceiling, honouring the env override."""
    raw = os.environ.get("COGNISOM_OPTITYPE_TIMEOUT")
    if not raw:
        return DEFAULT_OPTITYPE_TIMEOUT
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "COGNISOM_OPTITYPE_TIMEOUT=%r is not an integer; using %ds",
            raw, DEFAULT_OPTITYPE_TIMEOUT,
        )
        return DEFAULT_OPTITYPE_TIMEOUT
    if value <= 0:
        logger.warning(
            "COGNISOM_OPTITYPE_TIMEOUT=%d is not positive; using %ds",
            value, DEFAULT_OPTITYPE_TIMEOUT,
        )
        return DEFAULT_OPTITYPE_TIMEOUT
    return value


def is_optitype_available() -> bool:
    """Check if OptiType is installed and executable."""
    try:
        result = subprocess.run(
            ["OptiTypePipeline.py", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check Docker
    try:
        result = subprocess.run(
            ["docker", "images", "-q", OPTITYPE_IMAGE],
            capture_output=True, text=True, timeout=5,
        )
        return bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return False


def type_hla_from_bam(
    bam_path: str,
    sample_id: str = "sample",
    use_docker: bool = True,
    output_dir: Optional[str] = None,
) -> List[str]:
    """Type HLA-I alleles from a BAM file using OptiType.

    This extracts reads mapping to the HLA region (chr6:29,000,000-34,000,000),
    runs OptiType's integer linear programming solver, and returns the
    optimal 6-allele combination.

    Args:
        bam_path: Path to aligned BAM file (germline/normal sample).
        sample_id: Sample identifier for output naming.
        use_docker: Use Docker image if OptiType not installed natively.
        output_dir: Where to write OptiType results (temp dir if None).

    Returns:
        List of 6 HLA alleles, e.g.:
        ['HLA-A*02:01', 'HLA-A*03:01', 'HLA-B*07:02', 'HLA-B*44:02',
         'HLA-C*05:01', 'HLA-C*07:02']

    Raises:
        RuntimeError: If OptiType fails or is not available.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="optitype_")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract HLA reads from BAM
    hla_fastq = _extract_hla_reads(bam_path, output_dir)
    if not hla_fastq:
        raise RuntimeError("Failed to extract HLA reads from BAM")

    # Step 2: Run OptiType
    if is_optitype_available():
        result_tsv = _run_optitype_native(hla_fastq, output_dir)
    elif use_docker:
        result_tsv = _run_optitype_docker(hla_fastq, output_dir)
    else:
        raise RuntimeError(
            "OptiType not available. Install via: conda install -c bioconda optitype "
            f"OR: docker pull {OPTITYPE_IMAGE}"
        )

    # Step 3: Parse results
    alleles = _parse_optitype_result(result_tsv)
    logger.info("OptiType HLA typing for %s: %s", sample_id, alleles)
    return alleles


def type_hla_from_fastq(
    fastq_r1: str,
    fastq_r2: Optional[str] = None,
    sample_id: str = "sample",
    use_docker: bool = True,
    output_dir: Optional[str] = None,
) -> List[str]:
    """Type HLA-I alleles directly from FASTQ files.

    Skips the BAM extraction step — useful when you have pre-filtered
    HLA reads or want to type from raw FASTQ.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="optitype_")

    if is_optitype_available():
        result_tsv = _run_optitype_native(fastq_r1, output_dir, fastq_r2)
    elif use_docker:
        result_tsv = _run_optitype_docker(fastq_r1, output_dir, fastq_r2)
    else:
        raise RuntimeError("OptiType not available")

    alleles = _parse_optitype_result(result_tsv)
    logger.info("OptiType HLA typing for %s: %s", sample_id, alleles)
    return alleles


def _extract_hla_reads(bam_path: str, output_dir: str) -> Optional[str]:
    """Extract reads from the HLA region (chr6:29-34 Mb) using samtools."""
    hla_region = "chr6:29000000-34000000"
    output_fastq = os.path.join(output_dir, "hla_reads.fastq")

    try:
        # Extract HLA-mapped reads
        cmd = (
            f"samtools view -b {bam_path} {hla_region} | "
            f"samtools fastq -0 {output_fastq} -"
        )
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0 and os.path.exists(output_fastq):
            size = os.path.getsize(output_fastq)
            logger.info("Extracted HLA reads: %d bytes", size)
            return output_fastq
        else:
            logger.error("samtools failed: %s", result.stderr)
            return None
    except Exception as e:
        logger.error("HLA read extraction failed: %s", e)
        return None


def _run_optitype_native(
    fastq_path: str, output_dir: str, fastq_r2: str = None,
) -> str:
    """Run OptiType via native installation."""
    cmd = [
        "OptiTypePipeline.py",
        "-i", fastq_path,
        "--dna",
        "-o", output_dir,
        "-v",
    ]
    if fastq_r2:
        cmd.insert(3, fastq_r2)

    timeout = optitype_timeout()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"OptiType exceeded its {timeout}s ceiling. Raise "
            f"COGNISOM_OPTITYPE_TIMEOUT, or reduce the input by extracting "
            f"chr6:29-34Mb reads first."
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(f"OptiType failed: {result.stderr}")

    # Find result TSV
    for f in Path(output_dir).rglob("*_result.tsv"):
        return str(f)
    raise RuntimeError("OptiType result file not found")


def _run_optitype_docker(
    fastq_path: str, output_dir: str, fastq_r2: str = None,
) -> str:
    """Run OptiType via Docker container.

    The image is a bioconda build whose entrypoint is a conda activation
    wrapper, so the pipeline script has to be named explicitly as the
    command rather than assumed to be the entrypoint.
    """
    fastq_dir = os.path.dirname(os.path.abspath(fastq_path))
    inputs = [f"/data/{os.path.basename(fastq_path)}"]

    if fastq_r2:
        r2_dir = os.path.dirname(os.path.abspath(fastq_r2))
        if r2_dir != fastq_dir:
            raise ValueError(
                "Paired FASTQs must share a directory: only that one directory "
                f"is mounted into the container. R1 is in {fastq_dir}, "
                f"R2 in {r2_dir}."
            )
        inputs.append(f"/data/{os.path.basename(fastq_r2)}")

    cmd = [
        "docker", "run", "--rm",
        # Without this the results land root-owned on the host.
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
        "-v", f"{fastq_dir}:/data:ro",
        "-v", f"{os.path.abspath(output_dir)}:/out",
        OPTITYPE_IMAGE,
        "OptiTypePipeline.py",
        "-i", *inputs,
        "--dna",
        "-o", "/out",
        "-c", OPTITYPE_CONFIG,
        "-v",
    ]

    timeout = optitype_timeout()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"OptiType exceeded its {timeout}s ceiling. Raise "
            f"COGNISOM_OPTITYPE_TIMEOUT, or reduce the input by extracting "
            f"chr6:29-34Mb reads first."
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(f"OptiType Docker failed: {result.stderr}")

    for f in Path(output_dir).rglob("*_result.tsv"):
        return str(f)
    raise RuntimeError("OptiType Docker result file not found")


def _parse_optitype_result(tsv_path: str) -> List[str]:
    """Parse OptiType result TSV into HLA allele list.

    OptiType output format (TSV):
        A1      A2      B1      B2      C1      C2      Reads   Objective
        A*02:01 A*03:01 B*07:02 B*44:02 C*05:01 C*07:02 1234    567.89
    """
    with open(tsv_path) as f:
        lines = f.readlines()

    # Find the data line (skip header)
    for line in lines:
        if line.startswith("\t") or line[0].isdigit():
            parts = line.strip().split("\t")
            # First 6 columns are alleles
            alleles = []
            for i, part in enumerate(parts[:6]):
                part = part.strip()
                if part and "*" in part:
                    if not part.startswith("HLA-"):
                        part = f"HLA-{part}"
                    alleles.append(part)
            if len(alleles) >= 4:
                return alleles

    raise RuntimeError(f"Could not parse OptiType result: {tsv_path}")


def get_hla_typing_status() -> Dict[str, bool]:
    """Check status of HLA typing tools."""
    return {
        "optitype_native": is_optitype_available(),
        "optitype_docker": _check_docker_image(OPTITYPE_IMAGE),
        "samtools": _check_command("samtools"),
    }


def _check_docker_image(image: str) -> bool:
    try:
        result = subprocess.run(
            ["docker", "images", image, "--format", "{{.ID}}"],
            capture_output=True, text=True, timeout=5,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def _check_command(cmd: str) -> bool:
    try:
        result = subprocess.run(
            [cmd, "--version"], capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False
