"""
Cognisom Orchestrator
======================

Autonomous pipeline that runs the complete patient analysis workflow:
FASTQ → Parabricks → VCF → Annotation → HLA → Neoantigens →
Digital Twin → Treatment Simulation → Clinical Report

The orchestrator:
1. Detects available input data (FASTQ, BAM, or VCF)
2. Starts GPU if needed for Parabricks
3. Runs each pipeline step in sequence
4. Tracks progress and logs
5. Generates the final clinical report
6. Stops GPU when complete

Can run from the dashboard (interactive) or as a background job.

Usage:
    orchestrator = CognisomOrchestrator()
    result = orchestrator.run("MAYO-001",
        vcf_path="/path/to/patient.vcf",  # or vcf_text="...",
        # or fastq_r1="/path/to/R1.fq.gz", fastq_r2="/path/to/R2.fq.gz",
    )
    print(result.status)  # "completed"
    print(result.report)  # Clinical report dict
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class PipelineStep(str, Enum):
    INIT = "init"
    GPU_START = "gpu_start"
    PARABRICKS = "parabricks"
    VCF_PARSE = "vcf_parse"
    VARIANT_ANNOTATE = "variant_annotate"
    HLA_TYPING = "hla_typing"
    NEOANTIGEN = "neoantigen"
    DIGITAL_TWIN = "digital_twin"
    TREATMENT_SIM = "treatment_sim"
    MAD_BOARD = "mad_board"
    CLINICAL_REPORT = "clinical_report"
    GPU_STOP = "gpu_stop"
    COMPLETE = "complete"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class StepResult:
    step: PipelineStep
    status: StepStatus = StepStatus.PENDING
    message: str = ""
    duration_seconds: float = 0.0
    data: Dict = field(default_factory=dict)


@dataclass
class OrchestratorResult:
    """Complete result of an orchestrator run."""
    patient_id: str = ""
    status: str = "pending"  # pending, running, completed, failed
    started_at: str = ""
    completed_at: str = ""
    total_duration_seconds: float = 0.0
    steps: List[StepResult] = field(default_factory=list)

    # Pipeline outputs
    profile: Any = None  # PatientProfile
    twin: Any = None  # DigitalTwinConfig
    treatments: List[Any] = field(default_factory=list)  # List[TreatmentResult]
    mad_decision: Any = None  # Optional BoardDecision (from MAD Agent)
    report: Dict = field(default_factory=dict)

    # Summary
    variants_found: int = 0
    drivers_found: int = 0
    neoantigens_found: int = 0
    treatments_simulated: int = 0
    best_treatment: str = ""
    best_response: str = ""

    def to_dict(self) -> Dict:
        return {
            "patient_id": self.patient_id,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_duration_seconds": round(self.total_duration_seconds, 1),
            "variants_found": self.variants_found,
            "drivers_found": self.drivers_found,
            "neoantigens_found": self.neoantigens_found,
            "treatments_simulated": self.treatments_simulated,
            "best_treatment": self.best_treatment,
            "best_response": self.best_response,
            "steps": [
                {"step": s.step.value, "status": s.status.value,
                 "message": s.message, "duration": round(s.duration_seconds, 1)}
                for s in self.steps
            ],
        }


class CognisomOrchestrator:
    """Autonomous pipeline orchestrator.

    Runs the complete Cognisom analysis pipeline from any input format
    (FASTQ, BAM, VCF) to clinical report.
    """

    def __init__(self, progress_callback: Optional[Callable] = None):
        """
        Args:
            progress_callback: Called with (step_name, status, message)
                              for real-time progress updates.
        """
        self.progress = progress_callback or (lambda *a: None)
        self._store = None

    @property
    def store(self):
        if self._store is None:
            from cognisom.library.store import EntityStore
            self._store = EntityStore()
        return self._store

    def run(self,
            patient_id: str,
            vcf_text: Optional[str] = None,
            vcf_path: Optional[str] = None,
            fastq_r1: Optional[str] = None,
            fastq_r2: Optional[str] = None,
            bam_path: Optional[str] = None,
            pipeline_type: str = "germline",  # germline, somatic, rnaseq
            tumor_r1: Optional[str] = None,
            tumor_r2: Optional[str] = None,
            normal_r1: Optional[str] = None,
            normal_r2: Optional[str] = None,
            auto_stop_gpu: bool = True,
            use_healthomics: bool = False,
            ) -> OrchestratorResult:
        """Run the complete analysis pipeline.

        Automatically detects the appropriate entry point based on
        provided inputs and runs all downstream steps.

        Args:
            patient_id: Patient identifier
            vcf_text: VCF content as string (highest priority)
            vcf_path: Path to VCF file
            fastq_r1/r2: Paired FASTQ paths (triggers Parabricks)
            bam_path: Pre-aligned BAM (triggers variant calling only)
            pipeline_type: "germline", "somatic", "rnaseq"
            tumor_r1/r2, normal_r1/r2: For somatic pipeline
            auto_stop_gpu: Stop GPU instance after completion

        Returns:
            OrchestratorResult with all outputs
        """
        result = OrchestratorResult(
            patient_id=patient_id,
            status="running",
            started_at=datetime.now().isoformat(),
        )
        t0 = time.time()

        try:
            # ── Step 1: Initialize ──────────────────────────────────
            self._run_step(result, PipelineStep.INIT, "Initializing pipeline",
                           lambda: self._step_init(patient_id))

            # ── Step 2: GPU + Parabricks (if FASTQ/BAM) ────────────
            if fastq_r1 or bam_path:
                if use_healthomics:
                    # HealthOmics serverless path — no GPU needed
                    result.steps.append(StepResult(
                        PipelineStep.GPU_START, StepStatus.SKIPPED,
                        "Using HealthOmics (serverless)"))
                    self._run_step(result, PipelineStep.PARABRICKS,
                                   "Running HealthOmics Parabricks (serverless)",
                                   lambda: self._step_healthomics(
                                       fastq_r1, fastq_r2, patient_id,
                                       pipeline_type,
                                       tumor_r1=tumor_r1, tumor_r2=tumor_r2,
                                       normal_r1=normal_r1, normal_r2=normal_r2))
                else:
                    # Self-managed GPU path
                    self._run_step(result, PipelineStep.GPU_START,
                                   "Starting GPU instance",
                                   lambda: self._step_start_gpu())
                    self._run_step(result, PipelineStep.PARABRICKS,
                                   f"Running Parabricks ({pipeline_type})",
                                   lambda: self._step_parabricks(
                                       fastq_r1, fastq_r2, bam_path,
                                       pipeline_type, patient_id,
                                       tumor_r1, tumor_r2, normal_r1, normal_r2))

                # Get VCF from Parabricks output
                pb_data = result.steps[-1].data
                vcf_text = pb_data.get("vcf_text", "")
            else:
                result.steps.append(StepResult(
                    PipelineStep.GPU_START, StepStatus.SKIPPED,
                    "VCF provided, GPU not needed"))
                result.steps.append(StepResult(
                    PipelineStep.PARABRICKS, StepStatus.SKIPPED,
                    "VCF provided, Parabricks not needed"))

            # ── Step 3: Load VCF ────────────────────────────────────
            if not vcf_text and vcf_path:
                vcf_text = Path(vcf_path).read_text()

            if not vcf_text:
                raise ValueError("No VCF data available. Provide vcf_text, vcf_path, or FASTQ files.")

            # ── Step 4: Parse + Annotate + HLA + Neoantigens ────────
            self._run_step(result, PipelineStep.VCF_PARSE,
                           "Parsing VCF variants",
                           lambda: self._step_parse_vcf(vcf_text, patient_id))

            profile = result.steps[-1].data.get("profile")
            if not profile:
                raise ValueError("VCF parsing failed — no profile generated")

            result.profile = profile
            result.variants_found = len(profile.variants)
            result.drivers_found = len(profile.cancer_driver_mutations)
            result.neoantigens_found = len(profile.predicted_neoantigens)

            # Steps 5-6 happen inside PatientProfileBuilder automatically
            result.steps.append(StepResult(
                PipelineStep.VARIANT_ANNOTATE, StepStatus.COMPLETED,
                f"Annotated {result.drivers_found} cancer drivers",
                data={"drivers": [v.gene for v in profile.cancer_driver_mutations]}))
            result.steps.append(StepResult(
                PipelineStep.HLA_TYPING, StepStatus.COMPLETED,
                f"HLA: {', '.join(profile.hla_alleles or [])}",
                data={"hla": profile.hla_alleles}))
            result.steps.append(StepResult(
                PipelineStep.NEOANTIGEN, StepStatus.COMPLETED,
                f"{result.neoantigens_found} neoantigens, "
                f"{len(profile.vaccine_neoantigens)} vaccine candidates"))

            # ── Step 7: Digital Twin ────────────────────────────────
            self._run_step(result, PipelineStep.DIGITAL_TWIN,
                           "Building personalized digital twin",
                           lambda: self._step_digital_twin(profile))

            twin = result.steps[-1].data.get("twin")
            result.twin = twin

            # ── Step 8: Treatment Simulation ────────────────────────
            self._run_step(result, PipelineStep.TREATMENT_SIM,
                           "Simulating treatments (entity-driven)",
                           lambda: self._step_treatment_sim(twin))

            treatments = result.steps[-1].data.get("treatments", [])
            result.treatments = treatments
            result.treatments_simulated = len(treatments)

            if treatments:
                best = min(treatments, key=lambda t: t.best_response)
                result.best_treatment = best.treatment_name
                result.best_response = best.response_category

            # ── Step 8b: MAD Board (optional) ─────────────────────────
            self._run_step(result, PipelineStep.MAD_BOARD,
                           "MAD Board deliberation",
                           lambda: self._step_mad_board(profile, twin, treatments))

            mad_data = result.steps[-1].data
            if mad_data:
                result.mad_decision = mad_data.get("decision")

            # ── Step 9: Clinical Report ─────────────────────────────
            self._run_step(result, PipelineStep.CLINICAL_REPORT,
                           "Generating clinical report",
                           lambda: self._step_clinical_report(
                               profile, twin, treatments, patient_id))

            result.report = result.steps[-1].data.get("report", {})

            # ── Step 10: GPU Stop ───────────────────────────────────
            if auto_stop_gpu and (fastq_r1 or bam_path):
                self._run_step(result, PipelineStep.GPU_STOP,
                               "Stopping GPU instance",
                               lambda: self._step_stop_gpu())
            else:
                result.steps.append(StepResult(
                    PipelineStep.GPU_STOP, StepStatus.SKIPPED))

            # ── Complete ────────────────────────────────────────────
            result.status = "completed"
            result.steps.append(StepResult(
                PipelineStep.COMPLETE, StepStatus.COMPLETED,
                f"Pipeline complete: {result.variants_found} variants, "
                f"{result.drivers_found} drivers, {result.treatments_simulated} treatments"))

        except Exception as e:
            result.status = "failed"
            logger.exception("Orchestrator failed for %s", patient_id)
            result.steps.append(StepResult(
                PipelineStep.COMPLETE, StepStatus.FAILED, str(e)))

        result.completed_at = datetime.now().isoformat()
        result.total_duration_seconds = time.time() - t0

        logger.info(
            "Orchestrator %s for %s in %.1fs: %d variants, %d drivers, "
            "%d treatments, best=%s (%s)",
            result.status, patient_id, result.total_duration_seconds,
            result.variants_found, result.drivers_found,
            result.treatments_simulated, result.best_treatment, result.best_response,
        )
        return result

    def _run_step(self, result: OrchestratorResult, step: PipelineStep,
                  message: str, func: Callable) -> StepResult:
        """Run a pipeline step with timing and error handling."""
        self.progress(step.value, "running", message)
        step_result = StepResult(step, StepStatus.RUNNING, message)
        result.steps.append(step_result)

        t0 = time.time()
        try:
            data = func()
            step_result.status = StepStatus.COMPLETED
            step_result.data = data or {}
            step_result.duration_seconds = time.time() - t0
            self.progress(step.value, "completed", message)
        except Exception as e:
            step_result.status = StepStatus.FAILED
            step_result.message = f"{message}: {e}"
            step_result.duration_seconds = time.time() - t0
            self.progress(step.value, "failed", str(e))
            raise

        return step_result

    # ── Step Implementations ──────────────────────────────────────

    def _step_init(self, patient_id: str) -> Dict:
        """Initialize pipeline."""
        return {"patient_id": patient_id, "entity_count": self.store.stats()["total_entities"]}

    def _step_start_gpu(self) -> Dict:
        """Start GPU instance if needed."""
        from cognisom.infrastructure.gpu_connector import (
            get_gpu_instance_state, start_gpu_instance, wait_for_kit)

        state = get_gpu_instance_state()
        if state == "running":
            return {"state": "already_running"}

        ok, msg = start_gpu_instance()
        if not ok:
            raise RuntimeError(f"GPU start failed: {msg}")

        # Wait for instance to be ready (not Kit, just SSH/SSM)
        import boto3
        ec2 = boto3.client("ec2", region_name="us-west-2")
        waiter = ec2.get_waiter("instance_status_ok")
        waiter.wait(InstanceIds=["i-0ac9eb88c1b046163"])

        return {"state": "started"}

    def _step_parabricks(self, fastq_r1, fastq_r2, bam_path,
                          pipeline_type, patient_id,
                          tumor_r1, tumor_r2, normal_r1, normal_r2) -> Dict:
        """Run Parabricks pipeline."""
        from cognisom.genomics.parabricks_runner import ParabricksRunner
        runner = ParabricksRunner()

        if pipeline_type == "somatic" and tumor_r1:
            job_id = runner.run_somatic(tumor_r1, tumor_r2, normal_r1, normal_r2, patient_id)
        elif fastq_r1:
            job_id = runner.run_germline(fastq_r1, fastq_r2, patient_id)
        else:
            return {"skipped": True, "reason": "No FASTQ provided"}

        # Wait for completion (poll every 30s)
        for _ in range(240):  # Max 2 hours
            time.sleep(30)
            status = runner.get_job_status(job_id)
            if status.get("state") == "completed":
                vcf_text = runner.get_result_vcf(job_id)
                return {"job_id": job_id, "vcf_text": vcf_text}
            elif status.get("state") == "failed":
                raise RuntimeError(f"Parabricks failed: {status}")

        raise RuntimeError("Parabricks timed out after 2 hours")

    def _step_healthomics(self, fastq_r1: str, fastq_r2: str,
                            patient_id: str, pipeline_type: str,
                            tumor_r1: str = None, tumor_r2: str = None,
                            normal_r1: str = None, normal_r2: str = None,
                            ) -> Dict:
        """Run pipeline via AWS HealthOmics Ready2Run.

        Supports both germline (single sample) and somatic (matched
        tumor-normal) workflows. Somatic requires 3-step execution:
        1. Align tumor FASTQs → tumor BAM (Parabricks fq2bam)
        2. Align normal FASTQs → normal BAM (Parabricks fq2bam)
        3. Somatic calling: tumor BAM + normal BAM → VCF (Mutect2)
        """
        from cognisom.infrastructure.healthomics import HealthOmicsRunner
        import boto3
        import tempfile

        runner = HealthOmicsRunner()
        bucket = "cognisom-genomics"

        if pipeline_type == "somatic":
            return self._run_healthomics_somatic(
                runner, patient_id, bucket,
                tumor_r1 or fastq_r1, tumor_r2 or fastq_r2,
                normal_r1, normal_r2,
            )

        # --- Germline path ---
        run_id = runner.run_germline_pipeline(
            fastq_r1=fastq_r1,
            fastq_r2=fastq_r2,
            sample_id=patient_id,
            use_parabricks=True,
        )

        if not run_id:
            raise RuntimeError("HealthOmics run failed to start")

        status = runner.wait_for_completion(run_id, timeout_minutes=60)
        if status.get("status") != "COMPLETED":
            raise RuntimeError(f"HealthOmics run failed: {status}")

        vcf_text = self._download_vcf_from_s3(bucket, patient_id)
        return {"run_id": run_id, "vcf_text": vcf_text, "execution": "healthomics"}

    def _run_healthomics_somatic(self, runner, patient_id: str, bucket: str,
                                   tumor_r1: str, tumor_r2: str,
                                   normal_r1: str, normal_r2: str) -> Dict:
        """Run matched tumor-normal somatic pipeline via HealthOmics.

        3-step process:
        1. Align tumor FASTQs → BAM
        2. Align normal FASTQs → BAM
        3. Mutect2 somatic calling on both BAMs
        """
        if not normal_r1 or not normal_r2:
            raise ValueError(
                "Somatic pipeline requires both tumor and normal FASTQ pairs. "
                "Provide normal_r1 and normal_r2 (matched germline sample)."
            )

        # Step 1: Align tumor
        self.progress(
            "parabricks", "running",
            f"Aligning tumor sample (HealthOmics)...",
        )
        tumor_run_id = runner.run_germline_pipeline(
            fastq_r1=tumor_r1, fastq_r2=tumor_r2,
            sample_id=f"{patient_id}-tumor",
            use_parabricks=True,
        )
        if not tumor_run_id:
            raise RuntimeError("Tumor alignment failed to start")

        # Step 2: Align normal (can run concurrently)
        self.progress(
            "parabricks", "running",
            f"Aligning normal sample (HealthOmics)...",
        )
        normal_run_id = runner.run_germline_pipeline(
            fastq_r1=normal_r1, fastq_r2=normal_r2,
            sample_id=f"{patient_id}-normal",
            use_parabricks=True,
        )
        if not normal_run_id:
            raise RuntimeError("Normal alignment failed to start")

        # Wait for both alignments
        tumor_status = runner.wait_for_completion(tumor_run_id, timeout_minutes=90)
        if tumor_status.get("status") != "COMPLETED":
            raise RuntimeError(f"Tumor alignment failed: {tumor_status}")

        normal_status = runner.wait_for_completion(normal_run_id, timeout_minutes=90)
        if normal_status.get("status") != "COMPLETED":
            raise RuntimeError(f"Normal alignment failed: {normal_status}")

        # Find BAM outputs
        tumor_bam = self._find_s3_file(bucket, f"results/{patient_id}-tumor/", ".bam")
        normal_bam = self._find_s3_file(bucket, f"results/{patient_id}-normal/", ".bam")

        if not tumor_bam or not normal_bam:
            raise RuntimeError("BAM files not found in HealthOmics output")

        # Step 3: Somatic calling (Mutect2)
        self.progress(
            "parabricks", "running",
            "Running Mutect2 somatic calling (HealthOmics)...",
        )
        somatic_run_id = runner.run_somatic_pipeline(
            tumor_bam=f"s3://{bucket}/{tumor_bam}",
            normal_bam=f"s3://{bucket}/{normal_bam}",
            sample_id=patient_id,
        )
        if not somatic_run_id:
            raise RuntimeError("Somatic calling failed to start")

        somatic_status = runner.wait_for_completion(somatic_run_id, timeout_minutes=120)
        if somatic_status.get("status") != "COMPLETED":
            raise RuntimeError(f"Somatic calling failed: {somatic_status}")

        vcf_text = self._download_vcf_from_s3(bucket, patient_id)
        return {
            "run_id": somatic_run_id,
            "tumor_alignment_run": tumor_run_id,
            "normal_alignment_run": normal_run_id,
            "vcf_text": vcf_text,
            "execution": "healthomics_somatic",
            "pipeline": "matched_tumor_normal",
        }

    def _download_vcf_from_s3(self, bucket: str, patient_id: str) -> str:
        """Download VCF from S3 results directory."""
        import boto3
        import tempfile

        vcf_key = self._find_s3_file(bucket, f"results/{patient_id}/", ".vcf")
        if not vcf_key:
            raise RuntimeError(f"No VCF found in s3://{bucket}/results/{patient_id}/")

        s3 = boto3.client("s3", region_name="us-west-2")
        with tempfile.NamedTemporaryFile(suffix=".vcf", delete=False) as tmp:
            s3.download_file(bucket, vcf_key, tmp.name)
            vcf_text = Path(tmp.name).read_text()
        return vcf_text

    def _find_s3_file(self, bucket: str, prefix: str, suffix: str) -> Optional[str]:
        """Find a file with given suffix in S3 prefix."""
        import boto3
        s3 = boto3.client("s3", region_name="us-west-2")
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if key.endswith(suffix) or key.endswith(suffix + ".gz"):
                return key
        return None

    def _step_parse_vcf(self, vcf_text: str, patient_id: str) -> Dict:
        """Parse VCF and build patient profile (includes HLA + neoantigens)."""
        from cognisom.genomics.patient_profile import PatientProfileBuilder
        builder = PatientProfileBuilder()
        profile = builder.from_vcf_text(vcf_text, patient_id=patient_id)
        return {"profile": profile}

    def _step_digital_twin(self, profile) -> Dict:
        """Build digital twin from profile."""
        from cognisom.genomics.twin_config import DigitalTwinConfig
        twin = DigitalTwinConfig.from_profile_only(profile)
        return {
            "twin": twin,
            "mhc1_downreg": twin.mhc1_downregulation,
            "neoantigen_count": twin.neoantigen_count,
        }

    def _step_treatment_sim(self, twin) -> Dict:
        """Simulate all recommended treatments."""
        from cognisom.genomics.treatment_simulator import TreatmentSimulator
        sim = TreatmentSimulator(store=self.store)
        recommended = sim.get_recommended_treatments(twin)

        # Also add top entity-driven drugs
        available = sim.get_available_treatments()
        all_treatments = list(set(recommended + available[:10]))

        results = sim.compare_treatments(all_treatments[:12], twin, 180)
        return {
            "treatments": results,
            "recommended": recommended,
        }

    def _step_mad_board(self, profile, twin, treatments) -> Dict:
        """Run the MAD Board (Molecular AI Decision) multi-agent analysis."""
        try:
            from cognisom.mad.board import BoardModerator
            moderator = BoardModerator()
            decision = moderator.run_full_analysis(
                patient_id=profile.patient_id,
                profile=profile,
                twin=twin,
                treatment_results=treatments,
            )
            return {
                "decision": decision,
                "recommended": decision.recommended_treatment,
                "consensus": decision.consensus_level,
                "confidence": decision.confidence,
            }
        except Exception as e:
            logger.warning(f"MAD Board step failed (non-fatal): {e}")
            return {}

    def _step_clinical_report(self, profile, twin, treatments, patient_id) -> Dict:
        """Generate clinical report data."""
        report = {
            "report_type": "cognisom_autonomous_analysis",
            "version": "2.0",
            "generated": datetime.now().isoformat(),
            "patient_id": patient_id,
            "pipeline": "autonomous_orchestrator",
            "genomic_summary": {
                "total_variants": len(profile.variants),
                "cancer_drivers": len(profile.cancer_driver_mutations),
                "tmb": profile.tumor_mutational_burden,
                "msi_status": profile.msi_status,
                "tmb_high": profile.is_tmb_high,
                "dna_repair_defect": profile.has_dna_repair_defect,
                "hla_alleles": profile.hla_alleles,
            },
            "neoantigens": {
                "total_predicted": len(profile.predicted_neoantigens),
                "vaccine_candidates": len(profile.vaccine_neoantigens),
                "strong_binders": profile.strong_binder_count,
                "vaccine_eligible": profile.neoantigen_vaccine_candidate,
            },
            "digital_twin": {
                "mhc1_downregulation": twin.mhc1_downregulation,
                "immune_score": twin.immune_score,
                "neoantigen_count": twin.neoantigen_count,
            },
            "treatments": [
                {
                    "name": t.treatment_name,
                    "response": t.response_category,
                    "best_response_pct": f"{(1-t.best_response)*100:.0f}%",
                    "pfs_days": t.progression_free_days,
                    "irae_risk": round(t.immune_related_adverse_events, 2),
                }
                for t in treatments
            ],
            "recommendations": profile.get_therapy_recommendations(),
            "driver_mutations": profile.get_driver_details(),
            "entity_library": {
                "entities_used": self.store.stats()["total_entities"],
                "drugs_available": len([t for t in treatments]),
                "source": "entity_driven",
            },
        }
        return {"report": report}

    def _step_stop_gpu(self) -> Dict:
        """Stop GPU instance."""
        try:
            import boto3
            ec2 = boto3.client("ec2", region_name="us-west-2")
            ec2.stop_instances(InstanceIds=["i-0ac9eb88c1b046163"])
            return {"state": "stopping"}
        except Exception as e:
            return {"state": "stop_failed", "error": str(e)}
