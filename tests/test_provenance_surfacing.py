"""
Provenance: every derived number must name where it came from.

Four places asserted a source rather than reporting one:

  * ``OncoKBClient`` answered from a hardcoded gene-level table when the
    API was unreachable, and ``GenomicsAgent`` labelled the result
    "OncoKB: BRCA2 p.E1143fs" -- a curated guess entering the evidence
    chain and the FDA audit hash under a database lookup's name, with a
    +0.02 confidence bump per variant.

  * Three separate sites hardcoded ``"pwm-v1"`` as the neoantigen
    predictor version. In production MHCflurry is installed and does
    run, so the audit record named a model that did not produce the
    numbers.

  * ``HLATyper`` returned a bare list of alleles from five very different
    paths -- OptiType typing, a VCF annotation, or a fixed population
    profile identical for every patient -- with nothing distinguishing
    them. ``type_from_bam``, the only real typing path, had no callers.

  * ``TreatmentSimulator`` seeded NumPy's global RNG from ``hash()`` of a
    string, which Python salts per process, so a "seeded" trajectory
    differed on every run and unrelated draws elsewhere were disturbed.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.genomics.hla_typer import HLATyper
from cognisom.genomics.mhcflurry_binding import (
    MHCFLURRY_SCORER,
    NO_SCORER,
    PWM_SCORER,
    active_scorer_name,
)
from cognisom.genomics.oncokb_client import OncoKBClient, OncoKBAnnotation
from cognisom.genomics.treatment_simulator import TreatmentSimulator
from cognisom.genomics.twin_config import DigitalTwinConfig
from cognisom.mad.model_cards import get_neoantigen_predictor_card


# ── OncoKB fallback declares itself ─────────────────────────────────

def test_offline_annotation_is_marked_as_the_builtin_table():
    client = OncoKBClient(api_token=None)
    ann = client.annotate_mutation("BRCA2", "p.E1143fs")

    assert ann.source == "builtin-kb"
    assert ann.is_fallback is True
    assert ann.matched_protein_change is False
    assert "gene symbol only" in ann.source_detail


def test_the_offline_table_is_gene_level_not_variant_level():
    """Every variant in a gene gets the same verdict, so say so."""
    client = OncoKBClient(api_token=None)
    real = client.annotate_mutation("BRCA2", "p.E1143fs")
    nonsense = client.annotate_mutation("BRCA2", "p.Q9999Z")

    assert real.oncogenic == nonsense.oncogenic
    assert real.is_fallback and nonsense.is_fallback


def test_a_live_annotation_defaults_to_the_api_source():
    ann = OncoKBAnnotation(
        gene="BRAF", variant="V600E", oncogenic="Oncogenic",
        mutation_effect="Gain-of-function",
        highest_sensitive_level="LEVEL_1", highest_resistance_level="",
    )
    assert ann.source == "oncokb-api"
    assert ann.is_fallback is False
    assert ann.matched_protein_change is True


def test_evidence_from_the_offline_table_is_not_labelled_oncokb():
    from cognisom.mad.agents import GenomicsAgent
    from cognisom.genomics.patient_profile import PatientProfileBuilder
    from cognisom.genomics.synthetic_vcf import SYNTHETIC_PROSTATE_VCF
    from cognisom.genomics import gene_protein_mapper as gpm

    gpm.GeneProteinMapper._fetch_from_uniprot = lambda self, gene: None
    profile = PatientProfileBuilder().from_vcf_text(
        SYNTHETIC_PROSTATE_VCF, "prov-test"
    )
    twin = DigitalTwinConfig.from_profile_only(profile)
    opinion = GenomicsAgent().analyze(twin=twin, profile=profile)

    for item in opinion.evidence_items:
        if item.source_id.startswith("oncokb:"):
            # Only a genuine API hit may claim the OncoKB name.
            assert item.supporting_data.get("source") == "oncokb-api"
        if item.source_id.startswith("builtin-kb:"):
            assert "OncoKB" not in item.source_name


# ── Scorer identity ─────────────────────────────────────────────────

def test_active_scorer_name_never_raises_and_is_one_of_the_known_values():
    assert active_scorer_name() in {MHCFLURRY_SCORER, PWM_SCORER, NO_SCORER}


def test_the_model_card_describes_the_scorer_that_is_actually_in_effect():
    card = get_neoantigen_predictor_card()
    scorer = active_scorer_name()

    if scorer == MHCFLURRY_SCORER:
        assert card.version == MHCFLURRY_SCORER
        assert card.model_type == "neural-network-ensemble"
    else:
        assert card.model_type == "position-weight-matrix"


def test_provenance_record_resolves_the_scorer_rather_than_asserting_one():
    from cognisom.mad.provenance import DataProvenance

    versions = DataProvenance().module_versions
    assert versions["neoantigen_predictor"] != "pwm-v1.0"
    assert versions["neoantigen_predictor"] == active_scorer_name()


# ── HLA typing declares its method ──────────────────────────────────

def test_a_default_population_profile_is_not_reported_as_patient_specific():
    typer = HLATyper()
    alleles = typer.type_from_variants([], patient_id="never-seen-before")

    assert len(alleles) >= 4
    assert typer.typing_method == HLATyper.METHOD_POPULATION_DEFAULT
    assert typer.is_patient_specific is False


def test_two_different_patients_get_the_same_default_alleles():
    """Which is exactly why it must be flagged."""
    a = HLATyper().type_from_variants([], patient_id="patient-a")
    b = HLATyper().type_from_variants([], patient_id="patient-b")
    assert a == b


def test_a_predefined_profile_counts_as_patient_specific():
    from cognisom.genomics.hla_typer import SYNTHETIC_HLA_PROFILES

    known = next(k for k in SYNTHETIC_HLA_PROFILES if k != "default")
    typer = HLATyper()
    typer.type_from_variants([], patient_id=known)

    assert typer.typing_method == HLATyper.METHOD_PREDEFINED
    assert typer.is_patient_specific is True


def test_typing_method_is_reported_before_any_call():
    assert HLATyper().typing_method == "not-run"


def test_profile_carries_the_hla_typing_method():
    from cognisom.genomics.patient_profile import PatientProfileBuilder
    from cognisom.genomics.synthetic_vcf import SYNTHETIC_PROSTATE_VCF
    from cognisom.genomics import gene_protein_mapper as gpm

    gpm.GeneProteinMapper._fetch_from_uniprot = lambda self, gene: None
    profile = PatientProfileBuilder().from_vcf_text(
        SYNTHETIC_PROSTATE_VCF, "hla-test"
    )
    payload = profile.to_dict()

    assert payload["hla_typing_method"] == profile.hla_typing_method
    assert isinstance(payload["hla_is_patient_specific"], bool)


# ── Reproducibility ─────────────────────────────────────────────────

def test_tumor_trajectory_is_identical_across_processes():
    """`hash()` on a str is salted per process (PYTHONHASHSEED).

    Seeding from it meant the same patient and regimen produced a
    different curve on every run -- the one thing a seed prevents.
    """
    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "from cognisom.genomics.treatment_simulator import TreatmentSimulator\n"
        "from cognisom.genomics.twin_config import DigitalTwinConfig\n"
        "v = TreatmentSimulator()._simulate_tumor_dynamics(\n"
        "    0.6, {'name': 'Olaparib', 'effect_onset_days': 14},\n"
        "    DigitalTwinConfig(), 30)\n"
        "print(v[-1])\n" % str(REPO_ROOT)
    )

    outputs = []
    for hash_seed in ("0", "1", "12345"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True,
            env={"PYTHONHASHSEED": hash_seed, "PATH": "/usr/bin:/bin"},
        )
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout.strip())

    assert len(set(outputs)) == 1, f"trajectory varied by hash seed: {outputs}"


def test_simulation_does_not_disturb_the_global_numpy_rng():
    """np.random.seed() reaches out and changes unrelated draws."""
    np.random.seed(1234)
    expected = np.random.random()

    np.random.seed(1234)
    TreatmentSimulator()._simulate_tumor_dynamics(
        0.6, {"name": "Olaparib", "effect_onset_days": 14},
        DigitalTwinConfig(), 30,
    )
    assert np.random.random() == expected


def test_same_regimen_gives_the_same_curve_within_a_process():
    sim = TreatmentSimulator()
    args = (0.6, {"name": "Pembrolizumab", "effect_onset_days": 14},
            DigitalTwinConfig(), 30)
    assert sim._simulate_tumor_dynamics(*args) == sim._simulate_tumor_dynamics(*args)


def test_different_regimens_give_different_noise():
    sim = TreatmentSimulator()
    twin = DigitalTwinConfig()
    a = sim._simulate_tumor_dynamics(
        0.6, {"name": "Olaparib", "effect_onset_days": 14}, twin, 30)
    b = sim._simulate_tumor_dynamics(
        0.6, {"name": "Pembrolizumab", "effect_onset_days": 14}, twin, 30)
    assert a != b


# ── HLA typing from reads ───────────────────────────────────────────

def test_fastq_typing_is_reported_as_patient_specific(monkeypatch):
    """OptiType output is the patient's genotype, unlike the default."""
    from cognisom.genomics import optitype_hla

    monkeypatch.setattr(optitype_hla, "is_optitype_available", lambda: True)
    monkeypatch.setattr(
        optitype_hla, "type_hla_from_fastq",
        lambda r1, r2=None, sample_id="s", **kw: [
            "HLA-A*02:01", "HLA-A*11:01", "HLA-B*15:01",
            "HLA-B*40:01", "HLA-C*03:03", "HLA-C*03:04",
        ],
    )

    typer = HLATyper()
    alleles = typer.type_from_fastq("normal_R1.fastq.gz", "normal_R2.fastq.gz",
                                    sample_id="HCC1395BL")

    assert typer.typing_method == HLATyper.METHOD_OPTITYPE
    assert typer.is_patient_specific is True
    assert "HLA-A*11:01" in alleles


def test_missing_optitype_falls_back_and_says_so(monkeypatch):
    from cognisom.genomics import optitype_hla
    monkeypatch.setattr(optitype_hla, "is_optitype_available", lambda: False)

    typer = HLATyper()
    alleles = typer.type_from_fastq("r1.fastq.gz")

    assert typer.typing_method == HLATyper.METHOD_POPULATION_DEFAULT
    assert typer.is_patient_specific is False
    assert len(alleles) == 6


def test_a_failing_optitype_run_does_not_masquerade_as_typing(monkeypatch):
    """A crash must not leave the result looking patient-specific."""
    from cognisom.genomics import optitype_hla

    def boom(*a, **kw):
        raise RuntimeError("razers3 died")

    monkeypatch.setattr(optitype_hla, "is_optitype_available", lambda: True)
    monkeypatch.setattr(optitype_hla, "type_hla_from_fastq", boom)

    typer = HLATyper()
    typer.type_from_fastq("r1.fastq.gz")
    assert typer.is_patient_specific is False


def test_the_builder_accepts_reads_and_records_the_method(monkeypatch):
    from cognisom.genomics import optitype_hla
    from cognisom.genomics.patient_profile import PatientProfileBuilder
    from cognisom.genomics.synthetic_vcf import SYNTHETIC_PROSTATE_VCF
    from cognisom.genomics import gene_protein_mapper as gpm

    gpm.GeneProteinMapper._fetch_from_uniprot = lambda self, gene: None
    monkeypatch.setattr(optitype_hla, "is_optitype_available", lambda: True)
    monkeypatch.setattr(
        optitype_hla, "type_hla_from_fastq",
        lambda r1, r2=None, sample_id="s", **kw: [
            "HLA-A*02:01", "HLA-A*11:01", "HLA-B*15:01",
            "HLA-B*40:01", "HLA-C*03:03", "HLA-C*03:04",
        ],
    )

    profile = PatientProfileBuilder().from_vcf_text(
        SYNTHETIC_PROSTATE_VCF, "reads-test",
        normal_fastq=("n_R1.fastq.gz", "n_R2.fastq.gz"),
    )

    assert profile.hla_typing_method == HLATyper.METHOD_OPTITYPE
    assert profile.hla_is_patient_specific is True
    assert "HLA-A*11:01" in profile.hla_alleles
