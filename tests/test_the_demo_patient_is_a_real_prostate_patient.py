"""The demo runs a real prostate patient, and says what is wrong with him.

The demo ran on HCC1395 -- a BREAST cancer cell line -- while the tissue
geometry, the AR pathway and most of the drug library are prostate. The
mismatch was the honest thing to fix, and fixing it needed a real
prostate callset.

``SC_9184_T`` from the SU2C/PCF cohort (Abida 2019, PMID 31061129) was
chosen for one reason the simulator can act on: **AR p.F877L**, the
substitution that switches enzalutamide and apalutamide from antagonists
to agonists (Joseph 2013 PMID 23779130; Balbas 2013 PMID 23580326), in a
man whose record reads ``ABI_ENZA_EXPOSURE_STATUS = Exposed``. Genome and
treatment history both bear on a mechanism the engine implements rather
than merely describes.

Three things these tests exist to hold down:

1. **The callset is committed, not read from ``data/``.** ``data/`` is
   gitignored, so a demo that read the 64,325-row cohort MAF directly
   would work on the machine that downloaded it and nowhere else -- not
   in CI, not on the server. That is this codebase's most common defect
   shape, so the test asserts the VCF is present and parses without the
   cohort.

2. **The disclosures travel with the patient.** The biopsy is a LIVER
   METASTASIS laid out on a measured prostate section. Nothing in the
   callset says so, so it is declared where the cohort is declared and
   has to reach the page.

3. **The attribution travels with the data.** The cohort is ODbL, which
   is share-alike and requires attribution -- the second non-Apache
   asset in this repository.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
VCF = REPO / "cognisom" / "genomics" / "su2c_prostate_demo.vcf"


@pytest.fixture(scope="module")
def vcf_text() -> str:
    return VCF.read_text()


# ── the callset is in the repository ──────────────────────────────


def test_the_callset_is_committed_not_read_from_the_gitignored_data_dir():
    assert VCF.exists(), (
        f"{VCF} is missing. It must be COMMITTED: data/ is gitignored, so "
        "a demo that reads the cohort MAF works only on the machine that "
        "downloaded it. Regenerate with scripts/su2c_to_vcf.py.")
    import subprocess
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(VCF.relative_to(REPO))],
        cwd=REPO, capture_output=True, text=True)
    assert tracked.returncode == 0, (
        "the demo VCF is not tracked by git, so CI and the server will "
        "not have it")


def test_the_cohort_maf_itself_is_not_redistributed():
    """ODbL is share-alike; we extract one patient, not the cohort."""
    import subprocess
    listed = subprocess.run(
        ["git", "ls-files", "data/cbioportal"],
        cwd=REPO, capture_output=True, text=True)
    assert listed.stdout.strip() == "", (
        "the SU2C cohort download must stay out of the repository; only "
        "the single-patient extract is committed")


# ── the header carries its own provenance ─────────────────────────


def test_the_header_attributes_the_source_and_its_licence(vcf_text):
    for expected in ("Abida", "31061129", "prad_su2c_2019",
                     "Open Data Commons", "ODbL"):
        assert expected in vcf_text, f"{expected!r} missing from the header"


def test_the_repository_notice_carries_the_odbl_attribution():
    notice = (REPO / "NOTICE").read_text()
    assert "prad_su2c_2019" in notice and "ODbL" in notice, (
        "ODbL requires attribution on public use of the database or works "
        "produced from it; NOTICE is where this repository records that")
    assert "31061129" in notice


def test_the_header_says_the_normal_genotype_is_not_our_evidence(vcf_text):
    """The one thing a reader most needs to know about this file.

    The source publishes ``Mutation_Status = Somatic`` against a matched
    normal but publishes NO normal read counts. The VCF's ``0/0``
    therefore asserts the consortium's finding; it is not read evidence
    held here. Without that genotype the parser would classify every
    call ``UNKNOWN: no_matched_normal``, so the assertion is load-bearing
    and has to be disclosed rather than quietly relied on.
    """
    assert "cognisom_normal_evidence" in vcf_text
    note = next(line for line in vcf_text.splitlines()
                if line.startswith("##cognisom_normal_evidence="))
    for expected in ("NO read depth", "ASSERTS", "not read evidence"):
        assert expected in note, (expected, note)


def test_the_normal_column_really_carries_no_depth(vcf_text):
    """What the header promises, the records must do."""
    for line in vcf_text.splitlines():
        if line.startswith("#"):
            continue
        normal = line.split("\t")[9]
        assert normal.startswith("0/0"), normal
        assert not re.search(r"0/0:\d", normal), (
            f"a normal call carries real depth: {normal!r}. The source "
            "publishes none, so this would be invented data.")


def test_the_indel_placeholder_alleles_are_disclosed(vcf_text):
    """MAF dash alleles have no VCF anchor base and we ship no genome."""
    assert "cognisom_indel_alleles" in vcf_text
    note = next(line for line in vcf_text.splitlines()
                if line.startswith("##cognisom_indel_alleles="))
    assert "placeholder N" in note and "NOT a usable genomic allele" in note


# ── it parses, and it parses as somatic ───────────────────────────


@pytest.fixture(scope="module")
def parsed():
    from cognisom.genomics.vcf_parser import VCFParser
    parser = VCFParser()
    return parser, parser.parse_file(str(VCF))


def test_the_callset_parses_as_a_matched_somatic_vcf(parsed):
    parser, variants = parsed
    assert parser.is_somatic_vcf, (
        "without both sample headers the parser cannot do the "
        "tumour/normal comparison at all")
    assert parser.tumor_sample == "SC_9184_T"
    assert parser.normal_sample == "SC_9184_N"
    assert len(variants) == 93, len(variants)
    assert all(v.somatic_status == "SOMATIC" for v in variants), (
        "the consortium called these somatic against a matched normal; "
        "if this fails the normal genotype stopped reaching the "
        "classifier and every call silently became UNKNOWN")


def test_the_resistance_mutation_is_present_with_real_read_support(parsed):
    _parser, variants = parsed
    ar = [v for v in variants if v.gene == "AR"]
    assert len(ar) == 1, [(v.gene, v.protein_change) for v in ar]
    assert ar[0].protein_change == "p.F877L"
    # 190 alt of 817 reads. Pinned beside the claim that it is real
    # read support and not a marginal call; re-measure from the MAF, do
    # not re-pin.
    assert ar[0].tumor_alt_count == 190
    assert ar[0].tumor_af == pytest.approx(0.2326, abs=0.001)
    assert ar[0].somatic_status == "SOMATIC"


# ── the cohort has one owner ──────────────────────────────────────


def test_the_default_demo_patient_is_the_prostate_one():
    from cognisom.genomics.demo_patients import (
        DEFAULT_DEMO_PATIENT, demo_patient)
    assert DEFAULT_DEMO_PATIENT == "su2c_prostate"
    assert demo_patient(DEFAULT_DEMO_PATIENT).cancer_type == "prostate", (
        "every other part of this platform is prostate; a breast cell "
        "line was the odd one out")


def test_an_unknown_demo_patient_fails_loudly():
    from cognisom.genomics.demo_patients import demo_patient
    with pytest.raises(KeyError) as excinfo:
        demo_patient("nobody")
    assert "su2c_prostate" in str(excinfo.value), (
        "the error must name the keys that DO exist")


def test_every_declared_demo_patient_has_a_committed_callset():
    from cognisom.genomics.demo_patients import DEMO_PATIENTS
    for key, patient in DEMO_PATIENTS.items():
        assert patient.vcf_path.exists(), f"{key}: {patient.vcf_path}"
        assert patient.caveats, (
            f"{key} declares no caveats. An empty disclosure list is a "
            "claim that nothing weakens this subject, which is never "
            "true of a demo callset.")


def test_the_recording_script_goes_through_the_cohort_module():
    """The test that would have caught the HLA key mismatch.

    Everything else here can pass while the recording script names its
    own VCF and its own patient id -- which is exactly how the demo came
    to ask for "HCC1395" while the measured typing was registered under
    "SEQC2". The id the script builds under must be the id the registry
    is consulted with, and that holds only if one module owns both.
    """
    source = (REPO / "scripts" / "record_demo_run.py").read_text()
    assert "demo_patients" in source, (
        "record_demo_run.py must load its subject through "
        "genomics.demo_patients, not by naming a VCF path itself")
    assert "seqc2_demo.vep.vcf" not in source, (
        "the script still hard-codes a callset path")
    assert "DEFAULT_DEMO_PATIENT" in source


# ── the disclosures reach the page ────────────────────────────────


@pytest.fixture(scope="module")
def subject():
    os.environ.setdefault("COGNISOM_ALLOW_PWM_FALLBACK", "1")
    from cognisom.dashboard.sim_request import describe_subject
    from cognisom.genomics.demo_patients import load_demo_patient
    with contextlib.redirect_stdout(io.StringIO()):
        profile = load_demo_patient("su2c_prostate")
    return profile, describe_subject(profile)


@pytest.mark.slow
def test_the_liver_biopsy_is_disclosed(subject):
    """The caveat a reader is most likely to assume away.

    The genome is this patient's; the tissue architecture is not. No
    choice of layout fixes it -- there is no measured liver-metastasis
    geometry in this repository -- so the only honest option is to say
    so wherever this run is shown.
    """
    _profile, subj = subject
    blob = " ".join(subj.caveats).upper()
    assert "LIVER" in blob and "METASTASIS" in blob
    assert "PROSTATE SECTION" in blob


@pytest.mark.slow
def test_the_prior_treatment_is_disclosed(subject):
    """Any drug the simulator applies is added to a history it ignores."""
    _profile, subj = subject
    blob = " ".join(subj.caveats).lower()
    assert "enzalutamide" in blob and "taxane" in blob


@pytest.mark.slow
def test_the_unmeasured_hla_is_still_disclosed(subject):
    """SU2C publishes no HLA typing, and that must not be glossed.

    This is the caveat the breast demo escaped by having a real OptiType
    typing. Trading it away for a real prostate genome is a deliberate
    choice, so it has to be visible.
    """
    _profile, subj = subject
    assert subj.hla_measured is False
    assert any("population-default" in c for c in subj.caveats)


@pytest.mark.slow
def test_a_profile_built_without_the_cohort_module_carries_no_disclosures():
    """The disclosures are not magic: they come from the declaration.

    A profile built straight from the VCF has an empty
    ``provenance_caveats``, and that is correct -- nothing then knows
    the biopsy site. It is also why the recording script must go through
    the cohort module, which the test above asserts.
    """
    os.environ.setdefault("COGNISOM_ALLOW_PWM_FALLBACK", "1")
    from cognisom.dashboard.sim_request import describe_subject
    from cognisom.genomics.patient_profile import PatientProfileBuilder
    with contextlib.redirect_stdout(io.StringIO()):
        bare = PatientProfileBuilder(cancer_type="prostate").from_vcf_file(
            str(VCF), patient_id="SC_9184_T")
    assert bare.provenance_caveats == []
    blob = " ".join(describe_subject(bare).caveats).upper()
    assert "LIVER" not in blob


# ── and the resistance mechanism actually fires ───────────────────


@pytest.mark.slow
def test_this_patients_own_variants_make_enzalutamide_an_agonist(subject):
    """The reason this patient was chosen, end to end.

    The variant labels come from the real profile, not from a literal,
    so this fails if the callset stops carrying AR F877L or if the
    label format drifts from what the agonist table matches on.
    """
    profile, _subj = subject
    from cognisom.workflow import parameter_bridge as pb
    from cognisom.workflow.scenario_builder import variant_labels_from_profile

    labels = variant_labels_from_profile(profile)
    assert "AR:F877L" in labels

    assert pb.anti_androgen_is_agonist("Enzalutamide", labels) == "F877L"
    assert pb.anti_androgen_is_agonist("Apalutamide", labels) == "F877L"
    # The actionable half: darolutamide is a full antagonist on F876L
    # (Moilanen 2015, PMID 26137992), so this patient is not out of
    # options and the model must not say he is.
    assert pb.anti_androgen_is_agonist("Darolutamide", labels) is None


@pytest.mark.slow
def test_the_bridge_turns_this_patients_enzalutamide_into_activation(subject):
    profile, _subj = subject
    import contextlib as _c
    from cognisom.library.models import SimulationScenario
    from cognisom.library.store import EntityStore
    from cognisom.workflow import parameter_bridge as pb
    from cognisom.workflow.scenario_builder import variant_labels_from_profile

    with _c.redirect_stdout(io.StringIO()):
        store = EntityStore()
    drug = store.find_entity_by_name("Enzalutamide", "drug")
    assert drug is not None
    scenario = SimulationScenario(
        name="t", display_name="t", scenario_type="drug_response",
        drug_ids=[drug.entity_id],
        tags=["drug_response"]
        + [f"variant:{label}" for label in variant_labels_from_profile(profile)])

    bridge = pb.ParameterBridge(store)
    resolved = bridge.resolve_scenario(scenario, include_defaults=False)
    assert (resolved["ar_signaling"]["ar_transcriptional_activity"]
            == pb.AR_AGONIST_TRANSCRIPTIONAL_ACTIVITY)
    assert "ar_transcriptional_block" not in resolved["ar_signaling"]
    assert any("AGONIST" in note for note in bridge.last_notes)
