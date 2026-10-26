"""The patient's genome reaches the simulation.

Item 3 of the realism plan. Until now every simulated cancer cell
carried ``KRAS_G12D`` on a generic three-allele HLA panel, whoever the
patient was, and ``EngineRunner`` had no way to receive a patient. The
genomics pipeline ran end to end on the real HCC1395 genome -- two
somatic coding drivers, ARID1A p.N218T and NOTCH1 p.L2065Q, typed
against that patient's measured HLA -- and stopped short of the engine.

Three things had to change, and each is pinned here:

* ``EngineRunner(patient=...)`` carries the patient's somatic missense
  mutations and HLA into the cellular and immune modules, via
  ``patient_engine.engine_inputs``, which checks every mutation against
  the reference protein and reports each one it leaves out.
* The engine's reference proteome covered three genes. It now falls
  back to the 19,293-gene MANE table, so ARID1A and NOTCH1 can be cut
  into peptides at all.
* The founding cancer cells now present the patient's mutations at
  build time. Before, none of them displayed anything until the
  molecular module happened to introduce a KRAS mutation mid-run.

Measured with MHCflurry 2.0.6 on HCC1395: all 80 founding cancer cells
display both mutations' peptides on HLA-A*29:02, B*45:01 and C*06:02;
SYYPTRSAY binds A*29:02 at 31 nM.
"""

from __future__ import annotations

import contextlib
import io
import os
from types import SimpleNamespace as N

import numpy as np

from cognisom.core.event_bus import EventTypes
import pytest

from cognisom.dashboard.engine_runner import EngineRunner
from cognisom.dashboard.patient_engine import engine_inputs, parse_missense
from cognisom.genomics.hla_typer import hla_is_read_derived

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOTATED = os.path.join(HERE, "cognisom", "genomics", "seqc2_demo.vep.vcf")
HCC_HLA = ["HLA-A*29:02", "HLA-A*29:02", "HLA-B*45:01",
           "HLA-B*45:01", "HLA-C*06:02", "HLA-C*06:02"]


def _variant(gene, change, consequence="missense"):
    return N(gene=gene, protein_change=change, consequence=consequence)


def _patient(variants, hla=HCC_HLA, method="optitype", drivers=()):
    return N(patient_id="TEST", hla_alleles=list(hla),
             hla_is_patient_specific=method != "population-default",
             hla_typing_method=method, coding_variants=list(variants),
             cancer_driver_mutations=list(drivers))


HCC_LIKE = _patient([_variant("ARID1A", "p.N218T"), _variant("NOTCH1", "p.L2065Q")])


def _build(**kwargs):
    np.random.seed(1)
    runner = EngineRunner(dt=0.05, duration=0.5, scenario="Baseline", **kwargs)
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        runner.build()
    return runner


def _founders(runner):
    cells = runner.engine.modules["cellular"].cells.values()
    return [c for c in cells if c.cell_type == "cancer"]


# ── parsing a protein change ──────────────────────────────────────


@pytest.mark.parametrize("change,expected", [
    ("p.N218T", ("N", 218, "T")),
    ("p.Asn218Thr", ("N", 218, "T")),
    ("p.(Leu2065Gln)", ("L", 2065, "Q")),
    ("G12D", ("G", 12, "D")),
    ("p.R130*", None),          # nonsense
    ("p.Arg130Ter", None),      # nonsense, three-letter
    ("p.E1143fs", None),        # frameshift
    ("p.S99S", None),           # synonymous
    ("", None),
    (None, None),
])
def test_only_single_residue_missense_changes_parse(change, expected):
    assert parse_missense(change) == expected


# ── what crosses over, and what is reported instead ───────────────


def test_every_missense_mutation_crosses_over_with_its_hla():
    inputs = engine_inputs(HCC_LIKE)
    assert [m.label for m in inputs.mutations] == ["ARID1A_N218T", "NOTCH1_L2065Q"]
    assert inputs.skipped == []
    assert inputs.hla_measured is True


def test_a_skipped_mutation_is_reported_with_its_reason():
    """The engine's own presentation step skips a reference mismatch
    silently. The mapping refuses to: every mutation left out is listed
    with why."""
    inputs = engine_inputs(_patient([
        _variant("AR", "p.T877A"),                       # historical numbering
        _variant("PTEN", "p.R130*", "nonsense"),
        _variant("BRCA2", "p.E1143fs", "frameshift"),
        _variant("NOTAGENE", "p.A1V"),
        _variant("TP53", "p.R248W"),                     # fine
    ]))
    assert [m.label for m in inputs.mutations] == ["TP53_R248W"]
    reasons = {gene: reason for gene, _change, reason in inputs.skipped}
    assert "reference has F at 877, not T" in reasons["AR"]
    assert "nonsense" in reasons["PTEN"]
    assert "frameshift" in reasons["BRCA2"]
    assert "not in the reference proteome" in reasons["NOTAGENE"]


def test_the_ar_hotspot_is_reported_not_silently_renumbered():
    """AR T877A is the historical name; the MANE threonine is at 878.
    Other code keys on "T877A", so renumbering it here would be its own
    kind of fabrication. It is reported instead."""
    inputs = engine_inputs(_patient([_variant("AR", "p.T877A")]))
    assert inputs.mutations == []
    assert inputs.skipped and inputs.skipped[0][0] == "AR"


def test_drivers_are_flagged_and_duplicates_collapse():
    drivers = [_variant("ARID1A", "p.N218T")]
    inputs = engine_inputs(_patient(
        [_variant("ARID1A", "p.N218T"), _variant("ARID1A", "p.Asn218Thr"),
         _variant("NOTCH1", "p.L2065Q")], drivers=drivers))
    assert [(m.label, m.oncogenic) for m in inputs.mutations] == [
        ("ARID1A_N218T", True), ("NOTCH1_L2065Q", False)]


def test_presentation_scores_each_allele_once_and_nk_gets_the_diploid_typing():
    configs = engine_inputs(HCC_LIKE).module_configs()
    assert configs["cellular"]["hla_alleles"] == [
        "HLA-A*29:02", "HLA-B*45:01", "HLA-C*06:02"]
    assert configs["immune"]["patient_hla"] == HCC_HLA
    assert configs["immune"]["patient_hla_measured"] is True


# ── measured means measured ───────────────────────────────────────


@pytest.mark.parametrize("method,measured", [
    ("optitype", True),
    ("predefined-profile", False),     # invented demo alleles
    ("vcf-hla-annotation", False),
    ("population-default", False),
])
def test_only_a_typing_from_the_patients_reads_counts_as_measured(method, measured):
    """hla_typer counts its predefined profiles as patient-specific,
    which is fair for a synthetic patient whose whole genome is
    synthetic. They are not MEASURED, and the NK licensing model requires
    types "actually determined from a patient's reads"."""
    patient = _patient([], method=method)
    assert engine_inputs(patient).hla_measured is measured
    assert hla_is_read_derived(patient) is measured


def test_the_twin_config_no_longer_hands_the_nk_model_an_invented_typing():
    """twin_config fed hla_is_patient_specific straight into the NK
    model's patient_hla_measured, so the synthetic demo patient's
    invented alleles reached the innate simulation marked as measured --
    the exact case that model was written to refuse."""
    import inspect
    from cognisom.genomics import twin_config

    source = inspect.getsource(twin_config)
    assert "hla_is_read_derived(profile)" in source
    assert "getattr(profile, 'hla_is_patient_specific', False))" not in source


# ── the engine can find the proteins ──────────────────────────────


def test_the_engine_finds_proteins_outside_its_three_gene_reference():
    from cognisom.genomics.gene_protein_mapper import BUILTIN_PROTEINS
    from cognisom.modules.molecular_module import MolecularModule

    module = MolecularModule({})
    with contextlib.redirect_stdout(io.StringIO()):
        module.initialize()
    arid1a = module.get_reference_protein("ARID1A")
    assert arid1a == BUILTIN_PROTEINS["ARID1A"].sequence
    assert arid1a[217] == "N", "ARID1A N218 is where HCC1395's mutation sits"
    assert module.get_reference_protein("NOTAGENE") == ""


def test_genes_the_engine_already_held_keep_their_own_translation():
    """KRAS, TP53 and BRAF come from the engine's DNA reference, which it
    mutates base by base. The fallback must not replace them: the
    engine's KRAS is 188 residues, the MANE table's 189."""
    from cognisom.modules.molecular_module import MolecularModule

    module = MolecularModule({})
    with contextlib.redirect_stdout(io.StringIO()):
        module.initialize()
    assert len(module.get_reference_protein("KRAS")) == 188


# ── the engine, with and without a patient ────────────────────────


def test_without_a_patient_nothing_changes():
    runner = _build()
    assert runner.patient_inputs is None
    assert runner.presentation_report is None
    founders = _founders(runner)
    assert founders and all(c.mutations == ["KRAS_G12D"] for c in founders)
    cellular = runner.engine.modules["cellular"]
    assert cellular.hla_alleles == ["HLA-A*02:01", "HLA-A*24:02", "HLA-B*07:02"]


def test_the_founding_tumour_carries_the_patients_mutations_and_alleles():
    runner = _build(patient=HCC_LIKE)
    founders = _founders(runner)
    assert all(c.mutations == ["ARID1A_N218T", "NOTCH1_L2065Q"] for c in founders)
    assert runner.engine.modules["cellular"].hla_alleles == [
        "HLA-A*29:02", "HLA-B*45:01", "HLA-C*06:02"]
    immune = runner.module_configs["immune"]
    assert immune["patient_hla"] == HCC_HLA and immune["patient_hla_measured"]


def test_an_explicit_override_still_beats_the_patient():
    runner = _build(patient=HCC_LIKE,
                    overrides={"cellular": {"hla_alleles": ["HLA-A*02:01"]}})
    assert runner.engine.modules["cellular"].hla_alleles == ["HLA-A*02:01"]


def test_presentation_runs_at_build_and_says_what_it_did():
    report = _build(patient=HCC_LIKE).presentation_report
    assert report["cancer_cells"] == 80
    assert report["mutations"] == ["ARID1A_N218T", "NOTCH1_L2065Q"]
    # Every mutation is either presented or recorded as unpresented with
    # the best binding it reached -- never simply absent.
    accounted = set(report["presented"]) | {u["mutation"] for u in report["unpresented"]}
    assert accounted == set(report["mutations"])


def test_presentation_without_reference_proteins_says_so():
    from cognisom.modules.cellular_module import CellularModule

    module = CellularModule({"patient_mutations": [
        {"gene": "ARID1A", "aa_change": "N218T", "label": "ARID1A_N218T",
         "oncogenic": True}]})
    report = module.present_patient_mutations()
    assert "no molecular module" in report["reason"]


def test_a_patient_with_nothing_presentable_gets_a_tumour_that_presents_nothing():
    """A patient given with no presentable mutations is not "no patient".
    Conflating the two gave this tumour the historical KRAS_G12D, which
    made a "nothing presentable" control quietly present KRAS -- and the
    T cells killed it."""
    silent = _patient([])
    runner = _build(patient=silent)
    founders = _founders(runner)
    assert founders and all(c.mutations == [] for c in founders)
    assert all(not c.mhc1_displayed_peptides for c in founders)
    assert runner.presentation_report["reason"] == "no patient mutations"


def test_a_cell_that_joins_a_patients_tumour_takes_that_patients_mutations():
    """A normal cell transformed mid-run joins THIS patient's tumour. It
    used to be labelled KRAS_G12D, and the molecular module wrote a KRAS
    G12D mutation into it, whoever the patient was."""
    runner = _build(patient=HCC_LIKE)
    cellular = runner.engine.modules["cellular"]
    normal = next(cid for cid, c in cellular.cells.items() if c.cell_type == "normal")
    with contextlib.redirect_stdout(io.StringIO()):
        cellular.transform_cell(normal)
    cell = cellular.cells[normal]
    assert cell.cell_type == "cancer"
    assert cell.mutations == ["ARID1A_N218T", "NOTCH1_L2065Q"]
    assert "KRAS_G12D" not in cell.mutations


def test_the_molecular_module_writes_no_default_kras_into_a_patients_tumour():
    runner = _build(patient=HCC_LIKE)
    molecular = runner.engine.modules["molecular"]
    assert molecular.default_transform_driver is None
    legacy = _build()
    assert legacy.engine.modules["molecular"].default_transform_driver == ("KRAS", "G12D")


def test_without_a_patient_a_transformed_cell_is_still_labelled_kras():
    runner = _build()
    cellular = runner.engine.modules["cellular"]
    normal = next(cid for cid, c in cellular.cells.items() if c.cell_type == "normal")
    with contextlib.redirect_stdout(io.StringIO()):
        cellular.transform_cell(normal)
    assert cellular.cells[normal].mutations == ["KRAS_G12D"]


# ── with the real predictor ───────────────────────────────────────


def _mhcflurry():
    from cognisom.genomics.mhcflurry_binding import is_mhcflurry_available
    if not is_mhcflurry_available():
        pytest.skip("MHCflurry unavailable here (it needs Python <= 3.12); "
                    "binder results are only meaningful from the real predictor")


def test_every_founding_cell_displays_the_patients_neoantigens():
    _mhcflurry()
    runner = _build(patient=HCC_LIKE)
    report = runner.presentation_report
    assert report["cells_displaying"] == report["cancer_cells"] == 80
    assert set(report["presented"]) == {"ARID1A_N218T", "NOTCH1_L2065Q"}
    peptides = {(p.peptide.sequence, p.hla_allele)
                for c in _founders(runner) for p in c.mhc1_displayed_peptides}
    assert ("SYYPTRSAY", "HLA-A*29:02") in peptides
    assert {a for _, a in peptides} <= {"HLA-A*29:02", "HLA-B*45:01", "HLA-C*06:02"}


@pytest.mark.slow
@pytest.mark.timeout(400)
def test_the_real_hcc1395_genome_reaches_the_cancer_cells():
    _mhcflurry()
    if not os.path.exists(ANNOTATED):
        pytest.skip("annotated demo callset not generated")
    from cognisom.genomics.patient_profile import PatientProfileBuilder

    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        profile = PatientProfileBuilder(cancer_type="breast").from_vcf_file(
            ANNOTATED, patient_id="SEQC2")
    runner = _build(patient=profile)
    assert runner.patient_inputs.hla_measured
    assert runner.presentation_report["cells_displaying"] == 80
    assert runner.engine.modules["immune"].get_nk_licensing_profile() is not None


@pytest.mark.slow
@pytest.mark.timeout(500)
def test_the_genome_decides_whether_t_cells_can_kill():
    """The plan's bar for item 3: different genomes, identical settings,
    different T-cell outcomes. With NK cells and macrophages removed so
    innate killing cannot mask it, measured over 6 h on three seeds:

        HCC1395, ARID1A and NOTCH1 presented   T-cell kills 4, 3, 5
        same patient, nothing presentable       T-cell kills 0, 0, 0

    (9, 10, 13 before kill calibration on 2026-10-26: a T-cell contact now
    adds decaying perforin damage instead of killing on one draw. Counted
    by killer: a run with no NK cells configured used to recruit them.)

    A tumour presenting the patient's neoantigens is killed; one
    presenting nothing is invisible to T cells.
    """
    _mhcflurry()

    def t_cell_kills(patient):
        np.random.seed(1)
        runner = EngineRunner(
            dt=0.05, duration=6.0, scenario="Baseline", patient=patient,
            overrides={"immune": {"n_nk_cells": 0, "n_macrophages": 0}},
            snapshot_interval_hours=1.0)
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            runner.build()
            runner.run()
        return sum(1 for e in runner.event_log
                   if e["event"] == EventTypes.CANCER_KILLED
                   and e["data"].get("killer_type") == "T_cell")

    assert t_cell_kills(_patient([])) == 0
    assert t_cell_kills(HCC_LIKE) > 0
