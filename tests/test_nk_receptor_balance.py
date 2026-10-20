"""
NK receptor balance, KIR-HLA licensing, and the macrophage PD-1 gate.

The innate arm had NO test coverage of any kind. Instrumenting the three
branches of ``ImmuneModule._target_kill_probability`` and running the
whole suite recorded 29 calls, all 29 on the T-cell Hill path and zero
on the NK/macrophage branch: every ImmuneModule config in tests/ sets
``n_nk_cells: 0`` and ``n_macrophages: 0``. This module is the first
regression net for that half of the immune system.

Three things are asserted here, in order of how much they matter:

1. The KIR ligand-group table is right. Each entry carries residues
   76-83 read out of its IPD-IMGT/HLA sequence record plus that record's
   own ``matching.kir_ligand`` assignment, and the tests check that the
   residue rule REPRODUCES the database's assignment for every HLA-B and
   HLA-C entry. A hand-copied serotype table could not be checked this
   way; this one can.
2. The mechanism behaves like the licensing literature says. Licensed
   subsets lose inhibition as the target loses MHC-I; unlicensed subsets
   are weaker but MHC-independent; broader ligand coverage licenses more
   of the compartment. These are the properties, not the magnitudes --
   the declared parameters are tested for ordering only, which is all
   the cited sources support.
3. The flagship T-cell closed loop is untouched. The last test
   re-runs the neoantigen trace with the NK receptor model configured
   and asserts the same 1/1/2/1/1/1 event counts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import logging

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.core import SimulationEngine, SimulationConfig
from cognisom.core.event_bus import EventTypes
from cognisom.engine.py.immune import macrophage_phagocytosis as mphage
from cognisom.engine.py.immune.nk_receptor import (
    HLA_LIGAND_TABLE,
    IPD_IMGT_HLA_RELEASE,
    KIR,
    KIR_AVIDITY,
    KIR_COGNATE_LIGANDS,
    LigandGroup,
    POPULATION_PRIOR_KIR_SOURCE,
    UnknownAlleleError,
    activating_signal,
    draw_kir_subset,
    genotyped_kir,
    hla_ligand_profile,
    licensing_profile,
    ligand_group,
    ligand_group_from_motif,
    missing_self_recognized,
    nk_clone_decision,
    nk_decision,
    normalize_allele,
    population_prior_kir,
    subset_signal,
)
from cognisom.genomics.hla_typer import MEASURED_HLA_TYPINGS
from cognisom.modules.cellular_module import CellularModule
from cognisom.modules.immune_module import ImmuneModule
from cognisom.modules.molecular_module import MolecularModule


SEQC2_ALLELES = list(MEASURED_HLA_TYPINGS["SEQC2"].alleles)

#: A genotype carrying all three KIR ligand groups, so every modelled
#: inhibitory receptor is licensed. C*06:02 is C2, C*07:01 is C1,
#: B*51:01 is Bw4-80I (all three verified against IPD-IMGT/HLA).
BROADLY_LICENSED = [
    "HLA-A*02:01", "HLA-A*24:02",
    "HLA-B*51:01", "HLA-B*07:02",
    "HLA-C*06:02", "HLA-C*07:01",
]


def _profile(alleles, **kw):
    return licensing_profile(
        hla_ligand_profile(alleles, is_patient_specific=True,
                           source="test", **kw)
    )


def _subset(profile, receptor):
    for s in profile.subsets:
        if s.receptor is receptor:
            return s
    raise AssertionError(f"{receptor} not in compartment")


# ── 1. The ligand table and the residue rule ─────────────────────────

def test_the_residue_rule_reproduces_ipds_own_kir_ligand_assignment():
    # Every HLA-B and HLA-C record in the table carries IPD-IMGT/HLA's
    # own matching.kir_ligand string. Deriving the group from residues
    # 80/83 of the same record's mature protein must agree with it on
    # every single entry -- otherwise either the motif or the rule is
    # wrong, and a patient's licensing would be decided by the error.
    checked = 0
    for allele, record in HLA_LIGAND_TABLE.items():
        if record.ipd_kir_ligand is None:
            continue
        expected = record.ipd_kir_ligand.replace(" ", "")
        assert ligand_group(allele).value == expected, (
            f"{allele} ({record.ipd_accession}): motif "
            f"{record.residues_76_83} derives "
            f"{ligand_group(allele).value}, IPD {IPD_IMGT_HLA_RELEASE} "
            f"says {record.ipd_kir_ligand!r}"
        )
        checked += 1
    # 31 of the 43 entries are B or C; HLA-A gets no IPD assignment.
    assert checked == 31, f"expected 31 IPD-assigned entries, got {checked}"


def test_hla_a_bw4_positive_set_matches_the_published_allotypes():
    # IPD assigns no kir_ligand to HLA-A, so the A-locus Bw4 call rests
    # on the residue rule alone. The allotypes it must pick out are the
    # four long-established HLA-A KIR3DL1 ligands: A*23, A*24, A*25,
    # A*32 (Gumperz et al., J Exp Med 1995; Thananchai et al.,
    # J Immunol 2007). A*29:02 -- the measured SEQC2 allele -- is not
    # among them, which is what makes SEQC2's Bw4 status checkable.
    bw4 = {
        a for a in HLA_LIGAND_TABLE
        if a.startswith("A*")
        and ligand_group(a) in (LigandGroup.BW4_80I, LigandGroup.BW4_80T)
    }
    assert bw4 == {"A*23:01", "A*24:02", "A*25:01", "A*32:01"}
    assert ligand_group("A*29:02") is LigandGroup.NO_KIR_LIGAND


def test_hla_c_is_split_by_residue_80_only():
    # Asn80 -> C1, Lys80 -> C2 (Winter & Long, J Immunol 1997).
    assert ligand_group_from_motif("C", "VSLRNLRG") is LigandGroup.C1
    assert ligand_group_from_motif("C", "VNLRKLRG") is LigandGroup.C2
    # Residue 83 is irrelevant at the C locus: flip it and the group
    # must not move.
    assert ligand_group_from_motif("C", "VSLRNLRR") is LigandGroup.C1


def test_bw4_is_decided_by_arg83_and_split_by_residue_80():
    assert ligand_group_from_motif("B", "ENLRIALR") is LigandGroup.BW4_80I
    assert ligand_group_from_motif("B", "ENLRTALR") is LigandGroup.BW4_80T
    assert ligand_group_from_motif("B", "ESLRNLRG") is LigandGroup.BW6
    # B*27:05's 77-83 is not the ENLR.ALR consensus, but Arg83 + Thr80
    # still make it Bw4-80T -- and IPD agrees.
    assert ligand_group_from_motif("B", "EDLRTLLR") is LigandGroup.BW4_80T
    assert HLA_LIGAND_TABLE["B*27:05"].ipd_kir_ligand == "Bw4 - 80T"


def test_an_allele_off_the_record_refuses_instead_of_defaulting():
    # A guessed ligand group silently decides which NK subsets are
    # licensed, so it must not be guessable.
    with pytest.raises(UnknownAlleleError):
        ligand_group("C*99:99")
    with pytest.raises(UnknownAlleleError):
        ligand_group("B*99:01")
    # One unknown allele in a genotype refuses the whole profile rather
    # than dropping that allele -- dropping it would remove a ligand
    # group and so unlicense a subset, invisibly.
    with pytest.raises(UnknownAlleleError):
        hla_ligand_profile(SEQC2_ALLELES + ["HLA-C*99:99"])
    with pytest.raises(UnknownAlleleError):
        hla_ligand_profile([])


def test_allele_names_normalize_to_two_fields():
    for spelling in ("HLA-C*06:02", "C*06:02", "hla-c*06:02:01:01",
                     "C*06:02:01"):
        assert normalize_allele(spelling) == "C*06:02"
    with pytest.raises(UnknownAlleleError):
        normalize_allele("C06:02")
    with pytest.raises(UnknownAlleleError):
        normalize_allele(None)


def test_bw6_is_not_a_kir_ligand_anywhere_in_the_avidity_table():
    for receptor, row in KIR_AVIDITY.items():
        assert LigandGroup.BW6 not in row
        assert LigandGroup.NO_KIR_LIGAND not in row
    # Bw6 and the A-locus Bw4-negatives contribute no ligand group.
    prof = hla_ligand_profile(SEQC2_ALLELES)
    assert prof.groups == (LigandGroup.C2,)


def test_avidity_and_licensing_orderings_are_the_cited_claim():
    # The magnitudes in KIR_AVIDITY are declared parameters; these
    # orderings are what the literature actually supports, so they are
    # what the suite pins.
    a = KIR_AVIDITY
    # KIR2DL2 is a stronger C1 receptor than KIR2DL3 (Moesta 2008).
    assert a[KIR.KIR2DL2][LigandGroup.C1] > a[KIR.KIR2DL3][LigandGroup.C1]
    # KIR2DL1-C2 is the strongest HLA-C pairing.
    assert a[KIR.KIR2DL1][LigandGroup.C2] >= a[KIR.KIR2DL2][LigandGroup.C1]
    # KIR2DL2 cross-reacts onto C2, but weakly, and KIR2DL1 does not
    # cross-react onto C1 at all.
    assert 0 < a[KIR.KIR2DL2][LigandGroup.C2] < a[KIR.KIR2DL2][LigandGroup.C1]
    assert LigandGroup.C1 not in a[KIR.KIR2DL1]
    # KIR3DL1 prefers Bw4-80I over Bw4-80T (Carr 2005).
    assert a[KIR.KIR3DL1][LigandGroup.BW4_80I] > a[KIR.KIR3DL1][LigandGroup.BW4_80T]
    # Cross-reactivity does not educate: only the cognate group does.
    assert KIR_COGNATE_LIGANDS[KIR.KIR2DL2] == (LigandGroup.C1,)
    assert LigandGroup.C2 in KIR_AVIDITY[KIR.KIR2DL2]


# ── 2. SEQC2: where a real measurement reaches the simulation ────────

def test_the_measured_seqc2_typing_is_homozygous_c2_with_no_bw4_ligand():
    # SEQC2 = HCC1395BL matched normal, typed with OptiType 1.3.5 from
    # SRR7890874 on 1198 reads: A*29:02/A*29:02, B*45:01/B*45:01,
    # C*06:02/C*06:02. Every allele is homozygous, so this patient
    # carries exactly one KIR ligand group.
    prof = hla_ligand_profile(SEQC2_ALLELES, is_patient_specific=True,
                              source="measured:SEQC2")
    assert prof.has_c2 and not prof.has_c1 and not prof.has_bw4
    assert prof.groups == (LigandGroup.C2,)
    assert dict(prof.per_allele) == {
        "A*29:02": LigandGroup.NO_KIR_LIGAND,
        "B*45:01": LigandGroup.BW6,
        "C*06:02": LigandGroup.C2,
    }
    assert prof.is_patient_specific


def test_seqc2_licenses_kir2dl1_and_nothing_else():
    prof = _profile(SEQC2_ALLELES)
    assert _subset(prof, KIR.KIR2DL1).licensed is True
    for receptor in (KIR.KIR2DL2, KIR.KIR2DL3, KIR.KIR3DL1):
        assert _subset(prof, receptor).licensed is False, receptor
    # KIR2DL1's share of the compartment under the carriage prior:
    # 0.95 / (0.95 + 0.53 + 0.88 + 0.96).
    assert prof.licensed_fraction == pytest.approx(0.95 / 3.32, abs=1e-6)
    # Unlicensed KIR2DL2 still sees C2 by cross-reactivity, so it is
    # partially inhibited despite never having been educated.
    assert _subset(prof, KIR.KIR2DL2).inhibitory_avidity > 0
    # KIR2DL3 and KIR3DL1 have no ligand at all on this patient.
    assert _subset(prof, KIR.KIR2DL3).inhibitory_avidity == 0.0
    assert _subset(prof, KIR.KIR3DL1).inhibitory_avidity == 0.0


def test_broader_ligand_coverage_licenses_more_of_the_compartment():
    seqc2 = _profile(SEQC2_ALLELES)
    broad = _profile(BROADLY_LICENSED)
    assert broad.licensed_fraction == pytest.approx(1.0)
    assert broad.licensed_fraction > seqc2.licensed_fraction
    assert all(s.licensed for s in broad.subsets)


# ── 3. The mechanism ─────────────────────────────────────────────────

def test_a_licensed_compartment_loses_inhibition_as_the_target_loses_mhc1():
    # Missing self (Karre et al., Nature 1986): less surface MHC-I ->
    # less KIR inhibition -> more killing. The legacy rule was a step
    # with identical output at MHC-I 0.00, 0.20 and 0.39 and none at
    # 0.40; this must be graded.
    prof = _profile(SEQC2_ALLELES)
    ps = [
        nk_decision(prof, mhc1_expression=m, max_kill_probability=0.8)
        .kill_probability
        for m in (0.0, 0.1, 0.2, 0.3, 0.4)
    ]
    assert ps == sorted(ps, reverse=True), ps
    assert ps[0] > ps[-1], "no missing-self gradient at all"
    assert len(set(round(p, 6) for p in ps)) == len(ps), (
        f"still a step function: {ps}"
    )


def test_a_fully_licensed_compartment_is_self_tolerant_at_normal_mhc1():
    # Every subset educated and engaged -> inhibition dominates and the
    # autologous target at full MHC-I is spared outright.
    broad = _profile(BROADLY_LICENSED)
    at_full = nk_decision(broad, mhc1_expression=1.0,
                          max_kill_probability=0.8)
    assert at_full.kill_probability == pytest.approx(0.0)
    assert at_full.recognized is False
    at_none = nk_decision(broad, mhc1_expression=0.0,
                          max_kill_probability=0.8)
    assert at_none.kill_probability > 0.0
    assert at_none.recognized is True
    # And it is a stronger missing-self responder than the narrowly
    # licensed SEQC2 compartment, because more of it is educated.
    seqc2_none = nk_decision(_profile(SEQC2_ALLELES), mhc1_expression=0.0,
                             max_kill_probability=0.8)
    assert at_none.kill_probability > seqc2_none.kill_probability


def test_unlicensed_subsets_are_hyporesponsive_but_mhc_independent():
    # KIR3DL1 on a Bw4-negative patient: no cognate ligand, so never
    # educated -- and nothing for MHC-I to inhibit it through. It should
    # kill weakly and identically at every MHC-I level, which is the
    # licensing model's signature prediction.
    prof = licensing_profile(
        hla_ligand_profile(SEQC2_ALLELES, is_patient_specific=True,
                           source="test"),
        genotyped_kir([KIR.KIR3DL1]),
    )
    only = prof.subsets[0]
    assert only.receptor is KIR.KIR3DL1
    assert only.licensed is False and prof.licensed_fraction == 0.0
    assert 0.0 < only.competence < 1.0
    ps = {
        nk_decision(prof, mhc1_expression=m, max_kill_probability=0.8)
        .kill_probability
        for m in (0.0, 0.25, 0.5, 0.75, 1.0)
    }
    assert len(ps) == 1, f"unlicensed subset is MHC-sensitive: {ps}"
    # ... and weaker than the same receptor would be if licensed.
    licensed = licensing_profile(
        hla_ligand_profile(BROADLY_LICENSED, is_patient_specific=True,
                           source="test"),
        genotyped_kir([KIR.KIR3DL1]),
    )
    assert _subset(licensed, KIR.KIR3DL1).licensed is True
    assert (nk_decision(licensed, mhc1_expression=0.0,
                        max_kill_probability=0.8).kill_probability
            > ps.pop())


def test_activating_signal_is_monotone_in_mutation_load_and_saturates():
    values = [activating_signal(n) for n in (0, 1, 5, 20, 100, 10_000)]
    assert values == sorted(values)
    assert values[0] < values[-1]
    assert values[-1] <= 1.0
    # Half-max at the declared half_max_mutations.
    assert activating_signal(5) == pytest.approx(0.75)
    assert activating_signal(0) == pytest.approx(0.5)
    # Mutation load raises killing through the activating arm.
    prof = _profile(SEQC2_ALLELES)
    low = nk_decision(prof, mhc1_expression=0.5, n_mutations=0,
                      max_kill_probability=0.8).kill_probability
    high = nk_decision(prof, mhc1_expression=0.5, n_mutations=50,
                       max_kill_probability=0.8).kill_probability
    assert high > low


def test_the_kir_prior_never_passes_for_a_genotype():
    prior = population_prior_kir()
    assert prior.is_patient_specific is False
    assert "Bontadini" in prior.source and "prior" in prior.source
    assert prior.source == POPULATION_PRIOR_KIR_SOURCE
    assert prior.weight(KIR.KIR2DL1) == 0.95
    assert prior.weight(KIR.KIR2DL2) == 0.53
    typed = genotyped_kir([KIR.KIR2DL1, KIR.KIR3DL1])
    assert typed.is_patient_specific is True
    assert typed.weight(KIR.KIR2DL1) == 1.0
    assert typed.weight(KIR.KIR2DL3) == 0.0
    # The HLA side being measured does not make the KIR side measured.
    prof = _profile(SEQC2_ALLELES)
    assert prof.is_patient_specific is True
    assert prof.kir.is_patient_specific is False


def test_an_empty_kir_compartment_refuses():
    with pytest.raises(ValueError):
        licensing_profile(
            hla_ligand_profile(SEQC2_ALLELES, is_patient_specific=True,
                               source="test"),
            genotyped_kir([]),
        )


def test_the_licensing_profile_carries_its_own_provenance():
    d = _profile(SEQC2_ALLELES).as_dict()
    assert d["ligand_groups"] == ["C2"]
    assert d["hla_is_patient_specific"] is True
    assert d["kir_is_patient_specific"] is False
    assert d["ipd_release"] == IPD_IMGT_HLA_RELEASE
    assert len(d["subsets"]) == 4


# ── 3b. The per-clone decision, which is what the engine runs ───────

class _FixedRNG:
    """An rng whose .random() returns a scripted sequence."""

    def __init__(self, *values):
        self.values = list(values)
        self.i = 0

    def random(self):
        v = self.values[self.i % len(self.values)]
        self.i += 1
        return v


def test_the_clone_draw_walks_the_compartment_weights():
    # Cumulative shares under the carriage prior (total 3.32):
    # 2DL1 to 0.28614, 2DL2 to 0.44578, 2DL3 to 0.71084, 3DL1 to 1.0.
    prof = _profile(SEQC2_ALLELES)
    cases = [
        (0.0, KIR.KIR2DL1), (0.2860, KIR.KIR2DL1),
        (0.2862, KIR.KIR2DL2), (0.44, KIR.KIR2DL2),
        (0.50, KIR.KIR2DL3), (0.70, KIR.KIR2DL3),
        (0.72, KIR.KIR3DL1), (0.999999, KIR.KIR3DL1),
    ]
    for u, expected in cases:
        assert draw_kir_subset(prof, _FixedRNG(u)).receptor is expected, u
    # And the draw reproduces the weights in aggregate.
    rng = np.random.default_rng(0)
    counts = {}
    for _ in range(20_000):
        r = draw_kir_subset(prof, rng).receptor
        counts[r] = counts.get(r, 0) + 1
    for subset in prof.subsets:
        assert abs(counts[subset.receptor] / 20_000 - subset.weight) < 0.02


def test_a_clone_decision_is_the_single_subset_arithmetic():
    prof = _profile(SEQC2_ALLELES)
    licensed = _subset(prof, KIR.KIR2DL1)
    d = nk_clone_decision(licensed, mhc1_expression=0.2, n_mutations=0,
                          max_kill_probability=0.8)
    # activating 0.5, competence 1.0, avidity 1.0 -> signal 0.5 - 0.2.
    assert d.activating == pytest.approx(0.5)
    assert subset_signal(licensed, 0.5, 0.2) == pytest.approx(0.3)
    assert d.kill_probability == pytest.approx(0.8 * 0.3)
    assert d.per_subset == (("KIR2DL1", pytest.approx(0.3),
                             pytest.approx(0.24)),)


def test_a_licensed_clone_will_not_engage_a_target_with_intact_mhc1():
    # Self-tolerance is a refusal to ENGAGE, not merely weak killing --
    # which is why the engagement decision has to read the integrated
    # signal. Averaging the compartment inside the kill roll instead
    # left every NK cell engaging every target.
    prof = _profile(SEQC2_ALLELES)
    licensed = _subset(prof, KIR.KIR2DL1)
    intact = nk_clone_decision(licensed, mhc1_expression=1.0,
                               max_kill_probability=0.8)
    assert intact.recognized is False
    assert intact.kill_probability == 0.0
    lost = nk_clone_decision(licensed, mhc1_expression=0.0,
                             max_kill_probability=0.8)
    assert lost.recognized is True
    assert lost.kill_probability == pytest.approx(0.4)


def test_an_unlicensed_clone_engages_at_any_mhc1_but_kills_weakly():
    prof = _profile(SEQC2_ALLELES)
    naive = _subset(prof, KIR.KIR3DL1)   # no Bw4 on this patient
    ps = {
        nk_clone_decision(naive, mhc1_expression=m,
                          max_kill_probability=0.8).kill_probability
        for m in (0.0, 0.5, 1.0)
    }
    assert len(ps) == 1
    weak = ps.pop()
    assert 0.0 < weak
    licensed_uninhibited = nk_clone_decision(
        _subset(prof, KIR.KIR2DL1), mhc1_expression=0.0,
        max_kill_probability=0.8).kill_probability
    assert weak < licensed_uninhibited


def test_nk_agents_carry_a_drawn_kir_identity_and_others_do_not():
    m = _immune(patient_id="SEQC2")
    nk_id = m.add_immune_cell([100, 100, 50], "NK_cell")
    t_id = m.add_immune_cell([100, 100, 50], "T_cell")
    mac_id = m.add_immune_cell([100, 100, 50], "macrophage")
    assert m.immune_cells[nk_id].kir_subset in {k.value for k in KIR}
    assert m.immune_cells[t_id].kir_subset is None
    assert m.immune_cells[mac_id].kir_subset is None
    # With no typing on record there is no compartment to draw from.
    bare = _immune()
    bare_nk = bare.add_immune_cell([100, 100, 50], "NK_cell")
    assert bare.immune_cells[bare_nk].kir_subset is None


def test_the_clone_draw_does_not_consume_the_global_numpy_stream():
    # The closed-loop T-cell test is reproducible only because the
    # global stream is. Drawing clone identities from it would shift
    # every downstream roll.
    m = _immune(patient_id="SEQC2")
    np.random.seed(1234)
    m.add_immune_cell([0, 0, 0], "NK_cell")
    m.add_immune_cell([0, 0, 0], "NK_cell")
    after = np.random.random()
    np.random.seed(1234)
    assert after == np.random.random()
    # And the draw itself is reproducible from nk_seed.
    a = _immune(patient_id="SEQC2", nk_seed=7)
    b = _immune(patient_id="SEQC2", nk_seed=7)
    for mod in (a, b):
        for _ in range(8):
            mod.add_immune_cell([0, 0, 0], "NK_cell")
    assert ([c.kir_subset for c in a.immune_cells.values()]
            == [c.kir_subset for c in b.immune_cells.values()])


# ── 4. The macrophage PD-1 gate ──────────────────────────────────────

def test_pdl1_zero_leaves_the_phagocytic_baseline_exactly_unchanged():
    # The gate must be a bit-exact no-op on a cell that neither
    # IFN-gamma nor an oncogene has touched, which is every cell in the
    # existing baselines.
    for baseline in (0.0, 0.3, 0.8, 0.95, 1.0):
        assert mphage.phagocytosis_probability(baseline, 0.0) == baseline
        assert mphage.pd1_attenuation(0.0) == 1.0


def test_target_pdl1_suppresses_phagocytosis_and_blockade_rescues_it():
    # Gordon et al., Nature 2017: TAM PD-1 expression correlates
    # negatively with phagocytic potency, and PD-1/PD-L1 blockade
    # increases phagocytosis in vivo.
    full = mphage.phagocytosis_probability(0.8, 0.0)
    half = mphage.phagocytosis_probability(0.8, 0.5)
    none = mphage.phagocytosis_probability(0.8, 1.0)
    assert full > half > none > 0.0
    rescued = mphage.phagocytosis_probability(0.8, 1.0, checkpoint_block=1.0)
    assert rescued == pytest.approx(full)
    partial = mphage.phagocytosis_probability(0.8, 1.0, checkpoint_block=0.5)
    assert none < partial < full
    out = mphage.phagocytosis_outcome(0.8, 1.0, checkpoint_block=0.25)
    assert out.effective_signal == pytest.approx(0.75)
    assert out.probability == pytest.approx(0.8 * out.function_retained)


def test_zero_max_suppression_restores_the_previous_flat_behaviour():
    for pdl1 in (0.0, 0.5, 1.0):
        assert mphage.phagocytosis_probability(
            0.8, pdl1, max_suppression=0.0) == 0.8


# ── 5. The module: per-lineage parameters and the two NK rules ───────

def _immune(**cfg):
    base = {"n_t_cells": 0, "n_nk_cells": 0, "n_macrophages": 0}
    base.update(cfg)
    return ImmuneModule(base)


class _Target:
    """Minimal stand-in for CellState's fields that the innate arm reads."""

    def __init__(self, mhc1=1.0, pdl1=0.0, mutations=()):
        self.mhc1_expression = mhc1
        self.pdl1_expression = pdl1
        self.mutations = list(mutations)


def test_nk_and_macrophage_no_longer_share_one_flat_number():
    # Both lineages used to read self.kill_probability at a single site.
    m = _immune(nk_kill_probability=0.2, macrophage_kill_probability=0.9)
    assert m.nk_kill_probability == 0.2
    assert m.macrophage_kill_probability == 0.9
    nk = ImmuneCellStub("NK_cell")
    mac = ImmuneCellStub("macrophage")
    assert m._target_kill_probability(nk, _Target()) == pytest.approx(0.2)
    assert m._target_kill_probability(mac, _Target()) == pytest.approx(0.9)
    # The legacy key still drives both, so existing configs and
    # gpu/orchestrator (which reads module.kill_probability) are
    # unaffected.
    legacy = _immune(kill_probability=0.95)
    assert legacy.kill_probability == 0.95
    assert legacy.nk_kill_probability == 0.95
    assert legacy.macrophage_kill_probability == 0.95
    # validation/calibrator.py has declared nk_kill_probability as an
    # immune-module parameter all along with nothing reading it.
    from cognisom.validation.calibrator import DEFAULT_PARAMETERS
    declared = {p.name for p in DEFAULT_PARAMETERS if p.module == "immune"}
    assert "nk_kill_probability" in declared


class ImmuneCellStub:
    def __init__(self, cell_type, polarization=0.0, position=None):
        self.cell_type = cell_type
        self.active_tcr_match = None
        # Fields a live ImmuneCell carries. polarization 0.0 is M0, at
        # which the macrophage polarization factor is exactly 1.0, so a
        # default stub exercises the PD-1 gate alone. position is read
        # by the arginase gate's neighbour scan.
        self.polarization = polarization
        self.cell_id = -1
        self.in_blood = False
        self.position = (
            np.zeros(3, dtype=np.float32) if position is None
            else np.asarray(position, dtype=np.float32)
        )


def test_the_missing_self_threshold_is_configurable_not_a_literal():
    assert missing_self_recognized(0.39) is True
    assert missing_self_recognized(0.4) is False
    assert missing_self_recognized(0.6, threshold=0.7) is True
    m = _immune(nk_missing_self_threshold=0.7)
    assert m.nk_missing_self_threshold == 0.7
    nk = ImmuneCellStub("NK_cell")
    assert m._recognize(nk, _Target(mhc1=0.6))[0] is True
    assert m._recognize(nk, _Target(mhc1=0.8))[0] is False


def test_without_a_patient_typing_the_legacy_rule_runs_and_says_so():
    m = _immune()
    assert m.get_nk_licensing_profile() is None
    state = m.get_state()
    assert state["nk_model"] == "missing_self_threshold"
    assert state["nk_patient_specific"] is False
    assert state["nk_licensed_fraction"] is None
    nk = ImmuneCellStub("NK_cell")
    # The measured legacy behaviour: a hard step at 0.4.
    assert m._recognize(nk, _Target(mhc1=0.39))[0] is True
    assert m._recognize(nk, _Target(mhc1=0.4))[0] is False
    assert m._target_kill_probability(nk, _Target(mhc1=0.0)) == pytest.approx(0.8)


def test_an_unknown_patient_id_does_not_invent_a_typing():
    # No measured typing on file -> the receptor model stays off. It
    # must NOT fall back to a synthetic or population-default profile:
    # a licensing prediction from invented alleles is indistinguishable
    # from one built on real reads.
    m = _immune(patient_id="NO-SUCH-PATIENT")
    assert m.get_nk_licensing_profile() is None
    assert m.get_state()["nk_model"] == "missing_self_threshold"


def test_the_measured_seqc2_typing_reaches_the_simulation():
    m = _immune(patient_id="SEQC2")
    prof = m.get_nk_licensing_profile()
    assert prof is not None
    assert prof.hla.groups == (LigandGroup.C2,)
    assert "SEQC2" in prof.hla.source and "SRR7890874" in prof.hla.source
    state = m.get_state()
    assert state["nk_model"] == "kir_hla_balance"
    assert state["nk_patient_specific"] is True
    assert state["nk_kir_patient_specific"] is False
    assert state["nk_ligand_groups"] == ["C2"]
    assert state["nk_licensed_fraction"] == pytest.approx(0.95 / 3.32,
                                                          abs=1e-6)
    # And the NK branch is now graded rather than a step at 0.4.
    nk = ImmuneCellStub("NK_cell")
    ps = [m._target_kill_probability(nk, _Target(mhc1=x))
          for x in (0.0, 0.2, 0.39, 0.41)]
    assert ps == sorted(ps, reverse=True)
    assert len(set(round(p, 9) for p in ps)) == 4, ps


def test_an_explicit_allele_list_is_unmeasured_unless_declared():
    m = _immune(patient_hla=["HLA-C*06:02", "HLA-B*45:01"])
    prof = m.get_nk_licensing_profile()
    assert prof is not None
    assert prof.hla.is_patient_specific is False
    assert m.get_state()["nk_patient_specific"] is False
    declared = _immune(patient_hla=["HLA-C*06:02", "HLA-B*45:01"],
                       patient_hla_measured=True)
    assert declared.get_nk_licensing_profile().hla.is_patient_specific is True


def test_an_unrecognised_allele_degrades_visibly_instead_of_killing_the_engine(caplog):
    """An allele outside the ligand table must not take the module down.

    This originally raised UnknownAlleleError out of __init__. Refusing to
    guess a KIR ligand group is right -- the receptor model must not invent
    one -- but raising from the constructor turns a data-coverage gap into a
    dead engine: the ligand table covers a subset of several thousand known
    class-I alleles, and one unrecognised allele would make the whole immune
    module unconstructable for that patient, taking every simulation with it.

    So it degrades to the legacy missing-self step rule instead. What makes
    that acceptable rather than the silent substitution this codebase has
    spent its recent history removing is that the degradation is VISIBLE
    three ways, all asserted here: a WARNING naming the refused allele,
    get_nk_licensing_profile() returning None, and nk_model reported in
    get_state(). A caller can tell which rule ran.
    """
    with caplog.at_level(logging.WARNING):
        module = _immune(patient_hla=["HLA-C*99:99"])

    # 1. it constructed
    assert module is not None
    # 2. the receptor model is off, and says so rather than guessing
    assert module.get_nk_licensing_profile() is None
    assert module.get_state()["nk_model"] != "receptor_balance"
    # 3. the refusal is on the record, naming the allele
    warnings = " ".join(r.getMessage() for r in caplog.records
                        if r.levelno >= logging.WARNING)
    assert "C*99:99" in warnings, f"the refused allele is not named: {warnings!r}"
    assert "not guessed" in warnings or "not on record" in warnings


def test_the_ligand_table_still_refuses_an_unknown_allele_at_its_own_layer():
    """The module degrades; the ligand map itself must still refuse.

    Fail-soft belongs at the module boundary, not in the receptor model --
    hla_ligand_profile inventing a ligand group would be the actual defect.
    """
    with pytest.raises(UnknownAlleleError):
        hla_ligand_profile(["HLA-C*99:99"])


def test_macrophage_engagement_reads_target_state():
    # `return True` ignored the don't-eat-me side entirely: macrophage
    # kills were identical at target MHC-I 0.0, 0.4 and 1.0.
    m = _immune(macrophage_kill_probability=0.8)
    mac = ImmuneCellStub("macrophage")
    clean = m._target_kill_probability(mac, _Target(pdl1=0.0))
    hot = m._target_kill_probability(mac, _Target(pdl1=1.0))
    assert clean == pytest.approx(0.8)
    assert hot < clean
    # Checkpoint blockade now reaches the innate arm too, from the same
    # knob that drives the T-cell branch.
    blocked = _immune(macrophage_kill_probability=0.8, checkpoint_block=1.0)
    assert blocked._target_kill_probability(mac, _Target(pdl1=1.0)) == \
        pytest.approx(0.8)
    # Full suppression refuses engagement outright.
    total = _immune(macrophage_pd1_max_suppression=1.0)
    assert total._recognize(mac, _Target(pdl1=1.0))[0] is False
    assert total._recognize(mac, _Target(pdl1=0.0))[0] is True


def test_the_licensing_event_fires_only_when_a_typing_exists():
    seen = []
    engine = SimulationEngine(SimulationConfig(dt=0.01, duration=0.01,
                                               use_gpu=False))
    engine.register_module("immune", ImmuneModule, {
        "n_t_cells": 0, "n_nk_cells": 2, "n_macrophages": 0,
        "patient_id": "SEQC2",
    })
    engine.event_bus.subscribe(EventTypes.NK_LICENSING_RESOLVED, seen.append)
    engine.initialize()
    engine.event_bus.process_events()
    assert len(seen) == 1, seen
    assert seen[0]["ligand_groups"] == ["C2"]
    assert seen[0]["hla_is_patient_specific"] is True

    quiet = []
    engine2 = SimulationEngine(SimulationConfig(dt=0.01, duration=0.01,
                                                use_gpu=False))
    engine2.register_module("immune", ImmuneModule, {
        "n_t_cells": 0, "n_nk_cells": 2, "n_macrophages": 0,
    })
    engine2.event_bus.subscribe(EventTypes.NK_LICENSING_RESOLVED, quiet.append)
    engine2.initialize()
    engine2.event_bus.process_events()
    assert quiet == []


def _colocated_nk_run(mhc, *, immune_cfg, subset=None, n=6,
                      duration=1.0, seed=0, mutations=()):
    """Put one NK cell on top of each cancer cell and run.

    The patrol regime is encounter-limited -- a 10 um detection radius
    in a 160 um box means only a handful of NK cells meet a target in
    four hours, which buries the receptor model in sampling noise. The
    closed-loop T-cell test colocates for the same reason. ``subset``
    forces every NK clone's KIR identity so the engagement decision is
    deterministic.
    """
    np.random.seed(seed)
    engine = SimulationEngine(SimulationConfig(dt=0.01, duration=duration,
                                               use_gpu=False))
    engine.register_module("cellular", CellularModule, {
        "n_normal_cells": 0, "n_cancer_cells": 0,
    })
    cfg = {"n_t_cells": 0, "n_nk_cells": n, "n_macrophages": 0}
    cfg.update(immune_cfg)
    engine.register_module("immune", ImmuneModule, cfg)
    killers = []
    engine.event_bus.subscribe(
        EventTypes.CANCER_KILLED, lambda d: killers.append(d["killer_type"]))
    # Which NK cells engaged at least once. total_activations counts
    # engagements, and a contact that does not kill now ends in release and
    # re-engagement, so it counts attempts; "this clone engages this
    # target" is the set of cells that ever did.
    engaged = set()
    engine.event_bus.subscribe(
        EventTypes.IMMUNE_ACTIVATED, lambda d: engaged.add(d["immune_id"]))
    engine.initialize()
    cellular = engine.modules["cellular"]
    immune = engine.modules["immune"]
    immune.engaged_cells = engaged
    immune.set_cellular_module(cellular)
    for i, ic in enumerate(immune.immune_cells.values()):
        pos = [100.0 + 20.0 * i, 100.0, 50.0]
        cid = cellular.add_cell(position=pos, cell_type="cancer")
        cell = cellular.cells[cid]
        cell.mhc1_expression = mhc
        cell.mutations = list(mutations)
        ic.position = np.array(pos, dtype=np.float32)
        if subset is not None:
            ic.kir_subset = KIR(subset).value
    engine.run()
    return immune, killers


def test_a_licensed_clone_spares_an_intact_mhc_target_through_the_engine():
    # KIR2DL1 on a C2-carrying patient: avidity 1.0 against an
    # autologous target, so at MHC-I 1.0 the inhibitory term is 1.0 and
    # the activating term cannot exceed it whatever the mutation load.
    # Not weak killing -- no engagement at all. This is self-tolerance,
    # and it is what the previous rule got exactly backwards above its
    # 0.4 threshold (no NK activity at all, licensed or not).
    cfg = {"patient_id": "SEQC2"}
    intact, killers = _colocated_nk_run(
        1.0, immune_cfg=cfg, subset=KIR.KIR2DL1,
        mutations=("TP53:R175H",) * 50,
    )
    assert intact.total_activations == 0
    assert intact.total_kills == 0 and killers == []
    # The same clone on an MHC-I-null target engages every target and
    # kills: missing self.
    lost, lost_killers = _colocated_nk_run(
        0.0, immune_cfg=cfg, subset=KIR.KIR2DL1)
    assert len(lost.engaged_cells) == 6
    assert lost.total_kills > 0
    assert set(lost_killers) == {"NK_cell"}


def test_an_unlicensed_clone_is_mhc_independent_through_the_engine():
    # KIR3DL1 on a Bw4-negative patient has no cognate ligand, so
    # nothing for MHC-I to inhibit it through. Engagement must be
    # identical at MHC-I 0.0 and 1.0 -- the licensing model's signature
    # prediction, and the reason an MHC-intact tumour is not simply
    # invisible to the whole NK compartment.
    cfg = {"patient_id": "SEQC2"}
    counts = []
    for mhc in (0.0, 0.5, 1.0):
        mod, _ = _colocated_nk_run(mhc, immune_cfg=cfg, subset=KIR.KIR3DL1)
        counts.append((mod.total_activations, mod.total_kills,
                       len(mod.engaged_cells)))
    assert len({c[0] for c in counts}) == 1, counts
    assert all(c[2] == 6 for c in counts), counts
    assert all(c[1] > 0 for c in counts), counts


def test_the_legacy_step_rule_still_behaves_exactly_as_measured():
    # With no typing on record the innate arm must reproduce what was
    # measured before this change: a hard step at 0.4.
    for mhc, expect_engagement in ((0.0, True), (0.39, True),
                                   (0.4, False), (1.0, False)):
        mod, killers = _colocated_nk_run(mhc, immune_cfg={})
        assert mod.get_state()["nk_model"] == "missing_self_threshold"
        if expect_engagement:
            assert len(mod.engaged_cells) == 6, mhc
            assert mod.total_kills > 0, mhc
        else:
            assert mod.total_activations == 0, mhc
            assert mod.total_kills == 0 and killers == [], mhc


def test_a_narrowly_licensed_patient_keeps_more_nk_activity_at_high_mhc():
    # SEQC2 carries one ligand group, so 71% of its compartment is
    # unlicensed and MHC-independent. A patient carrying C1, C2 and Bw4
    # has every subset licensed, so an intact-MHC target shuts almost
    # all of it down. Same target, different patient, different innate
    # response -- which is the whole point of resolving this from a
    # typing.
    broad = {"patient_hla": BROADLY_LICENSED, "patient_hla_measured": True}
    seqc2 = {"patient_id": "SEQC2"}
    acts = {}
    for label, cfg in (("broad", broad), ("seqc2", seqc2)):
        total = 0
        for receptor in (KIR.KIR2DL1, KIR.KIR2DL2, KIR.KIR2DL3, KIR.KIR3DL1):
            mod, _ = _colocated_nk_run(1.0, immune_cfg=cfg, subset=receptor)
            total += mod.total_activations
        acts[label] = total
    assert acts["seqc2"] > acts["broad"], acts
    # And the ordering reverses once MHC-I is lost, because the broad
    # patient's compartment is fully educated.
    lost = {}
    for label, cfg in (("broad", broad), ("seqc2", seqc2)):
        total = 0
        for receptor in (KIR.KIR2DL1, KIR.KIR2DL2, KIR.KIR2DL3, KIR.KIR3DL1):
            mod, _ = _colocated_nk_run(0.0, immune_cfg=cfg, subset=receptor)
            total += mod.total_kills
        lost[label] = total
    assert lost["broad"] >= lost["seqc2"], lost


# ── 6. The flagship T-cell loop is untouched ─────────────────────────

def test_the_tcell_closed_loop_is_unchanged_by_the_innate_rewrite():
    # Same trace as test_closed_loop_neoantigen, but with the NK
    # receptor model configured and the macrophage PD-1 gate live, to
    # prove the innate changes do not reach the flagship path.
    np.random.seed(0)
    engine = SimulationEngine(SimulationConfig(dt=0.01, duration=0.5,
                                               use_gpu=False))
    engine.register_module("molecular", MolecularModule, {
        "transcription_rate": 0.0, "exosome_release_rate": 0.0,
        "mutation_rate": 0.0,
    })
    engine.register_module("cellular", CellularModule, {
        "n_normal_cells": 0, "n_cancer_cells": 0,
        "hla_alleles": ["HLA-A*02:01", "HLA-A*24:02", "HLA-B*07:02"],
    })
    engine.register_module("immune", ImmuneModule, {
        "n_t_cells": 1, "n_nk_cells": 0, "n_macrophages": 0,
        "tcr_recognition_threshold": 0.0, "tcr_repertoire_size": 200,
        "tcr_seed": 0, "costimulation": 1.0, "checkpoint_block": 0.0,
        "patient_id": "SEQC2",
        # The subject is the causal chain, not kill timing. A lethal T-cell
        # contact lasts ~75 min by the library's measured value, longer than
        # this run, so it is made one minute long here.
        "lethal_contact_minutes": {"T_cell": 1.0},
    })
    engine.initialize()
    molecular = engine.modules["molecular"]
    cellular = engine.modules["cellular"]
    immune = engine.modules["immune"]
    cellular.set_molecular_module(molecular)
    immune.set_cellular_module(cellular)

    cancer_id = cellular.add_cell(position=[100, 100, 50], cell_type="cancer")
    cellular.cells[cancer_id].mhc1_expression = 0.9
    molecular.add_cell(cancer_id)
    tid = next(iter(immune.immune_cells))
    pos = np.array([100.0, 100.0, 50.0], dtype=np.float32)
    cellular.cells[cancer_id].position = pos.copy()
    immune.immune_cells[tid].position = pos.copy()

    molecular.introduce_mutation(cancer_id, "TP53", "R175H")
    engine.run()

    counts = engine.event_bus.get_statistics()["event_counts"]
    assert counts.get(EventTypes.MUTATION_OCCURRED, 0) == 1
    assert counts.get(EventTypes.PEPTIDE_GENERATED, 0) == 1
    assert counts.get(EventTypes.PEPTIDE_PRESENTED, 0) == 2
    assert counts.get(EventTypes.IMMUNE_ACTIVATED, 0) == 1
    assert counts.get(EventTypes.CANCER_KILLED, 0) == 1
    assert counts.get(EventTypes.CELL_KILLED_BY_TCELL, 0) == 1
    assert immune.total_kills == 1
    assert immune.total_tcell_kills == 1
    assert immune.total_tcr_recognitions == 1
    assert immune.total_exhaustion_transitions == 0
    assert cellular.total_peptides_presented == 2


def test_the_tcell_hill_branch_is_not_gated_by_the_innate_changes():
    # The T-cell kill probability must be byte-identical whether or not
    # the NK receptor model is configured.
    from cognisom.engine.py.immune.tcr_repertoire import TCRRepertoire

    def p_kill(**extra):
        cfg = {"n_t_cells": 1, "n_nk_cells": 0, "n_macrophages": 0,
               "tcr_recognition_threshold": 0.0, "tcr_repertoire_size": 200,
               "tcr_seed": 0, "costimulation": 1.0}
        cfg.update(extra)
        m = ImmuneModule(cfg)
        m._tcr_repertoire = TCRRepertoire(size=200, recognition_threshold=0.0,
                                          seed=0)
        cell = ImmuneCellStub("T_cell")
        cell.active_tcr_match = _FakeMatch(0.8)
        return m._target_kill_probability(cell, _Target(mhc1=0.9))

    assert p_kill() == p_kill(patient_id="SEQC2")
    assert p_kill() == p_kill(macrophage_pd1_max_suppression=1.0)
    assert p_kill() > 0.5


class _FakeTCR:
    tcr_id = None
    cdr3 = "CASSX"


class _FakeMatch:
    def __init__(self, affinity):
        self.affinity = affinity
        self.is_exhausted = False
        self.tcr = _FakeTCR()
        self.presentation = None


def test_a_licensed_clone_tolerates_self_imperfectly():
    """Pin a KNOWN LIMITATION so it is visible and cannot silently worsen.

    A licensed NK clone should never engage a target it is licensed against.
    It currently can, at high mutation load against full self-MHC, because
    the activating and inhibitory terms are summed without a shared scale.
    See the KNOWN LIMITATION block in nk_receptor.py: both obvious constant
    rescales were measured and each destroys the graded MHC response, so the
    repair is structural rather than a tweak.

    This test does not assert the behaviour is right. It asserts it is no
    worse than measured, and that the graded response it trades against is
    still there -- so a future fix has to improve one without losing the
    other.
    """
    hla = hla_ligand_profile(
        ["HLA-A*02:01", "HLA-B*57:01", "HLA-C*01:02", "HLA-C*06:02"])
    profile = licensing_profile(hla, population_prior_kir())
    assert all(s.licensed for s in profile.subsets), "not fully licensed"

    # The limitation, bounded: self-reactivity exists but stays small.
    at_zero = nk_decision(profile, mhc1_expression=1.0, n_mutations=0,
                          max_kill_probability=0.8)
    assert at_zero.kill_probability == 0.0, "self-reactive with no mutations"

    at_fifty = nk_decision(profile, mhc1_expression=1.0, n_mutations=50,
                           max_kill_probability=0.8)
    assert 0.0 < at_fifty.kill_probability <= 0.15, (
        f"self-reactivity against intact MHC-I is {at_fifty.kill_probability:.4f}; "
        f"measured at 0.1225 when this was documented. Above 0.15 the "
        f"licensing model is not doing its job at all."
    )

    # And the thing any repair must not destroy: a graded response to MHC loss.
    graded = [
        nk_decision(profile, mhc1_expression=m, n_mutations=2,
                    max_kill_probability=0.8).kill_probability
        for m in (1.0, 0.75, 0.5, 0.25, 0.05)
    ]
    assert len(set(round(v, 6) for v in graded)) == 5, (
        f"the graded MHC response collapsed to {len(set(round(v,6) for v in graded))} "
        f"distinct values: {graded}. Rescaling the inhibitory or activating "
        f"constants to fix self-tolerance does exactly this."
    )
    assert graded == sorted(graded), f"not monotonic in MHC loss: {graded}"
