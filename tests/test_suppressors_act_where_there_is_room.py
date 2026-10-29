"""Phase 3 of realism item 1: Tregs and MDSCs, measured before built.

The brief warned that the T-cell kill path is already heavily
suppressed, and that stacking two more multiplicative suppressors onto
it would either do nothing visible or crush T cells -- and that either
way the mechanism would be unfalsifiable. So the headroom was measured
FIRST, across 53,388 instrumented T-cell contacts in 27 slab runs
(Baseline, Aggressive Tumor and Immunotherapy, 24 h, shipped and
all-T-cell infiltrates), before a line of this was written.

WHAT THE MEASUREMENT SAID
-------------------------
  quantity                             Baseline/shipped   Immuno/shipped
  smallest p_kill in 53,388 contacts   0.007775 (global minimum)
  contacts with p_kill < 0.001         0 of 53,388
  arginase multiplier, mean            0.4057  (floor 0.400)   0.4005
  local M2 arginine burden == 1.0      98.39% of contacts      99.43%
  exhaustion multiplier                1.0 at EVERY contact    1.0
  sympathetic multiplier               1.0 at EVERY contact    1.0
  lactate multiplier                   1.0 at EVERY contact    1.0

Two answers fell out of that, and they point in opposite directions.

The path is NOT at a floor, so a suppressor CAN move the outcome: the
engine turns p_kill into a Poisson perforin-hit rate, so a factor f on
p scales expected hits by exactly f, and no contact is near zero.

But the ARGINASE gate specifically is at its floor wherever the shipped
presets put macrophages. That is what decided the MDSC question, below.

WHICH MECHANISM EACH SUPPRESSOR USES, AND WHETHER IT SHARES OR ADDS
-------------------------------------------------------------------
  arginase      extracellular L-arginine supply   Rodriguez 2004
  sympathetic   beta2AR -> cAMP -> PKA            Farooq 2023 etc
  lactate       acidosis                          Fischer 2007
  exhaustion    the PD-1-hi fixed state           Philip 2017
  Treg (NEW)    TGF-beta repression of perforin   Thomas & Massague 2005
  MDSC          extracellular L-arginine supply   Ochoa 2007  -> SHARES
                                                  the arginase gate, and
                                                  so was NOT built

Every mechanism in that table is distinct except MDSC, whose mechanism
is the one already on the path. MDSC therefore had to contribute to the
SAME arginine burden rather than multiply a second gate -- and the
burden is already saturated, so it cannot matter in a shipped preset.
See test_an_mdsc_would_have_to_share_the_arginase_gate_and_could_not_matter.
"""
from __future__ import annotations

import contextlib
import io
from collections import Counter
from typing import Dict, List

import numpy as np
import pytest

from cognisom.core import SimulationConfig, SimulationEngine
from cognisom.core.event_bus import EventTypes
from cognisom.engine.py.immune import treg_suppression as ts
from cognisom.engine.py.immune.macrophage_polarization import (
    DEFAULT_ARGINASE_MAX_SUPPRESSION, DEFAULT_CYTOKINE_SATURATION_COUNT,
    arginase_attenuation, local_signal)
from cognisom.engine.py.immune.tcell_kill import kill_probability
from cognisom.engine.py.immune.tcr_repertoire import (
    EXHAUSTED_KILL_MULTIPLIER, EXHAUSTION_ONSET_HOURS, TCRRepertoire)
from cognisom.modules.cellular_module import CellularModule
from cognisom.modules.immune_module import ImmuneModule
from cognisom.modules.molecular_module import MolecularModule

#: Surface class I a cancer cell is seeded at, and the costimulation
#: and affinity the instrumented runs measured, so the pure-function
#: checks below sit at the regime a real contact is actually in.
MHC1_SEEDED = 0.30
COSTIM = 0.6
#: Mean TCR-pMHC affinity over the 53,388 instrumented contacts.
MEASURED_AFFINITY = 0.9395
#: Mean MHC-I of the cells those contacts engaged.
MEASURED_MHC1 = 0.2933
#: Mean combined Hill signal, against the rule's EC50 of 0.30. Below
#: EC50 the Hill is quartic, which is what makes costimulation a 3.6x
#: elasticity lever -- and what made the ICB-rescue term's sign matter.
MEASURED_SIGNAL = 0.1654


# ── the arena: effectors against cells at the seeded phenotype ────────

def _presentations(alleles):
    """One cheap MHCPresentation per allele, built by hand so the
    measurement does not depend on which binding predictor is installed
    (MHCflurry is not importable on py3.13)."""
    from cognisom.engine.py.immune.mhc_loading import MHCPresentation
    from cognisom.engine.py.molecular.peptidome import Peptide
    out = []
    for i, allele in enumerate(alleles):
        peptide = Peptide(
            sequence='KLVVVGADG', source_gene='KRAS', length=9,
            is_mutant=True, wild_type_sequence='KLVVVGAGG',
            mutation_label='G12D', anchor_position_in_peptide=7,
            parent_position_1based=5 + i, cleavage_score=0.9)
        out.append(MHCPresentation(
            peptide=peptide, hla_allele=allele, ic50_nm=30.0,
            binding_level='strong', presentation_score=0.95))
    return out


def _lineup(n_t: int = 0, hours: float = 6.0, dt: float = 0.05,
            checkpoint_block: float = 0.0, **immune_over):
    """An engine with no seeded population and ``n_t`` T cells.

    Only the keys a caller names reach the module, so the "absent" case
    is genuinely the config this repository shipped and not a config
    with the new keys set to their defaults.
    """
    engine = SimulationEngine(SimulationConfig(
        dt=dt, duration=hours, use_gpu=False))
    engine.register_module('molecular', MolecularModule, {
        'transcription_rate': 0.0, 'exosome_release_rate': 0.0,
        'mutation_rate': 0.0})
    engine.register_module('cellular', CellularModule, {
        'n_normal_cells': 0, 'n_cancer_cells': 0,
        # No division and no spontaneous death, so every death is a kill.
        'division_time_cancer': 1e9, 'division_time': 1e9,
        'apoptosis_rate': 0.0})
    cfg = {
        'n_t_cells': n_t, 'n_nk_cells': 0, 'n_macrophages': 0,
        'tcr_recognition_threshold': 0.0, 'tcr_repertoire_size': 200,
        'tcr_seed': 0, 'costimulation': COSTIM,
        'checkpoint_block': checkpoint_block,
        'macrophage_polarization_enabled': False,
    }
    cfg.update(immune_over)
    engine.register_module('immune', ImmuneModule, cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        engine.initialize()
    cellular = engine.modules['cellular']
    immune = engine.modules['immune']
    immune.set_cellular_module(cellular)
    return engine, cellular, immune


def arena(n_t: int = 20, seed: int = 1, n_each: int = 20,
          hours: float = 6.0, **kw) -> Dict[str, object]:
    """Kills, perforin hits and the per-contact kill probability.

    ``mean_p`` is the deterministic quantity -- the mean of
    ``_target_kill_probability`` over every T-cell contact -- so a
    suppressor's per-contact effect is readable exactly, separately from
    the Poisson and Bernoulli draws downstream of it.
    """
    np.random.seed(seed)
    engine, cellular, immune = _lineup(n_t=n_t, hours=hours, **kw)
    rng = np.random.default_rng(seed)
    centre = np.array([100.0, 100.0, 50.0])
    for _ in range(n_each):
        pos = centre + rng.uniform(-11, 11, 3)
        cid = cellular.add_cell(position=pos.tolist(), cell_type='cancer')
        cell = cellular.cells[cid]
        cell.mhc1_expression = MHC1_SEEDED
        cell.mhc1_baseline = MHC1_SEEDED
        cell.mhc1_displayed_peptides = _presentations(cellular.hla_alleles)
        cell.pdl1_expression = 0.0
        cell.ifn_gamma_level = None
    for icell in immune.immune_cells.values():
        icell.position = centre + rng.uniform(-11, 11, 3)

    kills: Counter = Counter()
    engine.event_bus.subscribe(
        EventTypes.CANCER_KILLED,
        lambda d: kills.update([d.get('killer_type')]))
    probs: List[float] = []
    exhausted: List[bool] = []
    real = immune._target_kill_probability

    def spy(immune_cell, target_cell):
        p = real(immune_cell, target_cell)
        if (immune_cell.cell_type == 'T_cell'
                and immune_cell.active_tcr_match is not None):
            probs.append(p)
            exhausted.append(
                immune._live_is_exhausted(immune_cell.active_tcr_match))
        return p

    immune._target_kill_probability = spy
    with contextlib.redirect_stdout(io.StringIO()):
        engine.run()
    return {
        'kills': sum(kills.values()),
        'perforin_hits': immune.total_perforin_hits,
        'contacts': len(probs),
        'exhausted_contacts': sum(exhausted),
        'mean_p': float(np.mean(probs)) if probs else 0.0,
        'treg_mult': immune._treg_cytolytic_attenuation(),
        'clones_exhausted': immune._tcr_repertoire.exhausted_count(),
    }


def _totals(seeds, **kw):
    out = [arena(seed=s, **kw) for s in seeds]
    return {
        'kills': sum(o['kills'] for o in out),
        'hits': sum(o['perforin_hits'] for o in out),
        'contacts': sum(o['contacts'] for o in out),
        'mean_p': float(np.mean([o['mean_p'] for o in out])),
        'treg_mult': out[0]['treg_mult'],
    }


# ── part 1: the mechanism is SHARED, not doubled ─────────────────────

def test_an_mdsc_would_have_to_share_the_arginase_gate_and_could_not_matter():
    """Why no MDSC was built, in arithmetic.

    MDSCs suppress T cells by depleting extracellular L-arginine --
    "Arginase production by MDSC depletes arginine from the tumor
    microenvironment and impairs T-cell signal transduction and
    function" (Ochoa 2007, PMID 17255300). That is the mechanism the
    arginase gate on this path already models: ``arginase_attenuation``
    cites Rodriguez 2004 (PMID 15313928), the paper showing
    tumour-associated myeloid cells "deplete extracellular L-Arg in
    vitro". One arginine pool, two cell names.

    So an MDSC adds ARG1 to the SAME saturating burden sum, and the
    difference between sharing and doubling is not cosmetic. At the
    burden the shipped Baseline actually sits at -- exactly 1.0 on
    98.39% of 7,506 measured contacts -- sharing changes the multiplier
    by nothing at all, while a second gate of the same strength would
    cut it a further 20% out of a pool that is already empty:

        burden   arginase   +1 MDSC shared   +1 MDSC as a 2nd gate
        0.000     1.000         0.800               0.800
        0.333     0.800         0.600               0.640
        0.667     0.600         0.400               0.480
        1.000     0.400         0.400               0.320

    Projected over the measured contacts, one MDSC-equivalent in the
    shared sum moves the Baseline T-cell hit rate -0.29% (three move it
    -0.60%), and the T-cell arm does 9 of that preset's 128 kills, so
    the population effect is about -0.04%. A separate gate would have
    read -60%. No MDSC code exists.
    """
    sat = DEFAULT_CYTOKINE_SATURATION_COUNT
    one_mdsc = 1.0 / sat

    # Sharing: one burden, and at the ceiling an extra source is inert.
    assert arginase_attenuation(local_signal(3.0, sat)) == pytest.approx(0.4)
    at_ceiling_shared = arginase_attenuation(min(1.0, 1.0 + one_mdsc))
    assert at_ceiling_shared == pytest.approx(0.400000)

    # Doubling: a second gate manufactures suppression out of nothing.
    at_ceiling_doubled = (arginase_attenuation(1.0)
                          * arginase_attenuation(one_mdsc))
    assert at_ceiling_doubled == pytest.approx(0.320000)
    assert at_ceiling_doubled < at_ceiling_shared

    # The floor is the gate's whole range, and it is already reached.
    assert DEFAULT_ARGINASE_MAX_SUPPRESSION == 0.6
    assert arginase_attenuation(1.0) == pytest.approx(
        1.0 - DEFAULT_ARGINASE_MAX_SUPPRESSION)

    # Below the ceiling sharing and doubling differ too, and sharing is
    # the one that cannot exceed the single mechanism's measured floor.
    for burden in (0.0, 1 / 3, 2 / 3):
        shared = arginase_attenuation(min(1.0, burden + one_mdsc))
        doubled = (arginase_attenuation(burden)
                   * arginase_attenuation(one_mdsc))
        assert shared >= 1.0 - DEFAULT_ARGINASE_MAX_SUPPRESSION
        if burden > 0.0:
            assert doubled > shared     # a second gate always cuts more

    # And the decision is recorded in the module, not only in prose.
    report = ts.calibration_report()
    assert 'MDSC' in report['not_built']
    assert '17255300' in report['not_built']     # Ochoa, the MDSC paper
    assert '15313928' in report['not_built']     # Rodriguez, the gate's


def test_the_treg_mechanism_is_not_any_existing_gate_under_a_new_name():
    """The Treg term uses TGF-beta repression of the cytolytic
    machinery -- "TGF-beta acts on cytotoxic T lymphocytes (CTLs) to
    specifically inhibit the expression of five cytolytic gene
    products -- namely, perforin, granzyme A, granzyme B, Fas ligand,
    and interferon gamma" (Thomas & Massague 2005, PMID 16286245) --
    which is the route by which Tregs abolished CD8 cytotoxicity in
    vivo while leaving expansion and IFN-gamma intact (Chen 2005,
    PMID 15623559, and resistant in dnTGF-betaR CD8 cells).

    Perforin hits are what this engine's T-cell kill path counts, so
    the term is on cytolytic capacity and not on recognition.

    It is independent of macrophage density, which is the test that it
    has not quietly become a second arginase gate: with the Treg
    fraction fixed, the multiplier is the same at every M2 burden.
    """
    for burden in (0.0, 0.5, 1.0):
        # The Treg function takes no density argument at all; this
        # asserts the separation at the level the module composes them.
        assert ts.treg_cytolytic_attenuation(0.3) == pytest.approx(0.846939,
                                                                   abs=1e-6)
        assert arginase_attenuation(burden) == pytest.approx(
            1.0 - burden * DEFAULT_ARGINASE_MAX_SUPPRESSION)

    report = ts.calibration_report(0.3)
    assert '16286245' in report['mechanism']     # Thomas & Massague
    assert '15623559' in report['mechanism']     # Chen
    assert 'perforin' in report['mechanism']
    assert 'not ICB-rescuable' in report['mechanism']


def test_tam_tgf_beta_is_deliberately_not_fed_into_the_treg_term():
    """M2 macrophages in this engine already secrete TGF_beta1, and it
    is NOT wired into the Treg term, for the same reason MDSCs were not
    built: the M2 compartment already suppresses this exact path
    through arginase, at its floor on 98.39% of shipped Baseline
    contacts, so a second M2-density-driven multiplier would charge one
    cell population twice.

    The polarization arm makes the identical call -- it folds TGF-beta
    into its IL-10 term rather than adding a second one. This test pins
    that TGF_beta1 reaches no T-cell kill multiplier: changing a
    macrophage's polarization moves the arginase gate and nothing else.
    """
    from cognisom.engine.py.immune.macrophage_polarization import (
        cytokine_output)
    pure_m2 = cytokine_output(1.0)
    assert pure_m2['TGF_beta1'] == 1.0 and pure_m2['ARG1'] == 1.0
    # Both are emitted, and only one of them gates T cells. The Treg
    # multiplier is a function of treg_fraction alone.
    assert ts.treg_cytolytic_attenuation(None) == 1.0
    assert ts.treg_cytolytic_attenuation(0.0) == 1.0


# ── part 2: no Tregs and no MDSCs is EXACTLY as before ───────────────

@pytest.mark.timeout(300)
def test_a_configuration_with_no_tregs_is_bit_identical_to_before():
    """The provable no-op, and it has to be exact rather than close:
    nothing in this platform measures a Treg fraction for any patient
    today, so every preset and every other test in this repository runs
    the path this test pins.

    Three seeds of a 20-T-cell 6 h arena. The key ABSENT from the
    config, the key None, and the key 0.0 all give the same 10 kills,
    the same 49 perforin hits, the same 1,249 contacts and the same
    per-contact kill probability to 12 decimal places -- and so do the
    two no-op forms of the pre-exhaustion key.
    """
    seeds = (1, 2, 3)
    base = _totals(seeds)                                  # key absent
    assert (base['kills'], base['hits'], base['contacts']) == (10, 49, 1249)
    assert base['treg_mult'] == 1.0

    for label, kw in (('treg None', {'treg_fraction': None}),
                      ('treg 0.0', {'treg_fraction': 0.0}),
                      ('exhaustion None', {'t_cell_initial_exhaustion': None}),
                      ('exhaustion 0.0', {'t_cell_initial_exhaustion': 0.0})):
        got = _totals(seeds, **kw)
        assert got['kills'] == base['kills'], label
        assert got['hits'] == base['hits'], label
        assert got['contacts'] == base['contacts'], label
        assert got['mean_p'] == pytest.approx(base['mean_p'], abs=1e-12), label
        assert got['treg_mult'] == 1.0, label


def test_the_no_op_is_exact_in_the_pure_functions_too():
    """1.0 exactly, with no epsilon left over, at every no-op input --
    the same property local_signal and lactate_attenuation are built to
    have, so the gates compose to exactly 1.0 and not to 0.99999."""
    for f in (None, 0.0, -0.0):
        assert ts.treg_occupancy(f) == 0.0
        assert ts.treg_cytolytic_attenuation(f) == 1.0
    # A negative or NaN fraction is clamped, not propagated.
    assert ts.treg_cytolytic_attenuation(-0.5) == 1.0
    assert ts.treg_cytolytic_attenuation(float('nan')) == 1.0
    # And a repertoire with no pre-exhaustion is identical to one built
    # before the parameter existed: same clones, no draw taken.
    plain = TCRRepertoire(size=100, seed=3)
    none_ = TCRRepertoire(size=100, seed=3, initial_exhausted_fraction=None)
    zero = TCRRepertoire(size=100, seed=3, initial_exhausted_fraction=0.0)
    assert [t.cdr3 for t in plain.tcrs] == [t.cdr3 for t in none_.tcrs]
    assert [t.cdr3 for t in plain.tcrs] == [t.cdr3 for t in zero.tcrs]
    assert (plain.exhausted_count(), none_.exhausted_count(),
            zero.exhausted_count()) == (0, 0, 0)


def test_pre_exhaustion_does_not_disturb_the_tcr_sequences():
    """The draw has its OWN generator. Reusing the repertoire's would be
    safe today -- it is read only by _generate, which has already run --
    but would couple the TCR sequences to this fraction the moment
    anything else drew from it, and the clones are what recognition is
    built on."""
    plain = TCRRepertoire(size=200, seed=7)
    loaded = TCRRepertoire(size=200, seed=7, initial_exhausted_fraction=0.4)
    assert [t.cdr3 for t in plain.tcrs] == [t.cdr3 for t in loaded.tcrs]
    assert [t.clone_frequency for t in plain.tcrs] == [
        t.clone_frequency for t in loaded.tcrs]
    # Exactly round(f * size), not a binomial draw around it: the input
    # is a measured fraction and should not come back as a sample.
    assert loaded.exhausted_count() == 80
    again = TCRRepertoire(size=200, seed=7, initial_exhausted_fraction=0.4)
    assert ([s for s in again._states.values()]
            == [s for s in loaded._states.values()])


# ── part 3: the effect is visible where the headroom said it is ──────

def test_the_treg_term_scales_the_per_contact_kill_probability_exactly():
    """The headroom measurement said no contact is near zero (smallest
    p_kill in 53,388 contacts: 0.007775, none below 0.001), so a factor
    on the path is not absorbed. It is applied exactly: the mean
    per-contact kill probability over a 3-seed arena is the unsuppressed
    0.08635884 times the Treg multiplier, to 8 decimal places.

      treg_fraction   multiplier   mean p_kill
         0.0           1.000000     0.08635884
         0.1           0.960317     0.08293190
         0.2           0.910714     0.07864823
         0.3           0.846939     0.07314065
         0.5           0.642857     0.05551640
         0.737         0.000000     0.00000000
    """
    seeds = (1, 2, 3)
    base = _totals(seeds)
    assert base['mean_p'] == pytest.approx(0.08635884, abs=1e-7)
    for f, mult, mean_p in ((0.1, 0.960317, 0.08293190),
                            (0.2, 0.910714, 0.07864823),
                            (0.3, 0.846939, 0.07314065),
                            (0.5, 0.642857, 0.05551640),
                            (0.737, 0.0, 0.0)):
        got = _totals(seeds, treg_fraction=f)
        assert got['treg_mult'] == pytest.approx(mult, abs=1e-6)
        assert got['mean_p'] == pytest.approx(mean_p, abs=1e-7)
        # Exactness against the UNROUNDED multiplier: the table above is
        # printed to six places, and 0.960317 vs 0.9603174603 is itself
        # a 4e-8 difference on this mean.
        assert got['mean_p'] == pytest.approx(
            base['mean_p'] * got['treg_mult'], abs=1e-12)


@pytest.mark.timeout(600)
def test_the_treg_term_changes_the_outcome_not_only_the_probability():
    """And it reaches the kills. 12 seeds of a 12 h arena, 20 T cells
    against 20 cancer cells, which is the smallest arena that resolves
    the lever above the Poisson noise on 69 kills:

      treg_fraction   multiplier   kills   perforin hits   contacts
         0.0           1.000000      69         339          8,538
         0.3           0.846939      59         316          9,275
         0.5           0.642857      53         264          9,283
         0.737         0.000000       0           0         10,515

    Contacts RISE as killing falls, which is the feedback that makes
    this an outcome and not an arithmetic restatement: suppressed T
    cells leave more live targets to keep engaging.

    0.737 is the Treg fraction at which the Treg-per-effector ratio
    reaches Chen 2005's measured 2.8, where that paper reports specific
    cytotoxicity abolished -- so zero kills is the sourced endpoint, not
    a crash.
    """
    seeds = tuple(range(1, 13))
    rows = {f: _totals(seeds, hours=12.0, treg_fraction=f)
            for f in (0.0, 0.3, 0.5, 0.737)}
    assert rows[0.0]['kills'] == 69
    assert rows[0.3]['kills'] == 59
    assert rows[0.5]['kills'] == 53
    assert rows[0.737]['kills'] == 0
    assert [rows[f]['hits'] for f in (0.0, 0.3, 0.5, 0.737)] == [
        339, 316, 264, 0]
    # Monotone in the Treg load, and the contact count moves the other
    # way, so the kill fall is not just fewer tries.
    kills = [rows[f]['kills'] for f in (0.0, 0.3, 0.5, 0.737)]
    assert kills == sorted(kills, reverse=True)
    contacts = [rows[f]['contacts'] for f in (0.0, 0.3, 0.5, 0.737)]
    assert contacts == sorted(contacts)


@pytest.mark.timeout(300)
def test_pre_exhaustion_reaches_a_24_hour_run_where_the_clock_cannot():
    """Exhaustion was measured to be exactly 1.0 at every one of the
    53,388 instrumented contacts, because EXHAUSTION_ONSET_HOURS is 10
    days and no preset runs that long. So a patient measured to have an
    exhausted infiltrate reached no run at all. This key is the only
    route by which that measurement can, and it adds no new constant:
    the fraction is the patient's and the 0.1 residual cytotoxicity was
    already sourced.

    3 seeds of a 20-T-cell 6 h arena: 0 exhausted contacts and 10 kills
    without it; every contact exhausted, 4 perforin hits and 0 kills
    with it, and the per-contact probability is the unsuppressed one
    times exactly EXHAUSTED_KILL_MULTIPLIER.
    """
    assert EXHAUSTION_ONSET_HOURS == 240.0        # 10 days
    seeds = (1, 2, 3)
    base = [arena(seed=s) for s in seeds]
    assert sum(o['exhausted_contacts'] for o in base) == 0
    assert sum(o['kills'] for o in base) == 10

    loaded = [arena(seed=s, t_cell_initial_exhaustion=1.0) for s in seeds]
    assert sum(o['clones_exhausted'] for o in loaded) == 600   # 200 x 3
    assert sum(o['exhausted_contacts'] for o in loaded) == 1335
    assert sum(o['kills'] for o in loaded) == 0
    assert sum(o['perforin_hits'] for o in loaded) == 4
    assert (float(np.mean([o['mean_p'] for o in loaded]))
            == pytest.approx(float(np.mean([o['mean_p'] for o in base]))
                             * EXHAUSTED_KILL_MULTIPLIER, abs=1e-9))


def test_a_pre_exhausted_fraction_is_a_coin_flip_on_the_dominant_clone():
    """FOUND, not fixed, and the reason this key is a population dose
    the engine can only realise per-run as all-or-nothing.

    ``best_match`` returns the single highest-affinity clone for a
    presentation, so every T cell recognising that peptide funnels to
    ONE clone -- the same funnelling the encounter-count rule was
    already documented to suffer from (see EXHAUSTION_ONSET_HOURS's
    comment). A fraction phi of clones pre-exhausted therefore gives
    roughly a phi chance that the run's one relevant clone is
    exhausted, and within a single run the answer is binary: at phi 0.5
    and phi 1.0 the arena above gives identical results, 1,335
    exhausted contacts and 4 perforin hits, to the hit.

    Across 12 repertoire seeds -- which is what moves which clone is
    dominant -- it does track phi:

      phi    dominant clone exhausted in
      0.0            0 / 12
      0.1            1 / 12
      0.3            4 / 12
      0.5            4 / 12
      0.9           11 / 12
      1.0           12 / 12

    So the key is honest as a cohort-level input and must not be read
    as a graded per-run severity. Fixing it means making recognition
    draw a clone by clonal frequency instead of taking the argmax,
    which changes the flagship closed loop and is not this change.
    """
    def dominant_exhausted(phi, tcr_seed):
        rep = TCRRepertoire(size=200, recognition_threshold=0.0,
                            seed=tcr_seed, initial_exhausted_fraction=phi)
        match = rep.best_match(_presentations(['HLA-A*02:01'])[0])
        return bool(match.is_exhausted)

    counts = {phi: sum(dominant_exhausted(phi, s) for s in range(12))
              for phi in (0.0, 0.1, 0.3, 0.5, 0.9, 1.0)}
    assert counts == {0.0: 0, 0.1: 1, 0.3: 4, 0.5: 4, 0.9: 11, 1.0: 12}
    # Monotone across repertoires, binary within one.
    vals = [counts[p] for p in (0.0, 0.1, 0.3, 0.5, 0.9, 1.0)]
    assert vals == sorted(vals)
    assert counts[0.0] == 0 and counts[1.0] == 12


# ── part 4: the costimulation route, and why it was not taken ────────

def test_the_icb_rescue_term_makes_the_costimulation_route_unusable():
    """FOUND, not fixed. The brief preferred routing a Treg through the
    kill rule's existing costimulation input, which would need no new
    multiplier at all. The measurement ruled it out, and the reason is a
    defect in the existing rule rather than anything about Tregs.

    ``kill_probability`` adds an ICB-rescue term
    ``(1 - costimulation) * checkpoint_block * 0.5``, which is LARGEST
    at zero costimulation. Since affinity x MHC-I measures 0.2755 in a
    real contact and the rescue slope is 0.5, the combined signal starts
    RISING as costimulation falls once ``checkpoint_block`` passes
    0.551. Projected over the measured contacts, a Treg routed through
    costimulation gives -27.4% expected perforin hits in Baseline
    (checkpoint_block 0) and +29.5% in the Immunotherapy preset
    (checkpoint_block 1.0): under blockade the model would make an
    anergic T cell kill BETTER than a costimulated one, and a Treg would
    read as a pro-immune intervention.

    The literature says which way round it should be. PD-1 "suppresses
    T cell function primarily by inactivating CD28 signaling" (Hui 2017,
    PMID 28280247) and "Rescue of exhausted CD8 T cells by PD-1-targeted
    therapies is CD28-dependent", with "a cell-intrinsic requirement of
    CD28 for CD8 T cell proliferation after PD-1 blockade" (Kamphorst
    2017, PMID 28280249). ICB rescue should scale WITH costimulation.

    Rewriting that term changes the flagship Immunotherapy preset, so
    it is recorded here and the costimulation route stays unused.
    """
    a, m = MEASURED_AFFINITY, MEASURED_MHC1
    # Without blockade, falling costimulation falls the kill probability,
    # which is the direction a Treg needs.
    open_full = kill_probability(a, m, COSTIM, 0.0)
    open_none = kill_probability(a, m, 0.0, 0.0)
    assert open_full > open_none
    assert open_none == 0.0

    # Under full blockade the sign flips: no costimulation kills better.
    blocked_full = kill_probability(a, m, COSTIM, 1.0)
    blocked_none = kill_probability(a, m, 0.0, 1.0)
    assert blocked_none > blocked_full, (blocked_none, blocked_full)
    assert blocked_none / blocked_full > 1.25

    # The crossover is where affinity x MHC-I meets the rescue slope.
    crossover = a * m / 0.5
    assert crossover == pytest.approx(0.5512, abs=1e-3)
    assert kill_probability(a, m, 0.0, crossover - 0.05) < kill_probability(
        a, m, COSTIM, crossover - 0.05)
    assert kill_probability(a, m, 0.0, crossover + 0.05) > kill_probability(
        a, m, COSTIM, crossover + 0.05)

    # The route this module took instead is blockade-independent, so it
    # cannot invert: the Treg multiplier is a plain factor on the path.
    for block in (0.0, 0.5, 1.0):
        p0 = kill_probability(a, m, COSTIM, block)
        for f in (0.1, 0.3, 0.5):
            mult = ts.treg_cytolytic_attenuation(f)
            assert p0 * mult < p0


def test_the_hill_rule_sits_below_its_ec50_in_a_real_contact():
    """Why the costimulation route would have been a big lever if the
    rescue term were right, and why the measurement had to be taken
    before choosing: at the measured mean signal of 0.1654 against the
    rule's EC50 of 0.30, the Hill is in its quartic region, so
    d(ln p)/d(ln costim) is 3.66 at the mean inputs (3.61 as the mean
    of the per-contact elasticities, which is the quantity the runs
    recorded) -- a 10% costimulation cut would be a 31%
    kill-probability cut. Under full blockade the same elasticity
    measures 1.26, a 2.9x difference between presets."""
    a, m = MEASURED_AFFINITY, MEASURED_MHC1
    signal = a * m * COSTIM
    assert signal == pytest.approx(MEASURED_SIGNAL, abs=5e-4)
    assert signal < 0.30                          # below the Hill's EC50
    p = kill_probability(a, m, COSTIM, 0.0)
    elasticity = 4.0 * (1.0 - p)
    assert elasticity == pytest.approx(3.662, abs=0.01)
    p_blocked = kill_probability(a, m, COSTIM, 1.0)
    assert 4.0 * (1.0 - p_blocked) == pytest.approx(1.257, abs=0.01)
    assert elasticity / (4.0 * (1.0 - p_blocked)) == pytest.approx(
        2.91, abs=0.05)


# ── part 5: sensitivity to each new constant ─────────────────────────

def test_sensitivity_to_the_two_new_constants():
    """Both new numbers, over their stated brackets.

    ``treg_ratio_full_suppression`` is 2.8 with bracket 0.5-4.0. 1.6-4.0
    is Chen 2005's 2.8 +/- 1.2 Treg per CD8 and its own SD, which is an
    uncertainty on the OBSERVED ratio; the 0.5 low end is an ESTIMATE of
    how early saturation could occur, and it is there because Chen's
    design is presence/absence -- it shows cytotoxicity abolished AT 2.8
    and cannot locate where abolition begins, so an SD on the
    observation cannot bound a threshold. ``treg_max_cytolytic_loss`` is
    1.0 with low end 0.5, because "abolished" is a figure-level readout.

      multiplier at treg_fraction    0.1      0.2      0.3      0.5
      ratio_full 0.5 (ESTIMATE low) 0.7778   0.5000   0.1429   0.0000
      ratio_full 1.6 (paper -1 SD)  0.9306   0.8438   0.7321   0.3750
      ratio_full 2.8 (default)      0.9603   0.9107   0.8469   0.6429
      ratio_full 4.0 (paper +1 SD)  0.9722   0.9375   0.8929   0.7500
      max_loss   0.5 (low)          0.9802   0.9554   0.9235   0.8214
      max_loss   1.0 (default)      0.9603   0.9107   0.8469   0.6429

    At the repository's own HIGH_TREG threshold of 0.3 the paper's own
    SD spans 0.732 to 0.893, a factor of 1.22 on the retained
    multiplier; the FULL declared bracket spans 0.143 to 0.893, a factor
    of 6.25. So the ratio constant dominates this model, and its
    uncertainty is a modelling choice rather than a measurement --
    which is exactly why its status is 'derived' and not 'sourced'.
    max_loss over 0.5-1.0 spans 0.923 to 0.847 at the same point, a
    factor of 1.09.
    """
    want = {
        (0.5, 'ratio'): {0.1: 0.777778, 0.2: 0.500000,
                         0.3: 0.142857, 0.5: 0.000000},
        (1.6, 'ratio'): {0.1: 0.930556, 0.2: 0.843750,
                         0.3: 0.732143, 0.5: 0.375000},
        (2.8, 'ratio'): {0.1: 0.960317, 0.2: 0.910714,
                         0.3: 0.846939, 0.5: 0.642857},
        (4.0, 'ratio'): {0.1: 0.972222, 0.2: 0.937500,
                         0.3: 0.892857, 0.5: 0.750000},
        (0.5, 'loss'): {0.1: 0.980159, 0.2: 0.955357,
                        0.3: 0.923469, 0.5: 0.821429},
        (1.0, 'loss'): {0.1: 0.960317, 0.2: 0.910714,
                        0.3: 0.846939, 0.5: 0.642857},
    }
    for (v, kind), row in want.items():
        for f, expected in row.items():
            kw = ({'ratio_full': v} if kind == 'ratio' else {'max_loss': v})
            assert ts.treg_cytolytic_attenuation(f, **kw) == pytest.approx(
                expected, abs=1e-6), (v, kind, f)

    # The declared brackets are the ones the sensitivity was run over.
    assert (ts.TREG_RATIO_FULL_SUPPRESSION.value,
            ts.TREG_RATIO_FULL_SUPPRESSION.low,
            ts.TREG_RATIO_FULL_SUPPRESSION.high) == (2.8, 0.5, 4.0)
    assert ts.TREG_MAX_CYTOLYTIC_LOSS.value == 1.0
    assert ts.TREG_MAX_CYTOLYTIC_LOSS.low == 0.5

    # Spread at the classifier's own suppression threshold, over the
    # paper's SD and over the whole declared bracket.
    at_03 = [ts.treg_cytolytic_attenuation(0.3, ratio_full=r)
             for r in (1.6, 4.0)]
    assert at_03[1] / at_03[0] == pytest.approx(1.219, abs=0.01)
    full = [ts.treg_cytolytic_attenuation(0.3, ratio_full=r)
            for r in (0.5, 4.0)]
    assert full[1] / full[0] == pytest.approx(6.25, abs=0.01)
    loss_03 = [ts.treg_cytolytic_attenuation(0.3, max_loss=m)
               for m in (1.0, 0.5)]
    assert loss_03[1] / loss_03[0] == pytest.approx(1.090, abs=0.01)


def test_the_ratio_is_derived_and_not_sourced_and_the_source_says_why():
    """Chen 2005's 2.8 is real and verified verbatim, but the experiment
    is presence/absence: it shows cytotoxicity abolished AT a ratio that
    happened to be 2.8, not that abolition requires 2.8. Saturation
    could sit anywhere in (0, 2.8], and the module's own shape note says
    the same thing about the curve ("Chen 2005 measured TWO points").
    An SD on an observed ratio is not an uncertainty on a threshold, so
    the status cannot be 'sourced'.

    Its provenance has to be stated too, because it is the one constant
    here that is not re-retrievable: efetch on 15623559 returns an
    abstract with no ratio in it, efetch db=pmc id=PMC544311 answers
    that the publisher does not allow full-text download, and the
    Europe PMC fullTextXML endpoint errors. The quote came from the PDF.
    """
    p = ts.TREG_RATIO_FULL_SUPPRESSION
    assert p.status == 'derived'
    assert 'presence/absence' in p.source
    assert 'not re-retrievable' in p.source
    assert '15623559' in p.source
    # The direction the two-point design leaves open is the one that
    # makes this lever large, so the bracket has to reach under 1.0.
    assert p.low < 1.0
    assert ts.treg_cytolytic_attenuation(0.3, ratio_full=p.low) < 0.25


@pytest.mark.timeout(300)
def test_the_outcome_moves_with_each_constant_over_its_bracket():
    """The same sensitivity at the outcome, 3 seeds at treg_fraction 0.3:

      constant                 multiplier   kills   perforin hits
      ratio_full 1.6 (low)      0.732143      10          53
      ratio_full 2.8 (default)  0.846939      11          51
      ratio_full 4.0 (high)     0.892857       8          50
      max_loss   0.5 (low)      0.923469       8          50

    The kill column is 8-11 on about 50 hits, so at this arena size the
    bracket is NOT resolved in kills -- the perforin hits are, and they
    move monotonically with the multiplier. Reporting it the other way
    round would be reading noise.
    """
    seeds = (1, 2, 3)
    rows = {
        'ratio_low': _totals(seeds, treg_fraction=0.3,
                             treg_ratio_full_suppression=1.6),
        'default': _totals(seeds, treg_fraction=0.3),
        'ratio_high': _totals(seeds, treg_fraction=0.3,
                              treg_ratio_full_suppression=4.0),
        'loss_low': _totals(seeds, treg_fraction=0.3,
                            treg_max_cytolytic_loss=0.5),
    }
    assert rows['ratio_low']['treg_mult'] == pytest.approx(0.732143, abs=1e-6)
    assert rows['default']['treg_mult'] == pytest.approx(0.846939, abs=1e-6)
    assert rows['ratio_high']['treg_mult'] == pytest.approx(0.892857, abs=1e-6)
    assert rows['loss_low']['treg_mult'] == pytest.approx(0.923469, abs=1e-6)
    # mean p_kill tracks the multiplier exactly; that is the resolved axis.
    unsuppressed = _totals(seeds)['mean_p']
    for row in rows.values():
        assert row['mean_p'] == pytest.approx(
            unsuppressed * row['treg_mult'], abs=1e-9)
    assert rows['ratio_low']['hits'] == 53
    assert rows['default']['hits'] == 51
    assert rows['ratio_high']['hits'] == 50
    assert rows['loss_low']['hits'] == 50


def test_the_quoted_size_of_the_treg_lever_is_the_multiplier_itself():
    """The docstring of ``treg_cytolytic_attenuation`` quotes the size of
    this lever, and it quoted the WRONG route's numbers until
    2026-10-28: -6.2 / -12.8 / -27.4 / -44.0 / -80.1% on expected
    perforin hits, which is ~3x the implemented term and is in fact the
    projection for the COSTIMULATION route the module rejected.

    Expected hits are EXACTLY linear in this multiplier -- immune_module
    multiplies the Hill probability by it once, and contact_hits has
    mean hits_per_contact * hill_probability -- so the reduction is
    1 - multiplier and nothing else. Pinned here so the quoted magnitude
    cannot drift from the code again.
    """
    want = {0.05: 0.0188, 0.1: 0.0397, 0.2: 0.0893,
            0.3: 0.1531, 0.5: 0.3571}
    for f, loss in want.items():
        assert 1.0 - ts.treg_cytolytic_attenuation(f) == pytest.approx(
            loss, abs=5e-5), f
    # The classifier's own HIGH_TREG threshold: -15.3%, not -44%.
    assert 1.0 - ts.treg_cytolytic_attenuation(0.3) == pytest.approx(
        0.1531, abs=5e-5)
    # Full suppression at the ratio Chen measured, which is the only
    # point where the lever reaches -100%.
    full = ts.TREG_RATIO_FULL_SUPPRESSION.value
    f_full = full / (1.0 + full)
    assert f_full == pytest.approx(0.7368, abs=1e-4)
    assert ts.treg_cytolytic_attenuation(f_full) == pytest.approx(0.0)
    # And the corrected table is the one in the docstring, not the old.
    doc = ts.treg_cytolytic_attenuation.__doc__
    assert '-15.3%' in doc
    assert '-44.0%' not in doc.split('CORRECTED')[0]

    # The costimulation route, recomputed, is where those numbers came
    # from: the Hill is quartic below its EC50, so routing the same
    # occupancy through costimulation amplifies it ~3x.
    base = kill_probability(0.95, MHC1_SEEDED, COSTIM)
    via_costim = {}
    for f in (0.2, 0.3):
        occ = ts.treg_occupancy(f)
        via_costim[f] = 1.0 - kill_probability(
            0.95, MHC1_SEEDED, COSTIM * (1.0 - occ)) / base
    assert via_costim[0.2] == pytest.approx(0.291, abs=0.005)
    assert via_costim[0.3] == pytest.approx(0.460, abs=0.005)
    assert via_costim[0.3] / 0.1531 == pytest.approx(3.0, abs=0.1)


def test_the_producer_only_keys_are_consumed_and_reported():
    """Both keys were emitted by twin_config and read by nothing. The
    comment that stood in immune_module said so: "They stay
    producer-only, and that is a known gap rather than an oversight."

    They are consumed now, and the run REPORTS the realised multiplier
    and clone count rather than only the fraction it was handed --
    because a fraction that reaches nothing looks identical to one that
    does.

    HALF THE GAP, not all of it: see
    test_a_treg_fraction_reaches_a_real_run_but_a_patient_cannot_set_it.
    """
    _, _, immune = _lineup(n_t=2, treg_fraction=0.3,
                           t_cell_initial_exhaustion=0.25)
    state = immune.get_state()
    assert state['treg_fraction'] == 0.3
    assert state['treg_cytolytic_retained'] == pytest.approx(0.846939,
                                                             abs=1e-6)
    assert state['t_cell_initial_exhaustion'] == 0.25
    assert state['n_clones_exhausted'] == 50          # 0.25 x 200
    report = state['treg_calibration']
    assert report['treg_ratio_full_suppression']['status'] == 'derived'
    assert report['occupancy'] == pytest.approx(0.153061, abs=1e-6)

    # Unmeasured stays visible as unmeasured, not as zero.
    _, _, plain = _lineup(n_t=2)
    plain_state = plain.get_state()
    assert plain_state['treg_fraction'] is None
    assert plain_state['treg_cytolytic_retained'] == 1.0
    assert plain_state['t_cell_initial_exhaustion'] is None
    assert plain_state['n_clones_exhausted'] == 0


@pytest.mark.timeout(300)
def test_a_treg_fraction_reaches_a_real_run_but_a_patient_cannot_set_it():
    """Exactly how far the producer-only gap is closed, both halves
    asserted, because "this module closes the gap" would be this
    repository's own dominant defect class worn as a fix.

    CLOSED: a Treg load set on a run reaches the engine and changes the
    realised multiplier. Through EngineRunner, not a hand-built module,
    because every other test here builds ImmuneModule directly and that
    proves nothing about whether a run can carry the key.

    NOT CLOSED: no shipped preset sets either key, and the PRODUCER
    reaches no engine. DigitalTwinConfig.to_simulation_params() emits
    both, and its only two non-test callers are a st.download_button and
    a st.json expander on _pages/28_digital_twin.py;
    PatientEngineInputs.module_configs() -- the one path that injects
    patient data into module configs -- emits neither. So a measured
    patient Treg fraction still has no route to a run, and every
    launchable configuration has treg_mult exactly 1.0.
    """
    from cognisom.dashboard.engine_runner import EngineRunner, SCENARIOS
    from cognisom.dashboard.patient_engine import PatientEngineInputs
    from cognisom.genomics.twin_config import DigitalTwinConfig

    # CLOSED: the key reaches the engine through a run's immune config.
    np.random.seed(5)
    runner = EngineRunner(
        tissue='ring', dt=0.05, duration=0.1, scenario='Baseline',
        record_interval=1000, snapshot_interval_hours=None,
        overrides={'immune': {'treg_fraction': 0.3,
                              't_cell_initial_exhaustion': 0.25}})
    with contextlib.redirect_stdout(io.StringIO()):
        runner.build()
    state = runner.engine.modules['immune'].get_state()
    assert state['treg_fraction'] == 0.3
    assert state['treg_cytolytic_retained'] == pytest.approx(0.846939,
                                                             abs=1e-6)
    assert state['n_clones_exhausted'] > 0

    # NOT CLOSED: no preset carries either key, so every shipped run is
    # the provable no-op.
    for name, preset in SCENARIOS.items():
        immune = preset.get('immune', {})
        assert 'treg_fraction' not in immune, name
        assert 't_cell_initial_exhaustion' not in immune, name
    np.random.seed(5)
    shipped = EngineRunner(
        tissue='ring', dt=0.05, duration=0.1, scenario='Baseline',
        record_interval=1000, snapshot_interval_hours=None)
    with contextlib.redirect_stdout(io.StringIO()):
        shipped.build()
    assert shipped.engine.modules['immune'].get_state()[
        'treg_cytolytic_retained'] == 1.0

    # NOT CLOSED: the producer emits both and the patient path drops them.
    emitted = DigitalTwinConfig(
        immune_profile_measured=True, treg_fraction=0.3,
        t_cell_exhaustion_fraction=0.25).to_simulation_params()['immune']
    assert emitted['treg_fraction'] == 0.3
    assert emitted['t_cell_initial_exhaustion'] == 0.25
    patient_immune = PatientEngineInputs(
        patient_id='x', hla_alleles=['HLA-A*02:01'], hla_measured=True,
        hla_method='test').module_configs()['immune']
    assert 'treg_fraction' not in patient_immune, (
        'the patient path now carries treg_fraction -- the gap is '
        'closed end to end, so say so and re-pin this test')
    assert 't_cell_initial_exhaustion' not in patient_immune


def test_twin_config_emits_the_fraction_and_not_the_mean_score():
    """The units defect this phase found in the producer.

    ``to_simulation_params`` emitted ``mean_exhaustion`` under the key
    ``t_cell_initial_exhaustion``. mean_exhaustion is the MEAN of
    per-cell exhaustion SCORES (cell_state_classifier line 159,
    ``np.mean(exhaustion_scores)``), and a mean score is not a fraction
    of cells: a cohort where every T cell scores 0.4 and one where 40%
    score 1.0 and 60% score 0.0 both give 0.4, and only the second has
    exhausted cells in it. The quantity that IS a fraction is
    ``exhausted_fraction`` -- "Fraction with score > 0.6" -- which the
    config already carried as ``t_cell_exhaustion_fraction`` and did
    not emit. Nothing caught it because the key had no consumer.
    """
    from cognisom.genomics.twin_config import DigitalTwinConfig
    twin = DigitalTwinConfig(immune_profile_measured=True,
                      mean_exhaustion=0.42,
                      t_cell_exhaustion_fraction=0.13)
    emitted = twin.to_simulation_params()['immune']
    assert emitted['t_cell_initial_exhaustion'] == 0.13
    assert emitted['t_cell_initial_exhaustion'] != twin.mean_exhaustion

    # Unmeasured still routes None, not 0.0: a tumour measured to have
    # no exhausted T cells and one nobody looked at are different.
    unmeasured = DigitalTwinConfig()
    assert (unmeasured.to_simulation_params()['immune']
            ['t_cell_initial_exhaustion'] is None)
