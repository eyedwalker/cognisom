"""Macrophage M1/M2 polarisation in the simulation.

Grafted from the macrophage arm of the second independent design. The
NK arm of this consolidation is A1's per-clone KIR-HLA receptor balance
and is tested in tests/test_nk_receptor_balance.py; the NK tests that
accompanied this module's original have been dropped rather than
duplicated, because they exercised a different NK API.

What is claimed here is the GATING -- that a macrophage's phenotype
decides its behaviour and that an M2 compartment suppresses T cells --
not a cytokine concentration field. There is no diffusion solver; local
signals are counted from neighbouring cells on demand, in the manner of
spatial/ecm_barrier.ecm_density_at.

Immunology relied on:
  Mills CD et al., J Immunol 2000;164:6166-73       M1/M2 axis
  Mantovani A et al., Trends Immunol 2002;23:549-55  drivers
  Mosser DM & Edwards JP, Nat Rev Immunol 2008;8:958-69
  Qian BZ & Pollard JW, Cell 2010;141:39-51          M2/TAM promotion
  Colegio OR et al., Nature 2014;513:559-63          lactate -> M2
  Henze AT & Mazzone M, J Clin Invest 2016;126:3672-9 hypoxia -> M2
  Rodriguez PC et al., Cancer Res 2004;64:5839-49    arginase-1 -> T-cell arrest
  Bronte V & Zanovello P, Nat Rev Immunol 2005;5:641-54
  Oyler-Yaniv A et al., Immunity 2017;46:609-20      paracrine range (30 um)
"""

import math

import numpy as np
import pytest

from cognisom.core import SimulationConfig, SimulationEngine
from cognisom.core.event_bus import EventTypes
from cognisom.engine.py.immune.macrophage_polarization import (
    DEFAULT_ARGINASE_MAX_SUPPRESSION,
    DEFAULT_KILL_SPAN,
    DEFAULT_TAU_HOURS,
    M0,
    M1,
    M2,
    arginase_attenuation,
    cytokine_output,
    hypoxia_signal,
    initial_polarization,
    kill_factor,
    local_signal,
    polarization_label,
    polarization_step,
    polarization_target,
)
from cognisom.modules.cellular_module import CellularModule
from cognisom.modules.immune_module import ImmuneModule
from cognisom.modules.molecular_module import MolecularModule

def test_kill_factor_is_exactly_one_for_an_unpolarized_macrophage():
    # This exactness is the whole reason the default behaviour is
    # unchanged by the new field.
    assert kill_factor(M0) == 1.0
def test_m1_is_more_tumoricidal_than_m0_than_m2():
    assert kill_factor(M1) > kill_factor(M0) > kill_factor(M2)
    assert kill_factor(M1) == pytest.approx(1.0 + DEFAULT_KILL_SPAN)
    assert kill_factor(M2) == pytest.approx(1.0 - DEFAULT_KILL_SPAN)
def test_kill_factor_never_goes_negative():
    assert kill_factor(M2, span=5.0) == 0.0
def test_arginase_is_exactly_one_with_no_m2_burden():
    # The property that makes the T-cell gate a provable no-op in a
    # configuration with no macrophages.
    assert arginase_attenuation(0.0) == 1.0
def test_arginase_is_bounded_by_its_max_suppression():
    assert arginase_attenuation(1.0) == pytest.approx(
        1.0 - DEFAULT_ARGINASE_MAX_SUPPRESSION)
    for b in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert 1.0 - DEFAULT_ARGINASE_MAX_SUPPRESSION <= (
            arginase_attenuation(b)) <= 1.0
def test_arginase_is_monotone_decreasing_in_m2_burden():
    prev = 1.1
    for i in range(11):
        v = arginase_attenuation(i / 10.0)
        assert v < prev
        prev = v
def test_arginase_composes_multiplicatively_with_the_sympathetic_gate():
    # Both are final multiplicative gates on the same path, so their
    # composition must be order-independent and bounded by each.
    from cognisom.engine.py.immune.sympathetic import sympathetic_attenuation
    a = arginase_attenuation(0.5)
    s = sympathetic_attenuation(0.5)
    assert a * s == s * a
    assert a * s < min(a, s)
def test_ifn_gamma_drives_m1_and_il10_tgfb_hypoxia_drive_m2():
    assert polarization_target(ifn_gamma=1.0) == M1
    assert polarization_target(il10=1.0) == M2
    assert polarization_target(tgf_beta=1.0) == M2
    assert polarization_target(hypoxia=1.0) == M2
def test_opposing_signals_of_equal_strength_cancel_to_m0():
    assert polarization_target(ifn_gamma=1.0, il10=1.0) == pytest.approx(M0)
    assert polarization_target(ifn_gamma=0.5, hypoxia=0.5) == pytest.approx(M0)
def test_no_signal_leaves_the_target_at_m0():
    assert polarization_target() == M0
def test_the_m2_arm_saturates_rather_than_exceeding_the_axis():
    assert polarization_target(il10=1.0, tgf_beta=1.0, hypoxia=1.0) == M2
def test_polarization_relaxes_toward_its_target_on_a_24_hour_constant():
    # One time constant of explicit Euler at dt = tau/2400 should land
    # within a fraction of a percent of the analytic 1 - exp(-1).
    p = M0
    steps = 2400
    for _ in range(steps):
        p = polarization_step(p, M1, DEFAULT_TAU_HOURS / steps)
    analytic = -(1.0 - math.exp(-1.0))
    assert p == pytest.approx(analytic, abs=1e-3)
def test_a_two_hour_run_moves_a_macrophage_only_a_few_percent():
    # Documenting the real timescale: M1<->M2 repolarization takes
    # about a day, so the mechanism is nearly invisible at the 0.5-4 h
    # durations the demo scripts use. If it needs to show in a demo,
    # the run duration is what should change, not the time constant.
    p = M0
    for _ in range(200):          # 200 steps x 0.01 h = 2 h
        p = polarization_step(p, M1, 0.01)
    assert 0.07 < abs(p) < 0.09
def test_polarization_step_cannot_overshoot_its_target():
    assert polarization_step(M0, M1, 100.0) == M1
    assert polarization_step(M0, M2, DEFAULT_TAU_HOURS * 2) == M2
def test_polarization_stays_on_the_axis():
    p = M0
    for _ in range(5000):
        p = polarization_step(p, M2, 1.0)
        assert M1 <= p <= M2
def test_a_zero_length_step_changes_nothing():
    assert polarization_step(0.3, M1, 0.0) == 0.3
def test_polarization_labels_match_the_transcriptomic_classifier():
    # The simulated axis and the marker-score axis must use the same
    # sign convention and the same cutoffs, or a simulated M1 and a
    # classified M1 would not be the same statement. This asserts
    # against cell2sentence's own function, not a copy of its numbers.
    from cognisom.genomics.cell2sentence import Cell2SentenceModel
    scorer = Cell2SentenceModel._score_polarization_markers
    # M1 at -1, mixed at 0, M2 at +1 -- cell2sentence returns
    # "score: -1 = pure M1, +1 = pure M2" with +/- 0.3 cutoffs.
    assert polarization_label(M1) == "M1"
    assert polarization_label(M0) == "mixed"
    assert polarization_label(M2) == "M2"
    assert polarization_label(-0.31) == "M1"
    assert polarization_label(-0.29) == "mixed"
    assert polarization_label(0.31) == "M2"
    assert polarization_label(0.29) == "mixed"
    # And the classifier agrees on a pure-M1 and a pure-M2 sentence.
    from cognisom.genomics.cell2sentence import M1_MARKERS, M2_MARKERS
    m1_label, m1_score = scorer(" ".join(M1_MARKERS))
    m2_label, m2_score = scorer(" ".join(M2_MARKERS))
    assert (m1_label, m2_label) == ("M1", "M2")
    assert m1_score < 0 < m2_score
    assert polarization_label(m1_score) == "M1"
    assert polarization_label(m2_score) == "M2"
def test_cytokine_output_matches_the_entity_library_rows():
    # M1 -> TNFa / IL1b / IL6 / IL12 / NO; M2 -> IL10 / TGFb. These
    # are the curated cytokines_secreting fields on the library's
    # "Macrophage M1" and "Macrophage M2" rows.
    m1 = cytokine_output(M1)
    m2 = cytokine_output(M2)
    for species in ("TNF_alpha", "IL1_beta", "IL6", "IL12", "NO"):
        assert m1[species] == 1.0
        assert m2[species] == 0.0
    for species in ("IL10", "TGF_beta1", "ARG1"):
        assert m2[species] == 1.0
        assert m1[species] == 0.0
def test_an_unpolarized_macrophage_secretes_nothing():
    assert set(cytokine_output(M0).values()) == {0.0}
def test_cytokine_output_is_graded_not_binary():
    half = cytokine_output(0.5)
    assert half["IL10"] == pytest.approx(0.5)
    assert half["TNF_alpha"] == 0.0
def test_initial_polarization_of_an_unmeasured_profile_is_m0():
    assert initial_polarization(None) == M0
def test_initial_polarization_seeds_the_compartment_mean():
    # A population fraction constrains the mean, not any individual
    # cell. f = 0.5 must give exactly M0, so a balanced compartment
    # and an unmeasured one start from the same place.
    assert initial_polarization(0.5) == M0
    assert initial_polarization(1.0) == M2
    assert initial_polarization(0.0) == M1
    assert initial_polarization(0.75) == pytest.approx(0.5)
def test_hypoxia_is_zero_at_normoxia_and_one_at_anoxia():
    assert hypoxia_signal(0.21) == 0.0
    assert hypoxia_signal(0.5) == 0.0        # hyperoxic is not hypoxic
    assert hypoxia_signal(0.0) == 1.0
    assert hypoxia_signal(0.105) == pytest.approx(0.5)
def test_missing_oxygen_data_is_not_hypoxia():
    assert hypoxia_signal(None) == 0.0
    assert hypoxia_signal(float("nan")) == 0.0
def test_local_signal_is_exactly_zero_with_nothing_nearby():
    # No epsilon left over: zero secreting cells means zero signal,
    # which is what makes the arginase no-op exact.
    assert local_signal(0) == 0.0
    assert local_signal(1, saturation_count=3.0) == pytest.approx(1 / 3)
    assert local_signal(3, saturation_count=3.0) == 1.0
    assert local_signal(99, saturation_count=3.0) == 1.0
def _immune(**config):
    cfg = {'n_t_cells': 0, 'n_nk_cells': 0, 'n_macrophages': 0}
    cfg.update(config)
    module = ImmuneModule(cfg)
    module.initialize()
    return module
class _Target:
    """A minimal stand-in for a CellState."""

    def __init__(self, mhc1=1.0, mutations=None, oxygen=0.21, position=None):
        self.cell_id = 999
        self.mhc1_expression = mhc1
        self.mutations = mutations or []
        self.oxygen = oxygen
        self.alive = True
        self.cell_type = 'cancer'
        self.position = np.array(
            position if position is not None else [100.0, 100.0, 50.0],
            dtype=np.float32)
        self.mhc1_displayed_peptides = []
        self.pdl1_expression = 0.0
        self.local_ecm_density = 0.0
def test_the_two_innate_lineages_default_to_the_shared_kill_probability():
    module = _immune()
    assert module.nk_kill_probability == module.kill_probability == 0.8
    assert module.macrophage_kill_probability == 0.8
def test_the_shared_kill_probability_still_moves_both_lineages():
    module = _immune(kill_probability=0.5)
    assert module.nk_kill_probability == 0.5
    assert module.macrophage_kill_probability == 0.5
def test_each_lineage_can_now_be_set_independently():
    module = _immune(nk_kill_probability=0.1, macrophage_kill_probability=0.9)
    assert module.nk_kill_probability == 0.1
    assert module.macrophage_kill_probability == 0.9
    assert module.kill_probability == 0.8      # the shared default is intact
def test_an_unpolarized_macrophage_still_recognizes_every_target():
    # The rule this replaced was `return True`. At M0 the new rule
    # must give the same answer for every target.
    module = _immune(n_macrophages=1)
    mac = module.immune_cells[next(iter(module.immune_cells))]
    assert mac.polarization == M0
    for mhc in (0.0, 0.4, 1.0):
        assert module._recognize(mac, _Target(mhc1=mhc))[0] is True
def test_macrophage_recognition_now_reads_state_instead_of_returning_true():
    module = _immune(n_macrophages=1)
    mac = module.immune_cells[next(iter(module.immune_cells))]
    mac.polarization = M2
    # An M2 macrophage is pro-tumour and stops attacking.
    assert module._recognize(mac, _Target())[0] is False
    mac.polarization = M1
    assert module._recognize(mac, _Target())[0] is True
def test_an_m1_macrophage_kills_more_than_an_m2_one():
    module = _immune(n_macrophages=2, macrophage_kill_probability=0.5)
    ids = sorted(module.immune_cells)
    m1_cell, m2_cell = (module.immune_cells[i] for i in ids[:2])
    m1_cell.polarization = M1
    m2_cell.polarization = M2
    p1 = module._target_kill_probability(m1_cell, _Target())
    p2 = module._target_kill_probability(m2_cell, _Target())
    assert p1 > 0.5 > p2
    assert p1 == pytest.approx(0.5 * (1.0 + DEFAULT_KILL_SPAN))
    assert p2 == pytest.approx(0.5 * (1.0 - DEFAULT_KILL_SPAN))
def test_macrophage_kill_probability_stays_a_probability():
    module = _immune(n_macrophages=1, macrophage_kill_probability=0.9)
    mac = module.immune_cells[next(iter(module.immune_cells))]
    mac.polarization = M1
    assert module._target_kill_probability(mac, _Target()) <= 1.0
def test_an_unpolarized_macrophage_keeps_the_baseline_probability():
    module = _immune(n_macrophages=1, macrophage_kill_probability=0.37)
    mac = module.immune_cells[next(iter(module.immune_cells))]
    assert module._target_kill_probability(mac, _Target()) == 0.37
def test_the_arginase_gate_is_a_provable_noop_with_no_macrophages():
    # Every immune configuration in this suite, and the flagship
    # closed-loop configuration, sets n_macrophages = 0.
    module = _immune(n_t_cells=1)
    tcell = module.immune_cells[next(iter(module.immune_cells))]
    assert module._local_m2_burden(tcell.position) == 0.0
    assert arginase_attenuation(0.0) == 1.0
def test_unpolarized_macrophages_impose_no_arginase_burden():
    module = _immune(n_t_cells=1, n_macrophages=8)
    tcell = next(c for c in module.immune_cells.values()
                 if c.cell_type == 'T_cell')
    for cell in module.immune_cells.values():
        cell.position = tcell.position.copy()
    assert module._local_m2_burden(tcell.position) == 0.0
def test_m2_macrophages_suppress_a_colocated_t_cell():
    module = _immune(n_t_cells=1, n_macrophages=3)
    tcell = next(c for c in module.immune_cells.values()
                 if c.cell_type == 'T_cell')
    for cell in module.immune_cells.values():
        cell.position = tcell.position.copy()
        if cell.cell_type == 'macrophage':
            cell.polarization = M2
    burden = module._local_m2_burden(tcell.position)
    assert burden == 1.0
    assert arginase_attenuation(
        burden, max_suppression=module.arginase_max_suppression
    ) == pytest.approx(1.0 - DEFAULT_ARGINASE_MAX_SUPPRESSION)
def test_the_arginase_gate_multiplies_onto_the_tcell_hill_rule():
    # Measured headroom: an unconditional multiplier on this branch
    # leaves the suite green down to 0.3, so a gate bottoming out at
    # 0.4 is inside the envelope.
    from cognisom.engine.py.immune.tcr_repertoire import TCRRepertoire
    module = _immune(n_t_cells=1, n_macrophages=3,
                     tcr_recognition_threshold=0.0, tcr_repertoire_size=50)
    tcell = next(c for c in module.immune_cells.values()
                 if c.cell_type == 'T_cell')
    repertoire = TCRRepertoire(size=50, recognition_threshold=0.0, seed=0)
    module._tcr_repertoire = repertoire

    class _Match:
        affinity = 0.9
        is_exhausted = False
        tcr = None

    tcell.active_tcr_match = _Match()
    target = _Target(mhc1=0.9)

    # Macrophages far away -> no gate.
    for cell in module.immune_cells.values():
        if cell.cell_type == 'macrophage':
            cell.position = np.array([2000.0, 2000.0, 2000.0],
                                     dtype=np.float32)
            cell.polarization = M2
    p_far = module._target_kill_probability(tcell, target)

    # Macrophages colocated and M2 -> gate bites.
    for cell in module.immune_cells.values():
        if cell.cell_type == 'macrophage':
            cell.position = tcell.position.copy()
    p_near = module._target_kill_probability(tcell, target)

    assert p_near < p_far
    assert p_near == pytest.approx(
        p_far * (1.0 - DEFAULT_ARGINASE_MAX_SUPPRESSION))
def test_polarization_moves_toward_m1_beside_activated_t_cells():
    module = _immune(n_t_cells=3, n_macrophages=1,
                     polarization_tau_hours=1.0)
    mac = next(c for c in module.immune_cells.values()
               if c.cell_type == 'macrophage')
    for cell in module.immune_cells.values():
        cell.position = mac.position.copy()
        if cell.cell_type == 'T_cell':
            cell.activated = True
    assert module._local_ifn_gamma(mac.position, mac.cell_id) == 1.0
    before = mac.polarization
    for _ in range(100):
        module._update_polarization(mac, 0.01)
    assert mac.polarization < before
    assert polarization_label(mac.polarization) in ("M1", "mixed")
def test_resting_t_cells_secrete_no_ifn_gamma():
    module = _immune(n_t_cells=3, n_macrophages=1)
    mac = next(c for c in module.immune_cells.values()
               if c.cell_type == 'macrophage')
    for cell in module.immune_cells.values():
        cell.position = mac.position.copy()
    assert module._local_ifn_gamma(mac.position, mac.cell_id) == 0.0
def test_a_macrophage_does_not_count_its_own_cytokines():
    module = _immune(n_macrophages=1, macrophage_m2_fraction=1.0)
    mac = next(iter(module.immune_cells.values()))
    assert mac.polarization == M2
    # Self-excluded, so an isolated M2 macrophage sees no M2 signal
    # and does not lock itself in.
    assert module._local_m2_secretion(
        mac.position, 'IL10', mac.cell_id) == 0.0
    assert module._local_m2_secretion(mac.position, 'IL10') == pytest.approx(
        1.0 / 3.0)
def test_polarization_moves_toward_m2_under_hypoxia():
    module = _immune(n_macrophages=1, polarization_tau_hours=1.0)
    mac = next(iter(module.immune_cells.values()))

    class _Cellular:
        def __init__(self, pos):
            self.cells = {0: _Target(oxygen=0.0, position=pos)}

        def ecm_density_at(self, position):
            return 0.0

    module.set_cellular_module(_Cellular(mac.position.copy()))
    assert module._local_hypoxia(mac.position) == 1.0
    for _ in range(100):
        module._update_polarization(mac, 0.01)
    assert mac.polarization > 0.0
def test_absent_oxygen_data_does_not_read_as_hypoxia():
    module = _immune(n_macrophages=1)
    mac = next(iter(module.immune_cells.values()))
    assert module.cellular_module is None
    assert module._local_hypoxia(mac.position) == 0.0
def test_the_polarization_drive_can_be_switched_off():
    module = _immune(n_macrophages=1,
                     macrophage_polarization_enabled=False)
    assert module.get_innate_state()[
        'macrophage_polarization_enabled'] is False
def test_polarization_is_reported_unmeasured_by_default():
    # The platform has no measured M1/M2 ratio for any patient: the
    # only runtime path to CellStateClassifier ends at
    # generate_synthetic_expression(), which plants the M2 markers it
    # then rediscovers. 'unmeasured' is the correct answer.
    state = _immune(n_macrophages=4).get_innate_state()
    assert state['macrophage_polarization_source'] == 'unmeasured'
    assert state['macrophage_m2_fraction'] is None
    assert state['immune_profile_measured'] is False
    assert state['mean_macrophage_polarization'] == M0
def test_no_macrophages_reports_an_absent_mean_not_zero():
    state = _immune(n_macrophages=0).get_innate_state()
    assert state['mean_macrophage_polarization'] is None
    assert state['n_macrophages_in_tissue'] == 0
def test_an_m2_fraction_seeds_the_compartment():
    module = _immune(n_macrophages=5, macrophage_m2_fraction=0.8)
    pols = [c.polarization for c in module.immune_cells.values()]
    assert pols == [pytest.approx(0.6)] * 5
def test_an_m2_fraction_without_a_measurement_is_not_called_measured():
    state = _immune(n_macrophages=2,
                    macrophage_m2_fraction=0.8).get_innate_state()
    assert state['macrophage_polarization_source'] == 'configured_unmeasured'
    assert state['immune_profile_measured'] is False
def test_a_declared_measurement_is_reported_as_measured():
    state = _immune(n_macrophages=2, macrophage_m2_fraction=0.8,
                    measured=True).get_innate_state()
    assert state['macrophage_polarization_source'] == 'patient_measured'
def test_twin_config_m2_fraction_plumbing_now_has_a_consumer():
    # to_simulation_params() has emitted immune.m2_fraction and
    # immune.measured all along with no consumer in the repo -- the
    # only reader was st.json() on a dashboard page.
    from cognisom.genomics.twin_config import DigitalTwinConfig
    params = DigitalTwinConfig(
        m2_macrophage_fraction=0.7, immune_profile_measured=True,
    ).to_simulation_params()
    assert params['immune']['m2_fraction'] == 0.7
    assert params['immune']['measured'] is True
    module = ImmuneModule({**params['immune'],
                           'n_t_cells': 0, 'n_nk_cells': 0,
                           'n_macrophages': 2})
    module.initialize()
    state = module.get_innate_state()
    assert state['macrophage_m2_fraction'] == 0.7
    assert state['macrophage_polarization_source'] == 'patient_measured'
    assert state['mean_macrophage_polarization'] == pytest.approx(0.4)
def test_an_unmeasured_twin_profile_stays_unmeasured_through_the_module():
    # twin_config now reports an unmeasured immune profile as None
    # rather than 0. That None must survive into the module as
    # 'unmeasured', not become a polarization of M1.
    from cognisom.genomics.twin_config import DigitalTwinConfig
    params = DigitalTwinConfig().to_simulation_params()
    assert params['immune']['m2_fraction'] is None
    assert params['immune']['measured'] is False
    module = ImmuneModule({**params['immune'], 'n_t_cells': 0,
                           'n_nk_cells': 0, 'n_macrophages': 2})
    module.initialize()
    assert module.get_innate_state()[
        'macrophage_polarization_source'] == 'unmeasured'
    assert module.macrophage_m2_fraction is None
def test_recruited_macrophages_start_from_the_compartment_mean():
    module = _immune(n_macrophages=0, macrophage_m2_fraction=1.0)
    cell_id = module.add_immune_cell([100, 100, 50], 'macrophage')
    assert module.immune_cells[cell_id].polarization == M2
    t_id = module.add_immune_cell([100, 100, 50], 'T_cell')
    assert module.immune_cells[t_id].polarization == M0
def test_get_state_surfaces_the_innate_summary():
    state = _immune(n_macrophages=2).get_state()
    assert state['mean_macrophage_polarization'] == M0
    assert state['nk_missing_self_threshold'] == pytest.approx(0.4)
    assert _immune(n_macrophages=0).get_state()[
        'mean_macrophage_polarization'] is None
def _flagship_engine():
    engine = SimulationEngine(SimulationConfig(
        dt=0.01, duration=0.5, use_gpu=False))
    engine.register_module('molecular', MolecularModule, {
        'transcription_rate': 0.0, 'exosome_release_rate': 0.0,
        'mutation_rate': 0.0,
    })
    engine.register_module('cellular', CellularModule, {
        'n_normal_cells': 0, 'n_cancer_cells': 0,
        'hla_alleles': ['HLA-A*02:01', 'HLA-A*24:02', 'HLA-B*07:02'],
    })
    engine.register_module('immune', ImmuneModule, {
        'n_t_cells': 1, 'n_nk_cells': 0, 'n_macrophages': 0,
        'tcr_recognition_threshold': 0.0, 'tcr_repertoire_size': 200,
        'tcr_seed': 0, 'costimulation': 1.0, 'checkpoint_block': 0.0,
        # The subject is the causal chain, not kill timing. A lethal T-cell
        # contact lasts ~75 min by the library's measured value, longer than
        # this run, so it is made one minute long here.
        'lethal_contact_minutes': {'T_cell': 1.0},
    })
    engine.initialize()
    return engine
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_flagship_closed_loop_trace_is_unchanged(seed):
    # Pinned to the trace measured on HEAD 1a53937 BEFORE this
    # upgrade, across seeds 0/1/2:
    #   mutation_occurred 1 -> peptide_generated 1 ->
    #   peptide_presented 2 -> immune_activated 1 ->
    #   cancer_killed 1 -> cell_killed_by_tcell 1
    # with total_kills 1, tcell_kills 1, tcr_recognitions 1,
    # exhaustion_transitions 0 and 38 peptides generated.
    np.random.seed(seed)
    engine = _flagship_engine()
    molecular = engine.modules['molecular']
    cellular = engine.modules['cellular']
    immune = engine.modules['immune']
    cellular.set_molecular_module(molecular)
    immune.set_cellular_module(cellular)

    cancer_id = cellular.add_cell(position=[100, 100, 50],
                                  cell_type='cancer')
    cellular.cells[cancer_id].mhc1_expression = 0.9
    molecular.add_cell(cancer_id)
    t_id = next(iter(immune.immune_cells))
    pos = np.array([100.0, 100.0, 50.0], dtype=np.float32)
    cellular.cells[cancer_id].position = pos.copy()
    immune.immune_cells[t_id].position = pos.copy()

    seen = {}
    for event in (EventTypes.MUTATION_OCCURRED,
                  EventTypes.PEPTIDE_GENERATED,
                  EventTypes.PEPTIDE_PRESENTED,
                  EventTypes.IMMUNE_ACTIVATED,
                  EventTypes.CANCER_KILLED,
                  EventTypes.CELL_KILLED_BY_TCELL):
        def _count(_data, key=event):
            seen[key] = seen.get(key, 0) + 1
        engine.event_bus.subscribe(event, _count)

    molecular.introduce_mutation(cancer_id, 'TP53', 'R175H')
    engine.run()

    assert seen.get(EventTypes.MUTATION_OCCURRED) == 1
    assert seen.get(EventTypes.PEPTIDE_GENERATED) == 1
    assert seen.get(EventTypes.PEPTIDE_PRESENTED) == 2, (
        f"presented {seen.get(EventTypes.PEPTIDE_PRESENTED)} peptides, "
        f"expected 2. Mutations that bound nothing: "
        f"{engine.modules['cellular'].unpresented_mutations}")
    assert seen.get(EventTypes.IMMUNE_ACTIVATED) == 1
    assert seen.get(EventTypes.CANCER_KILLED) == 1
    assert seen.get(EventTypes.CELL_KILLED_BY_TCELL) == 1
    assert immune.total_kills == 1
    assert immune.total_tcell_kills == 1
    assert immune.total_tcr_recognitions == 1
    assert immune.total_exhaustion_transitions == 0
    assert cellular.total_peptides_generated == 38
    assert cellular.total_peptides_presented == 2
def test_the_flagship_config_has_no_macrophage_so_the_gate_cannot_fire():
    engine = _flagship_engine()
    immune = engine.modules['immune']
    assert immune.n_macrophages == 0
    assert not [c for c in immune.immune_cells.values()
                if c.cell_type == 'macrophage']
    tcell = next(iter(immune.immune_cells.values()))
    assert immune._local_m2_burden(tcell.position) == 0.0
