"""Immune escape can now be inherited, so selection has raw material.

An established tumour is by definition one that escaped. This engine
could not produce escape, because nothing varied: ``cellular_module``
seeded EVERY cancer cell at ``mhc1_expression = 0.3`` (three sites) and
``_divide_cell`` copied ``mhc1_expression``, ``mhc1_baseline`` and
``pdl1_expression`` to the daughter exactly. Immune pressure therefore
had nothing to select on and escape could only ever be imposed by hand.

Now a daughter MAY differ, at a rate with a source
(``engine/py/immune/escape_variation.py``), and the two routes stay
structurally apart because the acceptance test turns on it:

* one HLA ALLELE lost -- surface class I falls to (n-1)/n of
  baseline and the presentations on the lost allele are dropped. The
  cell is still recognised, so checkpoint blockade still reaches it.
  Rate 1.55e-6 per division, the geometric mean of five measured
  per-cell-per-generation rates (PMIDs 6969452, 10029664, 70075,
  9443039 x2).
* a SECOND hit on an already-hemizygous cell -- no surface class I,
  no displayed peptides. Below ``_recognize``'s 0.05 floor the cell is
  invisible to T cells and blockade cannot reach it. ESTIMATE: no
  per-division B2M rate exists.

Two things are pinned hardest here.

ONE: nothing moved. Every draw comes from a generator of its own, so
the global stream that every pinned run depends on is byte-identical
with escape on and off -- which is what lets escape default to ON. The
expectation that makes ON safe is computed on CANCER divisions, because
``_maybe_escape`` returns None for every other cell type: a 24 h
Baseline SLAB run makes 53-56 cancer divisions of ~849, i.e. 8.7e-5
expected events, and a 24 h Baseline RING run makes 35 divisions of
which ZERO are a cancer cell's, so the ring never reaches the draw at
all. A run that shows no event is behaving correctly, not broken.

That last fact matters to how this file is read. The 24 h ring on/off
identity below cannot exercise the draw, so the code path is pinned
instead by test_at_the_literature_rate_a_short_run_changes_nothing_at_all
(200 cancer divisions, 200 draws, 0 events, stream identical) and by
test_a_six_hour_slab_run_is_identical_with_escape_on_and_off (12 cancer
divisions through the real preset path).

TWO: the 0.05 recognition floor is MIRRORED in escape_variation, not
imported, because the immune module must not depend on the cellular
one. It is pinned here by behaviour, so a change to either side fails.
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.core import SimulationConfig, SimulationEngine
from cognisom.engine.py.immune import escape_variation as ev
from cognisom.modules.cellular_module import (
    MHC1_IFN_SATURATION, CellularModule)
from cognisom.modules.immune_module import ImmuneModule
from cognisom.modules.molecular_module import MolecularModule


def _cellular(**config) -> CellularModule:
    """A cellular module with no seeded population, so every cell in a
    test is one the test put there."""
    with contextlib.redirect_stdout(io.StringIO()):
        module = CellularModule({'n_normal_cells': 0, 'n_cancer_cells': 0,
                                 **config})
        module.initialize()
    return module


def _cancer(module: CellularModule, mhc1: float = 0.3) -> int:
    cell_id = module.add_cell(position=[100, 100, 50], cell_type='cancer')
    module.cells[cell_id].mhc1_expression = mhc1
    return cell_id


def _divide(module: CellularModule, cell_id: int) -> int:
    module._divide_cell(cell_id)
    return max(module.cells)


# ── every number has a source or a range ─────────────────────────────────

def test_every_escape_number_is_sourced_or_an_estimate_with_a_range():
    report = ev.calibration_report()
    assert set(report) == {
        'mhc1_allele_loss_per_division', 'mhc1_second_hit_per_division',
        'surface_fraction_per_allele_lost', 'pdl1_heritable_gain_per_division',
        'pdl1_heritable_gain_magnitude', 'founder_hla_loh_fraction'}
    for name, entry in report.items():
        low, high = entry['range']
        assert low <= entry['value'] <= high, name
        assert entry['status'] in {'sourced', 'derived', 'ESTIMATE'}, name
        assert entry['source'], name
    # The one route with a literature rate is 'derived' from it; the one
    # without is an ESTIMATE and says so.
    assert report['mhc1_allele_loss_per_division']['status'] == 'derived'
    assert report['mhc1_second_hit_per_division']['status'] == 'ESTIMATE'


def test_the_rate_is_the_geometric_mean_of_five_measured_rates():
    rates = [r for r, _pmid, _what in ev.MHC1_LOSS_RATE_MEASUREMENTS]
    assert len(rates) == 5
    expected = np.exp(np.mean(np.log(rates)))
    assert ev.MHC1_ALLELE_LOSS_PER_DIVISION.value == pytest.approx(expected)
    assert ev.MHC1_ALLELE_LOSS_PER_DIVISION.value == pytest.approx(
        1.55e-6, rel=0.01)
    # The bracket is the measured extremes, a 14x spread, not a guess.
    assert ev.MHC1_ALLELE_LOSS_PER_DIVISION.low == 5.0e-7
    assert ev.MHC1_ALLELE_LOSS_PER_DIVISION.high == 6.9e-6
    for pmid in ('6969452', '10029664', '70075', '9443039'):
        assert pmid in ev.MHC1_ALLELE_LOSS_PER_DIVISION.source


def test_the_five_values_are_not_five_independent_determinations():
    """Five reported values, four papers, two laboratories -- and the
    source has to say so, because a geometric mean reads as an average
    of independent measurements.

    Re-fetched by efetch 2026-10-28: PMID 6969452 reports no new rate,
    it restates its own group's earlier one ("...the overall rate of
    spontaneous variation for HLA alleles, which we previously estimated
    at about 5 x 10(-7) per cell per generation"), and that group is
    also PMID 70075. de Nooij-van Dalen contributes 3 of the 5.
    """
    by_pmid = {}
    for rate, pmid, what in ev.MHC1_LOSS_RATE_MEASUREMENTS:
        by_pmid.setdefault(pmid, []).append(rate)
    assert len(ev.MHC1_LOSS_RATE_MEASUREMENTS) == 5
    assert len(by_pmid) == 4
    assert len(by_pmid['9443039']) == 2        # two lines, one paper
    # The restated entry is labelled as restated, in the entry itself.
    restated = next(what for rate, pmid, what
                    in ev.MHC1_LOSS_RATE_MEASUREMENTS if pmid == '6969452')
    assert 'PREVIOUSLY ESTIMATED' in restated
    assert 'previously estimated' in restated   # the paper's own words
    assert '70075' in restated                  # named as the same group
    # Dropping it moves the point value 33%, which is inside the bracket
    # -- so the caveat is about independence, not about the number.
    rest = [r for r, pmid, _ in ev.MHC1_LOSS_RATE_MEASUREMENTS
            if pmid != '6969452']
    without = float(np.exp(np.mean(np.log(rest))))
    assert without == pytest.approx(2.06e-6, rel=0.01)
    assert (ev.MHC1_ALLELE_LOSS_PER_DIVISION.low <= without
            <= ev.MHC1_ALLELE_LOSS_PER_DIVISION.high)
    assert '2.06e-6' in ev.MHC1_ALLELE_LOSS_PER_DIVISION.source


def test_the_magnitude_is_per_allele_and_says_it_is_not_a_haplotypes():
    """The name and the implementation disagreed until 2026-10-28: the
    trait was called a haplotype loss while one ALLELE was drawn from
    the panel. A class I haplotype is an A, a B and a C allele in cis,
    so on a 6-allele genotype it would leave 3/6 = 0.5, not 5/6 -- three
    times the surface cost. The code was right and the name was wrong."""
    assert ev.TRAIT_ALLELE_LOSS == 'mhc1_allele_loss'
    assert 'haplotype' not in ev.TRAIT_ALLELE_LOSS
    assert ev.surviving_surface_fraction(6) == pytest.approx(5 / 6)
    assert ev.surviving_surface_fraction(3) == pytest.approx(2 / 3)
    # The source text has to carry the warning, because 5/6 is what a
    # reader will otherwise quote for an HLA LOH event.
    source = ev.SURFACE_FRACTION_PER_ALLELE_LOST.source
    assert 'per ALLELE' in source
    assert "NOT a haplotype's magnitude" in source
    # And one allele of a 6-allele panel is what the engine does.
    module = _cellular(hla_alleles=['HLA-A*02:01', 'HLA-A*24:02',
                                    'HLA-B*07:02', 'HLA-B*08:01',
                                    'HLA-C*07:01', 'HLA-C*07:02'],
                       escape_mhc1_loss_per_division=1.0)
    daughter = module.cells[_divide(module, _cancer(module, mhc1=0.3))]
    assert len(daughter.escape_lost_hla) == 1
    assert daughter.mhc1_expression == pytest.approx(0.3 * 5 / 6)


def test_the_scale_problem_is_in_the_module_as_numbers():
    """Why a short preset run shows no de novo escape, stated so nobody
    has to rediscover it by staring at a zero.

    The argument is CANCER divisions. Only a cancer daughter draws, so
    a run's total division count is the wrong denominator: the 24 h
    Baseline slab makes 53-56 cancer divisions out of 846-849, and the
    24 h Baseline ring makes 0 out of 35.
    """
    assert ev.divisions_per_expected_event() == pytest.approx(6.5e5, rel=0.02)
    # Slab Baseline, 24 h: cancer divisions, not the 849 total.
    assert ev.expected_events(56) == pytest.approx(8.7e-5, rel=0.02)
    # Ring Baseline, 24 h: no cancer cell divides, so the draw is never
    # reached and the expectation is exactly zero -- not the 5.4e-5 that
    # using the 35 total divisions gave.
    assert ev.expected_events(0) == 0.0
    assert ev.expected_events(35) == pytest.approx(5.4e-5, rel=0.02)
    assert ev.expected_events(3000) == pytest.approx(4.6e-3, rel=0.02)
    assert ev.expected_events(1e9) == pytest.approx(1550, rel=0.02)


def test_the_second_hit_is_a_second_hit_not_an_independent_route():
    # Total loss needs two hits, so de novo it is ~mu^2 -- never. The
    # structure, not the number, is the claim.
    mu = ev.MHC1_ALLELE_LOSS_PER_DIVISION.value
    assert ev.MHC1_SECOND_HIT_PER_DIVISION.value == mu
    assert mu * ev.MHC1_SECOND_HIT_PER_DIVISION.value < 1e-11
    source = ev.MHC1_SECOND_HIT_PER_DIVISION.source
    assert '29070816' in source          # Sade-Feldman 2017
    assert '27433843' in source          # Zaretsky 2016


def test_heritable_pdl1_is_zero_and_the_module_says_why():
    # The literature gives prevalences in established tumours and no
    # per-division rate, and the engine already carries the dominant
    # inducible routes -- so the honest number is zero, not an invention.
    assert ev.PDL1_HERITABLE_GAIN_PER_DIVISION.value == 0.0
    source = ev.PDL1_HERITABLE_GAIN_PER_DIVISION.source
    for pmid in ('29902298', '30576871', '27281199'):
        assert pmid in source
    assert 'inducible' in source
    assert _cellular().escape_pdl1_gain_per_division == 0.0


def test_the_founder_fraction_is_a_target_not_a_rate():
    assert ev.FOUNDER_HLA_LOH_FRACTION.value == 0.0
    assert '29107330' in ev.FOUNDER_HLA_LOH_FRACTION.source   # McGranahan
    assert 'ESTIMATE' == ev.FOUNDER_HLA_LOH_FRACTION.status
    # The range is the whole unit interval, not McGranahan's 40%. That
    # 40% is a per-TUMOUR prevalence and this is a per-CELL fraction, so
    # a clonal event puts 1.0 of the cells there -- and the headline
    # measurements of this item all run at 0.5, which the old [0, 0.40]
    # bracket called out of range while reporting no such thing.
    assert (ev.FOUNDER_HLA_LOH_FRACTION.low,
            ev.FOUNDER_HLA_LOH_FRACTION.high) == (0.0, 1.0)
    at_half = ev.calibration_report(founder_fraction=0.5)[
        'founder_hla_loh_fraction']
    assert at_half['scaled_from_calibration'] is True
    assert at_half['out_of_range'] is False


def test_a_value_outside_a_declared_bracket_has_to_say_so():
    """scaled_from_calibration only says a run differs from the point
    value. A run can sit outside every bracket in the module and that
    flag reads the same, which is a much weaker statement than the
    brackets are there to support."""
    inside = ev.calibration_report(mhc1_loss=2e-6)[
        'mhc1_allele_loss_per_division']
    assert inside['scaled_from_calibration'] is True
    assert inside['out_of_range'] is False       # within 5e-7..6.9e-6
    outside = ev.calibration_report(mhc1_loss=0.5)[
        'mhc1_allele_loss_per_division']
    assert outside['out_of_range'] is True
    # Nothing is flagged when nothing was configured.
    assert 'out_of_range' not in ev.calibration_report()[
        'mhc1_allele_loss_per_division']


def test_a_run_that_scaled_mu_cannot_report_the_literature_rate():
    report = ev.calibration_report(mhc1_loss=1e-2)
    entry = report['mhc1_allele_loss_per_division']
    assert entry['value'] == ev.MHC1_ALLELE_LOSS_PER_DIVISION.value
    assert entry['configured'] == 1e-2
    assert entry['scaled_from_calibration'] is True
    assert ev.calibration_report(
        mhc1_loss=ev.MHC1_ALLELE_LOSS_PER_DIVISION.value
    )['mhc1_allele_loss_per_division']['scaled_from_calibration'] is False


def test_a_six_allele_patient_panel_is_not_reported_as_the_default_three():
    """The calibrated surface fraction is 2/3 because the engine's
    default panel has three alleles. A run on a real class-I genotype
    uses 5/6, and must say so rather than reporting 2/3."""
    entry = ev.calibration_report(n_alleles=6)[
        'surface_fraction_per_allele_lost']
    assert entry['value'] == pytest.approx(2 / 3)
    assert entry['configured'] == pytest.approx(5 / 6)
    assert entry['scaled_from_calibration'] is True

    module = _cellular(hla_alleles=['HLA-A*02:01', 'HLA-A*24:02',
                                    'HLA-B*07:02', 'HLA-B*08:01',
                                    'HLA-C*07:01', 'HLA-C*07:02'],
                       escape_mhc1_loss_per_division=1.0)
    daughter = module.cells[_divide(module, _cancer(module, mhc1=0.3))]
    assert daughter.mhc1_expression == pytest.approx(0.3 * 5 / 6)
    assert module.escape_distribution()['calibration'][
        'surface_fraction_per_allele_lost']['configured'] == pytest.approx(
            5 / 6)


# ── nothing moved: off, and on-with-no-event ─────────────────────────────

def _division_leaves_global_stream_at(module_kwargs, n_divisions=5):
    """Divide a cancer cell n times, then return the global stream's
    state and the daughter. Seeds the global stream first, so the only
    thing that can differ between calls is what the division draws."""
    module = _cellular(**module_kwargs)
    np.random.seed(4242)
    cell_id = _cancer(module)
    last = cell_id
    for _ in range(n_divisions):
        last = _divide(module, cell_id)
    return np.random.get_state(), module, module.cells[last]


def test_with_escape_off_the_daughter_and_the_stream_are_untouched():
    off_state, off_module, off_daughter = _division_leaves_global_stream_at(
        {'escape_enabled': False})
    parent = off_module.cells[min(off_module.cells)]

    assert off_daughter.mhc1_expression == parent.mhc1_expression
    assert off_daughter.mhc1_baseline == parent.mhc1_baseline
    assert off_daughter.pdl1_expression == parent.pdl1_expression
    assert off_daughter.escape_traits == ()
    assert off_daughter.escape_born_at_h is None
    assert off_daughter.escape_lost_hla == ()
    assert off_module.escape_events == {name: 0 for name in ev.ESCAPE_TRAITS}

    # ...and the global stream sits exactly where it did: a division
    # draws its 3-vector offset and nothing else.
    np.random.seed(4242)
    np.random.randn(3 * 5)
    expected = np.random.get_state()
    assert off_state[1].tolist() == expected[1].tolist()
    assert off_state[2:] == expected[2:]


def test_at_the_literature_rate_a_short_run_changes_nothing_at_all():
    """ON by default is safe because the sourced rate is 1.55e-6: this is
    the measurement that defends the default, not an opinion."""
    off_state, _off_module, off_daughter = _division_leaves_global_stream_at(
        {'escape_enabled': False}, n_divisions=200)
    on_state, on_module, on_daughter = _division_leaves_global_stream_at(
        {'escape_enabled': True}, n_divisions=200)

    assert on_state[1].tolist() == off_state[1].tolist()
    assert on_state[2:] == off_state[2:]
    assert on_daughter.mhc1_expression == off_daughter.mhc1_expression
    assert on_daughter.escape_traits == ()
    # 200 divisions at 1.55e-6 expect 3.1e-4 events. Zero is the right
    # answer; if this ever fires the rate was changed, not the code.
    assert sum(on_module.escape_events.values()) == 0


def test_the_draw_comes_from_the_dedicated_stream_even_when_it_fires():
    on_state, module, daughter = _division_leaves_global_stream_at(
        {'escape_mhc1_loss_per_division': 1.0}, n_divisions=1)
    assert daughter.escape_traits == (ev.TRAIT_ALLELE_LOSS,)

    np.random.seed(4242)
    np.random.randn(3)
    expected = np.random.get_state()
    assert on_state[1].tolist() == expected[1].tolist()
    assert on_state[2:] == expected[2:]
    # The dedicated stream is reproducible from the config seed alone.
    other = _cellular(escape_mhc1_loss_per_division=1.0)
    np.random.seed(999)
    second = other.cells[_divide(other, _cancer(other))]
    assert second.escape_lost_hla == daughter.escape_lost_hla


def test_two_escape_seeds_give_two_different_lineages():
    lost = []
    for seed in (0, 1, 2, 3):
        module = _cellular(escape_mhc1_loss_per_division=1.0,
                           escape_seed=seed)
        cell_id = _cancer(module)
        lost.append(tuple(
            module.cells[_divide(module, cell_id)].escape_lost_hla))
    assert len(set(lost)) > 1, "escape_seed does not reach the draw"


# ── the realised rate is the configured one ──────────────────────────────

@pytest.mark.parametrize('p', [0.05, 0.2])
def test_the_realised_rate_matches_the_configured_probability(p):
    n = 4000
    module = _cellular(escape_mhc1_loss_per_division=p, escape_seed=7)
    cell_id = _cancer(module)
    hits = 0
    for _ in range(n):
        daughter = module.cells[_divide(module, cell_id)]
        hits += bool(daughter.escape_traits)
    # Binomial, generous: 4 sigma.
    sigma = (p * (1 - p) / n) ** 0.5
    assert abs(hits / n - p) < 4 * sigma, (hits / n, p, sigma)
    assert module.escape_events[ev.TRAIT_ALLELE_LOSS] == hits


def test_a_total_loss_needs_a_first_hit_first():
    """One division of an intact cell can never produce the invisible
    phenotype, however high the second-hit probability is: that is the
    two-hit structure, and it is why B2M loss appears under therapy
    pressure rather than in a naive tumour."""
    module = _cellular(escape_mhc1_loss_per_division=0.0,
                       escape_mhc1_second_hit_per_division=1.0)
    cell_id = _cancer(module)
    for _ in range(50):
        daughter = module.cells[_divide(module, cell_id)]
        assert daughter.escape_traits == ()
    assert module.escape_events[ev.TRAIT_TOTAL_LOSS] == 0


# ── the magnitude is the patient's ───────────────────────────────────────

def test_an_allele_loss_scales_surface_class_i_by_the_allele_count():
    module = _cellular(escape_mhc1_loss_per_division=1.0)
    cell_id = _cancer(module, mhc1=0.3)
    daughter = module.cells[_divide(module, cell_id)]
    fraction = ev.surviving_surface_fraction(len(module.hla_alleles))
    assert fraction == pytest.approx(2.0 / 3.0)
    assert daughter.mhc1_expression == pytest.approx(0.3 * fraction)
    # The level it returns to without IFN-gamma falls too, or the first
    # IFN-gamma step would undo the loss.
    assert daughter.mhc1_baseline == pytest.approx(0.3 * fraction)
    assert len(daughter.escape_lost_hla) == 1
    assert daughter.escape_lost_hla[0] in module.hla_alleles


def test_the_repertoire_hit_is_the_patients_own_peptides():
    """No constant for 'how much an allele loss costs': the cell loses
    exactly the presentations bound to the lost allele, so the cost is
    computed from where this patient's neoantigens bind (McGranahan's
    CRUK0020 lost 92% of its predicted neoantigens to its lost
    alleles)."""
    module = _cellular()
    cell_id = _cancer(module, mhc1=0.9)
    cell = module.cells[cell_id]
    cell.mhc1_displayed_peptides = _fake_presentations(module.hla_alleles)
    before = len(cell.mhc1_displayed_peptides)

    module._lose_one_allele(cell)
    lost = cell.escape_lost_hla[0]
    assert all(p.hla_allele != lost for p in cell.mhc1_displayed_peptides)
    assert len(cell.mhc1_displayed_peptides) < before
    assert all(p.hla_allele in module.hla_alleles
               for p in cell.mhc1_displayed_peptides)


def test_an_escaped_cell_cannot_re_present_on_the_allele_it_lost():
    """The escape event has to survive the next mutation. Presentation
    is an INGEST path (on_mutation_occurred appends to the cell), so
    without a filter there an escaped cell that acquires a mutation --
    or a founder seeded before the patient's mutations are presented --
    would display it on the allele it had lost and be recognisable
    again, quietly undoing the event."""
    module = _cellular()
    cell = module.cells[_cancer(module, mhc1=0.3)]
    module._lose_one_allele(cell)
    lost = cell.escape_lost_hla[0]

    fresh = _fake_presentations(module.hla_alleles)
    kept = module._presentable_on(cell, fresh)
    assert all(p.hla_allele != lost for p in kept)
    assert len(kept) == len(module.hla_alleles) - 1

    # A totally lost cell presents nothing from any allele.
    module._lose_mhc1_entirely(cell)
    assert module._presentable_on(cell, fresh) == []
    # An untouched cell is filtered by nothing at all -- same list back.
    intact = module.cells[_cancer(module)]
    assert module._presentable_on(intact, fresh) is fresh


def test_a_founder_seeded_tumour_presents_on_its_surviving_alleles_only():
    """The order trap this filter closes: founder seeding runs in
    initialize(), the patient's mutations are presented afterwards."""
    module = _cellular(n_cancer_cells=40, n_normal_cells=0,
                       escape_founder_hla_loh_fraction=1.0, escape_seed=2)
    assert module.escape_founder_cells == 40
    for cell in module.cells.values():
        assert len(cell.escape_lost_hla) == 1
        kept = module._presentable_on(cell, _fake_presentations(
            module.hla_alleles))
        assert all(p.hla_allele not in cell.escape_lost_hla for p in kept)


def test_the_real_presentation_path_respects_a_lost_allele():
    """Through present_patient_mutations, not through the filter
    directly: the peptides are cut from the reference protein and scored
    on this patient's panel, and a founder that lost an allele must not
    come back displaying on it."""
    engine = SimulationEngine(SimulationConfig(
        dt=0.01, duration=0.1, use_gpu=False))
    engine.register_module('molecular', MolecularModule, {
        'transcription_rate': 0.0, 'exosome_release_rate': 0.0,
        'mutation_rate': 0.0})
    engine.register_module('cellular', CellularModule, {
        'n_normal_cells': 0, 'n_cancer_cells': 12,
        'escape_founder_hla_loh_fraction': 1.0, 'escape_seed': 4,
        'patient_mutations': [{'gene': 'KRAS', 'aa_change': 'G12D',
                               'label': 'KRAS_G12D', 'oncogenic': True}],
    })
    with contextlib.redirect_stdout(io.StringIO()):
        engine.initialize()
    cellular = engine.modules['cellular']
    cellular.set_molecular_module(engine.modules['molecular'])
    with contextlib.redirect_stdout(io.StringIO()):
        report = cellular.present_patient_mutations()

    cancer = [c for c in cellular.cells.values() if c.cell_type == 'cancer']
    assert len(cancer) == 12
    assert all(c.escape_lost_hla for c in cancer)
    assert report['cells_displaying'] > 0, (
        "precondition failed: nothing presented, so the assertion below "
        "would pass on an empty list")
    for cell in cancer:
        assert all(p.hla_allele not in cell.escape_lost_hla
                   for p in cell.mhc1_displayed_peptides), (
            f"{cell.cell_id} presents on {cell.escape_lost_hla}, the "
            "allele it lost -- the escape event was undone by the "
            "presentation path")
    # Not vacuous. The uniform draw covered the whole panel...
    lost_seen = {a for c in cancer for a in c.escape_lost_hla}
    assert lost_seen == set(cellular.hla_alleles)
    # ...and this patient DOES present on the alleles that were lost, so
    # the filter removed real binders rather than agreeing with peptides
    # that never bound them. Measured on the same tumour with no founder
    # loss: KRAS G12D binds all three panel alleles.
    intact = _cellular(
        n_cancer_cells=1,
        patient_mutations=[{'gene': 'KRAS', 'aa_change': 'G12D',
                            'label': 'KRAS_G12D', 'oncogenic': True}])
    intact.set_molecular_module(engine.modules['molecular'])
    with contextlib.redirect_stdout(io.StringIO()):
        intact.present_patient_mutations()
    intact_cell = next(iter(intact.cells.values()))
    intact_alleles = {p.hla_allele
                      for p in intact_cell.mhc1_displayed_peptides}
    assert intact_alleles, "precondition failed: the intact cell shows nothing"
    assert intact_alleles & lost_seen, (
        "no founder lost an allele this patient presents on, so the "
        "filter had nothing to remove and this test proves nothing")
    # What the event COSTS is a function of which allele went, not of
    # the cell -- and that is the patient-specific coupling, measured:
    #
    #   real MHCflurry predictor (py3.11): KRAS G12D binds HLA-A*02:01
    #   and nothing else on this panel, so the 2 founders of 12 that
    #   lost A*02:01 display NOTHING -- T-cell-invisible at surface
    #   MHC-I 0.2 -- while the other 10 keep their single binder. For
    #   this patient one allele carries 100% of the repertoire, which
    #   is McGranahan's CRUK0020 (92%) taken to its limit.
    #
    #   PWM fallback (py3.13, no MHCflurry): binders on all three
    #   alleles, so every founder refills its five display slots from
    #   the alleles it kept and none goes dark.
    #
    # So this assertion is deliberately NOT "some cell goes dark": that
    # depends on the patient and on which scorer is installed. What is
    # invariant is that the cost is set by the allele.
    by_lost: dict = {}
    for cell in cancer:
        by_lost.setdefault(cell.escape_lost_hla, set()).add(
            len(cell.mhc1_displayed_peptides))
    assert all(len(sizes) == 1 for sizes in by_lost.values()), by_lost
    assert len(by_lost) == len(cellular.hla_alleles)

    # The report must not call this tumour fully visible: the dark cells
    # are above the surface floor, so only n_escaped_without_display
    # finds them.
    dist = cellular.escape_distribution()
    dark = sum(1 for c in cancer if not c.mhc1_displayed_peptides)
    assert dist['n_escaped_without_display'] == dark
    if dark:
        assert dist['n_below_recognition'] == 0, (
            "these cells still carry surface class I; if the floor "
            "caught them the two routes are no longer distinguishable")


def test_the_lost_allele_is_drawn_uniformly_not_toward_the_best_target():
    """Mutation supply must be blind: biasing the draw toward the allele
    carrying the most neoantigens would impose the selection the immune
    module is supposed to discover, and a uniform supply under selection
    is what produces McGranahan's parallel evolution."""
    module = _cellular(escape_mhc1_loss_per_division=1.0, escape_seed=11)
    cell_id = _cancer(module)
    counts = {allele: 0 for allele in module.hla_alleles}
    n = 900
    for _ in range(n):
        daughter = module.cells[_divide(module, cell_id)]
        counts[daughter.escape_lost_hla[0]] += 1
    expected = n / len(counts)
    sigma = (expected * (1 - 1 / len(counts))) ** 0.5
    for allele, count in counts.items():
        assert abs(count - expected) < 4 * sigma, (allele, counts)


def test_total_loss_cannot_be_raised_back_by_interferon_gamma():
    """A B2M-null cell does not answer IFN-gamma with surface class I.
    _update_mhc1 relaxes toward baseline * (1 + gain * drive), so the
    baseline has to go to zero as well as the level -- and the gain and
    saturation themselves are untouched, because that mechanism is the
    one here that passed an external test on 592 TCGA tumours."""
    module = _cellular()
    cell = module.cells[_cancer(module, mhc1=0.3)]
    module._lose_mhc1_entirely(cell)
    assert cell.mhc1_expression == 0.0 and cell.mhc1_baseline == 0.0

    cell.ifn_gamma_level = MHC1_IFN_SATURATION * 10
    for _ in range(100):
        module._update_mhc1(cell, 1.0)
    assert cell.mhc1_expression == 0.0

    # A partial-loss cell DOES answer it, from its lowered baseline.
    partial = module.cells[_cancer(module, mhc1=0.3)]
    module._lose_one_allele(partial)
    partial.ifn_gamma_level = MHC1_IFN_SATURATION * 10
    for _ in range(100):
        module._update_mhc1(partial, 1.0)
    assert partial.mhc1_expression > partial.mhc1_baseline


# ── the structural difference the acceptance test turns on ───────────────

def _fake_presentations(alleles):
    """One cheap MHCPresentation per allele. Built by hand rather than
    through the predictor so the test does not depend on which scorer is
    installed -- MHCflurry is not importable on py3.13."""
    from cognisom.engine.py.immune.mhc_loading import MHCPresentation
    from cognisom.engine.py.molecular.peptidome import Peptide
    out = []
    for i, allele in enumerate(alleles):
        peptide = Peptide(
            sequence='KLVVVGADG', source_gene='KRAS', length=9,
            is_mutant=True, wild_type_sequence='KLVVVGAGG',
            mutation_label='G12D', anchor_position_in_peptide=7,
            parent_position_1based=5 + i, cleavage_score=0.9,
        )
        out.append(MHCPresentation(
            peptide=peptide, hla_allele=allele, ic50_nm=30.0,
            binding_level='strong', presentation_score=0.95))
    return out


def _recognition_engine():
    """One T cell with a permissive repertoire, and the cellular module
    whose divisions it will be shown. Mirrors the setup in
    test_division_inheritance."""
    engine = SimulationEngine(SimulationConfig(
        dt=0.01, duration=0.5, use_gpu=False))
    engine.register_module('molecular', MolecularModule, {
        'transcription_rate': 0.0, 'exosome_release_rate': 0.0,
        'mutation_rate': 0.0})
    engine.register_module('cellular', CellularModule, {
        'n_normal_cells': 0, 'n_cancer_cells': 0,
        'escape_mhc1_loss_per_division': 1.0,
        'escape_mhc1_second_hit_per_division': 1.0})
    engine.register_module('immune', ImmuneModule, {
        'n_t_cells': 1, 'n_nk_cells': 0, 'n_macrophages': 0,
        'tcr_recognition_threshold': 0.0, 'tcr_repertoire_size': 200,
        'tcr_seed': 0, 'costimulation': 1.0, 'checkpoint_block': 0.0})
    with contextlib.redirect_stdout(io.StringIO()):
        engine.initialize()
    cellular = engine.modules['cellular']
    immune = engine.modules['immune']
    immune.set_cellular_module(cellular)
    t_cell = immune.immune_cells[next(iter(immune.immune_cells))]
    return cellular, immune, t_cell


def test_partial_loss_stays_visible_and_total_loss_does_not():
    """The load-bearing assertion of this file. A recognised parent must
    produce a still-recognised daughter on the partial route and an
    unrecognised one on the total route -- otherwise the two routes are
    the same event and checkpoint blockade cannot tell them apart."""
    np.random.seed(0)
    cellular, immune, t_cell = _recognition_engine()

    parent_id = _cancer(cellular, mhc1=0.3)
    parent = cellular.cells[parent_id]
    parent.mhc1_displayed_peptides = _fake_presentations(cellular.hla_alleles)
    recognized, _ = immune._recognize(t_cell, parent)
    assert recognized, "precondition failed: the parent is not recognised"

    # First hit: one allele. Still above the 0.05 floor, still
    # presenting on the alleles it kept, so still recognised.
    daughter = cellular.cells[_divide(cellular, parent_id)]
    assert daughter.escape_traits == (ev.TRAIT_ALLELE_LOSS,)
    assert daughter.mhc1_expression >= ev.T_CELL_RECOGNITION_MHC1_FLOOR
    assert daughter.mhc1_displayed_peptides
    assert immune._recognize(t_cell, daughter)[0], (
        "a one-allele loss made the cell invisible -- the partial and "
        "total routes have collapsed into one")

    # Second hit on the already-hemizygous daughter: invisible.
    grandchild = cellular.cells[_divide(cellular, daughter.cell_id)]
    assert ev.TRAIT_TOTAL_LOSS in grandchild.escape_traits
    assert grandchild.mhc1_expression < ev.T_CELL_RECOGNITION_MHC1_FLOOR
    assert not immune._recognize(t_cell, grandchild)[0]


def test_a_patient_whose_whole_repertoire_sat_on_the_lost_allele_goes_dark():
    """Looks like a bug, is not. A cell can keep surface class I and
    still drop out of the T-cell loop, because _recognize also needs a
    non-empty presentation list -- so if ALL of this patient's
    neoantigens bound the lost allele, blockade has nothing left to
    un-block. That is the correct reading of McGranahan's CRUK0020 (92%
    of 1,220 predicted neoantigens on the lost alleles) and it is why
    the cost of this route is the patient's, not a constant."""
    np.random.seed(0)
    _ignored, immune, t_cell = _recognition_engine()
    # An escape_seed whose first allele draw is the panel's first entry,
    # so the test knows which allele the one event will take.
    seed = next(s for s in range(100)
                if np.random.default_rng([s, 0]).integers(3) == 0)
    cellular = _cellular(escape_seed=seed)
    doomed = cellular.hla_alleles[0]

    cell = cellular.cells[_cancer(cellular, mhc1=0.9)]
    cell.mhc1_displayed_peptides = _fake_presentations([doomed])
    assert immune._recognize(t_cell, cell)[0]

    assert cellular._lose_one_allele(cell) == ev.TRAIT_ALLELE_LOSS
    assert cell.escape_lost_hla == (doomed,)
    assert cell.mhc1_expression >= ev.T_CELL_RECOGNITION_MHC1_FLOOR
    assert cell.mhc1_displayed_peptides == []
    assert not immune._recognize(t_cell, cell)[0]


def test_the_recognition_floor_mirrors_the_immune_modules_own_literal():
    """escape_variation duplicates _recognize's 0.05 rather than
    importing it, because the immune module must not depend on this one.
    Pinned by behaviour on both sides of the line, so a change to either
    file fails here instead of silently disagreeing."""
    np.random.seed(0)
    cellular, immune, t_cell = _recognition_engine()
    cell = cellular.cells[_cancer(cellular, mhc1=1.0)]
    cell.mhc1_displayed_peptides = _fake_presentations(cellular.hla_alleles)

    floor = ev.T_CELL_RECOGNITION_MHC1_FLOOR
    cell.mhc1_expression = floor * 1.001
    assert immune._recognize(t_cell, cell)[0]
    cell.mhc1_expression = floor * 0.999
    assert not immune._recognize(t_cell, cell)[0]


def test_an_escaped_lineage_keeps_its_traits_into_the_next_generation():
    """Traits must be clonal: a grandchild of an escaped cell carries
    what its grandparent acquired, with no further event of its own."""
    module = _cellular(escape_mhc1_loss_per_division=1.0,
                       escape_mhc1_second_hit_per_division=0.0)
    parent_id = _cancer(module, mhc1=0.3)
    daughter = module.cells[_divide(module, parent_id)]
    assert daughter.escape_traits == (ev.TRAIT_ALLELE_LOSS,)

    grandchild = module.cells[_divide(module, daughter.cell_id)]
    great = module.cells[_divide(module, grandchild.cell_id)]
    for descendant in (grandchild, great):
        assert descendant.escape_traits == daughter.escape_traits
        assert descendant.escape_lost_hla == daughter.escape_lost_hla
        assert descendant.escape_born_at_h == daughter.escape_born_at_h
        assert descendant.mhc1_expression == pytest.approx(
            daughter.mhc1_expression)
    # One event, three cells carrying it: that is inheritance, not supply.
    assert module.escape_events[ev.TRAIT_ALLELE_LOSS] == 1
    assert module.escape_distribution()['traits'][
        ev.TRAIT_ALLELE_LOSS] == 3


def test_the_birth_time_is_the_lineages_first_escape():
    module = _cellular(escape_mhc1_loss_per_division=1.0,
                       escape_mhc1_second_hit_per_division=1.0)
    parent_id = _cancer(module, mhc1=0.3)
    module.clock_h = 3.0
    daughter = module.cells[_divide(module, parent_id)]
    module.clock_h = 9.0
    grandchild = module.cells[_divide(module, daughter.cell_id)]
    assert ev.TRAIT_TOTAL_LOSS in grandchild.escape_traits
    # The second hit does not re-date the lineage: the measurable
    # quantity is how long it has been under pressure.
    assert grandchild.escape_born_at_h == 3.0


# ── the run reports it ───────────────────────────────────────────────────

def test_the_report_counts_what_is_there():
    module = _cellular(escape_mhc1_loss_per_division=1.0,
                       escape_mhc1_second_hit_per_division=0.0,
                       escape_seed=3)
    intact_id = _cancer(module, mhc1=0.3)
    partial = module.cells[_divide(module, intact_id)]
    total = module.cells[_divide(module, partial.cell_id)]
    module._lose_mhc1_entirely(total)
    _cancer(module, mhc1=0.3)          # a second intact cell, never divided

    dist = module.escape_distribution()
    assert dist['enabled'] is True and dist['seed'] == 3
    assert dist['n_cancer'] == 4
    assert dist['n_escaped'] == 2
    assert dist['fraction_escaped'] == pytest.approx(0.5)
    assert dist['traits'][ev.TRAIT_ALLELE_LOSS] == 2
    assert dist['traits'][ev.TRAIT_TOTAL_LOSS] == 1
    assert dist['n_below_recognition'] == 1
    assert dist['fraction_below_recognition'] == pytest.approx(0.25)
    assert dist['recognition_threshold'] == ev.T_CELL_RECOGNITION_MHC1_FLOOR
    # Neither escaped cell was ever given a peptide to display, so both
    # count as dark -- the surface count above would have reported one.
    assert dist['n_escaped_without_display'] == 2
    assert dist['mean_mhc1'] == pytest.approx((0.3 + 0.2 + 0.0 + 0.3) / 4)
    assert sum(dist['lost_hla_alleles'].values()) == 2
    assert dist['first_escape_at_h'] == 0.0
    # Supply over the run's life, which selection then acts on: ONE
    # allele-loss event, carried by two cells. The population side
    # above and this side part company once cells divide or die.
    assert dist['events'][ev.TRAIT_ALLELE_LOSS] == 1
    assert dist['events'][ev.TRAIT_TOTAL_LOSS] == 1
    assert dist['calibration'][
        'mhc1_allele_loss_per_division']['configured'] == 1.0


def test_the_engine_surfaces_escape_in_get_state():
    module = _cellular()
    _cancer(module)
    state = module.get_state()
    assert 'escape' in state
    assert state['escape']['n_cancer'] == 1
    assert state['escape']['calibration'][
        'mhc1_allele_loss_per_division']['status'] == 'derived'


def test_an_empty_tumour_reports_zeroes_not_a_crash():
    dist = _cellular().escape_distribution()
    assert dist['n_cancer'] == 0
    assert dist['fraction_escaped'] == 0.0
    assert dist['mean_mhc1'] == 0.0
    assert dist['first_escape_at_h'] is None


# ── founder escape variants ──────────────────────────────────────────────

def test_the_founder_fraction_is_off_by_default_and_takes_no_draw():
    module = _cellular(n_cancer_cells=20, n_normal_cells=0)
    assert module.escape_founder_hla_loh_fraction == 0.0
    assert module.escape_founder_cells == 0
    assert all(not c.escape_traits for c in module.cells.values())
    # The founder sub-stream was not advanced, so turning the founder
    # fraction on later cannot move anything that was drawn before it.
    untouched = np.random.default_rng([module.escape_seed, 2]).random()
    assert module._escape_rng_founder.random() == untouched


def test_founder_seeding_puts_escape_in_the_founding_tumour():
    module = _cellular(n_cancer_cells=200, n_normal_cells=0,
                       escape_founder_hla_loh_fraction=0.4, escape_seed=5)
    dist = module.escape_distribution()
    assert dist['n_cancer'] == 200
    assert dist['fraction_escaped'] == pytest.approx(0.4, abs=0.08)
    assert module.escape_founder_cells == dist['n_escaped']
    # Seeded at a lowered baseline, still above the recognition floor:
    # the founder route is the PARTIAL one, which blockade can reach.
    assert dist['n_below_recognition'] == 0
    assert dist['calibration']['founder_hla_loh_fraction'][
        'scaled_from_calibration'] is True
    escaped = [c for c in module.cells.values() if c.escape_traits]
    assert all(c.mhc1_expression == pytest.approx(0.3 * 2 / 3)
               for c in escaped)


def test_escape_enabled_false_turns_off_founder_seeding_too():
    """One flag, one meaning. Until 2026-10-28 ``_seed_founder_escape``
    guarded only on the founder fraction, so a run that declared escape
    DISABLED and also named a founder fraction started with most of its
    tumour carrying HLA LOH while escape_distribution() reported
    'enabled': False.

    Measured then, 50 cancer cells at founder 0.5, escape_seed 1:
    escape_enabled False gave founder_cells 30, n_escaped 30 and
    mean_mhc1 0.24 -- identical to the enabled run, against the 0.30
    every cancer cell is seeded at.
    """
    off = _cellular(n_cancer_cells=50, n_normal_cells=0,
                    escape_enabled=False,
                    escape_founder_hla_loh_fraction=0.5, escape_seed=1)
    dist = off.escape_distribution()
    assert dist['enabled'] is False
    assert off.escape_founder_cells == 0
    assert dist['n_escaped'] == 0
    assert dist['founder_events'] == {name: 0 for name in ev.ESCAPE_TRAITS}
    assert dist['mean_mhc1'] == pytest.approx(0.30, abs=1e-9)
    assert all(not c.escape_traits for c in off.cells.values())

    # Not vacuous: the same configuration with the flag ON does seed.
    on = _cellular(n_cancer_cells=50, n_normal_cells=0,
                   escape_enabled=True,
                   escape_founder_hla_loh_fraction=0.5, escape_seed=1)
    assert on.escape_founder_cells > 0
    assert on.escape_distribution()['mean_mhc1'] < 0.30


def test_the_snapshot_carries_both_ways_out_of_the_t_cell_loop():
    """engine_runner's time series used to carry only n_below_recognition,
    the readout cellular_module's own docstring calls misleading on its
    own: _recognize also needs a non-empty presentation list, so an
    escaped cell with surface class I and nothing to display is just as
    invisible. On the default panel with the real MHCflurry predictor
    that is the DOMINANT form -- KRAS G12D binds HLA-A*02:01 and nothing
    else, so founders that lost A*02:01 go dark while
    n_below_recognition stays 0.

    So the snapshot carries both and their union, and this test is the
    only thing standing between a renamed key in escape_distribution and
    a silently zeroed series.
    """
    from cognisom.dashboard.engine_runner import EngineRunner

    np.random.seed(7)
    runner = EngineRunner(
        tissue='ring', dt=0.05, duration=0.5, scenario='Baseline',
        record_interval=1, snapshot_interval_hours=None,
        overrides={'cellular': {
            'escape_founder_hla_loh_fraction': 1.0, 'escape_seed': 3}})
    with contextlib.redirect_stdout(io.StringIO()):
        runner.build()
        runner.run()
    cells = runner.engine.modules['cellular']
    dist = cells.escape_distribution()
    snap = runner.history[-1]

    # Precondition: the founder fraction actually seeded something, or
    # every assertion below would pass on zeros.
    assert dist['n_escaped'] > 0

    assert snap.n_escaped == dist['n_escaped']
    assert snap.mean_mhc1 == pytest.approx(dist['mean_mhc1'])
    assert snap.n_mhc1_lost == dist['n_below_recognition']
    assert snap.n_escaped_without_display == dist[
        'n_escaped_without_display']
    assert snap.n_tcell_invisible == dist['n_tcell_invisible']
    # The union is not the sum: a total-loss cell is in both counts.
    assert snap.n_tcell_invisible <= (
        snap.n_mhc1_lost + snap.n_escaped_without_display)
    assert snap.n_tcell_invisible >= max(
        snap.n_mhc1_lost, snap.n_escaped_without_display)

    # And the series is reachable from a page: get_time_series left all
    # five keys out, so the snapshot fields existed and nothing could
    # plot them.
    series = runner.get_time_series()
    for key in ('n_escaped', 'mean_mhc1', 'n_mhc1_lost',
                'n_escaped_without_display', 'n_tcell_invisible'):
        assert key in series, key
        assert len(series[key]) == len(series['time'])
    assert series['n_escaped'][-1] == dist['n_escaped']


def test_the_union_of_the_two_invisible_routes_is_not_their_sum():
    """A totally lost cell has neither surface class I nor peptides, so
    it is counted by n_below_recognition AND by n_escaped_without_
    display. Adding them would report it twice."""
    module = _cellular(escape_mhc1_loss_per_division=1.0,
                       escape_mhc1_second_hit_per_division=0.0,
                       escape_seed=3)
    intact_id = _cancer(module, mhc1=0.3)
    partial = module.cells[_divide(module, intact_id)]
    total = module.cells[_divide(module, partial.cell_id)]
    module._lose_mhc1_entirely(total)

    dist = module.escape_distribution()
    assert dist['n_below_recognition'] == 1          # the total-loss cell
    assert dist['n_escaped_without_display'] == 2    # both, never dressed
    # Two distinct cells are invisible, not three.
    assert dist['n_tcell_invisible'] == 2


# ── what it does to a run (slow) ─────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.timeout(900)
def test_a_pinned_24h_run_is_identical_with_escape_on_and_off():
    """The measurement that defends ON by default. Measured on the ring
    Baseline preset the pinned 24 h tests use (seed 12345, dt 0.05):
    35 divisions, 10 cancer deaths, 115 normal cells alive, and zero de
    novo escape events -- the same with escape on and off.

    WHAT THIS DOES NOT SHOW, UPDATED 2026-11-12 -- and it has gone back
    and forth, so read the history rather than the latest line. Until
    2026-11-08 not one division here was a cancer cell's, so this run
    never reached the draw and the identity held on an unentered code
    path. Fixing the ECM deposition rate slowed the infiltrate's arrival
    enough for the tumour to get one division in first, which made the
    identity near-vacuous instead of wholly vacuous: one draw at
    1.55e-6 expects 1.5e-6 events, indistinguishable from none.

    It is back to ZERO cancer divisions as of the macrophage phagocytic
    licence (Dooling 2023; see
    tests/test_a_macrophage_needs_the_target_marked.py), and not for the
    reason one would guess. Macrophages now kill nothing in this preset,
    so one would expect the tumour to last LONGER and divide more. The
    run instead clears all ten cancer cells in ten kills with none
    dividing, where before it took eleven kills and one division. The
    cause is the shared RNG stream, not a faster infiltrate: removing
    the macrophage draws reorders every subsequent random number, and
    this preset's outcome sits on that knife edge. So THE IDENTITY
    BELOW IS WHOLLY VACUOUS AGAIN -- it is identity on a code path this
    run does not enter -- and it is asserted as zero, with the division
    count stated, so no reader is told otherwise. The draw itself is
    pinned by
    test_at_the_literature_rate_a_short_run_changes_nothing_at_all (200
    cancer divisions through the module) and by the 6 h slab run that
    follows (12 cancer divisions through the real preset path), which is
    where the non-vacuous version of this claim lives.
    """
    from cognisom.dashboard.engine_runner import EngineRunner
    from cognisom.core.event_bus import EventTypes

    out = {}
    cancer_divisions = {}
    for enabled in (True, False):
        np.random.seed(12345)
        runner = EngineRunner(
            tissue='ring', dt=0.05, duration=24.0, scenario='Baseline',
            record_interval=1000, snapshot_interval_hours=None,
            overrides={'cellular': {'escape_enabled': enabled}})
        runner.build()
        by_type: dict = {}

        def count(data, _by=by_type):
            _by[data.get('cell_type')] = _by.get(data.get('cell_type'), 0) + 1

        runner.engine.event_bus.subscribe(EventTypes.CELL_DIVIDED, count)
        with contextlib.redirect_stdout(io.StringIO()):
            runner.run()
        cells = runner.engine.modules['cellular']
        out[enabled] = (cells.get_state()['n_cells'],
                        cells.get_state()['n_cancer'],
                        cells.total_divisions, cells.total_deaths,
                        sum(cells.escape_events.values()))
        cancer_divisions[enabled] = by_type.get('cancer', 0)

    assert out[True] == out[False], out
    # 35 -> 10 divisions since division_time_normal became 90 h: this
    # ring preset divides no cancer cells (asserted below), so every
    # division here is a normal one and the count falls with the period.
    # Verified identical under both scorers.
    # 10 -> 11 with the ECM fix: 10 normal divisions and ONE cancer.
    # 11 -> 10 with the macrophage licence, both scorers: ten normal
    # divisions and no cancer one, because the tumour is cleared in ten
    # kills with nothing dividing. See the docstring.
    assert out[True][2] == 10, f"the pinned division count moved: {out[True]}"
    assert out[True][4] == 0, "a de novo event at 1.55e-6 over 10 divisions"
    # The limitation, measured rather than assumed. The docstring above
    # was rewritten when this moved 0 -> 1 and again when it moved back,
    # as its own message says to.
    assert cancer_divisions[True] == 0, (
        f'cancer divisions here moved to {cancer_divisions[True]}; this '
        'run reaches the escape draw that many times, and the docstring '
        'above states the count -- update it rather than re-pinning')
    # Zero draws expect zero events, which is what makes the on/off
    # identity above vacuous rather than evidence.
    assert ev.expected_events(cancer_divisions[True]) == 0.0


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_a_six_hour_slab_run_is_identical_with_escape_on_and_off():
    """The on/off identity on a run that DOES reach the draw.

    The ring presets divide no cancer cells in 24 h, so the pinned ring
    identity above is identity on an unentered code path. The slab does
    divide them: 6 h of Baseline slab, seed 12345, dt 0.05, makes 208
    divisions of which 12 are a cancer cell's -- 12 draws taken from
    self._escape_rng_mhc1 at mu = 1.55e-6, expecting 1.9e-5 events.

    Measured 2026-10-28, escape on and off: 2,241 cells, 85 cancer, 208
    divisions, 6 deaths, mean surface MHC-I 0.354054035 and 0 de novo
    events, identical to the digit. 6 h rather than 24 h because this is
    about crossing the path, and a 24 h pair costs four times as much
    for four times the same answer.
    """
    from cognisom.dashboard.engine_runner import EngineRunner
    from cognisom.core.event_bus import EventTypes

    out = {}
    for enabled in (True, False):
        np.random.seed(12345)
        runner = EngineRunner(
            tissue='slab', dt=0.05, duration=6.0, scenario='Baseline',
            record_interval=1000, snapshot_interval_hours=None,
            overrides={'cellular': {'escape_enabled': enabled}})
        with contextlib.redirect_stdout(io.StringIO()):
            runner.build()
        cells = runner.engine.modules['cellular']
        by_type: dict = {}

        def count(data, _by=by_type):
            _by[data.get('cell_type')] = _by.get(data.get('cell_type'), 0) + 1

        runner.engine.event_bus.subscribe(EventTypes.CELL_DIVIDED, count)
        with contextlib.redirect_stdout(io.StringIO()):
            runner.run()
        state = cells.get_state()
        out[enabled] = (
            state['n_cells'], state['n_cancer'], cells.total_divisions,
            cells.total_deaths, sum(cells.escape_events.values()),
            by_type.get('cancer', 0),
            round(cells.escape_distribution()['mean_mhc1'], 9))

    assert out[True] == out[False], out
    # Was (2241, 85, 208, 6, 0, 12, 0.354054035) when normal cells
    # divided every 24 h. Since the period became 90 h: 142 fewer cells,
    # a third of the divisions (208 -> 66), and a slightly lower mean
    # MHC-I because fewer kills means less IFN-gamma. Cancer divisions
    # went 12 -> 13, which is RNG stream drift -- the cancer period did
    # not move. Verified identical under both scorers.
    # RE-MEASURED 2026-11-08 when the ECM deposition rate was fixed:
    # (2099, 86, 66, 6, 0, 13, 0.346884065) -> this. The freed infiltrate
    # kills more (deaths 6 -> 13), so the tumour is smaller (86 -> 77)
    # and fewer cells remain to divide (cancer divisions 13 -> 11); mean
    # MHC-I rose (0.3469 -> 0.3545) because more kills means more
    # IFN-gamma. Every move is in the direction the mechanism predicts.
    # RE-MEASURED 2026-11-11 when immune cells stopped being confined to
    # the ring's 20-180 um box (the slab reaches 17.5-182.5 um in x; see
    # test_the_infiltrate_can_reach_the_whole_tissue.py). Was
    # (2090, 77, 64, 13, 0, 11, 0.35446766) under both scorers. It is no
    # longer the same under both, so the earlier "identical under both
    # scorers" stopped being true here and each is pinned on its own.
    # Cancer divisions stayed 11 under both, which is what the
    # expectation below is computed from.
    #
    # RE-MEASURED 2026-11-12, both scorers, with the NK serial-kill limit
    # (tests/test_an_nk_cell_is_spent_after_a_few_kills.py). Was
    # (2094, 81, 64, 10, 0, 11, 0.359673066) and
    # (2092, 79, 64, 12, 0, 11, 0.359102551). Cancer divisions went 11 ->
    # 12 under both, so the expectation below moved with them.
    #
    # RE-MEASURED AGAIN the same day with the macrophage phagocytic
    # licence (Dooling 2023; tests/test_a_macrophage_needs_the_target
    # _marked.py), was (2094, 81, 65, 11, 0, 12, 0.356417534) and
    # (2090, 77, 65, 15, 0, 12, 0.355733008). Deaths fell on both --
    # 11 -> 8 and 15 -> 8 -- which is the licence doing exactly what it
    # was measured to do, and the tumour left standing grew with it
    # (81 -> 83, 77 -> 83). SIX OF THE SEVEN FIELDS ARE NOW THE SAME
    # UNDER BOTH SCORERS and only the mean MHC-I separates them, where
    # before the licence four fields differed: the kills that differed
    # between scorers were the ones whose timing ran through the T-cell
    # arm's affinities, and there are fewer of them left. Cancer
    # divisions stayed 12, so the expectation below did not move.
    from cognisom.genomics.mhcflurry_binding import is_mhcflurry_available
    measured = {
        True: (2096, 83, 65, 8, 0, 12, 0.355237414),     # MHCflurry
        False: (2096, 83, 65, 8, 0, 12, 0.354799857),    # PWM fallback
    }[bool(is_mhcflurry_available())]
    assert out[True][:7] == measured, out[True]
    # The point of this test: the draw WAS reached.
    assert out[True][5] > 0
    assert ev.expected_events(out[True][5]) == pytest.approx(
        12 * 1.5497584e-06, rel=0.01)       # 12 cancer divisions
