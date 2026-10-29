"""Is immune escape SELECTED in this model, or does it only drift?

Phase 1 of realism item 1 made escape heritable: a daughter can now
differ from its parent at a sourced rate
(``engine/py/immune/escape_variation.py``), so immune pressure has raw
material. The plan's bar for the item is the next claim, and it is an
empirical one about this engine: "surviving cells are measurably
enriched for escape traits compared with the starting population, which
is selection observed rather than written in."

MEASURED 2026-10-28. THE CLAIM IS NOT MET IN THE SHIPPED PRESETS, and
the reason is specific enough to fix. Selection is real, large and
highly significant at the only place it can arise -- the contact -- and
it is invisible in the population because the arm that can see escape
does under a tenth of the killing.

"Not met" and not "false": the design cannot tell the predicted effect
from zero. This work's own kill data predicts the population shift at
+0.0035 (4.82 escaped cells spared over 12 runs, 0.40 per run against
~115 survivors), which is a THIRD of the +-0.0078 standard error on
the 12-seed contrast, whose 95% interval [-0.018, +0.014] contains both
0 and +0.0035. Resolving +0.0035 at 2 sigma needs the se under 0.00175,
about 240 seeds per arm at the shipped arm's measured sd of 0.0253 --
20x this design. What IS resolved is the all-T-cell arm, at +0.0554
with t = 4.43: the measurement can see selection when the kill mix
allows it, so the null below is about the kill mix, not about the
method.

WHICH SCORER. Every exact pin below is taken under a NAMED peptide
binding predictor, because the slab runs are not scorer-independent:
the 24 h Baseline slab run ends with 115 survivors and 21 kills under
the position-weight-matrix fallback (py3.13, no MHCflurry) and 110
survivors and 23 kills under the real MHCflurry 2.0.6 (py3.11, which
is what both GitHub workflows and production use). The arena tests in
parts 1, 2 and 4 build their presentations by hand and are identical
under both. See SHIPPED_24H_PINS.

The population result, which is a null
-------------------------------------
Baseline and Aggressive Tumor tissue slabs, 24 h, dt 0.05, half the
founding cancer cells (80 and 240) seeded with one-ALLELE HLA loss
(``escape_founder_hla_loh_fraction`` 0.5 -- the de novo rate cannot
supply an event at this scale, see below), 12 seeds per arm, 96 runs.
delta is the escaped fraction among survivors minus the escaped
fraction among founders, and every arm carries the SAME total
infiltrate so that only the lineage mix differs:

  preset       arm            T/NK/M     delta       sd      vs control  t
  Baseline     immune removed  0/0/0    +0.0062   0.0093          --     --
  Baseline     shipped        19/6/35   +0.0043   0.0253     -0.0020  -0.26
  Baseline     all T cells    60/0/0    +0.0617   0.0424     +0.0554  +4.43
  Baseline     no T cells      0/9/51   -0.0198   0.0390     -0.0260  -2.25
  Aggressive   immune removed  0/0/0    -0.0026   0.0055          --     --
  Aggressive   shipped        10/3/18   +0.0031   0.0078     +0.0057  +2.06
  Aggressive   all T cells    31/0/0    +0.0178   0.0099     +0.0204  +6.21
  Aggressive   no T cells      0/4/27   -0.0083   0.0103     -0.0057  -1.71

THE NOISE FLOOR IS SET BY THE NOISIER ARM, AND THAT IS THE TREATMENT
ARM, NOT THE CONTROL. The drift arm's sd is 0.0093 per run on Baseline
and 0.0055 on Aggressive, but the shipped Baseline arm's is 0.0253, so
the CONTRAST's standard error is 0.0078 and the smallest delta this
design resolves on Baseline is 0.0143 -- not the 0.005 the control
arm's sd alone would suggest. The shipped arms do not clear it
(t = -0.26, and +2.06 against a 95% two-sided bound of 2.07 at 22 df --
a miss, not a pass). The all-T-cell arms clear it by a factor of four.
Immune killing is itself what makes the measurement noisy on Baseline,
sd 0.0093 -> 0.0253.

Differential proliferation is not hiding anything in the shipped arms:
the escaped share of cancer divisions was 0.5058 against a population
share of 0.5053 (Baseline) and 0.5017 against 0.5023 (Aggressive). In
the all-T-cell arm it does diverge, 0.5574 against 0.5340, which is
the downstream consequence of escaped cells living longer and not an
independent growth advantage -- nothing in the cellular module's
division path reads mhc1_expression.

The mechanism result, which is not a null
-----------------------------------------
Kills join to the phenotype of the cell killed, against a TIME-MATCHED
null -- for each kill, the escaped fraction of the living tumour at that
instant, summed to E. O/E below 1 means escaped cells were spared. Two
errors are given because they answer different objections: the naive z
treats each kill as independent, and the robust t treats each SEED as
the unit, which is the honest one because a daughter is placed next to
its parent and inherits its phenotype, so kills cluster by clone.

  preset      arm       lineage    kills    O      E     O/E   z     t
  Baseline    shipped   T cell        23    1   11.96  0.084  -4.6  -4.8
  Baseline    shipped   NK cell      124   67   62.62  1.070  +0.8  +0.9
  Baseline    shipped   macrophage    87   45   43.24  1.041  +0.4  +0.4
  Baseline    shipped   all          234  113  117.82  0.959  -0.6  -0.8
  Baseline    all T     T cell       302  104  165.55  0.628  -7.2  -5.1
  Aggressive  shipped   T cell        15    1    7.36  0.136  -3.3  -4.0
  Aggressive  shipped   all          224  106  112.73  0.940  -0.9  -1.0
  Aggressive  all T     T cell       218   25  111.04  0.225 -11.7 -11.3

The T-cell arm selects FOR escape overwhelmingly on both presets -- one
escaped cell killed where twelve were due on Baseline, one where seven
were due on Aggressive -- and it is 23 of 234 and 15 of 224 kills. The
rest are neutral, so the population sees nothing: the shipped null is
DILUTION, not cancellation. Three facts, each pinned below, explain
every row:

* the macrophage path never reads MHC-I. Its per-contact probability is
  0.05 at surface class I 0.00, 0.20, 0.30, 0.45 and 1.00 alike -- 58%
  of the shipped slab infiltrate is blind to the escape trait by
  construction.
* the NK path is the legacy missing-self STEP at 0.4 with a flat 0.15.
  An escaped cell at 0.20 and an intact one at 0.30 are both below it,
  so NK is neutral at the seeded levels -- and its measured O/E of 1.07
  is noise, not biology.

  BUT THE SHIPPED RUN DOES NOT STAY IN THAT REGIME, and the earlier
  version of this paragraph said it did. Sampling the same 24 h
  Baseline slab run every hour and taking the MAXIMUM over the run
  rather than the end-of-run snapshot:

    scorer           hourly samples with an intact cell at/above 0.4
    PWM fallback       4 of 24 (h 17.3-20.3, 1 intact cell), max 0.4010
    real MHCflurry    16 of 24 (h 8.3 onward, 2-3 cells), max 0.4252

  Under the real predictor -- CI and production -- intact cancer cells
  are over the step for two thirds of the run and 2 of them are over it
  at the final frame. Escaped cells never cross it under either scorer
  (max 0.2653 and 0.2835). So the counter-selection is LIVE in a
  shipped preset, not latent: an intact cell leaves NK's view while its
  escaped neighbour stays in it, so NK kills the escaped one. The
  mechanism in isolation: raise IFN-gamma to saturation and the intact
  cell reaches 0.30 x 1.5 = 0.45 while the escaped one reaches only
  0.20 x 1.5 = 0.30, and in a 12 h arena at that drive NK cells killed
  187 escaped cells and 0 intact ones over 12 seeds -- escape becomes a
  liability, which is backwards, and it is an artefact of an unsourced
  step function rather than of any biology. The end-of-run snapshot
  hides it under the PWM scorer, which is why the regime is now
  asserted on the hourly series.
* the T-cell Hill rule is the only one that grades its killing by
  MHC-I, and it grades it steeply: 0.0955 per contact at 0.30 against
  0.0204 at 0.20, a 4.7x drop for one allele, and exactly 0 below
  the 0.05 recognition floor.

What would make selection visible
---------------------------------
Not a longer run and not a higher mutation rate -- a different kill
budget. The T-cell differential is already significant at 23 kills; the
innate arm's 211 neutral kills are what dilute it. With the same total
infiltrate made all T cells, the same 12 h arena gives 3 escaped kills
against 36 intact (escaped share 0.077, z = -5.3); with the shipped
19/6/35 ratio it gives 65 against 83 (0.439, z = -1.5) and with full
IFN drive 90 against 94 (0.489, z = -0.3). That is the finding: this
engine's selection coefficient on HLA loss is set by its T/NK/
macrophage kill mix, which realism item 2 is separately calibrating,
and not by anything in escape_variation.

For the de novo route the scale problem is arithmetic and already in
escape_variation: 1.55e-6 per division needs 6.5e5 CANCER divisions for
one expected event, against 53-56 in a 24 h Baseline slab run -- of 846
to 849 divisions in total, because only a cancer daughter draws
(_maybe_escape returns None for every other cell type). Every number
above therefore comes from a FOUNDER-SEEDED run. A run that waits for
de novo escape needs ~11,000x more cancer divisions than 24 h on this
slab provides. On the RING presets that factor is infinite: a 24 h ring
Baseline run makes 35 divisions and none of them is a cancer cell's, so
a ring run never reaches the draw at all.

Does the HERITABLE route reach selection, and not only an imposed
phenotype? Yes, in one configuration, and it is the last test here
----------------------------------------------------------------------
Every measurement above imposes the phenotype -- by hand in the arenas,
by founder seeding in the preset runs -- so on its own this file would
show selection acting on written-in variation and never on variation
the engine produced. test_de_novo_escape_arises_and_is_then_selected
joins the two halves: zero escaped founders, mu scaled DELIBERATELY to
0.5 per division and labelled as scaled, 12 T cells against a growing
tumour. 85 de novo events arise from nothing, the escaped share of
survivors rises to 0.547 against an immune-off control's 0.447, and the
kills select against escape at O/E 0.32, z = -3.5 over 114 kills. That
is the only configuration in which this engine can show escape ARISING
and then being SELECTED, and the mu it needs is 3 x 10^5 times the
literature rate.

De novo escape DID happen twice, and both times it was seed 5
-------------------------------------------------------------
Across the 12 Aggressive Tumor runs whose de novo counter could be read
separately from founder seeding, there were 2,707 cancer divisions,
0.0042 expected events, and ONE observed -- a 1-in-240 outcome, which
is the sort of thing that gets a sourced rate "corrected". The
all-T-cell arm at the same seeds produced one event too, and that one
landed on a cell already carrying an allele loss, so it fired the
SECOND hit and left exactly one T-cell-invisible cell in the run
(n_below_recognition 1, n_escaped_without_display 1). Both events were
seed 5, which is the diagnosis: one unlucky stream, hit at the same
position, landing on an intact cell in one arm and a hemizygous one in
the other, so the same draw produced a partial loss in one run and a
total loss in the other. It is not the rate: the stream
``numpy.random.default_rng([5, 0])`` carries a draw of 1.360e-6 inside
its first 260 values, below mu = 1.5498e-6.
Checked over 2,000 such streams, 2 carry a sub-mu draw in 260 against
0.81 expected, the mean of the per-stream minimum is 0.00385 against
the theoretical 1/(N+1) = 0.00383, the median 0.00269 against 0.00266,
and a KS test of the transformed minima against U(0,1) gives p = 0.98.
The seeding is uniform and seed 5 was unlucky.
"""
from __future__ import annotations

import contextlib
import io
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.core import SimulationConfig, SimulationEngine
from cognisom.core.event_bus import EventTypes
from cognisom.engine.py.immune import escape_variation as ev
from cognisom.engine.py.immune.kill_calibration import (
    MACROPHAGE_KILL_PER_CONTACT, NK_KILL_PER_CONTACT)
from cognisom.engine.py.immune.macrophage_phagocytosis import (
    phagocytosis_probability)
from cognisom.engine.py.immune.nk_receptor import (
    DEFAULT_MISSING_SELF_THRESHOLD, missing_self_recognized)
from cognisom.engine.py.immune.tcell_kill import kill_probability
from cognisom.modules.cellular_module import MHC1_IFN_GAIN, CellularModule
from cognisom.modules.immune_module import ImmuneModule
from cognisom.modules.molecular_module import MolecularModule

#: Surface class I an untouched cancer cell is seeded at (three sites in
#: cellular_module), and what the loss of ONE ALLELE leaves on the
#: engine's 3-allele default panel. (n-1)/n is a per-allele magnitude:
#: a whole class I haplotype is three alleles in cis and would leave
#: 0.5 on a full 6-allele genotype, not 2/3 -- see escape_variation.)
MHC1_SEEDED = 0.30
MHC1_AFTER_ONE_ALLELE_LOST = MHC1_SEEDED * ev.surviving_surface_fraction(3)

#: T / NK / macrophage in the shipped tissue slab, split by the TCGA PRAD
#: CIBERSORT medians (engine/py/immune/kill_calibration). Macrophages are
#: 58% of it and read no MHC-I at all.
SLAB_INFILTRATE = (19, 6, 35)


# ── the arena ────────────────────────────────────────────────────────────
#
# Both phenotypes in ONE run, interleaved in position so neither sits
# closer to the effectors, with division and spontaneous death switched
# off so every death is a kill. That makes the escaped share of kills a
# direct selection coefficient per contact, against a null of 0.5 -- and
# it resolves in seconds what a 24 h preset run resolves in 20 kills.

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


def _lineup(n_t: int = 0, n_nk: int = 0, n_mac: int = 0,
            checkpoint_block: float = 0.0, dt: float = 0.05,
            hours: float = 6.0):
    """An engine with no seeded population and the given effectors."""
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
    engine.register_module('immune', ImmuneModule, {
        'n_t_cells': n_t, 'n_nk_cells': n_nk, 'n_macrophages': n_mac,
        # A permissive repertoire, so recognition turns on the cell's
        # MHC-I and its presentations rather than on whether this seed
        # drew a matching clone.
        'tcr_recognition_threshold': 0.0, 'tcr_repertoire_size': 200,
        'tcr_seed': 0, 'costimulation': 0.6,
        'checkpoint_block': checkpoint_block,
        # M0: the polarization factor is an exact identity there, so the
        # macrophage measurement is the PD-1 gate and nothing else.
        'macrophage_polarization_enabled': False})
    with contextlib.redirect_stdout(io.StringIO()):
        engine.initialize()
    cellular = engine.modules['cellular']
    immune = engine.modules['immune']
    immune.set_cellular_module(cellular)
    return engine, cellular, immune


def _dress(cellular: CellularModule, cell, phenotype: str,
           ifn_drive: float) -> None:
    """Put one cancer cell into the named phenotype.

    ``ifn_drive`` places the cell at the STEADY STATE of the IFN-gamma
    response, mhc1_baseline * (1 + MHC1_IFN_GAIN * drive), with no field
    present -- so the measurement is the threshold effect and not how
    far up a 6 h exponential the run got. The externally validated gain
    itself is untouched.
    """
    cell.mhc1_expression = MHC1_SEEDED
    cell.mhc1_displayed_peptides = _presentations(cellular.hla_alleles)
    cell.pdl1_expression = 0.0
    if phenotype == 'allele_loss':
        cellular._lose_one_allele(cell)
    elif phenotype == 'total_loss':
        cellular._lose_mhc1_entirely(cell)
    elif phenotype == 'pdl1_high':
        cell.pdl1_expression = 1.0
    elif phenotype != 'intact':
        raise ValueError(phenotype)
    cell.mhc1_baseline = float(cell.mhc1_expression)
    cell.mhc1_expression = float(
        cell.mhc1_baseline * (1.0 + MHC1_IFN_GAIN * ifn_drive))
    cell.ifn_gamma_level = None


def arena(escaped_phenotype: str, n_t: int = 0, n_nk: int = 0,
          n_mac: int = 0, checkpoint_block: float = 0.0, seed: int = 1,
          n_each: int = 20, hours: float = 6.0,
          ifn_drive: float = 0.0) -> Dict[str, object]:
    """Kills on ``n_each`` intact and ``n_each`` escaped cells, mixed.

    Returns the kills by (lineage, phenotype) plus the two totals. The
    pair at each site is offset by 1.5 um either side of one draw, so
    distance to every effector is matched.
    """
    np.random.seed(seed)
    engine, cellular, immune = _lineup(
        n_t=n_t, n_nk=n_nk, n_mac=n_mac,
        checkpoint_block=checkpoint_block, hours=hours)
    rng = np.random.default_rng(seed)
    centre = np.array([100.0, 100.0, 50.0])
    phenotype_of: Dict[int, str] = {}
    for _ in range(n_each):
        base = centre + rng.uniform(-11, 11, 3)
        for which, offset in (('intact', -1.5), ('escaped', 1.5)):
            pos = base + np.array([offset, 0.0, 0.0])
            cid = cellular.add_cell(position=pos.tolist(),
                                    cell_type='cancer')
            _dress(cellular, cellular.cells[cid],
                   'intact' if which == 'intact' else escaped_phenotype,
                   ifn_drive)
            phenotype_of[cid] = which
    for icell in immune.immune_cells.values():
        icell.position = centre + rng.uniform(-11, 11, 3)

    kills: Counter = Counter()
    engine.event_bus.subscribe(
        EventTypes.CANCER_KILLED,
        lambda d: kills.update(
            [(d.get('killer_type'), phenotype_of.get(d['cell_id'], '?'))]))
    with contextlib.redirect_stdout(io.StringIO()):
        engine.run()
    return {
        'kills': dict(kills),
        'escaped': sum(v for k, v in kills.items() if k[1] == 'escaped'),
        'intact': sum(v for k, v in kills.items() if k[1] == 'intact'),
    }


def _z_vs_half(escaped: int, intact: int) -> Optional[float]:
    """z for the escaped share of kills against a neutral 0.5."""
    n = escaped + intact
    if n == 0:
        return None
    return (escaped / n - 0.5) / math.sqrt(0.25 / n)


# ── part 1: what each lineage can see, as pure functions ─────────────────

def test_the_macrophage_arm_is_blind_to_mhc1_to_the_bit():
    """58% of the shipped slab infiltrate cannot see the escape trait at
    all, which is most of why the population shows no selection.
    _macrophage_kill_probability reads pdl1_expression and the
    effector's polarization; MHC-I appears nowhere in it."""
    probs = {
        mhc: phagocytosis_probability(
            baseline=MACROPHAGE_KILL_PER_CONTACT.value,
            pdl1_expression=0.0, checkpoint_block=0.0)
        for mhc in (0.0, MHC1_AFTER_ONE_ALLELE_LOST, MHC1_SEEDED,
                    0.45, 1.0)}
    assert set(probs.values()) == {MACROPHAGE_KILL_PER_CONTACT.value}
    assert MACROPHAGE_KILL_PER_CONTACT.value == 0.05


def test_the_nk_arm_is_a_step_and_escape_sits_on_the_wrong_side_of_it():
    """The legacy missing-self rule is ``mhc1 < 0.4`` with a flat 0.15.
    An escaped cell at 0.20 and an intact one at 0.30 are both below it,
    so NK is NEUTRAL on escape -- until IFN-gamma lifts the intact cell
    to 0.45, over the step, while the escaped one reaches only 0.30.
    Then NK kills the escaped cell and not its intact neighbour, which
    is counter-selection, and it is the sign the shipped mix pays."""
    thr = DEFAULT_MISSING_SELF_THRESHOLD
    assert thr == 0.4
    # No IFN-gamma: both visible, so the differential is exactly zero.
    assert missing_self_recognized(MHC1_SEEDED, thr)
    assert missing_self_recognized(MHC1_AFTER_ONE_ALLELE_LOST, thr)
    # Full IFN-gamma drive: only the escaped cell is still visible.
    intact_hi = MHC1_SEEDED * (1.0 + MHC1_IFN_GAIN)
    escaped_hi = MHC1_AFTER_ONE_ALLELE_LOST * (1.0 + MHC1_IFN_GAIN)
    assert intact_hi == pytest.approx(0.45)
    assert escaped_hi == pytest.approx(0.30)
    assert not missing_self_recognized(intact_hi, thr)
    assert missing_self_recognized(escaped_hi, thr)
    # A total loss is maximally visible to NK, which is correct biology
    # and the opposite of an escape.
    assert missing_self_recognized(0.0, thr)
    assert NK_KILL_PER_CONTACT.value == 0.15


def test_only_the_t_cell_arm_grades_its_killing_by_mhc1():
    """And it grades it steeply: one allele costs 4.7x per contact,
    and below the 0.05 recognition floor the probability is exactly 0.
    This is the whole selection differential the engine has."""
    p_intact = kill_probability(0.95, MHC1_SEEDED, 0.6)
    p_escaped = kill_probability(0.95, MHC1_AFTER_ONE_ALLELE_LOST, 0.6)
    assert p_intact == pytest.approx(0.095481, abs=1e-6)
    assert p_escaped == pytest.approx(0.020425, abs=1e-6)
    assert p_intact / p_escaped == pytest.approx(4.675, rel=1e-3)
    assert kill_probability(0.95, 0.0, 0.6) == 0.0


def test_the_t_cell_arm_never_reads_the_targets_pdl1():
    """A gap, found while measuring this, and the reason heritable PD-L1
    is not a T-cell escape route in this engine at all.

    ``_target_kill_probability``'s T-cell branch calls kill_outcome with
    affinity, the target's MHC-I, the module's costimulation and
    checkpoint_block. It does not pass pdl1_expression, and
    kill_probability's ICB-rescue term is (1 - costim) * block * 0.5 --
    target-independent. So blockade raises T-cell killing of every
    recognised cell by the same amount whether that cell is PD-L1-null
    or PD-L1-saturated. The macrophage path DOES read it, and that is
    the only place PD-L1 escape exists here.
    """
    np.random.seed(0)
    _engine, cellular, immune = _lineup(n_t=1, n_mac=1)
    by_type = {c.cell_type: c for c in immune.immune_cells.values()}
    t_cell, macrophage = by_type['T_cell'], by_type['macrophage']

    probs = []
    for pdl1 in (0.0, 0.5, 1.0):
        cid = cellular.add_cell(position=[100, 100, 50], cell_type='cancer')
        cell = cellular.cells[cid]
        _dress(cellular, cell, 'intact', 0.0)
        cell.pdl1_expression = pdl1
        recognized, match = immune._recognize(t_cell, cell)
        assert recognized
        t_cell.active_tcr_match = match
        probs.append(immune._target_kill_probability(t_cell, cell))
        if pdl1 == 1.0:
            # The same cell, same step, read by a macrophage instead.
            assert immune._macrophage_kill_probability(cell, macrophage) \
                < MACROPHAGE_KILL_PER_CONTACT.value
    assert len(set(probs)) == 1, (
        'the T-cell kill probability moved with the target PD-L1; the '
        'branch now reads it, so this gap is closed and the docstring '
        'above is stale')


# ── part 2: selection measured where it arises, at a contact ─────────────

def test_a_t_cell_never_kills_a_cell_that_lost_mhc1_even_under_blockade():
    """The structural claim of the whole item. 20 intact and 20
    total-loss cells in one arena, 20 T cells, 6 h, seeds 1-3: the
    escaped half takes 0 kills at checkpoint_block 0.0 AND at 1.0,
    while the intact half goes 13 -> 49. Blockade acts downstream of
    recognition, and _recognize has already returned (False, None)."""
    for block, expect_intact in ((0.0, 13), (1.0, 49)):
        escaped = intact = 0
        for seed in (1, 2, 3):
            out = arena('total_loss', n_t=20, checkpoint_block=block,
                        seed=seed)
            escaped += out['escaped']
            intact += out['intact']
        assert escaped == 0, (
            f'a T cell killed an MHC-I-null cell at block {block}')
        assert intact == expect_intact


def test_a_partial_loss_is_rescued_by_blockade_and_a_total_loss_is_not():
    """Why the two MHC-I routes had to stay structurally apart -- and the
    half of the item's done-when that this engine does NOT support.

    A one-allele loss keeps the cell recognised at a lower signal: it
    takes 0 kills against the intact half's 7 without blockade, and
    blockade brings it all the way back to PARITY, 30 against 31. A
    total loss is 0 at both (test above).

    WHAT THAT DOES AND DOES NOT ESTABLISH. It establishes that blockade
    separates PARTIAL from TOTAL MHC-I loss. It does NOT establish the
    done-when, which was "checkpoint blockade restores killing only
    where PD-L1 was the escape route" -- this docstring used to claim it
    did, and the test's own numbers contradict that claim. Blockade
    rescues the HLA-loss route, the one route with a sourced rate and
    the only one founder seeding uses, and it rescues it HARDER than it
    helps an intact cell. Measured per contact at affinity 0.95 and
    costimulation 0.6:

      target MHC-I   p_kill at block 0.0   at block 1.0   ratio
          0.20 (escaped)      0.0204           0.5455     26.7x
          0.30 (intact)       0.0955           0.7005      7.3x

    The escape penalty shrinks from 0.21 to 0.78 of the intact cell.
    The cause is structural and is pinned two tests above
    (test_the_t_cell_arm_never_reads_the_targets_pdl1): the rescue term
    is (1 - costim) * block * 0.5, which contains neither affinity nor
    MHC-I nor the target's PD-L1. Total loss resists blockade only
    because _recognize hard-gates below 0.05 -- the rule itself still
    returns p = 0.165 at MHC-I 0.0 and p = 0.233 at 0.04 under full
    blockade -- so it is a gate, not a graded mechanism. And the PD-L1
    half of the criterion is demonstrated only on a hand-set
    pdl1_expression of 1.0 against a MACROPHAGE (next test), because
    PDL1_HERITABLE_GAIN_PER_DIVISION is 0.0 and the T-cell arm never
    reads PD-L1 at all.

    So: total MHC-I loss is not rescuable by blockade; PD-L1 gain is,
    on the macrophage arm; partial MHC-I loss is ALSO rescued, because
    the ICB term is independent of MHC-I. Making the three separable
    means rewriting that term to scale with costimulation and with the
    target's PD-L1 (Hui 2017 PMID 28280247, Kamphorst 2017 PMID
    28280249), which changes the flagship Immunotherapy preset and is
    not this change.
    """
    open_ = [arena('allele_loss', n_t=20, checkpoint_block=0.0, seed=s)
             for s in (1, 2, 3)]
    blocked = [arena('allele_loss', n_t=20, checkpoint_block=1.0, seed=s)
               for s in (1, 2, 3)]
    assert (sum(o['escaped'] for o in open_),
            sum(o['intact'] for o in open_)) == (0, 7)
    assert (sum(b['escaped'] for b in blocked),
            sum(b['intact'] for b in blocked)) == (30, 31)

    # The per-contact numbers behind the paragraph above, so the claim
    # "blockade helps the escaped cell MORE" is pinned to the rule and
    # not left as prose. If the ICB term is ever made MHC-I-dependent
    # these ratios change and this assertion is where it shows.
    esc_open = kill_probability(0.95, MHC1_AFTER_ONE_ALLELE_LOST, 0.6, 0.0)
    esc_block = kill_probability(0.95, MHC1_AFTER_ONE_ALLELE_LOST, 0.6, 1.0)
    int_open = kill_probability(0.95, MHC1_SEEDED, 0.6, 0.0)
    int_block = kill_probability(0.95, MHC1_SEEDED, 0.6, 1.0)
    assert esc_open == pytest.approx(0.0204255, abs=1e-6)
    assert esc_block == pytest.approx(0.5454844, abs=1e-6)
    assert int_open == pytest.approx(0.0954810, abs=1e-6)
    assert int_block == pytest.approx(0.7004997, abs=1e-6)
    assert esc_block / esc_open > int_block / int_open
    assert esc_open / int_open == pytest.approx(0.2139, abs=1e-3)
    assert esc_block / int_block == pytest.approx(0.7787, abs=1e-3)
    # A cell just under the recognition floor would be rescued too, so
    # the "blockade cannot rescue MHC-I loss" result is the floor and
    # nothing else.
    assert kill_probability(0.95, 0.04, 0.6, 1.0) == pytest.approx(
        0.2332, abs=1e-3)
    assert kill_probability(0.95, 0.0, 0.6, 1.0) == pytest.approx(
        0.1649, abs=1e-3)


def test_blockade_restores_phagocytosis_of_a_pdl1_high_cell():
    """The other half of the specificity claim, and the only escape
    route blockade rescues here: PD-L1 gates a macrophage that has
    already recognised its target, so relieving the gate restores the
    killing. 20 macrophages, 6 h, seeds 1-3: the PD-L1-high half takes
    6 kills against the PD-L1-null half's 17, and under blockade 14
    against the same 17 -- the intact half does not move, because its
    PD-L1 was 0 and the gate was already the configured baseline."""
    open_ = [arena('pdl1_high', n_mac=20, checkpoint_block=0.0, seed=s)
             for s in (1, 2, 3)]
    blocked = [arena('pdl1_high', n_mac=20, checkpoint_block=1.0, seed=s)
               for s in (1, 2, 3)]
    assert (sum(o['escaped'] for o in open_),
            sum(o['intact'] for o in open_)) == (6, 17)
    assert (sum(b['escaped'] for b in blocked),
            sum(b['intact'] for b in blocked)) == (14, 17)


# 72 arenas, measured at 11.9 s on this machine under load, so this one
# stays in the fast lane -- pytest.ini's marker is for tests over ~30 s
# and the repo has been bitten by marking cheap engine tests slow. The
# explicit timeout is 25x the measured cost, not the global 120 s.
@pytest.mark.timeout(300)
def test_the_t_cell_arm_selects_for_escape_and_the_innate_arm_against_it():
    """The measurement that explains the population null, over 12 seeds
    of a 12 h arena with 20 effectors and 20 cells of each phenotype.

    The escaped share of kills against a neutral 0.5:

      effectors          IFN drive   escaped   intact   share   z
      20 T cells             0.0         3       36     0.077   -5.3
      20 T cells             1.0        71      101     0.413   -2.3
      20 NK cells            0.0       176      187     0.485   -0.6
      20 NK cells            1.0       187        0     1.000  +13.7
      20 macrophages       0.0 / 1.0   100      109     0.478   -0.6

    Three different signs from three lineages on the same trait. The
    macrophage row is identical at both IFN drives to the kill, because
    that path reads no MHC-I; the NK row flips from exactly neutral to
    total counter-selection as IFN-gamma carries the INTACT cell over
    the 0.4 step.
    """
    def total(**kw):
        out = [arena('allele_loss', hours=12.0, seed=s, **kw)
               for s in range(1, 13)]
        return (sum(o['escaped'] for o in out),
                sum(o['intact'] for o in out))

    assert total(n_t=20, ifn_drive=0.0) == (3, 36)
    assert total(n_t=20, ifn_drive=1.0) == (71, 101)
    assert total(n_nk=20, ifn_drive=0.0) == (176, 187)
    # Counter-selection, total: every intact cell is NK-invisible at
    # 0.45 and every escaped one is still a target at 0.30.
    assert total(n_nk=20, ifn_drive=1.0) == (187, 0)
    # MHC-blind, so IFN-gamma changes nothing at all.
    assert total(n_mac=20, ifn_drive=0.0) == (100, 109)
    assert total(n_mac=20, ifn_drive=1.0) == (100, 109)

    assert _z_vs_half(3, 36) < -5.0          # T cells: selection FOR
    assert _z_vs_half(187, 0) > 13.0         # NK + IFN: selection AGAINST
    assert -1.0 < _z_vs_half(100, 109) < 0.0  # macrophages: neutral


@pytest.mark.timeout(300)   # 24 arenas, measured at 3.9 s under load
def test_in_the_shipped_infiltrate_the_two_arms_cancel():
    """The shipped slab ratio, 19 T / 6 NK / 35 macrophage scaled to 20,
    over 12 seeds of the same 12 h arena. The escaped share of kills is
    not distinguishable from neutral in either IFN-gamma regime: 65
    against 83 (0.439, z = -1.5) without it, and 90 against 94 (0.489,
    z = -0.3) with it. The two rows fail for different reasons, and the
    difference is worth keeping: at drive 0 the T-cell signal is merely
    DILUTED by two neutral lineages, and at drive 1 it is also OFFSET,
    because NK has stopped seeing the intact cell. Compare the
    all-T-cell arm's z = -5.3 at the SAME total number of effectors:
    what suppresses selection is the kill mix, not the escape
    mechanism."""
    mix = {'n_t': 6, 'n_nk': 2, 'n_mac': 12}
    out = [arena('allele_loss', hours=12.0, seed=s, ifn_drive=0.0, **mix)
           for s in range(1, 13)]
    hi = [arena('allele_loss', hours=12.0, seed=s, ifn_drive=1.0, **mix)
          for s in range(1, 13)]
    flat = (sum(o['escaped'] for o in out), sum(o['intact'] for o in out))
    lifted = (sum(o['escaped'] for o in hi), sum(o['intact'] for o in hi))
    assert flat == (65, 83)
    assert lifted == (90, 94)
    assert abs(_z_vs_half(*flat)) < 1.96
    assert abs(_z_vs_half(*lifted)) < 1.96


# ── part 3: the population, which is where the plan's bar is ─────────────

def _preset_run(arm: str, seed: int, founder: float = 0.5,
                duration: float = 24.0) -> Dict[str, object]:
    """One 24 h Baseline tissue-slab run, with the escape phenotype of
    every cancer cell joined to how it died.

    ``arm`` is 'immune_on' (the shipped infiltrate) or 'immune_off' (the
    immune module removed, which is the drift control).

    Two things are recorded beyond the end state, and both were added
    because the end state alone said something false.

    DIVISIONS BY PARENT TYPE, because only a cancer daughter draws for
    escape (_maybe_escape returns None for every other cell type), so
    the total division count is the wrong denominator for any
    expectation: 846-849 total against 53-56 cancer, a factor of 15.

    THE MHC-I REGIME HOURLY, not at the final frame. The NK arm sees a
    cell only below mhc1 0.4, so whether intact cells cross that step
    decides the SIGN of the NK contribution -- and under the PWM
    fallback they cross it mid-run (h 17-20) and fall back under it by
    hour 24, so a final-frame assertion reports a regime the run was not
    in. The engine's step is wrapped rather than using
    snapshot_interval_hours because the snapshot carries no per-cell
    MHC-I maximum.
    """
    from cognisom.dashboard.engine_runner import EngineRunner
    modules = {
        'cellular': True, 'immune': arm == 'immune_on', 'vascular': True,
        'lymphatic': True, 'molecular': True, 'spatial': True,
        'epigenetic': True, 'circadian': True, 'morphogen': True,
        'cell_mechanics': False, 'ar_signaling': True}
    np.random.seed(seed)
    runner = EngineRunner(
        tissue='slab', dt=0.05, duration=duration, scenario='Baseline',
        modules_enabled=modules, record_interval=1000,
        snapshot_interval_hours=None,
        overrides={'cellular': {
            'escape_founder_hla_loh_fraction': founder,
            'escape_seed': seed}})
    with contextlib.redirect_stdout(io.StringIO()):
        runner.build()
    cells = runner.engine.modules['cellular']
    escaped_of = {cid: bool(c.escape_traits)
                  for cid, c in cells.cells.items()
                  if c.cell_type == 'cancer'}
    founders = dict(escaped_of)
    kills: List[Tuple[str, bool, float]] = []
    divisions_by_parent: Counter = Counter()

    def on_divided(data):
        divisions_by_parent[data.get('cell_type')] += 1
        daughter = cells.cells.get(data.get('daughter_id'))
        if daughter is not None and daughter.cell_type == 'cancer':
            escaped_of[daughter.cell_id] = bool(daughter.escape_traits)

    def on_killed(data):
        # The cellular module subscribed in initialize(), so the target
        # is already removed by the time this runs; add it back to get
        # the composition the killer was drawing from.
        living = [c for c in cells.cells.values()
                  if c.alive and c.cell_type == 'cancer']
        n = len(living) + 1
        esc = sum(1 for c in living if c.escape_traits)
        esc += 1 if escaped_of.get(data['cell_id']) else 0
        kills.append((data.get('killer_type'),
                      bool(escaped_of.get(data['cell_id'])), esc / n))

    runner.engine.event_bus.subscribe(EventTypes.CELL_DIVIDED, on_divided)
    runner.engine.event_bus.subscribe(EventTypes.CANCER_KILLED, on_killed)

    # Hourly MHC-I regime, sampled by wrapping the engine's step.
    regime: List[Dict[str, object]] = []
    engine = runner.engine
    original_step = engine.step
    next_sample = [0.0]

    def sampled_step(*args, **kwargs):
        out = original_step(*args, **kwargs)
        # SimulationEngine.step advances self.time by dt (line 195), so
        # this reads the time the sample belongs to. Reading a
        # non-existent attribute with a 0.0 default silently produced
        # ONE sample and a time series that never advanced.
        now = float(engine.time)
        if now >= next_sample[0]:
            next_sample[0] = now + 1.0
            live = [c for c in cells.cells.values()
                    if c.alive and c.cell_type == 'cancer']
            i = [c.mhc1_expression for c in live if not c.escape_traits]
            e = [c.mhc1_expression for c in live if c.escape_traits]
            regime.append({
                'h': now,
                'intact_max': max(i) if i else None,
                'intact_above': sum(
                    1 for x in i if x >= DEFAULT_MISSING_SELF_THRESHOLD),
                'escaped_max': max(e) if e else None,
                'escaped_above': sum(
                    1 for x in e if x >= DEFAULT_MISSING_SELF_THRESHOLD)})
        return out

    engine.step = sampled_step
    with contextlib.redirect_stdout(io.StringIO()):
        runner.run()

    survivors = [c for c in cells.cells.values()
                 if c.alive and c.cell_type == 'cancer']
    intact = [c.mhc1_expression for c in survivors if not c.escape_traits]
    escaped = [c.mhc1_expression for c in survivors if c.escape_traits]
    return {
        'founders': len(founders),
        'founders_escaped': sum(founders.values()),
        'survivors': len(survivors),
        'survivors_escaped': len(escaped),
        'kills': kills,
        # De novo only. Founder seeding is counted in
        # escape_founder_events, which it was NOT before this phase: it
        # shared _record_escape with the division path, so a
        # founder-seeded run reported 42 'events' and the de novo count
        # -- the number that says whether the literature rate can do
        # anything at this scale -- was unreadable.
        'de_novo': dict(cells.escape_events),
        'founder_events': dict(cells.escape_founder_events),
        # The IFN-gamma regime, which decides the SIGN of the NK arm's
        # contribution: NK sees a cell only below mhc1 0.4.
        'mhc1_intact_mean': float(np.mean(intact)) if intact else None,
        'mhc1_intact_max': float(np.max(intact)) if intact else None,
        'intact_above_nk_step': sum(
            1 for x in intact if x >= DEFAULT_MISSING_SELF_THRESHOLD),
        'mhc1_escaped_mean': float(np.mean(escaped)) if escaped else None,
        'escaped_above_nk_step': sum(
            1 for x in escaped if x >= DEFAULT_MISSING_SELF_THRESHOLD),
        # Only a cancer daughter draws, so this -- not total_divisions --
        # is the denominator for expected de novo events.
        'divisions_total': int(sum(divisions_by_parent.values())),
        'cancer_divisions': int(divisions_by_parent['cancer']),
        # The regime over the WHOLE run, which is what decides the NK
        # arm's sign; the final frame can be in a different one.
        'n_regime_samples': len(regime),
        'intact_samples_above_nk_step': sum(
            1 for s in regime if s['intact_above']),
        'max_intact_over_run': max(
            (s['intact_max'] for s in regime if s['intact_max'] is not None),
            default=None),
        'max_escaped_over_run': max(
            (s['escaped_max'] for s in regime if s['escaped_max'] is not None),
            default=None),
        'escaped_ever_above_nk_step': any(
            s['escaped_above'] for s in regime),
    }


def _scorer() -> str:
    """Which peptide binding predictor this interpreter has.

    The slab runs below are NOT scorer-independent -- MHCflurry and the
    position-weight-matrix fallback give different neoantigen
    repertoires, so a founder that lost an allele goes dark under one
    and not the other, and the run diverges from there. Both GitHub
    workflows and production run Python 3.11 with mhcflurry from
    requirements.txt; py3.13 cannot import it (mhcflurry 2.0.6 needs the
    stdlib `pipes` module) and falls back when
    COGNISOM_ALLOW_PWM_FALLBACK is set.
    """
    from cognisom.genomics.mhcflurry_binding import is_mhcflurry_available
    return 'mhcflurry' if is_mhcflurry_available() else 'pwm'


#: The exact end state of the shipped 24 h Baseline slab run at seed 1,
#: founder fraction 0.5, under EACH scorer. Measured 2026-10-28.
#:
#: This pin used to carry only the PWM row, which made the test RED
#: under the configuration CI and production use: the real predictor
#: gives (110, 63) where the fallback gives (115, 60), and three further
#: assertions moved with it. The properties the test is actually about
#: -- zero T-cell kills, zero de novo events, escaped cells never over
#: the NK step -- hold under both and are asserted outside this table.
SHIPPED_24H_PINS = {
    'pwm': {
        'survivors': (115, 60),
        'delta': -0.00326,
        'n_kills': 21,
        'by_lineage': {'macrophage': 11, 'NK_cell': 10},
        'observed': 13,
        'expected': 10.96,
        'intact_mean': 0.34797,
        'escaped_mean': 0.22904,
        'intact_above_at_end': 0,
        'intact_samples_above': 4,
        'max_intact_over_run': 0.40102,
        'cancer_divisions': 56,
        'divisions_total': 849,
    },
    'mhcflurry': {
        'survivors': (110, 63),
        'delta': 0.04773,
        'n_kills': 23,
        'by_lineage': {'macrophage': 11, 'NK_cell': 12},
        'observed': 12,
        'expected': 12.72,
        'intact_mean': 0.33901,
        'escaped_mean': 0.22162,
        'intact_above_at_end': 2,
        'intact_samples_above': 16,
        'max_intact_over_run': 0.42523,
        'cancer_divisions': 53,
        'divisions_total': 846,
    },
}


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_the_shipped_24h_run_is_not_enriched_for_escape():
    """The plan's bar, pinned at the null it actually measures.

    Baseline slab, 24 h, seed 1, half the 80 founders seeded with HLA
    LOH. Under the PWM fallback: 42 of 80 escaped founders -> 60 of 115
    survivors, so the escaped fraction goes 0.5250 to 0.5217 and moves
    DOWN by 0.0033. Under the real MHCflurry predictor the same seed
    gives 63 of 110 and +0.0477. Over 12 seeds (PWM) the mean move is
    +0.0043 (sd 0.0253, 95% CI -0.0118 to +0.0203) against +0.0062
    (sd 0.0093) with the immune module removed -- a difference of
    -0.0020 +- 0.0078, t = -0.26, and a single seed's delta is not
    evidence either way against that scatter. What this test pins is
    that the shipped infiltrate's delta stays inside the measured
    run-to-run scatter and that the MECHANISM behind it does not move.

    The kill composition in this one seed says why there is nothing to
    find: 21 kills (PWM) or 23 (MHCflurry), all of them NK and
    macrophage and ZERO T-cell. The arm that can see the escape trait
    did not kill anything, under either scorer.

    The MHC-I regime is pinned on the HOURLY SERIES and not on the final
    frame, because the final frame lies under the PWM scorer: intact
    cells cross the 0.4 missing-self step at 4 of 24 hourly samples
    (h 17.3-20.3, max 0.4010) and are back under it at hour 24. Under
    the real predictor they are over it at 16 of 24 samples and 2 are
    over it at the end. So the NK arm's counter-selection regime IS
    entered in a shipped preset. Escaped cells never cross the step
    under either scorer, which is the asymmetry that makes it
    counter-selection rather than noise.
    """
    scorer = _scorer()
    pin = SHIPPED_24H_PINS[scorer]
    out = _preset_run('immune_on', seed=1)

    # ── scorer-independent: founder seeding is a draw on cancer cells
    # and the binding predictor cannot touch it.
    assert (out['founders'], out['founders_escaped']) == (80, 42)
    assert out['founder_events'] == {
        'mhc1_allele_loss': 42, 'mhc1_total_loss': 0,
        'pdl1_heritable_gain': 0}

    # ── scorer-independent: the mechanism behind the null.
    by_lineage = Counter(k[0] for k in out['kills'])
    assert by_lineage['T_cell'] == 0, (
        'a T cell killed in the shipped slab run; the diluted-T-cell '
        'explanation of the population null needs re-measuring')
    # The delta sits inside the measured run-to-run scatter (sd 0.0253
    # over 12 seeds), which is all one seed can say.
    p0 = out['founders_escaped'] / out['founders']
    p1 = out['survivors_escaped'] / out['survivors']
    assert abs(p1 - p0) < 3 * 0.0253

    # ── scorer-independent: the regime is entered, and only by intact
    # cells. Asserted over the run, not at the end.
    assert out['n_regime_samples'] == 24
    assert out['intact_samples_above_nk_step'] > 0, (
        'intact cells no longer cross the 0.4 NK step in the shipped '
        'run -- if the step or the IFN-gamma gain was fixed, re-pin '
        'this and say so; it was a live defect when measured')
    assert out['max_intact_over_run'] >= DEFAULT_MISSING_SELF_THRESHOLD
    assert out['escaped_ever_above_nk_step'] is False
    assert out['escaped_above_nk_step'] == 0
    assert out['max_escaped_over_run'] < DEFAULT_MISSING_SELF_THRESHOLD

    # ── scorer-dependent: the exact end state.
    assert (out['survivors'], out['survivors_escaped']) == pin['survivors']
    assert p1 - p0 == pytest.approx(pin['delta'], abs=5e-5)
    assert len(out['kills']) == pin['n_kills']
    assert by_lineage == Counter(pin['by_lineage'])
    observed = sum(1 for k in out['kills'] if k[1])
    expected = sum(k[2] for k in out['kills'])
    assert observed == pin['observed']
    assert expected == pytest.approx(pin['expected'], abs=0.01)
    assert out['mhc1_intact_mean'] == pytest.approx(
        pin['intact_mean'], abs=5e-5)
    assert out['mhc1_escaped_mean'] == pytest.approx(
        pin['escaped_mean'], abs=5e-5)
    assert out['intact_above_nk_step'] == pin['intact_above_at_end']
    assert out['intact_samples_above_nk_step'] == pin[
        'intact_samples_above']
    assert out['max_intact_over_run'] == pytest.approx(
        pin['max_intact_over_run'], abs=5e-5)

    # ── and the division counts, because only a CANCER daughter draws.
    assert out['cancer_divisions'] == pin['cancer_divisions']
    assert out['divisions_total'] == pin['divisions_total']
    assert out['cancer_divisions'] < 0.1 * out['divisions_total'], (
        'the de novo expectation must be computed on cancer divisions')

    # And no de novo event, as 1.55e-6 per division over ~850 divisions
    # requires. Every number above is about SEEDED variation, and the
    # two counters have to stay apart for that sentence to be checkable.
    assert sum(out['de_novo'].values()) == 0
    assert out['founder_events'] == {
        'mhc1_allele_loss': 42, 'mhc1_total_loss': 0,
        'pdl1_heritable_gain': 0}


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_drift_alone_moves_the_composition_as_much_as_immune_pressure():
    """The control the claim needs: the same run with the immune module
    removed, so nothing can select and only sampling and division
    remain. Seed 1 goes 42 of 80 -> 78 of 145, a move of +0.0129 --
    LARGER than the +0.0043 mean the shipped infiltrate produces and of
    the opposite sign to that seed's own -0.0033. Over 12 seeds drift
    alone is +0.0062 (sd 0.0093), and that sd is the noise floor every
    number in this file is measured against.

    With no immune cells there is also no IFN-gamma, so surface MHC-I
    stays exactly where it was seeded -- 0.300 intact, 0.200 after one
    allele -- which is worth asserting because it shows the control
    arm is the clean drift arm it claims to be and not a second regime.
    """
    out = _preset_run('immune_off', seed=1)
    assert (out['founders'], out['founders_escaped']) == (80, 42)
    assert (out['survivors'], out['survivors_escaped']) == (145, 78)
    assert out['kills'] == []
    p0 = out['founders_escaped'] / out['founders']
    p1 = out['survivors_escaped'] / out['survivors']
    assert p1 - p0 == pytest.approx(0.01293, abs=5e-5)
    assert out['mhc1_intact_mean'] == pytest.approx(MHC1_SEEDED, abs=1e-9)
    assert out['mhc1_escaped_mean'] == pytest.approx(
        MHC1_AFTER_ONE_ALLELE_LOST, abs=1e-9)
    assert sum(out['de_novo'].values()) == 0


# ── part 4: what WOULD show selection, as arithmetic ─────────────────────

def test_the_de_novo_route_cannot_supply_an_event_in_a_preset_run():
    """So every selection measurement here had to seed the variation.

    The denominator is CANCER divisions, not a run's division count:
    only a cancer daughter draws (_maybe_escape returns None for every
    other cell type), and the 24 h Baseline slab run makes 53-56 cancer
    divisions out of 846-849. 6.5e5 against 56 is a factor of ~11,500,
    not the ~770 this test asserted while using the total -- and on the
    RING presets the factor is infinite, because a 24 h ring Baseline
    run makes 35 divisions and none of them is a cancer cell's.
    """
    cancer_divisions_24h = 56
    total_divisions_24h = 849
    need = ev.divisions_per_expected_event()
    assert need == pytest.approx(6.45e5, rel=0.02)
    assert ev.expected_events(cancer_divisions_24h) < 1e-4
    assert need / cancer_divisions_24h > 10000
    # The factor the earlier version of this test reported, so the
    # correction is on the record: using totals understates the gap 15x.
    assert need / total_divisions_24h == pytest.approx(760, rel=0.05)


def test_the_one_de_novo_event_observed_is_the_seed_and_not_the_rate():
    """2,707 cancer divisions over the 12 Aggressive Tumor runs whose de
    novo counter was readable, 0.0042 expected events, one observed --
    a 1-in-240 outcome, and exactly the observation that invites someone
    to "correct" a sourced rate downward.

    It is the seed. ``default_rng([5, 0])`` carries a draw of 1.360e-6
    within its first 260 values, under mu = 1.5498e-6, and the escape
    draws come from that stream. The seeding itself is uniform, which is
    what this pins: over 2,000 streams only 2 carry a sub-mu draw in 260
    against 0.81 expected (Poisson p = 0.19), and the mean and median of
    the per-stream minimum match the Beta(1, N) theory to three
    significant figures. If a future change to how these generators are
    seeded biases the early stream, the realised escape rate would drift
    away from the literature one silently, and this is where it shows.
    """
    mu = ev.MHC1_ALLELE_LOSS_PER_DIVISION.value
    n_draws = 260
    minima = []
    for s in range(2000):
        minima.append(float(np.random.default_rng([s, 0]).random(n_draws).min()))
    # Seed 5 is the one that fired in the measured run.
    assert minima[5] < mu
    assert minima[5] == pytest.approx(1.360e-6, rel=1e-3)
    hits = sum(1 for m in minima if m < mu)
    assert hits == 2
    expected_hits = 2000 * (1.0 - (1.0 - mu) ** n_draws)
    assert expected_hits == pytest.approx(0.806, abs=5e-3)
    # The minimum of N uniforms is Beta(1, N): mean 1/(N+1), median
    # 1 - 0.5**(1/N). Three significant figures each.
    assert float(np.mean(minima)) == pytest.approx(
        1.0 / (n_draws + 1), rel=0.02)
    assert float(np.median(minima)) == pytest.approx(
        1.0 - 0.5 ** (1.0 / n_draws), rel=0.02)
    # And the arithmetic that makes one event a 1-in-240 outcome.
    cancer_divisions = 2707
    assert ev.expected_events(cancer_divisions) == pytest.approx(
        0.0042, abs=5e-5)
    assert 1.0 / (1.0 - math.exp(-ev.expected_events(cancer_divisions))) \
        == pytest.approx(239.0, abs=2.0)


def test_the_recorded_kill_mix_arithmetic_is_internally_consistent():
    """ARITHMETIC ON A RECORDED MEASUREMENT, and nothing more.

    Named honestly after what it does: the table below is typed in from
    a 12-seed measurement the suite does NOT re-derive (96 runs, ~3
    hours), so this function touches no product code and cannot fail
    from an engine change. It was previously called
    test_the_limit_on_selection_is_the_kill_mix_not_the_escape_rate,
    which promised a sensitivity result it does not establish. The
    engine-measured kill composition that CAN fail lives in
    test_the_shipped_24h_run_is_not_enriched_for_escape (seed 1) and in
    test_de_novo_escape_arises_and_is_then_selected.

    What the arithmetic says. Over 12 Baseline runs the T-cell arm was
    23 of 234 kills, 9.8%. Its own selection ratio was O/E = 0.084 (one
    escaped cell killed where 12 were due, z = -4.64); the innate arm's
    was 1.07 and 1.04. Mixing those at the shipped weights gives an
    overall 0.96, which at 234 kills is z = -0.64 -- indistinguishable
    from neutral. If T cells did half the killing at the same
    per-lineage ratios the aggregate would be under 0.6.

    And the one preset that DOES have that kill mix is not the
    confirmation it looks like: the Immunotherapy slab measures T cells
    at 52% of kills (100 of 191, seeds 1-3) and escaped cells are still
    not spared, O/E 1.10, because that preset runs checkpoint_block 1.0
    and the ICB-rescue term rescues a partially escaped cell harder than
    an intact one. See escape_variation's third consequence.
    """
    measured = {  # lineage: (kills, O, E)
        'T_cell': (23, 1, 11.96),
        'NK_cell': (124, 67, 62.62),
        'macrophage': (87, 45, 43.24)}
    n = sum(v[0] for v in measured.values())
    O = sum(v[1] for v in measured.values())
    E = sum(v[2] for v in measured.values())
    assert n == 234
    assert O / E == pytest.approx(0.959, abs=5e-4)
    assert measured['T_cell'][0] / n == pytest.approx(0.098, abs=5e-4)
    # The T-cell arm on its own is unambiguous; the aggregate is not.
    t_kills, t_O, t_E = measured['T_cell']
    assert t_O / t_E < 0.1
    assert (t_O - t_E) / math.sqrt(t_E * (1 - t_E / t_kills)) < -4.0
    # The same per-lineage ratios, if T cells did half the killing.
    half = 0.5 * t_O / t_E + 0.5 * (
        (measured['NK_cell'][1] + measured['macrophage'][1])
        / (measured['NK_cell'][2] + measured['macrophage'][2]))
    assert half < 0.6
    assert SLAB_INFILTRATE[2] / sum(SLAB_INFILTRATE) > 0.5


# ── part 5: the two halves of the item, joined ───────────────────────────
#
# Everything above imposes the escape phenotype -- by hand in the arenas
# (_dress calls cellular._lose_one_allele), by founder seeding in the
# preset runs. Measured: monkeypatching CellularModule._maybe_escape to
# a no-op failed 11 tests in test_escape_can_be_inherited.py and ZERO in
# this file. So without the test below, "selection observed rather than
# written in" would be observed entirely on written-in variation, and
# the heritable route and the selection measurement would never meet.

def _growing_tumour(n_t: int, seed: int, mu: float,
                    hours: float = 24.0, n_cancer: int = 30,
                    costimulation: float = 1.0) -> Dict[str, object]:
    """A dividing tumour with NO escaped founders and mu scaled high.

    Division is ON (unlike ``arena``, which switches it off so every
    death is a kill) because the de novo route only fires at a division.
    Spontaneous apoptosis is off, so every death is still a kill.
    """
    np.random.seed(seed)
    engine = SimulationEngine(SimulationConfig(
        dt=0.05, duration=hours, use_gpu=False))
    engine.register_module('molecular', MolecularModule, {
        'transcription_rate': 0.0, 'exosome_release_rate': 0.0,
        'mutation_rate': 0.0})
    engine.register_module('cellular', CellularModule, {
        'n_normal_cells': 0, 'n_cancer_cells': n_cancer,
        'apoptosis_rate': 0.0,
        # DELIBERATELY SCALED, and calibration_report says so: 0.5 per
        # division is 3.2e5 times the sourced 1.55e-6. At the literature
        # rate 30 cells over 24 h supply ~200 cancer divisions and
        # 3e-4 expected events, so there would be nothing to select.
        'escape_mhc1_loss_per_division': mu,
        # One route only, so the measured selection is on the partial
        # phenotype whose rate is sourced.
        'escape_mhc1_second_hit_per_division': 0.0,
        'escape_seed': seed})
    engine.register_module('immune', ImmuneModule, {
        'n_t_cells': n_t, 'n_nk_cells': 0, 'n_macrophages': 0,
        'tcr_recognition_threshold': 0.0, 'tcr_repertoire_size': 200,
        'tcr_seed': 0, 'costimulation': costimulation,
        'checkpoint_block': 0.0,
        'macrophage_polarization_enabled': False})
    with contextlib.redirect_stdout(io.StringIO()):
        engine.initialize()
    cells = engine.modules['cellular']
    immune = engine.modules['immune']
    immune.set_cellular_module(cells)
    for cell in cells.cells.values():
        cell.mhc1_displayed_peptides = _presentations(cells.hla_alleles)
    rng = np.random.default_rng(seed)
    centre = np.mean([c.position for c in cells.cells.values()], axis=0)
    for icell in immune.immune_cells.values():
        icell.position = centre + rng.uniform(-20, 20, 3)

    escaped_of = {cid: bool(c.escape_traits) for cid, c in cells.cells.items()}
    kills: List[Tuple[bool, float]] = []

    def on_divided(data):
        daughter = cells.cells.get(data.get('daughter_id'))
        if daughter is not None:
            escaped_of[daughter.cell_id] = bool(daughter.escape_traits)

    def on_killed(data):
        living = [c for c in cells.cells.values()
                  if c.alive and c.cell_type == 'cancer']
        n = len(living) + 1
        esc = sum(1 for c in living if c.escape_traits)
        esc += 1 if escaped_of.get(data['cell_id']) else 0
        kills.append((bool(escaped_of.get(data['cell_id'])), esc / n))

    engine.event_bus.subscribe(EventTypes.CELL_DIVIDED, on_divided)
    engine.event_bus.subscribe(EventTypes.CANCER_KILLED, on_killed)
    with contextlib.redirect_stdout(io.StringIO()):
        engine.run()

    dist = cells.escape_distribution()
    return {
        'n_cancer': dist['n_cancer'],
        'n_escaped': dist['n_escaped'],
        'founder_cells': dist['founder_cells'],
        'de_novo': sum(dist['events'].values()),
        'divisions': cells.total_divisions,
        'kills': len(kills),
        'O': sum(1 for k in kills if k[0]),
        'E': sum(k[1] for k in kills),
    }


@pytest.mark.timeout(300)   # 12 runs, measured at 2.1 s under load
def test_de_novo_escape_arises_and_is_then_selected():
    """The only configuration in which this engine shows escape ARISING
    and then being SELECTED, which is the item's claim with both halves
    joined.

    Zero escaped founders, escape_mhc1_loss_per_division scaled to 0.5
    and labelled as scaled, 30 cancer cells dividing for 24 h, 6 seeds
    per arm, against the same tumour with no immune cells:

      arm          cancer  escaped  escaped share  de novo  kills  O/E
      no immune       313      140      0.4473        155      0     --
      12 T cells      139       76      0.5468         85    114   0.32

    Three statements, each needing the heritable route to exist:

    * the phenotype was NOT imposed. founder_cells is 0 in every run and
      every one of the 85 and 155 events came from a division.
    * the escaped share RISES above the drift arm, +0.0995, which is 4x
      the whole measured 12-seed spread of the shipped slab arm (sd
      0.0253) and the enrichment the item asks for.
    * the kills select against escape on a TIME-MATCHED null: 7 escaped
      cells killed where 21.8 were due, O/E 0.32, z = -3.5 over 114
      kills. The population shift and the contact mechanism agree in
      this arm, which they cannot be shown to do in the shipped one.

    THE PRICE is on the record: mu is 3.2e5 times the sourced rate. At
    1.55e-6 these 199 cancer divisions expect 3.1e-4 events, so this
    arm demonstrates that the MACHINERY works and says nothing about
    whether a real tumour at this scale would escape -- which is why the
    founder route exists. calibration_report's scaled_from_calibration
    flag is asserted below so a reader of the run cannot miss it.
    """
    mu = 0.5
    control = [_growing_tumour(0, s, mu) for s in range(1, 7)]
    treated = [_growing_tumour(12, s, mu) for s in range(1, 7)]

    # The phenotype was not imposed: no founder carries it.
    assert all(r['founder_cells'] == 0 for r in control + treated)
    # ...and it arose at divisions, in both arms.
    assert sum(r['de_novo'] for r in control) == 155
    assert sum(r['de_novo'] for r in treated) == 85

    def share(rows):
        c = sum(r['n_cancer'] for r in rows)
        return sum(r['n_escaped'] for r in rows) / c, c

    drift, n_drift = share(control)
    selected, n_selected = share(treated)
    assert (n_drift, n_selected) == (313, 139)
    assert drift == pytest.approx(0.4473, abs=5e-4)
    assert selected == pytest.approx(0.5468, abs=5e-4)
    # The enrichment, against the shipped arm's whole 12-seed sd.
    assert selected - drift == pytest.approx(0.0995, abs=1e-3)
    assert selected - drift > 3 * 0.0253

    # And the contact mechanism, on the time-matched null.
    n_kills = sum(r['kills'] for r in treated)
    observed = sum(r['O'] for r in treated)
    expected = sum(r['E'] for r in treated)
    assert sum(r['kills'] for r in control) == 0
    assert (n_kills, observed) == (114, 7)
    assert expected == pytest.approx(21.8215, abs=1e-3)
    assert observed / expected == pytest.approx(0.3208, abs=1e-3)
    z = (observed - expected) / math.sqrt(expected * (1 - expected / n_kills))
    assert z < -3.0

    # The scaling is declared, not silent.
    entry = ev.calibration_report(mhc1_loss=mu)[
        'mhc1_allele_loss_per_division']
    assert entry['scaled_from_calibration'] is True
    assert entry['out_of_range'] is True
    assert mu / ev.MHC1_ALLELE_LOSS_PER_DIVISION.value > 3e5
