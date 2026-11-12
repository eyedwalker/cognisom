#!/usr/bin/env python3
"""
Cellular Module
===============

Handles cell-level dynamics: metabolism, division, death, migration.

Features:
- Cell cycle (G1, S, G2, M phases)
- Metabolism (O2, glucose, ATP)
- Cell division
- Cell death (apoptosis, necrosis)
- Cell migration
"""

import math
import sys
sys.path.insert(0, '..')

import numpy as np
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from cognisom.core.module_base import SimulationModule
from cognisom.core.event_bus import EventTypes
from cognisom.engine.py.immune import escape_variation as escape
from cognisom.engine.py.immune.mhc_loading import MHCLoader, MHCPresentation
from cognisom.engine.py.molecular.peptidome import (
    Peptide,
    generate_neoantigen_peptides,
)


# Default patient HLA-I alleles for the simulation. Same three-allele
# panel used by the neoantigen vaccine pipeline (HLA-A/B/C). Override
# via module config 'hla_alleles'.
DEFAULT_HLA_ALLELES = ("HLA-A*02:01", "HLA-A*24:02", "HLA-B*07:02")

logger = logging.getLogger(__name__)



# Oncogenes known to drive intrinsic PD-L1 expression (Type III TME
# in Teng et al. 2015). The per-cell baseline is how much PD-L1
# expression each oncogene contributes when the cell carries an
# oncogenic flag for that gene. Multiple matching oncogenes stack
# additively, clamped at 1.0.
# Sources: Spranger 2015 (PI3K/AKT); Casey 2016 (MYC); Akbay 2013
# (EGFR mutant NSCLC drives PD-L1 via ERK/MAPK).
_INTRINSIC_PDL1_INDUCERS = {
    "PIK3CA": 0.35,
    "MYC": 0.30,
    "EGFR": 0.40,
}


def _intrinsic_pdl1_for_oncogenes(oncogenes) -> float:
    """Sum the intrinsic PD-L1 contributions of any matching oncogenes,
    clamped to [0, 1]. ``oncogenes`` is an iterable of gene names."""
    if not oncogenes:
        return 0.0
    total = sum(_INTRINSIC_PDL1_INDUCERS.get(g, 0.0) for g in oncogenes)
    return float(max(0.0, min(1.0, total)))


#: Oxygen below which a cell is ANOXIC and starts to die, on the engine's
#: normalised scale (0.21 == arterial 100 mmHg; VascularModule). Low oxygen
#: is not lethal: murine fibroblasts and human tumour cells grew normally at
#: 0.5% O2, and only severe hypoxia below 0.01% O2 induced apoptosis
#: (Papandreou 2005, PMID 15833847). 0.01% of a 713 mmHg dry-gas pressure is
#: 0.071 mmHg. This used to be 0.02 (9.5 mmHg, ~1.3% O2) and killed a normal
#: cell on the step it crossed it -- a level ~2.6x ABOVE the one the paper
#: found harmless -- which is why the Hypoxia preset lost 83% of its normal
#: tissue in 12 h.
ANOXIC_O2_LEVEL = 0.21 * (0.0001 * 713.0) / 100.0
#: Glucose below which a cell is starving, mM. Unchanged from the module's
#: original threshold; unsourced.
STARVATION_GLUCOSE_MM = 0.5
#: Death rate of a cell, normal or cancer, while anoxic or starving, per
#: hour. ESTIMATE, range 1/72-1/6: Papandreou 2005 reports apoptosis under
#: anoxia without a survival time in its abstract, and no measured survival
#: curve was found to source it from. A mean survival of a day is the
#: centre. Normal and cancer cells share it because the paper found the
#: same threshold for both. This replaces instant death for a normal cell
#: and an illustrative 1.0/h for a cancer cell.
ANOXIC_DEATH_PER_HOUR = 1.0 / 24.0


#: Lactate production, mM per hour on the cell's own volume: the rates
#: _update_metabolism has always used for the intracellular pool, now
#: named so the spatial field can source from the same numbers. 3:1
#: cancer:normal is the module's own ratio. Both ESTIMATE -- see
#: SpatialModule's lactate budget for what the field actually needs.
LACTATE_PRODUCTION_CANCER_PER_H = 0.3
LACTATE_PRODUCTION_NORMAL_PER_H = 0.1

#: Tissue DHT in androgen-stimulated benign prostate: 13.7 pmol/g, i.e.
#: ~13.7 nM at tissue density 1 g/mL (Titus 2005, PMID 16000557; LC-MS/MS,
#: 18 specimens, blood flow preserved to the moment of freezing).
TISSUE_DHT_REFERENCE_NM = 13.7
#: The fraction of tissue DHT that does NOT come from circulating
#: testosterone. Medical castration in healthy men cut serum T by 94% and
#: prostatic DHT by only 80% (Page 2006, PMID 16882745, n=13); in
#: patients, tissue DHT stayed at ~25% of baseline after six months of
#: castration + flutamide while serum DHT fell to 7.5% (Nishiyama 2004,
#: PMID 15534082, n=30 paired). Adrenal precursors converted inside the
#: gland are the source both papers name. 0.20 is Page's figure, the
#: cleaner design; Nishiyama's 0.25 is the check. A vessel-fed field
#: alone predicts 0.06 and is therefore wrong by itself -- measured.
DHT_INTRACRINE_FRACTION = 0.20
#: AR signalling at or above this fraction of eugonadal counts as intact:
#: no division slowing, no withdrawal hazard, and no random draw. The
#: eugonadal complex sits a few tenths of a percent under its bound in a
#: running population (daughters inherit a parent that was still
#: settling), and without a dead band that drift would stretch every
#: division time by 0.5% and roll a die per dependent cell per step in a
#: run where nothing was withdrawn.
AR_INTACT_ABOVE = 0.95

#: Division periods on the MEASURED clock, hours: thymidine labelling
#: index with a ~10 h S phase (Meyer 1982, PMID 6185700; the derivation is
#: beside division_time_normal in CellularModule.__init__). Normal
#: prostate epithelium ~347 days, adenocarcinoma ~46 days.
#:
#: The module's defaults (90 h and 12 h) are the same 7.5:1 ratio on a
#: clock compressed ~93x so that a one-day run shows divisions. That is a
#: choice about what a short run should show, and it puts growth on a
#: clock 93x faster than killing and drug binding, which run in real
#: time. A run that is meant to stand for days or weeks of a patient
#: should use these.
MEASURED_DIVISION_HOURS = {'normal': 8333.0, 'cancer': 1111.0}

#: IFN-gamma raises surface MHC-I on tumour cells by a mean 50% (five
#: glioma lines, Naganuma 1998, PMID 9821108; same direction in prostate,
#: Martini 2010, PMID 20304037, magnitude not given). The GAIN is from a
#: non-prostate line and the SATURATION level has no source: ESTIMATES.
MHC1_IFN_GAIN = 0.5
#: Field level at which the gain is full. The field's units are arbitrary
#: -- the secretion rate per effector is 1.0/h and nothing external fixes
#: it -- so this number is what gives them meaning, and it is an internal
#: CALIBRATION rather than a measurement: it is the level that three
#: activated effectors at contact range produce in the shipped block
#: (measured against the solver: one effector at 10 um gives 1.4e-4).
#: Three within 30 um is exactly the saturation point of the neighbour-
#: counting proxy this field replaces (DEFAULT_CYTOKINE_SATURATION_COUNT
#: in macrophage_polarization), so the two agree where the proxy is
#: defined and part company the way the physics says: with distance, and
#: with the density of the cells consuming the signal.
IFN_GAMMA_PROXY_SATURATION = 4.5e-4
#: The MHC-I response uses that calibration; so does macrophage
#: polarisation, through ImmuneModule._local_ifn_gamma. One number, so
#: the field cannot mean one thing to one consumer and another to the
#: next. (Kept under its old name too: it was published in a commit
#: message and a memory note as MHC1_IFN_SATURATION.)
MHC1_IFN_SATURATION = IFN_GAMMA_PROXY_SATURATION
#: Relaxation time of surface MHC-I toward its IFN-set level, hours. The
#: cited measurements are at 24-48 h; class I turns over in hours. The
#: value is an ESTIMATE within that bracket.
MHC1_RESPONSE_H = 6.0

#: Relaxation time of tissue DHT toward its supply-set level, hours. Not
#: measured: tissue levels are reported weeks after castration, which
#: bounds it above and no closer. ESTIMATE; it moves transients only.
DHT_RELAXATION_H = 2.0


@dataclass
class CellState:
    """State of a single cell"""
    cell_id: int
    position: np.ndarray
    cell_type: str  # 'normal', 'cancer', 'immune'
    phase: str  # 'G1', 'S', 'G2', 'M'
    age: float = 0.0
    alive: bool = True

    # Metabolism
    oxygen: float = 0.21
    glucose: float = 5.0
    atp: float = 1000.0
    lactate: float = 0.0
    # Androgen. testosterone_nM is the FREE interstitial testosterone at
    # the cell, written by SpatialModule from its field on every solve;
    # None means no field is attached and tissue DHT holds its reference.
    # dht_nM is tissue dihydrotestosterone, set by that supply plus the
    # intracrine term (see DHT_INTRACRINE_FRACTION).
    testosterone_nM: Optional[float] = None
    # Interstitial lactate at the cell, mM, from the spatial field (None =
    # no field). `lactate` above is the cell's own pool.
    lactate_mM: Optional[float] = None
    # Interstitial IFN-gamma at the cell, in [0, 1], from the spatial
    # field (None = no field). Raises MHC-I toward mhc1_ceiling.
    ifn_gamma_level: Optional[float] = None
    # Local level of every DECLARED signal (modules/signalling.py), by
    # name, written by the spatial module after each solve. Empty unless a
    # run declares one; the three attributes above are the hand-wired
    # signals that predate that path.
    signals: Dict[str, float] = field(default_factory=dict)
    dht_nM: float = TISSUE_DHT_REFERENCE_NM
    # Whether this cancer cell dies when androgen signalling is withdrawn.
    # Westin 1995 (PMID 7778676; 18 patients, biopsies the day before and
    # 7 days after castration): proliferation fell in 15 tumours, but the
    # apoptotic index rose in only 6 -- the WHO grade 1-2, p53/Bcl-2/
    # c-myc-negative ones. A third, then; here a deterministic third of
    # the cancer cells (cell_id % 3 == 0), inherited clonally, so no random
    # draw moves the pinned runs. The split is per tumour in the paper
    # and per cell here: ESTIMATE.
    ar_dependent: bool = False

    # Sublethal cytotoxic damage from T-cell perforin hits, in hits, as it
    # stood at cytotoxic_damage_at_h (the immune module's clock). It decays
    # between hits; see engine/py/immune/kill_calibration.py.
    cytotoxic_damage: float = 0.0
    cytotoxic_damage_at_h: float = 0.0

    # Cancer properties
    mhc1_expression: float = 1.0
    # Where mhc1_expression sits with no IFN-gamma around. Captured the
    # first time the IFN-gamma response runs, so that a cell seeded or
    # transformed at a downregulated level returns to THAT level when the
    # signal goes, not to 1.0.
    mhc1_baseline: Optional[float] = None
    mutations: List[str] = None

    # PD-L1 surface expression in [0, 1]. Two induction modes:
    #   * adaptive (Type I TME, Teng et al. 2015): bumped by
    #     IFN-gamma signalling from activated T cells in the
    #     vicinity (the immune module increments this on
    #     IMMUNE_ACTIVATED).
    #   * intrinsic (Type III TME): bumped at transformation time
    #     for cancer cells carrying oncogenes known to constitutively
    #     induce PD-L1 (PIK3CA, MYC, EGFR per the cancer-immunology
    #     literature -- Spranger / Pardoll).
    # See engine/py/immune/tme_classifier.py for how this gets read.
    pdl1_expression: float = 0.0

    # Local desmoplastic stroma / fibrotic ECM density in [0, 1]
    # (Upgrade 6). Cancer cells deposit ECM proportional to their
    # age + cell-type baseline; T-cell motility + detection radius
    # are attenuated by the ECM density sampled at their position
    # (see engine/py/spatial/ecm_barrier.py). High-ECM tumors
    # exclude TILs even when neoantigens are displayed -- the
    # "cold but mutated" PDAC phenotype.
    local_ecm_density: float = 0.0

    # Closed-loop neoantigen presentation (Upgrade 2). Populated when
    # MUTATION_OCCURRED fires for this cell; each entry carries the
    # peptide + HLA allele + binding affinity for downstream TCR-pMHC
    # matching. Empty for cells without displayed neoantigens.
    mhc1_displayed_peptides: List[MHCPresentation] = field(default_factory=list)

    # Heritable immune-escape state. Empty on every cell until an escape
    # event happens at division, so a run with escape off -- or one in
    # which no event occurred -- carries exactly the state it always did.
    #
    # escape_traits names the ROUTE(s) this lineage carries, from
    # engine/py/immune/escape_variation.ESCAPE_TRAITS. The two MHC-I
    # routes are not interchangeable: 'mhc1_allele_loss' leaves the
    # cell recognised at a lower signal (checkpoint blockade still
    # works), 'mhc1_total_loss' puts it below ImmuneModule._recognize's
    # 0.05 floor, where it is invisible to T cells and blockade cannot
    # reach it.
    #
    # escape_born_at_h is when the lineage FIRST escaped, on the
    # cellular module's clock, so selection can be measured against
    # trait age later. A second hit does not move it: the quantity of
    # interest is how long the escaped lineage has been under pressure.
    #
    # escape_lost_hla names the panel allele(s) lost to LOH, which is
    # what makes parallel evolution checkable (McGranahan 2017 saw the
    # same allele lost on distinct branches in four tumours). It stays
    # empty for a total-loss event, because B2M loss does not delete an
    # HLA gene.
    escape_traits: tuple = ()
    escape_born_at_h: Optional[float] = None
    escape_lost_hla: tuple = ()

    def __post_init__(self):
        if self.mutations is None:
            self.mutations = []


class CellularModule(SimulationModule):
    """
    Cellular simulation module
    
    Manages:
    - Cell population
    - Cell cycle
    - Metabolism
    - Division and death
    - Cell state
    
    Events Emitted:
    - CELL_DIVIDED: When cell divides
    - CELL_DIED: When cell dies
    - CELL_TRANSFORMED: When normal → cancer
    - CELL_MIGRATED: When cell moves
    
    Events Subscribed:
    - EXOSOME_UPTAKEN: Process molecular cargo
    - CANCER_KILLED: Remove killed cells
    - HYPOXIA_DETECTED: Respond to low O2
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)

        # Cell population
        self.cells: Dict[int, CellState] = {}
        self.next_cell_id = 0

        # Parameters
        # Division periods, hours. These are a COMPRESSED timescale whose
        # RATIO is the measured one -- read on before changing either.
        #
        # Meyer JS, Sufrin G, Martin SA, J Urol 1982;128:1353-6 (PMID
        # 6185700) gives thymidine labelling indices for prostate:
        #
        #     normal peripheral zone    0.12%   (central zone 0.11%)
        #     nodular hyperplasia       0.31%
        #     adenocarcinoma            0.90%   (range 0.14-3.90)
        #
        # With an in-vivo S-phase duration of ~10 h -- the median across
        # normal tissues and tumours in halogenated-pyrimidine studies --
        # turnover is Ts/LI: ~8,333 h (~347 d) for normal epithelium and
        # ~1,111 h (~46 d) for adenocarcinoma. A Ki-67 cross-check agrees:
        # benign Ki-67 ~1.3% is the GROWTH FRACTION, not the turnover, and
        # 1.3% of cells cycling at ~108 h each gives the same ~347 d.
        #
        # Those real values are not what is set here. At 8,333 h a 24 h run
        # shows no division at all, and every 24 h pin in the suite would
        # have to be re-taken against a multi-week default duration. So the
        # scale stays compressed and the RATIO is corrected, which is the
        # part that drives behaviour: measured normal:cancer is 7.5:1 and
        # this used to run 2:1 (24/12), i.e. normal tissue proliferated
        # 3.75x too fast RELATIVE to cancer. That ratio error is what
        # produced "normal tissue grows 37% in 12 h" while a tumour was
        # being killed beside it.
        #
        # So: cancer unchanged, normal = 12.0 * 7.5. Changing the cancer
        # period means changing this one by the same factor, or the ratio
        # silently breaks again. gpu/cell_ops.detect_division_candidates
        # and core/tissue_config carry their own copies -- move all three.
        self.division_time_normal = self.config.get('division_time_normal', 90.0)  # hours
        self.division_time_cancer = self.config.get('division_time_cancer', 12.0)  # hours
        # Androgen withdrawal, read from the AR module when linked. The
        # proliferation gate applies to every cancer cell (15 of 18 tumours
        # in Westin 1995 slowed by day 7); the apoptosis hazard applies to
        # the ar_dependent subset. Both magnitudes are ESTIMATES: the
        # paper's full text is a scanned PDF and its abstract gives
        # directions and counts, not rates. Normal epithelium is left
        # alone -- Page 2006 (PMID 16882745) found benign proliferation and
        # apoptosis unchanged 28 days into castration.
        self.ar_dependent_fraction = float(self.config.get('ar_dependent_fraction', 1.0 / 3.0))
        self.androgen_withdrawal_death_per_hour = float(
            self.config.get('androgen_withdrawal_death_per_hour', 0.01))   # ESTIMATE, /h at full withdrawal
        self.anoxic_death_per_hour = float(
            self.config.get('anoxic_death_per_hour', ANOXIC_DEATH_PER_HOUR))
        # Where the initial cells start in their cycle: 'asynchronous' (the
        # default) staggers them over the steady-state age distribution;
        # 'synchronous' starts every cell at age 0, as before -- for tests
        # whose subject is the division clock itself. See
        # _stagger_initial_ages.
        self.initial_cell_cycle = self.config.get('initial_cell_cycle', 'asynchronous')
        if self.initial_cell_cycle not in ('asynchronous', 'synchronous'):
            raise ValueError(
                "initial_cell_cycle must be 'asynchronous' or 'synchronous', "
                f"got {self.initial_cell_cycle!r}")
        self.cell_cycle_seed = int(self.config.get('cell_cycle_seed', 0))

        # ── Heritable immune escape ───────────────────────────────────
        #
        # ON BY DEFAULT, and the defence of that choice is a
        # measurement, not a preference. Without heritable variation
        # immune pressure has nothing to select on, so escape could only
        # ever be imposed by hand; a capability that is off by default
        # in this repo is a capability nobody passes in. What makes ON
        # safe is that the sourced rate is 1.55e-6 per division
        # (escape_variation), so one expected event needs 6.5e5 CANCER
        # divisions -- and every draw comes from a generator of its own,
        # so the global stream that every pinned run depends on is
        # untouched whether an event fires or not.
        #
        # MEASURED, 2026-10-28, escape on vs off, same seeds. The
        # denominator is CANCER divisions, because _maybe_escape returns
        # None for every other cell type -- the earlier version of this
        # table quoted totals and so overstated the expectation by 14x
        # on the slab, and by everything on the ring:
        #
        #   run (24 h)                total div  CANCER div  de novo
        #   ring Baseline  seed 12345        35           0      0
        #   ring Hypoxia   seed 12345        26           0      0
        #   slab Baseline  seed 1 (PWM)     849          56      0
        #   slab Baseline  seed 1 (flurry)  846          53      0
        #   slab Aggressive seeds 1-3  ~930-980     226-238      0
        #
        # All eight pairs are identical on every number a pinned test
        # asserts -- cell counts, divisions, deaths by cause, mean
        # oxygen, mean MHC-I -- and the 24 h ring Baseline (115 normal
        # alive, 35 divisions, 10 cancer deaths) and Hypoxia (44 normal
        # hypoxia deaths, 65 deaths, 32 hypoxic cells) pins both hold.
        # 56 cancer divisions expect 8.7e-5 events, so zero is the
        # correct behaviour at this scale rather than a broken mechanism:
        # see escape_variation.expected_events. NOTE that the ring
        # presets have ZERO cancer divisions in 24 h, so an on/off
        # identity measured on the ring never reaches the draw at all;
        # the slab Baseline run is the one that crosses it.
        # For escape to be PRESENT in a short run, seed it
        # (escape_founder_hla_loh_fraction) instead of raising mu.
        self.escape_enabled = bool(self.config.get('escape_enabled', True))
        #: Its own seed, so a new draw cannot shift any other module's.
        #: Three independent sub-streams are derived from it -- one per
        #: route plus the founder seeder -- so enabling PD-L1 escape does
        #: not move the MHC-I draws, which an interleaved single stream
        #: would.
        #:
        #: It defaults to 0 INDEPENDENTLY of the run seed, which matters
        #: to anyone sweeping seeds: np.random.seed(s) does not reach
        #: these streams, so 12 replicates that differ only in the global
        #: seed draw the SAME founder-escape pattern and are correlated
        #: on the one axis a founder-seeded run is measuring. It is
        #: harmless for the de novo route at 1.55e-6 and load-bearing for
        #: founder seeding. Every measurement in realism item 1 passes
        #: escape_seed=seed explicitly, which is what made the 12-seed
        #: sd meaningful; a caller sweeping seeds must do the same. Left
        #: as an explicit knob rather than derived from the global stream
        #: because drawing it at initialize() would advance that stream
        #: and break the on/off byte-identity this default rests on.
        self.escape_seed = int(self.config.get('escape_seed', 0))
        self._escape_rng_mhc1 = np.random.default_rng([self.escape_seed, 0])
        self._escape_rng_pdl1 = np.random.default_rng([self.escape_seed, 1])
        self._escape_rng_founder = np.random.default_rng([self.escape_seed, 2])
        self.escape_mhc1_loss_per_division = float(self.config.get(
            'escape_mhc1_loss_per_division',
            escape.MHC1_ALLELE_LOSS_PER_DIVISION.value))
        self.escape_mhc1_second_hit_per_division = float(self.config.get(
            'escape_mhc1_second_hit_per_division',
            escape.MHC1_SECOND_HIT_PER_DIVISION.value))
        self.escape_pdl1_gain_per_division = float(self.config.get(
            'escape_pdl1_gain_per_division',
            escape.PDL1_HERITABLE_GAIN_PER_DIVISION.value))
        self.escape_pdl1_gain_magnitude = float(self.config.get(
            'escape_pdl1_gain_magnitude',
            escape.PDL1_HERITABLE_GAIN_MAGNITUDE.value))
        #: Fraction of FOUNDING cancer cells seeded already escaped. 0.0
        #: by default: McGranahan's 40% is a fraction of tumours under
        #: selection, and the fraction of cells inside one tumour -- the
        #: quantity this needs -- is not measured. Any non-zero value is
        #: the caller's ESTIMATE and the report says so.
        self.escape_founder_hla_loh_fraction = float(self.config.get(
            'escape_founder_hla_loh_fraction',
            escape.FOUNDER_HLA_LOH_FRACTION.value))
        #: DE NOVO events by route -- escape that happened at a division
        #: during this run. Deaths do not decrement them: they are
        #: mutation supply, not standing population, and the population
        #: side is escape_distribution().
        #:
        #: Founder seeding is counted SEPARATELY, below, and that
        #: separation is load-bearing rather than tidiness. Until it
        #: existed, _seed_founder_escape went through _record_escape and
        #: so landed in this counter, which meant a founder-seeded run
        #: reported 42 'events' and there was no way to read the de novo
        #: count back out -- the one number that says whether the
        #: literature rate can produce escape at this scale. It cannot:
        #: measured over 12 founder-seeded 24 h Baseline slab runs
        #: (~850 divisions each), de novo events are 0, as 1.55e-6 per
        #: division says they must be.
        self.escape_events: Dict[str, int] = {
            name: 0 for name in escape.ESCAPE_TRAITS}
        #: Escape imposed on the FOUNDING population, by route. An
        #: initial condition, not supply.
        self.escape_founder_events: Dict[str, int] = {
            name: 0 for name in escape.ESCAPE_TRAITS}
        self.escape_founder_cells = 0
        #: True only while _seed_founder_escape is running, so the two
        #: counters above cannot be conflated by a shared code path.
        self._seeding_founders = False
        #: The module's own clock in hours, so an escape event can be
        #: dated. Same pattern as ImmuneModule.clock_h.
        self.clock_h = 0.0

        self.glucose_consumption_normal = self.config.get('glucose_consumption', 0.2)
        self.glucose_consumption_cancer = self.config.get('glucose_consumption_cancer', 0.5)

        # ECM / desmoplasia dynamics (Upgrade 6). Cancer cells deposit
        # local stromal ECM proportional to ``ecm_deposition_rate``
        # (units: per-hour fraction). Anti-fibrotic therapy degrades
        # ECM at ``ecm_degradation_rate`` when ``anti_fibrotic_active``
        # is True.
        #
        # THE RATE WAS 0.05/h, AND THAT IS A ONE-DAY SATURATION.
        # 0.05 per hour fills the whole 0..1 range in 20 h, so measured
        # over a shipped 24 h Baseline run every cancer cell went:
        #
        #     hour   0    4    8   12   16   20
        #     ECM  0.00 0.21 0.41 0.61 0.81 1.00
        #
        # and because ECM attenuates immune motility (ecm_barrier), the
        # infiltrate's speed fell 5.00 -> 0.50 um/min across the same
        # run. That leaves the 2-12 um/min band measured for lymphocyte
        # migration at about hour 13 and ends four times below its
        # floor, so the engine's own migration-speed result held only in
        # the first half of every run. Worse, a 24 h run traversed the
        # entire range of the mechanism, which makes any ECM-dependent
        # outcome a function of WHEN it was read.
        #
        # A stromal reaction is a matter of weeks, not of one day.
        # Taking desmoplasia as developing to ~0.8 over ~6 weeks
        # (0.8 / 1008 h) gives ~8e-4 per hour, which moves ECM by ~0.02
        # over a 24 h run -- i.e. ECM is effectively STATIC on the
        # timescale this engine simulates, and is therefore a property
        # of the tissue a run starts from rather than something that
        # grows during it. ESTIMATE: no measured per-hour collagen
        # deposition rate for prostate adenocarcinoma was found, and the
        # 6-week figure is the order of magnitude of a stromal reaction,
        # not a measurement.
        #
        # Set ``cancer_baseline_ecm`` to give a run a fibrotic tissue;
        # that is the knob desmoplasia belongs on now. It still defaults
        # to 0.0 because no sourced ECM density for prostate exists.
        self.ecm_deposition_rate = float(
            self.config.get('ecm_deposition_rate', 8.0e-4)
        )
        self.ecm_degradation_rate = float(
            self.config.get('ecm_degradation_rate', 0.10)
        )
        self.anti_fibrotic_active = bool(
            self.config.get('anti_fibrotic_active', False)
        )
        # Cancer cells start with this baseline ECM at spawn time,
        # matching the lecture's "PDAC tumors arrive with 80% stroma"
        # observation. Override per cancer-type preset.
        self.cancer_baseline_ecm = float(
            self.config.get('cancer_baseline_ecm', 0.0)
        )

        # Initial layout of the cell population.
        #
        # 'ring' is the historical layout, and every preset that shipped
        # before TissueSlab uses it: see _seed_ring, and the measurement
        # at the top of cognisom/modules/tissue_slab.py for what it
        # actually looks like -- 80 cells on a circle of radius 50 um
        # with a 100 um hole in the middle, which is why the renders of
        # this engine have always looked like a bead necklace rather
        # than tissue.
        #
        # 'slab' seeds a close-packed tissue VOLUME with an embedded
        # tumour nodule instead. Its geometry lives in
        # cognisom.modules.tissue_slab.SLAB_SPEC, which is also what the
        # vascular and immune modules read, so the three cannot disagree
        # about where the tissue is. Per-parameter overrides go in a
        # 'slab' sub-dict; unknown keys there are rejected rather than
        # ignored.
        # 'section' seeds a VOLUME whose occupancy comes from a measured
        # human prostate section instead of being uniformly packed: see
        # cognisom.modules.tissue_section. The lattice pitch is the
        # slab's, because the measured packed cell diameter puts it in
        # the right band; what the section changes is WHERE the cells
        # are, so glands fill and lumens stay empty. The slab fills
        # every site and real prostate fills ~60% of them, so the slab
        # over-connects each cell by about 1.6x -- that difference is
        # the point of the layout, not a shortfall in it.
        self.seeding = str(self.config.get('seeding', 'ring'))
        self.slab_overrides = dict(self.config.get('slab', {}) or {})
        self.section_overrides = dict(self.config.get('section', {}) or {})
        #: Populated by _seed_slab with the geometry it used (extents,
        #: bounds, nodule centre and radius), for callers that need to
        #: frame a camera or check perfusion against it.
        self.slab_layout = None
        #: The same, for _seed_section. Separate from slab_layout so a
        #: consumer cannot read one layout's geometry believing it has
        #: the other's -- their extents and occupancy differ.
        self.section_layout = None

        # Patient HLA-I panel + MHC loader (Upgrade 2). Same scorer the
        # neoantigen predictor uses; auto-picks up MHCflurry when
        # available, otherwise PWM fallback.
        self.hla_alleles: List[str] = list(
            self.config.get('hla_alleles', DEFAULT_HLA_ALLELES)
        )
        #: The simulated patient's somatic missense mutations, as dicts
        #: with gene / aa_change / label / oncogenic -- built by
        #: cognisom.dashboard.patient_engine.engine_inputs, which checks
        #: each against the reference protein and reports every one it
        #: leaves out. Empty means no patient was given, and every cancer
        #: cell carries the historical KRAS_G12D, as before.
        self.patient_mutations: List[Dict[str, Any]] = [
            dict(m) for m in (self.config.get('patient_mutations') or [])
        ]
        #: Whether a patient was given at all. Separate from whether that
        #: patient has presentable mutations: a patient with none must get
        #: a tumour that presents nothing, not the historical KRAS_G12D.
        #: Conflating the two made a "nothing presentable" control quietly
        #: present KRAS.
        self.patient_given = self.config.get('patient_mutations') is not None
        #: label -> the fraction of FOUNDING cancer cells that carry that
        #: mutation. A label that is absent is in every cell, which is
        #: what every run before this did, so an empty mapping changes
        #: nothing. This exists because a tumour is not one genotype: a
        #: resistance mutation is usually in some cells and not others,
        #: and a drug then SELECTS rather than merely failing. Daughters
        #: copy their parent's mutations, so the fraction is free to move
        #: during the run -- that movement is the result.
        self.clonal_fractions: Dict[str, float] = {}
        _known = {m['label'] for m in self.patient_mutations}
        for _label, _fraction in (self.config.get('clonal_fractions') or {}).items():
            _fraction = float(_fraction)
            if _label not in _known:
                raise ValueError(
                    f"clonal fraction given for {_label!r}, which is not one "
                    f"of this patient's engine mutations {sorted(_known)}; "
                    f"refusing to run a subclone of a mutation the tumour "
                    f"does not carry")
            if not 0.0 < _fraction <= 1.0:
                raise ValueError(
                    f"clonal fraction for {_label} must be in (0, 1], got "
                    f"{_fraction}")
            self.clonal_fractions[_label] = _fraction
        #: How many founding cancer cells have been given their mutations;
        #: decides which of them carry a subclonal label.
        self._founding_rank = 0
        #: One record per (gene, mutation) whose peptides bound nothing on
        #: this patient's HLA panel: the best IC50 reached, on which
        #: allele, and which scorer said so. See on_mutation_occurred.
        self.unpresented_mutations: List[Dict[str, Any]] = []
        self._unpresented_seen: set = set()
        self.max_displayed_per_mutation = int(
            self.config.get('max_displayed_per_mutation', 5)
        )
        self._mhc_loader: Optional[MHCLoader] = None

        # Linked molecular module supplies WT protein sequences for
        # neoantigen peptide generation (set via set_molecular_module).
        self.molecular_module = None
        self.circadian_module = None
        self.ar_module = None
        self.androgen_withdrawal_deaths = 0

        # Statistics
        self.total_divisions = 0
        self.total_deaths = 0
        self.total_departed = 0
        self.total_transformations = 0
        self.total_peptides_generated = 0
        self.total_peptides_presented = 0
    
    def initialize(self):
        """Initialize cellular system"""
        print("  Creating initial cell population...")

        # Create initial cells
        n_normal = self.config.get('n_normal_cells', 80)
        n_cancer = self.config.get('n_cancer_cells', 20)

        if self.seeding == 'slab':
            n_normal, n_cancer = self._seed_slab(n_normal, n_cancer)
        elif self.seeding == 'section':
            n_normal, n_cancer = self._seed_section(n_normal, n_cancer)
        else:
            self._seed_ring(n_normal, n_cancer)

        # Initialize MHC loader (Upgrade 2). Construction reaches into
        # NeoantigenPredictor which instantiates a GeneProteinMapper;
        # this preseeds from BUILTIN_PROTEINS and does not touch the
        # network. Doing it here keeps the per-cell update path cheap.
        self._mhc_loader = MHCLoader()

        if self.initial_cell_cycle == 'asynchronous':
            self._stagger_initial_ages()

        # An established tumour has already escaped. Off by default, and
        # it takes no draw at all when off -- see _seed_founder_escape.
        n_founders = self._seed_founder_escape()

        # Subscribe to events
        self.subscribe(EventTypes.EXOSOME_UPTAKEN, self.on_exosome_uptaken)
        self.subscribe(EventTypes.CANCER_KILLED, self.on_cancer_killed)
        self.subscribe(EventTypes.MUTATION_OCCURRED, self.on_mutation_occurred)

        print(f"    ✓ {n_normal} normal cells")
        print(f"    ✓ {n_cancer} cancer cells")
        print(f"    ✓ MHC-I display: {len(self.hla_alleles)} HLA alleles "
              f"({', '.join(self.hla_alleles)})")
        if n_founders:
            print(f"    ✓ {n_founders} founding cancer cells seeded with "
                  f"HLA LOH (ESTIMATE, fraction "
                  f"{self.escape_founder_hla_loh_fraction:.2f})")
    
    def _stagger_initial_ages(self) -> None:
        """Start each initial cell at a random point in its cycle.

        Every cell used to start at age 0 with its lineage's division
        time, so a whole lineage divided on one step -- all 80 normal cells
        at exactly 24 h -- and whether that step fell inside a 24 h run
        depended on whether the float sum of the steps reached 24.0: at
        dt = 0.05 it did and the population doubled, at dt = 0.02 it fell
        short and nothing divided.

        Ages are drawn from the steady-state age distribution of a cycling
        population (core.cell_cycle.steady_state_ages), so the run starts
        from that steady state rather than from a synchronised step.

        The draws come from a generator of their own (cell_cycle_seed), so
        the global stream -- and with it the ring layout the Krogh
        calibration was measured on, and every other module's draws -- is
        unchanged.
        """
        from cognisom.core.cell_cycle import steady_state_ages

        cells = list(self.cells.values())
        periods = [self.division_time_cancer if c.cell_type == 'cancer'
                   else self.division_time_normal for c in cells]
        rng = np.random.default_rng(self.cell_cycle_seed)
        ages = steady_state_ages(periods, rng.random(len(cells)))
        for cell, age in zip(cells, ages):
            cell.age = float(age)

    def _seed_ring(self, n_normal: int, n_cancer: int):
        """The historical layout: normal cells on a circle, cancer in a blob.

        Kept byte-for-byte, including the order of the np.random draws,
        because every preset but TissueSlab uses it and the vascular
        module's Krogh calibration was measured on it.
        """
        # Normal cells (organized in circle)
        for i in range(n_normal):
            angle = 2 * np.pi * i / n_normal
            radius = 50
            x = 100 + radius * np.cos(angle)
            y = 100 + radius * np.sin(angle)
            z = 50 + np.random.uniform(-10, 10)

            self.add_cell(position=[x, y, z], cell_type='normal')

        # Cancer cells (clustered)
        for i in range(n_cancer):
            x = 120 + np.random.uniform(-15, 15)
            y = 120 + np.random.uniform(-15, 15)
            z = 50 + np.random.uniform(-5, 5)

            cell_id = self.add_cell(position=[x, y, z], cell_type='cancer')
            self.cells[cell_id].mutations = self._founding_mutations()
            self.cells[cell_id].mhc1_expression = 0.3  # Downregulated
            # KRAS alone does not drive intrinsic PD-L1, so this stays
            # at 0; PIK3CA/MYC/EGFR mutations would bump it here via
            # _intrinsic_pdl1_for_oncogenes.

    def _seed_slab(self, n_normal, n_cancer):
        """A close-packed tissue slab with a solid tumour nodule in it.

        ``n_normal_cells`` / ``n_cancer_cells`` are honoured when given,
        but they default to the counts in ``SLAB_SPEC`` rather than to
        this module's ring defaults: the slab's extent, its vessel
        lattice and its immune scatter are all derived from the same
        spec, and a population that disagreed with it would be a slab
        whose periphery sits outside every Krogh cylinder. The spec is
        the single source of truth; override ``slab`` consistently in
        the cellular, vascular and immune configs to change the
        geometry.

        Returns the (n_normal, n_cancer) actually seeded.
        """
        from cognisom.modules.tissue_slab import (
            rng_from, seed_slab, slab_spec,
        )

        spec = slab_spec(**self.slab_overrides)
        if 'n_normal_cells' in self.config:
            spec['n_normal'] = int(self.config['n_normal_cells'])
        if 'n_cancer_cells' in self.config:
            spec['n_cancer'] = int(self.config['n_cancer_cells'])

        layout = seed_slab(
            spec['n_normal'], spec['n_cancer'],
            spacing=spec['spacing'],
            layers=spec['layers'],
            center=spec['center'],
            nodule_offset=spec['nodule_offset'],
            jitter=spec['jitter'],
            rng=rng_from(spec['seed']),
        )

        for position in layout['normal']:
            self.add_cell(position=position, cell_type='normal')
        for position in layout['cancer']:
            cell_id = self.add_cell(position=position, cell_type='cancer')
            self.cells[cell_id].mutations = self._founding_mutations()
            self.cells[cell_id].mhc1_expression = 0.3  # Downregulated

        self.slab_layout = layout
        extents = layout['extents']
        print(f"    \u2713 slab {extents[0]:.0f} x {extents[1]:.0f} x "
              f"{extents[2]:.0f} um, {layout['layers']} layers at "
              f"{layout['spacing']:.1f} um")
        print(f"    \u2713 nodule r={layout['nodule_radius']:.1f} um at "
              f"({layout['nodule_center'][0]:.0f}, "
              f"{layout['nodule_center'][1]:.0f}, "
              f"{layout['nodule_center'][2]:.0f})")
        return len(layout['normal']), len(layout['cancer'])

    def _seed_section(self, n_normal, n_cancer):
        """A tissue volume at a measured section's density field.

        Mirrors :meth:`_seed_slab` -- same spec-is-the-single-source-of-
        truth rule, same founding mutations, same MHC-I downregulation
        -- and differs only in where the cells go. Override ``section``
        consistently in the cellular, vascular and immune configs to
        change the geometry, exactly as ``slab`` is overridden.

        The cells' POSITIONS here are lattice sites; what is measured is
        which sites are occupied. The cell TYPES are not measured at all
        -- they come from the population prior as in every other layout,
        and the nodule is carved geometrically -- so this seeds a real
        arrangement of tissue DENSITY and makes no claim about the
        spatial arrangement of cell types. ``section_report`` says so on
        every run; do not report otherwise.

        Returns the (n_normal, n_cancer) actually seeded.
        """
        from cognisom.modules.tissue_section import (
            rng_from, section_report, section_spec, seed_section,
        )

        spec = section_spec(**self.section_overrides)
        if 'n_normal_cells' in self.config:
            spec['n_normal'] = int(self.config['n_normal_cells'])
        if 'n_cancer_cells' in self.config:
            spec['n_cancer'] = int(self.config['n_cancer_cells'])

        layout = seed_section(
            spec['n_normal'], spec['n_cancer'],
            section=spec['section'],
            window=spec['window'],
            bin_um=spec['bin_um'],
            spacing=spec['spacing'],
            jitter=spec['jitter'],
            center=spec['center'],
            nodule_offset=spec['nodule_offset'],
            rng=rng_from(spec['seed']),
        )

        for position in layout['normal']:
            self.add_cell(position=position, cell_type='normal')
        for position in layout['cancer']:
            cell_id = self.add_cell(position=position, cell_type='cancer')
            self.cells[cell_id].mutations = self._founding_mutations()
            self.cells[cell_id].mhc1_expression = 0.3  # Downregulated

        self.section_layout = layout
        report = section_report(layout)
        extents = layout['extents']
        print(f"    ✓ section {extents[0]:.0f} x {extents[1]:.0f} x "
              f"{extents[2]:.0f} um, {layout['layers']} layers at "
              f"{layout['spacing']:.1f} um")
        print(f"    ✓ measured occupancy {report['layer_fill']:.2f} of a "
              f"packed layer, {report['void_fraction_20um']:.1%} void at "
              f"20 um")
        print(f"    ✓ nodule r={layout['nodule_radius']:.1f} um at "
              f"({layout['nodule_center'][0]:.0f}, "
              f"{layout['nodule_center'][1]:.0f}, "
              f"{layout['nodule_center'][2]:.0f})")
        print(f"    ✓ density from {layout['attribution']}")
        return len(layout['normal']), len(layout['cancer'])

    def _founding_mutations(self) -> List[str]:
        """The mutation labels a founding cancer cell carries.

        The patient's, when one was given; otherwise the historical
        KRAS_G12D every run used to carry whoever the patient was.
        """
        if self.patient_given:
            labels = [m['label'] for m in self.patient_mutations]
            if not self.clonal_fractions:
                return labels
            rank = self._founding_rank
            self._founding_rank += 1
            return [label for label in labels
                    if self._founder_carries(label, rank)]
        return ['KRAS_G12D']

    def _founder_carries(self, label: str, rank: int) -> bool:
        """Whether the ``rank``-th founding cancer cell carries ``label``.

        Deterministic and evenly spread through the founding order, NOT
        drawn: a draw would consume the shared random stream and move
        every later event in the run, and it would make the founding
        count itself a random variable. Of N founders exactly
        floor(N * fraction) carry the label.

        Evenly spread is a simplification. A real subclone is a
        contiguous patch, because its cells share an ancestor; this
        scatters them. It is right about HOW MANY and wrong about WHERE.
        """
        fraction = self.clonal_fractions.get(label, 1.0)
        if fraction >= 1.0:
            return True
        eps = 1e-9
        return (math.floor((rank + 1) * fraction + eps)
                > math.floor(rank * fraction + eps))

    def clone_counts(self) -> Dict[str, Dict[str, int]]:
        """For each subclonal label: live cancer cells with and without it."""
        out: Dict[str, Dict[str, int]] = {}
        for label in self.clonal_fractions:
            with_it = without = 0
            for cell in self.cells.values():
                if cell.cell_type != 'cancer' or not cell.alive:
                    continue
                if label in (cell.mutations or []):
                    with_it += 1
                else:
                    without += 1
            out[label] = {'carriers': with_it, 'non_carriers': without}
        return out

    def _present_patient_mutations_on(self, cell_id: int) -> None:
        """Run the patient's mutations through presentation on one cell."""
        carried = self.cells[cell_id].mutations or []
        for mutation in self.patient_mutations:
            # A cell outside the subclone does not make the mutant
            # protein, so it has no mutant peptide to display and a
            # T cell specific for it has nothing to see there.
            if self.clonal_fractions and mutation['label'] not in carried:
                continue
            self.on_mutation_occurred({
                'cell_id': cell_id,
                'gene': mutation['gene'],
                'mutation': mutation['label'],
                'aa_change': mutation['aa_change'],
                'oncogenic': bool(mutation.get('oncogenic', False)),
            })

    def present_patient_mutations(self) -> Dict[str, Any]:
        """Present the patient's mutations on every founding cancer cell.

        Runs each mutation through on_mutation_occurred -- the same path
        a mutation acquired during the run takes -- so the peptides are
        cut from the reference protein, scored against this patient's
        HLA, and kept on the cell only if they bind. Daughters copy their
        parent's displayed peptides, so presentation is inherited.

        Must run after the molecular module is linked, which supplies the
        reference proteins. Returns what happened, including every
        mutation that produced nothing presentable on this patient's
        alleles, because a mutation the immune system cannot see is a
        result, not an error.
        """
        cancer_ids = [cid for cid, cell in self.cells.items()
                      if cell.cell_type == 'cancer']
        report: Dict[str, Any] = {
            'cancer_cells': len(cancer_ids),
            'mutations': [m['label'] for m in self.patient_mutations],
            'hla_alleles': list(self.hla_alleles),
            'presented': [],
            'unpresented': [],
            'cells_displaying': 0,
        }
        if not self.patient_mutations:
            report['reason'] = 'no patient mutations'
            return report
        if self.molecular_module is None:
            report['reason'] = ('no molecular module linked, so there are no '
                                'reference proteins to cut peptides from')
            return report

        before = len(self.unpresented_mutations)
        for cell_id in cancer_ids:
            self._present_patient_mutations_on(cell_id)

        displayed_genes = set()
        for cell_id in cancer_ids:
            peptides = self.cells[cell_id].mhc1_displayed_peptides
            if peptides:
                report['cells_displaying'] += 1
            for pres in peptides:
                displayed_genes.add(pres.peptide.source_gene)
        report['presented'] = [m['label'] for m in self.patient_mutations
                               if m['gene'] in displayed_genes]
        report['unpresented'] = [dict(u) for u in
                                 self.unpresented_mutations[before:]]
        return report

    def add_cell(self, position, cell_type='normal'):
        """Add new cell"""
        cell_id = self.next_cell_id
        self.next_cell_id += 1

        cell = CellState(
            cell_id=cell_id,
            position=np.array(position, dtype=np.float32),
            cell_type=cell_type,
            phase='G1'
        )
        # Seed cancer cells with the configured stromal baseline (e.g.,
        # PDAC presets arrive with 0.5-0.7 already laid down; soft-tissue
        # tumors closer to 0.0).
        if cell_type == 'cancer' and self.cancer_baseline_ecm > 0.0:
            cell.local_ecm_density = float(self.cancer_baseline_ecm)

        if cell.cell_type == 'cancer' and self.ar_dependent_fraction > 0:
            cell.ar_dependent = (cell_id % max(1, round(1.0 / self.ar_dependent_fraction))) == 0
        self.cells[cell_id] = cell
        return cell_id

    def remove_cell(self, cell_id: int):
        """Remove cell"""
        if cell_id in self.cells:
            del self.cells[cell_id]

    def ecm_density_at(
        self,
        position,
        sample_radius_um: float = 30.0,
    ) -> float:
        """Sample the local ECM density at a 3D position.

        Returns the mean ``local_ecm_density`` of cancer cells within
        ``sample_radius_um`` of ``position``. Used by the immune module
        to gate T-cell motility + detection radius (Upgrade 6 ECM
        barrier). Result is clamped to [0, 1]; positions outside any
        tumor return 0.0.
        """
        from cognisom.engine.py.spatial.ecm_barrier import ecm_density_at
        cancers = [
            c for c in self.cells.values()
            if c.cell_type == 'cancer' and c.alive
        ]
        return ecm_density_at(position, cancers, sample_radius_um)
    
    def update(self, dt: float):
        """Update all cells"""
        cells_to_divide = []
        cells_to_die = []
        cells_to_die_withdrawn = []

        # The module's own clock, so an escape event can be dated.
        self.clock_h += dt

        for cell_id, cell in list(self.cells.items()):
            if not cell.alive:
                continue
            
            # Age
            cell.age += dt

            # Metabolism
            self._update_metabolism(cell, dt)
            self._update_androgen(cell, dt)
            self._update_mhc1(cell, dt)

            # ECM dynamics (Upgrade 6). Cancer cells continuously
            # remodel their local stroma; anti-fibrotic therapy
            # degrades ECM uniformly when active.
            if cell.cell_type == 'cancer':
                if self.anti_fibrotic_active:
                    cell.local_ecm_density = max(
                        0.0,
                        cell.local_ecm_density - self.ecm_degradation_rate * dt,
                    )
                else:
                    cell.local_ecm_density = min(
                        1.0,
                        cell.local_ecm_density + self.ecm_deposition_rate * dt,
                    )
            
            # Check for division
            division_time = (self.division_time_cancer if cell.cell_type == 'cancer' 
                           else self.division_time_normal)
            
            # Androgen withdrawal slows a cancer cell's cycle in proportion
            # to its lost AR signalling: a cell at a quarter of eugonadal
            # signalling takes four times as long to divide. Proportional
            # is the ESTIMATE; the direction is Westin 1995.
            signalling = self._ar_signalling(cell_id) if cell.cell_type == 'cancer' else None
            if signalling is not None and signalling < AR_INTACT_ABOVE:
                division_time = division_time / max(signalling, 0.05)
            if cell.age >= division_time and self._division_permitted(cell_id):
                cells_to_divide.append(cell_id)
            
            # Check for death (anoxia, starvation): a rate per hour, the
            # same for normal and cancer cells. See ANOXIC_O2_LEVEL.
            if (cell.oxygen < ANOXIC_O2_LEVEL
                    or cell.glucose < STARVATION_GLUCOSE_MM):
                if np.random.random() < -np.expm1(-self.anoxic_death_per_hour * dt):
                    cells_to_die.append(cell_id)
            # The AR-dependent third dies at a rate that grows with the
            # signalling it has lost. No draw at all while signalling is
            # intact, so an intact run's random stream is untouched.
            if (cell.cell_type == 'cancer' and cell.ar_dependent
                    and cell_id not in cells_to_die):
                s = self._ar_signalling(cell_id)
                if s is not None and s < AR_INTACT_ABOVE:
                    hazard = self.androgen_withdrawal_death_per_hour * (1.0 - s)
                    if np.random.random() < -np.expm1(-hazard * dt):
                        cells_to_die_withdrawn.append(cell_id)
        
        # Process divisions
        for cell_id in cells_to_divide:
            self._divide_cell(cell_id)
        
        # Process deaths
        for cell_id in cells_to_die:
            self._kill_cell(cell_id, cause='hypoxia')
        for cell_id in cells_to_die_withdrawn:
            self.androgen_withdrawal_deaths += 1
            self._kill_cell(cell_id, cause='androgen_withdrawal')
    
    def _update_metabolism(self, cell: CellState, dt: float):
        """Update cell metabolism"""
        if cell.cell_type == 'cancer':
            # Warburg effect
            cell.glucose -= self.glucose_consumption_cancer * dt
            cell.oxygen -= 0.1 * dt
            cell.lactate += LACTATE_PRODUCTION_CANCER_PER_H * dt
            cell.atp += 50 * dt  # Less efficient
        else:
            # Normal metabolism
            cell.glucose -= self.glucose_consumption_normal * dt
            cell.oxygen -= 0.15 * dt
            cell.lactate += LACTATE_PRODUCTION_NORMAL_PER_H * dt
            cell.atp += 100 * dt  # More efficient
        
        # Clamp values
        cell.glucose = max(0, cell.glucose)
        cell.oxygen = max(0, cell.oxygen)
        cell.lactate = max(0, cell.lactate)
        cell.atp = max(0, cell.atp)
    
    def _update_mhc1(self, cell: CellState, dt: float):
        """Surface MHC-I follows the local IFN-gamma.

        IFN-gamma raises class I on tumour cells: a mean +50% across five
        malignant glioma lines (Naganuma et al., J Neurooncol 1998;39:227,
        PMID 9821108), and the same direction in a prostate model, where
        TRAMP-C2 cells are MHC-I-low until IFN-gamma treatment induces
        antigen-processing components (Martini et al., Vaccine
        2010;28:3548, PMID 20304037 -- direction only, no magnitude).

        The GAIN is therefore 0.5 from a non-prostate line, and the
        saturation point (the field level at which the gain is full) has
        no source at all: both are ESTIMATES. What is not an estimate is
        the shape -- the cell relaxes toward the target rather than
        jumping, over MHC1_RESPONSE_H, because surface class I turns over
        in hours and the measurements are at 24-48 h.

        This is what the contact-only PD-L1 bump in the immune module
        cannot do: a BYSTANDER tumour cell near an engagement is made
        more visible, so the effect spreads over the signal's range
        instead of touching only the cell being killed.
        """
        if cell.ifn_gamma_level is None:
            return
        if cell.mhc1_baseline is None:
            cell.mhc1_baseline = float(cell.mhc1_expression)
        drive = min(1.0, max(0.0, float(cell.ifn_gamma_level)
                             / MHC1_IFN_SATURATION))
        target = cell.mhc1_baseline * (1.0 + MHC1_IFN_GAIN * drive)
        cell.mhc1_expression += (target - cell.mhc1_expression) * (
            1.0 - np.exp(-dt / MHC1_RESPONSE_H))

    def _update_androgen(self, cell: CellState, dt: float):
        """Tissue DHT follows its two supplies.

        The circulating supply is the free testosterone the field put at
        this cell, referenced to the eugonadal wall level at which tissue
        DHT is the Titus value; the intracrine supply does not move with
        castration. Linear in testosterone between the two measured points
        (eugonadal and castrate) -- the saturable 5a-reductase form waits
        on a sourced Km, and with two anchor points a Km is not
        identifiable anyway. First-order relaxation, so a step in serum
        testosterone reaches the tissue over DHT_RELAXATION_H.
        """
        if cell.testosterone_nM is None:
            return
        from cognisom.modules.vascular_module import EUGONADAL_FREE_TESTOSTERONE_NM
        supply = max(0.0, float(cell.testosterone_nM)) / EUGONADAL_FREE_TESTOSTERONE_NM
        target = TISSUE_DHT_REFERENCE_NM * (
            (1.0 - DHT_INTRACRINE_FRACTION) * supply + DHT_INTRACRINE_FRACTION)
        cell.dht_nM += (target - cell.dht_nM) * (1.0 - np.exp(-dt / DHT_RELAXATION_H))

    def _divide_cell(self, cell_id: int):
        """Cell division.

        Inheritance policy for the daughter's non-default state. Each
        field is a modelling decision, not a mechanical copy:

        * ``mutations`` / ``mhc1_expression`` -- clonal, inherited.
        * ``mhc1_displayed_peptides`` -- inherited. The daughter carries
          the parent's genome deltas (MolecularModule.on_cell_divided
          forks the parent's CellGenomeView on this event), so it
          transcribes the same mutant proteins and presents the same
          neoantigens on the same patient HLA panel. Dropping the list
          made every daughter of a displaying tumor cell invisible to
          ImmuneModule._recognize, which gates T-cell recognition on it
          being non-empty -- and no MUTATION_OCCURRED ever fires again
          for an inherited mutation, so the daughter stayed invisible
          for life. That was a silent bias toward tumor escape growing
          with every division.
          Copying is correct while presentation is a pure function of
          the genome. If it ever becomes state-dependent (proteasome
          composition, TAP transport, HLA LOH), regenerate from the
          forked genome here instead of copying.
          The list is copied, not aliased; MHCPresentation is a frozen
          dataclass so the entries themselves are safe to share.
        * ``pdl1_expression`` -- inherited. PD-L1 is signalling state,
          not genome, so neither full inheritance nor a reset to 0 is
          right: surface protein is partitioned between daughters and
          then decays with the IFN-gamma stimulus. Inheriting the
          parent's level is the closer approximation, since the
          daughter sits ~5 um away in the same cytokine field that
          induced the parent. Simplification: no halving at mitosis and
          no decay term -- adaptive PD-L1 induced once is retained for
          the clone's life.
        * ``local_ecm_density`` -- NOT inherited; re-sampled. ECM is a
          property of the location, not of the cell, so the daughter
          takes the stromal density already present where it lands.
          Sampled before the daughter is registered, so its own seeded
          baseline does not pull the neighbourhood mean toward
          ``cancer_baseline_ecm``.
        * ``escape_traits`` / ``escape_born_at_h`` / ``escape_lost_hla``
          -- clonal, inherited, and then the daughter MAY DIFFER. Copy
          alone was the defect this path carried: with mhc1_expression,
          mhc1_baseline and pdl1_expression copied exactly from a
          population seeded uniformly at 0.3, immune pressure had
          nothing to select on and escape could only be imposed. See
          _maybe_escape; with no escape event the daughter is exactly
          what it was before, down to the global RNG stream.
        """
        if cell_id not in self.cells:
            return
        
        parent = self.cells[cell_id]
        
        # Create daughter cell
        offset = np.random.randn(3) * 5  # Small offset
        daughter_pos = parent.position + offset

        # Sample the stroma at the daughter's position BEFORE the
        # daughter exists, so it is not averaged into its own sample.
        sampled_ecm = (
            self.ecm_density_at(daughter_pos)
            if parent.cell_type == 'cancer' else None
        )

        daughter_id = self.add_cell(
            position=daughter_pos,
            cell_type=parent.cell_type
        )
        
        daughter = self.cells[daughter_id]
        # The daughter shares the parent's cytoplasm: it is born at the
        # parent's tissue androgen state, not at the eugonadal reference.
        # Without this a 24 h castrate run read 29% of baseline instead
        # of 25%, lifted by the cells born in its last few hours.
        daughter.dht_nM = parent.dht_nM
        daughter.testosterone_nM = parent.testosterone_nM
        daughter.ar_dependent = parent.ar_dependent
        daughter.mutations = parent.mutations.copy()
        daughter.mhc1_expression = parent.mhc1_expression
        # ...and the level it returns to without IFN-gamma, or a daughter
        # born into a signal would decay to 1.0 when it faded.
        daughter.mhc1_baseline = parent.mhc1_baseline
        # Oxygen and glucose are concentrations, so both cells carry the
        # parent's. A daughter used to start at the CellState defaults --
        # arterial O2 and plasma glucose -- so one born in a hypoxic corner
        # was briefly rescued from it, and when a whole lineage divided on
        # one step the tissue's mean O2 read as if half of it sat in an
        # artery. The tissue engine's daughters already inherit their state.
        daughter.oxygen = parent.oxygen
        daughter.glucose = parent.glucose
        daughter.mhc1_displayed_peptides = list(
            parent.mhc1_displayed_peptides
        )
        daughter.pdl1_expression = parent.pdl1_expression
        if sampled_ecm is not None:
            daughter.local_ecm_density = sampled_ecm
        # Escape traits are clonal: a grandchild keeps what its
        # grandparent acquired. Copied BEFORE the draw below, which may
        # then add to them.
        daughter.escape_traits = parent.escape_traits
        daughter.escape_born_at_h = parent.escape_born_at_h
        daughter.escape_lost_hla = parent.escape_lost_hla
        self._maybe_escape(daughter)

        # Reset parent age
        parent.age = 0.0
        
        # Emit event
        self.emit_event(EventTypes.CELL_DIVIDED, {
            'cell_id': cell_id,
            'daughter_id': daughter_id,
            'cell_type': parent.cell_type,
            'position': parent.position.tolist()
        })
        
        self.total_divisions += 1
    
    # ── heritable immune escape ──────────────────────────────────────

    def _maybe_escape(self, daughter: CellState) -> Optional[str]:
        """A daughter may differ from its parent. Returns the route that
        fired, or None.

        The whole point of the module this lives in: an established
        tumour is one that escaped, and nothing here could escape
        because nothing varied. Every cancer cell was seeded at
        mhc1_expression 0.3 and every daughter copied its parent
        exactly, so selection had no raw material.

        What fires, and why those two routes and not one
        ------------------------------------------------
        * An intact cell loses ONE HLA haplotype at the sourced rate
          (escape_variation.MHC1_ALLELE_LOSS_PER_DIVISION, 1.55e-6
          per division, the geometric mean of five measured
          per-cell-per-generation rates). Surface class I falls to
          (n-1)/n of its baseline and the presentations bound to the
          lost allele are dropped. The cell is STILL recognised, at a
          lower signal, so checkpoint blockade still reaches it.
        * A cell that already lost one haplotype loses presentation
          altogether at MHC1_SECOND_HIT_PER_DIVISION (an ESTIMATE: no
          per-division B2M rate exists). mhc1_expression AND
          mhc1_baseline go to 0 and the displayed peptides are cleared,
          which puts the cell under _recognize's 0.05 floor -- invisible
          to T cells and NOT rescuable by blockade. Setting the baseline
          to 0, not just the level, is what makes that stick: a
          B2M-null cell does not answer IFN-gamma with surface class I,
          and _update_mhc1 relaxes toward baseline * (1 + gain), so a
          zero baseline is a zero target. The IFN-gamma gain and
          saturation are untouched -- that mechanism is the one thing
          here that passed an external test on 592 TCGA tumours.

        Making total loss a SECOND hit rather than an independent
        first-order route is the structural claim: biallelic
        inactivation needs two hits, so de novo total loss is ~mu^2 and
        effectively never, while total loss from an already-hemizygous
        cell is ~mu. That matches B2M loss showing up under therapy
        pressure (Sade-Feldman 2017) rather than in naive tumours.

        Which allele is lost is drawn UNIFORMLY from the panel. A draw
        biased toward the allele carrying the most neoantigens would
        impose the selection this module exists to let the immune module
        discover; a uniform supply under selection is what produces
        McGranahan's parallel evolution.

        PD-L1 is last and inert: the heritable rate is 0.0 because every
        PD-L1 genomic number found is a prevalence in an established
        tumour, never a rate, and the engine already carries the
        dominant inducible routes (see escape_variation).

        Cancer cells only. de Nooij measured these rates in NORMAL
        lymphoid lines, so the event is not tumour-specific -- but in
        this engine a normal cell's mhc1_expression is read by nothing
        except NK missing-self, where lowering it would invent an
        artefact at a rate too low to ever be observed. The consequence
        for any expectation arithmetic is that the denominator is CANCER
        divisions and not a run's division count: on the 24 h Baseline
        slab that is 53-56 of ~849, and on the ring presets it is ZERO,
        so a ring run never reaches this method's draw.

        THE CONVENTION, because a caller scaling mu needs to know which
        quantity it is scaling. The draw is taken ONCE PER DIVISION, on
        the new cell; the parent persists in self.cells, has its age
        reset and is never drawn for. The cited rates are explicitly
        "per cell per generation" (de Nooij-van Dalen 1998, PMID
        9443039), which applies to BOTH products of a division, so the
        realised rate here is mu per new cell and about half the
        literature quantity per division event. Immaterial at preset
        scale -- 53 or 974 divisions give 0 events either way -- and it
        is a factor of 2 on anything deliberately scaled.

        Every draw is from self._escape_rng_*, never the global stream,
        so a pinned run is unchanged whether or not an event fires.
        """
        if not self.escape_enabled or daughter.cell_type != 'cancer':
            return None

        route = None
        traits = daughter.escape_traits
        if escape.TRAIT_TOTAL_LOSS in traits:
            pass                      # nothing left to lose; no draw
        elif escape.TRAIT_ALLELE_LOSS in traits:
            if (self.escape_mhc1_second_hit_per_division > 0.0
                    and self._escape_rng_mhc1.random()
                    < self.escape_mhc1_second_hit_per_division):
                self._lose_mhc1_entirely(daughter)
                route = escape.TRAIT_TOTAL_LOSS
        elif (self.escape_mhc1_loss_per_division > 0.0
                and self._escape_rng_mhc1.random()
                < self.escape_mhc1_loss_per_division):
            route = self._lose_one_allele(daughter)

        if (self.escape_pdl1_gain_per_division > 0.0
                and escape.TRAIT_PDL1_GAIN not in daughter.escape_traits
                and self._escape_rng_pdl1.random()
                < self.escape_pdl1_gain_per_division):
            daughter.pdl1_expression = float(min(
                1.0, daughter.pdl1_expression
                + self.escape_pdl1_gain_magnitude))
            self._record_escape(daughter, escape.TRAIT_PDL1_GAIN)
            route = route or escape.TRAIT_PDL1_GAIN

        return route

    def _presentable_on(self, cell: CellState,
                        presentations: List[MHCPresentation]
                        ) -> List[MHCPresentation]:
        """The subset of ``presentations`` this cell can actually display.

        A cell with no surface class I displays nothing; a cell that has
        lost an HLA allele displays nothing bound to it. Both are
        no-ops on a cell with no escape trait, which is every cell in a
        run where nothing escaped, so the shipped runs are untouched.

        The result is a cell that cannot present a mutation, per cell
        rather than per patient. It is NOT recorded in
        ``unpresented_mutations``: that diagnostic runs in
        ``on_mutation_occurred`` on the RAW score_all output, before
        this filter is applied, so when score_all returns binders and
        this filter empties them execution falls through -- kept == [],
        nothing appended, no PEPTIDE_PRESENTED event, and
        unpresented_mutations stays empty. The population of cells that
        went dark this way is found only in
        ``escape_distribution()['n_escaped_without_display']``.
        """
        if not cell.escape_traits:
            return presentations
        if escape.TRAIT_TOTAL_LOSS in cell.escape_traits:
            return []
        if not cell.escape_lost_hla:
            return presentations
        return [p for p in presentations
                if p.hla_allele not in cell.escape_lost_hla]

    def _record_escape(self, cell: CellState, trait: str) -> None:
        """Add a trait, date the lineage's first escape, count the event.

        The count goes to escape_founder_events while the founding
        population is being seeded and to escape_events otherwise, so
        the de novo supply stays readable in a run that also imposed
        escape on its founders. Both routes share this method, which is
        why the flag is here rather than at the two call sites.
        """
        if trait not in cell.escape_traits:
            cell.escape_traits = tuple(cell.escape_traits) + (trait,)
        if cell.escape_born_at_h is None:
            cell.escape_born_at_h = float(self.clock_h)
        counter = (self.escape_founder_events if self._seeding_founders
                   else self.escape_events)
        counter[trait] = counter.get(trait, 0) + 1

    def _lose_one_allele(self, cell: CellState) -> str:
        """One panel allele goes: the partial route, still recognised.

        ONE ALLELE, not one haplotype, and the name says so since
        2026-10-28. A class I haplotype is an A, a B and a C allele in
        cis; this removes a single entry of self.hla_alleles, which is
        the quantity McGranahan 2017 measures ("Allele-Specific HLA
        Loss") and the quantity de Nooij's assay rates. The engine's
        panels carry no phase information, so no haplotype can be
        identified to drop; see escape_variation's "ALLELE, not
        HAPLOTYPE" section.

        The REPERTOIRE hit is the patient's own -- every presentation on
        the lost allele is dropped, so how much this costs the cell
        depends on where this patient's neoantigens bind (McGranahan's
        CRUK0020 lost 92% of its predicted neoantigens to its lost
        alleles; a patient whose peptides avoid that allele loses
        nothing). The SURFACE hit is arithmetic: (n-1)/n of the alleles
        are left, which is a PER-ALLELE magnitude -- a true haplotype
        loss on a 6-allele genotype would leave 0.5, not 5/6.

        A one-allele panel has no partial state, so the event is the
        total one instead -- a cell with no presenting allele left is
        the invisible phenotype whatever the mechanism was called. The
        trait actually recorded is returned, so the caller cannot report
        a partial loss that was in fact a total one.

        A consequence worth stating, because it looks like a bug and is
        not: a cell whose ENTIRE displayed repertoire sat on the lost
        allele becomes unrecognised even though it still has surface
        class I, since _recognize also requires a non-empty presentation
        list. That is the correct reading of McGranahan -- if all of a
        patient's neoantigens bind the lost haplotype, blockade has
        nothing to un-block -- and it is why the consequence of this
        route is patient-specific rather than a constant.
        """
        panel = [a for a in self.hla_alleles if a not in cell.escape_lost_hla]
        if not panel:
            self._lose_mhc1_entirely(cell)
            return escape.TRAIT_TOTAL_LOSS
        lost = panel[int(self._escape_rng_mhc1.integers(len(panel)))]
        fraction = escape.surviving_surface_fraction(len(panel))
        if fraction <= 0.0:
            cell.escape_lost_hla = tuple(cell.escape_lost_hla) + (lost,)
            self._lose_mhc1_entirely(cell)
            return escape.TRAIT_TOTAL_LOSS

        if cell.mhc1_baseline is None:
            # Same convention as _update_mhc1: the level a cell returns
            # to with no IFN-gamma around is where it sits now. Capture
            # it before scaling, or the drop would be undone the first
            # time the IFN-gamma response ran.
            cell.mhc1_baseline = float(cell.mhc1_expression)
        cell.mhc1_baseline = float(cell.mhc1_baseline) * fraction
        cell.mhc1_expression = float(cell.mhc1_expression) * fraction
        cell.escape_lost_hla = tuple(cell.escape_lost_hla) + (lost,)
        self._record_escape(cell, escape.TRAIT_ALLELE_LOSS)
        # Recorded first, then the repertoire is re-derived through the
        # same filter the ingest path uses, so there is one definition
        # of what a cell can display and not two that can drift apart.
        cell.mhc1_displayed_peptides = self._presentable_on(
            cell, cell.mhc1_displayed_peptides)
        return escape.TRAIT_ALLELE_LOSS

    def _lose_mhc1_entirely(self, cell: CellState) -> None:
        """No surface class I at all: the T-cell-invisible route.

        Baseline as well as level, so IFN-gamma cannot raise it back --
        _update_mhc1 relaxes toward mhc1_baseline * (1 + gain * drive),
        and a B2M-null cell has no class I to induce.
        """
        cell.mhc1_expression = 0.0
        cell.mhc1_baseline = 0.0
        cell.mhc1_displayed_peptides = []
        self._record_escape(cell, escape.TRAIT_TOTAL_LOSS)

    def _seed_founder_escape(self) -> int:
        """Seed a fraction of the FOUNDING cancer cells already escaped.

        Off by default (see escape_founder_hla_loh_fraction), because
        the prevalence data is per tumour and the quantity this needs is
        per cell. When off, no draw is taken at all, so the founder
        sub-stream is not advanced either.

        This exists because the de novo route is silent at preset scale:
        one expected event needs 6.5e5 divisions and a 24 h Baseline
        slab run makes 56-59 CANCER divisions (of 849 total; see
        _maybe_escape, which draws only for a cancer daughter). The
        item's own premise is the way out -- an established tumour has
        already escaped -- so a run that wants escape present starts
        with it present and says that is what it did, instead of raising
        mu until a demo works.

        ``escape_enabled`` gates this route too, which it did NOT before
        2026-10-28: the guard was on the founder fraction alone, so a
        run that declared escape disabled and also named a founder
        fraction started with 60-70% of its tumour carrying HLA LOH and
        a mean surface MHC-I of 0.23 instead of 0.30, while
        escape_distribution() reported 'enabled': False. Measured on 50
        cancer cells at founder 0.5, escape_enabled False gave
        founder_cells 30 and n_escaped 30 -- identical to the enabled
        run. One flag now means one thing: no escape, by either route.
        """
        if (not self.escape_enabled
                or self.escape_founder_hla_loh_fraction <= 0.0):
            return 0
        founders = [c for c in self.cells.values() if c.cell_type == 'cancer']
        n = 0
        # Counted as an initial condition, not as mutation supply: see
        # escape_events / escape_founder_events.
        self._seeding_founders = True
        try:
            for cell in founders:
                if (self._escape_rng_founder.random()
                        < self.escape_founder_hla_loh_fraction):
                    self._lose_one_allele(cell)
                    n += 1
        finally:
            self._seeding_founders = False
        self.escape_founder_cells = n
        return n

    def escape_distribution(self) -> Dict[str, Any]:
        """What escape traits the living tumour carries, and what it cost.

        Counts, not rates: ``events`` is the DE NOVO mutation supply over
        the life of the run (a cell that died still happened),
        ``founder_events`` is the escape imposed on the founding
        population before the run began, and ``traits`` is what the
        standing population carries now. All three are reported because
        they part company: supply and standing population part company
        under selection, which is the point, and supply and founder
        seeding must never be added together -- a founder-seeded run
        that reported 42 'events' would look like a run in which the
        literature rate produced 42, when measured de novo events over
        12 such runs are 0.

        ``n_below_recognition`` counts cells under
        escape_variation.T_CELL_RECOGNITION_MHC1_FLOOR -- the value
        mirrored from ImmuneModule._recognize, pinned to it by behaviour
        in tests/test_escape_can_be_inherited.py. It is NOT the whole
        T-cell-invisible population, and reporting only it would have
        been misleading: _recognize also needs a non-empty presentation
        list, so an escaped cell whose entire repertoire sat on the lost
        allele is invisible at surface MHC-I 0.2. Measured on the
        default panel with the real MHCflurry predictor, that is the
        DOMINANT form: KRAS G12D binds HLA-A*02:01 and nothing else, so
        2 of 12 founders seeded with HLA LOH displayed nothing while
        n_below_recognition was 0. ``n_escaped_without_display`` is that
        population, and ``n_tcell_invisible`` is the UNION of the two --
        the number to read if only one is read. The union, not the sum:
        a total-loss cell is in both, because it has neither surface
        class I nor peptides.
        """
        cancer = [c for c in self.cells.values()
                  if c.alive and c.cell_type == 'cancer']
        floor = escape.T_CELL_RECOGNITION_MHC1_FLOOR
        traits: Dict[str, int] = {name: 0 for name in escape.ESCAPE_TRAITS}
        lost_alleles: Dict[str, int] = {}
        for cell in cancer:
            for trait in cell.escape_traits:
                traits[trait] = traits.get(trait, 0) + 1
            for allele in cell.escape_lost_hla:
                lost_alleles[allele] = lost_alleles.get(allele, 0) + 1
        escaped = [c for c in cancer if c.escape_traits]
        invisible = [c for c in cancer if c.mhc1_expression < floor]
        dark = [c for c in escaped if not c.mhc1_displayed_peptides]
        n = len(cancer)
        return {
            'enabled': self.escape_enabled,
            'seed': self.escape_seed,
            'n_cancer': n,
            'traits': traits,
            'n_escaped': len(escaped),
            'fraction_escaped': (len(escaped) / n) if n else 0.0,
            'mean_mhc1': (float(np.mean([c.mhc1_expression for c in cancer]))
                          if n else 0.0),
            'recognition_threshold': floor,
            'n_below_recognition': len(invisible),
            'fraction_below_recognition': (len(invisible) / n) if n else 0.0,
            # Escaped cells displaying nothing at all: invisible to T
            # cells whatever their surface level, because _recognize
            # needs a presentation. The repertoire route out of the
            # T-cell loop, as opposed to the surface one above.
            'n_escaped_without_display': len(dark),
            # The UNION of the two routes, which is what a reader that
            # plots one number wants. It is NOT the sum: a total-loss
            # cell is below the floor AND displaying nothing, so it is
            # in both lists above and adding them double-counts it.
            'n_tcell_invisible': len(
                {c.cell_id for c in invisible} | {c.cell_id for c in dark}),
            'lost_hla_alleles': lost_alleles,
            'events': dict(self.escape_events),
            'founder_events': dict(self.escape_founder_events),
            'founder_cells': self.escape_founder_cells,
            'first_escape_at_h': (
                min(c.escape_born_at_h for c in escaped) if escaped else None),
            'calibration': escape.calibration_report(
                mhc1_loss=self.escape_mhc1_loss_per_division,
                second_hit=self.escape_mhc1_second_hit_per_division,
                pdl1_gain=self.escape_pdl1_gain_per_division,
                founder_fraction=self.escape_founder_hla_loh_fraction,
                n_alleles=len(self.hla_alleles),
            ),
        }

    def depart_cell(self, cell_id: int, destination: str = 'lymphatic'):
        """A living cell leaves the tissue -- intravasation into a
        lymphatic or a vessel. It is not a death and is not counted as
        one, but every module that tracks the cell must drop it, and
        they all listen for CELL_DIED, so that is the event it rides on
        with ``cause`` naming the exit. Nothing filters CELL_DIED by
        cause; a listener that starts to must treat 'intravasation' as
        a departure."""
        if cell_id not in self.cells:
            return
        cell = self.cells[cell_id]
        cell.alive = False
        self.emit_event(EventTypes.CELL_DIED, {
            'cell_id': cell_id,
            'cell_type': cell.cell_type,
            'cause': 'intravasation',
            'destination': destination,
            'position': cell.position.tolist()
        })
        self.total_departed += 1
        self.remove_cell(cell_id)

    def _kill_cell(self, cell_id: int, cause: str = 'unknown'):
        """Cell death"""
        if cell_id not in self.cells:
            return
        
        cell = self.cells[cell_id]
        cell.alive = False
        
        # Emit event
        self.emit_event(EventTypes.CELL_DIED, {
            'cell_id': cell_id,
            'cell_type': cell.cell_type,
            'cause': cause,
            'position': cell.position.tolist()
        })
        
        self.total_deaths += 1
        
        # Remove cell
        self.remove_cell(cell_id)
    
    def transform_cell(self, cell_id: int, mutations=None):
        """Transform a normal cell to cancer.

        ``mutations`` is the list of ``(gene, label)`` pairs responsible
        for the transformation, as written into the recipient's
        CellGenomeView. The phenotype labels recorded on the cell are
        derived from them.

        This used to append the literal string ``'KRAS_G12D'`` no matter
        what actually drove the transformation, so a cell transformed by
        transferred BRAF V600E carried a genome delta for BRAF and a
        phenotype label claiming KRAS. The genome layer was right and the
        label contradicted it.

        The default is kept for callers that transform without
        attributing a specific mutation (spontaneous transformation),
        which is the only way the legacy label is still produced.
        """
        if cell_id not in self.cells:
            return

        cell = self.cells[cell_id]
        if cell.cell_type != 'normal':
            return

        # Transform
        cell.cell_type = 'cancer'
        if mutations:
            for gene_name, mutation_label in mutations:
                label = f'{gene_name}_{mutation_label}'
                if label not in cell.mutations:
                    cell.mutations.append(label)
        elif self.patient_given:
            # The cell joins THIS patient's tumour: it carries the
            # patient's mutations and presents them, instead of the
            # historical KRAS_G12D the patient does not have.
            for label in self._founding_mutations():
                if label not in cell.mutations:
                    cell.mutations.append(label)
        else:
            cell.mutations.append('KRAS_G12D')
        cell.mhc1_expression = 0.3
        if self.patient_given and not mutations and self.molecular_module is not None:
            self._present_patient_mutations_on(cell_id)
        
        # Emit event
        self.emit_event(EventTypes.CELL_TRANSFORMED, {
            'cell_id': cell_id,
            'position': cell.position.tolist(),
            'mutations': cell.mutations
        })
        
        self.total_transformations += 1
    
    def get_state(self) -> Dict[str, Any]:
        """Return current cellular state"""
        alive_cells = [c for c in self.cells.values() if c.alive]
        cancer_cells = [c for c in alive_cells if c.cell_type == 'cancer']
        normal_cells = [c for c in alive_cells if c.cell_type == 'normal']
        
        return {
            'n_cells': len(alive_cells),
            'n_cancer': len(cancer_cells),
            'n_normal': len(normal_cells),
            'total_divisions': self.total_divisions,
            'total_deaths': self.total_deaths,
            'total_departed': self.total_departed,
            'androgen_withdrawal_deaths': self.androgen_withdrawal_deaths,
            'n_ar_dependent': sum(1 for c in alive_cells if c.cell_type == 'cancer' and c.ar_dependent),
            'total_transformations': self.total_transformations,
            'avg_oxygen': np.mean([c.oxygen for c in alive_cells]) if alive_cells else 0,
            'avg_glucose': np.mean([c.glucose for c in alive_cells]) if alive_cells else 0,
            'avg_dht_nM': float(np.mean([c.dht_nM for c in alive_cells])) if alive_cells else 0.0,
            # Mutations whose peptides bound nothing on this patient's HLA
            # panel, with the best IC50 each reached and the scorer that
            # said so. Empty when every mutation presented something.
            'unpresented_mutations': list(self.unpresented_mutations),
            # Heritable immune escape: what the standing tumour carries,
            # how many events the run has supplied, and the provenance
            # of every rate behind them. See escape_distribution.
            'escape': self.escape_distribution(),
            # The stromal barrier. Reported because it was INVISIBLE: ECM
            # attenuates immune motility on every step and no state
            # output mentioned it, so a run whose infiltrate had been
            # slowed tenfold said nothing about why. `max` is the field
            # that matters -- it saturated at 1.0 within a day under the
            # old deposition rate.
            'ecm': self.ecm_distribution(),
        }

    def ecm_distribution(self) -> Dict[str, Any]:
        """The stromal barrier across living cancer cells, and its basis.

        ECM gates immune motility and sensing through
        engine/py/spatial/ecm_barrier, whose blocking factor is an
        unsourced tuning constant exceeding the nearest measured anchor
        -- so a run has to be able to say what its own barrier was.
        """
        from cognisom.engine.py.spatial.ecm_barrier import (
            DEFAULT_MOTILITY_BLOCK, DEFAULT_MOTILITY_BLOCK_STATUS,
            MEASURED_COMPARTMENT_MOTILITY_RATIO, MEASURED_COMPARTMENT_SOURCE,
        )
        values = [float(c.local_ecm_density)
                  for c in self.cells.values()
                  if c.alive and c.cell_type == 'cancer']
        return {
            'n_cancer_cells': len(values),
            'min': min(values) if values else None,
            'mean': float(np.mean(values)) if values else None,
            'max': max(values) if values else None,
            'saturated_fraction': (
                sum(1 for v in values if v >= 0.999) / len(values)
                if values else None),
            'deposition_rate_per_h': self.ecm_deposition_rate,
            'degradation_rate_per_h': self.ecm_degradation_rate,
            'anti_fibrotic_active': self.anti_fibrotic_active,
            'baseline_at_spawn': self.cancer_baseline_ecm,
            'motility_block': {
                'value': DEFAULT_MOTILITY_BLOCK,
                'status': DEFAULT_MOTILITY_BLOCK_STATUS,
                'max_slowdown': (1.0 / (1.0 - DEFAULT_MOTILITY_BLOCK)
                                 if DEFAULT_MOTILITY_BLOCK < 1.0 else None),
                'measured_anchor': MEASURED_COMPARTMENT_MOTILITY_RATIO,
                'measured_anchor_source': MEASURED_COMPARTMENT_SOURCE,
            },
            # Only cancer cells carry ECM, so normal prostate stroma has
            # none -- which is backwards for a gland that is largely
            # stroma. No fibroblast exists to make it; see the roadmap.
            'produced_by': 'cancer cells only (no fibroblast lineage exists)',
        }
    
    def set_molecular_module(self, molecular_module):
        """Link to molecular module for reference protein lookups.

        Required for the MUTATION_OCCURRED -> PEPTIDE_GENERATED ->
        PEPTIDE_PRESENTED chain (Upgrade 2): the molecular module owns
        the canonical WT protein sequence; the cellular module owns
        per-cell MHC-I display state.
        """
        self.molecular_module = molecular_module

    def set_circadian_module(self, circadian_module):
        """Link to the circadian module, which gates division.

        The clock controls the timing of mitosis through WEE1 (Matsuo et
        al., Science 2003, PMID 12934012). Linked, a cell that has reached
        its division time waits for its own clock's permissive window;
        its age keeps accruing, so the division is deferred, not lost.
        Unlinked, division is as before: age alone decides.
        """
        self.circadian_module = circadian_module

    def set_ar_module(self, ar_module):
        """Link the AR signalling module: a cancer cell's division time and,
        for the ar_dependent subset, its death hazard follow its AR-DHT
        complex relative to the eugonadal state. Unlinked, androgen is
        invisible to cell fate, as before."""
        self.ar_module = ar_module

    def _ar_signalling(self, cell_id: int):
        ar = self.ar_module
        if ar is None:
            return None
        return ar.signalling_fraction(cell_id)

    def _division_permitted(self, cell_id: int) -> bool:
        clock = self.circadian_module
        if clock is None:
            return True
        return bool(clock.can_divide(cell_id))

    # Event handlers
    def on_exosome_uptaken(self, data):
        """Handle exosome uptake.

        Oncogenic cargo can transform a normal recipient (30% probability).
        On transformation, cargo mutations are written into the recipient's
        CellGenomeView as sparse deltas BEFORE the cell-type flip, so
        daughter cells inherit the transmitted mutation through
        CellGenomeView.fork() (Claim 4 architecture) and the
        MUTATION_OCCURRED chain engages for the recipient (Claim 1 chain).
        """
        cell_id = data['cell_id']

        if cell_id in self.cells and data['cargo']['oncogenic']:
            if self.cells[cell_id].cell_type == 'normal':
                if np.random.random() < 0.3:  # 30% chance
                    cargo_mutations = data.get('cargo', {}).get('mutations', [])
                    if self.molecular_module is not None:
                        for gene_name, mutation_label in cargo_mutations:
                            self.molecular_module.introduce_mutation(
                                cell_id, gene_name, mutation_label
                            )
                    self.transform_cell(cell_id, mutations=cargo_mutations)

    def on_cancer_killed(self, data):
        """Handle cancer cell killed by immune system"""
        cell_id = data['cell_id']
        self._kill_cell(cell_id, cause='immune_killed')

    def on_mutation_occurred(self, data):
        """Generate neoantigen peptides and load them onto MHC-I.

        Closed-loop neoantigen presentation, stage 1 -> 2 (Upgrade 2):
            MUTATION_OCCURRED  ->  PEPTIDE_GENERATED  ->  PEPTIDE_PRESENTED

        Required event fields:
            cell_id (int), gene (str), mutation (str), aa_change (str)
            optional: impact_score, oncogenic

        Skips silently when:
            * the cell is not tracked here,
            * no aa_change is present (synonymous / nonsense mutations
              do not produce a canonical missense neoantigen),
            * the linked molecular module cannot supply a WT protein,
            * aa_change is not parseable as "WTposMUT" (e.g., "G12D").
        """
        cell_id = data.get('cell_id')
        gene = data.get('gene')
        mutation_label = data.get('mutation')
        aa_change = data.get('aa_change')

        if cell_id not in self.cells or self.molecular_module is None:
            return

        # Intrinsic PD-L1 induction (Type III TME mechanism): some
        # oncogenes constitutively up-regulate PD-L1 via downstream
        # signalling independent of TIL pressure. Fire this as soon as
        # the mutation lands, regardless of whether we go on to
        # generate a presentable neoantigen.
        if gene and data.get('oncogenic') and gene in _INTRINSIC_PDL1_INDUCERS:
            cell = self.cells[cell_id]
            cell.pdl1_expression = float(min(
                1.0,
                cell.pdl1_expression + _INTRINSIC_PDL1_INDUCERS[gene],
            ))

        if not gene or not aa_change:
            return

        # Parse "G12D" -> ("G", 12, "D"). Reject anything not matching
        # the simple missense pattern; nonsense / frameshift / start-loss
        # paths are handled by separate event types in future work.
        wt_aa, pos_str, mut_aa = aa_change[:1], aa_change[1:-1], aa_change[-1:]
        if not (wt_aa.isalpha() and mut_aa.isalpha() and pos_str.isdigit()):
            return
        mut_position_1based = int(pos_str)

        wt_protein = self.molecular_module.get_reference_protein(gene)
        if not wt_protein or mut_position_1based < 1 or mut_position_1based > len(wt_protein):
            return
        # Defensive: skip silently if the declared WT residue does not
        # match the reference protein at this position (rather than
        # raising, which would propagate to the event bus).
        if wt_protein[mut_position_1based - 1] != wt_aa:
            return

        peptides: List[Peptide] = generate_neoantigen_peptides(
            wild_type_protein=wt_protein,
            mutant_position_1based=mut_position_1based,
            wild_type_aa=wt_aa,
            mutant_aa=mut_aa,
            source_gene=gene,
            mutation_label=mutation_label or aa_change,
        )
        if not peptides:
            return

        self.total_peptides_generated += len(peptides)
        self.emit_event(EventTypes.PEPTIDE_GENERATED, {
            'cell_id': cell_id,
            'gene': gene,
            'mutation': mutation_label,
            'aa_change': aa_change,
            'n_peptides': len(peptides),
            'peptide_sequences': [p.sequence for p in peptides],
        })

        # Score against the patient HLA panel; drop non-binders. Keep up
        # to max_displayed_per_mutation strongest binders (by IC50) for
        # downstream TCR matching.
        presentations = self._mhc_loader.score_all(peptides, self.hla_alleles)
        if not presentations:
            # Peptides were generated and none of them binds this patient's
            # panel, which is a real and interesting outcome -- it is how an
            # HLA-mismatched mutation stays invisible to T cells -- but
            # silently returning makes it indistinguishable from a broken
            # scorer. Say which scorer ran and how close the best peptide
            # came. Once per (gene, mutation): the same mutation is
            # re-scored on every cell on every step, and the second pass is
            # free because score_all has already cached every pair.
            key = (gene, mutation_label)
            if key not in self._unpresented_seen:
                self._unpresented_seen.add(key)
                scored = self._mhc_loader.score_all(
                    peptides, self.hla_alleles, include_non_binders=True)
                best = min(scored, key=lambda p: p.ic50_nm, default=None)
                # Kept on the module, not only logged: which of a patient's
                # mutations produced nothing presentable, and how close the
                # best peptide came, is a result worth reading -- it is how
                # an HLA-mismatched mutation stays invisible to T cells --
                # and a caller that wants it should not have to scrape logs.
                self.unpresented_mutations.append({
                    "gene": gene,
                    "mutation": mutation_label,
                    "n_peptides": len(peptides),
                    "hla_alleles": list(self.hla_alleles),
                    "weak_threshold_nm": self._mhc_loader.weak_threshold_nm,
                    "best_peptide": best.peptide.sequence if best else None,
                    "best_allele": best.hla_allele if best else None,
                    "best_ic50_nm": best.ic50_nm if best else None,
                    "scorer": getattr(
                        self._mhc_loader._scorer, "_binding_method", "unknown"),
                })
                logger.info(
                    "%s %s: %d peptides, none binding %s (weak-binder "
                    "threshold %.0f nM). Best was %s on %s at %.0f nM. "
                    "Scorer: %s.",
                    gene, mutation_label, len(peptides),
                    "/".join(self.hla_alleles),
                    self._mhc_loader.weak_threshold_nm,
                    best.peptide.sequence if best else "-",
                    best.hla_allele if best else "-",
                    best.ic50_nm if best else float("nan"),
                    getattr(self._mhc_loader._scorer, "_binding_method", "unknown"),
                )
            return
        cell = self.cells[cell_id]
        # A cell that has lost presentation cannot display a mutation on
        # the machinery it no longer has. Without this, an escaped cell
        # acquiring a new mutation -- or a founder seeded with HLA LOH
        # before the patient's mutations are presented -- would display
        # it on the lost allele and become recognisable again, which
        # would quietly undo the escape event. Filtered BEFORE the
        # top-N truncation, so the cell keeps its best binders among the
        # alleles it still has rather than the panel's best overall.
        presentations = self._presentable_on(cell, presentations)
        presentations.sort(key=lambda p: p.ic50_nm)
        kept = presentations[:self.max_displayed_per_mutation]

        cell.mhc1_displayed_peptides.extend(kept)
        self.total_peptides_presented += len(kept)

        for pres in kept:
            self.emit_event(EventTypes.PEPTIDE_PRESENTED, {
                'cell_id': cell_id,
                'gene': gene,
                'mutation': mutation_label,
                'peptide': pres.peptide.sequence,
                'hla_allele': pres.hla_allele,
                'ic50_nm': pres.ic50_nm,
                'binding_level': pres.binding_level,
                'presentation_score': pres.presentation_score,
            })


# Test
if __name__ == '__main__':
    from cognisom.core import SimulationEngine, SimulationConfig
    
    print("=" * 70)
    print("Cellular Module Test")
    print("=" * 70)
    print()
    
    # Create engine
    engine = SimulationEngine(SimulationConfig(dt=0.1, duration=2.0))
    
    # Register cellular module
    engine.register_module('cellular', CellularModule, {
        'n_normal_cells': 10,
        'n_cancer_cells': 2,
        'division_time_normal': 1.0,  # Fast for testing
        'division_time_cancer': 0.5
    })
    
    # Initialize and run
    engine.initialize()
    engine.run()
    
    # Get results
    state = engine.modules['cellular'].get_state()
    print("\nCellular State:")
    for key, value in state.items():
        print(f"  {key}: {value}")
    print()
    
    print("=" * 70)
    print("✓ Cellular module working!")
    print("=" * 70)
