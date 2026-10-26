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

import sys
sys.path.insert(0, '..')

import numpy as np
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from cognisom.core.module_base import SimulationModule
from cognisom.core.event_bus import EventTypes
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
        self.division_time_normal = self.config.get('division_time_normal', 24.0)  # hours
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
        self.glucose_consumption_normal = self.config.get('glucose_consumption', 0.2)
        self.glucose_consumption_cancer = self.config.get('glucose_consumption_cancer', 0.5)

        # ECM / desmoplasia dynamics (Upgrade 6). Cancer cells deposit
        # local stromal ECM proportional to ``ecm_deposition_rate``
        # (units: per-hour fraction). Anti-fibrotic therapy degrades
        # ECM at ``ecm_degradation_rate`` when ``anti_fibrotic_active``
        # is True. Defaults model a moderately fibrotic tumor; PDAC-
        # like configs can set the rate ~5-10x higher.
        self.ecm_deposition_rate = float(
            self.config.get('ecm_deposition_rate', 0.05)
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
        self.seeding = str(self.config.get('seeding', 'ring'))
        self.slab_overrides = dict(self.config.get('slab', {}) or {})
        #: Populated by _seed_slab with the geometry it used (extents,
        #: bounds, nodule centre and radius), for callers that need to
        #: frame a camera or check perfusion against it.
        self.slab_layout = None

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
        else:
            self._seed_ring(n_normal, n_cancer)

        # Initialize MHC loader (Upgrade 2). Construction reaches into
        # NeoantigenPredictor which instantiates a GeneProteinMapper;
        # this preseeds from BUILTIN_PROTEINS and does not touch the
        # network. Doing it here keeps the per-cell update path cheap.
        self._mhc_loader = MHCLoader()

        if self.initial_cell_cycle == 'asynchronous':
            self._stagger_initial_ages()

        # Subscribe to events
        self.subscribe(EventTypes.EXOSOME_UPTAKEN, self.on_exosome_uptaken)
        self.subscribe(EventTypes.CANCER_KILLED, self.on_cancer_killed)
        self.subscribe(EventTypes.MUTATION_OCCURRED, self.on_mutation_occurred)

        print(f"    ✓ {n_normal} normal cells")
        print(f"    ✓ {n_cancer} cancer cells")
        print(f"    ✓ MHC-I display: {len(self.hla_alleles)} HLA alleles "
              f"({', '.join(self.hla_alleles)})")
    
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

    def _founding_mutations(self) -> List[str]:
        """The mutation labels a founding cancer cell carries.

        The patient's, when one was given; otherwise the historical
        KRAS_G12D every run used to carry whoever the patient was.
        """
        if self.patient_given:
            return [m['label'] for m in self.patient_mutations]
        return ['KRAS_G12D']

    def _present_patient_mutations_on(self, cell_id: int) -> None:
        """Run the patient's mutations through presentation on one cell."""
        for mutation in self.patient_mutations:
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
        presentations.sort(key=lambda p: p.ic50_nm)
        kept = presentations[:self.max_displayed_per_mutation]

        cell = self.cells[cell_id]
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
