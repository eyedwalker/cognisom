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

#: (gene, mutation) pairs already reported as presenting nothing.
_NO_BINDER_REPORTED: set = set()


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


#: Death rate of a cancer cell below the hypoxia or starvation threshold,
#: per hour. ILLUSTRATIVE: it is the 0.01-per-step roll this replaced, read
#: at the module's design step of 0.01 h, so behaviour at that step is
#: unchanged and every other step now agrees with it. No measured
#: survival time under these thresholds was found to source it from.
HYPOXIC_CANCER_DEATH_PER_HOUR = 1.0


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

    # Cancer properties
    mhc1_expression: float = 1.0
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
        self.max_displayed_per_mutation = int(
            self.config.get('max_displayed_per_mutation', 5)
        )
        self._mhc_loader: Optional[MHCLoader] = None

        # Linked molecular module supplies WT protein sequences for
        # neoantigen peptide generation (set via set_molecular_module).
        self.molecular_module = None

        # Statistics
        self.total_divisions = 0
        self.total_deaths = 0
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
            self.cells[cell_id].mutations = ['KRAS_G12D']
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
            self.cells[cell_id].mutations = ['KRAS_G12D']
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
        
        for cell_id, cell in list(self.cells.items()):
            if not cell.alive:
                continue
            
            # Age
            cell.age += dt

            # Metabolism
            self._update_metabolism(cell, dt)

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
            
            if cell.age >= division_time:
                cells_to_divide.append(cell_id)
            
            # Check for death (hypoxia, starvation). A normal cell dies at
            # once; a cancer cell is more resistant and dies at a rate per
            # hour. That rate used to be a bare 0.01 per step, so a
            # starving cancer cell lived ~1 h at dt = 0.01 and ~5 h at
            # dt = 0.05.
            if cell.oxygen < 0.02 or cell.glucose < 0.5:
                if cell.cell_type == 'normal':
                    cells_to_die.append(cell_id)
                elif np.random.random() < -np.expm1(
                        -HYPOXIC_CANCER_DEATH_PER_HOUR * dt):
                    cells_to_die.append(cell_id)
        
        # Process divisions
        for cell_id in cells_to_divide:
            self._divide_cell(cell_id)
        
        # Process deaths
        for cell_id in cells_to_die:
            self._kill_cell(cell_id, cause='hypoxia')
    
    def _update_metabolism(self, cell: CellState, dt: float):
        """Update cell metabolism"""
        if cell.cell_type == 'cancer':
            # Warburg effect
            cell.glucose -= self.glucose_consumption_cancer * dt
            cell.oxygen -= 0.1 * dt
            cell.lactate += 0.3 * dt
            cell.atp += 50 * dt  # Less efficient
        else:
            # Normal metabolism
            cell.glucose -= self.glucose_consumption_normal * dt
            cell.oxygen -= 0.15 * dt
            cell.lactate += 0.1 * dt
            cell.atp += 100 * dt  # More efficient
        
        # Clamp values
        cell.glucose = max(0, cell.glucose)
        cell.oxygen = max(0, cell.oxygen)
        cell.lactate = max(0, cell.lactate)
        cell.atp = max(0, cell.atp)
    
    def _divide_cell(self, cell_id: int):
        """Cell division"""
        if cell_id not in self.cells:
            return
        
        parent = self.cells[cell_id]
        
        # Create daughter cell
        offset = np.random.randn(3) * 5  # Small offset
        daughter_pos = parent.position + offset
        
        daughter_id = self.add_cell(
            position=daughter_pos,
            cell_type=parent.cell_type
        )
        
        daughter = self.cells[daughter_id]
        daughter.mutations = parent.mutations.copy()
        daughter.mhc1_expression = parent.mhc1_expression
        # Oxygen and glucose are concentrations, so both cells carry the
        # parent's. A daughter used to start at the CellState defaults --
        # arterial O2 and plasma glucose -- so one born in a hypoxic corner
        # was briefly rescued from it, and when a whole lineage divided on
        # one step the tissue's mean O2 read as if half of it sat in an
        # artery. The tissue engine's daughters already inherit their state.
        daughter.oxygen = parent.oxygen
        daughter.glucose = parent.glucose
        
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
        else:
            cell.mutations.append('KRAS_G12D')
        cell.mhc1_expression = 0.3
        
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
            'total_transformations': self.total_transformations,
            'avg_oxygen': np.mean([c.oxygen for c in alive_cells]) if alive_cells else 0,
            'avg_glucose': np.mean([c.glucose for c in alive_cells]) if alive_cells else 0
        }
    
    def set_molecular_module(self, molecular_module):
        """Link to molecular module for reference protein lookups.

        Required for the MUTATION_OCCURRED -> PEPTIDE_GENERATED ->
        PEPTIDE_PRESENTED chain (Upgrade 2): the molecular module owns
        the canonical WT protein sequence; the cellular module owns
        per-cell MHC-I display state.
        """
        self.molecular_module = molecular_module

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
            if key not in _NO_BINDER_REPORTED:
                _NO_BINDER_REPORTED.add(key)
                scored = self._mhc_loader.score_all(
                    peptides, self.hla_alleles, include_non_binders=True)
                best = min(scored, key=lambda p: p.ic50_nm, default=None)
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
