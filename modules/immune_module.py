#!/usr/bin/env python3
"""
Immune Module
=============

Handles immune system: T cells, NK cells, macrophages.

Features:
- T cells (CD8+ cytotoxic) - MHC-I recognition
- NK cells (natural killer) - Missing-self detection
- Macrophages - Phagocytosis and polarization
- Immune surveillance and patrol
- Cancer recognition and killing
- Chemotaxis (follow gradients)
"""

import sys
sys.path.insert(0, '..')

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from core.module_base import SimulationModule
from core.event_bus import EventTypes
from engine.py.immune.mhc_loading import MHCPresentation
from engine.py.immune.tcr_repertoire import TCRMatch, TCRRepertoire
from engine.py.immune.tcell_kill import kill_outcome
from engine.py.immune.tme_classifier import (
    TMEClassification,
    classify_tme as _classify_tme,
)
from engine.py.spatial.ecm_barrier import (
    detection_attenuation,
    motility_attenuation,
)


@dataclass
class ImmuneCell:
    """State of an immune cell"""
    cell_id: int
    position: np.ndarray
    cell_type: str  # 'T_cell', 'NK_cell', 'macrophage'
    velocity: np.ndarray = None
    
    # State
    activated: bool = False
    target_cell_id: int = None
    in_blood: bool = False

    # TCR-pMHC match that activated this T cell (None for NK/macrophage
    # or for not-yet-activated T cells). Carried so _kill_target can
    # parametrize kill probability with the actual affinity that drove
    # recognition.
    active_tcr_match: Optional[TCRMatch] = None

    # Parameters
    speed: float = 10.0  # μm/min
    detection_radius: float = 10.0  # μm
    kill_radius: float = 5.0  # μm

    def __post_init__(self):
        if self.velocity is None:
            self.velocity = np.zeros(3)


class ImmuneModule(SimulationModule):
    """
    Immune system simulation module
    
    Manages:
    - T cells (CD8+ cytotoxic)
    - NK cells (natural killer)
    - Macrophages
    - Immune surveillance
    - Cancer killing
    
    Events Emitted:
    - IMMUNE_ACTIVATED: When immune cell activates
    - CANCER_KILLED: When cancer cell is killed
    - IMMUNE_RECRUITED: When new immune cell arrives
    
    Events Subscribed:
    - CELL_TRANSFORMED: Respond to new cancer
    - CELL_DIVIDED: Track cell population
    - EXOSOME_RELEASED: Detect danger signals
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)

        # Immune cell population
        self.immune_cells: Dict[int, ImmuneCell] = {}
        self.next_immune_id = 0

        # Parameters
        self.n_t_cells = config.get('n_t_cells', 15)
        self.n_nk_cells = config.get('n_nk_cells', 10)
        self.n_macrophages = config.get('n_macrophages', 8)
        self.patrol_speed = config.get('patrol_speed', 5.0)  # μm/min
        # Baseline kill probability used by NK / macrophage and as a
        # fallback when a T cell engages a cancer cell that has no
        # MHC-I-displayed neoantigens (e.g., immune-escape variants).
        self.kill_probability = config.get('kill_probability', 0.8)

        # TCR-pMHC matching for T cells (Upgrade 2). The repertoire is
        # patient-specific; seed is stable so the closed-loop test can
        # reproduce a recognized match.
        self.tcr_repertoire_size = int(config.get('tcr_repertoire_size', 1000))
        self.tcr_recognition_threshold = float(
            config.get('tcr_recognition_threshold', 0.7)
        )
        self.tcr_seed = int(config.get('tcr_seed', 0))
        # Costimulation strength on the tumor side. Real tumors
        # downregulate CD80/CD86; we leave this exposed for sensitivity
        # studies (immunotherapy ON / OFF).
        self.costimulation = float(config.get('costimulation', 0.6))
        self.checkpoint_block = float(config.get('checkpoint_block', 0.0))
        self._tcr_repertoire: Optional[TCRRepertoire] = None

        # Statistics
        self.total_kills = 0
        self.total_tcell_kills = 0
        self.total_activations = 0
        self.total_recruited = 0
        self.total_tcr_recognitions = 0
        # Cumulative count of clones that have transitioned from PD-1-lo
        # precursor to PD-1-hi exhausted during this run.
        self.total_exhaustion_transitions = 0

        # Reference to cellular module (for target cells)
        self.cellular_module = None
    
    def initialize(self):
        """Initialize immune system"""
        print("  Creating immune cell population...")
        
        # Create T cells (patrol tissue)
        for i in range(self.n_t_cells):
            x = np.random.uniform(50, 150)
            y = np.random.uniform(50, 150)
            z = np.random.uniform(40, 60)
            
            self.add_immune_cell(
                position=[x, y, z],
                cell_type='T_cell'
            )
        
        # Create NK cells
        for i in range(self.n_nk_cells):
            x = np.random.uniform(50, 150)
            y = np.random.uniform(50, 150)
            z = np.random.uniform(40, 60)
            
            self.add_immune_cell(
                position=[x, y, z],
                cell_type='NK_cell'
            )
        
        # Create macrophages
        for i in range(self.n_macrophages):
            x = np.random.uniform(50, 150)
            y = np.random.uniform(50, 150)
            z = np.random.uniform(40, 60)
            
            self.add_immune_cell(
                position=[x, y, z],
                cell_type='macrophage'
            )
        
        # Build the patient TCR repertoire (Upgrade 2). Same scorer the
        # closed-loop test relies on; reproducible from tcr_seed.
        self._tcr_repertoire = TCRRepertoire(
            size=self.tcr_repertoire_size,
            recognition_threshold=self.tcr_recognition_threshold,
            seed=self.tcr_seed,
        )

        # Subscribe to events
        self.subscribe(EventTypes.CELL_TRANSFORMED, self.on_cell_transformed)
        self.subscribe(EventTypes.CELL_DIVIDED, self.on_cell_divided)

        print(f"    ✓ {self.n_t_cells} T cells")
        print(f"    ✓ {self.n_nk_cells} NK cells")
        print(f"    ✓ {self.n_macrophages} macrophages")
        print(f"    ✓ TCR repertoire: {self.tcr_repertoire_size} clones "
              f"(threshold={self.tcr_recognition_threshold})")
    
    def add_immune_cell(self, position, cell_type):
        """Add immune cell"""
        cell_id = self.next_immune_id
        self.next_immune_id += 1
        
        cell = ImmuneCell(
            cell_id=cell_id,
            position=np.array(position, dtype=np.float32),
            cell_type=cell_type
        )
        
        self.immune_cells[cell_id] = cell
        return cell_id
    
    def set_cellular_module(self, cellular_module):
        """Link to cellular module for target access"""
        self.cellular_module = cellular_module

    def classify_tme(self, **kwargs) -> TMEClassification:
        """Classify the current tumor microenvironment into Teng's
        4-type scheme (Cancer Res 2015) and emit a TME_CLASSIFIED
        event with the result.

        Reads cancer cells from the linked cellular module and immune
        cells from this module's population. Keyword args are forwarded
        to ``engine.py.immune.tme_classifier.classify_tme`` so callers
        can override TIL / PD-L1 thresholds at the call site.

        Returns the TMEClassification dataclass; also emits a
        TME_CLASSIFIED event whose data dict contains the resolved
        ``tme_type`` (as a string), TIL / PD-L1 counts, and the
        clinical-readout fields ``predicted_icb_response`` and
        ``description``.
        """
        if self.cellular_module is None:
            raise RuntimeError(
                "ImmuneModule.classify_tme requires a linked cellular "
                "module; call set_cellular_module() first."
            )
        cancer_cells = [
            c for c in self.cellular_module.cells.values()
            if c.cell_type == 'cancer' and c.alive
        ]
        immune_cells = list(self.immune_cells.values())
        result = _classify_tme(cancer_cells, immune_cells, **kwargs)

        self.emit_event(EventTypes.TME_CLASSIFIED, {
            'tme_type': result.tme_type.value,
            'n_cancer_cells': result.n_cancer_cells,
            'n_til': result.n_til,
            'til_ratio': result.til_ratio,
            'pdl1_positive_fraction': result.pdl1_positive_fraction,
            'mean_ecm_density': result.mean_ecm_density,
            'ecm_excluded': result.ecm_excluded,
            'predicted_icb_response': result.predicted_icb_response,
            'description': result.description,
        })
        return result

    def update(self, dt: float):
        """Update immune system"""
        if not self.cellular_module:
            return
        
        # Get cancer cells from cellular module
        cancer_cells = {cid: cell for cid, cell in self.cellular_module.cells.items()
                       if cell.cell_type == 'cancer' and cell.alive}
        
        for immune_id, immune_cell in list(self.immune_cells.items()):
            if immune_cell.in_blood:
                continue
            
            # Patrol (random walk)
            if not immune_cell.activated:
                self._patrol(immune_cell, dt)

                # ECM barrier (Upgrade 6): the stromal density at the
                # T cell's position compresses its effective sensing
                # radius. In a fully fibrotic region a T cell may have
                # functionally zero detection range, even though
                # antigens are present nearby.
                local_ecm = self.cellular_module.ecm_density_at(
                    immune_cell.position
                )
                effective_detection = detection_attenuation(
                    immune_cell.detection_radius, local_ecm,
                )

                # Look for cancer cells
                for cancer_id, cancer_cell in cancer_cells.items():
                    distance = np.linalg.norm(immune_cell.position - cancer_cell.position)

                    if distance < effective_detection:
                        recognized, tcr_match = self._recognize(
                            immune_cell, cancer_cell
                        )
                        if recognized:
                            immune_cell.activated = True
                            immune_cell.target_cell_id = cancer_id
                            immune_cell.active_tcr_match = tcr_match
                            self.total_activations += 1
                            if tcr_match is not None:
                                self.total_tcr_recognitions += 1
                                # Adaptive PD-L1 induction (Type I TME
                                # mechanism): an activated TCR-pMHC
                                # engagement releases IFN-gamma, which
                                # paramedically up-regulates PD-L1 on
                                # the target tumor cell. Bumped per
                                # engagement, clamped to 1.0.
                                cancer_cell.pdl1_expression = float(min(
                                    1.0,
                                    cancer_cell.pdl1_expression + 0.5,
                                ))
                                # Chronic-antigen exhaustion bookkeeping
                                # (Dolina 2021, lecture slide 52): every
                                # recognized engagement increments the
                                # clone's encounter counter. When the
                                # exhaustion threshold is crossed, the
                                # clone transitions from PD-1-lo
                                # precursor to PD-1-hi exhausted and we
                                # fire a TCELL_EXHAUSTED event so
                                # downstream consumers (analytics, ICB
                                # response models) can see it.
                                new_count, did_exhaust = (
                                    self._tcr_repertoire.register_engagement(
                                        tcr_match.tcr.tcr_id
                                    )
                                )
                                if did_exhaust:
                                    self.total_exhaustion_transitions += 1
                                    self.emit_event(
                                        EventTypes.TCELL_EXHAUSTED,
                                        {
                                            'tcr_id': tcr_match.tcr.tcr_id,
                                            'cdr3': tcr_match.tcr.cdr3,
                                            'encounter_count': new_count,
                                            'target_id': cancer_id,
                                            'peptide': (
                                                tcr_match.presentation
                                                .peptide.sequence
                                            ),
                                            'mutation': (
                                                tcr_match.presentation
                                                .peptide.mutation_label
                                            ),
                                        },
                                    )

                            self.emit_event(EventTypes.IMMUNE_ACTIVATED, {
                                'immune_id': immune_id,
                                'immune_type': immune_cell.cell_type,
                                'target_id': cancer_id,
                                'position': immune_cell.position.tolist(),
                                'tcr_id': (
                                    tcr_match.tcr.tcr_id if tcr_match else None
                                ),
                                'affinity': (
                                    tcr_match.affinity if tcr_match else None
                                ),
                            })
                            break
            
            # Attack target
            else:
                if immune_cell.target_cell_id in cancer_cells:
                    target = cancer_cells[immune_cell.target_cell_id]

                    # Move toward target
                    direction = target.position - immune_cell.position
                    distance = np.linalg.norm(direction)

                    if distance > 0:
                        # ECM barrier (Upgrade 6) attenuates the T
                        # cell's speed at its current position. The
                        # stromal density on the path matters most;
                        # we sample at the T cell, which is the
                        # cheapest first-order approximation.
                        local_ecm = (
                            self.cellular_module.ecm_density_at(
                                immune_cell.position
                            )
                        )
                        effective_speed = motility_attenuation(
                            immune_cell.speed, local_ecm,
                        )
                        direction = direction / distance
                        immune_cell.velocity = direction * effective_speed
                        immune_cell.position += immune_cell.velocity * dt * 0.01  # Convert min to hours
                    
                    # Kill if close enough. T cells with a TCR match use
                    # the closed-loop kill_probability(affinity, mhc,
                    # costim); NK / macrophage retain the baseline flat
                    # probability.
                    if distance < immune_cell.kill_radius:
                        # Continuous-contact engagement: every step a
                        # T cell sits on its target, the clone
                        # accumulates engagement signal (real-biology
                        # chronic-antigen exposure). This is what
                        # drives precursor -> exhausted transition
                        # over time even when the tumor doesn't die.
                        if (
                            immune_cell.cell_type == 'T_cell'
                            and immune_cell.active_tcr_match is not None
                        ):
                            tcr_id = (
                                immune_cell.active_tcr_match.tcr.tcr_id
                            )
                            new_count, did_exhaust = (
                                self._tcr_repertoire.register_engagement(
                                    tcr_id
                                )
                            )
                            if did_exhaust:
                                self.total_exhaustion_transitions += 1
                                pres = (
                                    immune_cell.active_tcr_match.presentation
                                )
                                self.emit_event(
                                    EventTypes.TCELL_EXHAUSTED,
                                    {
                                        'tcr_id': tcr_id,
                                        'cdr3': (
                                            immune_cell.active_tcr_match
                                            .tcr.cdr3
                                        ),
                                        'encounter_count': new_count,
                                        'target_id': (
                                            immune_cell.target_cell_id
                                        ),
                                        'peptide': pres.peptide.sequence,
                                        'mutation': (
                                            pres.peptide.mutation_label
                                        ),
                                    },
                                )

                        p_kill = self._target_kill_probability(
                            immune_cell, target
                        )
                        if np.random.random() < p_kill:
                            self._kill_target(immune_cell, target)
                else:
                    # Target lost, deactivate
                    immune_cell.activated = False
                    immune_cell.target_cell_id = None
                    immune_cell.active_tcr_match = None
            
            # Keep in bounds
            immune_cell.position = np.clip(immune_cell.position, [20, 20, 20], [180, 180, 80])
    
    def _patrol(self, immune_cell: ImmuneCell, dt: float):
        """Random patrol movement, ECM-attenuated."""
        # Random walk
        direction = np.random.randn(3)
        direction = direction / (np.linalg.norm(direction) + 1e-6)

        # ECM barrier (Upgrade 6): patrol speed scales down with the
        # local stromal density. A T cell wandering into a fibrotic
        # region effectively stalls.
        effective_speed = self.patrol_speed
        if self.cellular_module is not None:
            local_ecm = self.cellular_module.ecm_density_at(
                immune_cell.position
            )
            effective_speed = motility_attenuation(
                self.patrol_speed, local_ecm,
            )
        immune_cell.velocity = direction * effective_speed
        immune_cell.position += immune_cell.velocity * dt * 0.01  # Convert min to hours
    
    def _recognize(self, immune_cell: ImmuneCell, cancer_cell):
        """Decide whether this immune cell recognizes this cancer cell.

        Returns (recognized: bool, tcr_match: Optional[TCRMatch]).
        TCRMatch is populated only for T-cell recognitions; NK and
        macrophage paths keep the legacy threshold heuristic.
        """
        if immune_cell.cell_type == 'T_cell':
            # Upgrade 2: TCR-pMHC matching. The cancer cell must have at
            # least one MHC-I-displayed neoantigen, AND a clone in the
            # patient TCR repertoire must clear the recognition
            # threshold against one of those presentations.
            presentations: List[MHCPresentation] = getattr(
                cancer_cell, 'mhc1_displayed_peptides', []
            )
            if not presentations or self._tcr_repertoire is None:
                return False, None
            # MHC-I downregulation gates display visibility too: a
            # cancer cell with mhc1_expression near 0 hides its
            # neoantigens. We require the displayed pool to be non-
            # empty *and* the surface expression to be above a small
            # epsilon -- the patent claim is the closed loop, not the
            # surface gate, so the threshold is intentionally lenient.
            if cancer_cell.mhc1_expression < 0.05:
                return False, None
            for pres in presentations:
                match = self._tcr_repertoire.best_match(pres)
                if match is not None and match.is_recognized:
                    return True, match
            return False, None

        elif immune_cell.cell_type == 'NK_cell':
            # NK cells detect MISSING MHC-I (missing-self).
            return cancer_cell.mhc1_expression < 0.4, None

        elif immune_cell.cell_type == 'macrophage':
            return True, None

        return False, None

    def _target_kill_probability(self, immune_cell: ImmuneCell, target_cell) -> float:
        """Per-encounter kill probability.

        T cells with an active TCR match use the affinity * MHC * costim
        Hill rule from tcell_kill.kill_probability; NK and macrophage
        retain the legacy flat self.kill_probability.
        """
        if (immune_cell.cell_type == 'T_cell'
                and immune_cell.active_tcr_match is not None):
            match = immune_cell.active_tcr_match
            outcome = kill_outcome(
                affinity=match.affinity,
                mhc_level=getattr(target_cell, 'mhc1_expression', 1.0),
                costimulation=self.costimulation,
                checkpoint_block=self.checkpoint_block,
                # Lecture slide 52: exhausted clones cannot be rescued
                # by checkpoint blockade. Pass through the match's
                # exhaustion state so kill_outcome gates the rescue
                # term and applies the exhaustion multiplier.
                is_exhausted=match.is_exhausted,
            )
            return outcome.kill_probability
        return self.kill_probability

    def _kill_target(self, immune_cell: ImmuneCell, target_cell):
        """Kill target cancer cell.

        Always emits CANCER_KILLED (backwards-compat with everything
        subscribed to it). T-cell kills additionally emit
        CELL_KILLED_BY_TCELL so the closed-loop event trace
        MUTATION_OCCURRED -> PEPTIDE_GENERATED -> PEPTIDE_PRESENTED ->
        CELL_KILLED_BY_TCELL is recoverable from the event log.
        """
        self.emit_event(EventTypes.CANCER_KILLED, {
            'cell_id': target_cell.cell_id,
            'killer_id': immune_cell.cell_id,
            'killer_type': immune_cell.cell_type,
            'position': target_cell.position.tolist()
        })
        self.total_kills += 1

        if (immune_cell.cell_type == 'T_cell'
                and immune_cell.active_tcr_match is not None):
            match = immune_cell.active_tcr_match
            self.total_tcell_kills += 1
            self.emit_event(EventTypes.CELL_KILLED_BY_TCELL, {
                'cell_id': target_cell.cell_id,
                'killer_id': immune_cell.cell_id,
                'tcr_id': match.tcr.tcr_id,
                'peptide': match.presentation.peptide.sequence,
                'hla_allele': match.presentation.hla_allele,
                'mutation': match.presentation.peptide.mutation_label,
                'source_gene': match.presentation.peptide.source_gene,
                'affinity': match.affinity,
                'position': target_cell.position.tolist(),
            })

        # Deactivate immune cell
        immune_cell.activated = False
        immune_cell.target_cell_id = None
        immune_cell.active_tcr_match = None
    
    def get_state(self) -> Dict[str, Any]:
        """Return current immune state"""
        active_immune = [ic for ic in self.immune_cells.values() if not ic.in_blood]
        activated_immune = [ic for ic in active_immune if ic.activated]
        
        t_cells = [ic for ic in active_immune if ic.cell_type == 'T_cell']
        nk_cells = [ic for ic in active_immune if ic.cell_type == 'NK_cell']
        macrophages = [ic for ic in active_immune if ic.cell_type == 'macrophage']
        
        return {
            'n_immune_cells': len(active_immune),
            'n_activated': len(activated_immune),
            'n_t_cells': len(t_cells),
            'n_nk_cells': len(nk_cells),
            'n_macrophages': len(macrophages),
            'total_kills': self.total_kills,
            'total_activations': self.total_activations,
            'total_recruited': self.total_recruited
        }
    
    # Event handlers
    def on_cell_transformed(self, data):
        """Handle cell transformation - recruit more immune cells"""
        # Recruit additional immune cell
        position = data['position']
        
        # Add new T cell near transformation site
        offset = np.random.randn(3) * 20
        new_pos = np.array(position) + offset
        
        self.add_immune_cell(new_pos, 'T_cell')
        self.total_recruited += 1
        
        self.emit_event(EventTypes.IMMUNE_RECRUITED, {
            'cell_type': 'T_cell',
            'position': new_pos.tolist(),
            'reason': 'cell_transformation'
        })
    
    def on_cell_divided(self, data):
        """Handle cell division - track population"""
        # Could recruit more immune cells if cancer is dividing rapidly
        if data['cell_type'] == 'cancer':
            if np.random.random() < 0.1:  # 10% chance
                position = data['position']
                offset = np.random.randn(3) * 20
                new_pos = np.array(position) + offset
                
                self.add_immune_cell(new_pos, 'NK_cell')
                self.total_recruited += 1


# Test
if __name__ == '__main__':
    from core import SimulationEngine, SimulationConfig
    from cellular_module import CellularModule
    
    print("=" * 70)
    print("Immune Module Test")
    print("=" * 70)
    print()
    
    # Create engine
    engine = SimulationEngine(SimulationConfig(dt=0.01, duration=1.0))
    
    # Register modules
    engine.register_module('cellular', CellularModule, {
        'n_normal_cells': 10,
        'n_cancer_cells': 5,
        'division_time_cancer': 10.0  # Slow for testing
    })
    
    engine.register_module('immune', ImmuneModule, {
        'n_t_cells': 5,
        'n_nk_cells': 3,
        'n_macrophages': 2
    })
    
    # Initialize
    engine.initialize()
    
    # Link modules
    immune = engine.modules['immune']
    cellular = engine.modules['cellular']
    immune.set_cellular_module(cellular)
    
    print("Modules linked")
    print()
    
    # Run simulation
    engine.run()
    
    # Get results
    print("\nImmune State:")
    immune_state = immune.get_state()
    for key, value in immune_state.items():
        print(f"  {key}: {value}")
    
    print("\nCellular State:")
    cellular_state = cellular.get_state()
    for key, value in cellular_state.items():
        print(f"  {key}: {value}")
    
    print()
    print("=" * 70)
    print("✓ Immune module working!")
    print("=" * 70)
