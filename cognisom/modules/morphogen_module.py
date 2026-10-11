#!/usr/bin/env python3
"""
Morphogen Module
================

Handles morphogen gradients and positional information.

Features:
- Morphogen sources (organizers)
- Gradient formation (diffusion + degradation)
- Positional sensing by cells
- Cell fate determination
- Boundary formation
- Pattern formation
"""

import sys
sys.path.insert(0, '..')

import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

from cognisom.core.module_base import SimulationModule
from cognisom.core.event_bus import EventTypes


@dataclass
class MorphogenGradient:
    """Morphogen gradient field"""
    name: str
    source_position: np.ndarray
    source_strength: float = 1.0
    diffusion_coeff: float = 100.0  # μm²/s
    degradation_rate: float = 0.1   # 1/hour
    
    # Concentration field (will be set by spatial module)
    concentration: np.ndarray = None


@dataclass
class PositionalIdentity:
    """Cell's positional identity based on morphogens"""
    cell_id: int
    position: np.ndarray
    
    # Morphogen concentrations sensed
    morphogen_levels: Dict[str, float] = None
    
    # Positional coordinates (French flag model)
    anterior_posterior: float = 0.5  # 0=anterior, 1=posterior
    dorsal_ventral: float = 0.5      # 0=dorsal, 1=ventral
    proximal_distal: float = 0.5     # 0=proximal, 1=distal
    
    # Cell fate determined by position
    cell_fate: str = "undetermined"
    
    def __post_init__(self):
        if self.morphogen_levels is None:
            self.morphogen_levels = {}
    
    def determine_fate(self):
        """Determine cell fate based on morphogen levels"""
        # Example: French flag model
        # High morphogen = posterior fate
        # Medium = middle fate
        # Low = anterior fate
        
        if 'BMP' in self.morphogen_levels:
            bmp = self.morphogen_levels['BMP']
            if bmp > 0.7:
                self.cell_fate = "posterior"
            elif bmp > 0.3:
                self.cell_fate = "middle"
            else:
                self.cell_fate = "anterior"


class MorphogenModule(SimulationModule):
    """
    Morphogen gradient simulation module
    
    Manages:
    - Morphogen sources (organizers)
    - Gradient formation and maintenance
    - Positional sensing by cells
    - Cell fate determination
    - Pattern formation
    
    Events Emitted:
    - FATE_DETERMINED: When cell fate is specified
    - BOUNDARY_FORMED: When sharp boundary detected
    
    Events Subscribed:
    - CELL_DIVIDED: Inherit positional identity
    - CELL_DIED: Remove identity
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        # Morphogen gradients
        self.gradients: Dict[str, MorphogenGradient] = {}
        
        # Cell positional identities
        self.cell_identities: Dict[int, PositionalIdentity] = {}
        
        # Parameters
        self.tissue_size = self.config.get('tissue_size', (200, 200, 100))
        self.enable_fate_determination = self.config.get('enable_fate_determination', True)
        
        # Statistics
        self.total_fates_determined = 0
        self.fate_counts = {}
        
        # References
        self.spatial_module = None
        self.cellular_module = None

        # Gradients already registered as a source on a spatial field
        self._registered_sources = set()
    
    def initialize(self):
        """Initialize morphogen system"""
        print("  Creating morphogen gradient system...")
        
        # Create classic morphogen gradients
        self._create_gradient('BMP', 
                            source_position=np.array([200, 100, 50]),
                            source_strength=1.0)
        
        self._create_gradient('Shh',
                            source_position=np.array([100, 0, 50]),
                            source_strength=0.8)
        
        self._create_gradient('Wnt',
                            source_position=np.array([0, 100, 50]),
                            source_strength=0.9)
        
        # Subscribe to events
        self.subscribe(EventTypes.CELL_DIVIDED, self.on_cell_divided)
        self.subscribe(EventTypes.CELL_DIED, self.on_cell_died)
        
        print(f"    ✓ {len(self.gradients)} morphogen gradients")
        print(f"    ✓ Positional sensing system")
        print(f"    ✓ Fate determination ready")
    
    def _create_gradient(self, name: str, source_position: np.ndarray, 
                        source_strength: float):
        """Create morphogen gradient"""
        self.gradients[name] = MorphogenGradient(
            name=name,
            source_position=source_position,
            source_strength=source_strength
        )
    
    def add_cell(self, cell_id: int, position: np.ndarray):
        """Add cell and determine its positional identity"""
        identity = PositionalIdentity(
            cell_id=cell_id,
            # np.array() rather than .copy(): CellMechanicsModule rebinds
            # cell.position to a plain tuple, which has no .copy().
            position=np.array(position, dtype=float)
        )
        
        # Sense morphogen levels
        self._sense_morphogens(identity)
        
        # Determine positional coordinates
        self._determine_position(identity)
        
        # Determine fate
        if self.enable_fate_determination:
            self._specify_fate(identity)
            
            # Counting now lives in _specify_fate(), so the
            # re-specification that update() performs shares one tally.
        
        self.cell_identities[cell_id] = identity
    
    def remove_cell(self, cell_id: int):
        """Remove cell identity"""
        if cell_id in self.cell_identities:
            del self.cell_identities[cell_id]
    
    def set_spatial_module(self, spatial_module):
        """Link to spatial module"""
        self.spatial_module = spatial_module
        self._registered_sources = set()
    
    def set_cellular_module(self, cellular_module):
        """Link to cellular module"""
        self.cellular_module = cellular_module
    
    def update(self, dt: float):
        """Update morphogen system"""
        # Update gradients (if using spatial module)
        if self.spatial_module:
            self._update_gradients_spatial(dt)
        else:
            # Simple exponential decay model
            self._update_gradients_simple(dt)
        
        # Update cell sensing. identity.position is a copy taken at
        # add_cell() time, so unless it is re-read from the cellular
        # module every cell senses the gradient at its birthplace for the
        # whole run. Re-deriving the coordinates and the fate belongs to
        # the same step: re-sensing alone left anterior_posterior and
        # cell_fate frozen at their birth values even when the sensed
        # concentrations changed.
        for identity in self.cell_identities.values():
            self._refresh_position(identity)
            self._sense_morphogens(identity)
            self._determine_position(identity)
            self._specify_fate(identity)
    
    def _update_gradients_spatial(self, dt: float):
        """Update gradients using spatial module"""
        # Register each morphogen source with its spatial field ONCE.
        # SpatialField.add_source() appends, and SpatialField.update()
        # deposits rate*dt for every entry, so re-adding each step grew
        # the list without bound and multiplied the deposited morphogen
        # by the step count. An organizer is static, so one entry each is
        # correct. (This path is unreachable today: set_spatial_module()
        # has no callers, and SpatialModule creates only oxygen, glucose
        # and cytokine fields, so the name test never matches anyway.)
        for name, gradient in self.gradients.items():
            if name in self._registered_sources:
                continue
            if name.lower() in self.spatial_module.fields:
                field = self.spatial_module.fields[name.lower()]
                
                # Add source
                field.add_source(gradient.source_position, 
                               gradient.source_strength)
                self._registered_sources.add(name)
    
    def _update_gradients_simple(self, dt: float):
        """Nothing to integrate: the simple model is evaluated on demand.

        The analytic form C = C0 * exp(-distance / decay_length) is a
        steady state with no stored field, and _sense_morphogens()
        evaluates it directly at each cell's position, so there is no
        per-step gradient state to advance. Kept as the explicit
        counterpart to _update_gradients_spatial() rather than deleted,
        so the branch in update() stays symmetric.
        """
        return
    
    def _refresh_position(self, identity: PositionalIdentity):
        """Re-read the cell's current position from the cellular module.

        Without a cellular module there is nothing to refresh against, so
        the birth position is kept.
        """
        if self.cellular_module is None:
            return

        cell = getattr(self.cellular_module, 'cells', {}).get(identity.cell_id)
        if cell is None:
            return

        position = getattr(cell, 'position', None)
        if position is None:
            return

        # CellularModule stores an ndarray, but CellMechanicsModule
        # rebinds cell.position to a plain tuple when it writes positions
        # back, so normalise rather than assuming either.
        identity.position = np.array(position, dtype=float)

    def _specify_fate(self, identity: PositionalIdentity):
        """Determine (or re-determine) the cell's fate, and count it.

        Counts only a genuine change of fate, so the statistic stays a
        count of specification decisions rather than of simulation steps.
        """
        if not self.enable_fate_determination:
            return

        previous_fate = identity.cell_fate
        identity.determine_fate()

        if identity.cell_fate == previous_fate:
            return

        if identity.cell_fate not in self.fate_counts:
            self.fate_counts[identity.cell_fate] = 0
        self.fate_counts[identity.cell_fate] += 1
        self.total_fates_determined += 1

    def _sense_morphogens(self, identity: PositionalIdentity):
        """Cell senses morphogen concentrations at its position"""
        for name, gradient in self.gradients.items():
            # Calculate concentration based on distance from source
            distance = np.linalg.norm(identity.position - gradient.source_position)
            
            # Exponential decay
            decay_length = 50.0  # μm
            concentration = gradient.source_strength * np.exp(-distance / decay_length)
            
            identity.morphogen_levels[name] = concentration
    
    def _determine_position(self, identity: PositionalIdentity):
        """Determine positional coordinates from morphogen levels"""
        # Anterior-Posterior: BMP gradient
        if 'BMP' in identity.morphogen_levels:
            identity.anterior_posterior = identity.morphogen_levels['BMP']
        
        # Dorsal-Ventral: Shh gradient
        if 'Shh' in identity.morphogen_levels:
            identity.dorsal_ventral = identity.morphogen_levels['Shh']
        
        # Proximal-Distal: Wnt gradient
        if 'Wnt' in identity.morphogen_levels:
            identity.proximal_distal = identity.morphogen_levels['Wnt']
    
    def get_morphogen_level(self, cell_id: int, morphogen_name: str) -> float:
        """Get morphogen level sensed by cell"""
        if cell_id in self.cell_identities:
            return self.cell_identities[cell_id].morphogen_levels.get(morphogen_name, 0)
        return 0
    
    def get_cell_fate(self, cell_id: int) -> str:
        """Get cell fate"""
        if cell_id in self.cell_identities:
            return self.cell_identities[cell_id].cell_fate
        return "undetermined"
    
    def get_positional_coordinates(self, cell_id: int) -> Tuple[float, float, float]:
        """Get cell's positional coordinates"""
        if cell_id in self.cell_identities:
            identity = self.cell_identities[cell_id]
            return (identity.anterior_posterior,
                   identity.dorsal_ventral,
                   identity.proximal_distal)
        return (0.5, 0.5, 0.5)
    
    def get_state(self) -> Dict[str, Any]:
        """Return current morphogen state"""
        # Census of the cells alive now. self.fate_counts is a
        # cumulative tally that is never decremented, so publishing it as
        # the "distribution" reported the fates of a population that had
        # already died: 90 fates against n_cells_tracked = 0.
        census: Dict[str, int] = {}
        for identity in self.cell_identities.values():
            census[identity.cell_fate] = census.get(identity.cell_fate, 0) + 1

        return {
            'n_gradients': len(self.gradients),
            'n_cells_tracked': len(self.cell_identities),
            'total_fates_determined': self.total_fates_determined,
            'fate_distribution': census,
            'cumulative_fate_assignments': dict(self.fate_counts),
            'gradient_names': list(self.gradients.keys())
        }
    
    # Event handlers
    def on_cell_divided(self, data):
        """Handle cell division - inherit identity"""
        parent_id = data['cell_id']
        daughter_id = data.get('daughter_id')
        daughter_position = data.get('daughter_position')
        
        if daughter_id is None:
            return

        if daughter_position is None and self.cellular_module is not None:
            # CellularModule's CELL_DIVIDED payload carries daughter_id
            # but no daughter_position, so no daughter was ever tracked
            # and n_cells_tracked decayed to zero over any run with
            # turnover. Look the position up instead of giving up.
            cell = getattr(self.cellular_module, 'cells', {}).get(daughter_id)
            if cell is not None:
                daughter_position = getattr(cell, 'position', None)

        if daughter_position is not None:
            # Daughter gets new identity based on her position
            self.add_cell(daughter_id, np.array(daughter_position, dtype=float))
    
    def on_cell_died(self, data):
        """Handle cell death - remove identity"""
        cell_id = data['cell_id']
        self.remove_cell(cell_id)


# Test
if __name__ == '__main__':
    from cognisom.core import SimulationEngine, SimulationConfig
    from cellular_module import CellularModule
    
    print("=" * 70)
    print("Morphogen Module Test")
    print("=" * 70)
    print()
    
    # Create engine
    engine = SimulationEngine(SimulationConfig(dt=0.01, duration=1.0))
    
    # Register modules
    engine.register_module('cellular', CellularModule, {
        'n_normal_cells': 20,
        'n_cancer_cells': 0
    })
    
    engine.register_module('morphogen', MorphogenModule, {
        'enable_fate_determination': True
    })
    
    # Initialize
    engine.initialize()
    
    # Add cells to morphogen tracking
    morphogen = engine.modules['morphogen']
    cellular = engine.modules['cellular']
    
    for cell_id, cell in cellular.cells.items():
        morphogen.add_cell(cell_id, cell.position)
    
    print(f"Added {len(cellular.cells)} cells to morphogen tracking")
    print()
    
    # Check initial identities
    print("Cell positional identities:")
    for cell_id in list(cellular.cells.keys())[:5]:
        if cell_id in morphogen.cell_identities:
            identity = morphogen.cell_identities[cell_id]
            print(f"\nCell {cell_id} at {identity.position}:")
            print(f"  Morphogens: BMP={identity.morphogen_levels.get('BMP', 0):.3f}, "
                  f"Shh={identity.morphogen_levels.get('Shh', 0):.3f}, "
                  f"Wnt={identity.morphogen_levels.get('Wnt', 0):.3f}")
            print(f"  Position: AP={identity.anterior_posterior:.2f}, "
                  f"DV={identity.dorsal_ventral:.2f}, "
                  f"PD={identity.proximal_distal:.2f}")
            print(f"  Fate: {identity.cell_fate}")
    print()
    
    # Run simulation
    engine.run()
    
    # Get results
    print("\nMorphogen State:")
    morphogen_state = morphogen.get_state()
    for key, value in morphogen_state.items():
        print(f"  {key}: {value}")
    
    print()
    print("=" * 70)
    print("✓ Morphogen module working!")
    print("=" * 70)
