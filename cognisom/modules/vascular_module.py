#!/usr/bin/env python3
"""
Vascular Module
===============

Handles blood vessel network and nutrient/waste exchange.

Features:
- Capillary network
- O2 delivery
- Glucose delivery
- Waste removal (CO2, lactate)
- Diffusion-based exchange
- Distance-dependent gradients
"""

import sys
sys.path.insert(0, '..')

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass

from cognisom.core.module_base import SimulationModule
from cognisom.core.event_bus import EventTypes


# Largest geometry factor krogh_weight will return. The exchange step is
# clamped to the driving gradient, so this only has to be big enough that
# permeability * cap * dt >= 1 for every dt the engine runs; at 0.79 /h
# and dt = 1e-4 h that needs > 1.3e4.
_WEIGHT_CAP = 1.0e6


def krogh_shape(r: float, r_cap: float, r_tissue: float) -> float:
    """Krogh-cylinder concentration-drop shape function, in μm^2:

        shape(r) = Rt^2 * ln(r / Rc) - (r^2 - Rc^2) / 2

    Zero at r = Rc and monotonically increasing on [Rc, Rt] (its
    derivative Rt^2/r - r vanishes only at r = Rt).
    """
    return r_tissue * r_tissue * np.log(r / r_cap) - 0.5 * (r * r - r_cap * r_cap)


def krogh_weight(distance: float, r_cap: float, r_tissue: float) -> float:
    """shape(Rt) / shape(distance): the dimensionless conductance weight,
    exactly 1.0 at the lethal corner and rising towards the vessel."""
    if not (r_tissue > r_cap > 0.0):
        return 0.0
    if distance <= r_cap:
        return _WEIGHT_CAP
    denom = krogh_shape(distance, r_cap, r_tissue)
    if denom <= 0.0:
        return _WEIGHT_CAP
    return float(min(krogh_shape(r_tissue, r_cap, r_tissue) / denom,
                     _WEIGHT_CAP))


@dataclass
class Capillary:
    """Blood capillary"""
    capillary_id: int
    start: np.ndarray
    end: np.ndarray
    
    # Blood contents
    oxygen: float = 0.21  # 21% O2 (arterial)
    glucose: float = 5.0  # mM
    
    # Flow
    flow_rate: float = 0.5  # mm/s
    diameter: float = 8.0  # μm
    
    # Exchange parameters.
    #
    # exchange_radius is the KROGH RADIUS: the radial distance at
    # which the most demanding cell type in the model reaches its own
    # death threshold at steady state (the classical lethal corner).
    # The inner boundary of that cylinder is the vessel wall, at
    # radius_um = diameter / 2, exposed as a property so that
    # `diameter` stays the single source of truth.
    #
    # The permeabilities are LUMPED CONDUCTANCES in 1/hour, not
    # diffusivities. Each is fixed outright, with no free parameter,
    # by the lethal-corner condition
    #     permeability = q_max / (C_arterial - C_death)
    # evaluated on this model's own numbers:
    #   O2       max(0.15, 0.10) / (0.21 - 0.02) = 0.789 /h
    #   glucose  max(0.20, 0.50) / (5.00 - 0.50) = 0.111 /h
    #   lactate  0.10 /h / 1.5 mM / weight(11.7um) = 0.034 /h
    # See VascularModule._exchange_with_tissue for the derivation.
    # NOTE: VascularModule.exchange_rate multiplies the conductance,
    # so it shrinks the effective lethal corner: the Krogh profile is
    # flat at r = Rt by construction, which makes that dial far more
    # sensitive near 1.0 than it looks (1.0 -> 50 um, 0.9 -> ~29 um,
    # 0.4 -> ~9 um).
    exchange_radius: float = 50.0            # μm (Krogh radius)
    permeability_O2: float = 0.7894737       # 1/h = 0.15 / 0.19
    permeability_glucose: float = 0.1111111  # 1/h = 0.50 / 4.50
    permeability_lactate: float = 0.0344828  # 1/h, MCT efflux
    
    def __post_init__(self):
        self.length = np.linalg.norm(self.end - self.start)

    @property
    def radius_um(self) -> float:
        """Vessel radius in μm: the inner boundary of the Krogh
        cylinder. Derived from ``diameter`` so the two cannot drift.
        """
        return 0.5 * self.diameter
    
    def get_position(self, t: float) -> np.ndarray:
        """Get position along capillary (t: 0-1)"""
        return self.start + t * (self.end - self.start)

    def distance_to(self, position: np.ndarray) -> float:
        """Perpendicular distance from ``position`` to this capillary,
        treated as a line SEGMENT (the projection is clamped to the
        endpoints). This is the radial coordinate of the Krogh cylinder,
        not the distance to an arbitrary sample point.
        """
        along = self.end - self.start
        along_sq = float(np.dot(along, along))
        if along_sq <= 0.0:
            return float(np.linalg.norm(position - self.start))
        s = float(np.dot(position - self.start, along) / along_sq)
        s = 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)
        return float(np.linalg.norm(position - (self.start + s * along)))

    def conductance_weight(self, distance: float) -> float:
        """Dimensionless Krogh geometry factor at radius ``distance``.

        The Krogh-cylinder steady state for uniform consumption M and
        Krogh diffusion coefficient K, with a no-flux boundary at the
        tissue radius Rt, is

            dP(r) = (M / 2K) * [ Rt^2 ln(r/Rc) - (r^2 - Rc^2)/2 ]
                  = (M / 2K) * shape(r)

        The prefactor M/2K cancels when that profile is normalised by
        its own value at the lethal corner, so shape(Rt)/shape(r) is a
        pure function of the geometry (Rc, Rt) and is therefore the same
        for every zeroth-order species. That is why one weight serves
        O2, glucose and lactate alike.

        Returns shape(Rt)/shape(r): exactly 1.0 at r = Rt, growing
        without bound as r approaches the vessel wall. Capped at
        _WEIGHT_CAP, which is harmless because the exchange step is
        clamped to the driving gradient -- any sufficiently large value
        gives the same correct answer of "equilibrates within a step".
        """
        return krogh_weight(distance, self.radius_um, self.exchange_radius)


class VascularModule(SimulationModule):
    """
    Vascular system simulation module
    
    Manages:
    - Capillary network
    - O2/glucose delivery
    - Waste removal
    - Diffusion-based exchange
    - Hypoxia detection
    
    Events Emitted:
    - HYPOXIA_DETECTED: When tissue becomes hypoxic
    - VESSEL_FORMED: When angiogenesis occurs
    
    Events Subscribed:
    - CELL_DIVIDED: Track nutrient demand
    - CELL_DIED: Reduce demand
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        # Capillary network
        self.capillaries: Dict[int, Capillary] = {}
        self.next_capillary_id = 0
        
        # Parameters. Read self.config, not the raw argument:
        # SimulationModule.__init__ normalises None to {}, so reading
        # `config` directly made VascularModule() raise
        # AttributeError: 'NoneType' object has no attribute 'get'.
        self.n_capillaries = self.config.get('n_capillaries', 8)
        self.arterial_O2 = self.config.get('arterial_O2', 0.21)
        self.arterial_glucose = self.config.get('arterial_glucose', 5.0)
        self.exchange_rate = self.config.get('exchange_rate', 1.0)
        
        # The per-cell O2 demand the Krogh calibration is anchored to:
        # the largest in the cellular module (normal cells, 0.15 /h).
        # Read by get_O2_at_position and by the lethal-corner test.
        self.reference_O2_demand = float(
            self.config.get('reference_O2_demand', 0.15)
        )

        # Statistics
        self.total_O2_delivered = 0.0
        self.total_glucose_delivered = 0.0
        self.total_waste_removed = 0.0
        self.hypoxic_regions = 0
        
        # Reference to cellular module
        self.cellular_module = None
    
    def initialize(self):
        """Initialize vascular network"""
        print("  Creating capillary network...")
        
        # Create radial capillary pattern (from center outward)
        center = np.array([100.0, 100.0, 50.0])
        
        for i in range(self.n_capillaries):
            angle = 2 * np.pi * i / self.n_capillaries
            
            # Start near center
            start = center + np.array([
                20 * np.cos(angle),
                20 * np.sin(angle),
                np.random.uniform(-5, 5)
            ])
            
            # End at periphery
            end = center + np.array([
                80 * np.cos(angle),
                80 * np.sin(angle),
                np.random.uniform(-5, 5)
            ])
            
            self.add_capillary(start, end)
        
        # Subscribe to events
        self.subscribe(EventTypes.CELL_DIVIDED, self.on_cell_divided)
        self.subscribe(EventTypes.CELL_DIED, self.on_cell_died)
        
        print(f"    ✓ {self.n_capillaries} capillaries")
        print(f"    ✓ Radial network pattern")
    
    def add_capillary(self, start, end):
        """Add capillary to network"""
        capillary_id = self.next_capillary_id
        self.next_capillary_id += 1
        
        capillary = Capillary(
            capillary_id=capillary_id,
            start=np.array(start, dtype=np.float32),
            end=np.array(end, dtype=np.float32)
        )
        
        self.capillaries[capillary_id] = capillary
        return capillary_id
    
    def set_cellular_module(self, cellular_module):
        """Link to cellular module for exchange"""
        self.cellular_module = cellular_module
    
    def update(self, dt: float):
        """Update vascular system"""
        if not self.cellular_module:
            return
        
        # Blood is treated as an inexhaustible arterial reservoir. This
        # is an ASSUMPTION, not a result.
        #
        # Real tissue runs at an O2 extraction ratio of roughly 0.3, so
        # convective headroom over demand is about 1/0.3 = 3.3x -- not
        # orders of magnitude. A 3x rise in local demand or a 70% fall
        # in flow makes a capillary flow-limited, and that is a
        # principal mechanism of tumour hypoxia which this module
        # deliberately omits. A flow-limited regime, if ever wanted,
        # belongs HERE as an explicit per-step budget in unit.um^3.
        #
        # The reset is kept for a separate and dimensional reason: the
        # deleted `capillary.oxygen -= transfer` moved an intensive
        # level 1:1 between a ~1000 um^3 cell and a ~5000 um^3
        # capillary segment, which is wrong whether or not it is then
        # reset. Nothing decrements capillary.oxygen any more, so this
        # reset no longer discards bookkeeping -- there is none to
        # discard.
        for capillary in self.capillaries.values():
            capillary.oxygen = self.arterial_O2
            capillary.glucose = self.arterial_glucose
        
        # Exchange with cells: one pass over cells, each served by
        # its nearest capillary only -- see _exchange_with_tissue.
        # Classify BEFORE delivery. _check_hypoxia reads cell.oxygen, and
        # the post-delivery level carries an O(q*dt) optimistic offset,
        # so the flag would depend on the timestep the user picked. The
        # post-consumption level is C* exactly and is splitting-invariant.
        self._check_hypoxia()

        self._exchange_with_tissue(dt)
    
    def _exchange_with_tissue(self, dt: float):
        """Deliver O2 and glucose to, and clear lactate from, every live
        cell, via the nearest capillary.

        UNITS
        -----
        cell.oxygen / capillary.oxygen are a DIMENSIONLESS normalised O2
        level on which 0.21 == arterial (the dataclass says so). Mapping
        0.21 -> an arterial 100 mmHg makes the cellular module's own
        0.05 hypoxia flag 24 mmHg and its 0.02 death threshold 9.5 mmHg,
        both physiological, so that is the mapping the code already
        implies. glucose and lactate are mM. distance is um, dt is hours
        (division_time_normal = 24.0), so a coefficient multiplying a
        level-difference by dt must be a CONDUCTANCE in 1/h.

        WHAT WAS WRONG
        --------------
        The old expression was

            gradient = (capillary.oxygen - cell.oxygen) / (distance + 1)
            flux     = permeability_O2 * gradient * dt

        summed over 10 sample points per capillary. Three compounding
        errors:

        1. permeability_O2 is already a lumped conductance over the
           exchange path, so dividing by distance double-counted path
           length. The form is wrong too: around a LINE source the
           diffusive resistance goes as ln(r/Rc), never as r.
        2. Summing over sample points made delivery proportional to
           n_samples, a pure discretisation parameter, and accidentally
           restored most of what (1) removed. Life and death were
           decided by a loop bound.
        3. `capillary.oxygen -= transfer` moved an intensive quantity
           1:1 between compartments of unequal volume, and was then
           discarded by the arterial reset at the top of update().

        Net: a measured conductance of 0.0926 /h for a median normal
        cell against a demand of 0.15 /h, so the steady state
        0.21 - 0.15/0.0926 = -1.41 did not exist. Every cell in the
        Baseline scenario was dead by 9.5 h -- 90 deaths, 0 divisions --
        with 2.159 O2 units delivered over 24 h.

        THE MODEL
        ---------
        Each cell is a well-mixed compartment served by the nearest
        capillary:

            dC/dt = K(d) * (C_blood - C) - q
            C*    = C_blood - q / K(d)              (steady state)

        d is the perpendicular distance to the capillary SEGMENT. Only
        the NEAREST capillary contributes, which is not an
        approximation: the Krogh solution below is derived with a
        no-flux boundary at the tissue radius, i.e. it already assumes
        the neighbouring vessels supply the far side of the territory.
        Adding conductances over every capillary in range would
        double-count exactly that.

        THE CONDUCTANCE
        ---------------
        K(d) = permeability * shape(Rt)/shape(d) * exchange_rate, with

            shape(r) = Rt^2 ln(r/Rc) - (r^2 - Rc^2)/2

        the Krogh-cylinder drop profile (Capillary.conductance_weight).
        shape(Rt)/shape(d) is 1 at the tissue radius and rises towards
        the vessel wall. The prefactor M/2K of the Krogh solution
        cancels in that ratio, so the weight is pure geometry and is the
        same function for every zeroth-order species.

        CALIBRATION -- no free parameters
        ---------------------------------
        exchange_radius IS the Krogh radius, i.e. the lethal corner. Its
        defining property fixes each conductance outright: a cell at
        r = Rt, where the weight is exactly 1, must sit exactly at its
        death threshold. So

            permeability = q_max / (C_arterial - C_death)

        O2:      max(0.15, 0.10) / (0.21 - 0.02) = 0.789 /h
        glucose: max(0.20, 0.50) / (5.00 - 0.50) = 0.111 /h

        q_max is the largest per-cell demand in cellular_module
        (_update_metabolism): O2 0.15 /h normal vs 0.10 /h cancer,
        glucose 0.20 mM/h normal vs 0.50 mM/h cancer. Taking the max is
        what makes the radius a genuine lethal corner for every cell
        type rather than for one of them.

        Two independent checks that this is not circular tuning:

        * Krogh's own numbers. At a resting tissue OCR of
          3 mL/100mL/min and K = D*alpha = 5.4e-10 mL/(cm s mmHg) with
          Rc = 4 um, the lethal corner for a 100 -> 5 mmHg drop is
          Rt = 89 um. The module's declared 50 um is the same order, so
          calling exchange_radius a Krogh radius is not a stretch.
        * Interstitial glucose. Nothing above targets a glucose level,
          yet the normal cells of the Baseline scenario settle at a mean
          4.18 mM against 5.0 mM arterial -- 84% of plasma, which is the
          measured interstitial fraction (80-95%). That number fell out
          of the lethal-corner condition.

        Lactate is a DIFFERENT transport process (MCT-mediated efflux
        and perfusive washout, not tissue-limited uptake) and must not
        inherit the glucose coefficient as it used to. It is calibrated
        instead to a stated target -- normal interstitial lactate
        1.5 mM -- giving 0.10 /h / 1.5 mM / weight(11.7 um) = 0.034 /h.

        WHAT THIS BUYS, AND WHAT IT DOES NOT
        ------------------------------------
        The resulting zonation is self-consistent and continuous: at
        exchange_rate 1.0 a normal cell is normoxic out to 25 um,
        hypoxic (< 0.05) from 25 to 50 um, and unsupplied beyond 50 um
        -- where the profile has already reached the death threshold, so
        the cutoff is physically meaningful rather than a cliff. A
        hypoxic core is reachable by GEOMETRY, which is the honest
        mechanism, instead of by a tuned decay length.

        What this is NOT is first-principles transport, and that must
        not be claimed. With real coefficients this geometry is not
        O2-limited at all: D_O2 = 6.5e6 um^2/h equilibrates the measured
        median 11.7 um gap in ~13 ms, and convective supply comfortably
        exceeds the demand of this scenario. The module's own numbers
        say so too:
        at q = 0.15 /h a cell's entire O2 pool lasts 1.4 h, where real
        dissolved tissue O2 lasts ~6 s. cell.oxygen is a phenomenological
        hour-scale oxygen-sufficiency state matched to this engine's
        24 h division time, and the Krogh solution is used here to fix
        the SHAPE of its spatial dependence and the SCALE of its
        coefficients self-consistently -- not to claim the cell is
        solving a diffusion equation.

        STABILITY
        ---------
        Each transfer is clamped to the driving gradient, so a cell can
        never overshoot blood level however large K*dt is. The scheme is
        monotone and bounded by C_blood unconditionally: no CFL
        condition, no hyperoxia. The clamp is also what makes the
        d -> Rc limit correct, since K diverges there and the physical
        answer is "equilibrates within a step".
        """
        if not self.cellular_module or not self.capillaries:
            return

        caps = list(self.capillaries.values())

        for cell in self.cellular_module.cells.values():
            if not cell.alive:
                continue

            # Nearest capillary, by perpendicular distance to the segment.
            cap = caps[0]
            distance = cap.distance_to(cell.position)
            for other in caps[1:]:
                d = other.distance_to(cell.position)
                if d < distance:
                    cap, distance = other, d

            if distance >= cap.exchange_radius:
                continue

            # Krogh geometry factor, shared by all three species.
            weight = cap.conductance_weight(distance) * self.exchange_rate
            if weight <= 0.0:
                continue

            # O2 in. min(..., drive) forbids overshooting blood level.
            drive = cap.oxygen - cell.oxygen
            if drive > 0.0:
                transfer = min(cap.permeability_O2 * weight * drive * dt,
                               drive)
                cell.oxygen += transfer
                self.total_O2_delivered += float(transfer)

            # Glucose in.
            drive = cap.glucose - cell.glucose
            if drive > 0.0:
                transfer = min(cap.permeability_glucose * weight * drive * dt,
                               drive)
                cell.glucose += transfer
                self.total_glucose_delivered += float(transfer)

            # Lactate out, against a blood lactate of 0.
            drive = cell.lactate
            if drive > 0.0:
                removal = min(cap.permeability_lactate * weight * drive * dt,
                              drive)
                cell.lactate -= removal
                self.total_waste_removed += float(removal)
    
    def _check_hypoxia(self):
        """Check for hypoxic regions"""
        if not self.cellular_module:
            return
        
        hypoxic_cells = 0
        
        for cell in self.cellular_module.cells.values():
            if cell.alive and cell.oxygen < 0.05:  # < 5% O2
                hypoxic_cells += 1
        
        changed = hypoxic_cells != self.hypoxic_regions
        # Report the CURRENT count, always. The old guard only assigned
        # when hypoxic_cells > 0, so hypoxic_regions latched at its last
        # non-zero value and the dashboard metric could only ratchet up.
        self.hypoxic_regions = hypoxic_cells

        if hypoxic_cells > 0 and changed:
            self.emit_event(EventTypes.HYPOXIA_DETECTED, {
                'n_hypoxic_cells': hypoxic_cells,
                'severity': 'severe' if hypoxic_cells > 10 else 'moderate'
            })
    
    def get_distance_to_nearest_vessel(self, position: np.ndarray) -> float:
        """Get distance from position to nearest capillary"""
        # Exact perpendicular distance to each segment. The old
        # 10-sample-point scan over-reported distance by up to half the
        # sample spacing, which is 60um/9 = 6.7um for the default network.
        if not self.capillaries:
            return float('inf')
        return min(c.distance_to(position) for c in self.capillaries.values())
    
    def get_O2_at_position(self, position: np.ndarray) -> float:
        """Estimate O2 level at position based on vessel distance"""
        # The same Krogh steady state the exchange path produces, so
        # this estimator and the simulated field cannot disagree. The old
        # exp(-d/50um) cartoon was a second, independent O2 model.
        if not self.capillaries:
            return 0.0
        cap = min(self.capillaries.values(),
                  key=lambda c: c.distance_to(position))
        distance = cap.distance_to(position)
        if distance >= cap.exchange_radius:
            return 0.0
        K = (cap.permeability_O2 * cap.conductance_weight(distance)
             * self.exchange_rate)
        if K <= 0.0:
            return 0.0
        O2 = self.arterial_O2 - self.reference_O2_demand / K
        return float(max(0.0, min(self.arterial_O2, O2)))
    
    def get_state(self) -> Dict[str, Any]:
        """Return current vascular state"""
        if self.cellular_module:
            avg_cell_O2 = np.mean([c.oxygen for c in self.cellular_module.cells.values() 
                                   if c.alive])
            avg_cell_glucose = np.mean([c.glucose for c in self.cellular_module.cells.values() 
                                        if c.alive])
        else:
            avg_cell_O2 = 0
            avg_cell_glucose = 0
        
        return {
            'n_capillaries': len(self.capillaries),
            'total_length': sum(c.length for c in self.capillaries.values()),
            'avg_cell_O2': avg_cell_O2,
            'avg_cell_glucose': avg_cell_glucose,
            'total_O2_delivered': self.total_O2_delivered,
            'total_glucose_delivered': self.total_glucose_delivered,
            'total_waste_removed': self.total_waste_removed,
            'hypoxic_regions': self.hypoxic_regions
        }
    
    # Event handlers
    def on_cell_divided(self, data):
        """Handle cell division - increased demand"""
        # Could trigger angiogenesis if demand is high
        pass
    
    def on_cell_died(self, data):
        """Handle cell death - reduced demand"""
        pass


# Test
if __name__ == '__main__':
    from cognisom.core import SimulationEngine, SimulationConfig
    from cellular_module import CellularModule
    
    print("=" * 70)
    print("Vascular Module Test")
    print("=" * 70)
    print()
    
    # Create engine
    engine = SimulationEngine(SimulationConfig(dt=0.01, duration=1.0))
    
    # Register modules
    engine.register_module('cellular', CellularModule, {
        'n_normal_cells': 20,
        'n_cancer_cells': 5
    })
    
    engine.register_module('vascular', VascularModule, {
        'n_capillaries': 8,
        'exchange_rate': 1.0
    })
    
    # Initialize
    engine.initialize()
    
    # Link modules
    vascular = engine.modules['vascular']
    cellular = engine.modules['cellular']
    vascular.set_cellular_module(cellular)
    
    print("Modules linked")
    print()
    
    # Check initial O2 levels
    print("Initial cell O2 levels:")
    for cell_id, cell in list(cellular.cells.items())[:5]:
        distance = vascular.get_distance_to_nearest_vessel(cell.position)
        print(f"  Cell {cell_id}: O2={cell.oxygen:.3f}, distance={distance:.1f}μm")
    print()
    
    # Run simulation
    engine.run()
    
    # Get results
    print("\nVascular State:")
    vascular_state = vascular.get_state()
    for key, value in vascular_state.items():
        print(f"  {key}: {value}")
    
    print("\nCellular State:")
    cellular_state = cellular.get_state()
    for key, value in cellular_state.items():
        print(f"  {key}: {value}")
    
    print()
    print("=" * 70)
    print("✓ Vascular module working!")
    print("=" * 70)
