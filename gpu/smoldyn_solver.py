"""
GPU-Accelerated Smoldyn-Style Spatial Stochastic Solver
========================================================

Particle-based spatial stochastic simulation with Brownian dynamics
and reaction-diffusion. Each molecule is tracked as an individual
particle with 3D position.

This module provides:
- Brownian motion with per-species diffusion coefficients
- Bimolecular reactions via binding radius (A + B → C)
- Unimolecular reactions (A → B, A → ∅)
- Zeroth-order reactions (∅ → A)
- Reflective, absorbing, and periodic boundaries
- Surface binding/unbinding

VCell Parity Phase 2 - Spatial stochastic simulation.

Usage::

    from cognisom.gpu.smoldyn_solver import SmoldynSolver, SmoldynSystem

    # Create a simple A + B -> C reaction system
    system = SmoldynSystem.simple_binding()

    solver = SmoldynSolver(system, n_max_particles=100000)
    solver.add_particles('A', positions=np.random.rand(1000, 3) * 100)
    solver.add_particles('B', positions=np.random.rand(1000, 3) * 100)

    for _ in range(1000):
        solver.step(dt=0.001)

    print(f"A: {solver.count_species('A')}, C: {solver.count_species('C')}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .backend import get_backend
from .physics_interface import (
    BasePhysicsModel,
    PhysicsBackendType,
    PhysicsModelType,
    PhysicsState,
    register_physics,
)

log = logging.getLogger(__name__)


# ── CUDA Kernels ─────────────────────────────────────────────────────────

_BROWNIAN_MOTION_KERNEL = r"""
extern "C" __global__
void brownian_motion_step(
    float* positions,           // (n_particles, 3)
    const int* species,         // (n_particles,)
    const int* alive,           // (n_particles,)
    const float* diffusion,     // (n_species,) diffusion coefficients
    unsigned long long* rng,    // (n_particles,) RNG state
    float dt,
    int n_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;
    if (!alive[i]) return;

    int sp = species[i];
    float D = diffusion[sp];
    float sigma = sqrtf(2.0f * D * dt);

    // xorshift64 RNG
    unsigned long long s = rng[i];
    #define NEXT() ({ s ^= s << 13; s ^= s >> 7; s ^= s << 17; s; })
    #define UNIFORM() ((NEXT() >> 11) * 2.220446049250313e-16)

    // Box-Muller for 3 Gaussian randoms
    float u1 = (float)UNIFORM();
    float u2 = (float)UNIFORM();
    float u3 = (float)UNIFORM();
    float u4 = (float)UNIFORM();

    if (u1 < 1e-10f) u1 = 1e-10f;
    if (u3 < 1e-10f) u3 = 1e-10f;

    float r1 = sqrtf(-2.0f * logf(u1));
    float r2 = sqrtf(-2.0f * logf(u3));

    float dx = sigma * r1 * cosf(6.2831853f * u2);
    float dy = sigma * r1 * sinf(6.2831853f * u2);
    float dz = sigma * r2 * cosf(6.2831853f * u4);

    positions[i * 3 + 0] += dx;
    positions[i * 3 + 1] += dy;
    positions[i * 3 + 2] += dz;

    rng[i] = s;
}
"""

_BOUNDARY_REFLECT_KERNEL = r"""
extern "C" __global__
void apply_boundaries_reflect(
    float* positions,       // (n_particles, 3)
    const int* alive,       // (n_particles,)
    float x_min, float x_max,
    float y_min, float y_max,
    float z_min, float z_max,
    int n_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;
    if (!alive[i]) return;

    float x = positions[i * 3 + 0];
    float y = positions[i * 3 + 1];
    float z = positions[i * 3 + 2];

    // Reflect off boundaries
    if (x < x_min) { x = 2.0f * x_min - x; }
    if (x > x_max) { x = 2.0f * x_max - x; }
    if (y < y_min) { y = 2.0f * y_min - y; }
    if (y > y_max) { y = 2.0f * y_max - y; }
    if (z < z_min) { z = 2.0f * z_min - z; }
    if (z > z_max) { z = 2.0f * z_max - z; }

    positions[i * 3 + 0] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;
}
"""

_UNIMOLECULAR_REACTION_KERNEL = r"""
extern "C" __global__
void unimolecular_reactions(
    int* species,               // (n_particles,)
    int* alive,                 // (n_particles,)
    const float* rates,         // (n_species,) decay rates
    const int* products,        // (n_species,) product species (-1 = death)
    unsigned long long* rng,    // (n_particles,)
    float dt,
    int n_particles,
    int n_species
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;
    if (!alive[i]) return;

    int sp = species[i];
    if (sp < 0 || sp >= n_species) return;

    float rate = rates[sp];
    if (rate <= 0.0f) return;

    // Probability of reaction in dt
    float prob = 1.0f - expf(-rate * dt);

    // xorshift RNG
    unsigned long long s = rng[i];
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    float u = (float)(s >> 11) * 2.220446049250313e-16f;
    rng[i] = s;

    if (u < prob) {
        int product = products[sp];
        if (product < 0) {
            alive[i] = 0;  // Death
        } else {
            species[i] = product;  // Conversion
        }
    }
}
"""

_SPATIAL_HASH_KERNEL = r"""
extern "C" __global__
void build_spatial_hash(
    const float* positions,     // (n_particles, 3)
    const int* alive,           // (n_particles,)
    int* cell_indices,          // (n_particles,) output cell index
    int* cell_counts,           // (n_cells,) particles per cell
    float cell_size,
    int grid_x, int grid_y, int grid_z,
    float x_min, float y_min, float z_min,
    int n_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;

    if (!alive[i]) {
        cell_indices[i] = -1;
        return;
    }

    float x = positions[i * 3 + 0];
    float y = positions[i * 3 + 1];
    float z = positions[i * 3 + 2];

    int cx = (int)((x - x_min) / cell_size);
    int cy = (int)((y - y_min) / cell_size);
    int cz = (int)((z - z_min) / cell_size);

    // Clamp to grid bounds
    cx = max(0, min(cx, grid_x - 1));
    cy = max(0, min(cy, grid_y - 1));
    cz = max(0, min(cz, grid_z - 1));

    int cell_idx = cx + cy * grid_x + cz * grid_x * grid_y;
    cell_indices[i] = cell_idx;

    atomicAdd(&cell_counts[cell_idx], 1);
}
"""

_BIMOLECULAR_REACTION_KERNEL = r"""
extern "C" __global__
void bimolecular_reactions(
    float* positions,           // (n_particles, 3)
    int* species,               // (n_particles,)
    int* alive,                 // (n_particles,)
    const int* cell_indices,    // (n_particles,)
    const int* cell_starts,     // (n_cells,) start index in sorted list
    const int* cell_counts,     // (n_cells,) count per cell
    const int* sorted_indices,  // (n_particles,) particle indices sorted by cell
    const float* binding_radii, // (n_reactions,) binding radius per reaction
    const int* reactant1,       // (n_reactions,) first reactant species
    const int* reactant2,       // (n_reactions,) second reactant species
    const int* product1,        // (n_reactions,) first product (-1 = none)
    const int* product2,        // (n_reactions,) second product (-1 = none)
    unsigned long long* rng,
    int n_particles,
    int n_reactions,
    int grid_x, int grid_y, int grid_z
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;
    if (!alive[i]) return;

    int sp_i = species[i];
    int cell_i = cell_indices[i];
    if (cell_i < 0) return;

    float xi = positions[i * 3 + 0];
    float yi = positions[i * 3 + 1];
    float zi = positions[i * 3 + 2];

    // Check all reactions where this particle could be reactant1
    for (int r = 0; r < n_reactions; r++) {
        if (sp_i != reactant1[r]) continue;

        int sp_j_target = reactant2[r];
        float bind_r = binding_radii[r];
        float bind_r2 = bind_r * bind_r;

        // Search neighboring cells
        int cx = cell_i % grid_x;
        int cy = (cell_i / grid_x) % grid_y;
        int cz = cell_i / (grid_x * grid_y);

        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int ncx = cx + dx;
                    int ncy = cy + dy;
                    int ncz = cz + dz;

                    if (ncx < 0 || ncx >= grid_x) continue;
                    if (ncy < 0 || ncy >= grid_y) continue;
                    if (ncz < 0 || ncz >= grid_z) continue;

                    int neighbor_cell = ncx + ncy * grid_x + ncz * grid_x * grid_y;
                    int start = cell_starts[neighbor_cell];
                    int count = cell_counts[neighbor_cell];

                    for (int k = 0; k < count; k++) {
                        int j = sorted_indices[start + k];
                        if (j <= i) continue;  // Avoid double-counting
                        if (!alive[j]) continue;
                        if (species[j] != sp_j_target) continue;

                        float xj = positions[j * 3 + 0];
                        float yj = positions[j * 3 + 1];
                        float zj = positions[j * 3 + 2];

                        float dx = xi - xj;
                        float dy = yi - yj;
                        float dz_val = zi - zj;
                        float dist2 = dx*dx + dy*dy + dz_val*dz_val;

                        if (dist2 < bind_r2) {
                            // Reaction occurs!
                            // Use atomic to prevent race conditions
                            int old_alive_i = atomicExch(&alive[i], 0);
                            int old_alive_j = atomicExch(&alive[j], 0);

                            if (old_alive_i && old_alive_j) {
                                // Both were alive, reaction proceeds
                                if (product1[r] >= 0) {
                                    species[i] = product1[r];
                                    alive[i] = 1;
                                    // Place product at midpoint
                                    positions[i * 3 + 0] = (xi + xj) / 2.0f;
                                    positions[i * 3 + 1] = (yi + yj) / 2.0f;
                                    positions[i * 3 + 2] = (zi + zj) / 2.0f;
                                }
                                if (product2[r] >= 0) {
                                    species[j] = product2[r];
                                    alive[j] = 1;
                                }
                                return;  // One reaction per particle per step
                            } else {
                                // Race condition: restore if partner already reacted
                                if (!old_alive_i) alive[i] = 0;
                                if (!old_alive_j) alive[j] = 0;
                            }
                        }
                    }
                }
            }
        }
    }
}
"""


# ── Data Classes ─────────────────────────────────────────────────────────

class ReactionType(str, Enum):
    """Types of chemical reactions."""
    ZEROTH = "zeroth"       # ∅ → A (creation)
    FIRST = "first"         # A → B or A → ∅ (unimolecular)
    SECOND = "second"       # A + B → C (bimolecular)


class BoundaryType(str, Enum):
    """Types of spatial boundaries."""
    REFLECT = "reflect"     # Particles bounce off
    ABSORB = "absorb"       # Particles are removed
    PERIODIC = "periodic"   # Particles wrap around
    TRANSPARENT = "transparent"  # No boundary (infinite domain)


@dataclass
class SmoldynSpecies:
    """Definition of a molecular species."""
    name: str
    diffusion_coeff: float      # µm²/s
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # RGB for visualization
    radius: float = 0.01        # µm, for visualization/collision


@dataclass
class SmoldynReaction:
    """Definition of a chemical reaction."""
    name: str
    reaction_type: ReactionType
    rate: float                 # s⁻¹ for first-order, µm³/s for second-order
    reactants: List[str]        # Species names
    products: List[str]         # Species names (empty for degradation)
    binding_radius: float = 0.0 # µm, for bimolecular reactions
    unbinding_radius: float = 0.0  # µm, separation distance for products


@dataclass
class SmoldynCompartment:
    """Definition of a spatial compartment."""
    name: str
    bounds: Tuple[float, float, float, float, float, float]  # (x_min, x_max, y_min, y_max, z_min, z_max)
    boundary_type: BoundaryType = BoundaryType.REFLECT


@dataclass
class SmoldynSystem:
    """
    Complete definition of a Smoldyn spatial stochastic system.

    Contains species, reactions, and compartment definitions.
    """
    species: List[SmoldynSpecies]
    reactions: List[SmoldynReaction]
    compartment: SmoldynCompartment
    name: str = "unnamed"

    @property
    def species_names(self) -> List[str]:
        return [s.name for s in self.species]

    @property
    def n_species(self) -> int:
        return len(self.species)

    def get_species_index(self, name: str) -> int:
        return self.species_names.index(name)

    @staticmethod
    def simple_binding() -> "SmoldynSystem":
        """
        Simple A + B → C binding reaction.

        Useful for testing bimolecular reaction mechanics.
        """
        return SmoldynSystem(
            species=[
                SmoldynSpecies("A", diffusion_coeff=10.0, color=(1, 0, 0)),
                SmoldynSpecies("B", diffusion_coeff=10.0, color=(0, 1, 0)),
                SmoldynSpecies("C", diffusion_coeff=5.0, color=(0, 0, 1)),
            ],
            reactions=[
                SmoldynReaction(
                    name="A_B_binding",
                    reaction_type=ReactionType.SECOND,
                    rate=1e6,  # µm³/s
                    reactants=["A", "B"],
                    products=["C"],
                    binding_radius=0.1,  # µm
                ),
            ],
            compartment=SmoldynCompartment(
                name="cytoplasm",
                bounds=(0, 100, 0, 100, 0, 100),  # 100 µm cube
                boundary_type=BoundaryType.REFLECT,
            ),
        )

    @staticmethod
    def enzyme_kinetics() -> "SmoldynSystem":
        """
        Michaelis-Menten enzyme kinetics: E + S ⇌ ES → E + P

        Classic enzyme-substrate binding with product release.
        """
        return SmoldynSystem(
            species=[
                SmoldynSpecies("E", diffusion_coeff=5.0, color=(1, 0, 0)),    # Enzyme
                SmoldynSpecies("S", diffusion_coeff=50.0, color=(0, 1, 0)),   # Substrate
                SmoldynSpecies("ES", diffusion_coeff=3.0, color=(1, 1, 0)),   # Complex
                SmoldynSpecies("P", diffusion_coeff=50.0, color=(0, 0, 1)),   # Product
            ],
            reactions=[
                SmoldynReaction(
                    name="binding",
                    reaction_type=ReactionType.SECOND,
                    rate=1e5,
                    reactants=["E", "S"],
                    products=["ES"],
                    binding_radius=0.05,
                ),
                SmoldynReaction(
                    name="unbinding",
                    reaction_type=ReactionType.FIRST,
                    rate=100.0,  # s⁻¹
                    reactants=["ES"],
                    products=["E", "S"],
                    unbinding_radius=0.1,
                ),
                SmoldynReaction(
                    name="catalysis",
                    reaction_type=ReactionType.FIRST,
                    rate=50.0,  # s⁻¹
                    reactants=["ES"],
                    products=["E", "P"],
                    unbinding_radius=0.1,
                ),
            ],
            compartment=SmoldynCompartment(
                name="reaction_volume",
                bounds=(0, 50, 0, 50, 0, 50),
                boundary_type=BoundaryType.REFLECT,
            ),
        )

    @staticmethod
    def min_oscillator() -> "SmoldynSystem":
        """
        MinD/MinE oscillation system (simplified E. coli).

        Spatial oscillator that demonstrates pole-to-pole dynamics.
        """
        return SmoldynSystem(
            species=[
                SmoldynSpecies("MinD_ATP", diffusion_coeff=2.5, color=(1, 0, 0)),
                SmoldynSpecies("MinD_ADP", diffusion_coeff=2.5, color=(0.5, 0, 0)),
                SmoldynSpecies("MinE", diffusion_coeff=2.5, color=(0, 1, 0)),
                SmoldynSpecies("MinDE", diffusion_coeff=0.0, color=(1, 1, 0)),  # Membrane-bound
            ],
            reactions=[
                SmoldynReaction(
                    name="MinD_membrane_binding",
                    reaction_type=ReactionType.FIRST,
                    rate=0.5,
                    reactants=["MinD_ATP"],
                    products=["MinD_ATP"],  # State change handled separately
                ),
                SmoldynReaction(
                    name="MinDE_complex",
                    reaction_type=ReactionType.SECOND,
                    rate=5e4,
                    reactants=["MinD_ATP", "MinE"],
                    products=["MinDE"],
                    binding_radius=0.1,
                ),
                SmoldynReaction(
                    name="MinDE_hydrolysis",
                    reaction_type=ReactionType.FIRST,
                    rate=0.7,
                    reactants=["MinDE"],
                    products=["MinD_ADP", "MinE"],
                    unbinding_radius=0.2,
                ),
            ],
            compartment=SmoldynCompartment(
                name="cell",
                bounds=(0, 4, 0, 1, 0, 1),  # Rod-shaped cell
                boundary_type=BoundaryType.REFLECT,
            ),
        )


# ── Particle System ──────────────────────────────────────────────────────

class ParticleSystem:
    """
    Container for all particles in the simulation.

    Manages particle positions, species, and alive status with
    efficient GPU memory layout.
    """

    def __init__(self, n_max: int, n_species: int):
        """
        Initialize particle system.

        Parameters
        ----------
        n_max : int
            Maximum number of particles
        n_species : int
            Number of species
        """
        self.n_max = n_max
        self.n_species = n_species

        self._backend = get_backend()
        xp = self._backend.xp

        # Particle data (pre-allocated on GPU if available)
        self.positions = xp.zeros((n_max, 3), dtype=np.float32)
        self.species = xp.full(n_max, -1, dtype=np.int32)
        self.alive = xp.zeros(n_max, dtype=np.int32)

        # RNG state per particle
        rng_seeds = np.random.randint(1, 2**32 - 1, size=n_max, dtype=np.uint64)
        self.rng_state = xp.asarray(rng_seeds)

        # Particle count per species
        self._counts = np.zeros(n_species, dtype=np.int32)
        self._next_free = 0

    def add_particles(
        self,
        species_idx: int,
        positions: np.ndarray,
    ) -> np.ndarray:
        """
        Add particles of a given species.

        Parameters
        ----------
        species_idx : int
            Species index
        positions : np.ndarray
            (n, 3) positions to add

        Returns
        -------
        np.ndarray
            Indices of added particles
        """
        n_add = len(positions)
        if self._next_free + n_add > self.n_max:
            raise RuntimeError(f"Cannot add {n_add} particles, only {self.n_max - self._next_free} slots available")

        xp = self._backend.xp
        start = self._next_free
        end = start + n_add

        self.positions[start:end] = xp.asarray(positions.astype(np.float32))
        self.species[start:end] = species_idx
        self.alive[start:end] = 1

        self._counts[species_idx] += n_add
        self._next_free = end

        return np.arange(start, end)

    def remove_particle(self, idx: int):
        """Mark a particle as dead."""
        if self.alive[idx]:
            sp = int(self._backend.to_numpy(self.species[idx]))
            self._counts[sp] -= 1
            self.alive[idx] = 0

    def count_species(self, species_idx: int) -> int:
        """Count alive particles of a species."""
        xp = self._backend.xp
        mask = (self.species == species_idx) & (self.alive == 1)
        return int(xp.sum(mask))

    def get_positions(self, species_idx: Optional[int] = None) -> np.ndarray:
        """Get positions of alive particles (optionally filtered by species)."""
        xp = self._backend.xp

        if species_idx is None:
            mask = self.alive == 1
        else:
            mask = (self.species == species_idx) & (self.alive == 1)

        return self._backend.to_numpy(self.positions[mask])

    @property
    def n_alive(self) -> int:
        """Total number of alive particles."""
        return int(self._backend.xp.sum(self.alive))


# ── Smoldyn Solver ───────────────────────────────────────────────────────

class SmoldynSolver:
    """
    GPU-accelerated Smoldyn-style spatial stochastic solver.

    Simulates particle-based reaction-diffusion with Brownian dynamics.

    Parameters
    ----------
    system : SmoldynSystem
        System definition with species, reactions, compartment
    n_max_particles : int
        Maximum number of particles to pre-allocate
    cell_size : float
        Size of spatial hash cells (should be >= max binding radius)

    Examples
    --------
    >>> system = SmoldynSystem.simple_binding()
    >>> solver = SmoldynSolver(system, n_max_particles=10000)
    >>>
    >>> # Add initial particles
    >>> solver.add_particles('A', np.random.rand(500, 3) * 100)
    >>> solver.add_particles('B', np.random.rand(500, 3) * 100)
    >>>
    >>> # Run simulation
    >>> for _ in range(10000):
    ...     solver.step(dt=0.0001)
    >>>
    >>> print(f"C molecules: {solver.count_species('C')}")
    """

    def __init__(
        self,
        system: SmoldynSystem,
        n_max_particles: int = 100000,
        cell_size: Optional[float] = None,
    ):
        self.system = system
        self.n_max = n_max_particles

        self._backend = get_backend()
        self._time = 0.0

        # Particle system
        self.particles = ParticleSystem(n_max_particles, system.n_species)

        # Diffusion coefficients array
        self._diffusion = np.array(
            [s.diffusion_coeff for s in system.species],
            dtype=np.float32
        )

        # Compartment bounds
        b = system.compartment.bounds
        self._bounds = {
            'x_min': b[0], 'x_max': b[1],
            'y_min': b[2], 'y_max': b[3],
            'z_min': b[4], 'z_max': b[5],
        }

        # Spatial hashing
        self._cell_size = cell_size or self._compute_cell_size()
        self._grid_dims = self._compute_grid_dims()
        self._n_cells = self._grid_dims[0] * self._grid_dims[1] * self._grid_dims[2]

        # Pre-process reactions
        self._setup_reactions()

        # Compile GPU kernels
        self._kernels = {}
        if self._backend.has_gpu:
            self._compile_kernels()

        log.info(f"SmoldynSolver: {system.n_species} species, "
                 f"{len(system.reactions)} reactions, "
                 f"grid={self._grid_dims}, "
                 f"GPU={'yes' if self._backend.has_gpu else 'no'}")

    def _compute_cell_size(self) -> float:
        """Compute cell size from max binding radius."""
        max_bind_r = 0.1  # Default minimum
        for rxn in self.system.reactions:
            if rxn.binding_radius > max_bind_r:
                max_bind_r = rxn.binding_radius
        return max_bind_r * 2  # Cells should be at least 2x binding radius

    def _compute_grid_dims(self) -> Tuple[int, int, int]:
        """Compute spatial hash grid dimensions."""
        b = self._bounds
        nx = max(1, int((b['x_max'] - b['x_min']) / self._cell_size))
        ny = max(1, int((b['y_max'] - b['y_min']) / self._cell_size))
        nz = max(1, int((b['z_max'] - b['z_min']) / self._cell_size))
        return (nx, ny, nz)

    def _setup_reactions(self):
        """Pre-process reactions into arrays for GPU kernels."""
        n_species = self.system.n_species

        # Unimolecular reactions: rate and product per species
        self._uni_rates = np.zeros(n_species, dtype=np.float32)
        self._uni_products = np.full(n_species, -1, dtype=np.int32)

        # Bimolecular reactions
        bimol_rxns = [r for r in self.system.reactions
                      if r.reaction_type == ReactionType.SECOND]
        n_bimol = len(bimol_rxns)

        self._bimol_reactant1 = np.zeros(n_bimol, dtype=np.int32)
        self._bimol_reactant2 = np.zeros(n_bimol, dtype=np.int32)
        self._bimol_product1 = np.full(n_bimol, -1, dtype=np.int32)
        self._bimol_product2 = np.full(n_bimol, -1, dtype=np.int32)
        self._bimol_bind_r = np.zeros(n_bimol, dtype=np.float32)

        for rxn in self.system.reactions:
            if rxn.reaction_type == ReactionType.FIRST:
                sp_idx = self.system.get_species_index(rxn.reactants[0])
                self._uni_rates[sp_idx] = rxn.rate
                if rxn.products:
                    self._uni_products[sp_idx] = self.system.get_species_index(rxn.products[0])
                else:
                    self._uni_products[sp_idx] = -1  # Death

        for i, rxn in enumerate(bimol_rxns):
            self._bimol_reactant1[i] = self.system.get_species_index(rxn.reactants[0])
            self._bimol_reactant2[i] = self.system.get_species_index(rxn.reactants[1])
            if len(rxn.products) > 0:
                self._bimol_product1[i] = self.system.get_species_index(rxn.products[0])
            if len(rxn.products) > 1:
                self._bimol_product2[i] = self.system.get_species_index(rxn.products[1])
            self._bimol_bind_r[i] = rxn.binding_radius

        self._n_bimol = n_bimol

    def _compile_kernels(self):
        """Compile CUDA kernels."""
        try:
            import cupy as cp
            self._kernels['brownian'] = cp.RawKernel(_BROWNIAN_MOTION_KERNEL, "brownian_motion_step")
            self._kernels['boundary'] = cp.RawKernel(_BOUNDARY_REFLECT_KERNEL, "apply_boundaries_reflect")
            self._kernels['unimol'] = cp.RawKernel(_UNIMOLECULAR_REACTION_KERNEL, "unimolecular_reactions")
            self._kernels['hash'] = cp.RawKernel(_SPATIAL_HASH_KERNEL, "build_spatial_hash")
            self._kernels['bimol'] = cp.RawKernel(_BIMOLECULAR_REACTION_KERNEL, "bimolecular_reactions")
            log.debug("Smoldyn CUDA kernels compiled successfully")
        except Exception as e:
            log.warning(f"Failed to compile Smoldyn CUDA kernels: {e}")

    def add_particles(self, species_name: str, positions: np.ndarray) -> np.ndarray:
        """
        Add particles of a given species.

        Parameters
        ----------
        species_name : str
            Name of the species
        positions : np.ndarray
            (n, 3) array of positions

        Returns
        -------
        np.ndarray
            Indices of added particles
        """
        sp_idx = self.system.get_species_index(species_name)
        return self.particles.add_particles(sp_idx, positions)

    def count_species(self, species_name: str) -> int:
        """Count alive particles of a species."""
        sp_idx = self.system.get_species_index(species_name)
        return self.particles.count_species(sp_idx)

    def get_positions(self, species_name: Optional[str] = None) -> np.ndarray:
        """Get positions of particles."""
        if species_name is None:
            return self.particles.get_positions()
        sp_idx = self.system.get_species_index(species_name)
        return self.particles.get_positions(sp_idx)

    def step(self, dt: float):
        """
        Advance simulation by one time step.

        Parameters
        ----------
        dt : float
            Time step in seconds
        """
        if self._backend.has_gpu and 'brownian' in self._kernels:
            self._step_gpu(dt)
        else:
            self._step_cpu(dt)

        self._time += dt

    def _step_gpu(self, dt: float):
        """GPU step using CUDA kernels."""
        import cupy as cp

        n = self.n_max
        block = 256
        grid = (n + block - 1) // block

        # 1. Brownian motion
        diffusion_gpu = cp.asarray(self._diffusion)
        self._kernels['brownian'](
            (grid,), (block,),
            (self.particles.positions, self.particles.species, self.particles.alive,
             diffusion_gpu, self.particles.rng_state,
             np.float32(dt), np.int32(n))
        )

        # 2. Apply boundaries
        b = self._bounds
        self._kernels['boundary'](
            (grid,), (block,),
            (self.particles.positions, self.particles.alive,
             np.float32(b['x_min']), np.float32(b['x_max']),
             np.float32(b['y_min']), np.float32(b['y_max']),
             np.float32(b['z_min']), np.float32(b['z_max']),
             np.int32(n))
        )

        # 3. Unimolecular reactions
        uni_rates_gpu = cp.asarray(self._uni_rates)
        uni_products_gpu = cp.asarray(self._uni_products)
        self._kernels['unimol'](
            (grid,), (block,),
            (self.particles.species, self.particles.alive,
             uni_rates_gpu, uni_products_gpu, self.particles.rng_state,
             np.float32(dt), np.int32(n), np.int32(self.system.n_species))
        )

        # 4. Bimolecular reactions (if any)
        if self._n_bimol > 0:
            self._bimolecular_reactions_gpu(dt)

        cp.cuda.Stream.null.synchronize()

    def _bimolecular_reactions_gpu(self, dt: float):
        """GPU bimolecular reactions with spatial hashing."""
        import cupy as cp

        n = self.n_max
        block = 256
        grid = (n + block - 1) // block

        # Build spatial hash
        cell_indices = cp.zeros(n, dtype=np.int32)
        cell_counts = cp.zeros(self._n_cells, dtype=np.int32)

        gx, gy, gz = self._grid_dims
        b = self._bounds

        self._kernels['hash'](
            (grid,), (block,),
            (self.particles.positions, self.particles.alive,
             cell_indices, cell_counts,
             np.float32(self._cell_size),
             np.int32(gx), np.int32(gy), np.int32(gz),
             np.float32(b['x_min']), np.float32(b['y_min']), np.float32(b['z_min']),
             np.int32(n))
        )

        # Compute cell_starts via prefix sum
        cell_starts = cp.zeros(self._n_cells, dtype=np.int32)
        cp.cumsum(cell_counts[:-1], out=cell_starts[1:])

        # Sort particles by cell
        sorted_indices = cp.argsort(cell_indices).astype(np.int32)

        # Run bimolecular reactions
        reactant1_gpu = cp.asarray(self._bimol_reactant1)
        reactant2_gpu = cp.asarray(self._bimol_reactant2)
        product1_gpu = cp.asarray(self._bimol_product1)
        product2_gpu = cp.asarray(self._bimol_product2)
        bind_r_gpu = cp.asarray(self._bimol_bind_r)

        self._kernels['bimol'](
            (grid,), (block,),
            (self.particles.positions, self.particles.species, self.particles.alive,
             cell_indices, cell_starts, cell_counts, sorted_indices,
             bind_r_gpu, reactant1_gpu, reactant2_gpu, product1_gpu, product2_gpu,
             self.particles.rng_state,
             np.int32(n), np.int32(self._n_bimol),
             np.int32(gx), np.int32(gy), np.int32(gz))
        )

    def _step_cpu(self, dt: float):
        """CPU fallback step."""
        positions = self._backend.to_numpy(self.particles.positions)
        species = self._backend.to_numpy(self.particles.species)
        alive = self._backend.to_numpy(self.particles.alive)

        n = self.n_max
        n_species = self.system.n_species

        # 1. Brownian motion
        for i in range(n):
            if not alive[i]:
                continue
            sp = species[i]
            D = self._diffusion[sp]
            sigma = np.sqrt(2 * D * dt)
            positions[i] += np.random.randn(3) * sigma

        # 2. Reflect boundaries
        b = self._bounds
        for i in range(n):
            if not alive[i]:
                continue
            for dim, (lo, hi) in enumerate([(b['x_min'], b['x_max']),
                                            (b['y_min'], b['y_max']),
                                            (b['z_min'], b['z_max'])]):
                if positions[i, dim] < lo:
                    positions[i, dim] = 2 * lo - positions[i, dim]
                if positions[i, dim] > hi:
                    positions[i, dim] = 2 * hi - positions[i, dim]

        # 3. Unimolecular reactions
        for i in range(n):
            if not alive[i]:
                continue
            sp = species[i]
            rate = self._uni_rates[sp]
            if rate > 0:
                prob = 1 - np.exp(-rate * dt)
                if np.random.rand() < prob:
                    product = self._uni_products[sp]
                    if product < 0:
                        alive[i] = 0
                    else:
                        species[i] = product

        # 4. Bimolecular reactions (O(n²) for CPU - simplified)
        for r in range(self._n_bimol):
            sp1 = self._bimol_reactant1[r]
            sp2 = self._bimol_reactant2[r]
            bind_r2 = self._bimol_bind_r[r] ** 2

            for i in range(n):
                if not alive[i] or species[i] != sp1:
                    continue
                for j in range(i + 1, n):
                    if not alive[j] or species[j] != sp2:
                        continue

                    dist2 = np.sum((positions[i] - positions[j]) ** 2)
                    if dist2 < bind_r2:
                        # React
                        alive[i] = 0
                        alive[j] = 0

                        prod1 = self._bimol_product1[r]
                        prod2 = self._bimol_product2[r]

                        midpoint = (positions[i] + positions[j]) / 2

                        if prod1 >= 0:
                            species[i] = prod1
                            positions[i] = midpoint
                            alive[i] = 1

                        if prod2 >= 0:
                            species[j] = prod2
                            positions[j] = midpoint
                            alive[j] = 1

                        break

        # Write back
        xp = self._backend.xp
        self.particles.positions = xp.asarray(positions)
        self.particles.species = xp.asarray(species)
        self.particles.alive = xp.asarray(alive)

    def get_state(self) -> Dict[str, Any]:
        """Get current simulation state."""
        counts = {
            sp.name: self.count_species(sp.name)
            for sp in self.system.species
        }
        return {
            'time': self._time,
            'n_alive': self.particles.n_alive,
            'species_counts': counts,
            'n_max': self.n_max,
        }

    @property
    def time(self) -> float:
        """Current simulation time."""
        return self._time


# ── Physics Model Registration ───────────────────────────────────────────

@register_physics(
    "cupy_smoldyn",
    version="1.0.0",
    backend=PhysicsBackendType.CUPY,
    model_type=PhysicsModelType.SMOLDYN,
)
class CuPySmoldynPhysics(BasePhysicsModel):
    """
    CuPy-based Smoldyn solver following PhysicsModel protocol.
    """

    def __init__(self, name: str = "cupy_smoldyn", config: Optional[Dict] = None):
        super().__init__(name, config)
        self._solver: Optional[SmoldynSolver] = None

    @property
    def backend_type(self) -> PhysicsBackendType:
        return PhysicsBackendType.CUPY

    @property
    def model_type(self) -> PhysicsModelType:
        return PhysicsModelType.SMOLDYN

    def initialize(self, state: PhysicsState, **config) -> None:
        """
        Initialize Smoldyn solver.

        Config options:
            system: SmoldynSystem or 'simple_binding', 'enzyme_kinetics', 'min_oscillator'
            n_max_particles: Maximum particles
            initial_particles: Dict of species_name -> positions array
        """
        system_config = config.get('system', 'simple_binding')
        if isinstance(system_config, SmoldynSystem):
            system = system_config
        elif system_config == 'enzyme_kinetics':
            system = SmoldynSystem.enzyme_kinetics()
        elif system_config == 'min_oscillator':
            system = SmoldynSystem.min_oscillator()
        else:
            system = SmoldynSystem.simple_binding()

        n_max = config.get('n_max_particles', 100000)
        self._solver = SmoldynSolver(system, n_max)

        # Add initial particles
        initial = config.get('initial_particles', {})
        for species_name, positions in initial.items():
            self._solver.add_particles(species_name, positions)

        self._initialized = True

    def step(self, dt: float, state: PhysicsState) -> PhysicsState:
        """Advance Smoldyn simulation by dt."""
        if not self._initialized or self._solver is None:
            raise RuntimeError("Smoldyn solver not initialized")

        self._solver.step(dt)

        # Update PhysicsState
        state.custom['smoldyn_state'] = self._solver.get_state()
        state.custom['smoldyn_time'] = self._solver.time

        return state


# ── Convenience Functions ────────────────────────────────────────────────

def create_smoldyn_solver(
    system: Union[str, SmoldynSystem] = 'simple_binding',
    n_max_particles: int = 100000,
    **kwargs
) -> SmoldynSolver:
    """
    Create a Smoldyn solver.

    Parameters
    ----------
    system : str or SmoldynSystem
        'simple_binding', 'enzyme_kinetics', 'min_oscillator', or SmoldynSystem
    n_max_particles : int
        Maximum particles
    **kwargs
        Additional arguments to SmoldynSolver

    Returns
    -------
    SmoldynSolver
        Configured solver instance
    """
    if isinstance(system, str):
        if system == 'enzyme_kinetics':
            system = SmoldynSystem.enzyme_kinetics()
        elif system == 'min_oscillator':
            system = SmoldynSystem.min_oscillator()
        else:
            system = SmoldynSystem.simple_binding()

    return SmoldynSolver(system, n_max_particles, **kwargs)
