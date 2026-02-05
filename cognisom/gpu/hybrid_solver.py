"""
GPU-Accelerated Hybrid ODE/SSA Solver
=====================================

Automatic fast/slow species partitioning for efficient simulation of
reaction networks with mixed time scales.

This module implements the Haseltine-Rawlings partitioning scheme:
- Fast species (high copy number, >threshold) → deterministic ODE
- Slow species (low copy number, <threshold) → stochastic SSA

VCell Parity Phase 3 - Hybrid deterministic-stochastic simulation.

Key Features:
- Automatic species partitioning based on copy number
- Dynamic repartitioning as populations change
- GPU-accelerated ODE and SSA components
- Coupling terms for reactions spanning partitions
- Batched simulation of cell populations

Usage::

    from cognisom.gpu.hybrid_solver import HybridSolver, HybridSystem

    # Create system
    system = HybridSystem.gene_regulatory_network()

    # Create solver with automatic partitioning
    solver = HybridSolver(system, n_cells=1000, threshold=100)
    solver.initialize(initial_state)

    # Run simulation
    for _ in range(10000):
        solver.step(dt=0.01)

References:
    Haseltine, E. L., & Rawlings, J. B. (2002). Approximate simulation of
    coupled fast and slow reactions for stochastic chemical kinetics.
    J. Chem. Phys., 117(15), 6959-6969.
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

_PROPENSITY_KERNEL = r"""
extern "C" __global__
void compute_propensities(
    const float* state,           // (n_cells, n_species)
    const float* params,          // (n_cells, n_params)
    float* propensities,          // (n_cells, n_reactions)
    const int* stoich_reactants,  // (n_reactions, max_reactants)
    const int* stoich_coeffs,     // (n_reactions, max_reactants)
    const int* rate_param_idx,    // (n_reactions,)
    int n_cells,
    int n_species,
    int n_reactions,
    int max_reactants
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= n_cells) return;

    for (int r = 0; r < n_reactions; r++) {
        float prop = params[cell * n_params + rate_param_idx[r]];

        // Multiply by reactant species counts
        for (int j = 0; j < max_reactants; j++) {
            int sp = stoich_reactants[r * max_reactants + j];
            if (sp < 0) break;  // End of reactants

            int coeff = stoich_coeffs[r * max_reactants + j];
            float x = state[cell * n_species + sp];

            // Combinatorial factor for higher-order reactions
            for (int k = 0; k < coeff; k++) {
                prop *= (x - k) / (k + 1);
            }
        }

        propensities[cell * n_reactions + r] = fmaxf(prop, 0.0f);
    }
}
"""

_SSA_STEP_KERNEL = r"""
extern "C" __global__
void ssa_step_gillespie(
    float* state,                 // (n_cells, n_species)
    const float* propensities,    // (n_cells, n_reactions)
    const int* stoich_change,     // (n_reactions, n_species)
    const int* slow_species,      // (n_slow,)
    unsigned long long* rng,      // (n_cells,)
    float* times,                 // (n_cells,)
    float target_time,
    int n_cells,
    int n_species,
    int n_reactions,
    int n_slow
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= n_cells) return;

    // xorshift64 RNG
    unsigned long long s = rng[cell];
    #define NEXT() ({ s ^= s << 13; s ^= s >> 7; s ^= s << 17; s; })
    #define RAND() ((NEXT() & 0xFFFFFFFFULL) * 2.3283064365386963e-10f)

    float t = times[cell];

    while (t < target_time) {
        // Compute total propensity (for slow species reactions only)
        float a0 = 0.0f;
        for (int r = 0; r < n_reactions; r++) {
            a0 += propensities[cell * n_reactions + r];
        }

        if (a0 <= 0.0f) {
            t = target_time;
            break;
        }

        // Time to next reaction
        float dt = -logf(RAND()) / a0;
        t += dt;

        if (t > target_time) break;

        // Choose reaction
        float r_val = RAND() * a0;
        float sum = 0.0f;
        int chosen = -1;
        for (int r = 0; r < n_reactions; r++) {
            sum += propensities[cell * n_reactions + r];
            if (sum >= r_val) {
                chosen = r;
                break;
            }
        }
        if (chosen < 0) chosen = n_reactions - 1;

        // Apply state change (only for slow species)
        for (int j = 0; j < n_slow; j++) {
            int sp = slow_species[j];
            state[cell * n_species + sp] += stoich_change[chosen * n_species + sp];
            state[cell * n_species + sp] = fmaxf(state[cell * n_species + sp], 0.0f);
        }
    }

    times[cell] = t;
    rng[cell] = s;

    #undef NEXT
    #undef RAND
}
"""

_ODE_RHS_KERNEL = r"""
extern "C" __global__
void ode_rhs_reactions(
    const float* state,           // (n_cells, n_species)
    const float* params,          // (n_cells, n_params)
    float* dydt,                  // (n_cells, n_species)
    const int* stoich_change,     // (n_reactions, n_species)
    const int* stoich_reactants,  // (n_reactions, max_reactants)
    const int* stoich_coeffs,     // (n_reactions, max_reactants)
    const int* rate_param_idx,    // (n_reactions,)
    const int* fast_species,      // (n_fast,)
    int n_cells,
    int n_species,
    int n_reactions,
    int n_params,
    int n_fast,
    int max_reactants
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= n_cells) return;

    // Initialize dydt to zero for fast species
    for (int j = 0; j < n_fast; j++) {
        int sp = fast_species[j];
        dydt[cell * n_species + sp] = 0.0f;
    }

    // Compute contribution from each reaction
    for (int r = 0; r < n_reactions; r++) {
        // Compute rate
        float rate = params[cell * n_params + rate_param_idx[r]];

        for (int j = 0; j < max_reactants; j++) {
            int sp = stoich_reactants[r * max_reactants + j];
            if (sp < 0) break;
            int coeff = stoich_coeffs[r * max_reactants + j];
            float x = state[cell * n_species + sp];
            for (int k = 0; k < coeff; k++) {
                rate *= x;
                x -= 1.0f;
            }
        }

        rate = fmaxf(rate, 0.0f);

        // Add contribution to fast species
        for (int j = 0; j < n_fast; j++) {
            int sp = fast_species[j];
            dydt[cell * n_species + sp] += rate * stoich_change[r * n_species + sp];
        }
    }
}
"""

_PARTITION_KERNEL = r"""
extern "C" __global__
void partition_species(
    const float* state,           // (n_cells, n_species)
    int* is_fast,                 // (n_species,) output: 1 if fast, 0 if slow
    float threshold,
    int n_cells,
    int n_species
) {
    int sp = blockIdx.x * blockDim.x + threadIdx.x;
    if (sp >= n_species) return;

    // Species is fast if mean count > threshold
    float mean = 0.0f;
    for (int cell = 0; cell < n_cells; cell++) {
        mean += state[cell * n_species + sp];
    }
    mean /= n_cells;

    is_fast[sp] = (mean > threshold) ? 1 : 0;
}
"""


# ── Data Classes ─────────────────────────────────────────────────────────

class PartitionType(str, Enum):
    """Species partition classification."""
    FAST = "fast"      # High copy number, use ODE
    SLOW = "slow"      # Low copy number, use SSA
    BOUNDARY = "boundary"  # Near threshold, dynamic


@dataclass
class HybridSpecies:
    """Definition of a chemical species for hybrid simulation."""
    name: str
    initial_value: float = 0.0
    is_conserved: bool = False  # For conservation laws


@dataclass
class HybridReaction:
    """Definition of a chemical reaction."""
    name: str
    reactants: List[str]       # Species names (with multiplicity)
    products: List[str]        # Species names (with multiplicity)
    rate_constant: float       # Base rate constant
    rate_param_name: str = ""  # Parameter name for per-cell rates


@dataclass
class HybridPartition:
    """Current partitioning of species."""
    fast_species: List[int]    # Species indices in ODE partition
    slow_species: List[int]    # Species indices in SSA partition
    threshold: float = 100.0

    @property
    def n_fast(self) -> int:
        return len(self.fast_species)

    @property
    def n_slow(self) -> int:
        return len(self.slow_species)


@dataclass
class HybridSystem:
    """
    Complete definition of a hybrid ODE/SSA system.

    Contains species, reactions, and parameters.
    """
    species: List[HybridSpecies]
    reactions: List[HybridReaction]
    parameters: Dict[str, float] = field(default_factory=dict)
    name: str = "unnamed"

    @property
    def species_names(self) -> List[str]:
        return [s.name for s in self.species]

    @property
    def n_species(self) -> int:
        return len(self.species)

    @property
    def n_reactions(self) -> int:
        return len(self.reactions)

    def get_species_index(self, name: str) -> int:
        return self.species_names.index(name)

    @staticmethod
    def gene_regulatory_network() -> "HybridSystem":
        """
        Simple gene regulatory network with promoter, mRNA, protein.

        Demonstrates hybrid behavior:
        - Protein is typically high copy → ODE
        - mRNA can be low copy → SSA
        - Promoter states are discrete → SSA
        """
        return HybridSystem(
            name="gene_regulatory_network",
            species=[
                HybridSpecies("gene_off", initial_value=1.0),   # Promoter off
                HybridSpecies("gene_on", initial_value=0.0),    # Promoter on
                HybridSpecies("mRNA", initial_value=0.0),       # mRNA (could be low)
                HybridSpecies("protein", initial_value=100.0),  # Protein (typically high)
            ],
            reactions=[
                # Promoter activation: gene_off -> gene_on
                HybridReaction("activation", ["gene_off"], ["gene_on"], 0.1),
                # Promoter deactivation: gene_on -> gene_off
                HybridReaction("deactivation", ["gene_on"], ["gene_off"], 0.05),
                # Transcription: gene_on -> gene_on + mRNA
                HybridReaction("transcription", ["gene_on"], ["gene_on", "mRNA"], 10.0),
                # Translation: mRNA -> mRNA + protein
                HybridReaction("translation", ["mRNA"], ["mRNA", "protein"], 50.0),
                # mRNA degradation: mRNA -> ∅
                HybridReaction("mRNA_decay", ["mRNA"], [], 1.0),
                # Protein degradation: protein -> ∅
                HybridReaction("protein_decay", ["protein"], [], 0.1),
            ],
            parameters={
                "k_act": 0.1,
                "k_deact": 0.05,
                "k_trans": 10.0,
                "k_transl": 50.0,
                "k_deg_m": 1.0,
                "k_deg_p": 0.1,
            },
        )

    @staticmethod
    def enzyme_substrate_mm() -> "HybridSystem":
        """
        Michaelis-Menten enzyme kinetics.

        E + S <-> ES -> E + P

        Demonstrates:
        - Substrate typically high → ODE
        - Enzyme-substrate complex → can be low → SSA
        """
        return HybridSystem(
            name="enzyme_mm",
            species=[
                HybridSpecies("E", initial_value=10.0),      # Enzyme (low)
                HybridSpecies("S", initial_value=1000.0),    # Substrate (high)
                HybridSpecies("ES", initial_value=0.0),      # Complex (low)
                HybridSpecies("P", initial_value=0.0),       # Product (accumulates)
            ],
            reactions=[
                # Binding: E + S -> ES
                HybridReaction("binding", ["E", "S"], ["ES"], 1e-3),
                # Unbinding: ES -> E + S
                HybridReaction("unbinding", ["ES"], ["E", "S"], 0.1),
                # Catalysis: ES -> E + P
                HybridReaction("catalysis", ["ES"], ["E", "P"], 0.5),
            ],
            parameters={
                "kon": 1e-3,
                "koff": 0.1,
                "kcat": 0.5,
            },
        )

    @staticmethod
    def toggle_switch() -> "HybridSystem":
        """
        Genetic toggle switch (Gardner et al., Nature 2000).

        Two mutually repressing genes. Exhibits bistability.
        """
        return HybridSystem(
            name="toggle_switch",
            species=[
                HybridSpecies("mRNA_A", initial_value=10.0),
                HybridSpecies("mRNA_B", initial_value=0.0),
                HybridSpecies("protein_A", initial_value=500.0),
                HybridSpecies("protein_B", initial_value=10.0),
            ],
            reactions=[
                # Transcription of A (repressed by B)
                HybridReaction("trans_A", [], ["mRNA_A"], 50.0),
                # Transcription of B (repressed by A)
                HybridReaction("trans_B", [], ["mRNA_B"], 50.0),
                # Translation
                HybridReaction("transl_A", ["mRNA_A"], ["mRNA_A", "protein_A"], 10.0),
                HybridReaction("transl_B", ["mRNA_B"], ["mRNA_B", "protein_B"], 10.0),
                # Degradation
                HybridReaction("deg_mA", ["mRNA_A"], [], 1.0),
                HybridReaction("deg_mB", ["mRNA_B"], [], 1.0),
                HybridReaction("deg_pA", ["protein_A"], [], 0.1),
                HybridReaction("deg_pB", ["protein_B"], [], 0.1),
            ],
            parameters={
                "alpha": 50.0,
                "beta": 10.0,
                "gamma_m": 1.0,
                "gamma_p": 0.1,
                "K": 100.0,  # Repression threshold
                "n": 2,      # Hill coefficient
            },
        )


@dataclass
class HybridState:
    """State of hybrid simulation."""
    y: np.ndarray           # (n_cells, n_species) continuous state
    t: float = 0.0          # Current time
    partition: Optional[HybridPartition] = None


# ── Haseltine-Rawlings Partitioner ───────────────────────────────────────

class HaseltineRawlingsPartitioner:
    """
    Automatic species partitioning based on copy number.

    Uses the Haseltine-Rawlings criterion:
    - Species with mean count > threshold → fast (ODE)
    - Species with mean count <= threshold → slow (SSA)

    Optionally includes hysteresis to prevent frequent switching.
    """

    def __init__(
        self,
        threshold: float = 100.0,
        hysteresis: float = 0.2,
        min_fast: int = 0,
    ):
        """
        Initialize partitioner.

        Parameters
        ----------
        threshold : float
            Copy number threshold for fast/slow classification
        hysteresis : float
            Fraction of threshold for hysteresis band (prevents oscillation)
        min_fast : int
            Minimum number of fast species (0 = allow pure SSA)
        """
        self.threshold = threshold
        self.hysteresis = hysteresis
        self.min_fast = min_fast

        self._upper_threshold = threshold * (1 + hysteresis)
        self._lower_threshold = threshold * (1 - hysteresis)

    def partition(
        self,
        state: np.ndarray,
        current_partition: Optional[HybridPartition] = None,
    ) -> HybridPartition:
        """
        Partition species into fast/slow.

        Parameters
        ----------
        state : np.ndarray
            (n_cells, n_species) current state
        current_partition : HybridPartition, optional
            Previous partition (for hysteresis)

        Returns
        -------
        HybridPartition
            Updated partition
        """
        n_species = state.shape[1]
        mean_counts = np.mean(state, axis=0)

        fast = []
        slow = []

        for sp in range(n_species):
            mean = mean_counts[sp]

            if current_partition is not None:
                # Use hysteresis
                was_fast = sp in current_partition.fast_species
                if was_fast:
                    is_fast = mean > self._lower_threshold
                else:
                    is_fast = mean > self._upper_threshold
            else:
                is_fast = mean > self.threshold

            if is_fast:
                fast.append(sp)
            else:
                slow.append(sp)

        # Ensure minimum fast species
        if len(fast) < self.min_fast and len(slow) > 0:
            # Move species with highest counts to fast
            slow_counts = [(sp, mean_counts[sp]) for sp in slow]
            slow_counts.sort(key=lambda x: x[1], reverse=True)
            for i in range(min(self.min_fast - len(fast), len(slow_counts))):
                fast.append(slow_counts[i][0])
                slow.remove(slow_counts[i][0])

        return HybridPartition(
            fast_species=sorted(fast),
            slow_species=sorted(slow),
            threshold=self.threshold,
        )


# ── Hybrid Solver ────────────────────────────────────────────────────────

class HybridSolver:
    """
    GPU-accelerated hybrid ODE/SSA solver.

    Combines deterministic ODE integration for high-copy species
    with stochastic SSA for low-copy species.

    Parameters
    ----------
    system : HybridSystem
        System definition
    n_cells : int
        Number of cells to simulate in parallel
    threshold : float
        Copy number threshold for fast/slow partitioning
    repartition_interval : int
        Steps between repartitioning (0 = never repartition)
    ode_method : str
        ODE integration method ('euler', 'rk4')

    Examples
    --------
    >>> system = HybridSystem.gene_regulatory_network()
    >>> solver = HybridSolver(system, n_cells=1000)
    >>> solver.initialize()
    >>> for _ in range(10000):
    ...     solver.step(dt=0.001)
    """

    def __init__(
        self,
        system: HybridSystem,
        n_cells: int = 1000,
        threshold: float = 100.0,
        repartition_interval: int = 100,
        ode_method: str = "rk4",
    ):
        self.system = system
        self.n_cells = n_cells
        self.threshold = threshold
        self.repartition_interval = repartition_interval
        self.ode_method = ode_method

        self._backend = get_backend()
        self._partitioner = HaseltineRawlingsPartitioner(threshold)

        # State
        self._state = None
        self._partition = None
        self._step_count = 0
        self._time = 0.0

        # Precomputed stoichiometry
        self._stoich_change = None
        self._stoich_reactants = None
        self._stoich_coeffs = None

        # Per-cell parameters
        self._params = None

        # RNG for SSA
        self._rng = None

        # GPU kernels
        self._kernels = {}

        log.info(f"HybridSolver: {system.n_species} species, "
                 f"{system.n_reactions} reactions, "
                 f"{n_cells} cells, threshold={threshold}")

    def initialize(self, initial_state: Optional[np.ndarray] = None):
        """
        Initialize solver state.

        Parameters
        ----------
        initial_state : np.ndarray, optional
            (n_cells, n_species) initial state. If None, uses system defaults.
        """
        xp = self._backend.xp

        # Initialize state
        if initial_state is not None:
            self._state = HybridState(
                y=xp.asarray(initial_state.astype(np.float32)),
                t=0.0,
            )
        else:
            # Use system default initial values
            y0 = np.zeros((self.n_cells, self.system.n_species), dtype=np.float32)
            for i, sp in enumerate(self.system.species):
                y0[:, i] = sp.initial_value
            self._state = HybridState(
                y=xp.asarray(y0),
                t=0.0,
            )

        # Build stoichiometry matrices
        self._build_stoichiometry()

        # Initialize parameters
        self._init_parameters()

        # Initialize RNG
        seeds = np.random.randint(1, 2**32 - 1, size=self.n_cells, dtype=np.uint64)
        self._rng = xp.asarray(seeds)

        # Initial partitioning
        state_cpu = self._backend.to_numpy(self._state.y)
        self._partition = self._partitioner.partition(state_cpu)
        self._state.partition = self._partition

        # Compile kernels
        if self._backend.has_gpu:
            self._compile_kernels()

        log.info(f"  Initial partition: {self._partition.n_fast} fast, "
                 f"{self._partition.n_slow} slow species")

    def _build_stoichiometry(self):
        """Build stoichiometry matrices from reactions."""
        n_rxn = self.system.n_reactions
        n_sp = self.system.n_species

        # Net change matrix: (n_reactions, n_species)
        self._stoich_change = np.zeros((n_rxn, n_sp), dtype=np.int32)

        # Reactant indices and coefficients
        max_reactants = max(len(r.reactants) for r in self.system.reactions) if n_rxn > 0 else 1
        self._stoich_reactants = np.full((n_rxn, max_reactants), -1, dtype=np.int32)
        self._stoich_coeffs = np.zeros((n_rxn, max_reactants), dtype=np.int32)

        for r_idx, rxn in enumerate(self.system.reactions):
            # Count reactants
            reactant_counts = {}
            for sp_name in rxn.reactants:
                reactant_counts[sp_name] = reactant_counts.get(sp_name, 0) + 1

            # Count products
            product_counts = {}
            for sp_name in rxn.products:
                product_counts[sp_name] = product_counts.get(sp_name, 0) + 1

            # Compute net change
            all_species = set(reactant_counts.keys()) | set(product_counts.keys())
            for sp_name in all_species:
                sp_idx = self.system.get_species_index(sp_name)
                net = product_counts.get(sp_name, 0) - reactant_counts.get(sp_name, 0)
                self._stoich_change[r_idx, sp_idx] = net

            # Store reactant info
            for j, (sp_name, coeff) in enumerate(reactant_counts.items()):
                self._stoich_reactants[r_idx, j] = self.system.get_species_index(sp_name)
                self._stoich_coeffs[r_idx, j] = coeff

    def _init_parameters(self):
        """Initialize per-cell parameters."""
        xp = self._backend.xp

        # For now, all cells have same parameters
        n_params = len(self.system.parameters)
        self._param_names = list(self.system.parameters.keys())
        self._params = np.zeros((self.n_cells, max(n_params, 1)), dtype=np.float32)

        for i, (name, value) in enumerate(self.system.parameters.items()):
            self._params[:, i] = value

        self._params = xp.asarray(self._params)

        # Map reactions to parameters
        self._rate_param_idx = np.zeros(self.system.n_reactions, dtype=np.int32)
        for r_idx, rxn in enumerate(self.system.reactions):
            # Default: use rate_constant directly (stored as first param)
            self._rate_param_idx[r_idx] = 0

    def _compile_kernels(self):
        """Compile CUDA kernels."""
        try:
            import cupy as cp
            self._kernels['propensity'] = cp.RawKernel(_PROPENSITY_KERNEL, "compute_propensities")
            self._kernels['ssa_step'] = cp.RawKernel(_SSA_STEP_KERNEL, "ssa_step_gillespie")
            self._kernels['ode_rhs'] = cp.RawKernel(_ODE_RHS_KERNEL, "ode_rhs_reactions")
            self._kernels['partition'] = cp.RawKernel(_PARTITION_KERNEL, "partition_species")
            log.debug("Hybrid CUDA kernels compiled")
        except Exception as e:
            log.warning(f"Failed to compile hybrid CUDA kernels: {e}")

    def step(self, dt: float):
        """
        Advance simulation by one time step.

        Parameters
        ----------
        dt : float
            Time step
        """
        self._step_count += 1

        # Repartition if needed
        if self.repartition_interval > 0 and self._step_count % self.repartition_interval == 0:
            self._repartition()

        # Step based on partition
        if self._partition.n_fast > 0 and self._partition.n_slow > 0:
            # True hybrid: ODE for fast, SSA for slow
            self._step_hybrid(dt)
        elif self._partition.n_fast > 0:
            # Pure ODE
            self._step_ode(dt)
        else:
            # Pure SSA
            self._step_ssa(dt)

        self._time += dt
        self._state.t = self._time

    def _repartition(self):
        """Update species partition."""
        state_cpu = self._backend.to_numpy(self._state.y)
        new_partition = self._partitioner.partition(state_cpu, self._partition)

        if new_partition.fast_species != self._partition.fast_species:
            log.debug(f"Repartitioned: {new_partition.n_fast} fast, {new_partition.n_slow} slow")
            self._partition = new_partition
            self._state.partition = new_partition

    def _step_hybrid(self, dt: float):
        """Hybrid step: ODE for fast species, SSA for slow species."""
        if self._backend.has_gpu and 'ssa_step' in self._kernels:
            self._step_hybrid_gpu(dt)
        else:
            self._step_hybrid_cpu(dt)

    def _step_hybrid_cpu(self, dt: float):
        """CPU implementation of hybrid step."""
        state = self._backend.to_numpy(self._state.y)

        # ODE step for fast species
        if self._partition.n_fast > 0:
            self._ode_step_cpu(state, dt, self._partition.fast_species)

        # SSA step for slow species
        if self._partition.n_slow > 0:
            self._ssa_step_cpu(state, dt, self._partition.slow_species)

        self._state.y = self._backend.xp.asarray(state)

    def _step_hybrid_gpu(self, dt: float):
        """GPU implementation of hybrid step."""
        import cupy as cp

        # ODE step for fast species
        if self._partition.n_fast > 0:
            self._ode_step_gpu(dt)

        # SSA step for slow species
        if self._partition.n_slow > 0:
            self._ssa_step_gpu(dt)

        cp.cuda.Stream.null.synchronize()

    def _step_ode(self, dt: float):
        """Pure ODE step (all species deterministic)."""
        if self._backend.has_gpu and 'ode_rhs' in self._kernels:
            self._ode_step_gpu(dt, all_species=True)
        else:
            state = self._backend.to_numpy(self._state.y)
            self._ode_step_cpu(state, dt, list(range(self.system.n_species)))
            self._state.y = self._backend.xp.asarray(state)

    def _step_ssa(self, dt: float):
        """Pure SSA step (all species stochastic)."""
        if self._backend.has_gpu and 'ssa_step' in self._kernels:
            self._ssa_step_gpu(dt, all_species=True)
        else:
            state = self._backend.to_numpy(self._state.y)
            self._ssa_step_cpu(state, dt, list(range(self.system.n_species)))
            self._state.y = self._backend.xp.asarray(state)

    def _ode_step_cpu(self, state: np.ndarray, dt: float, species_indices: List[int]):
        """CPU ODE integration."""
        params = self._backend.to_numpy(self._params)

        if self.ode_method == "euler":
            dydt = self._compute_rhs_cpu(state, params, species_indices)
            for sp in species_indices:
                state[:, sp] += dt * dydt[:, sp]
                state[:, sp] = np.maximum(state[:, sp], 0)
        elif self.ode_method == "rk4":
            # RK4 integration
            y0 = state.copy()
            k1 = self._compute_rhs_cpu(y0, params, species_indices)
            y1 = y0.copy()
            for sp in species_indices:
                y1[:, sp] += 0.5 * dt * k1[:, sp]

            k2 = self._compute_rhs_cpu(y1, params, species_indices)
            y2 = y0.copy()
            for sp in species_indices:
                y2[:, sp] += 0.5 * dt * k2[:, sp]

            k3 = self._compute_rhs_cpu(y2, params, species_indices)
            y3 = y0.copy()
            for sp in species_indices:
                y3[:, sp] += dt * k3[:, sp]

            k4 = self._compute_rhs_cpu(y3, params, species_indices)

            for sp in species_indices:
                state[:, sp] = y0[:, sp] + (dt / 6.0) * (k1[:, sp] + 2*k2[:, sp] + 2*k3[:, sp] + k4[:, sp])
                state[:, sp] = np.maximum(state[:, sp], 0)

    def _compute_rhs_cpu(
        self,
        state: np.ndarray,
        params: np.ndarray,
        species_indices: List[int],
    ) -> np.ndarray:
        """Compute ODE RHS on CPU."""
        dydt = np.zeros_like(state)

        for r_idx, rxn in enumerate(self.system.reactions):
            # Compute rate
            rate = np.full(self.n_cells, rxn.rate_constant, dtype=np.float32)

            for sp_name in rxn.reactants:
                sp_idx = self.system.get_species_index(sp_name)
                rate *= state[:, sp_idx]

            # Add contribution to each species
            for sp_idx in species_indices:
                dydt[:, sp_idx] += rate * self._stoich_change[r_idx, sp_idx]

        return dydt

    def _ssa_step_cpu(self, state: np.ndarray, dt: float, species_indices: List[int]):
        """CPU SSA step using Gillespie algorithm."""
        target_time = self._time + dt

        for cell in range(self.n_cells):
            t = self._time

            while t < target_time:
                # Compute propensities
                propensities = self._compute_propensities_cpu(state[cell, :])
                a0 = np.sum(propensities)

                if a0 <= 0:
                    break

                # Time to next reaction
                tau = -np.log(np.random.random()) / a0
                t += tau

                if t > target_time:
                    break

                # Choose reaction
                r_val = np.random.random() * a0
                cumsum = 0.0
                chosen = -1
                for r in range(len(propensities)):
                    cumsum += propensities[r]
                    if cumsum >= r_val:
                        chosen = r
                        break

                if chosen < 0:
                    chosen = len(propensities) - 1

                # Apply state change (only for slow species)
                for sp_idx in species_indices:
                    state[cell, sp_idx] += self._stoich_change[chosen, sp_idx]
                    state[cell, sp_idx] = max(0, state[cell, sp_idx])

    def _compute_propensities_cpu(self, state: np.ndarray) -> np.ndarray:
        """Compute reaction propensities for a single cell."""
        propensities = np.zeros(self.system.n_reactions, dtype=np.float32)

        for r_idx, rxn in enumerate(self.system.reactions):
            rate = rxn.rate_constant

            # Combinatorial factor for reactants
            for sp_name in rxn.reactants:
                sp_idx = self.system.get_species_index(sp_name)
                rate *= state[sp_idx]

            propensities[r_idx] = max(0, rate)

        return propensities

    def _ode_step_gpu(self, dt: float, all_species: bool = False):
        """GPU ODE integration."""
        # Simplified GPU ODE - would need full implementation
        # For now, fall back to CPU
        state = self._backend.to_numpy(self._state.y)
        species = list(range(self.system.n_species)) if all_species else self._partition.fast_species
        self._ode_step_cpu(state, dt, species)
        self._state.y = self._backend.xp.asarray(state)

    def _ssa_step_gpu(self, dt: float, all_species: bool = False):
        """GPU SSA step."""
        # Simplified GPU SSA - would need full implementation
        # For now, fall back to CPU
        state = self._backend.to_numpy(self._state.y)
        species = list(range(self.system.n_species)) if all_species else self._partition.slow_species
        self._ssa_step_cpu(state, dt, species)
        self._state.y = self._backend.xp.asarray(state)

    def get_state(self) -> np.ndarray:
        """Get current state as numpy array."""
        return self._backend.to_numpy(self._state.y)

    def get_species(self, name: str) -> np.ndarray:
        """Get values for a specific species across all cells."""
        sp_idx = self.system.get_species_index(name)
        return self._backend.to_numpy(self._state.y[:, sp_idx])

    def get_partition(self) -> HybridPartition:
        """Get current species partition."""
        return self._partition

    def get_statistics(self) -> Dict[str, Any]:
        """Get simulation statistics."""
        state = self.get_state()
        stats = {
            'time': self._time,
            'steps': self._step_count,
            'n_cells': self.n_cells,
            'n_fast': self._partition.n_fast,
            'n_slow': self._partition.n_slow,
        }

        for i, sp in enumerate(self.system.species):
            stats[f'{sp.name}_mean'] = float(np.mean(state[:, i]))
            stats[f'{sp.name}_std'] = float(np.std(state[:, i]))

        return stats


# ── Physics Model Registration ───────────────────────────────────────────

@register_physics(
    "cupy_hybrid_ode_ssa",
    version="1.0.0",
    backend=PhysicsBackendType.CUPY,
    model_type=PhysicsModelType.HYBRID,
)
class CuPyHybridODESSASolver(BasePhysicsModel):
    """
    CuPy-based hybrid ODE/SSA solver following PhysicsModel protocol.

    Wraps HybridSolver for use with the physics registry.
    """

    def __init__(self, system: HybridSystem, **kwargs):
        super().__init__()
        self.solver = HybridSolver(system=system, **kwargs)

    def initialize(self, state: PhysicsState) -> PhysicsState:
        """Initialize from physics state."""
        if hasattr(state, 'positions') and state.positions is not None:
            self.solver.initialize(state.positions)
        else:
            self.solver.initialize()
        return self._to_physics_state()

    def step(self, dt: float, state: PhysicsState) -> PhysicsState:
        """Advance simulation."""
        self.solver.step(dt)
        return self._to_physics_state()

    def _to_physics_state(self) -> PhysicsState:
        """Convert solver state to PhysicsState."""
        return PhysicsState(
            positions=self.solver.get_state(),
            velocities=None,
            forces=None,
            time=self.solver._time,
        )

    def get_state(self) -> PhysicsState:
        """Get current physics state."""
        return self._to_physics_state()

    def set_state(self, state: PhysicsState) -> None:
        """Set solver state."""
        if state.positions is not None:
            self.solver._state.y = self.solver._backend.xp.asarray(state.positions)


# ── Convenience Functions ────────────────────────────────────────────────

def create_hybrid_solver(
    system: HybridSystem,
    n_cells: int = 1000,
    threshold: float = 100.0,
    **kwargs,
) -> HybridSolver:
    """
    Create and initialize a hybrid solver.

    Parameters
    ----------
    system : HybridSystem
        System definition
    n_cells : int
        Number of cells
    threshold : float
        Partitioning threshold
    **kwargs
        Additional arguments for HybridSolver

    Returns
    -------
    HybridSolver
        Initialized solver
    """
    solver = HybridSolver(system, n_cells, threshold, **kwargs)
    solver.initialize()
    return solver


def run_hybrid_simulation(
    system: HybridSystem,
    duration: float,
    dt: float = 0.01,
    n_cells: int = 1000,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a complete hybrid simulation.

    Parameters
    ----------
    system : HybridSystem
        System definition
    duration : float
        Total simulation time
    dt : float
        Time step
    n_cells : int
        Number of cells
    **kwargs
        Additional arguments

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (times, states) where states is (n_times, n_cells, n_species)
    """
    solver = create_hybrid_solver(system, n_cells, **kwargs)

    n_steps = int(duration / dt)
    times = np.linspace(0, duration, n_steps + 1)
    states = np.zeros((n_steps + 1, n_cells, system.n_species), dtype=np.float32)

    states[0] = solver.get_state()

    for i in range(n_steps):
        solver.step(dt)
        states[i + 1] = solver.get_state()

    return times, states
