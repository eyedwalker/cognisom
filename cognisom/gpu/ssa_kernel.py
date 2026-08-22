"""
GPU-Accelerated Stochastic Simulation Algorithm (Phase 3 Priority)
===================================================================

Implements the Gillespie SSA and tau-leaping algorithms on GPU for
batch-parallel intracellular gene expression across thousands of cells.

Each cell has its own set of species (mRNA, protein copy numbers) and
reaction propensities. The GPU kernel advances all cells simultaneously
using independent random streams.

Architecture:
    - Species matrix:  (n_cells, n_species) int32
    - Stoich matrix:   (n_reactions, n_species) int32 — shared across cells
    - Rate vector:     (n_cells, n_reactions) float32 — cell-specific rates
    - RNG state:       per-cell xoshiro256** state

Two solver modes:
    1. Direct SSA (exact, slow):  One reaction per step per cell
    2. Tau-leaping (approximate): Poisson-sampled multi-reaction steps

Usage::

    from cognisom.gpu.ssa_kernel import BatchSSA, GeneExpressionModel

    model = GeneExpressionModel.prostate_cell()
    solver = BatchSSA(model, n_cells=10000)
    solver.advance(dt=0.01)   # advance all cells by dt hours
    mrna_counts = solver.get_species("AR_mRNA")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import math

import numpy as np

from .backend import get_backend


def mass_action_propensities(species: np.ndarray, rates: np.ndarray,
                             S: np.ndarray) -> np.ndarray:
    """Stochastic mass-action propensities for every cell and reaction.

    For a reaction consuming ``m`` copies of species ``s``, the number of
    distinct reactant combinations available is ``C(X_s, m)``, so the
    propensity carries a factor ``X(X-1)...(X-m+1)/m!`` -- not ``X``.

    The previous code multiplied by the count exactly once per species
    with negative stoichiometry, ignoring the magnitude: ``2A -> B`` was
    computed as ``c*A`` rather than ``c*A(A-1)/2``. That is correct only
    where every reactant stoichiometry is exactly -1, and silently wrong
    for any dimerisation. It also let a reaction needing two molecules
    fire when only one remained, since ``c*1 > 0``; ``C(1,2)`` is 0.

    Args:
        species: (n_cells, n_species) integer molecule counts.
        rates:   (n_cells, n_reactions) stochastic rate constants.
        S:       (n_reactions, n_species) stoichiometry matrix.

    Returns:
        (n_cells, n_reactions) float64 propensities.
    """
    n_reactions, n_species = S.shape
    a = np.array(rates, dtype=np.float64, copy=True)

    for r in range(n_reactions):
        for s in range(n_species):
            order = -int(S[r, s])
            if order <= 0:
                continue
            counts = species[:, s].astype(np.float64)
            term = np.ones_like(counts)
            for k in range(order):
                term *= np.maximum(counts - k, 0.0)
            if order > 1:
                term /= math.factorial(order)
            a[:, r] *= term

    return np.maximum(a, 0.0)

log = logging.getLogger(__name__)

# ── CUDA kernel source ────────────────────────────────────────────────

# Shared device code. Kept as one string so the two kernels below cannot
# drift apart on the definition of a propensity.
_PROPENSITY_DEVICE_SRC = r"""
// Number of distinct ways to choose `order` molecules from `count`:
// C(count, order) = count*(count-1)*...*(count-order+1) / order!
//
// This is the combinatorial factor in the stochastic mass-action
// propensity. Multiplying by the raw count once per reactant species
// (what this code did before) is correct only when every reactant
// stoichiometry is -1, and it lets a reaction that consumes two
// molecules fire when only one is left.
__device__ __forceinline__ float combinations(int count, int order) {
    if (order <= 1) return (order == 1) ? (float)count : 1.0f;
    if (count < order) return 0.0f;
    float num = 1.0f;
    for (int k = 0; k < order; k++) num *= (float)(count - k);
    float den = 1.0f;
    for (int k = 2; k <= order; k++) den *= (float)k;
    return num / den;
}
"""

_TAU_LEAP_KERNEL = _PROPENSITY_DEVICE_SRC + r"""
extern "C" __global__
void tau_leap_step(
    int*         species,     // (n_cells, n_species) — copy numbers
    const int*   S,           // (n_reactions, n_species) — stoichiometry
    const float* rates,       // (n_cells, n_reactions) — propensities
    const float  dt,          // time step
    unsigned long long* rng,  // (n_cells, 4) — xoshiro256** state
    int n_cells,
    int n_species,
    int n_reactions
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= n_cells) return;

    // ── xoshiro256** RNG (per-cell) ──────────────────────────────
    unsigned long long s0 = rng[cell * 4 + 0];
    unsigned long long s1 = rng[cell * 4 + 1];
    unsigned long long s2 = rng[cell * 4 + 2];
    unsigned long long s3 = rng[cell * 4 + 3];

    #define ROTL(x, k) ((x) << (k)) | ((x) >> (64 - (k)))
    #define NEXT() ({                                        \
        unsigned long long result = ROTL(s1 * 5, 7) * 9;    \
        unsigned long long t = s1 << 17;                     \
        s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3;           \
        s2 ^= t;  s3 = ROTL(s3, 45);                        \
        result;                                              \
    })

    // Uniform [0,1)
    #define UNIFORM() ((NEXT() >> 11) * 0x1.0p-53)

    // Poisson(lambda) via Knuth for small lambda, normal approx for large
    #define POISSON(lam) ({                                  \
        int _k;                                              \
        if ((lam) < 30.0) {                                  \
            double _L = exp(-(double)(lam));                  \
            double _p = 1.0;                                 \
            _k = 0;                                          \
            do { _k++; _p *= UNIFORM(); } while (_p > _L);   \
            _k -= 1;                                         \
        } else {                                             \
            /* Box-Muller + normal approx */                  \
            double _u1 = UNIFORM();                          \
            double _u2 = UNIFORM();                          \
            if (_u1 < 1e-15) _u1 = 1e-15;                   \
            double _z = sqrt(-2.0 * log(_u1)) * cos(6.283185307 * _u2); \
            _k = (int)((lam) + sqrt((lam)) * _z + 0.5);     \
            if (_k < 0) _k = 0;                              \
        }                                                    \
        _k;                                                  \
    })

    int base = cell * n_species;

    // A tau-leap holds the state fixed across the whole interval: every
    // reaction's firing count must be drawn at the start-of-leap state.
    // Applying each reaction's stoichiometry before evaluating the next
    // reaction's propensity (as this did) makes the result depend on the
    // order reactions are listed in.
    int firings[64];  // max 64 reactions, matching the direct kernel

    for (int r = 0; r < n_reactions && r < 64; r++) {
        float a = rates[cell * n_reactions + r];

        // Stochastic mass-action: rate * prod C(count_s, order_s)
        for (int s = 0; s < n_species; s++) {
            int stoich = S[r * n_species + s];
            if (stoich < 0) {
                a *= combinations(species[base + s], -stoich);
            }
        }

        firings[r] = (a > 0.0f) ? POISSON(a * dt) : 0;
    }

    // Then apply every reaction's stoichiometry together.
    for (int r = 0; r < n_reactions && r < 64; r++) {
        if (firings[r] == 0) continue;
        for (int s = 0; s < n_species; s++) {
            int delta = S[r * n_species + s] * firings[r];
            int new_val = species[base + s] + delta;
            species[base + s] = (new_val > 0) ? new_val : 0;
        }
    }

    // Write back RNG state
    rng[cell * 4 + 0] = s0;
    rng[cell * 4 + 1] = s1;
    rng[cell * 4 + 2] = s2;
    rng[cell * 4 + 3] = s3;
}
"""

_DIRECT_SSA_KERNEL = _PROPENSITY_DEVICE_SRC + r"""
extern "C" __global__
void direct_ssa_step(
    int*         species,     // (n_cells, n_species)
    const int*   S,           // (n_reactions, n_species)
    const float* rates,       // (n_cells, n_reactions)
    float*       times,       // (n_cells,) — current sim time per cell
    const float  dt,          // interval to advance EACH cell by
    unsigned long long* rng,  // (n_cells, 4)
    int n_cells,
    int n_species,
    int n_reactions,
    int max_steps             // safety limit
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= n_cells) return;

    unsigned long long s0 = rng[cell * 4 + 0];
    unsigned long long s1 = rng[cell * 4 + 1];
    unsigned long long s2 = rng[cell * 4 + 2];
    unsigned long long s3 = rng[cell * 4 + 3];

    #define ROTL(x, k) ((x) << (k)) | ((x) >> (64 - (k)))
    #define NEXT() ({                                        \
        unsigned long long result = ROTL(s1 * 5, 7) * 9;    \
        unsigned long long t = s1 << 17;                     \
        s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3;           \
        s2 ^= t;  s3 = ROTL(s3, 45);                        \
        result;                                              \
    })
    #define UNIFORM() ((NEXT() >> 11) * 0x1.0p-53)

    int base = cell * n_species;
    // Each cell advances by dt from ITS OWN clock. The launch used to pass
    // a single t_end = max(times) + dt for the whole grid, so any cell
    // behind the leader integrated further than dt -- the further behind,
    // the bigger the extra interval.
    float t = times[cell];
    const float t_end = t + dt;

    for (int step = 0; step < max_steps && t < t_end; step++) {
        // Compute propensities
        float a_total = 0.0f;
        float a_cumsum[64];  // max 64 reactions

        for (int r = 0; r < n_reactions && r < 64; r++) {
            float a = rates[cell * n_reactions + r];
            for (int s = 0; s < n_species; s++) {
                int stoich = S[r * n_species + s];
                if (stoich < 0) {
                    a *= combinations(species[base + s], -stoich);
                }
            }
            a_total += a;
            a_cumsum[r] = a_total;
        }

        if (a_total <= 0.0f) break;

        // Time to next reaction
        double r1 = UNIFORM();
        if (r1 < 1e-15) r1 = 1e-15;
        float tau = (float)(-log(r1) / (double)a_total);
        // The event lands outside the interval, so it has not happened
        // yet: advance to the boundary and stop. Storing the overshot
        // time let per-cell clocks drift apart run after run.
        if (t + tau > t_end) { t = t_end; break; }
        t += tau;

        // Select reaction
        double r2 = UNIFORM() * (double)a_total;
        int rxn = 0;
        for (int r = 0; r < n_reactions && r < 64; r++) {
            if ((double)a_cumsum[r] >= r2) { rxn = r; break; }
        }

        // Fire reaction
        for (int s = 0; s < n_species; s++) {
            int new_val = species[base + s] + S[rxn * n_species + s];
            species[base + s] = (new_val > 0) ? new_val : 0;
        }
    }

    times[cell] = t;
    rng[cell * 4 + 0] = s0;
    rng[cell * 4 + 1] = s1;
    rng[cell * 4 + 2] = s2;
    rng[cell * 4 + 3] = s3;
}
"""


# ── Reaction model ────────────────────────────────────────────────────

@dataclass
class Reaction:
    """A single chemical reaction for the SSA."""
    name: str
    stoichiometry: Dict[str, int]  # species_name -> change (+/-)
    rate_constant: float           # base rate (per hour)


@dataclass
class GeneExpressionModel:
    """Defines species and reactions for batch SSA.

    Species are indexed by position in a flat array.
    Reactions reference species by name; the solver maps to indices.
    """
    species_names: List[str]
    initial_counts: Dict[str, int]
    reactions: List[Reaction]

    @staticmethod
    def prostate_cell() -> GeneExpressionModel:
        """Standard prostate epithelial cell gene expression model.

        7 genes x 4 reactions each = 28 reactions
        Species: gene_{name}, mRNA_{name}, protein_{name} (gene species are
        constant catalysts — not consumed).
        """
        genes = [
            ("AR",    0.15, 12.0, 0.04, 0.008),   # Androgen Receptor
            ("TP53",  0.10, 10.0, 0.05, 0.010),   # p53
            ("PTEN",  0.08,  8.0, 0.06, 0.012),   # PTEN
            ("MYC",   0.20, 15.0, 0.07, 0.015),   # MYC (oncogene)
            ("BRCA2", 0.05,  5.0, 0.03, 0.005),   # BRCA2
            ("ERG",   0.12, 10.0, 0.05, 0.010),   # ERG (fusion oncogene)
            ("GAPDH", 0.50, 20.0, 0.05, 0.010),   # housekeeping
        ]

        species = []
        initial = {}
        reactions = []

        for gene_name, k_tx, k_tl, k_mrna_deg, k_prot_deg in genes:
            g = f"gene_{gene_name}"
            m = f"mRNA_{gene_name}"
            p = f"protein_{gene_name}"

            species.extend([g, m, p])
            initial[g] = 1       # one gene copy
            initial[m] = 0       # start with no mRNA
            initial[p] = 0       # start with no protein

            # Transcription: gene -> gene + mRNA  (gene is catalyst)
            reactions.append(Reaction(
                name=f"transcription_{gene_name}",
                stoichiometry={m: +1},  # gene unchanged (catalyst)
                rate_constant=k_tx,
            ))

            # Translation: mRNA -> mRNA + protein  (mRNA is catalyst)
            reactions.append(Reaction(
                name=f"translation_{gene_name}",
                stoichiometry={p: +1},  # mRNA unchanged (catalyst)
                rate_constant=k_tl,
            ))

            # mRNA degradation: mRNA -> null
            reactions.append(Reaction(
                name=f"mrna_degradation_{gene_name}",
                stoichiometry={m: -1},
                rate_constant=k_mrna_deg,
            ))

            # Protein degradation: protein -> null
            reactions.append(Reaction(
                name=f"protein_degradation_{gene_name}",
                stoichiometry={p: -1},
                rate_constant=k_prot_deg,
            ))

        return GeneExpressionModel(
            species_names=species,
            initial_counts=initial,
            reactions=reactions,
        )

    @staticmethod
    def cancer_cell() -> GeneExpressionModel:
        """Prostate cancer cell — elevated MYC/ERG, reduced PTEN/TP53."""
        model = GeneExpressionModel.prostate_cell()

        # Modify rates for cancer phenotype
        rate_mods = {
            "transcription_MYC": 3.0,    # MYC overexpression
            "transcription_ERG": 2.5,    # ERG fusion driver
            "transcription_AR":  2.0,    # AR amplification
            "transcription_PTEN": 0.2,   # PTEN loss
            "transcription_TP53": 0.3,   # TP53 mutation / loss
        }
        for rxn in model.reactions:
            if rxn.name in rate_mods:
                rxn.rate_constant *= rate_mods[rxn.name]

        return model


# ── Batch SSA solver ──────────────────────────────────────────────────

class BatchSSA:
    """GPU-accelerated batch stochastic simulation for many cells.

    On GPU: compiles and runs CUDA tau-leaping or direct SSA kernels.
    On CPU: vectorized NumPy tau-leaping (still parallel across cells).
    """

    def __init__(
        self,
        model: GeneExpressionModel,
        n_cells: int = 1000,
        seed: int = 42,
    ):
        self._model = model
        self._n_cells = n_cells
        self._backend = get_backend()
        xp = self._backend.xp

        n_species = len(model.species_names)
        n_reactions = len(model.reactions)

        self._species_idx = {name: i for i, name in enumerate(model.species_names)}
        self._n_species = n_species
        self._n_reactions = n_reactions

        # Build stoichiometry matrix (n_reactions, n_species)
        S = np.zeros((n_reactions, n_species), dtype=np.int32)
        for r, rxn in enumerate(model.reactions):
            for sp_name, delta in rxn.stoichiometry.items():
                S[r, self._species_idx[sp_name]] = delta
        self._S = xp.asarray(S)

        # Build rate vector (n_cells, n_reactions) — initially uniform
        rates = np.zeros((n_cells, n_reactions), dtype=np.float32)
        for r, rxn in enumerate(model.reactions):
            rates[:, r] = rxn.rate_constant
        self._rates = xp.asarray(rates)

        # Initialize species (n_cells, n_species)
        species = np.zeros((n_cells, n_species), dtype=np.int32)
        for name, count in model.initial_counts.items():
            species[:, self._species_idx[name]] = count
        self._species = xp.asarray(species)

        # Time per cell
        self._times = xp.zeros(n_cells, dtype=np.float32)

        # RNG state: 4 x uint64 per cell (xoshiro256**)
        rng = np.random.SeedSequence(seed)
        rng_states = np.zeros((n_cells, 4), dtype=np.uint64)
        for i, child in enumerate(rng.spawn(n_cells)):
            rng_states[i] = child.generate_state(4)
        self._rng = xp.asarray(rng_states)

        # Compile CUDA kernels if on GPU
        self._tau_kernel = None
        self._ssa_kernel = None
        if self._backend.has_gpu:
            try:
                import cupy as cp
                self._tau_kernel = cp.RawKernel(_TAU_LEAP_KERNEL, "tau_leap_step")
                self._ssa_kernel = cp.RawKernel(_DIRECT_SSA_KERNEL, "direct_ssa_step")
                log.info("SSA CUDA kernels compiled successfully")
            except Exception as e:
                log.warning("Failed to compile SSA CUDA kernels: %s", e)

    # ── Public API ────────────────────────────────────────────────

    def advance(self, dt: float, method: str = "tau") -> None:
        """Advance all cells by dt hours.

        Args:
            dt: time step in hours
            method: "tau" for tau-leaping (fast), "direct" for exact SSA
        """
        if method == "tau":
            self._advance_tau(dt)
        elif method == "direct":
            self._advance_direct(dt)
        else:
            raise ValueError(f"Unknown SSA method: {method}")

    def get_species(self, name: str) -> np.ndarray:
        """Get copy numbers for a species across all cells.

        Returns: (n_cells,) int array on CPU.
        """
        idx = self._species_idx[name]
        return self._backend.to_numpy(self._species[:, idx])

    def get_all_species(self) -> np.ndarray:
        """Get full species matrix.

        Returns: (n_cells, n_species) int array on CPU.
        """
        return self._backend.to_numpy(self._species)

    def get_mean_expression(self) -> Dict[str, float]:
        """Get mean copy number for each species across all cells."""
        data = self._backend.to_numpy(self._species).astype(float)
        return {
            name: float(data[:, idx].mean())
            for name, idx in self._species_idx.items()
        }

    def set_rates(self, cell_mask: np.ndarray, reaction_name: str, rate: float):
        """Override rate constant for specific cells.

        Args:
            cell_mask: boolean array (n_cells,) — which cells to modify
            reaction_name: reaction to modify
            rate: new rate constant
        """
        rxn_idx = None
        for r, rxn in enumerate(self._model.reactions):
            if rxn.name == reaction_name:
                rxn_idx = r
                break
        if rxn_idx is None:
            raise KeyError(f"Reaction not found: {reaction_name}")

        xp = self._backend.xp
        mask = xp.asarray(cell_mask) if not hasattr(cell_mask, '__cuda_array_interface__') else cell_mask
        self._rates[mask, rxn_idx] = rate

    @property
    def n_cells(self) -> int:
        return self._n_cells

    @property
    def species_names(self) -> List[str]:
        return self._model.species_names

    @property
    def reaction_names(self) -> List[str]:
        return [r.name for r in self._model.reactions]

    # ── GPU tau-leaping ───────────────────────────────────────────

    def _advance_tau(self, dt: float):
        """Tau-leaping: Poisson-sample all reactions simultaneously."""
        if self._tau_kernel is not None:
            self._gpu_tau_leap(dt)
        else:
            self._cpu_tau_leap(dt)

    def _gpu_tau_leap(self, dt: float):
        """Run tau-leaping CUDA kernel."""
        import cupy as cp

        block = 256
        grid = (self._n_cells + block - 1) // block

        self._tau_kernel(
            (grid,), (block,),
            (
                self._species,
                self._S,
                self._rates,
                np.float32(dt),
                self._rng,
                np.int32(self._n_cells),
                np.int32(self._n_species),
                np.int32(self._n_reactions),
            ),
        )
        cp.cuda.Stream.null.synchronize()

    def _cpu_tau_leap(self, dt: float):
        """Vectorized NumPy tau-leaping (all cells in parallel).

        A tau-leap holds the state fixed across the leap: every reaction's
        firing count is drawn from a propensity evaluated at the state at
        the *start* of the interval, and all the stoichiometry is applied
        together at the end.

        The previous loop applied each reaction's stoichiometry before
        computing the next reaction's propensity, so reaction ``r`` saw a
        state that reactions ``0..r-1`` had already changed. That makes the
        result depend on the order reactions happen to be listed in, and
        biases it -- it is sequential operator splitting, not a tau-leap.
        """
        species = self._backend.to_numpy(self._species)
        rates = self._backend.to_numpy(self._rates)
        S = self._backend.to_numpy(self._S)

        # All propensities at the common start-of-leap state.
        a = mass_action_propensities(species, rates, S)

        # All firing counts, still at that state.
        firings = np.random.poisson(np.maximum(a * dt, 0.0)).astype(np.int64)

        # Then apply every reaction's stoichiometry at once.
        delta = firings @ S.astype(np.int64)      # (n_cells, n_species)
        updated = species.astype(np.int64) + delta
        np.clip(updated, 0, None, out=updated)
        species[:] = updated.astype(species.dtype)

        self._species = self._backend.xp.asarray(species)

    # ── GPU direct SSA ────────────────────────────────────────────

    def _advance_direct(self, dt: float):
        """Direct Gillespie SSA — exact but slower."""
        if self._ssa_kernel is not None:
            self._gpu_direct_ssa(dt)
        else:
            self._cpu_direct_ssa(dt)

    def _gpu_direct_ssa(self, dt: float):
        """Run direct SSA CUDA kernel."""
        import cupy as cp

        block = 256
        grid = (self._n_cells + block - 1) // block

        self._ssa_kernel(
            (grid,), (block,),
            (
                self._species,
                self._S,
                self._rates,
                self._times,
                np.float32(dt),
                self._rng,
                np.int32(self._n_cells),
                np.int32(self._n_species),
                np.int32(self._n_reactions),
                np.int32(100000),  # max steps safety
            ),
        )
        cp.cuda.Stream.null.synchronize()

    def _cpu_direct_ssa(self, dt: float):
        """CPU direct SSA — loops over cells sequentially."""
        species = self._backend.to_numpy(self._species)
        rates = self._backend.to_numpy(self._rates)
        S = self._backend.to_numpy(self._S)
        times = self._backend.to_numpy(self._times)

        for cell in range(self._n_cells):
            t = float(times[cell])
            t_end = t + dt

            for _ in range(100000):
                if t >= t_end:
                    break

                # Compute propensities (with the combinatorial factor, so a
                # reaction needing two molecules cannot fire on one).
                a = mass_action_propensities(
                    species[cell:cell + 1], rates[cell:cell + 1], S
                )[0]

                a_total = a.sum()
                if a_total <= 0:
                    break

                # Time to next reaction
                r1 = np.random.random()
                if r1 < 1e-15:
                    r1 = 1e-15
                tau = -np.log(r1) / a_total
                if t + tau > t_end:
                    # The next event falls outside this interval, so it does
                    # not happen yet. Standard SSA advances the clock to the
                    # interval boundary and stops. Storing the overshot time
                    # instead -- as this did -- let each cell's clock drift
                    # past t_end by a different amount, so cells silently
                    # desynchronised as the simulation ran.
                    t = t_end
                    break
                t += tau

                # Select reaction
                r2 = np.random.random() * a_total
                cumsum = 0.0
                rxn = 0
                for r in range(self._n_reactions):
                    cumsum += a[r]
                    if cumsum >= r2:
                        rxn = r
                        break

                # Fire
                species[cell] += S[rxn]
                np.clip(species[cell], 0, None, out=species[cell])

            times[cell] = t

        self._species = self._backend.xp.asarray(species)
        self._times = self._backend.xp.asarray(times)
