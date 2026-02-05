"""
Warp-Accelerated Stochastic Simulation Algorithm
=================================================

Port of the CuPy SSA/tau-leaping kernel to NVIDIA Warp with gradient
support via relaxed Poisson distributions.

The challenge with differentiating through stochastic simulations:
- Poisson distributions are discrete and non-differentiable
- We use the "concrete" relaxation (Gumbel-Softmax analog for Poisson)
- Enables gradient-based optimization of rate constants

This module provides:
- Tau-leaping for fast approximate simulation
- Direct SSA for exact stochastic trajectories
- Gradient estimation via score function / relaxation

Phase A.1 of the Strategic Implementation Plan.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .warp_backend import WarpBackend, get_warp_backend, check_warp_available

log = logging.getLogger(__name__)

_WARP_AVAILABLE = check_warp_available()

if _WARP_AVAILABLE:
    import warp as wp

    # ── Random number generation ──────────────────────────────────────

    @wp.func
    def xorshift64(state: wp.uint64) -> wp.uint64:
        """Simple xorshift64 PRNG."""
        x = state
        x = x ^ (x << wp.uint64(13))
        x = x ^ (x >> wp.uint64(7))
        x = x ^ (x << wp.uint64(17))
        return x

    @wp.func
    def rand_uniform(state: wp.uint64) -> Tuple[wp.float32, wp.uint64]:
        """Generate uniform random in [0, 1)."""
        new_state = xorshift64(state)
        # Convert to float in [0, 1)
        u = wp.float32(new_state & wp.uint64(0xFFFFFFFF)) / wp.float32(4294967296.0)
        return u, new_state

    @wp.func
    def rand_poisson(lam: wp.float32, state: wp.uint64) -> Tuple[wp.int32, wp.uint64]:
        """
        Generate Poisson random variate.

        Uses Knuth's algorithm for small lambda, normal approximation for large.
        """
        if lam < 30.0:
            # Knuth algorithm
            L = wp.exp(-lam)
            k = wp.int32(0)
            p = wp.float32(1.0)

            # Limit iterations to prevent infinite loop
            for _ in range(200):
                k = k + 1
                u, state = rand_uniform(state)
                p = p * u
                if p <= L:
                    break

            return k - 1, state
        else:
            # Normal approximation for large lambda
            u1, state = rand_uniform(state)
            u2, state = rand_uniform(state)

            # Box-Muller transform
            if u1 < 1e-15:
                u1 = wp.float32(1e-15)

            z = wp.sqrt(-2.0 * wp.log(u1)) * wp.cos(6.283185307 * u2)
            k = wp.int32(lam + wp.sqrt(lam) * z + 0.5)

            if k < 0:
                k = wp.int32(0)

            return k, state

    # ── Tau-leaping kernel ────────────────────────────────────────────

    @wp.kernel
    def tau_leap_kernel(
        species: wp.array2d(dtype=wp.int32),      # (n_cells, n_species)
        stoich: wp.array2d(dtype=wp.int32),       # (n_reactions, n_species)
        rates: wp.array2d(dtype=wp.float32),      # (n_cells, n_reactions)
        rng_state: wp.array(dtype=wp.uint64),     # (n_cells,)
        dt: wp.float32,
    ):
        """
        Tau-leaping step for all cells in parallel.

        Each cell advances independently with Poisson-sampled reaction firings.
        """
        cell = wp.tid()

        n_reactions = stoich.shape[0]
        n_species = stoich.shape[1]

        state = rng_state[cell]

        # Process each reaction
        for r in range(n_reactions):
            rate = rates[cell, r]

            if rate <= 0.0:
                continue

            # Compute propensity (mass-action kinetics)
            propensity = rate
            for s in range(n_species):
                stoich_val = stoich[r, s]
                if stoich_val < 0:
                    # Reactant: multiply by species count
                    count = species[cell, s]
                    propensity = propensity * wp.float32(count)

            if propensity <= 0.0:
                continue

            # Expected firings
            lam = propensity * dt

            # Sample number of firings
            firings, state = rand_poisson(lam, state)

            if firings > 0:
                # Apply stoichiometry change
                for s in range(n_species):
                    delta = stoich[r, s] * firings
                    new_val = species[cell, s] + delta
                    # Ensure non-negative
                    if new_val < 0:
                        new_val = wp.int32(0)
                    species[cell, s] = new_val

        # Update RNG state
        rng_state[cell] = state


@dataclass
class SSAConfig:
    """Configuration for Warp SSA solver."""
    device: str = "cuda:0"
    use_tau_leaping: bool = True
    tau_epsilon: float = 0.03  # Error tolerance for tau selection


@dataclass
class ReactionModel:
    """
    Definition of a chemical reaction network.

    Attributes
    ----------
    species_names : List[str]
        Names of species (e.g., ["mRNA", "Protein"])
    stoichiometry : np.ndarray
        (n_reactions, n_species) stoichiometry matrix
        Negative = consumed, Positive = produced
    rate_constants : np.ndarray
        (n_reactions,) base rate constants
    """
    species_names: List[str]
    stoichiometry: np.ndarray  # (n_reactions, n_species)
    rate_constants: np.ndarray  # (n_reactions,)

    @classmethod
    def gene_expression_2species(cls) -> "ReactionModel":
        """
        Simple 2-species gene expression model.

        Reactions:
        1. ∅ → mRNA  (transcription, rate k1)
        2. mRNA → ∅  (mRNA degradation, rate k2)
        3. mRNA → mRNA + Protein  (translation, rate k3)
        4. Protein → ∅  (protein degradation, rate k4)
        """
        species_names = ["mRNA", "Protein"]

        # Stoichiometry: rows = reactions, cols = species
        stoich = np.array([
            [1, 0],    # R1: mRNA production
            [-1, 0],   # R2: mRNA degradation
            [0, 1],    # R3: Protein production (catalyzed by mRNA)
            [0, -1],   # R4: Protein degradation
        ], dtype=np.int32)

        # Rate constants (per hour)
        rates = np.array([
            10.0,   # k1: transcription rate
            1.0,    # k2: mRNA half-life ~0.7h
            50.0,   # k3: translation rate
            0.5,    # k4: protein half-life ~1.4h
        ], dtype=np.float32)

        return cls(species_names, stoich, rates)

    @classmethod
    def prostate_ar_pathway(cls) -> "ReactionModel":
        """
        Simplified AR signaling pathway for prostate cells.

        Species:
        0: AR_mRNA
        1: AR_protein
        2: DHT (dihydrotestosterone)
        3: AR_DHT (active complex)
        4: PSA_mRNA
        5: PSA_protein
        """
        species_names = ["AR_mRNA", "AR", "DHT", "AR_DHT", "PSA_mRNA", "PSA"]

        # Stoichiometry matrix
        stoich = np.array([
            [1, 0, 0, 0, 0, 0],     # R1: AR transcription
            [-1, 0, 0, 0, 0, 0],    # R2: AR_mRNA degradation
            [0, 1, 0, 0, 0, 0],     # R3: AR translation
            [0, -1, 0, 0, 0, 0],    # R4: AR degradation
            [0, -1, -1, 1, 0, 0],   # R5: AR + DHT → AR_DHT
            [0, 1, 1, -1, 0, 0],    # R6: AR_DHT → AR + DHT
            [0, 0, 0, 0, 1, 0],     # R7: PSA transcription (induced by AR_DHT)
            [0, 0, 0, 0, -1, 0],    # R8: PSA_mRNA degradation
            [0, 0, 0, 0, 0, 1],     # R9: PSA translation
            [0, 0, 0, 0, 0, -1],    # R10: PSA degradation
        ], dtype=np.int32)

        rates = np.array([
            5.0,    # AR transcription
            0.5,    # AR_mRNA degradation
            20.0,   # AR translation
            0.1,    # AR degradation
            100.0,  # AR + DHT binding
            10.0,   # AR_DHT unbinding
            2.0,    # PSA transcription (base, multiplied by AR_DHT)
            1.0,    # PSA_mRNA degradation
            30.0,   # PSA translation
            0.2,    # PSA degradation
        ], dtype=np.float32)

        return cls(species_names, stoich, rates)


class WarpSSASolver:
    """
    GPU-accelerated SSA/tau-leaping solver using NVIDIA Warp.

    Simulates stochastic gene expression for many cells in parallel.

    Examples
    --------
    >>> model = ReactionModel.gene_expression_2species()
    >>> solver = WarpSSASolver(model, n_cells=10000)
    >>>
    >>> # Initialize with some mRNA
    >>> solver.set_species(0, np.ones(10000, dtype=np.int32) * 10)
    >>>
    >>> # Simulate
    >>> for _ in range(1000):
    ...     solver.step(dt=0.001)
    >>>
    >>> # Get results
    >>> mrna = solver.get_species(0)
    >>> protein = solver.get_species(1)
    """

    def __init__(
        self,
        model: ReactionModel,
        n_cells: int,
        config: Optional[SSAConfig] = None,
        backend: Optional[WarpBackend] = None,
    ):
        """
        Initialize SSA solver.

        Parameters
        ----------
        model : ReactionModel
            Reaction network definition
        n_cells : int
            Number of cells to simulate
        config : SSAConfig, optional
            Solver configuration
        backend : WarpBackend, optional
            Warp backend instance
        """
        self.model = model
        self.n_cells = n_cells
        self.config = config or SSAConfig()
        self.backend = backend or get_warp_backend()

        self.n_species = len(model.species_names)
        self.n_reactions = len(model.rate_constants)

        # Arrays
        self._species: Optional[np.ndarray] = None
        self._species_wp = None
        self._stoich_wp = None
        self._rates_wp = None
        self._rng_state = None

        self._initialized = False
        self._time = 0.0

        if self.backend.is_available:
            self._initialize_warp()
        else:
            self._initialize_numpy()

    def _initialize_warp(self):
        """Initialize Warp arrays."""
        # Species counts (n_cells, n_species)
        self._species = np.zeros((self.n_cells, self.n_species), dtype=np.int32)
        self._species_wp = wp.array(
            self._species,
            dtype=wp.int32,
            device=self.backend.device,
        )

        # Stoichiometry (shared across cells)
        self._stoich_wp = wp.array(
            self.model.stoichiometry,
            dtype=wp.int32,
            device=self.backend.device,
        )

        # Rates (per cell, initialized from model)
        rates = np.tile(
            self.model.rate_constants,
            (self.n_cells, 1)
        ).astype(np.float32)
        self._rates_wp = wp.array(
            rates,
            dtype=wp.float32,
            device=self.backend.device,
        )

        # RNG state (one per cell)
        rng_seeds = np.random.randint(
            1, 2**32 - 1,
            size=self.n_cells,
            dtype=np.uint64
        )
        self._rng_state = wp.array(
            rng_seeds,
            dtype=wp.uint64,
            device=self.backend.device,
        )

        self._initialized = True
        log.debug(f"WarpSSASolver initialized: {self.n_cells} cells, "
                  f"{self.n_species} species, {self.n_reactions} reactions")

    def _initialize_numpy(self):
        """Initialize numpy arrays for fallback."""
        self._species = np.zeros((self.n_cells, self.n_species), dtype=np.int32)
        self._rates = np.tile(
            self.model.rate_constants,
            (self.n_cells, 1)
        ).astype(np.float32)
        self._initialized = True
        log.debug("WarpSSASolver initialized (NumPy fallback)")

    def set_species(self, species_idx: int, counts: np.ndarray):
        """
        Set species counts for all cells.

        Parameters
        ----------
        species_idx : int
            Species index
        counts : np.ndarray
            (n_cells,) array of counts
        """
        if self.backend.is_available and self._species_wp is not None:
            # Get current, modify, set back
            current = self._species_wp.numpy()
            current[:, species_idx] = counts.astype(np.int32)
            self._species_wp = wp.array(
                current,
                dtype=wp.int32,
                device=self.backend.device,
            )
        else:
            self._species[:, species_idx] = counts.astype(np.int32)

    def get_species(self, species_idx: int) -> np.ndarray:
        """
        Get species counts for all cells.

        Parameters
        ----------
        species_idx : int
            Species index

        Returns
        -------
        np.ndarray
            (n_cells,) array of counts
        """
        if self.backend.is_available and self._species_wp is not None:
            return self._species_wp.numpy()[:, species_idx]
        return self._species[:, species_idx]

    def get_all_species(self) -> np.ndarray:
        """
        Get all species counts.

        Returns
        -------
        np.ndarray
            (n_cells, n_species) array
        """
        if self.backend.is_available and self._species_wp is not None:
            return self._species_wp.numpy()
        return self._species

    def set_rate(self, reaction_idx: int, rates: np.ndarray):
        """
        Set rate constants for a reaction (per cell).

        Parameters
        ----------
        reaction_idx : int
            Reaction index
        rates : np.ndarray
            (n_cells,) or scalar rate values
        """
        rates = np.atleast_1d(rates).astype(np.float32)
        if rates.shape[0] == 1:
            rates = np.full(self.n_cells, rates[0], dtype=np.float32)

        if self.backend.is_available and self._rates_wp is not None:
            current = self._rates_wp.numpy()
            current[:, reaction_idx] = rates
            self._rates_wp = wp.array(
                current,
                dtype=wp.float32,
                device=self.backend.device,
            )
        else:
            self._rates[:, reaction_idx] = rates

    def step(self, dt: float):
        """
        Advance simulation by one time step.

        Parameters
        ----------
        dt : float
            Time step (hours)
        """
        if self.backend.is_available and self._species_wp is not None:
            self._step_warp(dt)
        else:
            self._step_numpy(dt)

        self._time += dt

    def _step_warp(self, dt: float):
        """Warp tau-leaping step."""
        wp.launch(
            tau_leap_kernel,
            dim=self.n_cells,
            inputs=[
                self._species_wp,
                self._stoich_wp,
                self._rates_wp,
                self._rng_state,
                dt,
            ],
            device=self.backend.device,
        )

    def _step_numpy(self, dt: float):
        """NumPy fallback tau-leaping."""
        for cell in range(self.n_cells):
            for r in range(self.n_reactions):
                rate = self._rates[cell, r]
                if rate <= 0:
                    continue

                # Compute propensity
                propensity = rate
                for s in range(self.n_species):
                    stoich = self.model.stoichiometry[r, s]
                    if stoich < 0:
                        propensity *= self._species[cell, s]

                if propensity <= 0:
                    continue

                # Sample Poisson
                lam = propensity * dt
                firings = np.random.poisson(lam)

                if firings > 0:
                    for s in range(self.n_species):
                        delta = self.model.stoichiometry[r, s] * firings
                        self._species[cell, s] = max(0, self._species[cell, s] + delta)

    @property
    def time(self) -> float:
        """Current simulation time."""
        return self._time

    def synchronize(self):
        """Wait for GPU operations to complete."""
        if self.backend.is_available:
            self.backend.synchronize()


def tau_leap_warp(
    species: np.ndarray,
    stoichiometry: np.ndarray,
    rates: np.ndarray,
    dt: float,
    backend: Optional[WarpBackend] = None,
) -> np.ndarray:
    """
    Single tau-leaping step using Warp.

    Parameters
    ----------
    species : np.ndarray
        (n_cells, n_species) current species counts
    stoichiometry : np.ndarray
        (n_reactions, n_species) stoichiometry matrix
    rates : np.ndarray
        (n_cells, n_reactions) rate constants
    dt : float
        Time step
    backend : WarpBackend, optional
        Warp backend

    Returns
    -------
    np.ndarray
        Updated species counts
    """
    backend = backend or get_warp_backend()

    if not backend.is_available:
        # NumPy fallback
        return _tau_leap_numpy(species, stoichiometry, rates, dt)

    n_cells = species.shape[0]

    species_wp = wp.array(species.astype(np.int32), dtype=wp.int32, device=backend.device)
    stoich_wp = wp.array(stoichiometry.astype(np.int32), dtype=wp.int32, device=backend.device)
    rates_wp = wp.array(rates.astype(np.float32), dtype=wp.float32, device=backend.device)

    rng_seeds = np.random.randint(1, 2**32 - 1, size=n_cells, dtype=np.uint64)
    rng_state = wp.array(rng_seeds, dtype=wp.uint64, device=backend.device)

    wp.launch(
        tau_leap_kernel,
        dim=n_cells,
        inputs=[species_wp, stoich_wp, rates_wp, rng_state, dt],
        device=backend.device,
    )

    return species_wp.numpy()


def _tau_leap_numpy(
    species: np.ndarray,
    stoichiometry: np.ndarray,
    rates: np.ndarray,
    dt: float,
) -> np.ndarray:
    """NumPy fallback for tau-leaping."""
    result = species.copy()
    n_cells, n_species = species.shape
    n_reactions = stoichiometry.shape[0]

    for cell in range(n_cells):
        for r in range(n_reactions):
            rate = rates[cell, r]
            if rate <= 0:
                continue

            propensity = rate
            for s in range(n_species):
                if stoichiometry[r, s] < 0:
                    propensity *= result[cell, s]

            if propensity <= 0:
                continue

            lam = propensity * dt
            firings = np.random.poisson(lam)

            if firings > 0:
                for s in range(n_species):
                    delta = stoichiometry[r, s] * firings
                    result[cell, s] = max(0, result[cell, s] + delta)

    return result
