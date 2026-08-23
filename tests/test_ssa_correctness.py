"""
Stochastic simulation correctness.

The SSA implementations were biased in three independent ways, none of
which produced an error or a visibly wrong trajectory:

  1. Propensities multiplied by the reactant count once per species with
     negative stoichiometry, ignoring the magnitude. ``2A -> B`` was
     computed as ``c*A`` instead of ``c*A(A-1)/2``. Besides the wrong
     rate, this let a dimerisation fire when a single molecule remained
     (``c*1 > 0``), creating matter from nothing.

  2. Direct SSA advanced ``t += tau`` and only then checked whether it
     had passed ``t_end``, storing the overshot time. Every cell's clock
     therefore ran past the interval boundary by a different amount, so
     a population silently desynchronised as the run went on.

  3. Tau-leaping applied each reaction's stoichiometry before computing
     the next reaction's propensity, so reaction *r* was evaluated
     against a state reactions *0..r-1* had already changed. That makes
     the outcome depend on the order reactions happen to be listed in.

These tests check the propensity algebra directly, and check the
resulting dynamics against distributions with closed forms.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.gpu.ssa_kernel import (
    BatchSSA,
    GeneExpressionModel,
    Reaction,
    mass_action_propensities,
)


# ── Propensity algebra ──────────────────────────────────────────────

def _propensity(counts, stoich, rate=1.0):
    species = np.array([counts], dtype=np.int64)
    S = np.array([stoich], dtype=np.int64)
    return mass_action_propensities(species, np.array([[rate]]), S)[0, 0]


@pytest.mark.parametrize("a_count,expected", [
    (0, 0.0), (1, 0.0), (2, 1.0), (3, 3.0), (5, 10.0), (10, 45.0),
])
def test_dimerisation_uses_the_combinatorial_factor(a_count, expected):
    """2A -> B has propensity c*A(A-1)/2, not c*A."""
    assert _propensity([a_count, 0], [-2, 1]) == pytest.approx(expected)


def test_a_dimerisation_cannot_fire_on_a_single_molecule():
    """The specific regression: C(1,2) is 0, but the old code gave 1.0."""
    assert _propensity([1, 0], [-2, 1]) == 0.0


def test_trimerisation_uses_the_binomial_coefficient():
    assert _propensity([5, 0], [-3, 1]) == pytest.approx(math.comb(5, 3))
    assert _propensity([2, 0], [-3, 1]) == 0.0


def test_first_order_reactions_are_unchanged():
    """The old code was correct where every stoichiometry is -1."""
    assert _propensity([7, 0], [-1, 1], rate=2.0) == pytest.approx(14.0)
    # A + B -> C
    assert _propensity([4, 5, 0], [-1, -1, 1], rate=2.0) == pytest.approx(40.0)


def test_products_and_catalysts_do_not_enter_the_propensity():
    """Only species with negative stoichiometry are consumed."""
    # Gene -> Gene + mRNA: gene is a catalyst (stoich 0), so the propensity
    # is the bare rate constant.
    assert _propensity([1, 0], [0, 1], rate=3.0) == pytest.approx(3.0)


def test_propensities_are_computed_for_every_cell_independently():
    species = np.array([[2, 0], [5, 0], [10, 0]])
    rates = np.ones((3, 1))
    S = np.array([[-2, 1]])
    got = mass_action_propensities(species, rates, S)[:, 0]
    assert got == pytest.approx([1.0, 10.0, 45.0])


# ── Dynamics against closed forms ───────────────────────────────────

def _death_process(n_cells=4000, a0=50, k=0.5, seed=7):
    """A -> 0. Each molecule decays independently at rate k."""
    model = GeneExpressionModel(
        species_names=["A"],
        initial_counts={"A": a0},
        reactions=[Reaction(name="decay", stoichiometry={"A": -1},
                            rate_constant=k)],
    )
    return BatchSSA(model, n_cells=n_cells, seed=seed)


@pytest.mark.parametrize("method", ["tau", "direct"])
def test_death_process_matches_the_analytic_mean(method):
    """N(t) is Binomial(N0, exp(-kt)); the mean is N0*exp(-kt)."""
    np.random.seed(0)
    a0, k, t = 50, 0.5, 1.0
    sim = _death_process(a0=a0, k=k)

    steps = 200
    for _ in range(steps):
        sim.advance(t / steps, method=method)

    observed = sim.get_species("A").mean()
    expected = a0 * math.exp(-k * t)
    assert observed == pytest.approx(expected, rel=0.03)


def test_death_process_matches_the_analytic_variance():
    """Binomial variance N0*p*(1-p) — a mean-only check misses bias."""
    np.random.seed(1)
    a0, k, t = 50, 0.5, 1.0
    sim = _death_process(a0=a0, k=k, n_cells=6000)

    steps = 200
    for _ in range(steps):
        sim.advance(t / steps, method="tau")

    p = math.exp(-k * t)
    observed = sim.get_species("A").var()
    expected = a0 * p * (1 - p)
    assert observed == pytest.approx(expected, rel=0.15)


def test_counts_never_go_negative():
    np.random.seed(2)
    sim = _death_process(a0=5, k=5.0, n_cells=500)
    for _ in range(50):
        sim.advance(0.05, method="tau")
        assert sim.get_species("A").min() >= 0


# ── Clock discipline ────────────────────────────────────────────────

def test_direct_ssa_clock_does_not_overshoot_the_interval():
    """Every cell must land exactly on the interval boundary.

    The old loop stored the time of the first event *past* t_end, so
    cells drifted forward by differing amounts and the population lost a
    common clock.
    """
    np.random.seed(3)
    sim = _death_process(a0=100, k=10.0, n_cells=200)

    dt, n_steps = 0.01, 20
    for _ in range(n_steps):
        sim.advance(dt, method="direct")

    times = np.asarray(sim._backend.to_numpy(sim._times))
    assert times == pytest.approx(dt * n_steps, abs=1e-5)
    assert times.max() == pytest.approx(times.min(), abs=1e-6)


# ── Order independence ──────────────────────────────────────────────

def test_tau_leap_is_independent_of_reaction_ordering():
    """A leap evaluates every propensity at the start-of-leap state.

    With the old sequential update, listing the same reactions in a
    different order gave a different answer.
    """
    decay = Reaction(name="decay", stoichiometry={"A": -1}, rate_constant=0.4)
    convert = Reaction(name="convert", stoichiometry={"A": -1, "B": 1},
                       rate_constant=0.6)

    def run(reactions, seed):
        model = GeneExpressionModel(
            species_names=["A", "B"],
            initial_counts={"A": 200, "B": 0},
            reactions=list(reactions),
        )
        sim = BatchSSA(model, n_cells=3000, seed=seed)
        np.random.seed(seed)
        for _ in range(100):
            sim.advance(0.01, method="tau")
        return sim.get_species("A").mean(), sim.get_species("B").mean()

    forward = run([decay, convert], seed=11)
    reversed_ = run([convert, decay], seed=11)

    assert forward[0] == pytest.approx(reversed_[0], rel=0.05)
    assert forward[1] == pytest.approx(reversed_[1], rel=0.05)


def test_conversion_conserves_total_molecules():
    """A -> B creates nothing and destroys nothing."""
    np.random.seed(4)
    model = GeneExpressionModel(
        species_names=["A", "B"],
        initial_counts={"A": 100, "B": 0},
        reactions=[Reaction(name="convert", stoichiometry={"A": -1, "B": 1},
                            rate_constant=1.0)],
    )
    sim = BatchSSA(model, n_cells=1000, seed=5)
    for _ in range(100):
        sim.advance(0.01, method="tau")

    total = sim.get_species("A") + sim.get_species("B")
    assert total.min() == 100 and total.max() == 100
