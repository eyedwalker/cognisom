"""First-order decay, and the mRNA pool that had no bound.

Three defects, all of them about ``dt``:

  * ``IntracellularModel.degrade_mrna`` and ``degrade_protein`` used the
    LINEARISED probability ``ln2/T * dt`` where the exact one is
    ``1 - exp(-dt*ln2/T)``. At the default mRNA half-life of 2 h this
    over-degrades by 0.9% at dt = 0.05 h, 18.3% at dt = 1.0 h and 38.6%
    at dt = 2.0 h; compounded over 24 h at dt = 1.0 h it leaves 0.15x
    the correct amount. Worse, the linearised value EXCEEDS 1 once
    dt > T/ln2, and ``np.random.binomial`` then raises ValueError rather
    than saturating -- 2.885 h for mRNA, 14.43 h for protein, both
    timesteps a user is free to choose.

  * ``NucleicAcid.update(dt)`` took ``dt`` but used it only to age the
    molecule: degradation was a flat 10% chance per CALL, so the
    effective half-life was a property of how often the caller happened
    to call. It also degraded only once ``age > half_life``, which is not
    what a half-life means.

  * ``MolecularModule`` appended a full-length RNA object per
    transcription to a per-cell list that nothing decayed or trimmed,
    with exactly one consumer that reads index [0] and nothing else.
    Measured over a 24 h Baseline run: 6435 retained objects across 188
    cells, median 16 per cell, max 93, ~3.9 MB of sequence strings.

Plus one namespace trap: ``molecular/mutations.py`` held a ``Mutation``
class with a ``pass`` body that SHADOWED the real ``@dataclass Mutation``
in ``nucleic_acids.py``, so importing ``Mutation`` from the package
returned the empty placeholder.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.engine.py.intracellular import IntracellularModel, Protein, mRNA
from cognisom.engine.py.molecular.nucleic_acids import RNA


LN2 = float(np.log(2.0))


# ── The exact probability ───────────────────────────────────────────

@pytest.mark.parametrize("dt,half_life", [
    (3.0, 2.0),      # just past mRNA's 2.885 h linearised-p-exceeds-1 point
    (5.0, 2.0),
    (100.0, 2.0),
])
def test_degrade_mrna_no_longer_raises_at_a_large_step(dt, half_life):
    """`ln2/2.0 * 3.0` = 1.04, and np.random.binomial rejects p > 1."""
    model = IntracellularModel()
    model.mrnas["TP53"] = mRNA(gene_name="TP53", sequence_length=1000,
                               copy_number=100, half_life=half_life)
    model.degrade_mrna(dt)          # must not raise
    remaining = model.mrnas["TP53"].copy_number if "TP53" in model.mrnas else 0
    assert 0 <= remaining <= 100


@pytest.mark.parametrize("dt", [20.0, 50.0, 1000.0])
def test_degrade_protein_no_longer_raises_at_a_large_step(dt):
    """Same defect against the default protein half-life of 10 h, which
    put the threshold at 14.43 h."""
    model = IntracellularModel()
    model.proteins["P"] = Protein(name="P", gene_source="P",
                                  copy_number=100, half_life=10.0)
    model.degrade_protein(dt)       # must not raise
    remaining = model.proteins["P"].copy_number if "P" in model.proteins else 0
    assert 0 <= remaining <= 100


def test_mrna_decay_matches_the_analytic_survival_fraction():
    """A population with half-life T must retain 2^(-dt/T) of itself over
    an interval dt. At dt = 2.0 h and T = 2.0 h that is exactly a half;
    the linearised scheme removed 69.3% instead of 50%.
    """
    rng = np.random.default_rng(7)
    kept = []
    for _ in range(400):
        model = IntracellularModel()
        model.mrnas["G"] = mRNA(gene_name="G", sequence_length=100,
                                copy_number=2000, half_life=2.0)
        np.random.seed(int(rng.integers(0, 2 ** 31 - 1)))
        model.degrade_mrna(2.0)
        kept.append(model.mrnas["G"].copy_number / 2000.0
                    if "G" in model.mrnas else 0.0)
    assert float(np.mean(kept)) == pytest.approx(0.5, abs=0.01)


def test_mrna_decay_is_independent_of_how_the_step_is_split():
    """The defining property of a dt-correct scheme: one step of 1.0 h
    must agree with two of 0.5 h. The linearised form does not -- it
    removes 34.66% in one step and 31.66% in two.
    """
    def survival(n_steps, dt):
        rng = np.random.default_rng(11)
        kept = []
        for _ in range(300):
            model = IntracellularModel()
            model.mrnas["G"] = mRNA(gene_name="G", sequence_length=100,
                                    copy_number=4000, half_life=2.0)
            np.random.seed(int(rng.integers(0, 2 ** 31 - 1)))
            for _ in range(n_steps):
                model.degrade_mrna(dt)
            kept.append(model.mrnas["G"].copy_number / 4000.0
                        if "G" in model.mrnas else 0.0)
        return float(np.mean(kept))

    one = survival(1, 1.0)
    two = survival(2, 0.5)
    four = survival(4, 0.25)
    exact = 2.0 ** (-1.0 / 2.0)
    assert one == pytest.approx(exact, abs=0.01)
    assert two == pytest.approx(exact, abs=0.01)
    assert four == pytest.approx(exact, abs=0.01)


# ── NucleicAcid.update ──────────────────────────────────────────────

def test_nucleic_acid_update_decay_depends_on_dt():
    """It was a flat 10% chance per call, whatever dt was, and it only
    applied once age exceeded the half-life."""
    def survival(dt, n_steps, seed):
        np.random.seed(seed)
        alive = 0
        trials = 2000
        for _ in range(trials):
            rna = RNA("AUGGCCUAA" * 10, name="G")
            rna.half_life = 2.0
            rna.age = 0.0
            ok = True
            for _ in range(n_steps):
                if not rna.update(dt):
                    ok = False
                    break
            alive += int(ok)
        return alive / trials

    # Half of a 2 h-half-life population is gone after 2 h, however the
    # interval is divided.
    assert survival(2.0, 1, 3) == pytest.approx(0.5, abs=0.04)
    assert survival(1.0, 2, 5) == pytest.approx(0.5, abs=0.04)
    assert survival(0.5, 4, 9) == pytest.approx(0.5, abs=0.04)
    # A larger step must degrade more, not the same.
    assert survival(4.0, 1, 13) < survival(1.0, 1, 13)


def test_nucleic_acid_can_degrade_before_reaching_its_half_life():
    """`if self.age > self.half_life` meant a molecule was immortal for
    its first half-life, which is not what an exponential lifetime is."""
    np.random.seed(17)
    survived = 0
    for _ in range(2000):
        rna = RNA("AUGGCCUAA", name="G")
        rna.half_life = 2.0
        rna.age = 0.0
        if rna.update(0.5):
            survived += 1
    # 1 - exp(-0.5*ln2/2) = 15.9% should have gone.
    assert survived / 2000.0 == pytest.approx(1 - 0.159, abs=0.04)


# ── The unbounded pool ──────────────────────────────────────────────

def test_the_per_cell_mrna_pool_is_bounded():
    from cognisom.modules.molecular_module import MolecularModule

    module = MolecularModule({})
    assert module.MAX_MRNA_PER_CELL > 0


def test_a_full_run_keeps_the_mrna_pool_bounded():
    """6435 retained objects before; bounded by cells x MAX_MRNA_PER_CELL
    after, with the consumer still finding a transcript at index [0]."""
    from cognisom.dashboard.engine_runner import EngineRunner

    np.random.seed(12345)
    runner = EngineRunner(dt=0.05, duration=6.0, scenario="Baseline",
                          record_interval=20)
    runner.build()
    runner.run()
    molecular = runner.engine.modules["molecular"]
    cap = molecular.MAX_MRNA_PER_CELL

    sizes = [len(v) for v in molecular.cell_mrnas.values()]
    assert sizes, "no cells were tracked at all"
    assert max(sizes) <= cap
    assert molecular.total_transcriptions > 0
    # Eviction happened, i.e. the bound is actually load-bearing here.
    assert molecular.total_mrna_evicted > 0
    # The one consumer still has something to read.
    assert any(v for v in molecular.cell_mrnas.values())


# ── The shadowed dataclass ──────────────────────────────────────────

def test_the_package_exports_the_real_mutation_dataclass():
    """`from .mutations import Mutation` shadowed the real
    `@dataclass Mutation` in nucleic_acids with a `pass` body."""
    import dataclasses

    from cognisom.engine.py.molecular import Mutation
    from cognisom.engine.py.molecular.nucleic_acids import (
        Mutation as RealMutation,
    )

    assert Mutation is RealMutation
    assert dataclasses.is_dataclass(Mutation)


def test_the_placeholder_modules_are_gone():
    import importlib

    for name in ("proteins", "mutations"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"cognisom.engine.py.molecular.{name}")


# ── start() no longer discards set_mode() ───────────────────────────

def test_start_keeps_a_mode_set_beforehand():
    """`start(mode=REALTIME)` assigned unconditionally, so set_mode(FAST)
    followed by start() ran at REALTIME's 60 fps cap."""
    from cognisom.omniverse.realtime_sim import (
        RealtimeSimulation, SimulationMode,
    )

    sim = RealtimeSimulation(engine=None)
    sim.set_mode(SimulationMode.FAST)
    assert sim.mode == SimulationMode.FAST
    try:
        sim.start()
        assert sim.mode == SimulationMode.FAST
    finally:
        sim.stop()


def test_start_still_honours_an_explicit_mode():
    from cognisom.omniverse.realtime_sim import (
        RealtimeSimulation, SimulationMode,
    )

    sim = RealtimeSimulation(engine=None)
    sim.set_mode(SimulationMode.FAST)
    try:
        sim.start(mode=SimulationMode.STEPPED)
        assert sim.mode == SimulationMode.STEPPED
    finally:
        sim.stop()
