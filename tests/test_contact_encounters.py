"""Killing is a rate per hour, set by measured contact times, not by dt.

The immune module rolled its per-contact kill probability on every step of
contact. At up to 0.8 per roll a cell killed within a step or two of
touching, whatever the step was, so median kill time scaled with dt (0.07 h
at dt = 0.01, 0.35 h at dt = 0.05), and NK cells and macrophages cleared the
demo tumours within the first half hour. The T-cell exhaustion counter
counted those steps against a threshold of five encounters and exhausted a
clone after three minutes on one target.

A contact is now one encounter with one outcome, decided when it begins:
it kills with the per-contact probability, and it lasts a time drawn from
the lineage's measured contact duration -- lethal (contact to death) or
non-lethal (contact to release). The durations come from the library:

    CD8+ T cell, lethal       75 min    human, PMID:23935904 (range midpoint)
    CD8+ T cell, non-lethal   8.5 min   mouse in vivo, PMID:26872694
    NK cell, lethal           19 min    human, PMID:31270246

and an illustrative default for the rest. Cancer cells starving below the
hypoxia threshold die at a rate per hour rather than a probability per step.
"""
from __future__ import annotations

import contextlib
import io
import math

import numpy as np
import pytest

from cognisom.core.event_bus import EventBus, EventTypes
from cognisom.modules.cellular_module import (
    HYPOXIC_CANCER_DEATH_PER_HOUR, CellState, CellularModule,
)
from cognisom.modules.immune_module import (
    DEFAULT_LETHAL_CONTACT_MIN, DEFAULT_NONLETHAL_CONTACT_MIN,
    MINUTES_PER_HOUR, ImmuneModule,
)

TARGET = 7
NK_LETHAL_MIN = 19.0


class _Tissue:
    def __init__(self):
        self.cells = {}

    def ecm_density_at(self, position):
        return 0.0


def _nk_module(**cfg):
    with contextlib.redirect_stdout(io.StringIO()):
        m = ImmuneModule({"n_t_cells": 0, "n_nk_cells": 0, "n_macrophages": 0, **cfg})
    m.event_bus = EventBus()
    m.set_cellular_module(_Tissue())
    nk = m.immune_cells[m.add_immune_cell([100, 100, 50], "NK_cell")]
    return m, nk


def _one_contact(m, nk, dt):
    """Put a fresh MHC-null target on the NK cell and run one contact to
    its end. Returns (hours from contact to resolution, killed?)."""
    m.cellular_module.cells = {TARGET: CellState(
        cell_id=TARGET, cell_type="cancer", phase="G1",
        position=np.array([100, 100, 50], dtype=np.float32), mhc1_expression=0.0)}
    nk.position = np.array([100, 100, 50], dtype=np.float32)
    nk.activated, nk.target_cell_id = True, TARGET
    kills = m.total_kills
    m.update(dt)                                  # the contact begins
    start = m.clock_h
    while nk.contact_ends_at is not None or nk.activated:
        m.update(dt)
        if m.total_kills > kills or not nk.activated:
            break
    return m.clock_h - start, m.total_kills > kills


# ── the kill time is a property of the cells, not of the step ───────────

@pytest.mark.parametrize("dt", [0.01, 0.05])
def test_time_from_contact_to_kill_is_the_measured_contact_at_any_step(dt):
    np.random.seed(1)
    m, nk = _nk_module(nk_kill_probability=1.0)
    times = [_one_contact(m, nk, dt)[0] * MINUTES_PER_HOUR for _ in range(400)]
    # An exponential with the measured mean, resolved at the next step:
    # the step adds dt/2 on average. 400 draws put the mean within ~1 min.
    expected = NK_LETHAL_MIN + dt * MINUTES_PER_HOUR / 2
    assert np.mean(times) == pytest.approx(expected, abs=3.5)


@pytest.mark.parametrize("dt", [0.01, 0.05])
def test_the_kill_probability_is_per_contact_at_any_step(dt):
    np.random.seed(2)
    m, nk = _nk_module(nk_kill_probability=0.3)
    outcomes = [_one_contact(m, nk, dt)[1] for _ in range(1000)]
    assert np.mean(outcomes) == pytest.approx(0.3, abs=0.045)


def test_a_contact_that_does_not_kill_releases_the_cell_after_its_duration():
    np.random.seed(3)
    m, nk = _nk_module(nk_kill_probability=0.0)
    times = [_one_contact(m, nk, 0.01)[0] * MINUTES_PER_HOUR for _ in range(400)]
    assert m.total_kills == 0
    assert nk.activated is False and nk.contact_ends_at is None
    # The NK non-lethal contact has no library value yet: the default.
    assert np.mean(times) == pytest.approx(DEFAULT_NONLETHAL_CONTACT_MIN + 0.3, abs=2.5)


# ── where the durations come from ───────────────────────────────────────

def test_each_contact_time_says_where_it_came_from():
    m, _ = _nk_module()
    src = m.get_state()["contact_time_source"]
    assert src["T_cell"]["lethal_contact_minutes"] == "CD8+ Effector T cell: derived:PMID:23935904"
    assert src["T_cell"]["nonlethal_contact_minutes"] == "CD8+ Effector T cell: estimated:PMID:26872694"
    assert src["NK_cell"]["lethal_contact_minutes"] == "NK cell: measured:PMID:31270246"
    assert src["NK_cell"]["nonlethal_contact_minutes"].startswith("default (illustrative)")
    assert src["macrophage"]["lethal_contact_minutes"].startswith("default (illustrative)")
    assert m._contact_minutes_for("T_cell", True) == 75.0
    assert m._contact_minutes_for("macrophage", True) == DEFAULT_LETHAL_CONTACT_MIN


def test_config_outranks_the_library():
    m, _ = _nk_module(lethal_contact_minutes={"NK_cell": 5.0})
    assert m._contact_minutes_for("NK_cell", True) == 5.0
    assert m.contact_time_source["NK_cell"]["lethal_contact_minutes"] == "config"


# ── exhaustion counts contacts, not steps ───────────────────────────────

def test_a_long_contact_is_one_engagement():
    """The closed-loop engine, one T cell on its target for an hour of
    steps. The clone's encounter count is the number of contacts."""
    from cognisom.core import SimulationConfig, SimulationEngine
    from cognisom.modules.molecular_module import MolecularModule

    np.random.seed(0)
    engine = SimulationEngine(SimulationConfig(dt=0.01, duration=1.0, use_gpu=False))
    engine.register_module("molecular", MolecularModule, {
        "transcription_rate": 0.0, "exosome_release_rate": 0.0, "mutation_rate": 0.0})
    engine.register_module("cellular", CellularModule, {
        "n_normal_cells": 0, "n_cancer_cells": 0,
        "hla_alleles": ["HLA-A*02:01", "HLA-A*24:02", "HLA-B*07:02"]})
    engine.register_module("immune", ImmuneModule, {
        "n_t_cells": 1, "n_nk_cells": 0, "n_macrophages": 0,
        "tcr_recognition_threshold": 0.0, "tcr_repertoire_size": 200, "tcr_seed": 0,
        "costimulation": 0.0, "checkpoint_block": 0.0,        # contacts never kill
        "nonlethal_contact_minutes": {"T_cell": 20.0}})
    with contextlib.redirect_stdout(io.StringIO()):
        engine.initialize()
    mol, cel, imm = (engine.modules[k] for k in ("molecular", "cellular", "immune"))
    cel.set_molecular_module(mol)
    imm.set_cellular_module(cel)
    cancer = cel.add_cell(position=[100.0, 100.0, 50.0], cell_type="cancer")
    cel.cells[cancer].mhc1_expression = 0.9
    mol.add_cell(cancer)
    tcell = next(iter(imm.immune_cells.values()))
    tcell.position = np.array([100.0, 100.0, 50.0], dtype=np.float32)
    imm._tcr_repertoire.exhaustion_threshold = 10_000
    in_contact = []
    original = imm._engage
    imm._engage = lambda *a: (in_contact.append(1), original(*a))
    mol.introduce_mutation(cancer, "KRAS", "G12D")
    with contextlib.redirect_stdout(io.StringIO()):
        engine.run(duration=1.0)
    encounters = sum(imm._tcr_repertoire._encounters.values())
    assert len(in_contact) > 50                       # steps spent in contact
    assert 1 <= encounters <= 5                        # ~60 min / 20 min contacts


# ── starving cancer cells ───────────────────────────────────────────────

@pytest.mark.parametrize("dt", [0.01, 0.05])
def test_a_starving_cancer_cell_dies_at_a_rate_per_hour(dt):
    np.random.seed(4)
    with contextlib.redirect_stdout(io.StringIO()):
        cel = CellularModule({"n_normal_cells": 0, "n_cancer_cells": 600,
                              "initial_cell_cycle": "synchronous"})
        cel.initialize()
    for _ in range(int(round(1.0 / dt))):
        for c in cel.cells.values():
            c.oxygen = 0.0
        cel.update(dt)
    dead = 1 - sum(c.alive for c in cel.cells.values()) / 600
    assert dead == pytest.approx(1 - math.exp(-HYPOXIC_CANCER_DEATH_PER_HOUR), abs=0.06)
