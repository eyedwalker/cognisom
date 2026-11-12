"""Signals between cells: one inventory, and one way to declare a new one.

Why this file exists
--------------------
The engine solved six diffusing signals, each wired by hand across two
modules, with no list of them and no record of which were read by
anything. Vessels, stroma, nerves and new immune cells all communicate by
secreted molecules, so they all need a shared path before they can be
built. ``cognisom/modules/signalling.py`` is that path; these tests hold
it to three things:

* the inventory cannot drift from the code or leave a field out;
* a declared signal really travels: secreted where declared, weaker with
  distance, readable by every cell;
* the plumbing has no side effects -- declaring a signal nothing reads
  must leave a run exactly as it was.
"""

from __future__ import annotations

import contextlib
import hashlib
import io

import numpy as np
import pytest

from cognisom.dashboard.engine_runner import EngineRunner
from cognisom.modules import signalling as sg
from cognisom.modules import spatial_module as sm

# Short-lived on purpose: sqrt(D / k) is about 70 um here, inside the
# 200 um ring, so "weaker with distance" is a property the tissue can show.
# At a half-life of 30 minutes the reach is 360 um and the field is flat.
TEST_SIGNAL = sg.SecretedSignal(
    name="test_factor", diffusion_um2_s=50.0, half_life_hours=0.02,
    uptake_per_hour=50.0,
    secreted_by=(sg.Secretion("tissue", "cancer", rate=1.0),))


def _run(signals=(), hours=1.0, **kw):
    np.random.seed(3)
    runner = EngineRunner(tissue="ring", dt=0.05, duration=hours,
                          overrides={"spatial": {"signals": list(signals)}}, **kw)
    with contextlib.redirect_stdout(io.StringIO()):
        runner.build()
        runner.run()
    return runner


def _fingerprint(runner) -> str:
    series = runner.get_time_series()
    blob = repr([(k, [round(float(x), 9) for x in series[k]])
                 for k in sorted(series) if k != "time"])
    return hashlib.sha256(blob.encode()).hexdigest()


# ── the inventory ────────────────────────────────────────────────────

def test_every_field_the_engine_solves_is_in_the_inventory():
    runner = _run(hours=0.1)
    solved = set(runner.engine.modules["spatial"].fields)
    listed = {entry.name for entry in sg.inventory()}
    assert solved == listed == set(sm.SpatialModule.BUILT_IN_FIELDS), (
        f"solved but unlisted: {sorted(solved - listed)}; "
        f"listed but not solved: {sorted(listed - solved)}")


def test_the_inventorys_numbers_are_the_modules_own():
    rows = {entry.name: entry for entry in sg.inventory()}
    assert rows["ifn_gamma"].diffusion.value == sm.IFN_GAMMA_DIFFUSION_UM2_S
    assert rows["ifn_gamma"].loss.value == sm.IFN_GAMMA_UPTAKE_PER_H
    assert rows["testosterone"].diffusion.value == sm.TESTOSTERONE_DIFFUSION_UM2_S
    assert rows["lactate"].diffusion.value == sm.LACTATE_DIFFUSION_UM2_S
    assert rows["cytokine"].loss.value == sm.CHEMOKINE_HALF_LIFE_HOURS
    runner = _run(hours=0.1)
    fields = runner.engine.modules["spatial"].fields
    for name, row in rows.items():
        assert fields[name].diffusion_coeff == row.diffusion.value, name


def test_a_field_nothing_reads_is_listed_as_not_wired():
    """Oxygen and glucose are solved and consumed by nothing. The
    inventory must say so rather than list them as working signals."""
    rows = {entry.name: entry for entry in sg.inventory()}
    pushed = {name for name, _attr in sm.SpatialModule.FIELDS_TO_CELLS}
    for name in ("oxygen", "glucose"):
        assert name not in pushed
        assert rows[name].wired is False
    for name in ("testosterone", "ifn_gamma", "cytokine"):
        assert rows[name].wired is True


def test_every_number_in_the_inventory_carries_a_status_and_a_source():
    for entry in sg.inventory():
        for quantity in (entry.diffusion, entry.loss):
            if quantity is None:
                continue
            assert quantity.status in sg.STATUSES
            assert quantity.source.strip(), entry.name
        assert entry.made_by and entry.read_by and entry.effect


# ── a declared signal travels ────────────────────────────────────────

@pytest.fixture(scope="module")
def with_signal():
    return _run([TEST_SIGNAL])


def test_a_declared_signal_reaches_every_cell(with_signal):
    cells = [c for c in with_signal.engine.modules["cellular"].cells.values()
             if c.alive]
    assert cells and all("test_factor" in c.signals for c in cells)
    assert max(c.signals["test_factor"] for c in cells) > 0.0


def test_it_is_strongest_where_it_is_made(with_signal):
    cells = [c for c in with_signal.engine.modules["cellular"].cells.values()
             if c.alive]
    cancer = [c for c in cells if c.cell_type == "cancer"]
    normal = [c for c in cells if c.cell_type != "cancer"]
    assert cancer, "the ring preset seeds cancer cells; the test needs them"
    made = np.mean([c.signals["test_factor"] for c in cancer])
    centre = np.mean([c.position for c in cancer], axis=0)
    far = sorted(normal, key=lambda c: -np.linalg.norm(c.position - centre))[:10]
    away = np.mean([c.signals["test_factor"] for c in far])
    assert made > 2.0 * away, (made, away)


def test_an_immune_cell_can_read_it_where_it_stands(with_signal):
    spatial = with_signal.engine.modules["spatial"]
    immune = with_signal.engine.modules["immune"]
    cell = next(iter(immune.immune_cells.values()))
    assert spatial.get_concentration("test_factor", cell.position) >= 0.0


def test_only_the_declared_cells_secrete():
    quiet = sg.SecretedSignal(
        name="nobody_makes_this", diffusion_um2_s=50.0, half_life_hours=0.5,
        secreted_by=(sg.Secretion("tissue", "no_such_cell", rate=1.0),))
    runner = _run([quiet], hours=0.5)
    field = runner.engine.modules["spatial"].fields["nobody_makes_this"]
    assert float(np.max(field.concentration)) == 0.0


def test_immune_cells_secrete_only_while_the_condition_holds():
    rule = sg.Secretion("immune", "T_cell", rate=1.0, when=sg.is_engaged)
    signal = sg.SecretedSignal(name="while_engaged", diffusion_um2_s=50.0,
                               half_life_hours=0.5, secreted_by=(rule,))
    idle = type("Cell", (), {"cell_type": "T_cell", "in_blood": False,
                             "activated": False, "position": np.zeros(3)})()
    busy = type("Cell", (), {"cell_type": "T_cell", "in_blood": False,
                             "activated": True, "position": np.ones(3)})()
    immune = type("Immune", (), {"immune_cells": {1: idle, 2: busy}})()
    sources = signal.sources(None, immune)
    assert len(sources) == 1 and sources[0][1] == 1.0
    assert np.array_equal(sources[0][0], np.ones(3))


# ── and changes nothing it was not asked to ──────────────────────────

def test_a_signal_nothing_reads_leaves_the_run_exactly_as_it_was(with_signal):
    """The plumbing draws no random numbers and touches no cell's fate,
    so a run with an unread signal is the run without it."""
    assert _fingerprint(with_signal) == _fingerprint(_run())


def test_with_nothing_declared_no_cell_carries_a_signal():
    runner = _run(hours=0.2)
    assert not runner.engine.modules["spatial"]._signals
    assert all(c.signals == {} for c in
               runner.engine.modules["cellular"].cells.values())


# ── declarations that would mislead are refused ──────────────────────

def test_a_signal_with_no_loss_is_refused():
    with pytest.raises(ValueError, match="no decay and no uptake"):
        sg.SecretedSignal(name="everywhere", diffusion_um2_s=50.0,
                          secreted_by=())


def test_a_name_that_is_already_a_field_is_refused():
    clash = sg.SecretedSignal(name="ifn_gamma", diffusion_um2_s=50.0,
                              half_life_hours=1.0, secreted_by=())
    with pytest.raises(ValueError, match="already a signal named"):
        sm.SpatialModule({"signals": [clash]})
    with pytest.raises(ValueError, match="already a signal named"):
        sm.SpatialModule({"signals": [TEST_SIGNAL, TEST_SIGNAL]})


def test_a_secretion_names_a_real_population():
    with pytest.raises(ValueError, match="'tissue' or 'immune'"):
        sg.Secretion("stroma", "fibroblast", rate=1.0)


def test_the_reach_follows_diffusion_over_loss():
    """sqrt(D / k): the relation the IFN-gamma field was calibrated on."""
    signal = sg.SecretedSignal(name="reach", diffusion_um2_s=50.0,
                               half_life_hours=np.log(2.0),   # k = 1 per hour
                               secreted_by=())
    assert sg.decay_length_um(signal) == pytest.approx(np.sqrt(50.0 * 3600.0))
    denser = sg.SecretedSignal(name="reach2", diffusion_um2_s=50.0,
                               uptake_per_hour=200.0, secreted_by=())
    assert sg.decay_length_um(denser, 0.39) < sg.decay_length_um(denser, 0.01)


def test_a_run_with_a_declared_signal_survives_a_checkpoint(tmp_path):
    from cognisom.core.checkpoint import load_checkpoint, save_checkpoint
    runner = _run([TEST_SIGNAL], hours=0.3)
    path = str(tmp_path / "run.ckpt")
    with contextlib.redirect_stdout(io.StringIO()):
        save_checkpoint(runner, path)
        resumed = load_checkpoint(path)
    assert "test_factor" in resumed.engine.modules["spatial"].fields
