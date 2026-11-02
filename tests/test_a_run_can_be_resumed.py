"""A resumed run must be indistinguishable from an uninterrupted one.

This file is the specification of :mod:`cognisom.core.checkpoint`. A
checkpoint is only worth having if restarting from it changes nothing, and
the failure mode that matters is not a crash -- it is a restored run that
looks entirely plausible and has silently lost some piece of state. So the
test does not check that a checkpoint can be written and read; it runs the
same simulation twice, once straight through and once interrupted, and
demands the two agree bit for bit.

``EngineRunner.snapshot_interval_hours`` is not this. Those snapshots are
cell positions for the 3D view and carry none of the module state that
produced them, which is why item 4 of the realism plan lists "no
checkpoint/restart" as outstanding even though that parameter exists.

The observables below are deliberately not a summary. They include the
diffusion field sums, because the fields were the one piece of state that
the checkpoint's "pickle what pickles" rule would have dropped (a cached
scipy LU handle made the whole field refuse), and the RNG stream position,
because the engine draws from the shared global stream -- restore every
cell perfectly and leave the stream where ``build`` left it, and the run
diverges on its next draw with nothing to show for it.
"""

import contextlib
import io
import os

import numpy as np
import pytest

from cognisom.core.checkpoint import (
    CheckpointError,
    FORMAT_VERSION,
    load_checkpoint,
    save_checkpoint,
)
from cognisom.dashboard.engine_runner import EngineRunner

HALF_HOURS = 2.0


def _observables(runner):
    """Everything a resumed run could plausibly get wrong."""
    engine = runner.engine
    cellular = engine.modules['cellular']
    immune = engine.modules['immune']
    vascular = engine.modules['vascular']
    ar = engine.modules['ar_signaling']
    spatial = engine.modules['spatial']
    alive = [c for c in cellular.cells.values() if c.alive]
    return {
        'time': round(float(engine.time), 9),
        'step_count': int(engine.step_count),
        'n_cells': len(cellular.cells),
        'n_alive': len(alive),
        'n_cancer': sum(1 for c in alive if c.cell_type == 'cancer'),
        'divisions': cellular.total_divisions,
        'deaths': cellular.total_deaths,
        'kills': immune.total_kills,
        'tcell_kills': immune.total_tcell_kills,
        'O2_delivered': repr(float(vascular.total_O2_delivered)),
        'hypoxic_regions': vascular.hypoxic_regions,
        'ar_rows': len(ar.ids),
        'ar_Y_sum': repr(float(np.sum(ar.Y))),
        'mean_mhc1': repr(float(np.mean([c.mhc1_expression for c in alive]))),
        'mean_O2': repr(float(np.mean([c.oxygen for c in alive]))),
        'fields': {name: repr(float(np.sum(field.concentration)))
                   for name, field in spatial.fields.items()},
        # Read the stream POSITION rather than drawing from it. Drawing
        # here would advance the stream and offset a checkpoint taken
        # afterwards by one value, which looks exactly like a lost-state
        # bug and is not one.
        'rng_pos': int(np.random.get_state()[2]),
    }


def _built(duration, seed=1):
    np.random.seed(seed)
    runner = EngineRunner(dt=0.05, duration=duration, scenario='Baseline',
                          tissue='slab', record_interval=100000,
                          snapshot_interval_hours=None)
    with contextlib.redirect_stdout(io.StringIO()):
        runner.build()
    return runner


def _ran(runner):
    with contextlib.redirect_stdout(io.StringIO()):
        runner.run()
    return runner


@pytest.fixture(scope='module')
def round_trip(tmp_path_factory):
    """Run straight through, and again with a checkpoint in the middle."""
    straight = _observables(_ran(_built(2 * HALF_HOURS)))

    first_half = _ran(_built(HALF_HOURS))
    at_checkpoint = _observables(first_half)
    path = str(tmp_path_factory.mktemp('ckpt') / 'run.ckpt')
    manifest = save_checkpoint(first_half, path)
    del first_half

    reloaded = load_checkpoint(path, duration=HALF_HOURS)
    at_reload = _observables(reloaded)
    resumed = _observables(_ran(reloaded))

    return {
        'straight': straight,
        'at_checkpoint': at_checkpoint,
        'at_reload': at_reload,
        'resumed': resumed,
        'manifest': manifest,
        'path': path,
    }


@pytest.mark.slow
def test_a_resumed_run_equals_an_uninterrupted_one(round_trip):
    """The whole point. If this fails, nothing else in the file matters.

    A difference here means some piece of state is being rebuilt instead
    of restored. Do NOT relax this to a tolerance: the two runs execute
    the same arithmetic in the same order on the same RNG stream, so
    equality is the correct assertion and a tolerance would hide exactly
    the bug this test exists to catch.
    """
    assert round_trip['resumed'] == round_trip['straight']


@pytest.mark.slow
def test_reloading_restores_the_state_the_checkpoint_was_taken_at(round_trip):
    """Separates "wrote it wrong" from "resumed it wrong".

    Without this, a failure of the test above says only that something is
    broken. With it, the two failures are distinguishable: this one fails
    if the checkpoint lost state, that one fails if the resumed run
    diverges from state that was restored correctly.
    """
    assert round_trip['at_reload'] == round_trip['at_checkpoint']


@pytest.mark.slow
def test_the_diffusion_fields_survive_the_round_trip(round_trip):
    """The fields are the trap, so they get their own assertion.

    ``SpatialField`` held a scipy ``SuperLU`` handle, which pickle
    refuses, so under the checkpoint's "whatever pickles is state" rule
    the entire ``fields`` dict would have been classified as wiring and
    left at its freshly-built value -- a restored run with the right
    cells and a reset chemical environment. ``SpatialField.__getstate__``
    drops the cache so the field travels.

    This asserts the fields are non-trivial as well as equal, because
    "all zero in both runs" would satisfy equality while proving nothing.
    """
    fields = round_trip['resumed']['fields']
    assert fields == round_trip['straight']['fields']
    assert fields, 'no fields at all -- the check above proves nothing'
    assert any(float(v) != 0.0 for v in fields.values()), (
        'every field sums to zero, so this test cannot tell a restored '
        'field from a rebuilt one')


@pytest.mark.slow
def test_the_checkpoint_rebuilds_only_wiring(round_trip):
    """Nothing unrecognised may be left to be rebuilt.

    The checkpoint decides what is state by offering each attribute to
    pickle, so anything that refuses is left as ``build`` made it. That is
    right for an event bus or a cached factorisation and catastrophic for
    state. This pins that the set of skipped attributes is only the known
    wiring; a new one shows up here rather than as a silent divergence.
    """
    assert round_trip['manifest']['unexpected_rebuilt'] == []
    rebuilt = round_trip['manifest']['rebuilt']
    assert rebuilt, 'expected the wiring to be excluded, got nothing'
    for module, names in rebuilt.items():
        assert 'event_bus' in names, (
            '%s kept its event bus, which reaches every other module: '
            'the checkpoint would carry the whole engine twice' % module)


@pytest.mark.slow
def test_a_checkpoint_from_a_future_format_is_refused(round_trip, tmp_path):
    """A format mismatch must be reported, not guessed at.

    Reading a payload whose layout this build does not know would restore
    a partial run, and a partial run is the failure mode with no symptom.
    """
    import pickle

    with open(round_trip['path'], 'rb') as handle:
        payload = pickle.load(handle)
    payload['format_version'] = FORMAT_VERSION + 1
    future = tmp_path / 'future.ckpt'
    with open(future, 'wb') as handle:
        pickle.dump(payload, handle)

    with pytest.raises(CheckpointError, match='format'):
        load_checkpoint(str(future))


@pytest.mark.slow
def test_an_unrecognised_unrestored_attribute_is_refused_by_default(
        round_trip, tmp_path):
    """``strict`` exists so an unknown skipped attribute cannot pass.

    If a module grows an attribute that refuses to pickle and is NOT
    wiring, the checkpoint has lost state. The reader refuses it by
    default and says what to check, rather than handing back a run that
    will quietly diverge.
    """
    import pickle

    with open(round_trip['path'], 'rb') as handle:
        payload = pickle.load(handle)
    payload['unexpected_rebuilt'] = ['some_new_unpicklable_state']
    suspect = tmp_path / 'suspect.ckpt'
    with open(suspect, 'wb') as handle:
        pickle.dump(payload, handle)

    with pytest.raises(CheckpointError, match='some_new_unpicklable_state'):
        load_checkpoint(str(suspect))

    # And can be accepted deliberately, which is the whole reason the
    # escape hatch is a named argument and not the default.
    runner = load_checkpoint(str(suspect), duration=0.05, strict=False)
    assert runner.engine is not None


@pytest.mark.slow
def test_checkpointing_before_build_is_an_error():
    """A runner with no engine has no state, and saying so beats writing
    an empty checkpoint that fails much later at reload."""
    runner = EngineRunner(dt=0.05, duration=0.1, scenario='Baseline',
                          tissue='slab')
    with pytest.raises(CheckpointError, match='build'):
        save_checkpoint(runner, os.devnull)
