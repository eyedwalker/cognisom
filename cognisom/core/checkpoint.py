"""Checkpoint and restart a modules-engine run.

WHY THIS EXISTS
---------------
A long run could not be resumed. ``EngineRunner.snapshot_interval_hours``
sounds like it would do this and does not: those snapshots are cell
positions for the 3D view, a few scalars per cell, with nothing of the
module state that produced them. So a run that died at hour 300 started
again at hour 0, and the realism plan's item 4 lists "no
checkpoint/restart" alongside the scaling work for that reason.

WHY PICKLE, AND NOT A ``state_dict`` PER MODULE
-----------------------------------------------
The dominant risk in checkpointing is not that the writer crashes -- it
is that it silently misses a field, and the restored run then diverges
from the uninterrupted one in a way no exception reports. Thirteen
hand-written ``state_dict`` methods maximise that risk: each one is a
list that has to be kept in step with its module's attributes forever,
and nothing fails when it is not.

Pickle cannot miss a field, because it captures the object graph by
construction. What it cannot do is capture WIRING. Probing the live
engine showed the unpicklable leaves are only three kinds, and not one of
them is simulation state:

  * ``SpatialField._lu`` -- six cached sparse LU factorisations, each
    rebuildable from its own matrix on the next solve;
  * ``ARSignalingModule.rhs`` -- a closure from the reaction builder,
    rebuilt by ``initialize``;
  * event-bus subscribers and ``_secretion_providers`` -- bound methods
    and local functions, i.e. the wiring ``initialize`` establishes.

So this module rebuilds the wiring and restores only the state. It builds
a fresh runner from the stored construction arguments -- which reproduces
the wiring exactly, because ``build`` is deterministic given them -- and
then overwrites each module's data.

HOW THE WIRING IS EXCLUDED WITHOUT A LIST OF NAMES
--------------------------------------------------
Each module attribute is offered to pickle individually. Whatever pickles
is state and is stored; whatever refuses is wiring or a cache and is left
as ``build`` reconstructed it. That needs no hand-maintained name list
and stays correct when a module grows a new callback, because every
wiring reference reaches an unpicklable bound method transitively -- a
sibling module holds the event bus, and the event bus holds bound
methods.

The one thing that had to be arranged for is ``SpatialModule.fields``,
which IS state (the concentration arrays) but held an unpicklable LU
handle: ``SpatialField.__getstate__`` drops the cache so the field itself
travels. Without that, the whole diffusion state would have been
classified as wiring and silently dropped -- which is precisely the
failure mode this design exists to avoid, and precisely why
:func:`save_checkpoint` records which attributes it skipped and
:func:`load_checkpoint` checks them against what it expected.

WHAT FAITHFUL MEANS HERE
------------------------
The engine draws from the SHARED numpy global RNG stream, so the stream
position is part of the state: restored without it, every later draw
differs and the run diverges immediately even though every cell is
correct. ``np.random.get_state()`` is captured, and restored AFTER
``build``, because building consumes draws of its own.

Verified by ``tests/test_a_run_can_be_resumed.py``, which runs N hours
straight and compares it against N/2 hours, a checkpoint, a reload into a
fresh object, and N/2 hours more -- every observable, bit for bit,
including the RNG stream position. That test is the specification; if it
fails, something has become state that this module treats as wiring.

WHAT THIS IS NOT
----------------
Not an archival format. It is pickle, so it is readable only by a
compatible build of this package and its dependencies; the format version
and the versions it was written under are stored, and a mismatch is
reported rather than guessed at. It is for resuming a run, not for
publishing one -- use the USD export for that.
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

#: Bumped when the payload layout changes in a way an older reader would
#: misread. A reader refuses a version it does not know rather than
#: silently restoring a partial run.
FORMAT_VERSION = 1

#: EngineRunner constructor arguments that decide the engine's shape.
#: Kept as an explicit list because a missing one would change the
#: rebuilt engine's STRUCTURE, which is the one thing the "try to pickle
#: it" rule cannot detect: each attribute would restore cleanly onto the
#: wrong scaffolding.
_RUNNER_ARGS: Tuple[str, ...] = (
    'dt', 'duration', 'record_interval', 'snapshot_interval_hours',
    'tissue',
)

#: Carried as already-merged state and reapplied BEFORE build, rather
#: than as the constructor arguments that produced them. The runner
#: merges preset, patient and caller overrides into `module_configs` and
#: keeps only the result, so the merged form is both the only thing
#: available and the more faithful thing to store: it captures the
#: effective configuration however it was arrived at, including a
#: patient's contribution.
_RUNNER_MERGED: Tuple[str, ...] = (
    'module_configs', 'modules_enabled', 'patient_inputs',
)

#: Attributes expected to refuse pickling, i.e. the wiring and caches
#: that `build` reconstructs. This is NOT how they are excluded -- that
#: is decided per attribute by actually trying -- it is what a
#: checkpoint is checked against, so that something NEW becoming
#: unpicklable is reported instead of being quietly dropped.
_EXPECTED_REBUILT: frozenset = frozenset({
    'event_bus',
    'rhs',
    '_secretion_providers',
    'ar_module',
    'cellular_module',
    'circadian_module',
    'immune_module',
    'molecular_module',
    'spatial_module',
    'vascular_module',
})


class CheckpointError(RuntimeError):
    """A checkpoint could not be written or read faithfully."""


def _pickles(value: Any) -> bool:
    try:
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        return False
    return True


def _module_state(module: Any) -> Tuple[Dict[str, Any], List[str]]:
    """Split a module's attributes into state and rebuilt-by-wiring.

    Returns (state, rebuilt_names). Membership is decided by offering
    each value to pickle, not by its name -- see this module's docstring
    for why that is the self-maintaining choice.
    """
    state: Dict[str, Any] = {}
    rebuilt: List[str] = []
    for name, value in vars(module).items():
        if _pickles(value):
            state[name] = value
        else:
            rebuilt.append(name)
    return state, sorted(rebuilt)


def save_checkpoint(runner: Any, path: str) -> Dict[str, Any]:
    """Write a resumable checkpoint of ``runner`` to ``path``.

    ``runner`` must already have been built. Returns the manifest that
    was written (without the bulk state), so a caller can log or assert
    what the checkpoint actually contains -- in particular which
    attributes were left to be rebuilt.
    """
    engine = getattr(runner, 'engine', None)
    if engine is None:
        raise CheckpointError(
            'runner has no engine: call build() before checkpointing')

    args = {}
    for name in _RUNNER_ARGS + _RUNNER_MERGED + ('scenario_name',):
        if not hasattr(runner, name):
            raise CheckpointError(
                'runner is missing %r, so the rebuilt engine could '
                'differ in shape' % name)
        value = getattr(runner, name)
        if not _pickles(value):
            raise CheckpointError(
                'runner.%s cannot be pickled, so the engine could not be '
                'rebuilt with the same shape. This is config, not state: '
                'it must be stored, not left to be rebuilt.' % name)
        args[name] = value

    modules: Dict[str, Dict[str, Any]] = {}
    rebuilt: Dict[str, List[str]] = {}
    for name, module in engine.modules.items():
        state, skipped = _module_state(module)
        modules[name] = state
        if skipped:
            rebuilt[name] = skipped

    unexpected = sorted({n for names in rebuilt.values() for n in names}
                        - _EXPECTED_REBUILT)

    payload = {
        'format_version': FORMAT_VERSION,
        'numpy_version': np.__version__,
        'pickle_protocol': pickle.HIGHEST_PROTOCOL,
        'runner_args': args,
        'engine_time': float(engine.time),
        'engine_step_count': int(engine.step_count),
        # The shared global stream. Without this the restored run has
        # every cell right and every later draw wrong.
        'rng_state': np.random.get_state(),
        'modules': modules,
        'rebuilt': rebuilt,
        'unexpected_rebuilt': unexpected,
    }

    with open(path, 'wb') as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {k: v for k, v in payload.items()
                if k not in ('modules', 'rng_state')}
    manifest['module_names'] = sorted(modules)
    manifest['n_attributes'] = {k: len(v) for k, v in modules.items()}
    return manifest


def load_checkpoint(path: str, duration: Optional[float] = None,
                    strict: bool = True) -> Any:
    """Rebuild a runner from the checkpoint at ``path`` and return it.

    ``duration`` sets how much FURTHER the returned runner will simulate
    when ``run()`` is called, in hours; the stored duration is used when
    it is omitted. ``run()`` restarts the recording clocks, so the
    returned runner's history covers the resumed segment only -- the
    earlier segment's history lives in the run that wrote the
    checkpoint.

    With ``strict`` (the default), a checkpoint whose skipped attributes
    are not the known wiring is refused, because an unrecognised one may
    be state that would be silently lost.
    """
    with open(path, 'rb') as handle:
        payload = pickle.load(handle)

    version = payload.get('format_version')
    if version != FORMAT_VERSION:
        raise CheckpointError(
            'checkpoint format %r, this build reads %r'
            % (version, FORMAT_VERSION))

    unexpected = payload.get('unexpected_rebuilt') or []
    if unexpected and strict:
        raise CheckpointError(
            'checkpoint left attributes unrestored that are not known '
            'wiring: %s. They may be state, in which case the resumed '
            'run would diverge silently. Teach _EXPECTED_REBUILT about '
            'them only after confirming they are rebuilt by build(), or '
            'pass strict=False to accept the risk.'
            % ', '.join(unexpected))

    from cognisom.dashboard.engine_runner import EngineRunner

    args = dict(payload['runner_args'])
    scenario = args.pop('scenario_name')
    merged = {name: args.pop(name) for name in _RUNNER_MERGED if name in args}
    if duration is not None:
        args['duration'] = float(duration)

    runner = EngineRunner(scenario=scenario, **args)
    # Before build, because build reads them to decide which modules to
    # construct and with what parameters. Reapplying the merged result
    # makes the rebuilt engine the same shape whatever mix of preset,
    # patient and overrides originally produced it.
    for name, value in merged.items():
        setattr(runner, name, value)
    runner.build()

    engine = runner.engine
    stored = payload['modules']
    missing = sorted(set(stored) - set(engine.modules))
    if missing:
        raise CheckpointError(
            'checkpoint holds modules the rebuilt engine does not have: '
            '%s' % ', '.join(missing))

    for name, state in stored.items():
        # update(), not assignment: build() has already wired this
        # module to its siblings and the event bus, and those references
        # are exactly what the checkpoint does not carry.
        engine.modules[name].__dict__.update(state)

    engine.time = float(payload['engine_time'])
    engine.step_count = int(payload['engine_step_count'])

    # Last, because build() consumed draws of its own.
    np.random.set_state(payload['rng_state'])

    return runner
