"""Every registered module must construct through the documented API.

``cognisom/modules/__init__.py`` documents ``create_module(name, config=None)``
as the public constructor, and ``core/module_base.py`` declares
``__init__(self, config: Dict = None)``. Nine of the ten built-in modules none
the less crashed on that path with
``AttributeError: 'NoneType' object has no attribute 'get'``.

Two facts collided. ``SimulationModule.__init__`` does the None-normalisation
itself -- ``self.config = config or {}`` -- so the merged dict only ever exists
on ``self.config``; ``super().__init__(config)`` cannot rebind the caller's
local, so the raw parameter is still ``None`` when the subclass reads it. And
nine modules then read their parameters off that raw parameter
(``config.get('n_t_cells', 15)``) rather than off ``self.config``. There were
48 such reads.

It stayed invisible because the one production caller launders it:
``SimulationEngine.register_module`` does ``module_class(config or {})``, and
every scenario, dashboard page and REST endpoint goes through that line. No
test in the suite constructed a module through ``create_module(name)`` with no
config -- the single ``create_module`` caller passed one explicitly.
``cell_mechanics`` escaped because it alone wrote the defensive form itself.

The parametrised test below is the one that would have caught it, and it is
driven off ``list_modules()`` so a module added later is covered for free.
"""
from __future__ import annotations

import pytest

from cognisom.modules import create_module, list_modules

ALL_MODULES = sorted(list_modules())


def test_there_are_modules_registered():
    """A silent empty registry would make every parametrised case vacuous."""
    assert ALL_MODULES, "no modules registered"
    # Ten built-ins today; assert the floor, not the exact number, so adding
    # one does not fail this file.
    assert len(ALL_MODULES) >= 10


@pytest.mark.parametrize("name", ALL_MODULES)
def test_module_constructs_with_no_config(name):
    """The documented no-argument path must work for every module."""
    module = create_module(name)

    assert module is not None
    # The real invariant is that the base class's normalisation is what the
    # subclass reads. Asserting == {} would over-constrain a future module
    # that legitimately merges defaults into self.config.
    assert module.config is not None
    assert isinstance(module.config, dict)


@pytest.mark.parametrize("name", ALL_MODULES)
def test_module_initializes_and_reports_state_with_no_config(name):
    """Constructing is not enough -- the defaults must be usable."""
    module = create_module(name)
    module.initialize()

    state = module.get_state()
    assert isinstance(state, dict)


@pytest.mark.parametrize("name", ALL_MODULES)
def test_an_explicit_config_is_still_honoured(name):
    """The fix must not stop a supplied config from reaching the module."""
    module = create_module(name, config={"a_key_no_module_reads": 1})

    assert module.config["a_key_no_module_reads"] == 1


def test_no_module_reads_the_raw_config_parameter():
    """Pin the shape of the bug, not just its symptom.

    A module could regress by adding one new ``config.get(...)`` read while
    still constructing fine for every *other* parameter -- the crash only
    surfaces for whichever read runs first. Scanning the source catches a
    reintroduction that behavioural tests would miss until that line matters.
    """
    import inspect
    import re

    # Matches `config.get(` but not `self.config.get(` or `..._config.get(`.
    raw_read = re.compile(r"(?<!self\.)(?<![\w.])config\.get\(")

    offenders = []
    for name in ALL_MODULES:
        module = create_module(name)
        source = inspect.getsource(type(module))
        for lineno, line in enumerate(source.splitlines(), 1):
            if raw_read.search(line):
                offenders.append(f"{type(module).__name__} line {lineno}: {line.strip()}")

    assert not offenders, (
        "these read the raw config parameter, which is None on the documented "
        "no-argument path; use self.config instead:\n  " + "\n  ".join(offenders)
    )
