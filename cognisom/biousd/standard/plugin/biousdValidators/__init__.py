"""biousdValidators: the Plug entry point for Bio-USD's UsdValidation validators.

OpenUSD's validation registry reads the ``plugInfo.json`` beside this
file when it is built, advertises ``biousdValidators:StageMetadata``,
``:CellPopulation`` and ``:BioUSDSuite`` from it, and the first time a
client asks for one it loads this ``"Type": "python"`` plugin by running
``import biousdValidators`` -- which is why the ``plugin/`` directory has
to be on ``sys.path``. ``import biousd`` (on a pxr that can register,
26.05+) and ``biousd.register_validators()`` hand this directory to
Plug and put ``plugin/`` on ``sys.path`` in the same step; a process
that names this directory in ``PXR_PLUGINPATH_NAME`` instead must add
``plugin/`` to ``PYTHONPATH``, or the registry's load fails with a
``Tf.ErrorException`` the first time any client runs all validators.

This module is deliberately empty of checks. Plug imports it as a
top-level module, outside the package, so it cannot import the package
relatively; it finds the package that ships it by path -- under whatever
name that package was imported, ``biousd`` or ``cognisom.biousd.standard``
-- and asks :func:`validators.register_from_plugin` to register the
same task functions the command-line validator runs. If nothing has
imported the package yet (a process that found this plugin through
``PXR_PLUGINPATH_NAME`` only), it is imported here as ``biousd``.

On a usd-core without Python validator registration (before OpenUSD
26.05) the import succeeds, the package logs the reason once, and
nothing is registered; the registry then treats the validator as one
that did not load, silently.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.dirname(os.path.dirname(_HERE))        # plugin/biousdValidators -> package


def _package():
    """The biousd package directory's module, whatever it is imported as."""
    for name, module in list(sys.modules.items()):
        init = getattr(module, "__file__", None)
        if (init and os.path.basename(init) == "__init__.py"
                and os.path.dirname(os.path.abspath(init)) == _PACKAGE_DIR):
            return module
    spec = importlib.util.spec_from_file_location(
        "biousd", os.path.join(_PACKAGE_DIR, "__init__.py"),
        submodule_search_locations=[_PACKAGE_DIR])
    module = importlib.util.module_from_spec(spec)
    sys.modules["biousd"] = module
    spec.loader.exec_module(module)
    return module


registered = importlib.import_module(_package().__name__ + ".validators").register_from_plugin()
