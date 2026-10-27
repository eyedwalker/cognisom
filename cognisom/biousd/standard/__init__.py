"""Bio-USD: biological cells on OpenUSD, as a convention anyone can write to.

This package has no dependency on the rest of Cognisom and imports only
``numpy`` and ``pxr``. It is laid out to be extracted into its own
repository unchanged::

    git subtree split --prefix=cognisom/biousd/standard -b biousd
    # ... push that branch to a repository whose top-level package is `biousd`

Inside Cognisom it is ``cognisom.biousd.standard``; extracted it is
``biousd``. Every import is relative so both are true.

What is here:

- :mod:`.vocabulary` -- the convention, as data; SPEC.md renders from it.
- :mod:`.write` -- a minimal writer: any cells in, a conforming stage out.
- :mod:`.read` -- the stage back into arrays.
- :mod:`.validate` -- file-level conformance, with a command-line entry.
- :mod:`.api_schema` -- the codeless ``BioCellAPI`` schema and its plugin.
- :mod:`.profiles` -- UsdProfiles capabilities and claims.
- :mod:`.validators` -- the same checks for OpenUSD's validation framework.
- :mod:`.check` -- ``biousd-check``: framework plus Bio-USD, one report.
- :mod:`.build_plugin` -- regenerates ``schema.usda`` and ``plugin/``.

IMPORT FIRST
------------
Importing this package registers the schema plugin with ``Plug``. USD
builds its schema registry once per process, the first time anything
needs a prim definition (opening a stage does), and a plugin registered
after that is never consulted: measured on usd-core 25.11, a registry
built first never learns ``BioCellAPI`` however many times the plugin
is registered afterwards. So ``import biousd`` before the first stage is
opened, or set ``PXR_PLUGINPATH_NAME`` to :func:`plugin_path`. The
import touches Plug only and does not build the registry itself (asking
whether the schema is known would, and would lock out any plugin a host
registers after the import; it did, once). :data:`schema_registered` is
that Plug-level fact; :func:`schema_is_known` asks the registry when
called, and a late import is found out there, with one WARNING, while
everything else still works: files open, the validator runs, only the
schema's fallbacks and documentation are missing. On a ``pxr`` that can
load them (26.05+) the import also hands the validation-framework plugin
to Plug (:func:`~.validators.advertise_validators`); on an older wheel
it advertises nothing, so nothing can fail to load. ``BIOUSD_NO_PLUGIN=1``
skips both registrations.
"""

import os as _os

from .vocabulary import (  # noqa: F401
    API_SCHEMA, CARRIER, CELL_PHASES, CELL_PRIMVARS, CELL_TYPES, CURIE_PREFIXES,
    LAYER_DATA, LIBRARY_NAME, METERS_PER_UNIT, REQUIRED_PRIMVARS, VALIDATOR_CODES,
    VERSION, Primvar, iri_for, is_curie, render_spec_markdown)

# The submodules are imported here, eagerly, and the public functions bound
# AFTER them on purpose. Importing a submodule sets an attribute of the
# same name on the package, so a package-level function called `validate`
# defined before `.validate` was imported was replaced by the module the
# first time anything imported it -- and the second call to
# biousd.validate() hit a module. Binding last keeps the function.
from . import api_schema as _api_schema_module      # noqa: E402
from . import profiles as _profiles_module          # noqa: E402
from . import read as _read_module                  # noqa: E402
from . import validate as _validate_module          # noqa: E402
from . import write as _write_module                # noqa: E402
from . import validators as _validators_module      # noqa: E402
from . import check as _check_module                # noqa: E402

write_cells = _write_module.write_cells
read_cells = _read_module.read_cells
read_stage = _read_module.read_stage
validate = _validate_module.validate
validate_stage = _validate_module.validate_stage
plugin_path = _api_schema_module.plugin_path
register_schema = _api_schema_module.register_schema
schema_is_known = _api_schema_module.schema_is_known
apply_api = _api_schema_module.apply_api
write_claims = _profiles_module.write_claims
check_claims = _profiles_module.check_claims
register_validators = _validators_module.register_validators
advertise_validators = _validators_module.advertise_validators
check_file = _check_module.check_file

# Register the plugins now, before any stage can have built a registry:
# the schema with Plug (quiet without pxr), and the framework validators
# only where this pxr can load them. Neither builds a registry; a late
# import is found out by schema_is_known(), with one WARNING (see above).
if _os.environ.get(_api_schema_module.ENV_NO_PLUGIN, "") not in ("", "0"):
    schema_registered = False
else:
    schema_registered = register_schema()
    _validators_module.advertise_validators()

__version__ = VERSION
__all__ = [
    "VERSION", "__version__", "METERS_PER_UNIT", "CELL_PRIMVARS", "CELL_TYPES",
    "CELL_PHASES", "CURIE_PREFIXES", "LAYER_DATA", "CARRIER", "REQUIRED_PRIMVARS",
    "VALIDATOR_CODES", "API_SCHEMA", "LIBRARY_NAME", "Primvar", "is_curie", "iri_for",
    "render_spec_markdown", "write_cells", "read_cells", "read_stage", "validate",
    "validate_stage", "plugin_path", "register_schema", "schema_is_known",
    "schema_registered", "apply_api", "write_claims", "check_claims",
    "register_validators", "advertise_validators", "check_file",
]
