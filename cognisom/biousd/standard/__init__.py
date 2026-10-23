"""Bio-USD: biological cells on OpenUSD, as a convention anyone can write to.

This package has no dependency on the rest of Cognisom and imports only
``numpy`` and, lazily, ``pxr``. It is laid out to be extracted into its
own repository unchanged::

    git subtree split --prefix=cognisom/biousd/standard -b biousd
    # ... push that branch to a repository whose top-level package is `biousd`

Inside Cognisom it is ``cognisom.biousd.standard``; extracted it is
``biousd``. Every import is relative so both are true.

What is here:

- :mod:`.vocabulary` -- the convention, as data; SPEC.md renders from it.
- :mod:`.write` -- a minimal writer: any cells in, a conforming stage out.
- :mod:`.read` -- the stage back into arrays.
- :mod:`.validate` -- file-level conformance, with a command-line entry.
"""

from .vocabulary import (  # noqa: F401
    CARRIER, CELL_PHASES, CELL_PRIMVARS, CELL_TYPES, CURIE_PREFIXES, LAYER_DATA,
    METERS_PER_UNIT, REQUIRED_PRIMVARS, VERSION, Primvar, iri_for, is_curie,
    render_spec_markdown)

# The submodules are imported here, eagerly, and the public functions bound
# AFTER them on purpose. Importing a submodule sets an attribute of the
# same name on the package, so a package-level function called `validate`
# defined before `.validate` was imported was replaced by the module the
# first time anything imported it -- and the second call to
# biousd.validate() hit a module. Binding last keeps the function.
from . import read as _read_module      # noqa: E402
from . import validate as _validate_module  # noqa: E402
from . import write as _write_module    # noqa: E402

write_cells = _write_module.write_cells
read_cells = _read_module.read_cells
read_stage = _read_module.read_stage
validate = _validate_module.validate
validate_stage = _validate_module.validate_stage

__version__ = VERSION
__all__ = [
    "VERSION", "__version__", "METERS_PER_UNIT", "CELL_PRIMVARS", "CELL_TYPES",
    "CELL_PHASES", "CURIE_PREFIXES", "LAYER_DATA", "CARRIER", "REQUIRED_PRIMVARS",
    "Primvar", "is_curie", "iri_for", "render_spec_markdown",
    "write_cells", "read_cells", "read_stage", "validate", "validate_stage",
]
