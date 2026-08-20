"""
Cognisom — personalized molecular digital twin platform.

This file makes `cognisom` a regular package rather than an implicit
namespace package. That matters for more than tidiness: as a namespace
package, `cognisom` could absorb portions from anywhere on sys.path, so
which file an import like `cognisom.plugins.examples.virus_plugin` actually
loaded depended on sys.path ordering — and the repository contains a second,
divergent copy of most of this tree at the root. With `__init__.py` present
and the package installed (see pyproject.toml), resolution is deterministic.

The top-level `core/`, `gpu/`, `modules/`, `engine/` ... directories are the
duplicate tree still being collapsed into this package. Import from
`cognisom.*`, never bare `core.*` / `gpu.*`, so that the copy you get is the
copy that ships.
"""

__version__ = "0.9.0"
