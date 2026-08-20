"""
Simulation engine internals.

Without this file the directory is an implicit namespace package, which
setuptools skips under `namespaces = false` — so the entire engine subtree
(immune, molecular, spatial) was silently omitted from the built wheel while
cognisom.modules imported from it.
"""
