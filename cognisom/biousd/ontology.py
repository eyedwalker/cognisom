"""What kind of cell this is, in a vocabulary other people already use.

Bio-USD's ``bio:cell:type`` is a ten-token enum invented here: ``normal``,
``cancer``, ``immune``, ``stromal``, ``endothelial``, ``basal``,
``luminal``, ``neuroendocrine``, ``stem``, ``fibroblast``. Nothing outside
this repository knows what those mean, which is a problem for a schema
whose whole purpose is to be read by tools that are not ours.

The vocabulary the rest of biology uses is the **Cell Ontology** (CL), and
it is not a niche choice: CZ CELLxGENE, the Human Cell Atlas and Tabula
Sapiens label every one of their cells with a CL term. Carrying the CURIE
alongside the enum means a scene written here can be joined to 154 million
annotated cells without a translation table.

TWO AXES, NOT ONE
-----------------
``CellType.NORMAL`` is not a cell type. It is a *state*, and conflating it
with identity is why that enum cannot be mapped cleanly: a malignant
fibroblast is still a fibroblast. So identity and disease are separate
here --- :data:`CELL_CURIE` answers "what kind of cell", and a disease term
belongs in its own attribute. CL has no term for "normal" and should not.

EXACT MATCHES ONLY, AND HERE IS WHY
-----------------------------------
Every CURIE below was resolved against EBI's OLS by **exact label match**
and nothing else. A fuzzy search was tried first and it is worse than
useless, because it always returns something:

    Inflamed endothelial cell     -> endothelial cell of umbilical vein
    Stromal fibroblast            -> stromal cell of endometrium
    Prostate cancer stem cell     -> basal cell of prostate epithelium
    Luminal epithelial cell       -> luminal epithelial cell of mammary gland
    CD8+ T cell                   -> granzyme K-associated CD8 T cell

Four wrong tissues and one over-specific subtype, every one returned with
the same confidence as a correct answer. Where no exact term exists the
entry is absent rather than approximate, because an unmapped cell is
honest and a mis-mapped one is a citation that will be believed.

WHAT THIS FIXES ON THE WAY
--------------------------
*Both Tregs collapse.* "Regulatory T cell" and "Regulatory T cell (Treg)"
are one cell under two names and defeat every name-based check; they
resolve to the same CURIE, so a consumer grouping by term sees one cell.

*M1 and M2 separate.* The library's census mapping sends both to the
generic "macrophage", losing the polarisation that is the entire reason
they are two entities. CL:0000863 and CL:0000890 exist.

*Exhaustion gets a term.* CL:0020031, which matters here specifically:
exhausted CD8 was geometrically identical to a naive one, on a platform
built to study immune escape.
"""

from __future__ import annotations

from typing import Optional

# The map itself now lives in the publishable package, so a stranger
# writing a Bio-USD file gets the same answers this platform does. This
# module keeps its name and API for everything in Cognisom that imports
# it; the data has one home.
from cognisom.biousd.standard.terms import (  # noqa: F401
    CELL_CURIE, TYPE_CURIE, coverage, curie_for)
from cognisom.biousd.standard.vocabulary import CURIE_PREFIXES as PREFIX_BASE  # noqa: F401
from cognisom.biousd.standard.vocabulary import iri_for  # noqa: F401
