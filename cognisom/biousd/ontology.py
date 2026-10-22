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

from typing import Dict, Optional, Tuple

#: Library entity name -> ontology CURIE. Cell Ontology where a term
#: exists, NCI Thesaurus for the oncology-specific cells it does not
#: cover. Both are OBO-registered, so a CURIE resolves either way.
CELL_CURIE: Dict[str, str] = {
    # ── immune lineages ────────────────────────────────────────────
    "Basophil": "CL:0000767",
    "CD4+ Th1 cell": "CL:0000545",
    "CD4+ Th17 cell": "CL:0000899",
    "CD4+ Th2 cell": "CL:0000546",
    "CD4+ T cell": "CL:0000624",
    "CD8+ Effector T cell": "CL:0001050",
    "CD8+ Memory T cell": "CL:0000907",
    "CD8+ Naive T cell": "CL:0000900",
    "CD8+ T cell": "CL:0000625",
    "Conventional dendritic cell (cDC)": "CL:0000990",
    "Eosinophil": "CL:0000771",
    # The exhausted state has its own term, and on this platform that is
    # not a nicety: exhaustion is what the MHC-I work is about.
    "Exhausted CD8+ T cell": "CL:0020031",
    "Gamma-delta T cell": "CL:0000798",
    "ILC1": "CL:0001067",
    "ILC2": "CL:0001069",
    "ILC3": "CL:0001071",
    # Polarisation, which the census mapping flattens to "macrophage".
    "Macrophage M1": "CL:0000863",
    "Macrophage M2": "CL:0000890",
    "Macrophage": "CL:0000235",
    "Mast cell": "CL:0000097",
    "Memory B cell": "CL:0000787",
    "Monocyte": "CL:0000576",
    "NK cell": "CL:0000623",
    "NKT cell": "CL:0000814",
    "Naive B cell": "CL:0000788",
    "Neutrophil": "CL:0000775",
    "Plasma cell": "CL:0000786",
    "Plasmacytoid dendritic cell (pDC)": "CL:0000784",
    # One cell, two library entities. Same CURIE collapses them.
    "Regulatory T cell": "CL:0000815",
    "Regulatory T cell (Treg)": "CL:0000815",
    "T follicular helper (Tfh)": "CL:0002038",

    # ── tissue and tumour ──────────────────────────────────────────
    "Basal epithelial cell": "CL:0002236",
    "Endothelial cell": "CL:0000115",
    "High endothelial venule (HEV) cell": "CL:0002652",
    # Inflammation is a STATE of an endothelial cell, not a kind of one.
    # The generic term is the honest mapping; the fuzzy search offered a
    # umbilical vein cell, which is a different vessel in a different
    # organ in a different person.
    "Inflamed endothelial cell": "CL:0000115",
    "Luminal epithelial cell": "CL:0002340",
    "Neuroendocrine cell": "CL:0000165",
    "Postcapillary venule endothelial cell": "CL:1000414",
    "Prostatic smooth muscle cell": "CL:1000487",
    # CL covers normal lineages; these two are oncology terms and live in
    # NCI Thesaurus instead.
    "Cancer-associated fibroblast": "NCIT:C168534",
    "Prostate cancer stem cell": "NCIT:C68706",
    "Prostatic adenocarcinoma cell": "NCIT:C2919",
    "Stromal fibroblast": "CL:0000057",
}

#: The coarse :class:`~cognisom.biousd.schema.CellType` enum, for consumers
#: that only have the token. ``normal`` is deliberately absent: it names a
#: state, and there is no cell type it could mean.
TYPE_CURIE: Dict[str, str] = {
    "cancer": "CL:0001064",          # malignant cell
    "immune": "CL:0000738",          # leukocyte
    "stromal": "CL:0000499",         # stromal cell
    "endothelial": "CL:0000115",
    "basal": "CL:0000646",           # basal cell
    "luminal": "CL:0002340",         # luminal cell of prostate epithelium
    "neuroendocrine": "CL:0000165",
    "stem": "CL:0000034",            # stem cell
    "fibroblast": "CL:0000057",
}

#: Where each prefix resolves, so a reader can follow a CURIE without
#: knowing the registries.
PREFIX_BASE: Dict[str, str] = {
    "CL": "http://purl.obolibrary.org/obo/CL_",
    "NCIT": "http://purl.obolibrary.org/obo/NCIT_",
}


def curie_for(name: Optional[str], cell_type: Optional[str] = None
              ) -> Optional[str]:
    """The CURIE for a library entity, falling back to its coarse type.

    Returns ``None`` rather than a guess. A cell with no term is a cell
    whose identity this schema does not assert, which is a thing a
    consumer can handle; a wrong term is not.
    """
    if name and name in CELL_CURIE:
        return CELL_CURIE[name]
    if cell_type:
        return TYPE_CURIE.get(str(cell_type).lower())
    return None


def iri_for(curie: Optional[str]) -> Optional[str]:
    """Expand a CURIE to a resolvable IRI, or ``None``."""
    if not curie or ":" not in curie:
        return None
    prefix, local = curie.split(":", 1)
    base = PREFIX_BASE.get(prefix)
    return f"{base}{local}" if base else None


def coverage() -> Tuple[int, int]:
    """``(cells mapped, distinct terms)`` -- two Tregs share one term."""
    return len(CELL_CURIE), len(set(CELL_CURIE.values()))
