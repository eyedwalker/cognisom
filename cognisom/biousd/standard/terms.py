"""A starter mapping from cell names to ontology terms.

The first question anyone writing a Bio-USD file asks is "which term do I
use for a macrophage?" This answers it for 43 cell identities and the
9 coarse type tokens, so a writer need not start from the ontology.

EVERY TERM BELOW WAS RESOLVED BY EXACT LABEL MATCH against EBI's Ontology
Lookup Service, and nothing else. A fuzzy search was tried first and it is
worse than useless, because it always returns something:

    Inflamed endothelial cell     -> endothelial cell of umbilical vein
    Stromal fibroblast            -> stromal cell of endometrium
    Prostate cancer stem cell     -> basal cell of prostate epithelium
    Luminal epithelial cell       -> luminal cell of mammary gland
    CD8+ T cell                   -> granzyme K-associated CD8 T cell

Four wrong tissues and one over-specific subtype, every one returned with
the same confidence as a correct answer. Where no exact term exists the
entry is absent rather than approximate: an unmapped cell is honest, and a
mis-mapped one is a citation that will be believed.

Two axes, not one. ``normal`` is a state, not a cell type, and has no
term; a malignant fibroblast is still a fibroblast. Identity goes in
``bio:cell:ontologyTerm``; disease belongs in its own attribute.

What the map settles on the way: "Regulatory T cell" and "Regulatory T
cell (Treg)" are one cell and share one term; M1 and M2 macrophages are
two (CL:0000863, CL:0000890), where a census-style mapping flattens both
to "macrophage"; the exhausted CD8 T cell has its own term (CL:0020031),
which matters on a platform built to study immune escape.

Cell Ontology for normal lineages; NCI Thesaurus for the oncology terms it
does not cover. Both are OBO-registered, so a CURIE resolves either way
(see :data:`~.vocabulary.CURIE_PREFIXES`).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from .vocabulary import is_curie

#: Cell identity (a library's display name) -> CURIE.
CELL_CURIE: Dict[str, str] = {
    'Basophil': 'CL:0000767',
    'CD4+ Th1 cell': 'CL:0000545',
    'CD4+ Th17 cell': 'CL:0000899',
    'CD4+ Th2 cell': 'CL:0000546',
    'CD4+ T cell': 'CL:0000624',
    'CD8+ Effector T cell': 'CL:0001050',
    'CD8+ Memory T cell': 'CL:0000907',
    'CD8+ Naive T cell': 'CL:0000900',
    'CD8+ T cell': 'CL:0000625',
    'Conventional dendritic cell (cDC)': 'CL:0000990',
    'Eosinophil': 'CL:0000771',
    'Exhausted CD8+ T cell': 'CL:0020031',
    'Gamma-delta T cell': 'CL:0000798',
    'ILC1': 'CL:0001067',
    'ILC2': 'CL:0001069',
    'ILC3': 'CL:0001071',
    'Macrophage M1': 'CL:0000863',
    'Macrophage M2': 'CL:0000890',
    'Macrophage': 'CL:0000235',
    'Mast cell': 'CL:0000097',
    'Memory B cell': 'CL:0000787',
    'Monocyte': 'CL:0000576',
    'NK cell': 'CL:0000623',
    'NKT cell': 'CL:0000814',
    'Naive B cell': 'CL:0000788',
    'Neutrophil': 'CL:0000775',
    'Plasma cell': 'CL:0000786',
    'Plasmacytoid dendritic cell (pDC)': 'CL:0000784',
    'Regulatory T cell': 'CL:0000815',
    'Regulatory T cell (Treg)': 'CL:0000815',
    'T follicular helper (Tfh)': 'CL:0002038',
    'Basal epithelial cell': 'CL:0002236',
    'Endothelial cell': 'CL:0000115',
    'High endothelial venule (HEV) cell': 'CL:0002652',
    'Inflamed endothelial cell': 'CL:0000115',
    'Luminal epithelial cell': 'CL:0002340',
    'Neuroendocrine cell': 'CL:0000165',
    'Postcapillary venule endothelial cell': 'CL:1000414',
    'Prostatic smooth muscle cell': 'CL:1000487',
    'Cancer-associated fibroblast': 'NCIT:C168534',
    'Prostate cancer stem cell': 'NCIT:C68706',
    'Prostatic adenocarcinoma cell': 'NCIT:C2919',
    'Stromal fibroblast': 'CL:0000057',
}

#: Coarse ``bio:cell:type`` token -> CURIE, for a consumer that has only
#: the token. ``normal`` is deliberately absent: it names a state.
TYPE_CURIE: Dict[str, str] = {
    'cancer': 'CL:0001064',
    'immune': 'CL:0000738',
    'stromal': 'CL:0000499',
    'endothelial': 'CL:0000115',
    'basal': 'CL:0000646',
    'luminal': 'CL:0002340',
    'neuroendocrine': 'CL:0000165',
    'stem': 'CL:0000034',
    'fibroblast': 'CL:0000057',
}


def curie_for(name: Optional[str], cell_type: Optional[str] = None
              ) -> Optional[str]:
    """The term for a named identity, else for its coarse type, else None.

    ``None`` rather than a guess: an unasserted identity is something a
    consumer can handle, a wrong one is not.
    """
    if name and name in CELL_CURIE:
        return CELL_CURIE[name]
    if cell_type:
        return TYPE_CURIE.get(str(cell_type).lower())
    return None


def coverage() -> Tuple[int, int]:
    """``(identities mapped, distinct terms)``; the two Tregs share one."""
    return len(CELL_CURIE), len(set(CELL_CURIE.values()))


assert all(is_curie(v) for v in CELL_CURIE.values())
assert all(is_curie(v) for v in TYPE_CURIE.values())
