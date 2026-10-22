"""Composed scene instances, in the shape the USD writer reads.

The dashboard's Scene Composer keeps what it builds as plain dicts --
``{id, proto_name, x, y, z, r, cr, cg, cb}`` -- because the exporter it
was written against assembled ``.usda`` text by hand and needed nothing
more. :func:`cognisom.biousd.usd_writer.write_cells` reads attributes
instead, defensively, so the gap between the two is small enough to
close here rather than by changing how the composer stores a scene.

Worth stating plainly, because the numbers are not marginal. On a
6,000-instance scene of three cell prototypes:

    hand-written .usda    4.74 MB    18,006 prims
    write_cells .usdc     0.31 MB        14 prims

One prim per entity plus a nested Material each, against one
PointInstancer with a prototype per distinct mesh. That is 15x the bytes
and 1,286x the prims, and the text path additionally carries no
Bio-USD attributes, no ontology terms, no shared material and no way to
cut away the stroma that hides the subject.
"""

from __future__ import annotations

import contextlib
import functools
import io
from typing import Any, Dict, Mapping, Optional, Sequence

#: Composer prototype name -> the cell library's own name for it.
#:
#: This is what turns an exported scene from one sphere prototype into
#: real cell shapes. The composer's prototype library knows three
#: geometries -- sphere, capsule, ellipsoid -- while the cell library
#: carries a built morphology per entity, so going through the name
#: gets a neutrophil its multilobed nucleus and a dendritic cell its
#: branches:
#:
#:     tumor_cell      -> irregular_ruffled      macrophage_m1 -> ruffled_pseudopod
#:     neutrophil      -> multilobed_granular    macrophage_m2 -> elongated_spindle
#:     dendritic_cell  -> stellate_branching     nk_cell       -> granular_sphere
#:     plasma_cell     -> eccentric_nucleus_er   mast_cell     -> dense_granular
#:     endothelial_cell-> flat_squamous
#:
#: Generic prototypes (``immune_cell``, ``white_blood_cell``,
#: ``generic_cell``) are deliberately absent. They name no lineage, so
#: there is no entity to point at, and they fall back to a sphere --
#: which is what they are.
LIBRARY_ENTITY_FOR_PROTOTYPE: Dict[str, str] = {
    "tumor_cell": "Prostatic adenocarcinoma cell",
    "t_cell_cd4": "CD4+ T cell",
    "t_cell_cd8": "CD8+ T cell",
    "b_cell": "Naive B cell",
    "plasma_cell": "Plasma cell",
    "nk_cell": "NK cell",
    "macrophage_m1": "Macrophage M1",
    "macrophage_m2": "Macrophage M2",
    "mast_cell": "Mast cell",
    "neutrophil": "Neutrophil",
    "dendritic_cell": "Conventional dendritic cell (cDC)",
    "endothelial_cell": "Endothelial cell",
}


@functools.lru_cache(maxsize=1)
def _store():
    """One store, opened quietly; this would run once per cell otherwise."""
    from cognisom.library.store import EntityStore
    with contextlib.redirect_stdout(io.StringIO()):
        return EntityStore()


@functools.lru_cache(maxsize=256)
def mesh_type_for_prototype(proto_name: str) -> Optional[str]:
    """The library's mesh for a composer prototype, or ``None``.

    ``None`` rather than a sphere, so the caller can tell "this entity
    has no shape in the library" from "this entity is round".
    """
    entity_name = LIBRARY_ENTITY_FOR_PROTOTYPE.get(str(proto_name).lower())
    if entity_name is None:
        return None
    try:
        entity = _store().find_entity_by_name(entity_name)
    except Exception:                       # pragma: no cover - no library
        return None
    return getattr(entity, "mesh_type", None) if entity is not None else None

#: Composer prototype name -> Bio-USD cell-type token.
#:
#: Only names the prototype library actually ships are listed. Anything
#: else stays ``normal``: a guessed lineage would flow straight into
#: ``bio:cell:type`` and from there into an ontology term, and a cell
#: labelled with a term it does not have is worse than one labelled with
#: none. The same reasoning governs
#: :mod:`cognisom.biousd.ontology`, where a fuzzy search was dropped for
#: returning a confident wrong answer every time.
#: Every key below is a name the prototype library actually ships, which
#: is not where this started: a first draft keyed on ``cancer_cell``,
#: ``t_cell``, ``macrophage``, ``fibroblast`` and ``stem_cell``, and the
#: library has none of those -- it has ``tumor_cell``, ``t_cell_cd4``,
#: ``t_cell_cd8``, ``macrophage_m1`` and ``macrophage_m2``, and no
#: fibroblast or stem cell at all. A mapping whose keys never match
#: fires never, and looks exactly like one that works. The test that
#: every key resolves against the registry is what keeps it honest.
CELL_TYPE_FOR_PROTOTYPE: Dict[str, str] = {
    "tumor_cell": "cancer",
    # The library draws no distinction finer than this for the generic
    # leukocyte, and the polarisations it does carry (M1/M2) are a state
    # the Bio-USD enum has no token for -- `bio:cell:type` is lineage.
    "immune_cell": "immune",
    "white_blood_cell": "immune",
    "b_cell": "immune",
    "plasma_cell": "immune",
    "t_cell_cd4": "immune",
    "t_cell_cd8": "immune",
    "nk_cell": "immune",
    "macrophage_m1": "immune",
    "macrophage_m2": "immune",
    "mast_cell": "immune",
    "neutrophil": "immune",
    "dendritic_cell": "immune",
    "endothelial_cell": "endothelial",
}


class SceneCell:
    """One composed instance, as ``write_cells`` expects to read it."""

    __slots__ = ("position", "cell_type", "scale_um", "volume", "name",
                 "cell_id", "mesh_type")

    def __init__(self, instance: Mapping[str, Any], index: int = 0):
        proto = str(instance.get("proto_name", "")).lower()
        self.position = (float(instance.get("x", 0.0)),
                         float(instance.get("y", 0.0)),
                         float(instance.get("z", 0.0)))
        self.cell_type = CELL_TYPE_FOR_PROTOTYPE.get(proto, "normal")
        self.mesh_type = mesh_type_for_prototype(proto)
        # `r` is a radius and `scale_um` is a diameter. The prototype is
        # authored at unit diameter, so confusing the two halves every
        # cell in the scene without raising anything.
        self.scale_um = float(instance.get("r", 5.0)) * 2.0
        self.volume = 1.0
        self.name = instance.get("proto_name")
        self.cell_id = index


def cells_from_instances(instances: Sequence[Mapping[str, Any]]
                         ) -> list[SceneCell]:
    """Adapt a composed scene for :func:`write_cells`."""
    return [SceneCell(inst, i) for i, inst in enumerate(instances)]
