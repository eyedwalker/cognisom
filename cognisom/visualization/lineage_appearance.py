"""What a cell of a given lineage is drawn as -- shape, size, colour -- in one place.

What was wrong
--------------
:func:`cognisom.visualization.run_usd.write_run` drew every cell of a run
as one of two plain spheres: ``/World/Prototypes/cell`` at a flat 10 um
and ``/World/Prototypes/immune`` at 8 um. A cancer cell, a normal cell,
a T cell, an NK cell and a macrophage differed only in the colour
primvar, and both diameters were invented in the exporter, beside a
comment saying that inventing them was a bad idea. Meanwhile three
other places each kept their own map from a type token to a library
shape: ``exporters.cell_mesh_type`` (ARCHETYPE_ENTITY plus an alias
table), ``scene_adapter.LIBRARY_ENTITY_FOR_PROTOTYPE`` (Scene Composer
prototype names) and ``usd_animation`` (its own Sphere per type and
state, and no callers). Four maps for one question.

What is now true
----------------
This module is the one answer. :func:`appearance_for` takes any token a
cell carries in this repository -- the engine's cellular ``normal`` /
``cancer``; the engine's immune ``T_cell`` / ``NK_cell`` /
``macrophage``, with the cell's polarization so that an M1 and an M2 are
the two library entities they are; the Bio-USD ``CellType`` tokens; and
the Scene Composer's prototype names -- and returns an
:class:`Appearance`: the library entity the cell is drawn as, that
entity's ``mesh_type``, its diameter and WHERE THE DIAMETER CAME FROM,
its colour, its ontology term, and a note saying why the mapping is what
it is. Sizes come only from
:func:`cognisom.visualization.cell_scene.resolved_scale_um`, so a
measured ``radius_um`` wins over the unprovenanced ``scale_um`` the
moment one lands in the library, and nothing in this file holds a
number for any cell.

As the library stands today (2026-10-27) the engine's lineages resolve
to the table below. It is :func:`appearance_table_text` pasted in, and a
test holds the two equal, so the table cannot drift from the library
again (an earlier copy of it stayed dated 2026-10-27 while
``seed_cell_size`` made six of the seven cited)::

    normal         Luminal epithelial cell        polarized_columnar   15.0 um  estimate: scale_um (no provenance)
    cancer         Prostatic adenocarcinoma cell  irregular_ruffled    10.7 um  radius_um PMID:23626725
    T_cell         CD8+ Effector T cell           smooth_sphere        11.3 um  radius_um PMID:39273504
    NK_cell        NK cell                        granular_sphere       8.3 um  radius_um PMID:21103395
    macrophage_M1  Macrophage M1                  ruffled_pseudopod    21.2 um  radius_um PMID:9400735
    macrophage_M2  Macrophage M2                  elongated_spindle    21.2 um  radius_um PMID:9400735
    macrophage     Macrophage                     ruffled_pseudopod    21.2 um  radius_um PMID:9400735

Six of seven cited; ``normal`` is the one estimate (the luminal cell's
``radius_um`` is a placeholder halved from a ``scale_um`` of 15, and
:mod:`cognisom.library.seed_cell_size` records why no diameter was
written). ``size_source`` says which, and :attr:`Appearance.cited` is
the test. A cited size also carries :attr:`Appearance.size_convention`,
the library's ``diameter_convention`` for that radius -- which matters,
because none of these measured a cell in a tissue section: the cancer
cell is a circulating tumour cell in blood, which the library's own
record calls a probable underestimate of the in-situ cell; the T cell
is a 72 h CD3/CD28 blast; the NK cell is IL-2-cultured and migrating;
the macrophage is alveolar. The convention travels into the stage's
record so a reader of a 10.7 um cancer cell drawn beside a 15 um
normal cell can see what the number is. An unpolarised macrophage and
an M1 are the same drawn object: the library gives ``Macrophage`` and
``Macrophage M1`` the same mesh, the same measured radius and the same
colour, so a cell crossing the -0.3 cutoff changes its appearance key
and the stage's record, not the picture. Crossing +0.3 changes the
picture (spindle for pseudopod).

Why ``normal`` is drawn as a luminal epithelial cell
----------------------------------------------------
In the still exporters ``normal`` is a state and a default -- what a
cell is called when nothing typed it -- and it rightly resolves to no
shape there (:data:`STATE_TOKENS`, :func:`mesh_type_for_cell_token`).
The engine's ``normal`` is not that. It is a prostate cell that runs
the androgen-receptor pathway and makes PSA, which is the luminal
secretory phenotype. Verified in the source on 2026-10-27:

* ``engine_runner._link_modules`` calls ``ar_signaling.add_cell(cell_id)``
  for every cell in ``cellular.cells``, normal and cancer alike, and
  ``ARSignalingModule._on_cell_divided`` adds every daughter with its
  parent's state.
* ``ARSignalingModule`` integrates "AR transcription, translation, DHT
  binding, PSA induction -- in every live cell with that DHT as its
  androgen supply" (its own docstring), and its reported state splits
  ``psa_total`` from ``psa_cancer_total`` precisely "because benign
  luminal cells make PSA".
* ``CellularModule._update_androgen`` sets every live cell's tissue DHT
  from the testosterone field (Titus 2005 at eugonadal supply).
* The string ``psa`` occurs nowhere in ``cellular_module.py`` or
  ``molecular_module.py``: PSA is a species of the AR module only.

And the limit of the claim: ``CellularModule._ar_signalling`` is read
only ``if cell.cell_type == 'cancer'`` (division slowing) and for the
``ar_dependent`` cancer subset (withdrawal death). A normal cell
synthesises PSA at its tissue DHT and its fate never hears about it. So
the engine's normal cell is AR-expressing and PSA-secreting but
hormone-indifferent in fate, and no AR-negative basal cell is modelled
at all. That is why the DRAWN shape and size are the luminal entity's
while the ``bio:cell:type`` token stays ``normal`` and its ontology term
stays unasserted: the appearance records "drawn as", not "is".

What a resolved size means on the lattice
-----------------------------------------
The tissue slab packs cells on an 11 um lattice (median nearest
neighbour 9.2 um), and on the last frame of the 2 h ring Baseline run
cells and immune cells together sit a median 4.3 um apart. Drawn at
their suspension diameters, 8-21 um cells overlap. ``write_run``
measures that overlap and reports it; nothing here shrinks a cell to
hide it.
"""

from __future__ import annotations

import contextlib
import functools
import io
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

log = logging.getLogger(__name__)

Colour = Tuple[float, float, float]

#: What an appearance with no builder instances. Named so a reader of
#: the stage can see which cells the library had no shape for.
UNRESOLVED_PROTOTYPE = "unresolved_sphere"

#: Diameter for a token the library has no row for. Openly an estimate
#: (``size_source`` says so); it replaces the two silent constants
#: ``run_usd`` used to carry for every cell and every immune cell.
ESTIMATE_DIAMETER_UM = 10.0

#: The lineages the engine produces, as the keys reports use.
ENGINE_LINEAGES: Tuple[str, ...] = (
    "normal", "cancer", "T_cell", "NK_cell",
    "macrophage_M1", "macrophage_M2", "macrophage")

#: Scene Composer prototype name -> the cell library's own name for it.
#:
#: This is what turns an exported composed scene from one sphere
#: prototype into real cell shapes. The composer's prototype library
#: knows three geometries -- sphere, capsule, ellipsoid -- while the
#: cell library carries a built morphology per entity, so going through
#: the name gets a neutrophil its multilobed nucleus and a dendritic
#: cell its branches:
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

#: Exporter and Bio-USD tokens that are another archetype under a
#: different name. ``immune`` names no lineage finer than leukocyte and
#: is drawn as the T-cell archetype; ``stem`` as the malignant
#: epithelium; ``luminal`` IS the luminal_secretory archetype, which the
#: old alias table did not know, so a Bio-USD ``luminal`` cell was a
#: sphere.
CELL_TYPE_MESH_ALIASES: Dict[str, str] = {
    "cancer": "cancer_epithelial",
    "stem": "cancer_epithelial",
    "immune": "t_cell",
    "luminal": "luminal_secretory",
}

#: Tokens the still exporters' vocabulary uses for what a cell is DOING,
#: not what kind of cell it is. A state has no morphology, so these
#: resolve to no shape in :func:`mesh_type_for_cell_token`. ``normal``
#: is among them THERE because in ``cells_to_usd`` and ``write_cells`` it
#: is also the default for an untyped cell (``_token_of`` and
#: ``SceneCell`` both fall back to it), and giving the default bucket a
#: lineage's shape would draw every unknown thing as a luminal cell.
STATE_TOKENS = frozenset({"normal", "dead", "dividing", "differentiated", "progenitor"})

#: Coarse Bio-USD token per tissue archetype, for the colour fallback
#: and the type-level ontology term.
_COARSE_FOR_ARCHETYPE: Dict[str, str] = {
    "luminal_secretory": "luminal",
    "basal": "basal",
    "fibroblast": "fibroblast",
    "endothelial": "endothelial",
    "neuroendocrine": "neuroendocrine",
    "macrophage": "immune",
    "t_cell": "immune",
    "nk_cell": "immune",
    "cancer_epithelial": "cancer",
    "smooth_muscle": "stromal",
}

_NORMAL_NOTE = (
    "drawn as the luminal secretory cell: the engine's normal cell runs the "
    "AR pathway and makes PSA (ARSignalingModule registers every cell; "
    "CellularModule reads AR signalling only for cancer cells, so a normal "
    "cell's fate ignores it). bio:cell:type stays 'normal' and no ontology "
    "term is asserted -- this is what it is drawn as, not what it is.")


@dataclass(frozen=True)
class Appearance:
    """What one lineage is drawn as, and where each part of that came from."""

    #: The lineage label reports use: 'normal', 'cancer', 'T_cell',
    #: 'NK_cell', 'macrophage_M1', 'macrophage_M2', 'macrophage', a
    #: Bio-USD token, or a composer prototype name.
    key: str
    entity_name: Optional[str]
    entity_type: Optional[str]
    mesh_type: Optional[str]
    has_builder: bool
    diameter_um: float
    #: ``radius_um <citation>`` when the library carries a provenanced
    #: radius, else ``estimate: ...`` saying what the number really is.
    size_source: str
    colour: Colour
    ontology_term: str
    note: str
    #: The library's ``diameter_convention`` for the radius the size came
    #: from -- what was measured, on what, in what preparation -- and
    #: empty when the size is not from a ``radius_um``. A cited size
    #: without this is a number with no way to tell a circulating tumour
    #: cell from a cell in tissue.
    size_convention: str = ""
    #: The radius's provenance grade (``measured``, ``derived``, ...) when
    #: the size came from ``radius_um``, else empty. ``size_source``
    #: carries the citation and not the grade, so without this a DERIVED
    #: or ESTIMATED radius with a PMID would be indistinguishable from a
    #: measurement once it reached the stage.
    size_grade: str = ""

    @property
    def cited(self) -> bool:
        """Whether the size traces to a citation rather than a guess."""
        return is_citation(self.size_source)

    @property
    def prototype(self) -> str:
        """The prototype prim name this appearance instances."""
        return self.mesh_type if self.has_builder and self.mesh_type else UNRESOLVED_PROTOTYPE

    @property
    def radius_um(self) -> float:
        return self.diameter_um / 2.0

    def describe(self) -> str:
        where = self.size_source.split(" ", 1)[1] if self.cited else "est"
        text = f"{self.key} {self.prototype} {self.diameter_um:.1f} um ({where})"
        if self.size_convention:
            text += f" -- {self.size_convention}"
        return text

    def as_dict(self) -> Dict[str, Any]:
        return {
            "entity": self.entity_name,
            "entity_type": self.entity_type,
            "mesh_type": self.mesh_type,
            "prototype": self.prototype,
            "diameter_um": float(self.diameter_um),
            "size_source": self.size_source,
            "size_grade": self.size_grade,
            "size_convention": self.size_convention,
            "cited": self.cited,
            "colour": [float(c) for c in self.colour],
            "ontology_term": self.ontology_term,
            "note": self.note,
        }


@dataclass(frozen=True)
class _Mapping:
    key: str
    entity_name: str
    entity_type: Optional[str]
    #: Whether the entity IS the cell (its ontology term may be asserted)
    #: or only what the cell is drawn as (the token's own term is kept).
    asserted: bool
    coarse: str
    note: str


def is_citation(size_source: Optional[str]) -> bool:
    """``radius_um PMID:...`` is a citation; ``radius_um placeholder``,
    ``scale_um (no provenance)`` and anything ``estimate: ...`` are not.

    :func:`~cognisom.visualization.cell_scene.resolved_scale_um` writes
    ``radius_um`` followed by the parameter's citation when it has one
    and by its provenance grade when it does not, so the grade words are
    the things to exclude.
    """
    head, _, rest = str(size_source or "").partition(" ")
    rest = rest.strip()
    if head != "radius_um" or not rest:
        return False
    try:
        from cognisom.library.parameters import Provenance
        grades = {p.value for p in Provenance}
    except Exception:                                # noqa: BLE001
        grades = {"placeholder", "estimated", "calibrated", "derived", "measured"}
    return rest.lower() not in grades


def _normalise(token: Any) -> str:
    if token is None:
        return ""
    return str(getattr(token, "value", token)).strip().lower()


@functools.lru_cache(maxsize=1)
def _store():
    """One store, opened quietly; the resolver runs once per cell per
    frame otherwise. ``None`` when the library cannot be opened, which
    every caller treats as "no entity" rather than an error."""
    try:
        from cognisom.library.store import EntityStore
        with contextlib.redirect_stdout(io.StringIO()):
            return EntityStore()
    except Exception as exc:                         # noqa: BLE001
        log.warning("entity library unavailable; every cell is a sphere: %s", exc)
        return None


@functools.lru_cache(maxsize=1)
def _mappings() -> Dict[str, _Mapping]:
    """Lower-cased token -> mapping, built once from the three vocabularies."""
    from cognisom.core.tissue_config import ARCHETYPE_ENTITY

    table: Dict[str, _Mapping] = {}

    def archetype(token: str, key: str, name: str, coarse: str, note: str,
                  asserted: bool = True) -> None:
        entity_name, entity_type = ARCHETYPE_ENTITY[name]
        table[token] = _Mapping(key, entity_name, entity_type, asserted, coarse, note)

    # The engine's cellular tokens.
    table["normal"] = _Mapping("normal", "Luminal epithelial cell", "cell_type",
                               False, "normal", _NORMAL_NOTE)
    # asserted=False on purpose: the entity's own term in CELL_CURIE is
    # NCIT:C2919, which is the DISEASE (prostate adenocarcinoma), and
    # terms.py keeps disease out of the identity slot. The stage's
    # bio:cell:ontologyTerm for the engine's 'cancer' token is the
    # coarse malignant-cell term CL:0001064, so the record says the same
    # thing the primvar does instead of a second, different term.
    archetype("cancer", "cancer", "cancer_epithelial", "cancer",
              "the engine's cancer cell: tissue_config's cancer_epithelial archetype, "
              "drawn as the prostatic adenocarcinoma entity; the term is the coarse "
              "malignant-cell term the stage asserts (CL:0001064), not the entity's "
              "NCIT:C2919, which names the disease",
              asserted=False)
    # The engine's immune tokens. The macrophage splits on polarization
    # at the cell2sentence cutoffs (+/-0.3), see appearance_for.
    archetype("t_cell", "T_cell", "t_cell", "immune",
              "the engine's T cell kills through TCR-pMHC recognition: the CD8+ effector archetype")
    archetype("nk_cell", "NK_cell", "nk_cell", "immune",
              "tissue_config's nk_cell archetype")
    archetype("macrophage", "macrophage", "macrophage", "immune",
              "unpolarised or mixed macrophage (|polarization| <= 0.3), or no polarization recorded")
    table["macrophage_m1"] = _Mapping("macrophage_M1", "Macrophage M1", "immune_cell", True,
                                      "immune", "macrophage with polarization < -0.3")
    table["macrophage_m2"] = _Mapping("macrophage_M2", "Macrophage M2", "immune_cell", True,
                                      "immune", "macrophage with polarization > +0.3")
    # Every tissue archetype under its own key, and the Bio-USD tokens
    # that coincide with one (basal, endothelial, fibroblast,
    # neuroendocrine).
    for name, (entity_name, entity_type) in ARCHETYPE_ENTITY.items():
        table.setdefault(name, _Mapping(name, entity_name, entity_type, True,
                                        _COARSE_FOR_ARCHETYPE.get(name, "normal"),
                                        f"tissue_config archetype {name}"))
    # Aliases: drawn as another archetype, the token's own term kept.
    for token, name in CELL_TYPE_MESH_ALIASES.items():
        if token in table:
            continue
        base = table[name]
        asserted = token == "luminal"
        table[token] = _Mapping(token, base.entity_name, base.entity_type, asserted, token,
                                f"drawn as the {name} archetype" if not asserted
                                else "the luminal_secretory archetype")
    # ``stromal`` names a compartment, not a cell. Both stromal archetypes
    # (fibroblast, smooth muscle) are elongated_spindle at 25 um in the
    # library, so the fibroblast's shape is right whichever it is.
    fib = table["fibroblast"]
    table["stromal"] = _Mapping("stromal", fib.entity_name, fib.entity_type, False, "stromal",
                                "drawn as the stromal fibroblast: both stromal archetypes are "
                                "elongated_spindle at 25 um; the token's own term is kept")
    # Scene Composer prototype names.
    for proto, entity_name in LIBRARY_ENTITY_FOR_PROTOTYPE.items():
        coarse = ("cancer" if proto == "tumor_cell"
                  else "endothelial" if proto == "endothelial_cell" else "immune")
        table.setdefault(proto, _Mapping(proto, entity_name, None, True, coarse,
                                         f"Scene Composer prototype {proto}"))
    return table


def _find_entity(name: str, entity_type: Optional[str]):
    store = _store()
    if store is None:
        return None
    try:
        entity = store.find_entity_by_name(name, entity_type) if entity_type else None
        if entity is None:
            # "NK cell" exists as two rows (cell_type and immune_cell);
            # the typed lookup picks the archetype's, this picks any.
            entity = store.find_entity_by_name(name)
    except Exception as exc:                         # noqa: BLE001
        log.warning("library lookup failed for %r: %s", name, exc)
        return None
    return entity


def _colour_of(entity, coarse: str) -> Colour:
    from cognisom.biousd.usd_writer import TYPE_COLOUR

    rgb = getattr(entity, "color_rgb", None) if entity is not None else None
    if rgb is not None and len(tuple(rgb)) >= 3:
        return tuple(float(c) for c in tuple(rgb)[:3])
    return tuple(TYPE_COLOUR.get(coarse, TYPE_COLOUR["normal"]))


def _term_for(name: Optional[str], coarse: str) -> str:
    try:
        from cognisom.biousd.standard.terms import curie_for
        return curie_for(name, cell_type=coarse) or ""
    except Exception:                                # noqa: BLE001
        return ""


def _fallback(key: str, coarse: str, reason: str, report_key: Optional[str] = None) -> Appearance:
    return Appearance(
        key=report_key or key or "unknown", entity_name=None, entity_type=None,
        mesh_type=None, has_builder=False, diameter_um=ESTIMATE_DIAMETER_UM,
        size_source=f"estimate: {ESTIMATE_DIAMETER_UM:g} um fallback, {reason}",
        colour=_colour_of(None, coarse), ontology_term=_term_for(None, coarse),
        note=reason)


@functools.lru_cache(maxsize=512)
def _resolve(key: str) -> Appearance:
    from cognisom.visualization.cell_scene import resolved_scale_um
    from cognisom.visualization.morphology import BUILDERS

    mapping = _mappings().get(key)
    if mapping is None:
        return _fallback(key, key, f"no lineage mapping for token {key!r}")
    entity = _find_entity(mapping.entity_name, mapping.entity_type)
    if entity is None:
        return _fallback(key, mapping.coarse,
                         f"the library has no row named {mapping.entity_name!r}",
                         report_key=mapping.key)
    diameter, source = resolved_scale_um(entity)
    grade, convention = "", ""
    if source.startswith("radius_um"):
        # The size is the library's radius: carry its grade and the
        # convention the sourcing pass wrote beside it. A size from
        # scale_um has neither, whatever the row's convention says about
        # a radius that was not used.
        try:
            from cognisom.library.parameters import get_parameter
            radius = get_parameter(entity, "radius_um")
            grade = radius.provenance.value if radius is not None else ""
        except Exception:                            # noqa: BLE001
            grade = ""
        sourced = getattr(entity, "sourced", None) or {}
        convention = str(sourced.get("diameter_convention") or "")
    mesh_type = getattr(entity, "mesh_type", None) or None
    etype = getattr(entity, "entity_type", None)
    etype = getattr(etype, "value", etype)
    return Appearance(
        key=mapping.key,
        entity_name=mapping.entity_name,
        entity_type=str(etype) if etype else mapping.entity_type,
        mesh_type=mesh_type,
        has_builder=bool(mesh_type) and mesh_type in BUILDERS,
        diameter_um=float(diameter),
        size_source=source if is_citation(source) else f"estimate: {source}",
        colour=_colour_of(entity, mapping.coarse),
        ontology_term=_term_for(mapping.entity_name if mapping.asserted else None,
                                mapping.coarse),
        note=mapping.note,
        size_convention=convention,
        size_grade=grade)


def appearance_for(token: Any, *, polarization: Optional[float] = None,
                   activated: Optional[bool] = None) -> Appearance:
    """The :class:`Appearance` for a cell carrying ``token``.

    Accepted, case-insensitively: the engine's cellular ``normal`` /
    ``cancer``; the engine's immune ``T_cell`` / ``NK_cell`` /
    ``macrophage`` (a macrophage with a ``polarization`` is labelled by
    :func:`~cognisom.engine.py.immune.macrophage_polarization.polarization_label`
    -- ``M1`` is "Macrophage M1", ``M2`` is "Macrophage M2", ``mixed`` or
    no polarization is "Macrophage"); the report keys ``macrophage_M1`` /
    ``macrophage_M2``; the Bio-USD ``CellType`` tokens; every
    ``ARCHETYPE_ENTITY`` key; and the Scene Composer prototype names. An
    unknown token gets no entity, no mesh, a fallback diameter and a
    ``size_source`` that begins ``estimate:``.

    ``activated`` is accepted and does not change the picture: no
    sourced morphology distinguishes an engaged killer from a patrolling
    one, and the state travels as its own primvar
    (``cognisom:immune:activated``).
    """
    key = _normalise(token)
    if key == "macrophage" and polarization is not None:
        from cognisom.engine.py.immune.macrophage_polarization import polarization_label
        label = polarization_label(float(polarization))
        if label == "M1":
            key = "macrophage_m1"
        elif label == "M2":
            key = "macrophage_m2"
    return _resolve(key)


def mesh_type_for_cell_token(token: Any) -> Optional[str]:
    """The still exporters' question: a library mesh with a builder, or ``None``.

    ``None`` for a :data:`STATE_TOKENS` entry -- including ``normal``,
    which in ``cells_to_usd`` / ``write_cells`` is also the default for an
    untyped cell -- and for any token whose entity has no buildable
    mesh. The run exporter does not go through here: its ``normal`` is
    the engine's asserted token, see the module docstring.
    """
    key = _normalise(token)
    if not key or key in STATE_TOKENS:
        return None
    appearance = _resolve(key)
    return appearance.mesh_type if appearance.has_builder else None


def diameter_for_cell_token(token: Any) -> Optional[Tuple[float, str]]:
    """The still exporters' other question: the library's diameter for
    a typed cell and where it came from, or ``None``.

    ``None`` for a :data:`STATE_TOKENS` entry (``normal`` among them,
    for the reason :func:`mesh_type_for_cell_token` gives) and for a
    token the library has no row for. Otherwise ``(diameter_um,
    size_source)`` exactly as the run exporter draws and records the
    same lineage, so a ``BioCell`` that reaches ``write_cells`` with a
    type and no ``scale_um`` is the size of the same cell in a run
    export rather than a 10 um cell of the right shape.
    """
    key = _normalise(token)
    if not key or key in STATE_TOKENS:
        return None
    appearance = _resolve(key)
    if appearance.entity_name is None:
        return None
    return appearance.diameter_um, appearance.size_source


def engine_appearances() -> Dict[str, Appearance]:
    """Every lineage the engine produces, keyed as reports key them."""
    return {key: appearance_for(key) for key in ENGINE_LINEAGES}


def appearance_table(tokens: Optional[Iterable[Any]] = None) -> Dict[str, Dict[str, Any]]:
    """Report form: key -> the appearance as a plain dict. Defaults to the
    engine lineages."""
    if tokens is None:
        apps = list(engine_appearances().values())
    else:
        apps = [appearance_for(t) for t in tokens]
    return {a.key: a.as_dict() for a in apps}


def appearance_table_text(tokens: Optional[Iterable[Any]] = None) -> str:
    """The module docstring's table, regenerated from the library: one
    line per lineage -- key, entity, prototype, diameter, size source.

    A test holds the docstring's table equal to this (whitespace aside),
    which is what keeps a dated table honest: when the library changes,
    the test fails until the docstring is pasted over.
    """
    lines = []
    for key, rec in appearance_table(tokens).items():
        lines.append(f"{key:<14} {rec['entity'] or '(no entity)':<30} "
                     f"{rec['prototype']:<19} {rec['diameter_um']:5.1f} um  "
                     f"{rec['size_source']}")
    return "\n".join(lines)


def clear_cache() -> None:
    """Forget resolved appearances (and the store), e.g. after the
    library changes under a running process."""
    _resolve.cache_clear()
    _mappings.cache_clear()
    _store.cache_clear()
