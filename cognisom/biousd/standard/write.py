"""The minimal Bio-USD writer: cells in, a conforming stage out.

Deliberately small. It draws every cell as a unit sphere so that the
convention can be written to by anyone with ``usd-core`` and nothing
else; a tool with real morphology authors its own prototypes and sets
``protoIndices`` itself. The primvars it writes are exactly
:data:`~.vocabulary.CELL_PRIMVARS`, read from that table, so this
writer cannot author something the validator does not check.

Cells may be objects or mappings. Fields read, each defensively:
``position`` (3, um), ``cell_id``, ``cell_type``, ``ontology_term``,
``phase``, ``age``, ``alive``, ``volume``, ``scale_um`` (the diameter,
um) and optionally ``color`` (3, 0-1).

Since 0.2.0 the instancer also applies the ``BioCellAPI`` schema and the
default prim carries UsdProfiles claims, both authored as plain metadata
that needs no plugin to write (:mod:`.api_schema`, :mod:`.profiles`).
Every primvar is still authored with its type and ``varying``
interpolation explicitly, so a reader without the plugin sees exactly
the file a 0.1 writer produced plus two metadata entries it ignores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .api_schema import apply_api
from .profiles import write_claims
from .vocabulary import (
    CELL_PHASES, CELL_PRIMVARS, CELL_TYPES, METERS_PER_UNIT, VERSION, is_curie)

DEFAULT_INSTANCER = "/World/Cells"
DEFAULT_PROTOTYPE = "/Prototypes/cell"


@dataclass
class WriteReport:
    path: str
    cells: int = 0
    primvars: List[str] = field(default_factory=list)
    #: Ontology terms that were not well-formed CURIEs and were written
    #: as empty instead, by original value.
    rejected_terms: List[str] = field(default_factory=list)
    #: ``BioCellAPI`` was applied to the instancer.
    api_applied: bool = False
    #: The prim the UsdProfiles claims were written to (the default prim).
    claims_prim: Optional[str] = None


def _get(cell: Any, name: str, default: Any = None) -> Any:
    if isinstance(cell, dict):
        return cell.get(name, default)
    return getattr(cell, name, default)


def _token(value: Any, allowed: Sequence[str], default: str) -> str:
    text = str(getattr(value, "value", value) or default)
    return text if text in allowed else default


def write_cells(path: str, cells: Iterable[Any], *,
                up_axis: str = "Y",
                instancer_path: str = DEFAULT_INSTANCER,
                prototype_path: str = DEFAULT_PROTOTYPE,
                colour_by: Optional[str] = None,
                colour_range: Optional[Sequence[float]] = None
                ) -> WriteReport:
    """Write ``cells`` to ``path`` as a conforming Bio-USD stage."""
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    cells = list(cells)
    report = WriteReport(path=path, cells=len(cells))

    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, getattr(UsdGeom.Tokens, up_axis.lower()))
    UsdGeom.SetStageMetersPerUnit(stage, METERS_PER_UNIT)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    layer_data: Dict[str, Any] = {"bio:version": VERSION}
    if colour_by:
        layer_data["bio:colourBy"] = str(colour_by)
        if colour_range is not None and len(colour_range) == 2:
            layer_data["bio:colourMin"] = float(colour_range[0])
            layer_data["bio:colourMax"] = float(colour_range[1])
    stage.GetRootLayer().customLayerData = layer_data

    # A unit-diameter sphere: the instance scale is then the cell's
    # diameter in micrometres, which is what the vocabulary says.
    sphere = UsdGeom.Sphere.Define(stage, prototype_path)
    sphere.GetRadiusAttr().Set(0.5)

    instancer = UsdGeom.PointInstancer.Define(stage, instancer_path)
    instancer.GetPrototypesRel().SetTargets([sphere.GetPath()])

    n = len(cells)
    positions = np.array([tuple(float(v) for v in _get(c, "position", (0, 0, 0)))
                          for c in cells], dtype="float32").reshape(n, 3)
    diameters = np.array([float(_get(c, "scale_um", 10.0) or 10.0) for c in cells],
                         dtype="float32")
    instancer.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(positions))
    instancer.GetProtoIndicesAttr().Set(Vt.IntArray.FromNumpy(
        np.zeros(n, dtype="int32")))
    instancer.GetScalesAttr().Set(Vt.Vec3fArray.FromNumpy(
        np.repeat(diameters[:, None], 3, axis=1).astype("float32")))
    if n:
        lo = (positions - diameters[:, None] / 2).min(axis=0)
        hi = (positions + diameters[:, None] / 2).max(axis=0)
        instancer.CreateExtentAttr().Set([Gf.Vec3f(*map(float, lo)),
                                          Gf.Vec3f(*map(float, hi))])

    api = UsdGeom.PrimvarsAPI(instancer.GetPrim())
    varying = UsdGeom.Tokens.varying

    def author(name: str, type_name, values) -> None:
        pv = api.CreatePrimvar(name, type_name, varying)
        pv.Set(values)
        report.primvars.append(name)

    values_by_name: Dict[str, Any] = {}
    values_by_name["bio:cell:id"] = Vt.IntArray.FromNumpy(np.array(
        [int(_get(c, "cell_id", i) if _get(c, "cell_id", None) is not None else i)
         for i, c in enumerate(cells)], dtype="int32"))
    values_by_name["bio:cell:type"] = Vt.TokenArray(
        [_token(_get(c, "cell_type", "normal"), CELL_TYPES, "normal") for c in cells])
    terms: List[str] = []
    for c in cells:
        term = str(_get(c, "ontology_term", _get(c, "ontologyTerm", "")) or "")
        if not is_curie(term):
            report.rejected_terms.append(term)
            term = ""
        terms.append(term)
    values_by_name["bio:cell:ontologyTerm"] = Vt.TokenArray(terms)
    values_by_name["bio:cell:phase"] = Vt.TokenArray(
        [_token(_get(c, "phase", "G1"), CELL_PHASES, "G1") for c in cells])
    values_by_name["bio:cell:age"] = Vt.FloatArray.FromNumpy(np.array(
        [max(0.0, float(_get(c, "age", 0.0) or 0.0)) for c in cells], dtype="float32"))
    values_by_name["bio:cell:alive"] = Vt.BoolArray(
        [bool(_get(c, "alive", True)) for c in cells])
    values_by_name["bio:cell:volume"] = Vt.FloatArray.FromNumpy(np.array(
        [max(0.0, float(_get(c, "volume", 1.0) or 1.0)) for c in cells], dtype="float32"))

    type_names = {
        "int[]": Sdf.ValueTypeNames.IntArray,
        "token[]": Sdf.ValueTypeNames.TokenArray,
        "float[]": Sdf.ValueTypeNames.FloatArray,
        "bool[]": Sdf.ValueTypeNames.BoolArray,
    }
    for primvar in CELL_PRIMVARS:
        author(primvar.name, type_names[primvar.usd_type], values_by_name[primvar.name])

    colours = [_get(c, "color", None) for c in cells]
    if n and all(col is not None for col in colours):
        api.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, varying).Set(
            Vt.Vec3fArray.FromNumpy(np.array(colours, dtype="float32").reshape(n, 3)))

    # The schema is applied AFTER the primvars are authored with their
    # explicit interpolation, so a reader without the plugin sees no
    # difference; and the claims say what the stage uses: cells always,
    # ontology terms when at least one cell asserts one.
    report.api_applied = apply_api(instancer.GetPrim())
    report.claims_prim = write_claims(stage, ontology=any(terms))

    stage.GetRootLayer().Save()
    return report
