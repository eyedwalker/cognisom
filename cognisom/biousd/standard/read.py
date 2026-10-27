"""Read a Bio-USD stage back into arrays.

The reader is what makes the convention checkable: a round trip through
:func:`~.write.write_cells` and this must return what went in, and the
validator is built on it. It reads any ``UsdGeom.PointInstancer`` that
authors ``bio:cell:id`` or applies ``BioCellAPI``; a stage may hold
several populations.

AUTHORED, NOT DEFINED
---------------------
The 0.1 reader gated each primvar on ``PrimvarsAPI.HasPrimvar()``, which
is ``IsDefined()``. Once the API schema plugin is loaded, a primvar the
schema declares is *defined* on every prim that applies the schema,
authored or not: the reader saw ``bio:cell:volume`` on a file that never
wrote it, read ``None``, and the validator reported ``primvar.length``
with exit 1 on a conforming file -- and only when the plugin was present,
so the same file passed and failed depending on the process. Every gate
here is on what the file authors (``GetAuthoredPrimvars()``); the
interpolation is read as it resolves (authored metadata, else the
schema's fallback, else USD's ``constant``) together with whether it was
authored, so the validator can say which.

AUTHORED TYPE, NOT RESOLVED
---------------------------
The same regression's mirror image: with the plugin loaded,
``Primvar.GetTypeName()`` resolves to the type the schema declares, so a
file authoring ``int[] primvars:bio:cell:age`` read back as ``float[]``
and the validator passed it -- and the same file failed
(``primvar.type``) in a process without the plugin (measured on 25.11
and 26.11; the value came back a ``Vt.IntArray`` either way). The type
recorded here is the first authored ``typeName`` in the attribute's
property stack, which the registry does not touch.

The applied-schema list is read from ``GetPrimTypeInfo()``, which keeps
every name the file authors; ``GetAppliedSchemas()`` drops the ones this
process's registry does not know, which is ``BioCellAPI`` itself in any
process without the plugin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .api_schema import applied_api_schemas
from .profiles import ClaimsInfo, read_claims
from .vocabulary import API_SCHEMA, CELL_PRIMVARS


@dataclass
class CellTable:
    """One population, as read."""

    path: str
    instancer: str
    n: int
    positions: np.ndarray                 # (n, 3) um
    diameters: np.ndarray                 # (n,) um, from the instance scale's x
    proto_indices: np.ndarray             # (n,)
    primvars: Dict[str, Any] = field(default_factory=dict)
    #: primvar name -> interpolation as it resolves (authored, else fallback)
    interpolation: Dict[str, str] = field(default_factory=dict)
    #: primvar name -> whether ``interpolation`` is authored in the file;
    #: a reader without the plugin sees only authored metadata
    interpolation_authored: Dict[str, bool] = field(default_factory=dict)
    #: primvar name -> Sdf type name as authored (from the property
    #: stack, not the resolved type the schema would supply)
    type_names: Dict[str, str] = field(default_factory=dict)
    invisible_ids: List[int] = field(default_factory=list)
    prototypes: List[str] = field(default_factory=list)
    #: API schemas authored on the carrier, known to this process or not
    applied_schemas: List[str] = field(default_factory=list)
    #: ``BioCellAPI`` is among them
    api_applied: bool = False

    def __getitem__(self, name: str) -> Any:
        return self.primvars[name]


@dataclass
class StageInfo:
    path: str
    up_axis: str
    meters_per_unit: float
    layer_data: Dict[str, Any]
    populations: List[CellTable]
    default_prim: Optional[str] = None
    #: (prim path, prim type) of prims applying ``BioCellAPI`` that are
    #: not PointInstancers; USD lets that happen, the validator does not
    misapplied: List[Tuple[str, str]] = field(default_factory=list)
    #: (prim path, identifier) of ``BioCellAPI_<n>`` applications, a
    #: schema-family version this reader does not know
    other_versions: List[Tuple[str, str]] = field(default_factory=list)
    #: UsdProfiles claims on the default prim, as metadata
    claims: Optional[ClaimsInfo] = None


def read_population(prim, path: Optional[str] = None) -> Optional[CellTable]:
    """The cell population on ``prim``, or ``None`` when it is not one.

    A prim is a cell population when it is a PointInstancer that
    authors ``bio:cell:id`` or applies ``BioCellAPI``; applying the
    schema is a claim, and a carrier that makes it without the required
    primvars is reported as missing them rather than ignored.
    """
    from pxr import Usd, UsdGeom

    if not prim.IsA(UsdGeom.PointInstancer):
        return None
    applied = applied_api_schemas(prim)
    api_applied = API_SCHEMA in applied
    api = UsdGeom.PrimvarsAPI(prim)
    authored = {pv.GetPrimvarName(): pv for pv in api.GetAuthoredPrimvars()}
    if "bio:cell:id" not in authored and not api_applied:
        return None
    if path is None:
        path = prim.GetStage().GetRootLayer().identifier

    instancer = UsdGeom.PointInstancer(prim)
    positions = np.asarray(instancer.GetPositionsAttr().Get() or [],
                           dtype="float32").reshape(-1, 3)
    scales = instancer.GetScalesAttr().Get()
    diameters = (np.asarray(scales, dtype="float32")[:, 0]
                 if scales is not None and len(scales) else np.ones(len(positions), "float32"))
    proto = np.asarray(instancer.GetProtoIndicesAttr().Get() or [], dtype="int32")
    table = CellTable(path=path, instancer=str(prim.GetPath()), n=len(positions),
                      positions=positions, diameters=diameters, proto_indices=proto,
                      applied_schemas=applied, api_applied=api_applied)
    for primvar in CELL_PRIMVARS:
        pv = authored.get(primvar.name)
        if pv is None:
            continue
        value = pv.Get()
        table.primvars[primvar.name] = list(value) if value is not None else []
        table.interpolation[primvar.name] = str(pv.GetInterpolation())
        table.interpolation_authored[primvar.name] = bool(
            pv.GetAttr().HasAuthoredMetadata("interpolation"))
        specs = pv.GetAttr().GetPropertyStack(Usd.TimeCode.Default())
        table.type_names[primvar.name] = next(
            (str(s.typeName) for s in specs if getattr(s, "typeName", None)),
            str(pv.GetTypeName()))
    invisible = instancer.GetInvisibleIdsAttr().Get()
    table.invisible_ids = [int(i) for i in invisible] if invisible else []
    table.prototypes = [str(p) for p in instancer.GetPrototypesRel().GetTargets()]
    return table


def read_stage_info(stage, path: Optional[str] = None) -> StageInfo:
    """Every cell population in an open stage, plus the stage metadata."""
    from pxr import Usd, UsdGeom

    if path is None:
        path = stage.GetRootLayer().identifier
    default_prim = stage.GetDefaultPrim()
    info = StageInfo(path=path, up_axis=str(UsdGeom.GetStageUpAxis(stage)),
                     meters_per_unit=float(UsdGeom.GetStageMetersPerUnit(stage)),
                     layer_data=dict(stage.GetRootLayer().customLayerData or {}),
                     populations=[],
                     default_prim=str(default_prim.GetPath()) if default_prim else None)
    parse = Usd.SchemaRegistry.ParseSchemaFamilyAndVersionFromIdentifier
    for prim in stage.Traverse():
        for name in applied_api_schemas(prim):
            family, version = parse(name)
            if family != API_SCHEMA:
                continue
            if version != 0:
                info.other_versions.append((str(prim.GetPath()), name))
            elif not prim.IsA(UsdGeom.PointInstancer):
                info.misapplied.append((str(prim.GetPath()),
                                        str(prim.GetTypeName()) or "(untyped)"))
        table = read_population(prim, path)
        if table is not None:
            info.populations.append(table)
    info.claims = read_claims(stage)
    return info


def read_stage(path: str) -> StageInfo:
    """Every cell population in the stage at ``path``, plus the stage metadata."""
    from pxr import Usd

    stage = Usd.Stage.Open(path)
    if stage is None:
        raise OSError(f"could not open {path!r} as a USD stage")
    return read_stage_info(stage, path)


def read_cells(path: str) -> Optional[CellTable]:
    """The first cell population in the stage, or ``None``."""
    info = read_stage(path)
    return info.populations[0] if info.populations else None
