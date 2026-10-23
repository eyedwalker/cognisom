"""Read a Bio-USD stage back into arrays.

The reader is what makes the convention checkable: a round trip through
:func:`~.write.write_cells` and this must return what went in, and the
validator is built on it. It reads any ``UsdGeom.PointInstancer`` that
carries ``bio:cell:id``; a stage may hold several populations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .vocabulary import CELL_PRIMVARS


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
    #: primvar name -> declared interpolation token, for those present
    interpolation: Dict[str, str] = field(default_factory=dict)
    #: primvar name -> Sdf type name as authored
    type_names: Dict[str, str] = field(default_factory=dict)
    invisible_ids: List[int] = field(default_factory=list)
    prototypes: List[str] = field(default_factory=list)

    def __getitem__(self, name: str) -> Any:
        return self.primvars[name]


@dataclass
class StageInfo:
    path: str
    up_axis: str
    meters_per_unit: float
    layer_data: Dict[str, Any]
    populations: List[CellTable]


def read_stage(path: str) -> StageInfo:
    """Every cell population in the stage, plus the stage metadata."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(path)
    if stage is None:
        raise OSError(f"could not open {path!r} as a USD stage")
    up_axis = str(UsdGeom.GetStageUpAxis(stage))
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
    layer_data = dict(stage.GetRootLayer().customLayerData or {})

    populations: List[CellTable] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.PointInstancer):
            continue
        instancer = UsdGeom.PointInstancer(prim)
        api = UsdGeom.PrimvarsAPI(prim)
        if not api.HasPrimvar("bio:cell:id"):
            continue
        positions = np.asarray(instancer.GetPositionsAttr().Get() or [], dtype="float32").reshape(-1, 3)
        scales = instancer.GetScalesAttr().Get()
        diameters = (np.asarray(scales, dtype="float32")[:, 0]
                     if scales is not None and len(scales) else np.ones(len(positions), "float32"))
        proto = np.asarray(instancer.GetProtoIndicesAttr().Get() or [], dtype="int32")
        table = CellTable(path=path, instancer=str(prim.GetPath()), n=len(positions),
                          positions=positions, diameters=diameters, proto_indices=proto)
        for primvar in CELL_PRIMVARS:
            if not api.HasPrimvar(primvar.name):
                continue
            pv = api.GetPrimvar(primvar.name)
            value = pv.Get()
            table.primvars[primvar.name] = list(value) if value is not None else []
            table.interpolation[primvar.name] = str(pv.GetInterpolation())
            table.type_names[primvar.name] = str(pv.GetTypeName())
        invisible = instancer.GetInvisibleIdsAttr().Get()
        table.invisible_ids = [int(i) for i in invisible] if invisible else []
        table.prototypes = [str(p) for p in instancer.GetPrototypesRel().GetTargets()]
        populations.append(table)

    return StageInfo(path=path, up_axis=up_axis, meters_per_unit=mpu,
                     layer_data=layer_data, populations=populations)


def read_cells(path: str) -> Optional[CellTable]:
    """The first cell population in the stage, or ``None``."""
    info = read_stage(path)
    return info.populations[0] if info.populations else None
