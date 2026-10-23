"""Does this file conform to Bio-USD? Checks exactly what SPEC.md says.

    python -m biousd.validate scene.usdc            (after extraction)
    python -m cognisom.biousd.standard.validate scene.usdc

Exit status 0 when the stage conforms, 1 when it does not, 2 when it
could not be read. Every finding is a line with a level, a code and a
message, so a CI job can grep it and a person can read it.

A validator is the difference between a convention and a document: a
third party cannot know whether their file is right without one, and
this project's earlier validator checked Python objects, not files, so
it could not have told them.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import List, Sequence

from .read import StageInfo, read_stage
from .vocabulary import (
    CELL_PRIMVARS, METERS_PER_UNIT, PRIMVARS_BY_NAME, REQUIRED_PRIMVARS,
    VERSION, is_curie)


@dataclass
class Issue:
    level: str          # "error" | "warning"
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.level.upper():7s} {self.code:24s} {self.message}"


@dataclass
class Report:
    path: str
    issues: List[Issue] = field(default_factory=list)
    populations: int = 0
    cells: int = 0

    @property
    def errors(self) -> List[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, code: str, message: str) -> None:
        self.issues.append(Issue("error", code, message))

    def warn(self, code: str, message: str) -> None:
        self.issues.append(Issue("warning", code, message))

    def describe(self) -> str:
        head = (f"{self.path}: {'CONFORMS' if self.ok else 'DOES NOT CONFORM'} to "
                f"Bio-USD {VERSION} -- {self.populations} population(s), "
                f"{self.cells:,} cells, {len(self.errors)} error(s), "
                f"{len(self.issues) - len(self.errors)} warning(s)")
        return "\n".join([head] + [f"  {i}" for i in self.issues])


_EXPECTED_TYPE = {"int[]": "int[]", "token[]": "token[]",
                  "float[]": "float[]", "bool[]": "bool[]"}


def validate_stage(info: StageInfo) -> Report:
    report = Report(path=info.path)

    if abs(info.meters_per_unit - METERS_PER_UNIT) > 1e-12:
        report.error("stage.metersPerUnit",
                     f"metersPerUnit is {info.meters_per_unit:g}; Bio-USD requires "
                     f"{METERS_PER_UNIT:g} (one unit = one micrometre)")
    if info.up_axis not in ("Y", "Z"):
        report.error("stage.upAxis", f"upAxis is {info.up_axis!r}; must be Y or Z")
    version = info.layer_data.get("bio:version")
    if version is None:
        report.error("layer.version", "customLayerData has no bio:version")
    elif str(version).split(".")[0] != VERSION.split(".")[0]:
        report.warn("layer.version",
                    f"written to Bio-USD {version}; this validator is {VERSION}")
    if "bio:colourBy" in info.layer_data and not (
            "bio:colourMin" in info.layer_data and "bio:colourMax" in info.layer_data):
        report.warn("layer.colourRange",
                    "bio:colourBy is set without bio:colourMin/Max; a legend cannot be drawn")

    if not info.populations:
        report.error("carrier.none",
                     "no PointInstancer carries bio:cell:id; nothing here is a cell population")
        return report

    for table in info.populations:
        report.populations += 1
        report.cells += table.n
        where = table.instancer
        for name in REQUIRED_PRIMVARS:
            if name not in table.primvars:
                report.error("primvar.missing", f"{where}: required primvar {name} is absent")
        if len(table.proto_indices) != table.n:
            report.error("carrier.protoIndices",
                         f"{where}: {len(table.proto_indices)} protoIndices for {table.n} positions")
        if table.n and (table.proto_indices.max() >= len(table.prototypes)
                        or table.proto_indices.min() < 0):
            report.error("carrier.prototypes",
                         f"{where}: a protoIndex is outside the {len(table.prototypes)} prototype(s)")
        if table.n and (table.diameters <= 0).any():
            report.error("carrier.scales", f"{where}: a cell has a non-positive diameter")
        for i in table.invisible_ids:
            if i < 0 or i >= table.n:
                report.error("carrier.invisibleIds", f"{where}: invisible id {i} is not an instance")
                break

        for name, values in table.primvars.items():
            spec = PRIMVARS_BY_NAME[name]
            authored = table.type_names.get(name, "")
            if authored != _EXPECTED_TYPE[spec.usd_type]:
                report.error("primvar.type",
                             f"{where}: {name} is {authored or 'untyped'}, must be {spec.usd_type}")
            if table.interpolation.get(name) != spec.interpolation:
                report.error("primvar.interpolation",
                             f"{where}: {name} is {table.interpolation.get(name)!r}, "
                             f"must be {spec.interpolation!r}")
            if len(values) != table.n:
                report.error("primvar.length",
                             f"{where}: {name} has {len(values)} values for {table.n} cells")
                continue
            if spec.tokens is not None:
                bad = sorted({str(v) for v in values if str(v) not in spec.tokens})
                if bad:
                    report.error("primvar.token",
                                 f"{where}: {name} has values outside its allowed set: {bad[:5]}")
            if spec.minimum is not None:
                try:
                    low = [v for v in values if float(v) < spec.minimum]
                except (TypeError, ValueError):
                    low = ["(non-numeric)"]
                if low:
                    report.error("primvar.range",
                                 f"{where}: {name} has {len(low)} value(s) below {spec.minimum:g}")
            if name == "bio:cell:ontologyTerm":
                bad = sorted({str(v) for v in values if not is_curie(str(v))})
                if bad:
                    report.error("primvar.curie",
                                 f"{where}: ontology terms that are not CL:/NCIT: CURIEs: {bad[:5]}")
                empty = sum(1 for v in values if str(v) == "")
                if empty == table.n and table.n:
                    report.warn("primvar.curie",
                                f"{where}: no cell asserts an ontology term")
            if name == "bio:cell:id":
                ids = [int(v) for v in values]
                if len(set(ids)) != len(ids):
                    report.error("primvar.id", f"{where}: bio:cell:id is not unique")

        for primvar in CELL_PRIMVARS:
            if not primvar.required and primvar.name not in table.primvars:
                report.warn("primvar.optional", f"{where}: {primvar.name} not present")

    return report


def validate(path: str) -> Report:
    try:
        info = read_stage(path)
    except Exception as exc:                       # noqa: BLE001 - reported
        report = Report(path=path)
        report.error("stage.open", f"could not read the stage: {exc}")
        return report
    return validate_stage(info)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 2
    worst = 0
    for path in args:
        report = validate(path)
        print(report.describe())
        if any(i.code == "stage.open" for i in report.issues):
            worst = max(worst, 2)
        elif not report.ok:
            worst = max(worst, 1)
    return worst


if __name__ == "__main__":                        # pragma: no cover
    raise SystemExit(main())
