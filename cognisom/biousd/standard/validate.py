"""Does this file conform to Bio-USD? Checks exactly what SPEC.md says.

    python -m biousd.validate scene.usdc            (after extraction)
    python -m cognisom.biousd.standard.validate scene.usdc

Exit status 0 when the stage conforms, 1 when it does not, 2 when it
could not be read. Every finding is a line with a level, a code and a
message, so a CI job can grep it and a person can read it; the codes
and their levels are :data:`~.vocabulary.VALIDATOR_CODES`, which SPEC.md
lists.

A validator is the difference between a convention and a document: a
third party cannot know whether their file is right without one, and
this project's earlier validator checked Python objects, not files, so
it could not have told them.

ONE IMPLEMENTATION
------------------
The checks are four functions over what :mod:`.read` returns --
:func:`check_stage`, :func:`check_schema`, :func:`check_population`,
:func:`check_profile` -- and :func:`validate_stage` only assembles their
findings into a :class:`Report`. OpenUSD's validation framework runs the
same four through :mod:`.validators`, so a finding is one thing whichever
path reports it; the first draft of the framework plugin copied the
vocabulary into its own module, which is how a second validator starts
disagreeing with the first.

THE PLUGIN MUST NOT CHANGE THE VERDICT
--------------------------------------
Conformance is a property of the file, so the same file must get the
same verdict in a process with the ``BioCellAPI`` plugin and in one
without it. Three cases broke that, each now a test that runs both ways
in subprocesses:

- an omitted optional primvar: the 0.1 validator, with the plugin, saw
  the schema-defined primvar and reported ``primvar.length`` (exit 1)
  on a conforming file; the reader now gates on authored data, and the
  case is one warning either way;
- an unauthored interpolation: with the plugin it resolved to
  ``varying`` through the schema fallback and the first 0.2 draft
  reported a warning, without it to USD's default ``constant`` and an
  error. The fallback is not the file: a reader without the plugin sees
  ``constant``, so the file must author ``varying`` and the case is
  ``primvar.interpolation`` either way;
- a wrong value type: with the plugin ``GetTypeName()`` resolved to the
  declared type, so ``int[] primvars:bio:cell:age`` passed; the reader
  now records the authored type, and the case is ``primvar.type``
  either way.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import List, Sequence

from .profiles import ClaimsInfo, claims_issues
from .read import CellTable, StageInfo, read_stage
from .vocabulary import (
    API_SCHEMA, CARRIER_PRIM_TYPE, CELL_PRIMVARS, METERS_PER_UNIT, PRIMVARS_BY_NAME,
    REQUIRED_PRIMVARS, VALIDATOR_CODES, VERSION, is_curie)


@dataclass
class Issue:
    level: str          # "error" | "warning"
    code: str
    message: str
    #: the prim the finding is about, when it is about one
    site: str = ""

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
    def warnings(self) -> List[Issue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def codes(self) -> List[str]:
        return [i.code for i in self.issues]

    def error(self, code: str, message: str, site: str = "") -> None:
        self.issues.append(Issue("error", code, message, site))

    def warn(self, code: str, message: str, site: str = "") -> None:
        self.issues.append(Issue("warning", code, message, site))

    def describe(self) -> str:
        head = (f"{self.path}: {'CONFORMS' if self.ok else 'DOES NOT CONFORM'} to "
                f"Bio-USD {VERSION} -- {self.populations} population(s), "
                f"{self.cells:,} cells, {len(self.errors)} error(s), "
                f"{len(self.issues) - len(self.errors)} warning(s)")
        return "\n".join([head] + [f"  {i}" for i in self.issues])


_EXPECTED_TYPE = {"int[]": "int[]", "token[]": "token[]",
                  "float[]": "float[]", "bool[]": "bool[]"}


def _error(code: str, message: str, site: str = "") -> Issue:
    assert code in VALIDATOR_CODES, code
    return Issue("error", code, message, site)


def _warning(code: str, message: str, site: str = "") -> Issue:
    assert code in VALIDATOR_CODES, code
    return Issue("warning", code, message, site)


def check_stage(info: StageInfo) -> List[Issue]:
    """Stage metadata: units, up axis, the version and the colour ramp."""
    out: List[Issue] = []
    if abs(info.meters_per_unit - METERS_PER_UNIT) > 1e-12:
        out.append(_error("stage.metersPerUnit",
                          f"metersPerUnit is {info.meters_per_unit:g}; Bio-USD requires "
                          f"{METERS_PER_UNIT:g} (one unit = one micrometre)"))
    if info.up_axis not in ("Y", "Z"):
        out.append(_error("stage.upAxis", f"upAxis is {info.up_axis!r}; must be Y or Z"))
    version = info.layer_data.get("bio:version")
    if version is None:
        out.append(_error("layer.version", "customLayerData has no bio:version"))
    elif str(version).split(".")[0] != VERSION.split(".")[0]:
        out.append(_warning("layer.version",
                            f"written to Bio-USD {version}; this validator is {VERSION}"))
    if "bio:colourBy" in info.layer_data and not (
            "bio:colourMin" in info.layer_data and "bio:colourMax" in info.layer_data):
        out.append(_warning("layer.colourRange",
                            "bio:colourBy is set without bio:colourMin/Max; a legend "
                            "cannot be drawn"))
    return out


def check_schema(info: StageInfo) -> List[Issue]:
    """Where ``BioCellAPI`` is applied: only to a PointInstancer, only version 0."""
    out: List[Issue] = []
    for path, type_name in info.misapplied:
        out.append(_error("schema.misapplied",
                          f"{path}: {API_SCHEMA} is applied to a {type_name}; it may only "
                          f"be applied to a {CARRIER_PRIM_TYPE}", site=path))
    for path, name in info.other_versions:
        out.append(_warning("schema.version",
                            f"{path}: {name} is applied; this validator knows "
                            f"{API_SCHEMA} (version 0) only", site=path))
    return out


def check_population(table: CellTable) -> List[Issue]:
    """Everything about one carrier and the primvars it authors."""
    out: List[Issue] = []
    where = table.instancer
    for name in REQUIRED_PRIMVARS:
        if name not in table.primvars:
            out.append(_error("primvar.missing",
                              f"{where}: required primvar {name} is absent", where))
    if len(table.proto_indices) != table.n:
        out.append(_error("carrier.protoIndices",
                          f"{where}: {len(table.proto_indices)} protoIndices for "
                          f"{table.n} positions", where))
    if table.n and len(table.proto_indices) and (
            table.proto_indices.max() >= len(table.prototypes) or table.proto_indices.min() < 0):
        out.append(_error("carrier.prototypes",
                          f"{where}: a protoIndex is outside the {len(table.prototypes)} "
                          "prototype(s)", where))
    if table.n and (table.diameters <= 0).any():
        out.append(_error("carrier.scales", f"{where}: a cell has a non-positive diameter",
                          where))
    for i in table.invisible_ids:
        if i < 0 or i >= table.n:
            out.append(_error("carrier.invisibleIds",
                              f"{where}: invisible id {i} is not an instance", where))
            break

    for name, values in table.primvars.items():
        spec = PRIMVARS_BY_NAME[name]
        authored = table.type_names.get(name, "")
        if authored != _EXPECTED_TYPE[spec.usd_type]:
            out.append(_error("primvar.type",
                              f"{where}: {name} is {authored or 'untyped'}, must be "
                              f"{spec.usd_type}", where))
        if not table.interpolation_authored.get(name, True):
            # The same message whichever way it resolves here: with the
            # plugin the schema fallback says 'varying', without it USD
            # says 'constant', and the verdict must not depend on which.
            out.append(_error("primvar.interpolation",
                              f"{where}: {name} authors no interpolation; the file must "
                              f"author {spec.interpolation!r} (a reader without the "
                              f"{API_SCHEMA} plugin sees USD's default, 'constant')", where))
        elif table.interpolation.get(name) != spec.interpolation:
            out.append(_error("primvar.interpolation",
                              f"{where}: {name} is {table.interpolation.get(name)!r}, "
                              f"must be {spec.interpolation!r}", where))
        if len(values) != table.n:
            out.append(_error("primvar.length",
                              f"{where}: {name} has {len(values)} values for {table.n} cells",
                              where))
            continue
        if spec.tokens is not None:
            bad = sorted({str(v) for v in values if str(v) not in spec.tokens})
            if bad:
                out.append(_error("primvar.token",
                                  f"{where}: {name} has values outside its allowed set: "
                                  f"{bad[:5]}", where))
        if spec.minimum is not None:
            try:
                low = [v for v in values if float(v) < spec.minimum]
            except (TypeError, ValueError):
                low = ["(non-numeric)"]
            if low:
                out.append(_error("primvar.range",
                                  f"{where}: {name} has {len(low)} value(s) below "
                                  f"{spec.minimum:g}", where))
        if name == "bio:cell:ontologyTerm":
            bad = sorted({str(v) for v in values if not is_curie(str(v))})
            if bad:
                out.append(_error("primvar.curie",
                                  f"{where}: ontology terms that are not CL:/NCIT: CURIEs: "
                                  f"{bad[:5]}", where))
            empty = sum(1 for v in values if str(v) == "")
            if empty == table.n and table.n:
                out.append(_warning("primvar.curie",
                                    f"{where}: no cell asserts an ontology term", where))
        if name == "bio:cell:id":
            ids = [int(v) for v in values]
            if len(set(ids)) != len(ids):
                out.append(_error("primvar.id", f"{where}: bio:cell:id is not unique", where))

    for primvar in CELL_PRIMVARS:
        if not primvar.required and primvar.name not in table.primvars:
            out.append(_warning("primvar.optional", f"{where}: {primvar.name} not present",
                                where))
    if not table.api_applied:
        out.append(_warning("schema.notApplied",
                            f"{where}: {API_SCHEMA} is not applied (a 0.1 file; still "
                            "conforming)", where))
    return out


def check_profile(info: StageInfo) -> List[Issue]:
    """UsdProfiles claims, for consistency with the stage. Warnings only."""
    claims = info.claims or ClaimsInfo()
    asserted = any(any(str(v) for v in t.primvars.get("bio:cell:ontologyTerm", []))
                   for t in info.populations)
    site = claims.prim or ""
    return [_warning(code, message, site)
            for code, message in claims_issues(
                claims, version=info.layer_data.get("bio:version"),
                populations=len(info.populations), ontology_asserted=asserted)]


def is_biousd_stage(info: StageInfo) -> bool:
    """Whether the stage says it is Bio-USD at all.

    ``bio:version`` in the layer data, a cell population, or
    ``BioCellAPI`` applied anywhere (to the wrong prim type or under a
    later schema version included). The framework validators run on
    every stage a client validates and leave any other stage alone;
    :func:`validate` is pointed at a Bio-USD file on purpose and does
    not ask.
    """
    return bool("bio:version" in info.layer_data or info.populations
                or info.misapplied or info.other_versions)


def validate_stage(info: StageInfo) -> Report:
    report = Report(path=info.path)
    report.issues += check_stage(info)
    report.issues += check_schema(info)
    if not info.populations:
        report.error("carrier.none",
                     f"no PointInstancer authors bio:cell:id or applies {API_SCHEMA}; "
                     "nothing here is a cell population")
        return report
    for table in info.populations:
        report.populations += 1
        report.cells += table.n
        report.issues += check_population(table)
    report.issues += check_profile(info)
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
