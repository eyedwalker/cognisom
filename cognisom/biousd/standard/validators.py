"""Bio-USD's checks as validators for OpenUSD's UsdValidation framework.

The framework (``pxr.UsdValidation``, usd-core 24.11+) is what a
framework client runs -- measured here as a ``ValidationContext`` over
every registered validator, the selection ``usdchecker
--useNewValidationFramework`` makes; ``usdchecker`` itself has not been
run with these, because no wheel here ships the binary and Apple's
loads no plugin. A validator declared there is found by keyword, by
schema type or by name, and its findings sit beside the builtin ones in
one report. These two validators and one suite put the checks of
:mod:`.validate` there:

- ``biousdValidators:StageMetadata`` (stage): :func:`~.validate.check_stage`,
  :func:`~.validate.check_schema`, :func:`~.validate.check_profile`;
- ``biousdValidators:CellPopulation`` (prim, gated on
  ``UsdGeomPointInstancer``): :func:`~.validate.check_population`;
- ``biousdValidators:BioUSDSuite``: both.

They are the SAME functions the command-line validator runs, so the two
cannot disagree; a code such as ``primvar.token`` is reported here as
the error name ``PrimvarToken`` (:func:`error_name`, :func:`code_for`).

NOT EVERY STAGE IS BIO-USD
--------------------------
Registered, the stage validator ran on every stage a framework client
validated, and an ordinary scene with no ``bio:*`` anywhere came back
with three Bio-USD errors (``StageMetersPerUnit``, ``LayerVersion``,
``CarrierNone``) and failed (measured on 26.11). :func:`stage_task` now
runs only on a stage that says it is Bio-USD -- ``bio:version`` in the
layer data, a cell population, or ``BioCellAPI`` applied anywhere
(:func:`~.validate.is_biousd_stage`) -- and reports ``carrier.none``
only when ``bio:version`` is present and no population is: the file says
it is Bio-USD and holds no cells. The standalone validator, which a
person points at a Bio-USD file on purpose, keeps reporting
``carrier.none`` on anything. :func:`population_task` already returned
nothing for an instancer that is not a population.

THE VERSION GATE, measured on usd-core 25.11 and 26.11:

- 25.11 binds ``UsdValidation`` with ``Get*``/``Has*`` only -- no
  ``Register*``, and ``Validator()`` cannot be instantiated from Python.
  OpenUSD 26.05 added Python registration (CHANGELOG: "Added Python
  wrappers for registering new (plugins or otherwise) validators",
  PR #4015). :func:`register_validators` therefore returns ``False`` on
  25.x with that reason, logged once and printed by ``biousd-check``;
  ``biousd-check`` then runs the standalone checks, which are the same
  checks.
- A ``"Type": "python"`` Plug plugin (``plugin/biousdValidators/``)
  carries the validator metadata in its ``plugInfo.json``; the
  validation registry, when it is built, advertises the names and
  imports the module lazily the first time one is requested. On 26.11
  the module's import registers the task functions by name and a
  context over all validators reports Bio-USD errors next to the
  builtin ones (5 ms for 10,000 cells).
- ADVERTISED ONLY WHERE IT CAN LOAD. The first draft let the schema
  plugin's ``plugInfo.json`` ``Includes`` this one, so any process that
  registered the schema -- ``import biousd``, or ``PXR_PLUGINPATH_NAME``
  in a tool nobody controls -- advertised the validators too, and the
  first framework client to run all validators raised
  ``Tf.ErrorException`` (four ``PlugPlugin::_Load ... failed``,
  ``ModuleNotFoundError: biousdValidators``) on BOTH wheels unless
  ``plugin/`` happened to be on ``PYTHONPATH``. Now
  :func:`advertise_validators`, called by ``import biousd`` and by
  :func:`register_validators`, hands ``plugin/biousdValidators`` to Plug
  and puts ``plugin/`` on ``sys.path`` in the same step, and only on a
  ``pxr`` that can register (26.05+); on 25.11 nothing is advertised and
  nothing can fail to load. ``PXR_PLUGINPATH_NAME`` set to
  :func:`~.api_schema.plugin_path` therefore gives the schema and the
  profiles alone; a framework client that wants these validators from
  the environment adds ``plugin/biousdValidators`` to that variable and
  ``plugin/`` to ``PYTHONPATH`` -- where, on 25.11, the module imports,
  registers nothing, and the registry skips the validator silently
  (measured: no finding, no error).
- ORDER. Like the schema registry, the validation registry reads
  ``plugInfo.json`` once, when it is first built (the first
  ``UsdValidation.ValidationRegistry()``), and a plugin handed to Plug
  after that is not advertised; :func:`register_validators` then
  registers the validators explicitly, with their full metadata, which
  needs no plugin. :func:`registration_available` looks at the registry
  class, not an instance, so asking it does not build the registry.
- A ``"Type": "library"`` plugInfo whose library does not exist breaks
  ``ValidationContext`` on 25.11, so the plugin is ``python``, never
  ``library`` with an empty path.
- The README calls the time-range type ``UsdValidation.TimeRange``; the
  bound class is ``ValidationTimeRange``. The task functions here accept
  whatever the framework passes and ignore it: Bio-USD has no
  time-dependent check.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

from .api_schema import plugin_path
from .read import read_population, read_stage_info
from .validate import (
    Issue, check_population, check_profile, check_schema, check_stage, is_biousd_stage)
from .vocabulary import (
    API_SCHEMA, CARRIER_PRIM_TYPE, METERS_PER_UNIT, REQUIRED_PRIMVARS, VALIDATOR_CODES,
    VERSION)

__all__ = [
    "PLUGIN_NAME", "KEYWORD", "STAGE_VALIDATOR", "POPULATION_VALIDATOR", "SUITE",
    "MINIMUM_VERSION", "VALIDATORS", "render_plug_info", "validators_plugin_path",
    "error_name", "code_for", "stage_task", "population_task",
    "registration_available", "advertise_validators", "register_validators",
    "register_from_plugin",
]

log = logging.getLogger(__name__)

#: The Plug plugin's name, which is also the module Plug imports and the
#: prefix of every validator name.
PLUGIN_NAME = "biousdValidators"
KEYWORD = "BioUSDValidators"
STAGE_VALIDATOR = f"{PLUGIN_NAME}:StageMetadata"
POPULATION_VALIDATOR = f"{PLUGIN_NAME}:CellPopulation"
SUITE = f"{PLUGIN_NAME}:BioUSDSuite"
POINT_INSTANCER_TYPE = "UsdGeomPointInstancer"

#: The first OpenUSD with Python validator registration.
MINIMUM_VERSION = (0, 26, 5)

#: The validators as ``plugInfo.json`` declares them (name without the
#: plugin prefix -> metadata). ``keywords`` at the top level applies to
#: all three, as the builtin plugins do it.
VALIDATORS: Dict[str, Dict[str, Any]] = {
    "StageMetadata": {
        "doc": (f"On a Bio-USD stage (bio:version in customLayerData, a cell population, "
                f"or {API_SCHEMA} applied; any other stage gets no finding): metersPerUnit "
                f"must be {METERS_PER_UNIT:g} (one stage unit is one micrometre), upAxis Y "
                "or Z, and the root layer's customLayerData must carry bio:version; "
                f"{API_SCHEMA} may be applied to a {CARRIER_PRIM_TYPE} only; UsdProfiles "
                "claims, when present, must agree with the stage; a stage that names "
                f"bio:version holds at least one cell population. Bio-USD {VERSION}."),
    },
    "CellPopulation": {
        "doc": (f"A {CARRIER_PRIM_TYPE} that authors primvars:bio:cell:id or applies "
                f"{API_SCHEMA} has " + " and ".join(REQUIRED_PRIMVARS) + " with the stated "
                "type, varying interpolation and one value per instance; tokens are in "
                "their allowed sets; ontology terms are CL:/NCIT: CURIEs or empty; "
                f"bio:cell:id is unique. Bio-USD {VERSION}."),
        "schemaTypes": [POINT_INSTANCER_TYPE],
        "keywords": ["BioUSDCells"],
    },
    "BioUSDSuite": {
        "doc": "Every Bio-USD validator.",
        "isSuite": True,
    },
}


def validators_plugin_path() -> str:
    """``plugin/biousdValidators``: the module Plug imports, with its plugInfo.json."""
    return os.path.join(plugin_path(), PLUGIN_NAME)


def render_plug_info() -> str:
    """``plugin/biousdValidators/plugInfo.json``, from :data:`VALIDATORS`."""
    info = {"Plugins": [{
        "Type": "python",
        "Name": PLUGIN_NAME,
        "Info": {"Validators": {"keywords": [KEYWORD], **VALIDATORS}},
    }]}
    return json.dumps(info, indent=4, sort_keys=True) + "\n"


def error_name(code: str) -> str:
    """``primvar.token`` -> ``PrimvarToken``: a validator code as a framework error name."""
    return "".join(part[:1].upper() + part[1:] for part in code.split("."))


_CODE_BY_NAME = {error_name(code): code for code in VALIDATOR_CODES}


def code_for(name: str) -> str:
    """The validator code behind a framework error name or identifier.

    Accepts ``PrimvarToken``, ``biousdValidators:CellPopulation.PrimvarToken``
    or the whole ``GetErrorAsString()`` line, ``Error: (<identifier>) <message>``;
    raises ``KeyError`` for a name that is not one of ours.
    """
    text = name
    if "(" in text:
        text = text.split("(", 1)[1].split(")", 1)[0]
    return _CODE_BY_NAME[text.rsplit(".", 1)[-1].strip()]


# ── task functions ──────────────────────────────────────────────────


def _errors(issues: List[Issue], stage) -> list:
    from pxr import Sdf, UsdValidation

    kinds = UsdValidation.ValidationErrorType
    out = []
    for issue in issues:
        path = Sdf.Path(issue.site) if issue.site else Sdf.Path.absoluteRootPath
        out.append(UsdValidation.ValidationError(
            error_name(issue.code), kinds.Error if issue.level == "error" else kinds.Warn,
            [UsdValidation.ValidationErrorSite(stage, path)], issue.message))
    return out


def stage_task(stage, time_range=None) -> list:
    """``biousdValidators:StageMetadata``: the stage-level checks, on a Bio-USD stage.

    A stage that does not say it is Bio-USD (:func:`~.validate.is_biousd_stage`)
    gets no finding: a framework client runs every registered validator
    on every file, and this one must not fail an ordinary scene.
    ``carrier.none`` is reported only when the stage names ``bio:version``
    and holds no population.
    """
    info = read_stage_info(stage)
    if not is_biousd_stage(info):
        return []
    issues = check_stage(info) + check_schema(info)
    if info.populations:
        issues += check_profile(info)
    elif "bio:version" in info.layer_data:
        issues.append(Issue("error", "carrier.none",
                            f"no PointInstancer authors bio:cell:id or applies {API_SCHEMA}; "
                            "nothing here is a cell population"))
    return _errors(issues, stage)


def population_task(prim, time_range=None) -> list:
    """``biousdValidators:CellPopulation``: one carrier's checks."""
    table = read_population(prim)
    if table is None:
        return []
    return _errors(check_population(table), prim.GetStage())


# ── registration ────────────────────────────────────────────────────

_said: set = set()
_state = {"advertised": False}


def _once(message: str, level: int = logging.INFO) -> None:
    if message not in _said:
        _said.add(message)
        log.log(level, message)


def registration_available() -> Tuple[bool, str]:
    """Whether this ``pxr`` can register a validator from Python, and why not.

    Looks at the ``ValidationRegistry`` class, never an instance:
    instantiating the singleton builds the registry, after which a
    plugin handed to Plug is not advertised (measured on both wheels).
    """
    try:
        from pxr import Usd, UsdValidation
    except ImportError:
        return False, "pxr.UsdValidation is not importable (usd-core 24.11 or later binds it)"
    if not hasattr(UsdValidation.ValidationRegistry, "RegisterStageValidator"):
        return False, (f"usd-core {'.'.join(map(str, Usd.GetVersion()))} has no Python "
                       "validator registration; OpenUSD 26.05 added it (PR #4015), so the "
                       "Bio-USD validators run standalone here (biousd.validate / "
                       "biousd-check), not inside the framework")
    return True, ""


def _ensure_importable() -> None:
    """Plug imports the plugin as ``import biousdValidators``; make that resolve."""
    parent = plugin_path()
    if parent not in sys.path:
        sys.path.insert(0, parent)


def advertise_validators() -> bool:
    """Hand ``plugin/biousdValidators`` to Plug, where it can load, and only there.

    A validation registry built afterwards advertises the three
    validators and imports the module on first request, so ``plugin/``
    goes on ``sys.path`` in the same step. On a ``pxr`` without Python
    validator registration (before 26.05) nothing is registered and
    nothing advertised, because an advertised validator that cannot load
    is what made every framework client raise. ``True`` when Plug has
    the plugin afterwards. Idempotent, about a millisecond, and it never
    builds a registry.
    """
    if _state["advertised"]:
        return True
    if not registration_available()[0]:
        return False
    from pxr import Plug, Tf

    _ensure_importable()
    path = validators_plugin_path()
    try:
        Plug.Registry().RegisterPlugins(path)
    except Tf.ErrorException as exc:           # a malformed plugInfo.json
        log.error("the %s plugin at %s did not register: %s", PLUGIN_NAME, path, exc)
        return False
    if Plug.Registry().GetPluginWithName(PLUGIN_NAME) is None:
        _once(f"no {PLUGIN_NAME} plugin at {path} (plugInfo.json missing; run "
              "python -m biousd.build_plugin)", logging.ERROR)
        return False
    _state["advertised"] = True
    return True


def _register_plugin_validators(reg) -> None:
    reg.RegisterPluginStageValidator(STAGE_VALIDATOR, stage_task)
    reg.RegisterPluginPrimValidator(POPULATION_VALIDATOR, population_task)
    reg.RegisterPluginValidatorSuite(
        SUITE, [reg.GetOrLoadValidatorByName(STAGE_VALIDATOR),
                reg.GetOrLoadValidatorByName(POPULATION_VALIDATOR)])


def _register_explicitly(reg) -> None:
    """Without Plug: full metadata, for a registry built before our plugInfo was seen."""
    from pxr import Tf, UsdValidation

    md = VALIDATORS["StageMetadata"]
    reg.RegisterStageValidator(
        UsdValidation.ValidatorMetadata(name=STAGE_VALIDATOR, doc=md["doc"],
                                        keywords=[KEYWORD]), stage_task)
    md = VALIDATORS["CellPopulation"]
    try:
        metadata = UsdValidation.ValidatorMetadata(
            name=POPULATION_VALIDATOR, doc=md["doc"], keywords=[KEYWORD] + md["keywords"],
            schemaTypes=[Tf.Type.FindByName(POINT_INSTANCER_TYPE)])
    except TypeError:                   # a binding without schemaTypes: the task gates itself
        metadata = UsdValidation.ValidatorMetadata(
            name=POPULATION_VALIDATOR, doc=md["doc"], keywords=[KEYWORD] + md["keywords"])
    reg.RegisterPrimValidator(metadata, population_task)
    reg.RegisterValidatorSuite(
        UsdValidation.ValidatorMetadata(name=SUITE, doc=VALIDATORS["BioUSDSuite"]["doc"],
                                        keywords=[KEYWORD], isSuite=True),
        [reg.GetOrLoadValidatorByName(STAGE_VALIDATOR),
         reg.GetOrLoadValidatorByName(POPULATION_VALIDATOR)])


def _registered(reg) -> bool:
    return bool(reg.HasValidator(STAGE_VALIDATOR) and reg.HasValidator(POPULATION_VALIDATOR))


def register_validators() -> bool:
    """Register the Bio-USD validators with this process's UsdValidation registry.

    ``True`` when ``biousdValidators:StageMetadata`` and
    ``:CellPopulation`` are registered afterwards (already, through the
    plugin, or explicitly); ``False``, with the reason logged once, on a
    ``pxr`` without Python registration (before OpenUSD 26.05).

    Two routes, because the validation registry, like the schema
    registry, reads ``plugInfo.json`` once when it is built: when our
    plugin was handed to Plug before that (``import biousd`` first, or
    ``PXR_PLUGINPATH_NAME`` naming ``plugin/biousdValidators``), the
    names are advertised and loading one imports the plugin module,
    which registers by name; when it was not, the validators are
    registered explicitly with their full metadata, which needs no
    plugin at all.
    """
    ok, why = registration_available()
    if not ok:
        _once(why)
        return False
    from pxr import UsdValidation

    advertise_validators()          # in time, when the registry is not yet built
    _ensure_importable()            # the env-var route advertises without the path
    reg = UsdValidation.ValidationRegistry()
    if _registered(reg):
        return True
    advertised = {m.name for m in reg.GetAllValidatorMetadata()}
    if STAGE_VALIDATOR in advertised:
        for name in (STAGE_VALIDATOR, POPULATION_VALIDATOR):
            reg.GetOrLoadValidatorByName(name)
        reg.GetOrLoadValidatorSuiteByName(SUITE)
        if _registered(reg):
            return True
        _once(f"{PLUGIN_NAME} is advertised but did not register through Plug; "
              "registering explicitly", logging.WARNING)
    _register_explicitly(reg)
    return _registered(reg)


def register_from_plugin() -> bool:
    """What ``plugin/biousdValidators/__init__.py`` calls when Plug imports it.

    Registers by name (the metadata is already in the registry from the
    plugInfo.json); on a ``pxr`` without ``RegisterPlugin*`` it logs once
    and registers nothing, which the registry treats as a validator that
    did not load.
    """
    ok, why = registration_available()
    if not ok:
        _once(why)
        return False
    from pxr import UsdValidation

    reg = UsdValidation.ValidationRegistry()
    if _registered(reg):
        return True
    _register_plugin_validators(reg)
    return _registered(reg)
