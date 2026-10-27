"""Bio-USD's checks inside OpenUSD's validation framework, gated on the wheel.

Three paths report the same findings on the same file: the standalone
validator, the chained `biousd-check` command, and -- on usd-core 26.05 or
later, which is where Python validator registration arrived (PR #4015) --
the framework itself with `biousdValidators:*` registered. On 25.11 the
framework path is skipped by name, not silently: `registration_available()`
says why, the chained command prints it, and still reports every finding
from the standalone pass.

The plugin is also loadable the way a framework client loads any plugin:
`PXR_PLUGINPATH_NAME` naming its directory and `import biousdValidators`
resolvable, with nothing of ours imported first.

What the first draft got wrong, each now pinned here:

- the schema plugin's plugInfo pulled the validators plugin in, so the
  README's `PXR_PLUGINPATH_NAME` route (schema only, no PYTHONPATH) and
  `import biousd` alone both left a ValidationContext over all validators
  raising `Load of biousdValidators failed` on both wheels;
- registered, the stage validator reported three errors on an ordinary
  non-Bio-USD scene, failing it;
- `biousd-check` raised out of the framework pass on 25.11 for any stage
  with a `UsdShade.Shader` (no shader definitions on that wheel), and a
  second call in the same process got spurious errors instead.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("pxr")

from cognisom.biousd import standard as biousd                          # noqa: E402
from cognisom.biousd.standard import check as check_module              # noqa: E402
from cognisom.biousd.standard import validators                         # noqa: E402
from cognisom.biousd.standard.validators import (                       # noqa: E402
    KEYWORD, PLUGIN_NAME, POPULATION_VALIDATOR, STAGE_VALIDATOR, SUITE, code_for,
    error_name, population_task, register_validators, registration_available, stage_task)
from cognisom.biousd.standard.vocabulary import VALIDATOR_CODES         # noqa: E402

PKG = pathlib.Path(biousd.__file__).resolve().parent
REPO = PKG.parents[2]
EXAMPLE = PKG / "examples" / "minimal_200_cells.usda"

#: What the deliberately broken file must yield, by code.
BROKEN_ERRORS = {"stage.metersPerUnit", "primvar.id", "primvar.token"}
BROKEN_WARNINGS = {"profile.absent"}        # the default prim is cleared, and the claims with it


def _usd_version():
    from pxr import Usd
    return tuple(Usd.GetVersion())


def _can_register() -> bool:
    return _usd_version() >= validators.MINIMUM_VERSION


def _cells(n=20):
    rng = np.random.default_rng(2)
    return [dict(position=tuple(map(float, p)), cell_id=i, cell_type="cancer",
                 ontology_term="NCIT:C2919", scale_um=10.0)
            for i, p in enumerate(rng.uniform(-30, 30, (n, 3)))]


@pytest.fixture
def broken(tmp_path):
    """The spike's deliberately broken copy: findings for both validators.

    Framework: no defaultPrim, a Gprim nested under the prototype Gprim.
    Bio-USD: metersPerUnit 0.01, a type outside the set, a duplicated id.
    """
    from pxr import Usd, UsdGeom, Vt

    path = str(tmp_path / "broken.usda")
    biousd.write_cells(path, _cells())
    stage = Usd.Stage.Open(path)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    stage.GetRootLayer().ClearDefaultPrim()
    UsdGeom.Sphere.Define(stage, "/Prototypes/cell/Inner")
    api = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath("/World/Cells"))
    types = list(api.GetPrimvar("bio:cell:type").Get())
    types[0] = "alien"
    api.GetPrimvar("bio:cell:type").Set(Vt.TokenArray(types))
    ids = list(api.GetPrimvar("bio:cell:id").Get())
    ids[1] = ids[0]
    api.GetPrimvar("bio:cell:id").Set(Vt.IntArray(ids))
    stage.GetRootLayer().Save()
    return path


# ── the plugin's metadata and names ─────────────────────────────────


def test_the_validators_plugin_info_is_rendered_from_the_table():
    path = PKG / "plugin" / PLUGIN_NAME / "plugInfo.json"
    assert path.read_text() == validators.render_plug_info()
    (plugin,) = json.loads(path.read_text())["Plugins"]
    assert plugin["Type"] == "python" and plugin["Name"] == PLUGIN_NAME
    declared = plugin["Info"]["Validators"]
    assert declared["keywords"] == [KEYWORD]
    assert set(declared) - {"keywords"} == {"StageMetadata", "CellPopulation", "BioUSDSuite"}
    assert declared["CellPopulation"]["schemaTypes"] == ["UsdGeomPointInstancer"]
    assert declared["BioUSDSuite"]["isSuite"] is True
    assert (PKG / "plugin" / PLUGIN_NAME / "__init__.py").exists()
    # the schema plugInfo does NOT include it: a python plugin advertised
    # wherever the schema is registered fails to load wherever plugin/ is
    # not on sys.path, and that was every framework client
    umbrella = json.loads((PKG / "plugin" / "plugInfo.json").read_text())
    assert "Includes" not in umbrella


def test_error_names_round_trip():
    for code in VALIDATOR_CODES:
        name = error_name(code)
        assert name[0].isupper() and "." not in name
        assert code_for(name) == code
    assert error_name("primvar.token") == "PrimvarToken"
    assert code_for(f"{POPULATION_VALIDATOR}.PrimvarToken") == "primvar.token"
    assert code_for(f"Error: ({POPULATION_VALIDATOR}.PrimvarId) /World/Cells: bio:cell:id "
                    "is not unique (really)") == "primvar.id"
    with pytest.raises(KeyError):
        code_for("usdValidation:StageMetadataChecker.MissingDefaultPrim")


# ── the gate ────────────────────────────────────────────────────────


def test_registration_is_gated_on_the_wheel(caplog, monkeypatch):
    ok, why = registration_available()
    if _can_register():
        assert ok and why == ""
        assert register_validators() is True
        from pxr import UsdValidation
        reg = UsdValidation.ValidationRegistry()
        assert reg.HasValidator(STAGE_VALIDATOR) and reg.HasValidator(POPULATION_VALIDATOR)
        assert reg.HasValidatorSuite(SUITE)
        md = reg.GetValidatorMetadata(POPULATION_VALIDATOR)
        assert KEYWORD in md.GetKeywords()
        assert "UsdGeomPointInstancer" in [str(t) for t in md.GetSchemaTypes()] or (
            [t.typeName for t in md.GetSchemaTypes()] == ["UsdGeomPointInstancer"])
    else:
        assert not ok and "26.05" in why
        # and says so once, not twice: the module remembers what it said,
        # so forget it first, then two calls must produce one record
        monkeypatch.setattr(validators, "_said", set())
        with caplog.at_level(logging.INFO, logger=validators.log.name):
            assert register_validators() is False
            assert register_validators() is False
        said = [r for r in caplog.records if "26.05" in r.getMessage()]
        assert len(said) == 1, [r.getMessage() for r in caplog.records]


# ── the task functions run on any wheel ─────────────────────────────


def test_the_task_functions_are_the_standalone_checks(broken):
    from pxr import Usd, UsdValidation

    E = UsdValidation.ValidationErrorType
    assert stage_task(Usd.Stage.Open(str(EXAMPLE))) == []
    stage = Usd.Stage.Open(broken)
    assert population_task(stage.GetPrimAtPath("/World")) == []        # not an instancer
    stage_errors = stage_task(stage)
    cells_errors = population_task(stage.GetPrimAtPath("/World/Cells"))
    # An error made by a task function called directly has no validator
    # behind it, so GetIdentifier() raises; GetName() is the error name.
    assert {code_for(e.GetName()) for e in stage_errors} == {"stage.metersPerUnit",
                                                             "profile.absent"}
    assert {code_for(e.GetName()) for e in cells_errors} == {"primvar.id", "primvar.token"}
    assert {e.GetType() for e in cells_errors} == {E.Error}
    assert all(str(s.GetPrim().GetPath()) == "/World/Cells"
               for e in cells_errors for s in e.GetSites())
    by_code = {code_for(e.GetName()): e for e in stage_errors}
    assert by_code["stage.metersPerUnit"].GetType() == E.Error
    assert by_code["profile.absent"].GetType() == E.Warn


# ── three paths, one set of findings ────────────────────────────────


def test_the_standalone_and_chained_paths_agree(broken):
    standalone = biousd.validate(broken)
    assert not standalone.ok
    assert {i.code for i in standalone.errors} == BROKEN_ERRORS
    assert {i.code for i in standalone.warnings} == BROKEN_WARNINGS

    result = check_module.check_file(broken)
    assert result.status == 1 and result.error == ""
    assert result.report.codes == standalone.codes
    builtin = {f.identifier for f in result.framework
               if not f.identifier.startswith(PLUGIN_NAME + ":")}
    assert "usdValidation:StageMetadataChecker.MissingDefaultPrim" in builtin
    assert "usdGeomValidators:EncapsulationChecker.InvalidNestedGprims" in builtin
    assert result.biousd_in_framework == _can_register()
    if result.biousd_in_framework:
        assert sorted(result.framework_codes) == sorted(standalone.codes)
    else:
        assert result.framework_codes == []
    text = check_module.describe(result)
    for code in standalone.codes:
        assert f"biousd:{code}" in text
    assert text.rstrip().endswith("=> DOES NOT CONFORM")

    clean = check_module.check_file(str(EXAMPLE))
    assert clean.status == 0 and clean.framework == [] and clean.report.issues == []


@pytest.mark.skipif(not _can_register(),
                    reason="the framework cannot register Python validators before "
                           "OpenUSD 26.05 (PR #4015)")
def test_the_framework_path_agrees(broken):
    from pxr import Usd, UsdValidation

    assert register_validators()
    context = UsdValidation.ValidationContext([KEYWORD], True)
    errors = context.Validate(Usd.Stage.Open(broken))
    ours = [e for e in errors if e.GetIdentifier().startswith(PLUGIN_NAME + ":")]
    assert sorted(code_for(e.GetIdentifier()) for e in ours) == sorted(biousd.validate(broken).codes)
    prefixes = {e.GetIdentifier().rsplit(".", 1)[0] for e in ours}
    assert prefixes == {STAGE_VALIDATOR, POPULATION_VALIDATOR}
    assert context.Validate(Usd.Stage.Open(str(EXAMPLE))) == []


def test_the_command_line_exit_codes(tmp_path, broken):
    from pxr import Usd

    def run(*args):
        proc = subprocess.run([sys.executable, "-m", "cognisom.biousd.standard.check", *args],
                              capture_output=True, text=True, cwd=str(REPO))
        return proc.returncode, proc.stdout

    code, out = run(str(EXAMPLE))
    assert code == 0 and "=> CONFORMS" in out, out
    # the report says whether Bio-USD's own validators ran inside the
    # framework and, on a wheel where they cannot, why
    if _can_register():
        assert f"{PLUGIN_NAME} among them" in out
    else:
        assert f"{PLUGIN_NAME} not in the framework" in out and "26.05" in out, out
    code, out = run(broken)
    assert code == 1 and "=> DOES NOT CONFORM" in out and "biousd:primvar.token" in out
    code, out = run(str(tmp_path / "missing.usdc"))
    assert code == 2 and "COULD NOT BE READ" in out
    assert run("--help")[0] == 2

    # a 0.1-style file: one warning, clean unless --strict
    old = str(tmp_path / "old.usda")
    biousd.write_cells(old, _cells())
    stage = Usd.Stage.Open(old)
    stage.GetPrimAtPath("/World/Cells").ClearMetadata("apiSchemas")
    stage.GetRootLayer().Save()
    assert biousd.validate(old).codes == ["schema.notApplied"]
    assert run(old)[0] == 0
    assert run("--strict", old)[0] == 1


# ── loaded the way a framework client loads any plugin ──────────────


def _subprocess_json(code: str, env: dict) -> dict:
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(REPO), env=env)
    assert run.returncode == 0, run.stdout + run.stderr
    return json.loads(run.stdout.strip().splitlines()[-1])


def test_plug_loads_the_validators_plugin_lazily_with_nothing_of_ours_imported():
    """PXR_PLUGINPATH_NAME naming BOTH directories + PYTHONPATH, then one
    GetOrLoad: the shim finds the package by path and imports it as
    `biousd`. Registers on 26.05+; advertised but inert before, with no
    error."""
    env = dict(os.environ)
    env["BIOUSD_NO_PLUGIN"] = "1"
    env["PXR_PLUGINPATH_NAME"] = (biousd.plugin_path() + os.pathsep
                                  + validators.validators_plugin_path())
    env["PYTHONPATH"] = biousd.plugin_path() + os.pathsep + env.get("PYTHONPATH", "")
    out = _subprocess_json(f"""
import json, sys
from pxr import Usd, UsdValidation
reg = UsdValidation.ValidationRegistry()
advertised = sorted(m.name for m in reg.GetAllValidatorMetadata() if m.name.startswith({PLUGIN_NAME!r}))
loaded = reg.GetOrLoadValidatorByName({STAGE_VALIDATOR!r})
print(json.dumps(dict(advertised=advertised, loaded=loaded is not None,
                      has=bool(reg.HasValidator({STAGE_VALIDATOR!r}) and reg.HasValidator({POPULATION_VALIDATOR!r})),
                      suite=bool(reg.HasValidatorSuite({SUITE!r})),
                      shim="biousdValidators" in sys.modules, as_biousd="biousd" in sys.modules,
                      cognisom=[m for m in sys.modules if m.startswith("cognisom")])))
""", env)
    assert out["advertised"] == sorted([STAGE_VALIDATOR, POPULATION_VALIDATOR, SUITE])
    assert out["shim"] and out["as_biousd"] and out["cognisom"] == []
    assert out["has"] == out["loaded"] == out["suite"] == _can_register()


def _all_validators_context(extra: str, env: dict, path: str) -> dict:
    """A usdchecker-style context over every non-usdz validator, in a fresh
    process, after ``extra`` has run; what was advertised, what ours
    reported, and whether ours are registered afterwards."""
    return _subprocess_json(f"""
import json
{extra}
from pxr import Plug, Usd, UsdValidation
reg = UsdValidation.ValidationRegistry()
md = [m for m in reg.GetAllValidatorMetadata() if "UsdzValidators" not in m.GetKeywords()]
errors = UsdValidation.ValidationContext(md, True).Validate(Usd.Stage.Open({path!r}))
print(json.dumps(dict(
    plug=Plug.Registry().GetPluginWithName({PLUGIN_NAME!r}) is not None,
    known=Usd.SchemaRegistry().FindAppliedAPIPrimDefinition("BioCellAPI") is not None,
    advertised=sorted(m.name for m in md if m.name.startswith({PLUGIN_NAME!r})),
    ours=sorted(e.GetIdentifier() for e in errors if e.GetIdentifier().startswith({PLUGIN_NAME!r})),
    others=sorted(e.GetIdentifier() for e in errors if not e.GetIdentifier().startswith({PLUGIN_NAME!r})),
    registered=bool(reg.HasValidator({STAGE_VALIDATOR!r})))))
""", env)


def test_the_schema_env_var_route_advertises_no_validator_and_a_full_context_runs():
    """PXR_PLUGINPATH_NAME = plugin_path() alone, nothing on PYTHONPATH,
    nothing of ours imported: the README's route for a tool nobody
    controls. The first draft's plugInfo pulled the validators plugin in
    here, and every context over all validators raised 'Load of
    biousdValidators failed' on both wheels."""
    env = dict(os.environ)
    env["BIOUSD_NO_PLUGIN"] = "1"
    env["PXR_PLUGINPATH_NAME"] = biousd.plugin_path()
    out = _all_validators_context("", env, str(EXAMPLE))
    assert out["known"], "the schema is what the env-var route is for"
    assert not out["plug"] and out["advertised"] == [] and out["ours"] == []
    assert out["others"] == [] and not out["registered"]


def test_importing_the_package_then_a_context_over_all_validators_never_raises(broken):
    """`import biousd`, then what any framework client does. The first
    draft advertised the validators without making them importable, so
    this raised on both wheels. Now: on 26.05+ the import hands the
    plugin to Plug with plugin/ on sys.path, and the context loads and
    runs ours beside the builtin validators; before 26.05 nothing is
    advertised, so nothing can fail to load."""
    env = dict(os.environ)
    env["BIOUSD_NO_PLUGIN"] = ""
    env.pop("PXR_PLUGINPATH_NAME", None)
    out = _all_validators_context("import cognisom.biousd.standard as b", env, broken)
    assert out["known"]
    assert "usdValidation:StageMetadataChecker.MissingDefaultPrim" in out["others"]
    if _can_register():
        assert out["plug"] and out["registered"]
        assert out["advertised"] == sorted([STAGE_VALIDATOR, POPULATION_VALIDATOR, SUITE])
        assert sorted(code_for(i) for i in out["ours"]) == sorted(biousd.validate(broken).codes)
    else:
        assert not out["plug"] and not out["registered"]
        assert out["advertised"] == [] and out["ours"] == []


# ── not every stage is Bio-USD ──────────────────────────────────────


def _plain_stage(path: str):
    """An ordinary scene: an Xform, a Sphere, a PointInstancer scattering
    it, no bio:* anywhere, metersPerUnit left at USD's default."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Sphere.Define(stage, "/World/Ball")
    grass = UsdGeom.PointInstancer.Define(stage, "/World/Grass")
    grass.GetPrototypesRel().SetTargets(["/World/Ball"])
    grass.GetPositionsAttr().Set([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    grass.GetProtoIndicesAttr().Set([0, 0])
    stage.GetRootLayer().Save()
    return stage


def test_a_stage_that_is_not_biousd_gets_no_finding_from_the_framework_validators(tmp_path):
    """Registered, the stage validator reported StageMetersPerUnit,
    LayerVersion and CarrierNone on this scene and failed it. The task
    functions run on any wheel; the context over all validators needs
    26.05+. The standalone validator, pointed at the file on purpose,
    still says carrier.none."""
    from pxr import Usd
    from cognisom.biousd.standard.validate import is_biousd_stage
    from cognisom.biousd.standard.read import read_stage_info

    path = str(tmp_path / "plain.usda")
    stage = _plain_stage(path)
    assert not is_biousd_stage(read_stage_info(stage))
    assert stage_task(stage) == []
    assert population_task(stage.GetPrimAtPath("/World/Grass")) == []
    report = biousd.validate(path)
    assert not report.ok and "carrier.none" in {i.code for i in report.errors}
    if _can_register():
        from pxr import UsdValidation
        assert register_validators()
        reg = UsdValidation.ValidationRegistry()
        md = [m for m in reg.GetAllValidatorMetadata() if "UsdzValidators" not in m.GetKeywords()]
        errors = UsdValidation.ValidationContext(md, True).Validate(Usd.Stage.Open(path))
        assert [e.GetIdentifier() for e in errors
                if e.GetIdentifier().startswith(PLUGIN_NAME + ":")] == []
    # a stage that SAYS it is Bio-USD and holds no cells: that is carrier.none
    stage.GetRootLayer().customLayerData = {"bio:version": biousd.VERSION}
    assert is_biousd_stage(read_stage_info(stage))
    assert "carrier.none" in {code_for(e.GetName()) for e in stage_task(stage)}


# ── a wheel without shader definitions ──────────────────────────────


def _shaded(path: str) -> None:
    """A conforming stage plus the material every Cognisom stage carries:
    a UsdPreviewSurface fed by a primvar reader, bound to the prototype."""
    from pxr import Sdf, Usd, UsdShade

    biousd.write_cells(path, _cells())
    stage = Usd.Stage.Open(path)
    material = UsdShade.Material.Define(stage, "/World/Looks/Cell")
    surface = UsdShade.Shader.Define(stage, "/World/Looks/Cell/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    reader = UsdShade.Shader.Define(stage, "/World/Looks/Cell/Reader")
    reader.CreateIdAttr("UsdPrimvarReader_float3")
    reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("displayColor")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        reader.ConnectableAPI(), "result")
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/Prototypes/cell")).Bind(material)
    stage.GetRootLayer().Save()


def test_check_file_never_raises_on_a_stage_with_a_shader(tmp_path):
    """On the 25.11 wheel the usdShadeValidators asked Sdr for shader
    definitions the wheel does not ship: Tf.ErrorException out of
    check_file on the first call, three MissingShaderIdInRegistry errors
    on the second. Now an exit status on every call, the same one, with
    the skipped validators named on the wheel that lacks them."""
    path = str(tmp_path / "shaded.usda")
    _shaded(path)
    first = check_module.check_file(path)
    second = check_module.check_file(path)
    for result in (first, second):
        assert result.error == "" and result.framework_note == ""
        assert result.status == 0, check_module.describe(result)
    assert first.framework_validators == second.framework_validators
    assert first.framework_skipped == second.framework_skipped
    text = check_module.describe(first)
    if check_module.sdr_resolves_standard_shaders():
        assert first.framework_skipped == [] and "skipped" not in text
    else:
        assert first.framework_skipped, text
        assert all(n.startswith(check_module.SDR_DEPENDENT_PREFIX) for n in first.framework_skipped)
        assert "skipped" in text and "shaderDefs.usda" in text


@pytest.mark.skipif(not _can_register(),
                    reason="explicit registration needs the Python API of OpenUSD 26.05+")
def test_register_validators_registers_explicitly_when_plug_never_saw_the_plugin(broken):
    """The validation registry built before our plugInfo was registered:
    the names are not advertised, and register_validators() falls back to
    full-metadata registration, which needs no plugin."""
    env = dict(os.environ)
    env["BIOUSD_NO_PLUGIN"] = "1"
    env.pop("PXR_PLUGINPATH_NAME", None)
    out = _subprocess_json(f"""
import json
from pxr import Usd, UsdValidation
reg = UsdValidation.ValidationRegistry()
before = sorted(m.name for m in reg.GetAllValidatorMetadata() if m.name.startswith({PLUGIN_NAME!r}))
import cognisom.biousd.standard as b
ok = b.register_validators()
md = reg.GetValidatorMetadata({POPULATION_VALIDATOR!r})
errors = UsdValidation.ValidationContext([{KEYWORD!r}], True).Validate(Usd.Stage.Open({broken!r}))
print(json.dumps(dict(before=before, ok=ok, keywords=list(md.GetKeywords()),
                      suite=bool(reg.HasValidatorSuite({SUITE!r})),
                      identifiers=sorted(e.GetIdentifier() for e in errors))))
""", env)
    assert out["before"] == [] and out["ok"] and out["suite"]
    assert KEYWORD in out["keywords"]
    assert sorted(code_for(i) for i in out["identifiers"]) == sorted(biousd.validate(broken).codes)
