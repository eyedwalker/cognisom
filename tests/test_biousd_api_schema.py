"""The codeless BioCellAPI schema: generated, registered in time, optional.

What these pin, each measured before it was written down:

- the plugin is a build product of vocabulary.py, byte for byte;
- USD's schema registry is built once per process, so the plugin must be
  registered before the first stage is opened -- `import biousd` does
  it, a late call warns and does not fail;
- a file carrying `apiSchemas` opens in a process without the plugin,
  where `GetAppliedSchemas()` drops the name and `GetPrimTypeInfo()`
  keeps it;
- the 0.1 reader regression: with the plugin loaded, a conforming file
  that omits `bio:cell:volume` came back DOES NOT CONFORM (exit 1),
  because a schema-declared primvar is *defined* whether or not it is
  authored. The case now runs both ways, in subprocesses, because the
  registry state cannot be changed inside one process;
- its two siblings: a primvar with no authored interpolation and a
  primvar of the wrong value type each got a different verdict with the
  plugin (the fallback said `varying`, `GetTypeName()` said `float[]`)
  and without it. Both are errors both ways now: the file is what is
  judged, not what the registry resolves;
- the import registers with Plug and does NOT build the schema registry
  (asking whether the schema is known would), so a host's own schema
  plugin registered after `import biousd` is still known. The first
  draft asked at import and locked such a plugin out.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("pxr")

from cognisom.biousd import standard as biousd                             # noqa: E402
from cognisom.biousd.standard import api_schema, build_plugin, profiles     # noqa: E402
from cognisom.biousd.standard.vocabulary import (                          # noqa: E402
    API_SCHEMA, CELL_PRIMVARS, VALIDATOR_CODES, VERSION)

PKG = pathlib.Path(biousd.__file__).resolve().parent
REPO = PKG.parents[2]
EXAMPLE = PKG / "examples" / "minimal_200_cells.usda"


def _cells(n=30, seed=0):
    rng = np.random.default_rng(seed)
    return [dict(position=tuple(map(float, p)), cell_id=i,
                 cell_type="cancer" if i % 2 else "immune",
                 ontology_term="NCIT:C2919" if i % 2 else "CL:0000625",
                 phase="S", age=float(i), alive=True, volume=1.0, scale_um=10.0)
            for i, p in enumerate(rng.uniform(-30, 30, (n, 3)))]


def run_py(code: str, *, plugin: bool, env_extra=None) -> dict:
    """Run ``code`` in a fresh interpreter and return the JSON it prints last.

    Same executable, same environment (so a PYTHONPATH that shadows pxr
    with another usd-core carries over); ``plugin`` decides whether
    importing the package registers the schema plugin.
    """
    env = dict(os.environ)
    env["BIOUSD_NO_PLUGIN"] = "" if plugin else "1"
    env.pop("PXR_PLUGINPATH_NAME", None)
    env.update(env_extra or {})
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(REPO), env=env)
    assert run.returncode == 0, run.stdout + run.stderr
    return json.loads(run.stdout.strip().splitlines()[-1])


# ── generated, byte for byte ─────────────────────────────────────────


def test_the_schema_file_is_rendered_from_the_vocabulary():
    text = (PKG / "schema.usda").read_text()
    assert text == api_schema.render_schema_usda()
    for primvar in CELL_PRIMVARS:
        assert f"{primvar.usd_type} primvars:{primvar.name} (" in text
    assert f'class "{API_SCHEMA}"' in text and "skipCodeGeneration = true" in text
    assert f'string[] impliesCapabilities = ["{profiles.CAPABILITY_CELLS}"]' in text


def test_the_checked_in_plugin_is_byte_identical_to_a_fresh_build(tmp_path):
    """usdGenSchema, run again now, must give exactly what is shipped."""
    try:
        fresh = build_plugin.build(str(tmp_path))
    except build_plugin.BuildError as exc:
        pytest.skip(f"usdGenSchema cannot run here: {str(exc)[:200]}")
    assert set(fresh) == set(build_plugin.GENERATED)
    for rel, text in fresh.items():
        assert (PKG / rel).read_bytes() == text.encode(), f"{rel} is stale; run build_plugin"


def test_the_plugin_declares_the_schema_type_profiles_and_validators():
    info = json.loads((PKG / "plugin" / "plugInfo.json").read_text())
    # No Includes: pulling the python validators plugin in from here made
    # every framework client that found the schema through
    # PXR_PLUGINPATH_NAME raise 'Load of biousdValidators failed'.
    assert "Includes" not in info
    (plugin,) = info["Plugins"]
    assert plugin["Name"] == api_schema.LIBRARY_NAME == "bioUsd"
    assert plugin["Type"] == "resource" and plugin["Root"] == "." and plugin["ResourcePath"] == "."
    assert plugin["LibraryPath"] == ""
    entry = plugin["Info"]["Types"][api_schema.TF_TYPE]
    assert entry["schemaIdentifier"] == API_SCHEMA == "BioCellAPI"
    assert entry["alias"] == {"UsdSchemaBase": API_SCHEMA}
    assert entry["schemaKind"] == "singleApplyAPI"
    assert entry["apiSchemaCanOnlyApplyTo"] == ["PointInstancer"]
    assert entry["impliesCapabilities"] == [profiles.CAPABILITY_CELLS]
    assert plugin["Info"]["Profiles"] == profiles.render_plug_info_profiles()

    from pxr import Sdf
    layer = Sdf.Layer.FindOrOpen(str(PKG / "plugin" / "generatedSchema.usda"))
    spec = layer.GetPrimAtPath(f"/{API_SCHEMA}")
    assert spec is not None
    assert {p.name for p in spec.properties} == {f"primvars:{p.name}" for p in CELL_PRIMVARS}
    phase = spec.properties["primvars:bio:cell:phase"]
    assert list(phase.allowedTokens) == ["G0", "G1", "S", "G2", "M"]
    assert phase.GetInfo("interpolation") == "varying"
    assert VERSION in spec.customData["userDocBrief"]


# ── registration order ──────────────────────────────────────────────


def test_importing_the_package_registers_the_schema_in_time():
    out = run_py("""
import json
import cognisom.biousd.standard as b
from pxr import Tf, Usd
d = Usd.SchemaRegistry().FindAppliedAPIPrimDefinition("BioCellAPI")
print(json.dumps(dict(registered=b.schema_registered, known=b.schema_is_known(),
                      tf_type=not Tf.Type.FindByName("BioUsdBioCellAPI").isUnknown,
                      properties=sorted(d.GetPropertyNames()) if d else None,
                      kind=str(Usd.SchemaRegistry.GetSchemaKind("BioCellAPI")))))
""", plugin=True)
    assert out["registered"] and out["known"] and out["tf_type"]
    assert out["properties"] == sorted(f"primvars:{p.name}" for p in CELL_PRIMVARS)
    assert out["kind"].endswith("SingleApplyAPI")


def test_without_the_plugin_the_file_still_opens_and_the_name_is_kept():
    out = run_py(f"""
import json
import cognisom.biousd.standard as b
from pxr import Usd
stage = Usd.Stage.Open({str(EXAMPLE)!r})
prim = stage.GetPrimAtPath("/World/Cells")
report = b.validate({str(EXAMPLE)!r})
print(json.dumps(dict(known=b.schema_is_known(), registered=b.schema_registered,
                      applied=list(prim.GetAppliedSchemas()),
                      type_info=list(prim.GetPrimTypeInfo().GetAppliedAPISchemas()),
                      ok=report.ok, codes=report.codes)))
""", plugin=False)
    assert not out["known"] and not out["registered"]
    assert out["applied"] == [] and out["type_info"] == [API_SCHEMA]
    assert out["ok"] and out["codes"] == []


def test_a_late_registration_warns_once_and_does_not_fail():
    """Open a stage first (that builds the registry), register after.

    Plug takes the plugin either way (register_schema is True); the
    registry, built already, never learns it (schema_is_known is False),
    and the WARNING is logged there, once, however often it is asked."""
    out = run_py(f"""
import json, logging
records = []
class H(logging.Handler):
    def emit(self, r): records.append(r.getMessage())
logging.getLogger().addHandler(H()); logging.getLogger().setLevel(logging.WARNING)
import cognisom.biousd.standard as b          # BIOUSD_NO_PLUGIN=1: nothing registered yet
from pxr import Usd
stage = Usd.Stage.Open({str(EXAMPLE)!r})
n = len(stage.GetPrimAtPath("/World/Cells").GetPrimDefinition().GetPropertyNames())
first, second = b.register_schema(), b.register_schema()
silent = list(records)
known = [b.schema_is_known(), b.schema_is_known()]
print(json.dumps(dict(first=first, second=second, silent=silent, known=known,
                      warnings=records, props=n, ok=b.validate({str(EXAMPLE)!r}).ok)))
""", plugin=False)
    assert out["first"] is True and out["second"] is True
    assert out["silent"] == [], "registering with Plug is not where the registry is asked"
    assert out["known"] == [False, False]
    assert len(out["warnings"]) == 1 and "after this process built" in out["warnings"][0]
    assert "PXR_PLUGINPATH_NAME" in out["warnings"][0]
    assert out["ok"]


def _renamed_plugin(tmp_path) -> str:
    """The shipped plugin under other names (bioTwo / CellTwoAPI), profiles
    dropped: a stand-in for a schema plugin of the host's own."""
    src = PKG / "plugin"
    dest = tmp_path / "plugin_two"
    dest.mkdir()
    info = json.loads((src / "plugInfo.json").read_text())
    (plugin,) = info["Plugins"]
    plugin["Name"] = "bioTwo"
    types = plugin["Info"]["Types"]
    entry = types.pop(api_schema.TF_TYPE)
    entry["alias"] = {"UsdSchemaBase": "CellTwoAPI"}
    entry["schemaIdentifier"] = "CellTwoAPI"
    entry.pop("impliesCapabilities", None)
    types["BioTwoCellTwoAPI"] = entry
    plugin["Info"].pop("Profiles", None)
    (dest / "plugInfo.json").write_text(json.dumps(info, indent=4, sort_keys=True))
    schema = (src / "generatedSchema.usda").read_text()
    assert schema.count(f'class "{API_SCHEMA}"') == 1
    (dest / "generatedSchema.usda").write_text(
        schema.replace(f'class "{API_SCHEMA}"', 'class "CellTwoAPI"'))
    return str(dest)


def test_importing_the_package_does_not_build_the_registry_for_a_later_plugin(tmp_path):
    """A host registers its own schema plugin AFTER `import biousd` and
    before any stage: both schemas must be known. The first draft asked
    the registry at import whether BioCellAPI was known, which built it,
    and the host's plugin was never seen (measured: registered first,
    both known; imported first, the other unknown)."""
    other = _renamed_plugin(tmp_path)
    out = run_py(f"""
import json
import cognisom.biousd.standard as b
from pxr import Plug, Usd
n = len(Plug.Registry().RegisterPlugins({other!r}))
print(json.dumps(dict(registered=b.schema_registered, n=n,
                      known_two=Usd.SchemaRegistry().FindAppliedAPIPrimDefinition("CellTwoAPI") is not None,
                      known=b.schema_is_known())))
""", plugin=True)
    assert out["registered"] and out["n"] == 1
    assert out["known_two"], "import biousd built the schema registry and locked the host out"
    assert out["known"]


def test_the_env_var_route_registers_without_importing_first():
    out = run_py("""
import json
from pxr import Usd
known = Usd.SchemaRegistry().FindAppliedAPIPrimDefinition("BioCellAPI") is not None
import cognisom.biousd.standard as b
print(json.dumps(dict(known_before_import=known, known=b.schema_is_known())))
""", plugin=False, env_extra={"PXR_PLUGINPATH_NAME": biousd.plugin_path()})
    assert out["known_before_import"] and out["known"]


# ── apply_api and the writer ────────────────────────────────────────


def test_apply_api_needs_no_plugin_and_refuses_a_non_instancer(caplog):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    cells = UsdGeom.PointInstancer.Define(stage, "/World/Cells").GetPrim()
    assert biousd.apply_api(world) is False
    assert api_schema.applied_api_schemas(world) == []
    assert biousd.apply_api(cells) is True and biousd.apply_api(cells) is True
    assert api_schema.applied_api_schemas(cells) == [API_SCHEMA]
    assert "not a PointInstancer" in caplog.text


def test_the_writer_applies_the_schema_and_still_authors_every_interpolation(tmp_path):
    """A 0.1 reader must see no difference: explicit metadata, not fallbacks."""
    from pxr import Usd, UsdGeom

    path = str(tmp_path / "w.usda")
    report = biousd.write_cells(path, _cells())
    assert report.api_applied and report.claims_prim == "/World"
    text = pathlib.Path(path).read_text()
    assert f'prepend apiSchemas = ["{API_SCHEMA}"]' in text
    assert text.count('interpolation = "varying"') == len(CELL_PRIMVARS)
    stage = Usd.Stage.Open(path)                 # kept: a prim outlives no stage
    prim = stage.GetPrimAtPath("/World/Cells")
    for primvar in CELL_PRIMVARS:
        attr = prim.GetAttribute(f"primvars:{primvar.name}")
        assert attr.HasAuthoredMetadata("interpolation"), primvar.name
        assert UsdGeom.Primvar(attr).GetInterpolation() == "varying"


# ── the reader regression, both ways ────────────────────────────────


def _drop_volume(path: str) -> None:
    from pxr import Usd

    stage = Usd.Stage.Open(path)
    assert stage.GetPrimAtPath("/World/Cells").RemoveProperty("primvars:bio:cell:volume")
    stage.GetRootLayer().Save()


def _make_0_1_style(path: str) -> None:
    """The spike's exact case: a 0.1.2 file given the API by hand, no claims."""
    from pxr import Usd

    stage = Usd.Stage.Open(path)
    stage.GetRootLayer().customLayerData = {"bio:version": "0.1.2"}
    world = stage.GetPrimAtPath("/World")
    world.ClearMetadata("apiSchemas")
    world.ClearCustomData()
    stage.GetRootLayer().Save()


def _validate_in_subprocess(path: str, plugin: bool) -> dict:
    return run_py(f"""
import json
import cognisom.biousd.standard as b
r = b.validate({path!r})
print(json.dumps(dict(known=b.schema_is_known(), ok=r.ok, codes=r.codes,
                      warnings=[str(i) for i in r.warnings],
                      errors=[i.message for i in r.errors])))
""", plugin=plugin)


@pytest.mark.parametrize("style", ["written_by_0_2", "spike_0_1_file"])
def test_an_omitted_optional_primvar_is_one_warning_with_and_without_the_plugin(tmp_path, style):
    path = str(tmp_path / f"{style}.usda")
    biousd.write_cells(path, _cells())
    if style == "spike_0_1_file":
        _make_0_1_style(path)
    _drop_volume(path)
    with_plugin = _validate_in_subprocess(path, plugin=True)
    without = _validate_in_subprocess(path, plugin=False)
    assert with_plugin["known"] and not without["known"]
    for out in (with_plugin, without):
        assert out["ok"], out
        assert out["codes"] == ["primvar.optional"], out["warnings"]
    assert with_plugin["warnings"] == without["warnings"]


def test_an_unauthored_interpolation_is_an_error_with_and_without_the_plugin(tmp_path):
    """The schema supplies `varying` as a fallback; a reader without it
    sees `constant`. The first 0.2 draft reported a warning with the
    plugin and an error without, so the verdict depended on the process
    that ran the validator. The fallback is not the file: an error, with
    one message, both ways."""
    from pxr import Usd

    path = str(tmp_path / "noint.usda")
    biousd.write_cells(path, _cells())
    stage = Usd.Stage.Open(path)
    stage.GetPrimAtPath("/World/Cells").GetAttribute("primvars:bio:cell:age").ClearMetadata(
        "interpolation")
    stage.GetRootLayer().Save()
    with_plugin = _validate_in_subprocess(path, plugin=True)
    without = _validate_in_subprocess(path, plugin=False)
    assert with_plugin["known"] and not without["known"]
    for out in (with_plugin, without):
        assert not out["ok"] and out["codes"] == ["primvar.interpolation"], out
    assert with_plugin["errors"] == without["errors"]
    assert "authors no interpolation" in with_plugin["errors"][0]


def test_a_wrong_value_type_is_an_error_with_and_without_the_plugin(tmp_path):
    """`int[] primvars:bio:cell:age`, authored at the Sdf level. With the
    plugin, `GetTypeName()` resolved to the schema's `float[]` and the file
    CONFORMED; without it, `primvar.type`. The reader now records the
    authored type from the property stack, which the registry never
    touches, so the verdict is the same in both processes."""
    from pxr import Sdf, Vt

    path = str(tmp_path / "intage.usda")
    biousd.write_cells(path, _cells())
    layer = Sdf.Layer.FindOrOpen(path)
    prim = layer.GetPrimAtPath("/World/Cells")
    del prim.properties["primvars:bio:cell:age"]
    attr = Sdf.AttributeSpec(prim, "primvars:bio:cell:age", Sdf.ValueTypeNames.IntArray)
    attr.default = Vt.IntArray([1] * 30)
    attr.SetInfo("interpolation", "varying")
    layer.Save()
    assert "int[] primvars:bio:cell:age" in pathlib.Path(path).read_text()
    with_plugin = _validate_in_subprocess(path, plugin=True)
    without = _validate_in_subprocess(path, plugin=False)
    assert with_plugin["known"] and not without["known"]
    for out in (with_plugin, without):
        assert not out["ok"] and out["codes"] == ["primvar.type"], out
        assert "is int[], must be float[]" in out["errors"][0]


# ── the schema codes ────────────────────────────────────────────────


def test_misapplied_is_an_error_and_not_applied_a_warning(tmp_path):
    from pxr import Usd

    path = str(tmp_path / "mis.usda")
    biousd.write_cells(path, _cells())
    stage = Usd.Stage.Open(path)
    assert stage.GetPrimAtPath("/World").AddAppliedSchema(API_SCHEMA)   # USD allows it
    stage.GetRootLayer().Save()
    report = biousd.validate(path)
    assert not report.ok
    assert [i.code for i in report.errors] == ["schema.misapplied"]
    assert report.errors[0].site == "/World"

    path = str(tmp_path / "not.usda")
    biousd.write_cells(path, _cells())
    stage = Usd.Stage.Open(path)
    stage.GetPrimAtPath("/World/Cells").ClearMetadata("apiSchemas")
    stage.GetRootLayer().Save()
    report = biousd.validate(path)
    assert report.ok and report.codes == ["schema.notApplied"]


def test_a_versioned_schema_identifier_is_a_warning(tmp_path):
    from pxr import Usd

    path = str(tmp_path / "v1.usda")
    biousd.write_cells(path, _cells())
    stage = Usd.Stage.Open(path)
    stage.GetPrimAtPath("/World/Cells").AddAppliedSchema(f"{API_SCHEMA}_1")
    stage.GetRootLayer().Save()
    report = biousd.validate(path)
    assert report.ok and report.codes == ["schema.version"]
    assert f"{API_SCHEMA}_1" in report.warnings[0].message


def test_a_0_1_file_still_conforms_with_one_warning(tmp_path):
    """No apiSchemas, no claims, bio:version 0.1.2: what every 0.1 writer made."""
    from pxr import Usd

    path = str(tmp_path / "old.usda")
    biousd.write_cells(path, _cells())
    _make_0_1_style(path)
    stage = Usd.Stage.Open(path)
    stage.GetPrimAtPath("/World/Cells").ClearMetadata("apiSchemas")
    stage.GetRootLayer().Save()
    report = biousd.validate(path)
    assert report.ok and report.codes == ["schema.notApplied"], report.describe()


def test_the_examples_apply_the_schema_and_validate_clean():
    for name in ("minimal_200_cells.usda", "tissue_slab_10k_cells.usdc",
                 "coloured_by_age_2k_cells.usdc"):
        report = biousd.validate(str(PKG / "examples" / name))
        assert report.ok and report.issues == [], report.describe()
        table = biousd.read_cells(str(PKG / "examples" / name))
        assert table.api_applied and API_SCHEMA in table.applied_schemas
    text = EXAMPLE.read_text()
    assert f'prepend apiSchemas = ["{API_SCHEMA}"]' in text
    assert 'prepend apiSchemas = ["ClaimsAPI"]' in text


def test_every_code_the_validator_emits_is_in_the_table_and_vice_versa():
    source = "".join((PKG / name).read_text() for name in ("validate.py", "profiles.py"))
    emitted = set(re.findall(r'(?:_error|_warning|report\.error|report\.warn)\(\s*"([a-z]+\.[A-Za-z]+)"',
                             source))
    emitted |= set(re.findall(r'\(\s*"(profile\.[A-Za-z]+)"', source))
    assert emitted == set(VALIDATOR_CODES), emitted ^ set(VALIDATOR_CODES)
    for code, (level, _meaning) in VALIDATOR_CODES.items():
        assert level in ("error", "warning", "error / warning"), code


def test_the_vendored_build_inputs_are_declared():
    notice = (PKG / "NOTICE").read_text()
    for rel in ("vendor/openusd/usd/schema.usda", "vendor/openusd/codegenTemplates/plugInfo.json"):
        assert (PKG / rel).exists(), rel
        assert rel in notice
    assert "v25.11" in notice and "Tomorrow Open Source Technology License" in notice
    assert (PKG / "vendor" / "openusd" / "LICENSE.txt").exists()
    usd_dir, templates = build_plugin.vendored_paths()
    assert os.path.isfile(os.path.join(usd_dir, "schema.usda"))
    assert os.path.isfile(os.path.join(templates, "plugInfo.json"))
