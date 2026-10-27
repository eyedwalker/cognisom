"""Bio-USD as a package a stranger could use: the checks that make it one.

Before this, "Bio-USD" was a schema file nothing applied, a writer whose
vocabulary differed from that schema by one attribute each way, a
validator that checked Python objects rather than files, a README that
claimed MIT beside a schema header that claimed Apache 2.0, and no
LICENSE at all. Each of those is a reason a third party could not use
it, and each has a test here.

The package lives at cognisom/biousd/standard and is importable both as
``cognisom.biousd.standard`` and, after extraction, as ``biousd``. It
must therefore import nothing from Cognisom.
"""

from __future__ import annotations

import ast
import os
import pathlib
import re
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("pxr")

from cognisom.biousd import standard as biousd                       # noqa: E402
from cognisom.biousd.standard.vocabulary import (                    # noqa: E402
    CELL_PRIMVARS, REQUIRED_PRIMVARS, VERSION, render_spec_markdown)

PKG = pathlib.Path(biousd.__file__).resolve().parent
REPO = PKG.parents[2]


def _cells(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return [dict(position=tuple(map(float, p)), cell_id=i,
                 cell_type="cancer" if i % 2 else "immune",
                 ontology_term="NCIT:C2919" if i % 2 else "CL:0000625",
                 phase="S", age=float(i), alive=True, volume=1.0, scale_um=10.0)
            for i, p in enumerate(rng.uniform(-30, 30, (n, 3)))]


# ── the package stands alone ────────────────────────────────────────


def test_the_package_imports_nothing_from_cognisom():
    """Extracted, there is no Cognisom to import."""
    for path in PKG.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            for name in names:
                assert not name.startswith("cognisom"), f"{path.name} imports {name}"


def test_every_package_import_is_relative():
    """So the same files work as `biousd` and as `cognisom.biousd.standard`."""
    for path in PKG.rglob("*.py"):
        if path.name == "make_examples.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and (
                    node.module.startswith("biousd") or node.module == "standard"):
                assert node.level > 0, f"{path.name}: absolute self-import {node.module}"


def test_the_licence_is_the_apache_text_and_is_granted():
    text = (PKG / "LICENSE").read_text()
    assert text.lstrip().startswith("Apache License")
    assert "Version 2.0, January 2004" in text
    assert "APPENDIX: How to apply the Apache License" in text
    assert len(text.splitlines()) == 201, "not the canonical text"
    notice = (PKG / "NOTICE").read_text()
    assert "Copyright" in notice and "Apache" in notice


def test_one_version_everywhere():
    assert biousd.__version__ == VERSION
    assert f'version = "{VERSION}"' in (PKG / "pyproject.toml").read_text()
    assert f"version: {VERSION}" in (PKG / "CITATION.cff").read_text()
    assert f"Bio-USD {VERSION}" in (PKG / "SPEC.md").read_text()
    assert f"v{VERSION}" in (PKG / "README.md").read_text()
    assert f"**{VERSION}**" in (PKG / "README.md").read_text(), "no Changes entry"
    assert f"Bio-USD {VERSION}" in (PKG / "schema.usda").read_text()
    assert f"(Bio-USD {VERSION})" in (PKG / "plugin" / "generatedSchema.usda").read_text()
    for name in ("minimal_200_cells.usda", "tissue_slab_10k_cells.usdc",
                 "coloured_by_age_2k_cells.usdc"):
        assert biousd.read_stage(str(PKG / "examples" / name)).layer_data["bio:version"] == VERSION


def test_the_spec_is_rendered_from_the_vocabulary():
    """Edit vocabulary.py, re-render; a hand edit to SPEC.md fails here."""
    assert (PKG / "SPEC.md").read_text() == render_spec_markdown()


def test_the_spec_names_every_primvar_and_only_those():
    spec = (PKG / "SPEC.md").read_text()
    for primvar in CELL_PRIMVARS:
        assert f"`{primvar.name}`" in spec
    found = set(re.findall(r"`(bio:cell:[A-Za-z]+)`", spec))
    assert found == {p.name for p in CELL_PRIMVARS}


# ── write, read, validate ───────────────────────────────────────────


def test_a_round_trip_returns_what_went_in(tmp_path):
    path = str(tmp_path / "rt.usdc")
    report = biousd.write_cells(path, _cells())
    assert report.cells == 40 and report.rejected_terms == []
    table = biousd.read_cells(path)
    assert table.n == 40
    assert list(table["bio:cell:id"]) == list(range(40))
    assert table["bio:cell:type"][1] == "cancer" and table["bio:cell:type"][0] == "immune"
    assert table["bio:cell:ontologyTerm"][1] == "NCIT:C2919"
    assert float(table.diameters[0]) == 10.0
    assert set(table.primvars) == {p.name for p in CELL_PRIMVARS}


def test_a_written_stage_conforms(tmp_path):
    path = str(tmp_path / "ok.usdc")
    biousd.write_cells(path, _cells())
    report = biousd.validate(path)
    assert report.ok, report.describe()
    assert report.cells == 40 and report.populations == 1


def test_a_malformed_term_is_written_empty_and_reported(tmp_path):
    cells = _cells(4)
    cells[2]["ontology_term"] = "T cell"
    report = biousd.write_cells(str(tmp_path / "t.usdc"), cells)
    assert report.rejected_terms == ["T cell"]
    assert biousd.read_cells(str(tmp_path / "t.usdc"))["bio:cell:ontologyTerm"][2] == ""


def _tampered(tmp_path, name, mutate):
    """A fresh file per tamper: Usd.Stage.Open returns a cached stage for
    a path already open in this process, so mutations would pile up."""
    from pxr import Usd, UsdGeom

    source = str(tmp_path / f"{name}_src.usdc")
    biousd.write_cells(source, _cells())
    stage = Usd.Stage.Open(source)
    instancer = UsdGeom.PointInstancer(stage.GetPrimAtPath("/World/Cells"))
    mutate(stage, instancer, UsdGeom.PrimvarsAPI(instancer.GetPrim()))
    out = str(tmp_path / f"{name}.usdc")
    stage.GetRootLayer().Export(out)
    return biousd.validate(out)


@pytest.mark.parametrize("name, code, mutate", [
    ("missing_type", "primvar.missing",
     lambda st, inst, api: api.RemovePrimvar("bio:cell:type")),
    ("short_age", "primvar.length",
     lambda st, inst, api: api.GetPrimvar("bio:cell:age").Set(__import__("pxr").Vt.FloatArray([1.0]))),
    ("bad_curie", "primvar.curie",
     lambda st, inst, api: api.GetPrimvar("bio:cell:ontologyTerm").Set(
         __import__("pxr").Vt.TokenArray(["T cell"] * 40))),
    ("bad_phase", "primvar.token",
     lambda st, inst, api: api.GetPrimvar("bio:cell:phase").Set(
         __import__("pxr").Vt.TokenArray(["mitosis"] * 40))),
    ("dup_ids", "primvar.id",
     lambda st, inst, api: api.GetPrimvar("bio:cell:id").Set(
         __import__("pxr").Vt.IntArray([7] * 40))),
    ("bad_units", "stage.metersPerUnit",
     lambda st, inst, api: __import__("pxr").UsdGeom.SetStageMetersPerUnit(st, 0.01)),
    ("no_version", "layer.version",
     lambda st, inst, api: setattr(st.GetRootLayer(), "customLayerData", {})),
])
def test_each_kind_of_damage_is_caught(tmp_path, name, code, mutate):
    report = _tampered(tmp_path, name, mutate)
    assert not report.ok
    assert code in {i.code for i in report.errors}, report.describe()


def test_a_file_that_is_not_usd_is_reported_not_raised(tmp_path):
    report = biousd.validate(str(tmp_path / "nothing.usdc"))
    assert not report.ok and report.errors[0].code == "stage.open"


def test_the_command_line_entry_exits_by_status(tmp_path):
    ok = str(tmp_path / "ok.usdc")
    biousd.write_cells(ok, _cells())
    run = subprocess.run([sys.executable, "-m", "cognisom.biousd.standard.validate", ok],
                         capture_output=True, text=True, cwd=str(REPO))
    assert run.returncode == 0, run.stdout + run.stderr
    assert "CONFORMS" in run.stdout
    run = subprocess.run([sys.executable, "-m", "cognisom.biousd.standard.validate",
                          str(tmp_path / "missing.usdc")], capture_output=True, text=True,
                         cwd=str(REPO))
    assert run.returncode == 2


# ── the examples ship conforming ────────────────────────────────────


@pytest.mark.parametrize("name", ["minimal_200_cells.usda", "tissue_slab_10k_cells.usdc",
                                  "coloured_by_age_2k_cells.usdc"])
def test_each_example_conforms_and_is_small(name):
    path = PKG / "examples" / name
    assert path.exists(), f"{name} missing; run make_examples"
    assert path.stat().st_size < 1_500_000
    report = biousd.validate(str(path))
    assert report.ok, report.describe()


def test_the_readable_example_is_readable():
    text = (PKG / "examples" / "minimal_200_cells.usda").read_text()
    assert text.startswith("#usda 1.0")
    assert 'primvars:bio:cell:ontologyTerm' in text
    assert '"bio:version"' in text


def test_the_readme_example_conforms(tmp_path):
    """The README's own usda block, saved verbatim, must conform and pass
    biousd-check: a reader who copies it got DOES NOT CONFORM before (no
    prototypes relationship behind protoIndices, and an ontologyTerm
    line without its interpolation)."""
    text = (PKG / "README.md").read_text()
    match = re.search(r"```\n(#usda 1\.0\n.*?)```", text, re.S)
    assert match, "the README has lost its usda example"
    path = tmp_path / "readme.usda"
    path.write_text(match.group(1))
    report = biousd.validate(str(path))
    assert report.ok, report.describe()
    assert report.populations == 1 and report.cells == 2
    assert biousd.check_file(str(path)).status == 0


# ── Cognisom's own writer and schema agree with the standard ────────


def test_cognisoms_writer_authors_exactly_the_vocabulary():
    """The two drifted by one attribute each way before; this pins them."""
    source = (REPO / "cognisom" / "biousd" / "usd_writer.py").read_text()
    authored = set(re.findall(r'primvar\("(bio:cell:[A-Za-z]+)"', source))
    assert authored == {p.name for p in CELL_PRIMVARS}, authored ^ {p.name for p in CELL_PRIMVARS}


def test_cognisoms_schema_file_declares_every_primvar():
    schema = (REPO / "cognisom" / "biousd" / "schemas" / "biousd_core.usda").read_text()
    block = schema[schema.index('class BioCell "BioCell"'):]
    block = block[:block.index("\n}")]
    for primvar in CELL_PRIMVARS:
        assert primvar.name in block, f"schema BioCell lacks {primvar.name}"


def test_a_cognisom_written_scene_conforms(tmp_path):
    """The full-featured writer must produce files its own standard accepts."""
    from cognisom.biousd.usd_writer import write_cells as cognisom_write

    class _Cell:
        def __init__(self, i, p):
            self.position = tuple(map(float, p)); self.cell_type = "cancer"
            self.scale_um = 12.0; self.volume = 1.0; self.cell_id = i
            self.name = "Prostatic adenocarcinoma cell"
    rng = np.random.default_rng(3)
    path = str(tmp_path / "cog.usdc")
    cognisom_write(path, [_Cell(i, p) for i, p in enumerate(rng.uniform(-40, 40, (30, 3)))])
    report = biousd.validate(path)
    assert report.ok, report.describe()
    table = biousd.read_cells(path)
    assert table["bio:cell:ontologyTerm"][0] == "NCIT:C2919"


def test_required_is_exactly_id_and_type():
    assert REQUIRED_PRIMVARS == ("bio:cell:id", "bio:cell:type")


# ── the starter term map ────────────────────────────────────────────


def test_every_starter_term_is_a_well_formed_curie():
    from cognisom.biousd.standard import terms

    for name, curie in {**terms.CELL_CURIE, **terms.TYPE_CURIE}.items():
        assert biousd.is_curie(curie) and curie, (name, curie)
    assert terms.coverage()[0] >= 40
    assert "normal" not in terms.TYPE_CURIE, "normal is a state, not a lineage"


def test_cognisoms_ontology_module_is_the_same_map():
    """One home for the data; Cognisom re-exports it."""
    from cognisom.biousd import ontology
    from cognisom.biousd.standard import terms

    assert ontology.CELL_CURIE is terms.CELL_CURIE
    assert ontology.curie_for("Macrophage M1") == "CL:0000863"
    assert ontology.curie_for("Macrophage M2") == "CL:0000890"
    assert ontology.curie_for("Regulatory T cell") == ontology.curie_for(
        "Regulatory T cell (Treg)")
    assert ontology.curie_for("no such cell") is None


# ── the extraction, rehearsed ───────────────────────────────────────


def _copy_package(tmp_path) -> pathlib.Path:
    import shutil

    dest = tmp_path / "biousd"
    shutil.copytree(PKG, dest, ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))
    return dest


def test_the_package_runs_standalone_as_biousd(tmp_path):
    """`python -I`: no PYTHONPATH, no user site, the repository nowhere on
    sys.path. The copy must import as `biousd`, register its plugin, and
    validate and check its own example."""
    import json

    root = _copy_package(tmp_path).parent
    example = str(root / "biousd" / "examples" / "minimal_200_cells.usda")
    # -I drops cwd from sys.path too (it implies -P), so the copy's parent
    # is put there by hand; the 26.11 shadow, when present, comes in the
    # same way because -I ignores PYTHONPATH.
    extra = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    code = f"""
import json, sys
sys.path[:0] = {extra!r} + [{str(root)!r}]
import biousd
from biousd.check import main as check_main
from biousd.validate import main as validate_main
print(json.dumps(dict(version=biousd.__version__, known=biousd.schema_is_known(),
                      registered=biousd.schema_registered,
                      plugin=biousd.plugin_path(),
                      validate=validate_main([{example!r}]), check=check_main([{example!r}]),
                      claims=biousd.check_claims(__import__("pxr").Usd.Stage.Open({example!r})),
                      cognisom=[m for m in sys.modules if m.startswith("cognisom")])))
"""
    run = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True,
                         cwd=str(tmp_path))
    assert run.returncode == 0, run.stdout + run.stderr
    out = json.loads(run.stdout.strip().splitlines()[-1])
    assert out["version"] == VERSION and out["known"] and out["registered"]
    assert out["plugin"] == str(root / "biousd" / "plugin")
    assert out["validate"] == 0 and out["check"] == 0 and out["claims"] == []
    assert out["cognisom"] == []
    assert "CONFORMS" in run.stdout


def test_a_wheel_ships_the_plugin_and_installs(tmp_path):
    """pip wheel on the copy, then import from the installed wheel alone.

    0.1.2's pyproject.toml could not build a wheel at all (keywords and
    classifiers filed under optional-dependencies); this is the check
    that was missing.
    """
    import json
    import zipfile

    src = _copy_package(tmp_path)
    wheels = tmp_path / "wheels"
    run = subprocess.run([sys.executable, "-m", "pip", "wheel", str(src), "--no-deps",
                          "--no-build-isolation", "-w", str(wheels), "-q"],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    (wheel,) = list(wheels.glob("biousd-*.whl"))
    assert f"biousd-{VERSION}-" in wheel.name
    names = set(zipfile.ZipFile(wheel).namelist())
    for rel in ("schema.usda", "plugin/plugInfo.json", "plugin/generatedSchema.usda",
                "plugin/biousdValidators/__init__.py", "plugin/biousdValidators/plugInfo.json",
                "vendor/openusd/usd/schema.usda", "vendor/openusd/codegenTemplates/plugInfo.json",
                "vendor/openusd/LICENSE.txt", "examples/minimal_200_cells.usda",
                "README.md", "SPEC.md", "LICENSE", "NOTICE", "CITATION.cff",
                "check.py", "api_schema.py", "profiles.py", "validators.py", "build_plugin.py"):
        assert f"biousd/{rel}" in names, rel
    entry_points = next(n for n in names if n.endswith("entry_points.txt"))
    scripts = zipfile.ZipFile(wheel).read(entry_points).decode()
    for script in ("biousd-validate", "biousd-check", "biousd-build-plugin",
                   "biousd-from-physicell"):
        assert f"{script} = " in scripts

    site = tmp_path / "site"
    run = subprocess.run([sys.executable, "-m", "pip", "install", str(wheel), "--no-deps",
                          "--target", str(site), "-q"], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    # Inherited PYTHONPATH entries are made ABSOLUTE against this process's
    # cwd, and the installation comes first. The subprocess runs in tmp_path,
    # which holds the source copy this wheel was built from, so a relative
    # entry -- `PYTHONPATH=.`, which is how the full suite runs -- resolved
    # to tmp_path and `import biousd` returned the copy, not the wheel. The
    # test then proved nothing about the installation and said so by
    # failing on the path assertion below.
    extra = [os.path.abspath(p)
             for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    code = f"""
import json, os, sys
sys.path[:0] = [{str(site)!r}] + {extra!r}
import biousd
example = os.path.join(os.path.dirname(biousd.__file__), "examples", "minimal_200_cells.usda")
print(json.dumps(dict(file=biousd.__file__, known=biousd.schema_is_known(),
                      plugin_exists=os.path.isfile(os.path.join(biousd.plugin_path(), "plugInfo.json")),
                      status=biousd.check_file(example).status)))
"""
    run = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True,
                         cwd=str(tmp_path))
    assert run.returncode == 0, run.stdout + run.stderr
    out = json.loads(run.stdout.strip().splitlines()[-1])
    assert out["file"].startswith(str(site)) and out["known"] and out["plugin_exists"]
    assert out["status"] == 0
