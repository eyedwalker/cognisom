"""Every .usda the repo ships or emits must open in OpenUSD.

Three of the five USDA writers built their output by string concatenation and
were never validated, so they shipped text that `Usd.Stage.Open` refuses:

  * `biousd/schemas/biousd_core.usda` used commas to separate dictionary
    entries, which USD does not accept;
  * `biousd/precision_schema.usda` left colon-bearing dictionary keys
    unquoted, which USD also does not accept;
  * `biousd/converter.py` emitted the same unquoted colon keys into
    `customLayerData`;
  * `biousd/sbml_converter.py` emitted `custom dictionary ... = {}` as an
    attribute, and `dictionary` is not a valid USDA attribute type -- a
    key/value block has to be prim metadata.

Each produced rich, plausible content. The two writers that use the `pxr` API
instead of templating text -- `kit_extension/.../pdb_to_usd.py` and
`visualization/exporters.cells_to_usd` -- were both valid on the first try.

The semantic validator in `biousd/validator.py` did not catch any of this
because it validates the Python dataclasses and never the emitted text. That
gap is what these tests close.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

pxr_usd = pytest.importorskip("pxr.Usd", reason="OpenUSD (usd-core) not installed")
from pxr import Sdf, Usd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Committed .usda files, excluding build output, vendored trees and
#: .claude/worktrees, which holds detached checkouts of other branches.
_EXCLUDE_PARTS = {
    "node_modules", "dist", "build", ".git", ".claude", "PhysiCell",
}


def _committed_usda_files() -> list[Path]:
    return sorted(
        p for p in REPO_ROOT.rglob("*.usda")
        if not _EXCLUDE_PARTS.intersection(p.parts)
    )


def _first_error_line(exc: Exception) -> str:
    m = re.search(r"(\S+\.usda:\d+:\d+: [^\n]+)", str(exc))
    return m.group(1) if m else str(exc)[:300]


def test_there_are_usda_files_to_check():
    """A silent zero here would make every test below vacuous."""
    assert _committed_usda_files(), "found no .usda files to validate"


@pytest.mark.parametrize(
    "usda", _committed_usda_files(), ids=lambda p: p.relative_to(REPO_ROOT).as_posix()
)
def test_committed_usda_file_opens(usda: Path):
    try:
        layer = Sdf.Layer.FindOrOpen(str(usda))
    except Exception as e:  # pragma: no cover - the failure is the point
        pytest.fail(f"{usda.relative_to(REPO_ROOT)} does not parse: {_first_error_line(e)}")

    assert layer is not None, f"{usda.relative_to(REPO_ROOT)} did not open"


def test_the_biousd_schema_defines_its_prim_types():
    """Opening is necessary but not sufficient -- the content must survive."""
    layer = Sdf.Layer.FindOrOpen(
        str(REPO_ROOT / "cognisom/biousd/schemas/biousd_core.usda")
    )
    names = {p.name for p in layer.rootPrims}

    for expected in ("BioUnit", "BioCell", "BioImmuneCell", "BioGene", "BioProtein"):
        assert expected in names, f"{expected} missing from the schema"


def test_schema_range_metadata_reads_back():
    """The commas lived inside these dictionaries, so check one reads back."""
    layer = Sdf.Layer.FindOrOpen(
        str(REPO_ROOT / "cognisom/biousd/schemas/biousd_core.usda")
    )
    for prim in layer.rootPrims:
        for prop in prim.properties:
            if prop.name == "bio:immune:detectionRadius":
                data = dict(prop.GetInfo("customData"))
                assert data == {"min": 1.0, "max": 100.0}
                return
    pytest.fail("bio:immune:detectionRadius not found in the schema")


def test_precision_schema_keeps_its_api_schemas():
    layer = Sdf.Layer.FindOrOpen(
        str(REPO_ROOT / "cognisom/biousd/precision_schema.usda")
    )
    names = {p.name for p in layer.rootPrims}

    assert "BioPrecisePointAPI" in names
    assert "BiologicalScaleAPI" in names


class TestEmittedUSD:
    """The writers, not just the committed files."""

    def test_biousd_scene_export_reopens(self, tmp_path):
        from cognisom.biousd.converter import export_demo_scene

        out = tmp_path / "demo.usda"
        export_demo_scene(str(out))

        assert out.exists(), "export wrote nothing"
        try:
            stage = Usd.Stage.Open(str(out))
        except Exception as e:
            pytest.fail(f"exported scene does not reopen: {_first_error_line(e)}")
        assert len(list(stage.Traverse())) > 1

    def test_sbml_conversion_reopens_and_keeps_its_parameters(self, tmp_path):
        from cognisom.biousd.sbml_converter import SBMLToUSD

        src = REPO_ROOT / "cognisom/models/pathways/gene_expression.xml"
        if not src.exists():
            pytest.skip("bundled SBML model not present")

        conv = SBMLToUSD()
        conv.from_file(str(src))
        out = tmp_path / "sbml.usda"
        conv.export_usda(str(out))

        try:
            stage = Usd.Stage.Open(str(out))
        except Exception as e:
            pytest.fail(f"SBML conversion does not reopen: {_first_error_line(e)}")

        world = stage.GetPrimAtPath("/World")
        assert world, "/World missing from the converted scene"
        # The kinetic parameters were the content that made the file invalid.
        data = world.GetCustomData()
        assert conv.model.parameters, "no parameters parsed from the SBML"
        for key in conv.model.parameters:
            assert key in data, f"kinetic parameter {key} did not survive export"


class TestPrecisionManager:
    def test_it_constructs_against_a_real_stage(self):
        """It raised unconditionally, which disabled the whole USD path."""
        from cognisom.omniverse.precision import PrecisionTransformManager

        stage = Usd.Stage.CreateInMemory()
        manager = PrecisionTransformManager(stage)

        assert manager is not None
        data = dict(stage.GetMetadata("customLayerData") or {})
        assert data.get("cognisom:precisionMode") == "localFloat_globalDouble"
        assert "cognisom:biologicalScale" in data

    def test_the_real_connector_creates_a_stage_on_its_own_defaults(self, tmp_path):
        """enable_precision_mode defaults to True, and this returned False.

        The precision call sits inside create_stage's try block, so the
        dashboard's "Generate USD Scene" button fell through to an error.
        """
        from cognisom.omniverse.real_connector import RealOmniverseConnector

        connector = RealOmniverseConnector(output_dir=str(tmp_path))

        assert connector.create_stage("smoke") is True

        written = list(tmp_path.glob("*.usda"))
        assert written, "connector reported success but wrote no stage"
        for path in written:
            try:
                Usd.Stage.Open(str(path))
            except Exception as e:
                pytest.fail(f"{path.name} does not reopen: {_first_error_line(e)}")
