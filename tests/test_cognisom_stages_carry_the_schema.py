"""Cognisom's own stages carry Bio-USD 0.2.0: the schema, the claims, in time.

What was wrong
--------------
Bio-USD 0.2.0 added a codeless ``BioCellAPI`` schema and UsdProfiles
claims, and the standard's minimal writer authored both. Cognisom's two
real writers -- ``usd_writer.write_cells`` (a still) and
``run_usd.write_run`` (the scrubbable timeline) -- authored neither, so
every stage the dashboard produced validated with ``schema.notApplied``
and ``profile.absent``, and a 26.08+ consumer asking
``IsCompatibleWith("biousd.profile_v1")`` got ``NoPath``. And nothing in
Cognisom registered the plugin, so even a reader in a Cognisom process
saw no ``allowedTokens`` on ``bio:cell:phase``.

What is now true
----------------
Both writers apply ``BioCellAPI`` to every cell-population instancer
after its primvars are authored and write the claims on ``/World``, with
``biousd.ontology`` claimed only when some cell asserts a term.
``import cognisom.biousd`` registers the plugin, both writers import it
at module level before any ``pxr`` call, and the dashboard imports it in
``app.py`` before navigation -- so the schema is known in the writing
process and no late-registration warning is logged. The order is the
subtle part: USD builds its schema registry once, the first time a stage
needs a prim definition, and a plugin registered after that is never
consulted; the subprocess tests here run each order on purpose.

And the exports must pass the chained check, which they did not: both
``UsdPrimvarReader`` shaders authored ``inputs:varname`` as ``token``
where Sdr declares ``string`` (``ShaderSdrCompliance.MismatchedPropertyType``,
on 26.11 and in Apple's ``usdchecker``), and five Bind sites bound a
material without applying ``MaterialBindingAPI``
(``MissingMaterialBindingAPI`` on the timeline's camera backdrop). Both
fixed; the last test here runs ``check_file`` and, where it exists,
``/usr/bin/usdchecker`` on a still and a timeline.

Every test runs the same way on usd-core 25.11 and 26.11
(``scripts/usd_matrix.sh``); the UsdProfiles reader test skips below
26.08, where ``pxr.UsdProfiles`` does not exist.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("pxr")

from pxr import Usd, UsdGeom                                              # noqa: E402

import cognisom.biousd as cognisom_biousd                                 # noqa: E402
from cognisom.biousd.standard.api_schema import applied_api_schemas      # noqa: E402
from cognisom.biousd.standard.profiles import (                           # noqa: E402
    CAPABILITY_CELLS, CAPABILITY_ONTOLOGY, PROFILE, check_claims, read_claims)
from cognisom.biousd.standard.read import read_cells                      # noqa: E402
from cognisom.biousd.standard.validate import validate                    # noqa: E402
from cognisom.biousd.standard.vocabulary import API_SCHEMA, CELL_PHASES   # noqa: E402
from cognisom.biousd.usd_writer import CELLS, write_cells                 # noqa: E402
from cognisom.dashboard.engine_runner import CellSnapshot                 # noqa: E402
from cognisom.visualization.run_usd import (                              # noqa: E402
    CELLS_PATH, IMMUNE_PATH, write_run)

REPO = pathlib.Path(__file__).resolve().parents[1]
APP = REPO / "cognisom" / "dashboard" / "app.py"

FULL_CLAIMS = {CAPABILITY_CELLS: "hard", CAPABILITY_ONTOLOGY: "enhancement"}


# ── fixtures: a hand-built run and a still ──────────────────────────


def _snapshot(t, cells, immune):
    """``cells``: (id, type, position); ``immune``: (id, type, position, polarization)."""
    return CellSnapshot(
        time=float(t),
        cell_positions=np.array([p for _, _, p in cells], dtype=float).reshape(-1, 3),
        cell_types=[ty for _, ty, _ in cells],
        cell_phases=["G1"] * len(cells),
        cell_oxygen=[0.2] * len(cells),
        cell_mhc1=[1.0] * len(cells),
        cell_ids=[i for i, _, _ in cells],
        cell_glucose=[5.0] * len(cells),
        cell_atp=[1000.0] * len(cells),
        cell_age=[1.0] * len(cells),
        immune_positions=np.array([p for _, _, p, _ in immune], dtype=float).reshape(-1, 3),
        immune_types=[ty for _, ty, _, _ in immune],
        immune_activated=[False] * len(immune),
        immune_ids=[i for i, _, _, _ in immune],
        immune_polarization=[pol for _, _, _, pol in immune])


def _two_frames(immune=True, cancer=True):
    """Two cells 30 um apart (one cancer unless ``cancer`` is False, which
    makes both ``normal`` -- a state, with no ontology term), and a T cell
    plus an M2 macrophage far away, unless ``immune`` is False."""
    cells = [(1, "normal", (0.0, 0.0, 0.0)), (2, "cancer" if cancer else "normal", (30.0, 0.0, 0.0))]
    imm = [(10, "T_cell", (200.0, 0.0, 0.0), 0.0), (12, "macrophage", (600.0, 0.0, 0.0), 1.0)] \
        if immune else []
    return [_snapshot(0.5 * k, cells, imm) for k in range(2)]


class _Cell:
    def __init__(self, i, p, name="Prostatic adenocarcinoma cell", cell_type="cancer"):
        self.position = tuple(map(float, p))
        self.cell_type = cell_type
        self.scale_um = 12.0
        self.volume = 1.0
        self.cell_id = i
        self.name = name


def _cells(n=20, **kw):
    rng = np.random.default_rng(5)
    return [_Cell(i, p, **kw) for i, p in enumerate(rng.uniform(-40, 40, (n, 3)))]


@pytest.fixture(scope="module")
def handmade_run(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("run") / "two.usdc")
    report = write_run(path, None, snapshots=_two_frames(), events=[])
    return path, report, Usd.Stage.Open(path)


def _no_schema_or_profile_findings(path):
    report = validate(path)
    assert report.ok, report.describe()
    offending = [c for c in report.codes if c.startswith(("schema.", "profile."))]
    assert offending == [], report.describe()
    return report


# ── write_run ───────────────────────────────────────────────────────


def test_a_write_run_stage_applies_the_api_to_both_populations_and_claims_the_profile(handmade_run):
    path, report, stage = handmade_run
    assert report.api_applied == [CELLS_PATH, IMMUNE_PATH]
    assert report.claims_prim == "/World"
    for prim_path in (CELLS_PATH, IMMUNE_PATH):
        prim = stage.GetPrimAtPath(prim_path)
        assert prim.IsA(UsdGeom.PointInstancer)
        assert applied_api_schemas(prim) == [API_SCHEMA], prim_path
    claims = read_claims(stage)
    assert claims.prim == "/World" and claims.applied and claims.present
    assert claims.usages == FULL_CLAIMS            # immune cells assert CL:0000738, cancer CL:0001064
    assert claims.profiles == {PROFILE: []}
    assert check_claims(stage) == []
    _no_schema_or_profile_findings(path)
    assert "BioCellAPI on 2 population(s), profile claims on /World" in report.describe()


def test_the_schema_is_live_on_a_cognisom_written_timeline(handmade_run):
    """The plugin was registered before this stage existed (the test
    process imported cognisom.biousd at collection), so the applied
    schema supplies what the file does not author: the allowed phases."""
    _, _, stage = handmade_run
    assert cognisom_biousd.schema_registered, "the plugin was not known when this process started"
    phase = stage.GetPrimAtPath(CELLS_PATH).GetAttribute("primvars:bio:cell:phase")
    assert phase and not phase.HasAuthoredMetadata("allowedTokens")
    assert list(phase.GetMetadata("allowedTokens")) == list(CELL_PHASES)


def test_write_run_claims_the_ontology_usage_only_when_a_cell_asserts_a_term(tmp_path):
    """Two normal cells and nobody else: no term anywhere, so the claims
    must not say the stage uses biousd.ontology, or the validator would
    call them inconsistent."""
    path = str(tmp_path / "normal_only.usdc")
    report = write_run(path, None, snapshots=_two_frames(immune=False, cancer=False), events=[])
    assert report.api_applied == [CELLS_PATH] and report.claims_prim == "/World"
    stage = Usd.Stage.Open(path)
    claims = read_claims(stage)
    assert claims.usages == {CAPABILITY_CELLS: "hard"}
    assert check_claims(stage) == []
    report = _no_schema_or_profile_findings(path)
    assert "primvar.curie" in report.codes      # the honest warning: no cell asserts a term


# ── write_cells ─────────────────────────────────────────────────────


def test_a_write_cells_stage_applies_the_api_and_claims_the_profile(tmp_path):
    path = str(tmp_path / "still.usdc")
    result = write_cells(path, _cells())
    assert result.api_applied and result.claims_prim == "/World"
    assert "BioCellAPI applied, profile claims on /World" in result.describe()
    stage = Usd.Stage.Open(path)
    assert applied_api_schemas(stage.GetPrimAtPath(CELLS)) == [API_SCHEMA]
    assert read_claims(stage).usages == FULL_CLAIMS and check_claims(stage) == []
    table = read_cells(path)
    assert table.api_applied and API_SCHEMA in table.applied_schemas
    # every primvar authored, every term asserted, schema and claims in
    # place: nothing left for the validator to remark on
    report = validate(path)
    assert report.ok and report.codes == [], report.describe()


def test_write_cells_claims_the_ontology_usage_only_when_a_cell_asserts_a_term(tmp_path):
    path = str(tmp_path / "invented.usdc")
    result = write_cells(path, _cells(4, name="Entirely Invented Cell", cell_type="normal"))
    assert result.unmapped == ["Entirely Invented Cell"]
    assert result.api_applied and result.claims_prim == "/World"
    stage = Usd.Stage.Open(path)
    assert read_claims(stage).usages == {CAPABILITY_CELLS: "hard"}
    assert check_claims(stage) == []
    assert _no_schema_or_profile_findings(path).codes == ["primvar.curie"]


def test_an_empty_population_claims_nothing(tmp_path):
    """No carrier, nothing to claim: a stage with no cells is not a cell
    population, and claiming biousd.cells on it would be a lie."""
    result = write_cells(str(tmp_path / "empty.usdc"), [])
    assert result.cells == 0 and not result.api_applied and result.claims_prim is None


# ── the process: registered in time, by the imports Cognisom already makes


def _run_py(code: str, *, env_extra=None) -> dict:
    """Run ``code`` in a fresh interpreter from the repository root and
    return the JSON it prints last. The environment is inherited, so a
    PYTHONPATH that shadows pxr with another usd-core carries over."""
    env = dict(os.environ)
    env["BIOUSD_NO_PLUGIN"] = ""
    env.pop("PXR_PLUGINPATH_NAME", None)
    env.update(env_extra or {})
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(REPO), env=env)
    assert run.returncode == 0, run.stdout + run.stderr
    return json.loads(run.stdout.strip().splitlines()[-1])


_CAPTURE = """
import json, logging
records = []
class _H(logging.Handler):
    def emit(self, r): records.append(r.getMessage())
logging.getLogger().addHandler(_H()); logging.getLogger().setLevel(logging.WARNING)
"""

_WRITE_AND_INSPECT = """
import numpy as np
from pxr import Usd
from cognisom.biousd.usd_writer import write_cells
from cognisom.biousd.standard.validate import validate
class _C:
    def __init__(self, i):
        self.position = (float(i) * 20.0, 0.0, 0.0); self.cell_type = "cancer"
        self.scale_um = 12.0; self.volume = 1.0; self.cell_id = i
        self.name = "Prostatic adenocarcinoma cell"
result = write_cells(PATH, [_C(i) for i in range(5)])
stage = Usd.Stage.Open(PATH)
prim = stage.GetPrimAtPath("/World/Cells")
phase = prim.GetAttribute("primvars:bio:cell:phase")
import cognisom.biousd as cb
print(json.dumps(dict(
    registered=cb.schema_registered, known=cb.schema_is_known(),
    api_applied=result.api_applied, claims=result.claims_prim,
    applied=list(prim.GetAppliedSchemas()),
    type_info=list(prim.GetPrimTypeInfo().GetAppliedAPISchemas()),
    allowed=list(phase.GetMetadata("allowedTokens") or []),
    ok=validate(PATH).ok, codes=validate(PATH).codes,
    late_warnings=[m for m in records if "after this process built" in m])))
"""


def test_importing_the_run_writer_registers_the_schema_before_it_creates_a_stage(tmp_path):
    """The dashboard's path: `from cognisom.visualization.run_usd import
    write_run` is the first Cognisom import that touches USD on the
    simulation page, and it must be enough."""
    path = str(tmp_path / "in_time.usdc")
    out = _run_py(_CAPTURE + f"PATH = {path!r}\n"
                  "from cognisom.visualization.run_usd import write_run\n" + _WRITE_AND_INSPECT)
    assert out["registered"] and out["known"]
    assert out["api_applied"] and out["claims"] == "/World"
    assert out["applied"] == out["type_info"] == [API_SCHEMA]
    assert out["allowed"] == list(CELL_PHASES)
    assert out["ok"] and out["codes"] == []
    assert out["late_warnings"] == []


def test_a_stage_opened_before_the_import_leaves_the_schema_unknown_but_the_file_right(tmp_path):
    """The trap, run on purpose: build the registry first (a typed prim
    on a stage), then import. Plug still takes the plugin (registered is
    the Plug-level fact), the registry never learns it (known is False,
    with one warning when asked), and the written stage is the same file
    -- apiSchemas kept by the prim type info, dropped by
    GetAppliedSchemas, no allowedTokens in THIS process, validator clean."""
    path = str(tmp_path / "late.usdc")
    out = _run_py(_CAPTURE + f"PATH = {path!r}\n" + """
from pxr import Usd, UsdGeom
early = Usd.Stage.CreateInMemory()
n = len(UsdGeom.Xform.Define(early, "/Early").GetPrim().GetPrimDefinition().GetPropertyNames())
""" + _WRITE_AND_INSPECT)
    assert out["registered"] is True and out["known"] is False
    assert out["api_applied"] and out["claims"] == "/World"
    assert out["applied"] == [] and out["type_info"] == [API_SCHEMA]
    assert out["allowed"] == []
    assert out["ok"] and out["codes"] == []
    assert len(out["late_warnings"]) == 1


def test_without_the_plugin_cognisom_still_writes_the_same_metadata(tmp_path):
    """BIOUSD_NO_PLUGIN=1: a writer needs no plugin, and says so by producing
    the same apiSchemas and claims with the registry never told."""
    path = str(tmp_path / "noplugin.usdc")
    out = _run_py(_CAPTURE + f"PATH = {path!r}\n" + _WRITE_AND_INSPECT,
                  env_extra={"BIOUSD_NO_PLUGIN": "1"})
    assert out["registered"] is False and out["known"] is False
    assert out["api_applied"] and out["claims"] == "/World"
    assert out["type_info"] == [API_SCHEMA] and out["applied"] == []
    assert out["ok"] and out["codes"] == [] and out["late_warnings"] == []


def test_the_dashboard_imports_the_package_before_any_page_can_open_a_stage():
    """app.py is the one process entry for every page; the import must
    come before navigation, which is where pages start running."""
    text = APP.read_text()
    where = text.index("import cognisom.biousd")
    assert where < text.index("st.navigation(")
    assert where < text.index("st.Page(")
    assert "schema registry" in text[max(0, where - 1200):where], "the ordering trap is not documented"


# ── the chained check, on the project's own exports ─────────────────


def test_biousd_check_and_usdchecker_pass_a_cognisom_still_and_timeline(handmade_run, tmp_path):
    """One report, one exit status, on what Cognisom actually writes.

    Before: on 26.11 both stages were DOES NOT CONFORM (exit 1) from the
    framework pass -- `inputs:varname` authored as token where Sdr says
    string, a material bound on the camera backdrop without
    MaterialBindingAPI -- and on 25.11 check_file raised instead of
    returning at all. Apple's usdchecker 0.25.2 failed on the same
    varname errors (its 'Coding Error' lines about ConnectableAPIBehavior
    registration are its own, printed with rc 0, and not asserted on).
    """
    from cognisom.biousd.standard.check import check_file, describe

    run_path, _, _ = handmade_run
    still_path = str(tmp_path / "still.usdc")
    write_cells(still_path, _cells())
    for path in (run_path, still_path):
        result = check_file(path)
        assert result.error == "" and result.framework_note == "", describe(result)
        assert result.status == 0, describe(result)
        assert [f.identifier for f in result.framework if f.level == "error"] == []
        assert result.report is not None and result.report.ok
        if os.path.exists("/usr/bin/usdchecker"):
            proc = subprocess.run(["/usr/bin/usdchecker", path], capture_output=True, text=True)
            assert proc.returncode == 0 and "Success!" in proc.stdout, proc.stdout + proc.stderr


# ── a 26.08+ consumer reads the claims off a Cognisom run ──────────


def test_a_profiles_aware_reader_accepts_a_cognisom_run(handmade_run):
    """Skips below OpenUSD 26.08, where pxr.UsdProfiles does not exist."""
    path, _, _ = handmade_run
    out = _run_py(f"""
import json
try:
    from pxr import UsdProfiles
except ImportError:
    print(json.dumps(dict(skip="no pxr.UsdProfiles")))
    raise SystemExit(0)
import cognisom.biousd                     # registers the capabilities with the schema
from pxr import Usd
stage = Usd.Stage.Open({path!r})
api = UsdProfiles.ClaimsAPI(stage.GetDefaultPrim())
status, results = api.IsCompatibleWith({PROFILE!r})
print(json.dumps(dict(status=str(status),
                      results=sorted((r.capability, str(r.status)) for r in results),
                      usages=dict(api.GetCapabilityUsages()),
                      populated=dict(api.PopulateCapabilityUsages()))))
""")
    if "skip" in out:
        pytest.skip(f"{out['skip']} on usd-core {'.'.join(map(str, Usd.GetVersion()))}; "
                    "UsdProfiles shipped in OpenUSD 26.08")
    assert out["status"].endswith("ValidPath"), out
    assert out["results"] == [[CAPABILITY_CELLS, "ValidPath"], [CAPABILITY_ONTOLOGY, "ValidPath"]]
    assert out["usages"] == FULL_CLAIMS
    # the applied BioCellAPI on both instancers implies biousd.cells; the
    # discovery agrees with what the writer claimed
    assert out["populated"] == FULL_CLAIMS
