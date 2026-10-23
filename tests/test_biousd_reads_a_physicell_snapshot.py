"""A PhysiCell snapshot becomes a conforming Bio-USD stage.

The outreach note promises Macklin's lab a PhysiCell -> Bio-USD exporter;
this is it, exercised on a real PhysiCell 1.14.1 output (125 cells,
MultiCellDS v2, tests/fixtures/mcds_sample/SOURCE.md). The label table, not an
assumed row order, says which matrix row is which; the cycle and phase
codes come from PhysiCell's own constants source.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("pxr")

from cognisom.biousd.standard import physicell as pc  # noqa: E402
from cognisom.biousd.standard.read import read_cells  # noqa: E402
from cognisom.biousd.standard.validate import validate  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "fixtures", "mcds_sample", "output00000001.xml")
TYPES = {0: "normal", 1: "cancer"}


@pytest.fixture(scope="module")
def snapshot():
    return pc.read_snapshot(XML)


def test_the_label_table_drives_the_read(snapshot):
    assert snapshot.software == "PhysiCell"
    assert snapshot.time_min == 60.0
    assert snapshot.n == 125 and snapshot.matrix.shape[0] == 102
    assert snapshot.labels["position"] == slice(1, 4)
    assert snapshot.units["position"] == "microns"
    # PhysiCell 1.14 writes elapsed_time_in_phase twice; both are kept, named apart
    assert "elapsed_time_in_phase" in snapshot.labels and "elapsed_time_in_phase@25" in snapshot.labels


def test_positions_and_ids_are_the_matrix_s(snapshot):
    from scipy.io import loadmat
    raw = next(v for k, v in loadmat(snapshot.cells_path).items() if not k.startswith("__"))
    cells = pc.cells_from_snapshot(snapshot, types=TYPES)
    assert [c["cell_id"] for c in cells] == list(raw[0].astype(int))
    assert np.allclose([c["position"] for c in cells], raw[1:4].T)


def test_dead_cells_are_not_alive_and_phases_follow_the_code_table(snapshot):
    report = pc.ConvertReport(path="")
    cells = pc.cells_from_snapshot(snapshot, types=TYPES, report=report)
    phase = snapshot.column("current_phase").astype(int)
    dead = snapshot.column("dead").astype(int)
    for c, p, d in zip(cells, phase, dead):
        assert c["alive"] == (d == 0 and p < pc.FIRST_DEATH_PHASE)
        if p == 10:
            assert c["phase"] == "S"
        if p >= 100:
            assert c["phase"] == "G0" and not c["alive"]
    # the sample has codes {4, 10, 14, 100}: G0G1 and 'live' are coarse
    assert set(phase.tolist()) == {4, 10, 14, 100}
    assert report.phase_coarse == int(((phase == 4) | (phase == 14)).sum())
    assert report.phase_unknown == 0
    assert report.alive == int(((dead == 0) & (phase < 100)).sum()) < report.cells


def test_volume_is_relative_to_the_type_s_median_and_scale_is_a_diameter(snapshot):
    cells = pc.cells_from_snapshot(snapshot, types=TYPES)
    vol = snapshot.column("total_volume")
    for token in TYPES.values():
        rel = [c["volume"] for c in cells if c["cell_type"] == token and c["alive"]]
        assert np.median(rel) == pytest.approx(1.0, abs=1e-6)
    d = cells[0]["scale_um"]
    assert np.pi / 6.0 * d ** 3 == pytest.approx(vol[0], rel=1e-6)


def test_unnamed_types_are_written_normal_and_reported(snapshot):
    report = pc.ConvertReport(path="")
    cells = pc.cells_from_snapshot(snapshot, types={1: "cancer"}, report=report)
    assert report.types_unmapped == [0]
    assert {c["cell_type"] for c in cells} == {"normal", "cancer"}
    assert all(c["ontology_term"] == "CL:0001064" for c in cells if c["cell_type"] == "cancer")
    assert all(c["ontology_term"] == "" for c in cells if c["cell_type"] == "normal")


def test_the_stage_validates_and_reads_back(tmp_path):
    out = str(tmp_path / "physicell.usdc")
    report = pc.convert(XML, out, types=TYPES, terms={1: "NCIT:C4863"})
    assert report.cells == 125 and os.path.getsize(out) > 0
    assert validate(out).ok
    table = read_cells(out)
    assert table.n == 125
    assert sorted(table["bio:cell:id"]) == sorted(int(i) for i in pc.read_snapshot(XML).column("ID"))
    assert set(table["bio:cell:type"]) == {"normal", "cancer"}
    assert "NCIT:C4863" in set(table["bio:cell:ontologyTerm"])
    assert sum(table["bio:cell:alive"]) == report.alive


def test_the_command_line_does_the_same(tmp_path):
    out = str(tmp_path / "cli.usda")
    proc = subprocess.run([sys.executable, "-m", "cognisom.biousd.standard.physicell", XML, out,
                           "--type", "0=normal", "--type", "1=cancer"],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "125 cells" in proc.stdout
    assert validate(out).ok


def test_a_file_that_is_not_a_snapshot_is_refused(tmp_path):
    bad = tmp_path / "x.xml"
    bad.write_text("<not_mcds/>")
    with pytest.raises(ValueError):
        pc.read_snapshot(str(bad))
