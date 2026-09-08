"""
A visual surface may not present fabricated data as simulation output.

Two surfaces did. The frame-by-frame inspector generated forty-eight
frames of cells at random coordinates, resampled independently each
frame so they teleported, under a heading promising to replay
simulation history. And the Kit diapedesis manager silently substituted
fabricated leukocyte motion when it could not import the engine,
warning only to a container log that no viewer ever sees.

Both are the same failure: invented data inside a frame that says it is
real. These tests pin the fixes.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.dashboard.engine_runner import CellSnapshot, EngineRunner


# ── The inspector reads real runs, or nothing ────────────────────────

def _runner_with_frames():
    runner = EngineRunner.__new__(EngineRunner)
    runner.cell_snapshots = [
        CellSnapshot(
            time=float(t),
            cell_positions=np.array([[10.0 + t, 20.0, 30.0], [40.0, 50.0, 60.0]]),
            cell_types=["normal", "cancer"],
            cell_phases=["G1", "S"],
            cell_oxygen=[0.19, 0.11],
            cell_mhc1=[1.0, 0.3],
            cell_ids=[1, 2],
            cell_glucose=[4.5, 3.2],
            cell_atp=[900.0, 640.0],
            cell_age=[float(t), float(t)],
            immune_types=["T_cell"],
        )
        for t in range(3)
    ]
    return runner


def test_inspection_history_is_empty_without_a_run():
    """No run means nothing to inspect. It must not invent frames."""
    runner = EngineRunner.__new__(EngineRunner)
    runner.cell_snapshots = []
    assert runner.get_inspection_history() == []


def test_inspection_history_carries_real_per_cell_state():
    frames = _runner_with_frames().get_inspection_history()

    assert len(frames) == 3
    first = frames[0]
    assert first["time"] == 0.0
    assert first["n_alive"] == 2
    assert first["n_cancer"] == 1
    assert first["n_normal"] == 1
    assert first["n_immune"] == 1

    cell = first["cells"][0]
    assert cell["cell_id"] == 1
    assert cell["cell_type"] == "normal"
    assert cell["phase"] == "G1"
    assert cell["oxygen"] == pytest.approx(0.19)
    assert cell["glucose"] == pytest.approx(4.5)
    assert cell["atp"] == pytest.approx(900.0)


def test_cells_have_continuous_trajectories():
    """The fabricated version resampled every position each frame, so
    cells teleported. A real run moves them continuously, and a single
    cell must be followable across frames by its id."""
    frames = _runner_with_frames().get_inspection_history()

    track = [
        next(c for c in f["cells"] if c["cell_id"] == 1)["position"]
        for f in frames
    ]
    assert track == [(10.0, 20.0, 30.0), (11.0, 20.0, 30.0), (12.0, 20.0, 30.0)]


def test_absent_quantities_are_absent_rather_than_invented():
    """The engine records no lineage and cells have no size. The
    inspector previously supplied a parent id and a volume anyway."""
    cell = _runner_with_frames().get_inspection_history()[0]["cells"][0]
    assert "parent_id" not in cell
    assert "volume" not in cell


def test_snapshot_records_the_metabolic_state_the_engine_already_has():
    """Glucose, ATP and age live on CellState and were simply not being
    recorded, which is what pushed the inspector into fabricating."""
    for f in ("cell_glucose", "cell_atp", "cell_age", "time"):
        assert f in CellSnapshot.__dataclass_fields__


def test_inspector_page_no_longer_fabricates_a_timeline():
    """Guard the source directly: the page must not build its frames
    from a random number generator."""
    page = (
        REPO_ROOT / "cognisom" / "dashboard" / "_pages"
        / "12_3d_visualization.py"
    ).read_text()
    inspector = page[page.index("with tab_inspect:"):page.index("with tab_lineage:")]

    assert "get_inspection_history" in inspector, (
        "the inspector must source its frames from a real run"
    )

    # Comments legitimately name the old fabrication to explain the fix,
    # so only executable lines are checked.
    code = "\n".join(
        line for line in inspector.split("\n")
        if not line.strip().startswith("#")
    )
    for banned in ("RandomState", "rng.uniform", "rng.normal"):
        assert banned not in code, (
            f"the inspector tab still fabricates data via {banned}"
        )


def test_inspector_does_not_halt_the_page_when_empty():
    """st.stop() halts the whole script, which is how the Omniverse tab
    took the Export tab down with it. The empty state must not repeat
    that."""
    page = (
        REPO_ROOT / "cognisom" / "dashboard" / "_pages"
        / "12_3d_visualization.py"
    ).read_text()
    inspector = page[page.index("with tab_inspect:"):page.index("with tab_lineage:")]
    assert "st.stop()" not in inspector


def test_dead_playback_widgets_are_gone():
    """Auto-play and Speed assigned variables nothing read, so the
    controls did nothing while implying playback."""
    page = (
        REPO_ROOT / "cognisom" / "dashboard" / "_pages"
        / "12_3d_visualization.py"
    ).read_text()
    inspector = page[page.index("with tab_inspect:"):page.index("with tab_lineage:")]
    assert "playback_speed" not in inspector
    assert 'key="play_speed"' not in inspector


# ── Fabricated Kit frames announce themselves ────────────────────────

def _manager_source() -> str:
    return (
        REPO_ROOT / "cognisom" / "omniverse" / "kit_extension"
        / "cognisom.sim" / "cognisom" / "sim" / "diapedesis_manager.py"
    ).read_text()


def test_mock_frames_are_stamped():
    """Without a stamp travelling with the data, fabricated leukocyte
    motion is indistinguishable downstream from real rolling."""
    src = _manager_source()
    assert 'MOCK_FRAME_KEY = "is_mock"' in src
    assert "f[self.MOCK_FRAME_KEY] = True" in src


def test_mock_fallback_says_what_it_is_returning():
    src = _manager_source()
    fallback = src[src.index("except ImportError"):src.index("def _generate_mock_frames")]
    assert "MOCK" in fallback


def test_dashboard_surfaces_the_mock_stamp():
    """A container log line is not a disclosure. The viewer must say so."""
    page = (
        REPO_ROOT / "cognisom" / "dashboard" / "_pages" / "25_diapedesis.py"
    ).read_text()
    assert 'f.get("is_mock")' in page
    assert "fabricated" in page.lower()
