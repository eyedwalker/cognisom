"""The USD cell export, which could not export a cell.

`SceneExporter.cells_to_usd` was only ever exercised on the dict fixtures
the visualization tests build.  Handed the engine's own cells it raised

    Vec3d.__init__(Vec3d, numpy.float32, numpy.float32, numpy.float32)
    did not match C++ signature

because `CellState.position` is a float32 numpy array and `Gf.Vec3d` takes
doubles.  Every path that would have shown a simulated population in USD
went through that line.

What the tests below pin, beyond the crash:

  * one `UsdGeomPointInstancer` instead of N spheres above a threshold --
    20,000 cells went from 20,015 prims and 6.4 MB to 14 prims and 2.2 MB;
  * a camera that actually contains the cells, because a stage with no
    camera renders at whatever the viewer guesses, and because
    `Gf.Matrix4d.SetLookAt` silently returns a degenerate matrix -- a
    blank frame -- when the up vector is parallel to the view;
  * colour that separates cell type from cell state, since a render whose
    hues mean nothing is worse than no render.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
import pytest

pytest.importorskip("pxr.Usd", reason="OpenUSD (usd-core) not installed")
from pxr import Gf, Usd, UsdGeom  # noqa: E402

from cognisom.visualization.exporters import (  # noqa: E402
    CELL_STATE_USD_TINTS,
    CELL_TYPE_USD_COLORS,
    SceneExporter,
    cell_display_color,
    cell_state_name,
)

USDRECORD = shutil.which("usdrecord") or "/usr/bin/usdrecord"


def _engine_cells(n=8, cell_type="normal"):
    """Real `CellState` objects -- float32 numpy positions and all."""
    from cognisom.modules.cellular_module import CellState

    rng = np.random.default_rng(0)
    return [
        CellState(
            cell_id=i,
            position=rng.uniform(0, 100, 3).astype(np.float32),
            cell_type=cell_type,
            phase="G1",
        )
        for i in range(n)
    ]


def _spheres(stage):
    return [p for p in stage.Traverse() if p.IsA(UsdGeom.Sphere)]


def _instancers(stage):
    return [p for p in stage.Traverse() if p.IsA(UsdGeom.PointInstancer)]


class TestItExportsRealCells:
    def test_engine_cell_objects_export(self, tmp_path):
        """The crash: cells straight out of CellularModule."""
        cells = _engine_cells(12)
        out = SceneExporter.cells_to_usd(cells, str(tmp_path / "cells.usda"))

        assert out, "export returned no path"
        stage = Usd.Stage.Open(out)
        assert len(_spheres(stage)) == 12

    def test_a_numpy_position_in_a_dict_exports(self, tmp_path):
        """The same float32 array, handed over in the dict form."""
        cells = [
            {"position": np.array([1, 2, 3], dtype=np.float32), "cell_type": "cancer"}
        ]
        out = SceneExporter.cells_to_usd(cells, str(tmp_path / "one.usda"))

        stage = Usd.Stage.Open(out)
        sphere = _spheres(stage)[0]
        xform = UsdGeom.Xformable(sphere).GetLocalTransformation()
        assert tuple(xform.ExtractTranslation()) == pytest.approx((1.0, 2.0, 3.0))

    def test_positions_survive_the_round_trip(self, tmp_path):
        cells = _engine_cells(5)
        out = SceneExporter.cells_to_usd(cells, str(tmp_path / "cells.usda"))

        stage = Usd.Stage.Open(out)
        got = sorted(
            tuple(
                round(v, 3)
                for v in UsdGeom.Xformable(p).GetLocalTransformation().ExtractTranslation()
            )
            for p in _spheres(stage)
        )
        want = sorted(tuple(round(float(v), 3) for v in c.position) for c in cells)
        assert got == want

    def test_an_empty_population_still_writes_a_stage(self, tmp_path):
        out = SceneExporter.cells_to_usd([], str(tmp_path / "empty.usda"))

        assert out and Usd.Stage.Open(out) is not None


class TestItScales:
    def test_below_the_threshold_one_prim_per_cell(self, tmp_path):
        out = SceneExporter.cells_to_usd(
            _engine_cells(20), str(tmp_path / "small.usda"), instancer_threshold=100
        )
        stage = Usd.Stage.Open(out)

        assert len(_spheres(stage)) == 20
        assert _instancers(stage) == []

    def test_above_the_threshold_one_instancer_for_everything(self, tmp_path):
        n = 600
        out = SceneExporter.cells_to_usd(
            _engine_cells(n), str(tmp_path / "big.usda"), instancer_threshold=256
        )
        stage = Usd.Stage.Open(out)

        instancers = _instancers(stage)
        assert len(instancers) == 1
        instancer = UsdGeom.PointInstancer(instancers[0])
        assert len(instancer.GetPositionsAttr().Get()) == n
        assert len(instancer.GetProtoIndicesAttr().Get()) == n
        assert len(instancer.GetPrototypesRel().GetTargets()) == 1
        # The whole point: the prim count stops tracking the cell count.
        assert len(list(stage.Traverse())) < 30

    def test_every_instance_carries_its_own_colour(self, tmp_path):
        """Per-instance primvar, not per-instance material: a material
        override on an instance is what breaks instance sharing."""
        cells = _engine_cells(300, "normal") + _engine_cells(300, "cancer")
        out = SceneExporter.cells_to_usd(
            cells, str(tmp_path / "colour.usda"), instancer_threshold=100
        )
        stage = Usd.Stage.Open(out)

        instancer = _instancers(stage)[0]
        primvar = UsdGeom.PrimvarsAPI(instancer).GetPrimvar("displayColor")
        assert primvar, "no displayColor on the instancer"
        assert primvar.GetInterpolation() == UsdGeom.Tokens.varying

        colours = primvar.Get()
        assert len(colours) == len(cells)
        assert tuple(colours[0]) == pytest.approx(CELL_TYPE_USD_COLORS["normal"])
        assert tuple(colours[-1]) == pytest.approx(CELL_TYPE_USD_COLORS["cancer"])


class TestItColoursByTypeAndState:
    def test_type_sets_the_hue(self):
        normal = cell_display_color({"cell_type": "normal"})
        cancer = cell_display_color({"cell_type": "cancer"})

        assert normal == pytest.approx(CELL_TYPE_USD_COLORS["normal"])
        assert cancer == pytest.approx(CELL_TYPE_USD_COLORS["cancer"])
        assert normal != cancer

    def test_an_unknown_type_does_not_borrow_a_meaningful_hue(self):
        unknown = cell_display_color({"cell_type": "chondrocyte"})

        assert unknown not in CELL_TYPE_USD_COLORS.values()
        assert unknown[0] == pytest.approx(unknown[1], abs=0.05)  # grey

    def test_state_shifts_the_hue_without_erasing_it(self):
        """Hypoxic cancer must read as hypoxic AND still as cancer."""
        alive = np.array(cell_display_color({"cell_type": "cancer"}))
        hypoxic = np.array(
            cell_display_color({"cell_type": "cancer", "oxygen": 0.03})
        )
        hypoxic_normal = np.array(
            cell_display_color({"cell_type": "normal", "oxygen": 0.03})
        )

        assert np.linalg.norm(hypoxic - alive) > 0.2
        assert np.linalg.norm(hypoxic - hypoxic_normal) > 0.2

    def test_death_erases_the_type(self):
        dead_cancer = cell_display_color({"cell_type": "cancer", "alive": False})
        dead_normal = cell_display_color({"cell_type": "normal", "alive": False})

        assert dead_cancer == dead_normal == pytest.approx(
            CELL_STATE_USD_TINTS["dead"][0]
        )

    @pytest.mark.parametrize(
        "cell, expected",
        [
            ({"oxygen": 0.21, "glucose": 5.0}, "alive"),
            ({"alive": False}, "dead"),
            ({"oxygen": 0.01}, "dying"),            # CellularModule kills < 0.02
            ({"glucose": 0.2}, "dying"),            # ... or glucose < 0.5
            ({"oxygen": 0.04}, "hypoxic"),          # VascularModule flags < 0.05
            ({"phase": "M"}, "dividing"),
            ({"activated": True}, "activated"),
            ({"oxygen": 0.01, "phase": "M"}, "dying"),   # precedence
        ],
    )
    def test_the_classifier_uses_the_engine_thresholds(self, cell, expected):
        assert cell_state_name(cell) == expected

    def test_immune_type_names_are_matched_case_insensitively(self):
        """The immune module says 'T_cell'; the colour table says 't_cell'.
        Before the lookup was folded these fell through to grey."""
        assert cell_display_color({"cell_type": "T_cell"}) == pytest.approx(
            CELL_TYPE_USD_COLORS["t_cell"]
        )
        assert cell_display_color({"cell_type": "NK_cell"}) == pytest.approx(
            CELL_TYPE_USD_COLORS["nk_cell"]
        )

    def test_the_legend_travels_with_the_stage(self, tmp_path):
        out = SceneExporter.cells_to_usd(
            _engine_cells(4), str(tmp_path / "legend.usda")
        )
        stage = Usd.Stage.Open(out)

        comment = stage.GetMetadata("comment") or ""
        assert "hue = cell type" in comment
        legend = dict(stage.GetMetadata("customLayerData"))["cognisom:colorLegend"]
        assert "type:cancer" in legend


class TestItFramesTheCells:
    @pytest.mark.parametrize("n", [20, 600])
    def test_every_cell_is_inside_the_camera_frustum(self, tmp_path, n):
        cells = _engine_cells(n)
        out = SceneExporter.cells_to_usd(
            cells, str(tmp_path / f"frame_{n}.usda"), instancer_threshold=256
        )
        stage = Usd.Stage.Open(out)

        camera = UsdGeom.Camera(stage.GetPrimAtPath("/World/Camera"))
        assert camera, "no camera authored"
        frustum = camera.GetCamera(Usd.TimeCode.Default()).frustum

        for cell in cells:
            point = Gf.Vec3d(*(float(v) for v in cell.position))
            assert frustum.Intersects(point), f"{point} outside the frame"

    def test_it_frames_a_straight_down_view_too(self, tmp_path, monkeypatch):
        """SetLookAt degenerates when up is parallel to the view, and a
        degenerate camera matrix renders an empty image rather than
        failing."""
        monkeypatch.setattr(SceneExporter, "USD_CAMERA_DIRECTION", (0.0, 0.0, 1.0))
        cells = _engine_cells(16)
        out = SceneExporter.cells_to_usd(cells, str(tmp_path / "top.usda"))

        stage = Usd.Stage.Open(out)
        camera = UsdGeom.Camera(stage.GetPrimAtPath("/World/Camera"))
        frustum = camera.GetCamera(Usd.TimeCode.Default()).frustum
        for cell in cells:
            assert frustum.Intersects(Gf.Vec3d(*(float(v) for v in cell.position)))

    def test_the_stage_lights_itself(self, tmp_path):
        out = SceneExporter.cells_to_usd(
            _engine_cells(4), str(tmp_path / "lit.usda")
        )
        stage = Usd.Stage.Open(out)

        names = {p.GetName() for p in stage.Traverse()}
        assert {"Key", "Fill", "DomeLight"} <= names


@pytest.mark.skipif(
    not os.path.exists(USDRECORD), reason="usdrecord not installed"
)
class TestItActuallyRenders:
    @pytest.mark.parametrize("n", [16, 400])
    def test_usdrecord_produces_a_non_empty_image(self, tmp_path, n):
        """An exporter that 'works' and renders a blank frame is the
        failure this guards: it is exactly what the degenerate camera
        matrix produced."""
        Image = pytest.importorskip("PIL.Image", reason="Pillow not installed")

        out = SceneExporter.cells_to_usd(
            _engine_cells(n), str(tmp_path / "render.usda"), instancer_threshold=256
        )
        png = tmp_path / "render.png"
        result = subprocess.run(
            [USDRECORD, "--imageWidth", "160", "--camera", "/World/Camera",
             out, str(png)],
            capture_output=True, text=True, timeout=180,
        )
        assert png.exists(), f"usdrecord wrote nothing: {result.stderr[-400:]}"

        pixels = np.asarray(Image.open(str(png)).convert("RGBA"))
        covered = float((pixels[..., 3] > 8).mean())
        assert covered > 0.02, f"only {covered:.3%} of the frame was drawn"
