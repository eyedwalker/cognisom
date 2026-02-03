"""
Tests for the visualization module.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from cognisom.visualization.cell_renderer import (
    CellPopulationRenderer, CELL_TYPE_COLORS, PHASE_COLORS,
)
from cognisom.visualization.field_renderer import SpatialFieldRenderer
from cognisom.visualization.network_renderer import InteractionNetworkRenderer
from cognisom.visualization.exporters import SceneExporter


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def demo_cells():
    return CellPopulationRenderer.generate_demo_cells(
        n_normal=20, n_cancer=10, n_immune=5,
    )


@pytest.fixture
def demo_fields():
    return SpatialFieldRenderer.generate_demo_fields(grid_size=(10, 10, 6))


@pytest.fixture
def demo_interactions(demo_cells):
    renderer = InteractionNetworkRenderer()
    return renderer.generate_demo_interactions(demo_cells, n_interactions=15)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ── CellPopulationRenderer ───────────────────────────────────────

class TestCellPopulationRenderer:
    def test_render_demo(self, demo_cells):
        renderer = CellPopulationRenderer()
        fig = renderer.render(demo_cells, color_by="type")
        assert fig is not None
        assert len(fig.data) > 0

    def test_render_empty(self):
        renderer = CellPopulationRenderer()
        fig = renderer.render([])
        assert fig is not None

    def test_color_by_options(self, demo_cells):
        renderer = CellPopulationRenderer()
        for mode in ["type", "phase", "oxygen", "glucose", "atp", "mhc1"]:
            fig = renderer.render(demo_cells, color_by=mode)
            assert fig is not None

    def test_size_by_options(self, demo_cells):
        renderer = CellPopulationRenderer()
        for mode in ["fixed", "volume", "atp"]:
            fig = renderer.render(demo_cells, size_by=mode)
            assert fig is not None

    def test_highlight_mutations(self, demo_cells):
        renderer = CellPopulationRenderer()
        fig = renderer.render(demo_cells, highlight_mutations=True)
        assert fig is not None

    def test_generate_demo_cells(self):
        cells = CellPopulationRenderer.generate_demo_cells(
            n_normal=10, n_cancer=5, n_immune=3,
        )
        assert len(cells) == 18
        types = {c["cell_type"] for c in cells}
        assert "normal" in types
        assert "cancer" in types

    def test_cell_type_colors_complete(self):
        assert "normal" in CELL_TYPE_COLORS
        assert "cancer" in CELL_TYPE_COLORS
        assert "t_cell" in CELL_TYPE_COLORS

    def test_phase_colors_complete(self):
        for phase in ["G1", "S", "G2", "M", "G0"]:
            assert phase in PHASE_COLORS


# ── SpatialFieldRenderer ────────────────────────────────────────

class TestSpatialFieldRenderer:
    def test_render_isosurface(self, demo_fields):
        renderer = SpatialFieldRenderer()
        fig = renderer.render_isosurface(demo_fields["O2"], name="O2")
        assert fig is not None
        assert len(fig.data) > 0

    def test_render_slices(self, demo_fields):
        renderer = SpatialFieldRenderer()
        fig = renderer.render_slices(demo_fields["Glucose"], name="Glucose")
        assert fig is not None

    def test_render_slices_all_axes(self, demo_fields):
        renderer = SpatialFieldRenderer()
        for axis in ["x", "y", "z"]:
            fig = renderer.render_slices(
                demo_fields["O2"], slice_axis=axis, slice_indices=[2],
            )
            assert fig is not None

    def test_render_multi_field(self, demo_fields):
        renderer = SpatialFieldRenderer()
        fig = renderer.render_multi_field(demo_fields)
        assert fig is not None
        assert len(fig.data) == len(demo_fields)

    def test_generate_demo_fields(self):
        fields = SpatialFieldRenderer.generate_demo_fields(grid_size=(8, 8, 4))
        assert "O2" in fields
        assert "Glucose" in fields
        assert "Cytokines" in fields
        assert "VEGF" in fields
        assert fields["O2"].shape == (8, 8, 4)

    def test_field_values_realistic(self, demo_fields):
        o2 = demo_fields["O2"]
        assert o2.min() >= 0
        assert o2.max() <= 0.22  # atmospheric O2 ~ 0.21
        glucose = demo_fields["Glucose"]
        assert glucose.min() >= 0


# ── InteractionNetworkRenderer ──────────────────────────────────

class TestInteractionNetworkRenderer:
    def test_render_spatial_network(self, demo_cells, demo_interactions):
        renderer = InteractionNetworkRenderer()
        fig = renderer.render_network(demo_cells, demo_interactions, layout="spatial")
        assert fig is not None
        assert len(fig.data) > 0

    def test_render_force_network(self, demo_cells, demo_interactions):
        renderer = InteractionNetworkRenderer()
        fig = renderer.render_network(demo_cells, demo_interactions, layout="force")
        assert fig is not None

    def test_render_empty(self):
        renderer = InteractionNetworkRenderer()
        fig = renderer.render_network([], [])
        assert fig is not None

    def test_render_contact_map(self, demo_cells):
        renderer = InteractionNetworkRenderer()
        fig = renderer.render_contact_map(demo_cells, distance_threshold=30.0)
        assert fig is not None

    def test_generate_demo_interactions(self, demo_cells):
        renderer = InteractionNetworkRenderer()
        interactions = renderer.generate_demo_interactions(demo_cells, n_interactions=10)
        assert len(interactions) <= 10
        for inter in interactions:
            assert "source" in inter
            assert "target" in inter
            assert "type" in inter
            assert "strength" in inter

    def test_edge_colors_defined(self):
        for itype in ["paracrine", "juxtacrine", "killing", "exosome", "chemotaxis"]:
            assert itype in InteractionNetworkRenderer.EDGE_COLORS


# ── SceneExporter ───────────────────────────────────────────────

class TestSceneExporter:
    def test_cells_to_pdb(self, demo_cells, tmp_dir):
        path = str(tmp_dir / "cells.pdb")
        result = SceneExporter.cells_to_pdb(demo_cells, path)
        assert Path(result).exists()
        content = Path(result).read_text()
        assert "ATOM" in content
        assert "END" in content
        # Should have one ATOM line per cell + REMARK + END
        atom_lines = [l for l in content.split("\n") if l.startswith("ATOM")]
        assert len(atom_lines) == len(demo_cells)

    def test_pdb_cell_types_mapped(self, tmp_dir):
        cells = [
            {"cell_id": 0, "position": [1, 2, 3], "cell_type": "cancer"},
            {"cell_id": 1, "position": [4, 5, 6], "cell_type": "normal"},
        ]
        path = str(tmp_dir / "types.pdb")
        SceneExporter.cells_to_pdb(cells, path)
        content = Path(path).read_text()
        lines = [l for l in content.split("\n") if l.startswith("ATOM")]
        # cancer -> element O, residue CAN
        assert "CAN" in lines[0]
        # normal -> element C, residue NRM
        assert "NRM" in lines[1]

    def test_cells_to_gltf(self, demo_cells, tmp_dir):
        path = str(tmp_dir / "cells.gltf")
        result = SceneExporter.cells_to_gltf(demo_cells, path)
        assert Path(result).exists()
        content = json.loads(Path(result).read_text())
        assert content["asset"]["version"] == "2.0"
        assert content["accessors"][0]["count"] == len(demo_cells)

    def test_gltf_empty(self, tmp_dir):
        result = SceneExporter.cells_to_gltf([], str(tmp_dir / "empty.gltf"))
        assert result == ""

    def test_field_to_vtk(self, demo_fields, tmp_dir):
        field = demo_fields["O2"]
        path = str(tmp_dir / "o2.vtk")
        result = SceneExporter.field_to_vtk(field, path, field_name="oxygen")
        assert Path(result).exists()
        content = Path(result).read_text()
        assert "STRUCTURED_POINTS" in content
        assert "SCALARS oxygen" in content
        nx, ny, nz = field.shape
        assert f"DIMENSIONS {nx} {ny} {nz}" in content

    def test_fields_to_vtk_multi(self, demo_fields, tmp_dir):
        path = str(tmp_dir / "multi.vtk")
        result = SceneExporter.fields_to_vtk(demo_fields, path)
        assert Path(result).exists()
        content = Path(result).read_text()
        for name in demo_fields:
            assert f"SCALARS {name}" in content

    def test_cells_to_csv(self, demo_cells, tmp_dir):
        path = str(tmp_dir / "cells.csv")
        result = SceneExporter.cells_to_csv(demo_cells, path)
        assert Path(result).exists()
        lines = Path(result).read_text().strip().split("\n")
        assert lines[0].startswith("cell_id,")
        assert len(lines) == len(demo_cells) + 1  # header + data

    def test_csv_empty(self, tmp_dir):
        result = SceneExporter.cells_to_csv([], str(tmp_dir / "empty.csv"))
        assert result == ""

    def test_scene_to_json(self, demo_cells, demo_fields, demo_interactions, tmp_dir):
        path = str(tmp_dir / "scene.json")
        result = SceneExporter.scene_to_json(
            cells=demo_cells,
            fields=demo_fields,
            interactions=demo_interactions,
            output_path=path,
        )
        assert Path(result).exists()
        data = json.loads(Path(result).read_text())
        assert data["version"] == "1.0"
        assert len(data["cells"]) == len(demo_cells)
        assert "fields" in data
        assert "interactions" in data
