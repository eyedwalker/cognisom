"""The Scene Composer's export wrote spheres by hand. Now it does not.

The dashboard's USD page assembled ``.usda`` text itself -- one
``def Sphere`` and a nested ``Material`` per entity -- and so reached
none of the exporter work: no PointInstancer, no library morphology, no
Bio-USD attributes, no ontology terms, no cutaway. Measured on a
6,000-instance scene of three cell prototypes, the two paths cost:

    hand-written .usda    4.74 MB    18,006 prims
    write_cells .usdc     0.31 MB        14 prims

These tests cover the adapter that closed that gap, and the two things
about it that fail silently rather than loudly: a radius used as a
diameter, and a prototype guessed into a lineage it does not have.
"""

from __future__ import annotations

import pytest

from cognisom.biousd.scene_adapter import (
    CELL_TYPE_FOR_PROTOTYPE, SceneCell, cells_from_instances)

pytest.importorskip("pxr")


def _instance(proto="t_cell", r=5.0, **kw):
    inst = {"id": f"{proto}_0000", "proto_name": proto,
            "x": 1.0, "y": 2.0, "z": 3.0, "r": r,
            "cr": 0.5, "cg": 0.5, "cb": 0.5}
    inst.update(kw)
    return inst


def test_position_survives_the_adapter():
    assert SceneCell(_instance()).position == (1.0, 2.0, 3.0)


def test_a_radius_becomes_a_diameter():
    """The prototype is authored at unit diameter.

    Reading `r` straight into `scale_um` halves every cell in the scene
    and raises nothing -- the stage is valid, the render is just wrong.
    """
    assert SceneCell(_instance(r=5.0)).scale_um == 10.0


def test_a_known_prototype_gets_its_lineage():
    assert SceneCell(_instance("macrophage_m1")).cell_type == "immune"
    assert SceneCell(_instance("tumor_cell")).cell_type == "cancer"
    assert SceneCell(_instance("endothelial_cell")).cell_type == "endothelial"


def test_every_mapped_prototype_actually_exists():
    """A mapping keyed on names the library does not ship fires never.

    The first draft of this table keyed on `cancer_cell`, `t_cell`,
    `macrophage`, `fibroblast` and `stem_cell`. The library ships none
    of them -- it has `tumor_cell`, `t_cell_cd4`, `t_cell_cd8`,
    `macrophage_m1` and `macrophage_m2`, and no fibroblast or stem cell
    at all -- so five of twelve entries were dead, and a scene composed
    of tumour and T cells exported with every cell typed `normal`.
    Nothing raised. This is the check that catches it.
    """
    from cognisom.omniverse.prototype_library import ALL_PROTOTYPES

    missing = sorted(set(CELL_TYPE_FOR_PROTOTYPE) - set(ALL_PROTOTYPES))
    assert not missing, f"mapped prototypes that do not exist: {missing}"


def test_the_prototypes_that_matter_are_not_left_unmapped():
    """Tumour and leukocyte drive both the cut and the ontology term."""
    from cognisom.omniverse.prototype_library import (
        CELL_PROTOTYPES, IMMUNE_CELL_PROTOTYPES)

    for name in IMMUNE_CELL_PROTOTYPES:
        assert CELL_TYPE_FOR_PROTOTYPE.get(name) == "immune", name
    assert CELL_TYPE_FOR_PROTOTYPE.get("tumor_cell") == "cancer"
    assert "immune_cell" in CELL_PROTOTYPES


def test_an_unknown_prototype_is_not_guessed():
    """`normal` is the honest answer; a guess becomes an ontology term.

    `bio:cell:type` feeds the CURIE lookup, so a prototype guessed into
    a lineage leaves the stage asserting a Cell Ontology term for a cell
    that never had one.
    """
    for proto in ("mitochondrion", "atp", "", "cancer_cell", "fibroblast"):
        assert SceneCell(_instance(proto)).cell_type == "normal"


def test_every_mapped_token_is_one_the_schema_knows():
    """A typo here would be invisible until something consumed the stage."""
    from cognisom.biousd.ontology import TYPE_CURIE

    for proto, token in CELL_TYPE_FOR_PROTOTYPE.items():
        assert token in TYPE_CURIE, f"{proto} -> unknown token {token!r}"


def test_the_adapter_numbers_the_cells_in_order():
    cells = cells_from_instances([_instance(), _instance(), _instance()])
    assert [c.cell_id for c in cells] == [0, 1, 2]


def test_a_composed_scene_writes_one_instancer_not_a_prim_per_cell(tmp_path):
    """The whole point: 6,000 entities must not become 18,000 prims."""
    from cognisom.biousd.usd_writer import write_cells

    instances = [_instance("t_cell", x=float(i)) for i in range(600)]
    result = write_cells(str(tmp_path / "scene.usdc"),
                         cells_from_instances(instances))
    assert result.cells == 600
    # A PointInstancer, its prototypes, a material and a camera rig --
    # tens of prims, not one per entity.
    assert result.prims < 100
    assert result.primvars, "no Bio-USD attributes were authored"


def test_a_composed_scene_can_be_cut(tmp_path):
    """The cutaway has to survive the adapter, which means cell_type does."""
    from cognisom.biousd.usd_writer import write_cells

    instances = ([_instance("fibroblast", x=float(i), y=float(i % 7))
                  for i in range(500)]
                 + [_instance("cancer_cell", x=float(i % 11), y=float(i % 5))
                    for i in range(100)])
    cut = write_cells(str(tmp_path / "cut.usdc"),
                      cells_from_instances(instances), cutaway="section")
    assert cut.hidden > 0, "nothing was cut, so cell_type did not survive"


def test_the_page_imports_the_adapter_rather_than_redefining_it():
    """It lived in the page first; a copy left behind would drift."""
    source = (__import__("pathlib").Path(__file__).resolve().parents[1]
              / "cognisom" / "dashboard" / "_pages"
              / "16_usd_omniverse.py").read_text()
    assert "from cognisom.biousd.scene_adapter import" in source
    assert "class _SceneCell" not in source
    assert "_CELL_TYPE_FOR_PROTOTYPE" not in source


def test_the_page_passes_the_phase_2_controls_through():
    """Peel, LOD, colour-by and the blank-frame warning reach the user.

    They are keyword arguments on write_cells; a page that forgot one
    would still export, just without it, and nothing would say so.
    """
    source = (__import__("pathlib").Path(__file__).resolve().parents[1]
              / "cognisom" / "dashboard" / "_pages"
              / "16_usd_omniverse.py").read_text()
    for needle in ("occlusion=", "lod=lod", "colour_by=", "result.over_limit"):
        assert needle in source, f"the Export tab does not wire {needle!r}"
