"""One resolution, stated twice, so the two copies are checked against each other.

``TissueScaleConfig`` carries ``grid_shape``, ``tissue_size_um`` AND
``resolution_um``, and the third is determined by the first two. It is kept
because it is serialised and read back by ``from_json``, which means it can
drift from them.

This repo has already paid for this exact shape of duplication once:
``scale_um`` was a second copy of a cell radius, it had no provenance, it
drove the renderer while ``radius_um`` drove the simulation, and it broke
precisely where a real measured radius had landed and nobody updated the other
copy. The failure here would be quieter and worse, because every
length-dependent quantity in the engine reads ``resolution_um`` alone -- the
diffusion length, the vessel lattice spacing, the uptake binned per voxel, and
the resolution handed to ``steady_diffusion``. A grid changed without it
leaves the solver answering a well-posed question about the wrong tissue, with
no symptom except wrong numbers.
"""

import pytest

from cognisom.core.tissue_config import TissueScaleConfig


@pytest.mark.parametrize("preset", ["small_test", "medium_test",
                                    "full_tissue"])
def test_every_preset_already_agrees(preset):
    cfg = getattr(TissueScaleConfig, preset)()
    for size, count in zip(cfg.tissue_size_um, cfg.grid_shape):
        assert size / count == pytest.approx(cfg.resolution_um)


def test_the_bare_default_agrees():
    cfg = TissueScaleConfig()
    assert cfg.tissue_size_um[0] / cfg.grid_shape[0] == pytest.approx(
        cfg.resolution_um)


def test_changing_the_grid_and_forgetting_the_resolution_is_refused():
    """The likely mistake: a smaller grid, the default 10 um left behind.

    This is what every ad-hoc config in a test or a script does, and it is
    silent without the check -- the grid would be 24 voxels across a 5 mm
    tissue, 208 um per voxel, while every length in the engine believed 10.
    """
    with pytest.raises(ValueError, match="disagrees with"):
        TissueScaleConfig(grid_shape=(24, 24, 24))


def test_non_cubic_voxels_are_refused():
    """The stencil assumes one resolution for all three axes."""
    with pytest.raises(ValueError, match="non-cubic"):
        TissueScaleConfig(grid_shape=(100, 100, 50),
                          tissue_size_um=(1000.0, 1000.0, 1000.0))


def test_a_consistent_custom_config_is_accepted():
    cfg = TissueScaleConfig(grid_shape=(24, 24, 24),
                            tissue_size_um=(240.0, 240.0, 240.0))
    assert cfg.resolution_um == pytest.approx(10.0)


def test_a_json_round_trip_survives():
    """The reason resolution_um is kept rather than derived."""
    cfg = TissueScaleConfig.small_test()
    again = TissueScaleConfig.from_json(cfg.to_json())
    assert again.resolution_um == pytest.approx(cfg.resolution_um)
    assert tuple(again.grid_shape) == tuple(cfg.grid_shape)


def test_the_check_does_not_fight_float_round_off():
    """A tolerance for JSON round-trips, and nothing wider.

    1000/3 is not representable, so a grid that does not divide its tissue
    evenly must still be accepted when the declared resolution is the nearest
    float to the quotient -- otherwise the check would reject arithmetic
    rather than mistakes.
    """
    size, count = 1000.0, 3
    cfg = TissueScaleConfig(grid_shape=(count, count, count),
                            tissue_size_um=(size, size, size),
                            resolution_um=size / count)
    assert cfg.grid_shape == (count, count, count)

    # But a genuine 1% error is still caught.
    with pytest.raises(ValueError, match="disagrees with"):
        TissueScaleConfig(grid_shape=(count, count, count),
                          tissue_size_um=(size, size, size),
                          resolution_um=size / count * 1.01)
