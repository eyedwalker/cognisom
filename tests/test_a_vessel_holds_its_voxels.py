"""A capillary is a boundary condition, not a source term.

The Krogh-Erlang inner boundary: a vessel holds the tissue beside it at the
level the blood carries, and a gradient forms between vessels because the
cells in between consume. Without it there is nothing for a gradient to form
against -- the field was initialised uniform and consumed uniformly, so
oxygen varied by 0.4% across 240 um and hypoxia was unreachable at any uptake
rate, because raising the rate lowers the whole field together.

The boundary was first applied in ``_update_partition``, which is the
explicit ``step()`` path. The engine calls ``step_implicit``. A boundary
applied only on the path nobody takes is a boundary applied nowhere: 8.6% of
masked voxels held their value and the rest drifted, which produced a profile
that fell the wrong way and read as a physical result. So these tests pin the
value through the method the engine actually calls.
"""

import numpy as np
import pytest

from cognisom.core.tissue_config import TissueScaleConfig
from cognisom.core.tissue_engine import TissueSimulationEngine


ARTERIAL = 0.21


@pytest.fixture(scope="module")
def engine():
    import io
    from contextlib import redirect_stdout

    cfg = TissueScaleConfig(
        n_cells=400, grid_shape=(24, 24, 24),
        tissue_size_um=(240., 240., 240.), n_gpus=1, dt=0.05,
        duration=0.5, seed=7, enable_ode=False, enable_diffusion=True,
        enable_cell_cycle=False, enable_mechanics=False)
    with redirect_stdout(io.StringIO()):
        eng = TissueSimulationEngine(cfg)
        eng.initialize()
        eng.run()
    return eng


def test_every_masked_voxel_holds_the_arterial_value(engine):
    """All of them, not most. 8.6% was the symptom of the wrong code path."""
    field = np.asarray(engine._diffusion._fields["oxygen"][0])
    mask = np.asarray(engine._diffusion._vessels["oxygen"][0])
    held = field[mask]
    assert held.size > 0, "no vessel voxels were masked at all"
    assert np.all(held == np.float32(ARTERIAL)), (
        f"{int((held != np.float32(ARTERIAL)).sum())} of {held.size} masked "
        "voxels drifted off the arterial value"
    )


def test_the_vasculature_is_sparse_enough_to_leave_tissue_between(engine):
    """A vessel plane on the domain face would pin a whole face.

    The lattice is offset by half a spacing for exactly that reason. If the
    masked fraction ever approached the whole grid there would be no tissue
    left to go hypoxic, and a test that only checked the pinning would still
    pass.
    """
    frac = engine._diffusion.vessel_fraction("oxygen")
    assert 0.0 < frac < 0.1, frac


def test_a_secreted_field_gets_no_vessel(engine):
    """Pinning a chemokine at a capillary would invent a source.

    Oxygen and glucose arrive in blood. CXCL12 is secreted by cells into
    tissue; holding it at the arterial level along every vessel would supply
    it from a place the vasculature does not supply it from.
    """
    vessels = engine._diffusion._vessels
    assert "oxygen" in vessels
    for name in engine._diffusion.field_names:
        if name not in ("oxygen", "glucose"):
            assert name not in vessels, name


def test_the_solver_refuses_a_saturating_sink_that_does_not_saturate(engine):
    """A km at or above the supply level is first-order in disguise.

    The whole reason to saturate is that km sits far below the concentration
    the tissue actually holds. A km of 0.21 against an arterial 0.21 would
    make C/(km+C) vary from 0.5 downward across the physiological range --
    which is the linear behaviour being replaced, silently relabelled.
    """
    with pytest.raises(ValueError, match="not below the reference"):
        engine._diffusion.set_saturating_uptake("oxygen", ARTERIAL, ARTERIAL)
    with pytest.raises(ValueError, match="unknown field"):
        engine._diffusion.set_saturating_uptake("unobtanium", 0.001, ARTERIAL)
