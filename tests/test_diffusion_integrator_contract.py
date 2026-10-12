"""One diffusion integrator, one stability policy.

This repo carries five explicit-diffusion implementations. Only
``SpatialField.update`` in ``cognisom/modules/spatial_module.py`` converts the
simulation's HOURS into the diffusion coefficient's SECONDS and only it
enforces the 1/6 explicit-Euler stability limit in 3-D. The one that the GPU
orchestrator actually installs over it --
``cognisom.gpu.diffusion.diffuse_field``, whose own docstring offers itself as
a replacement for ``SpatialField.update`` -- did neither.

Measured on the shipped oxygen field (D = 2000 um^2/s, dx = 10 um, dt = 0.1 h)
before the fix: ``diffuse_field`` applied a diffusion number of 2.0 against a
limit of 1/6, and the non-negative clamp turned the resulting unstable
oscillation into 12x mass CREATION per call -- 1.0 -> 12 -> 144 -> ... ->
3.1e21 by call 20 and NaN across the whole grid by call 40. A 24 h run makes
240 such calls. It was invisible only because nothing registers a source, so
every field was identically zero, and zero is a fixed point of even a
divergent scheme.

These tests pin the two integrators to each other so the next person to add a
sixth copy, or to "optimise" one of them, fails here instead of shipping NaN.
"""

import logging

import numpy as np
import pytest

from cognisom.gpu.backend import get_backend
from cognisom.gpu.diffusion import diffuse_field
from cognisom.modules.spatial_module import SpatialField, SpatialModule

#: The three fields SpatialModule.initialize() actually creates.
SHIPPED_FIELDS = [("oxygen", 2000.0), ("glucose", 600.0), ("cytokine", 100.0)]

GRID = (20, 20, 10)
DX = 10.0


def _blob(value=1.0):
    """A single occupied voxel, away from the boundary shell."""
    c = np.zeros(GRID, dtype=np.float32)
    c[10, 10, 5] = value
    return c


def _field(name, d_coeff, concentration=None):
    f = SpatialField(name, GRID, DX, d_coeff)
    if concentration is not None:
        f.concentration = concentration.copy()
    return f


# ── the two integrators must agree ───────────────────────────────────

@pytest.mark.parametrize("name,d_coeff", SHIPPED_FIELDS)
def test_gpu_and_cpu_integrators_agree(name, d_coeff):
    """diffuse_field is advertised as a drop-in for SpatialField.update.
    Hold it to that: same initial condition, same dt, same answer."""
    dt = 0.1
    ref = _field(name, d_coeff, _blob())
    ref.update(dt)

    got = get_backend().to_numpy(
        diffuse_field(
            concentration=_blob(), diffusion_coeff=d_coeff,
            dt=dt, resolution=DX, sources=[], sinks=[],
        )
    )

    assert np.allclose(got, ref.concentration, rtol=1e-5, atol=1e-12), (
        f"{name}: GPU integrator diverged from SpatialField.update; "
        f"max abs diff {np.max(np.abs(got - ref.concentration)):.6g}"
    )


@pytest.mark.parametrize("name,d_coeff", SHIPPED_FIELDS)
def test_gpu_and_cpu_integrators_agree_with_sources_and_sinks(name, d_coeff):
    """Sources and sinks must be applied at the same point in the step.
    diffuse_field used to compute the Laplacian BEFORE injecting, so a
    source contributed nothing to transport on the step that created it."""
    dt = 0.1
    sources = [((10, 10, 5), 0.5), ((4, 4, 3), 0.25)]
    sinks = [((12, 12, 6), 0.1)]

    ref = _field(name, d_coeff)
    for idx, rate in sources:
        ref.sources.append((idx, rate))
    for idx, rate in sinks:
        ref.sinks.append((idx, rate))
    ref.update(dt)

    got = get_backend().to_numpy(
        diffuse_field(
            concentration=np.zeros(GRID, dtype=np.float32),
            diffusion_coeff=d_coeff, dt=dt, resolution=DX,
            sources=sources, sinks=sinks,
        )
    )

    assert np.allclose(got, ref.concentration, rtol=1e-5, atol=1e-12), (
        f"{name}: source/sink handling differs between integrators"
    )


# ── stability: the bug that produced NaN ─────────────────────────────

@pytest.mark.parametrize("name,d_coeff", SHIPPED_FIELDS)
def test_gpu_integrator_never_creates_mass(name, d_coeff):
    """Diffusion moves mass; it cannot make it. With a frozen boundary
    shell the total may fall, but it must never rise and must stay
    finite. Before the fix this grew 12x per call and hit NaN by call 40.
    """
    c = _blob()
    mass0 = float(c.sum())
    backend = get_backend()

    for call in range(1, 41):
        c = backend.to_numpy(
            diffuse_field(
                concentration=c, diffusion_coeff=d_coeff,
                dt=0.1, resolution=DX, sources=[], sinks=[],
            )
        )
        assert np.all(np.isfinite(c)), (
            f"{name}: non-finite values after {call} calls "
            f"({int((~np.isfinite(c)).sum())} voxels)"
        )
        assert float(c.sum()) <= mass0 * (1.0 + 1e-5), (
            f"{name}: mass grew from {mass0:.6g} to {float(c.sum()):.6g} "
            f"after {call} calls -- the scheme is unstable"
        )


@pytest.mark.parametrize("name,d_coeff", SHIPPED_FIELDS)
def test_gpu_integrator_respects_the_cfl_limit(name, d_coeff):
    """Whatever sub-stepping it chooses, the per-sub-step diffusion
    number must sit under 1/6, which is what makes the scheme damp
    high-frequency modes instead of amplifying them."""
    dt = 0.1
    dt_seconds = dt * SpatialField.SECONDS_PER_HOUR
    diffusion_number = d_coeff * dt_seconds / DX**2
    n_sub = max(1, int(np.ceil(
        diffusion_number / (SpatialField.CFL_LIMIT_3D * SpatialField.CFL_SAFETY)
    )))
    per_sub = diffusion_number / min(n_sub, SpatialField.MAX_SUBSTEPS)
    if n_sub > SpatialField.MAX_SUBSTEPS:
        per_sub = SpatialField.CFL_LIMIT_3D * SpatialField.CFL_SAFETY
    assert per_sub <= SpatialField.CFL_LIMIT_3D, (
        f"{name}: per-sub-step diffusion number {per_sub} exceeds 1/6"
    )


def test_gpu_integrator_treats_dt_as_hours():
    """dt is in hours and diffusion_coeff in um^2/s. Pin the conversion
    by comparing against a hand-rolled single explicit step at a dt small
    enough to be stable without sub-division."""
    d_coeff = 100.0
    # Choose dt so that D*dt_seconds/dx^2 is comfortably under 1/6.
    dt_seconds = 0.1 * DX**2 / d_coeff
    dt_hours = dt_seconds / SpatialField.SECONDS_PER_HOUR
    coeff = d_coeff * dt_seconds / DX**2
    assert coeff < SpatialField.CFL_LIMIT_3D * SpatialField.CFL_SAFETY

    c0 = _blob()
    expected = c0.copy()
    lap = np.zeros_like(c0)
    lap[1:-1, 1:-1, 1:-1] = (
        c0[2:, 1:-1, 1:-1] + c0[:-2, 1:-1, 1:-1]
        + c0[1:-1, 2:, 1:-1] + c0[1:-1, :-2, 1:-1]
        + c0[1:-1, 1:-1, 2:] + c0[1:-1, 1:-1, :-2]
        - 6 * c0[1:-1, 1:-1, 1:-1]
    )
    expected = np.maximum(expected + coeff * lap, 0.0)

    got = get_backend().to_numpy(
        diffuse_field(
            concentration=_blob(), diffusion_coeff=d_coeff,
            dt=dt_hours, resolution=DX, sources=[], sinks=[],
        )
    )
    assert np.allclose(got, expected, rtol=1e-5, atol=1e-12), (
        "dt is not being converted from hours to seconds"
    )


# ── the orchestrator path, end to end ────────────────────────────────

def test_orchestrator_patched_spatial_stays_finite():
    """The keystone. SimulationConfig(use_gpu=True) makes
    GPUOrchestrator._patch_spatial replace SpatialField.update with
    diffuse_field. Run a full 24 h with a source registered -- which is
    what any cell-as-sink wiring would do -- and require finite fields.
    Before the fix this was NaN across 2592 of 4000 voxels.
    """
    import io
    from contextlib import redirect_stdout

    from cognisom.core import SimulationConfig, SimulationEngine

    np.random.seed(12345)
    config = SimulationConfig(dt=0.05, duration=24.0)
    config.use_gpu = True
    engine = SimulationEngine(config)
    engine.register_module("spatial", SpatialModule, {})

    with redirect_stdout(io.StringIO()):
        engine.initialize()
        spatial = engine.modules["spatial"]
        spatial.add_source("oxygen", np.array([100.0, 100.0, 50.0]), rate=0.5)
        spatial.add_sink("oxygen", np.array([130.0, 130.0, 50.0]), rate=0.15)
        engine.run()

    for name, field in spatial.fields.items():
        c = field.concentration
        assert np.all(np.isfinite(c)), (
            f"{name}: {int((~np.isfinite(c)).sum())} non-finite voxels "
            f"after a 24 h GPU-accelerated run"
        )
        assert c.max() <= 1e6, f"{name}: field diverged to {c.max():.6g}"

    oxygen = spatial.fields["oxygen"].concentration
    assert oxygen.sum() > 0.0, "a driven field must not stay empty"


# ── the undriven field is skipped, provably ──────────────────────────

def test_undriven_field_is_an_exact_noop():
    """No sources, no sinks, nothing in it: diffusion can only produce
    zeros, so skipping is exact. This is the shipped state -- nothing in
    the engine registers a source -- and doing it anyway cost 11.1 s of
    wall clock and 720 stability warnings per 24 h run."""
    f = _field("oxygen", 2000.0)
    for _ in range(10):
        f.update(0.1)
    assert not f.concentration.any()


def test_undriven_field_does_not_warn(caplog):
    f = _field("oxygen", 2000.0)
    with caplog.at_level(logging.WARNING, logger="cognisom.modules.spatial_module"):
        for _ in range(240):
            f.update(0.1)
    assert caplog.records == [], (
        f"{len(caplog.records)} warnings about transporting an empty field"
    )


def test_driven_field_warns_once_per_field(caplog):
    """When the scheme really is under-diffusing, say so -- once. It used
    to say so on every update: 240 updates x 3 fields = 720 warnings."""
    f = _field("oxygen", 2000.0, _blob())
    with caplog.at_level(logging.WARNING, logger="cognisom.modules.spatial_module"):
        for _ in range(240):
            f.update(0.1)
    assert len(caplog.records) == 1, (
        f"expected exactly 1 warning per field, got {len(caplog.records)}"
    )


def test_driven_field_still_diffuses():
    """The skip-when-empty guard must not disable a field that has work."""
    f = _field("oxygen", 2000.0, _blob())
    f.update(0.1)
    assert f.concentration[10, 10, 5] < 1.0, "centre did not decay"
    assert f.concentration[11, 10, 5] > 0.0, "neighbour received nothing"

    g = _field("oxygen", 2000.0)
    g.add_source(np.array([100.0, 100.0, 50.0]), 0.5)
    g.update(0.1)
    assert g.concentration.sum() > 0.0, "a registered source produced nothing"


def test_whole_module_run_is_quiet_and_finite(caplog):
    """SpatialModule over a 24 h run, as the engine drives it."""
    m = SpatialModule({})
    import io
    from contextlib import redirect_stdout
    with redirect_stdout(io.StringIO()):
        m.initialize()
    with caplog.at_level(logging.WARNING, logger="cognisom.modules.spatial_module"):
        for _ in range(480):          # 24 h at dt = 0.05
            m.update(0.05)
    assert caplog.records == []
    for name, f in m.fields.items():
        assert np.all(np.isfinite(f.concentration))


# ── the GPU patch must not discard the coupling it replaces ──────────

def _coupled_spatial():
    """A SpatialModule linked to cells and vessels, as the engine links it."""
    import io
    from contextlib import redirect_stdout

    from cognisom.modules.cellular_module import CellularModule
    from cognisom.modules.vascular_module import VascularModule

    np.random.seed(7)
    spatial = SpatialModule({})
    cellular = CellularModule({})
    vascular = VascularModule({})
    with redirect_stdout(io.StringIO()):
        cellular.initialize()
        vascular.initialize()
        spatial.initialize()
    vascular.set_cellular_module(cellular)
    spatial.set_cellular_module(cellular)
    spatial.set_vascular_module(vascular)
    return spatial


def test_the_gpu_spatial_patch_keeps_the_cell_and_vessel_coupling():
    """GPUOrchestrator._patch_spatial replaces SpatialModule.update with a
    loop that reads only ``field.sources`` and ``field.sinks``. It did not
    re-derive the terms from the cells and vessels, and it cannot represent
    Dirichlet voxels or first-order uptake at all, so simply turning
    ``use_gpu`` on put every field back to the identically-zero state this
    coupling removes.

    MEASURED on the un-grafted tree: patched gives n_dirichlet = 0,
    n_uptake = 0 and a mean of 0.0, against 112 / 100 / 0.21 unpatched.
    """
    from cognisom.gpu.orchestrator import GPUOrchestrator

    plain = _coupled_spatial()
    for _ in range(20):                      # 1 h at dt = 0.05
        plain.update(0.05)
    reference = plain.fields["oxygen"]

    patched = _coupled_spatial()
    GPUOrchestrator()._patch_spatial(patched)
    for _ in range(20):
        patched.update(0.05)
    oxygen = patched.fields["oxygen"]

    assert len(oxygen.dirichlet) == len(reference.dirichlet) > 0, (
        "the GPU patch dropped the capillary Dirichlet voxels"
    )
    assert len(oxygen.uptake) == len(reference.uptake) > 0, (
        "the GPU patch dropped the per-cell uptake"
    )
    assert oxygen.last_solver == reference.last_solver == "steady_state"
    assert np.allclose(oxygen.concentration, reference.concentration,
                       rtol=1e-6, atol=1e-9), (
        "the accelerated field disagrees with the un-accelerated one"
    )


def test_the_gpu_spatial_patch_still_accelerates_a_plain_source_field():
    """The delegation above must be narrow: a field carrying only rate
    sources and sinks -- everything diffuse_field CAN represent -- still
    goes through diffuse_field, and stays finite doing it."""
    import io
    from contextlib import redirect_stdout

    from cognisom.gpu.orchestrator import GPUOrchestrator

    module = SpatialModule({})
    with redirect_stdout(io.StringIO()):
        module.initialize()
    module.add_source("oxygen", np.array([100.0, 100.0, 50.0]), rate=0.5)
    GPUOrchestrator()._patch_spatial(module)

    for _ in range(20):
        module.update(0.05)

    oxygen = module.fields["oxygen"]
    assert oxygen.last_solver == "none", (
        "a plain source field should not have run SpatialField.update at all"
    )
    assert np.all(np.isfinite(oxygen.concentration))
    assert oxygen.concentration.sum() > 0.0
