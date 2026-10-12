"""
GPU-Accelerated 3D Diffusion Solver
====================================

Computes the 3D Laplacian stencil and explicit Euler diffusion
update on GPU via CuPy, with automatic NumPy fallback.

The Laplacian kernel is the single most compute-intensive operation
in the spatial module, operating on 200x200x100 = 4M voxels per
field, three fields per timestep.

On GPU this uses a CuPy RawKernel with shared memory tiling for
the 7-point stencil. On CPU it falls back to NumPy slicing.
"""

import logging
from typing import Optional, Tuple

import numpy as np

from .backend import get_backend

logger = logging.getLogger(__name__)

# ── CuPy CUDA kernel (compiled once, cached) ────────────────────

_LAPLACIAN_KERNEL_SRC = r"""
extern "C" __global__
void laplacian_3d(
    const float* __restrict__ C,
    float* __restrict__ L,
    int nx, int ny, int nz
) {
    // Global thread indices
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;  // skip boundary
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i >= nx - 1 || j >= ny - 1 || k >= nz - 1) return;

    int idx = i * ny * nz + j * nz + k;

    float center = C[idx];
    float lap = (
        C[(i+1)*ny*nz + j*nz + k] + C[(i-1)*ny*nz + j*nz + k] +
        C[i*ny*nz + (j+1)*nz + k] + C[i*ny*nz + (j-1)*nz + k] +
        C[i*ny*nz + j*nz + (k+1)] + C[i*ny*nz + j*nz + (k-1)] -
        6.0f * center
    );

    L[idx] = lap;
}
"""

_cuda_laplacian_kernel = None


def _get_cuda_kernel():
    """Compile and cache the CUDA Laplacian kernel."""
    global _cuda_laplacian_kernel
    if _cuda_laplacian_kernel is None:
        import cupy as cp
        _cuda_laplacian_kernel = cp.RawKernel(_LAPLACIAN_KERNEL_SRC, "laplacian_3d")
    return _cuda_laplacian_kernel


# ── Public API ───────────────────────────────────────────────────

def compute_laplacian(concentration: np.ndarray) -> np.ndarray:
    """Compute the 3D Laplacian of a concentration field.

    Uses CUDA kernel on GPU, NumPy slicing on CPU.

    Args:
        concentration: 3D array (nx, ny, nz), float32.

    Returns:
        Laplacian array (same shape), zero at boundaries.
    """
    backend = get_backend()
    xp = backend.xp

    # Ensure float32 and on correct device
    C = backend.to_device(np.ascontiguousarray(concentration, dtype=np.float32))

    if backend.has_gpu:
        return _laplacian_gpu(C, xp)
    else:
        return _laplacian_cpu(C)


#: Set once the explicit scheme has reported that it is under-diffusing.
#: SpatialField rate-limits per field instance; this is a free function
#: with no instance to hang a flag on, so it rate-limits per process.
#: Either way the point gets made once, not 240 times a run.
_under_diffused_warned = False


def _warn_under_diffused(diffusion_number: float, n_sub_needed: int,
                         max_stable: float) -> None:
    """Report an under-diffused interval at most once per process."""
    global _under_diffused_warned
    if _under_diffused_warned:
        return
    _under_diffused_warned = True

    from cognisom.modules.spatial_module import SpatialField

    logger.warning(
        "diffuse_field: D*dt/dx^2 = %.1f needs %d explicit sub-steps to stay "
        "stable, above the cap of %d. Running %d stable sub-steps instead, so "
        "the field is UNDER-DIFFUSED over this interval (about %.1f%% of the "
        "transport applied). At this diffusion number the field is "
        "effectively at steady state within one step, so an implicit or "
        "steady-state solve is the right method here, not explicit Euler. "
        "Logged once per process.",
        diffusion_number, n_sub_needed, SpatialField.MAX_SUBSTEPS,
        SpatialField.MAX_SUBSTEPS,
        100.0 * SpatialField.MAX_SUBSTEPS * max_stable / diffusion_number,
    )


def diffuse_field(
    concentration: np.ndarray,
    diffusion_coeff: float,
    dt: float,
    resolution: float,
    sources: Optional[list] = None,
    sinks: Optional[list] = None,
) -> np.ndarray:
    """Full diffusion step: sources/sinks + stability-limited Euler update.

    A GPU-accelerated stand-in for SpatialField.update(), and installed as
    one by GPUOrchestrator._patch_spatial. It therefore has to match that
    method's TIME semantics, not merely its signature: dt in hours against
    a coefficient in um^2/s, and the 1/6 explicit-Euler stability limit in
    3-D enforced by sub-division. It did neither, which made enabling GPU
    acceleration silently replace a stable integrator with a divergent one.

    Args:
        concentration: 3D float32 array.
        diffusion_coeff: Diffusion coefficient (um^2/s).
        dt: Time step (hours).
        resolution: Voxel size (um).
        sources: List of (index_tuple, rate) pairs.
        sinks: List of (index_tuple, rate) pairs.

    Returns:
        Updated concentration array (same device as input).
    """
    backend = get_backend()
    xp = backend.xp

    C = backend.to_device(np.array(concentration, dtype=np.float32, copy=True))

    # The time-integration policy -- the hours-to-seconds conversion and
    # the explicit-Euler stability limit -- belongs to SpatialField, which
    # is the one implementation in this repo that gets it right. Reading
    # the constants from there instead of restating them is the whole
    # point: this function is installed OVER SpatialField.update by
    # GPUOrchestrator._patch_spatial, so the two drifting apart is
    # precisely the failure that produced NaN fields. Imported inside the
    # function because cognisom.modules imports the engine core and the
    # GPU orchestrator imports this module, so a module-level import
    # would close a cycle.
    from cognisom.modules.spatial_module import SpatialField

    # A field with no sources, no sinks and no content is an exact fixed
    # point of the diffusion operator: skipping it cannot change the
    # answer. It is also the shipped state, since nothing in the engine
    # registers a source.
    if not sources and not sinks and not bool(xp.any(C)):
        return C

    # 1. Sources and sinks act over the whole interval, once, and BEFORE
    #    transport -- the same order as SpatialField.update. This used to
    #    compute the Laplacian first, so a source contributed nothing to
    #    transport on the step that created it.
    if sources:
        for idx, rate in sources:
            if _is_valid(idx, C.shape):
                C[idx] += rate * dt

    # 2. Sinks
    if sinks:
        for idx, rate in sinks:
            if _is_valid(idx, C.shape):
                removal = min(float(rate * dt), float(C[idx]))
                C[idx] -= removal

    # 3. Transport. dt is in HOURS (the simulation's time unit) and
    #    diffusion_coeff in um^2/s, so the conversion is required. Without
    #    it this advanced dt SECONDS of diffusion and understated the
    #    transported amount 3600-fold.
    dt_seconds = dt * SpatialField.SECONDS_PER_HOUR
    if diffusion_coeff > 0.0 and dt_seconds > 0.0:
        diffusion_number = diffusion_coeff * dt_seconds / (resolution ** 2)
        max_stable = SpatialField.CFL_LIMIT_3D * SpatialField.CFL_SAFETY

        n_sub = max(1, int(np.ceil(diffusion_number / max_stable)))
        coeff = diffusion_number / n_sub

        if n_sub > SpatialField.MAX_SUBSTEPS:
            # Degrade exactly as SpatialField does: run the cap's worth of
            # STABLE sub-steps and under-diffuse, rather than spreading an
            # over-limit coefficient across capped sub-steps, which leaves
            # every one of them unstable. The old single step did the
            # latter in its purest form -- one step at a diffusion number
            # of 2.0 against a limit of 1/6 -- and the clamp below turned
            # the resulting oscillation into 12x mass creation per call.
            _warn_under_diffused(diffusion_number, n_sub, max_stable)
            n_sub = SpatialField.MAX_SUBSTEPS
            coeff = max_stable

        for _ in range(n_sub):
            if backend.has_gpu:
                lap = _laplacian_gpu(C, xp)
            else:
                lap = _laplacian_cpu(C)
            C += coeff * lap

    # 4. Non-negative clamp
    C = xp.maximum(C, 0.0)

    return C


def diffuse_fields_batch(
    fields: dict,
    dt: float,
    resolution: float,
) -> dict:
    """Diffuse multiple fields in a batch (all on GPU if available).

    Args:
        fields: Dict of name -> (concentration, diffusion_coeff, sources, sinks).
        dt: Time step.
        resolution: Voxel size.

    Returns:
        Dict of name -> updated concentration.
    """
    results = {}
    for name, (conc, diff_coeff, sources, sinks) in fields.items():
        results[name] = diffuse_field(
            conc, diff_coeff, dt, resolution, sources, sinks,
        )
    return results


# ── Internal implementations ────────────────────────────────────

def _laplacian_gpu(C, xp):
    """GPU Laplacian via CUDA kernel."""
    nx, ny, nz = C.shape
    L = xp.zeros_like(C)

    kernel = _get_cuda_kernel()

    # Block/grid dimensions
    block = (8, 8, 8)
    grid = (
        (nx - 2 + block[0] - 1) // block[0],
        (ny - 2 + block[1] - 1) // block[1],
        (nz - 2 + block[2] - 1) // block[2],
    )

    kernel(grid, block, (C, L, np.int32(nx), np.int32(ny), np.int32(nz)))
    return L


def _laplacian_cpu(C: np.ndarray) -> np.ndarray:
    """CPU Laplacian via NumPy slicing (same as original SpatialField code)."""
    L = np.zeros_like(C)
    L[1:-1, 1:-1, 1:-1] = (
        C[2:, 1:-1, 1:-1] + C[:-2, 1:-1, 1:-1] +
        C[1:-1, 2:, 1:-1] + C[1:-1, :-2, 1:-1] +
        C[1:-1, 1:-1, 2:] + C[1:-1, 1:-1, :-2] -
        6 * C[1:-1, 1:-1, 1:-1]
    )
    return L


def _is_valid(idx: tuple, shape: tuple) -> bool:
    return all(0 <= i < s for i, s in zip(idx, shape))
