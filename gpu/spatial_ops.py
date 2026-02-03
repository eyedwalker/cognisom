"""
GPU-Accelerated Spatial Operations
====================================

Pairwise distance computation, neighbor search, and force
calculations for the immune module and vascular module.

The immune module's inner loop (each immune cell vs each cancer cell)
is an N-body problem — perfect for GPU parallelism. Same pattern
appears in vascular exchange (capillary points vs cells).

On GPU: CuPy pairwise distance kernels.
On CPU: NumPy broadcasting (still faster than Python loops).
"""

import logging
from typing import Tuple, Optional

import numpy as np

from .backend import get_backend

logger = logging.getLogger(__name__)

# ── CUDA kernel for pairwise distances ───────────────────────────

_PAIRWISE_DIST_KERNEL_SRC = r"""
extern "C" __global__
void pairwise_distance_3d(
    const float* __restrict__ A,  // (M, 3)
    const float* __restrict__ B,  // (N, 3)
    float* __restrict__ D,        // (M, N)
    int M, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= M || j >= N) return;

    float dx = A[i*3 + 0] - B[j*3 + 0];
    float dy = A[i*3 + 1] - B[j*3 + 1];
    float dz = A[i*3 + 2] - B[j*3 + 2];

    D[i*N + j] = sqrtf(dx*dx + dy*dy + dz*dz);
}
"""

_cuda_pairwise_kernel = None


def _get_pairwise_kernel():
    global _cuda_pairwise_kernel
    if _cuda_pairwise_kernel is None:
        import cupy as cp
        _cuda_pairwise_kernel = cp.RawKernel(_PAIRWISE_DIST_KERNEL_SRC, "pairwise_distance_3d")
    return _cuda_pairwise_kernel


# ── Public API ───────────────────────────────────────────────────

def pairwise_distances(
    positions_a: np.ndarray,
    positions_b: np.ndarray,
) -> np.ndarray:
    """Compute pairwise Euclidean distances between two sets of 3D points.

    Args:
        positions_a: (M, 3) array of positions.
        positions_b: (N, 3) array of positions.

    Returns:
        (M, N) distance matrix.
    """
    backend = get_backend()
    xp = backend.xp

    A = backend.to_device(np.ascontiguousarray(positions_a, dtype=np.float32))
    B = backend.to_device(np.ascontiguousarray(positions_b, dtype=np.float32))

    M = A.shape[0]
    N = B.shape[0]

    if M == 0 or N == 0:
        return xp.zeros((M, N), dtype=np.float32)

    if backend.has_gpu:
        return _pairwise_gpu(A, B, M, N, xp)
    else:
        return _pairwise_cpu(A, B)


def find_neighbors(
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    radius: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find all pairs within a distance threshold.

    Args:
        positions_a: (M, 3) positions of first set.
        positions_b: (N, 3) positions of second set.
        radius: Distance threshold.

    Returns:
        (indices_a, indices_b, distances): matching pairs.
    """
    D = pairwise_distances(positions_a, positions_b)
    backend = get_backend()
    xp = backend.xp

    mask = D < radius
    idx_a, idx_b = xp.nonzero(mask)

    # Extract distances for matching pairs
    dists = D[idx_a, idx_b]

    return (
        backend.to_numpy(idx_a),
        backend.to_numpy(idx_b),
        backend.to_numpy(dists),
    )


def nearest_neighbor(
    positions_a: np.ndarray,
    positions_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find the nearest point in B for each point in A.

    Args:
        positions_a: (M, 3) query points.
        positions_b: (N, 3) target points.

    Returns:
        (nearest_indices, distances): for each point in A.
    """
    D = pairwise_distances(positions_a, positions_b)
    backend = get_backend()
    xp = backend.xp

    if D.size == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    nearest_idx = xp.argmin(D, axis=1)
    nearest_dist = D[xp.arange(D.shape[0]), nearest_idx]

    return backend.to_numpy(nearest_idx), backend.to_numpy(nearest_dist)


def compute_directions(
    from_positions: np.ndarray,
    to_positions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute unit direction vectors and distances from A to B.

    Args:
        from_positions: (N, 3) start positions.
        to_positions: (N, 3) target positions (same N).

    Returns:
        (directions, distances): (N, 3) unit vectors, (N,) distances.
    """
    backend = get_backend()
    xp = backend.xp

    A = backend.to_device(np.ascontiguousarray(from_positions, dtype=np.float32))
    B = backend.to_device(np.ascontiguousarray(to_positions, dtype=np.float32))

    diff = B - A
    dist = xp.sqrt(xp.sum(diff ** 2, axis=1))

    # Avoid division by zero
    safe_dist = xp.maximum(dist, 1e-6)
    directions = diff / safe_dist[:, None]

    return backend.to_numpy(directions), backend.to_numpy(dist)


def immune_detection_batch(
    immune_positions: np.ndarray,
    cancer_positions: np.ndarray,
    detection_radius: float,
    cancer_mhc1: np.ndarray,
    immune_types: np.ndarray,
    mhc1_threshold_t: float = 0.2,
    mhc1_threshold_nk: float = 0.4,
) -> list:
    """Batch immune cell detection of cancer targets.

    Replaces the nested loop in ImmuneModule.update():
        for immune_cell in immune_cells:
            for cancer_cell in cancer_cells:
                if distance < detection_radius and can_recognize():
                    activate()

    Args:
        immune_positions: (M, 3) immune cell positions.
        cancer_positions: (N, 3) cancer cell positions.
        detection_radius: Detection range in um.
        cancer_mhc1: (N,) MHC-I expression levels of cancer cells.
        immune_types: (M,) type codes (0=T_cell, 1=NK_cell, 2=macrophage).
        mhc1_threshold_t: MHC-I threshold for T cell recognition.
        mhc1_threshold_nk: MHC-I threshold for NK cell recognition.

    Returns:
        List of (immune_idx, cancer_idx, distance) detection events.
    """
    if immune_positions.shape[0] == 0 or cancer_positions.shape[0] == 0:
        return []

    # Distance matrix
    D = pairwise_distances(immune_positions, cancer_positions)
    backend = get_backend()
    xp = backend.xp

    D_np = backend.to_numpy(D)
    mhc1_np = np.asarray(cancer_mhc1, dtype=np.float32)
    itypes_np = np.asarray(immune_types, dtype=np.int8)

    detections = []
    M, N = D_np.shape

    for i in range(M):
        for j in range(N):
            if D_np[i, j] >= detection_radius:
                continue

            itype = itypes_np[i]
            mhc = mhc1_np[j]

            # Recognition logic (mirrors ImmuneModule._can_recognize)
            recognized = False
            if itype == 0:  # T_cell — needs MHC-I
                recognized = mhc > mhc1_threshold_t
            elif itype == 1:  # NK_cell — detects missing MHC-I
                recognized = mhc < mhc1_threshold_nk
            elif itype == 2:  # macrophage — always recognizes
                recognized = True

            if recognized:
                detections.append((i, j, float(D_np[i, j])))
                break  # Each immune cell only detects one target

    return detections


# ── Internal implementations ────────────────────────────────────

def _pairwise_gpu(A, B, M, N, xp):
    """GPU pairwise distance via CUDA kernel."""
    D = xp.zeros((M, N), dtype=xp.float32)

    kernel = _get_pairwise_kernel()
    block = (16, 16)
    grid = (
        (M + block[0] - 1) // block[0],
        (N + block[1] - 1) // block[1],
    )

    kernel(grid, block, (A, B, D, np.int32(M), np.int32(N)))
    return D


def _pairwise_cpu(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """CPU pairwise distance via NumPy broadcasting."""
    # A: (M, 3), B: (N, 3) -> D: (M, N)
    diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]  # (M, N, 3)
    return np.sqrt(np.sum(diff ** 2, axis=-1))
