"""Solve a fast-relaxing field for its steady state instead of stepping it.

Oxygen crosses the 60 um gap between capillaries in L^2/D = 1.8 s. The tissue
engine's step is 180 s. The field has therefore settled a hundred times over
before the cells have aged once, so time-stepping it answers a question nobody
asked and pays for the answer with the operator-splitting error.

How much that error costs is not obvious from the usual way of quoting it.
``implicit_diffusion._uptake_field_step`` reports an error of order ``k*dt``
and notes that the engine runs at ``k*dt = 0.0071`` -- "0.71%". That is correct
and it is a fraction of the FIELD VALUE. What a tissue simulation needs from
an oxygen field is the DEPLETION, because the depletion is what decides
whether a cell is hypoxic, and at the configured consumption rate the true
depletion is 4.77e-06 against a field of 0.21 -- 0.002% of it. So an error
worth 0.71% of the field is three hundred times the entire quantity of
interest. Measured against an exact Newton solve of the same 7-point operator,
the stepped scheme's fixed point depletes by 1.49e-03 where the truth is
4.77e-06: over by 312x, with every voxel still inside 0.71% of arterial and
the field still looking almost uniform.

Stepping cannot be rescued by taking more substeps at any sensible price:
reaching 10% of the true depletion needs dt near 0.005 s, about 7,000 substeps
per engine step, which is the same order as the 4,320 the EXPLICIT scheme
needs merely for stability. Nor by iterating a pseudo-time relaxation to its
fixed point -- that fixed point IS the wrong answer, and the iterate goes
bit-for-bit stationary while the steady-state equation is still out of
balance.

So this module solves

    D grad^2 C = Vmax(x) C / (km + C)

directly, with capillary voxels held at the arterial level (the Krogh-Erlang
inner boundary) and the outer faces insulated by default. Newton on the
reaction term, matrix-free conjugate gradient for each Newton step. Matrix-free
because a sparse operator for the 500^3 grid this engine targets would be 1.25e8
rows; the stencil needs no storage beyond the field itself.

Two details that are easy to get wrong, and were:

**The faces.** ``adi_step`` never writes its outermost layer, which makes every
face a Dirichlet boundary at whatever the field was initialised to -- so the
edges of the tissue block supply oxygen exactly as a capillary does. For a
block cut out of a larger tissue that is wrong and it dominates once vessels
are more than a few voxels apart. The default here is ``"insulated"``, zero
flux, which is the symmetry condition a representative block actually has.
``"arterial"`` reproduces the stepped solver's behaviour, for comparison.

**The convergence test.** Measured against the depletion, for the reason above.
A criterion on the field value is satisfied long before the depletion means
anything, and a residual scaled by the sink over-reports the field error by
D/(dx^2 k) -- a factor of 168 at the shipped numbers -- while the same residual
scaled by the diffusive flux goes slack whenever the sink is weak and will
pronounce an untouched uniform field converged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

#: Faces carry no flux: the block is a representative piece of a larger
#: tissue, so its cut edges are symmetry planes and not sources.
INSULATED = "insulated"
#: Faces held at the boundary value, which is what ``adi_step`` does by never
#: writing them. Kept so the two solvers can be compared on equal terms.
ARTERIAL = "arterial"


@dataclass
class SteadyResult:
    """The solved field, and enough to judge whether to believe it."""

    field: np.ndarray
    #: Newton steps taken.
    newton_steps: int = 0
    #: Total CG iterations across all Newton steps.
    cg_steps: int = 0
    #: max |D grad^2 C - R(C)| over free voxels, in units of concentration per
    #: second. Zero for an exact solution.
    imbalance: float = 0.0
    #: The depletion the solution shows: boundary value minus the minimum.
    #: Quote ``imbalance`` against this and not against the field, or the
    #: error sounds three hundred times smaller than it is.
    depletion: float = 0.0
    #: Largest Newton correction on the final step, as a fraction of the
    #: depletion. This is the convergence measure.
    last_step_vs_depletion: float = 0.0
    converged: bool = False

    def __repr__(self) -> str:                             # pragma: no cover
        return (f"SteadyResult(newton={self.newton_steps}, "
                f"cg={self.cg_steps}, depletion={self.depletion:.3e}, "
                f"step/depletion={self.last_step_vs_depletion:.2e}, "
                f"converged={self.converged})")


def _laplacian(field: np.ndarray, resolution: float, faces: str,
               out: Optional[np.ndarray] = None) -> np.ndarray:
    """7-point Laplacian with the chosen face condition.

    ``INSULATED`` mirrors the edge value outward, which makes the normal
    derivative zero. ``ARTERIAL`` leaves the outermost layer out of the result
    entirely, matching a solver that never writes it.
    """
    if out is None:
        out = np.zeros_like(field)
    else:
        out.fill(0.0)

    inv = 1.0 / (resolution * resolution)
    if faces == INSULATED:
        padded = np.pad(field, 1, mode="edge")
        out[...] = (
            padded[2:, 1:-1, 1:-1] + padded[:-2, 1:-1, 1:-1]
            + padded[1:-1, 2:, 1:-1] + padded[1:-1, :-2, 1:-1]
            + padded[1:-1, 1:-1, 2:] + padded[1:-1, 1:-1, :-2]
            - 6.0 * field
        ) * inv
        return out

    core = (slice(1, -1),) * 3
    if min(field.shape) < 3:
        return out
    out[core] = (
        field[2:, 1:-1, 1:-1] + field[:-2, 1:-1, 1:-1]
        + field[1:-1, 2:, 1:-1] + field[1:-1, :-2, 1:-1]
        + field[1:-1, 1:-1, 2:] + field[1:-1, 1:-1, :-2]
        - 6.0 * field[core]
    ) * inv
    return out


def _reaction(C: np.ndarray, vmax: Any, km: float) -> np.ndarray:
    """R(C), the per-voxel consumption."""
    if km <= 0.0:
        return vmax * C
    return vmax * C / (km + np.abs(C))


def _reaction_slope(C: np.ndarray, vmax: Any, km: float) -> np.ndarray:
    """dR/dC, which Newton needs and which is what makes this converge fast.

    For the saturating form the slope collapses to nearly zero wherever the
    field is well above km -- that is the whole point of saturation -- so the
    Jacobian is close to the plain Laplacian over perfused tissue and only
    picks up a diagonal where oxygen has actually run down.
    """
    if km <= 0.0:
        return np.broadcast_to(np.asarray(vmax, dtype=C.dtype), C.shape)
    denom = km + np.abs(C)
    return vmax * km / (denom * denom)


def _cg(apply_op, rhs: np.ndarray, free: np.ndarray, diag: np.ndarray,
        tol: float, max_iter: int) -> Tuple[np.ndarray, int]:
    """Jacobi-preconditioned CG on a symmetric positive-definite operator.

    The operator handed in is ``-J``: minus the Laplacian is positive
    definite, and the reaction slope adds a non-negative diagonal, so the
    system is SPD and CG is the right tool rather than a general solver.

    Held voxels are eliminated rather than constrained: the correction is
    defined to be zero there, so their current values enter only through the
    residual and never through the search direction. That keeps the operator
    symmetric, which a penalty or a large diagonal would not.
    """
    x = np.zeros_like(rhs)
    r = rhs * free
    precond = np.where(diag > 0, 1.0 / diag, 1.0)
    z = r * precond * free
    p = z.copy()
    rz = float(np.vdot(r, z).real)
    if rz <= 0.0:
        return x, 0
    target = tol * tol * rz
    used = 0
    for used in range(1, max_iter + 1):
        Ap = apply_op(p)
        denom = float(np.vdot(p, Ap).real)
        if denom <= 0.0:
            # Loss of positive-definiteness means the preconditioner or the
            # operator is no longer what CG assumes. Stop with what we have
            # rather than take a step in a direction that may increase the
            # residual.
            break
        alpha = rz / denom
        x += alpha * p
        r -= alpha * Ap
        z = r * precond * free
        rz_new = float(np.vdot(r, z).real)
        if rz_new <= target:
            break
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, used


def solve_steady_state(
    field: np.ndarray,
    diffusion_coeff: float,
    resolution_um: float,
    *,
    uptake_k: Any = 0.0,
    uptake_km: float = 0.0,
    uptake_ref: float = 1.0,
    held: Optional[np.ndarray] = None,
    held_value: Optional[float] = None,
    faces: str = INSULATED,
    tol: float = 1e-2,
    max_newton: int = 30,
    max_cg: Optional[int] = None,
) -> SteadyResult:
    """Solve ``D grad^2 C = Vmax C/(km + C)`` for C.

    Args:
        field: starting guess, modified only in the returned copy. Warm-start
            from the previous solution and Newton typically needs one step.
        diffusion_coeff: D, in um^2/s.
        resolution_um: voxel edge, in um.
        uptake_k: per-voxel first-order rate in 1/s, scalar or field, with the
            same meaning as ``adi_step``'s: the rate AT ``uptake_ref``, so
            ``Vmax = uptake_k * uptake_ref`` and switching to the saturating
            form does not rescale the demand the caller configured.
        uptake_km: half-saturation concentration. 0 gives a first-order sink,
            which cannot produce a Krogh profile at any rate and is kept only
            for fields that genuinely are first order.
        held: boolean mask of Dirichlet voxels -- the capillaries.
        held_value: the value to hold them at. Defaults to the starting field's
            value at those voxels, so a mask with the arterial level already
            written needs nothing further.
        faces: ``INSULATED`` (default, zero flux) or ``ARTERIAL``.
        tol: convergence target for the Newton correction, as a fraction of
            the DEPLETION rather than of the field. See the module docstring.
        max_newton: Newton steps before giving up.
        max_cg: CG iterations per Newton step. The default scales with the grid
            and is generous; the vessel lattice makes this system very well
            conditioned, because no free voxel is more than a few cells from a
            Dirichlet one, so CG normally finishes far sooner.

    Returns:
        A :class:`SteadyResult`. Check ``converged``; if it is False, read
        ``imbalance`` against ``depletion`` before believing the field.
    """
    if faces not in (INSULATED, ARTERIAL):
        raise ValueError(
            f"faces must be {INSULATED!r} or {ARTERIAL!r}, got {faces!r}")
    if diffusion_coeff <= 0:
        raise ValueError(
            f"diffusion_coeff must be positive, got {diffusion_coeff}")
    if resolution_um <= 0:
        raise ValueError(
            f"resolution_um must be positive, got {resolution_um}")
    if uptake_km < 0:
        raise ValueError(f"uptake_km must be non-negative, got {uptake_km}")

    C = np.array(field, dtype=np.float64, copy=True)
    if C.ndim != 3:
        raise ValueError(f"field must be 3-D, got shape {C.shape}")

    if held is None:
        held_mask = np.zeros(C.shape, dtype=bool)
    else:
        held_mask = np.asarray(held, dtype=bool)
        if held_mask.shape != C.shape:
            raise ValueError(
                f"held shape {held_mask.shape} does not match field shape "
                f"{C.shape}")
    if faces == ARTERIAL:
        # The outermost layer is a boundary, so it belongs in the mask rather
        # than being silently skipped by the stencil.
        for axis in range(3):
            sl = [slice(None)] * 3
            sl[axis] = 0
            held_mask[tuple(sl)] = True
            sl[axis] = -1
            held_mask[tuple(sl)] = True

    level = held_value
    if level is None:
        level = float(C[held_mask].max()) if held_mask.any() else float(C.max())
    C[held_mask] = level

    free = (~held_mask).astype(np.float64)
    n_free = float(free.sum())
    if n_free == 0:
        return SteadyResult(field=C, converged=True)

    vmax = np.asarray(uptake_k, dtype=np.float64) * float(uptake_ref)
    if vmax.ndim:
        if vmax.shape != C.shape:
            raise ValueError(
                f"uptake_k shape {vmax.shape} does not match field shape "
                f"{C.shape}")
        if (vmax < 0).any():
            raise ValueError("uptake_k must be non-negative")
        vmax = vmax * free        # a held voxel's own sink is the boundary's
    elif float(vmax) < 0:
        raise ValueError("uptake_k must be non-negative")

    D = float(diffusion_coeff)
    dx = float(resolution_um)
    if max_cg is None:
        max_cg = max(50, 4 * max(C.shape))

    scratch = np.empty_like(C)
    result = SteadyResult(field=C)

    for step in range(1, max_newton + 1):
        residual = (D * _laplacian(C, dx, faces, scratch)
                    - _reaction(C, vmax, uptake_km)) * free
        slope = _reaction_slope(C, vmax, uptake_km) * free

        # -J = -(D L - diag(slope)) = -D L + diag(slope), SPD.
        def apply_op(x, _slope=slope):
            lap = _laplacian(x * free, dx, faces)
            return (-D * lap + _slope * x) * free

        delta, cg_used = _cg(apply_op, residual, free, slope + D * 6.0 / (dx * dx),
                             tol=1e-6, max_iter=max_cg)
        result.cg_steps += cg_used
        C += delta * free
        # The saturating sink is non-negative only while C is; Newton can
        # overshoot below zero on the first step from a bad guess, and R(C)
        # is not defined there in any useful sense.
        np.clip(C, 0.0, None, out=C)
        C[held_mask] = level

        depletion = max(level - float(C[free > 0].min()), 0.0)
        measure = float(np.abs(delta * free).max())
        rel = measure / depletion if depletion > 0 else (
            0.0 if measure == 0.0 else float("inf"))
        result.newton_steps = step
        result.depletion = depletion
        result.last_step_vs_depletion = rel
        if rel <= tol:
            result.converged = True
            break

    final = (D * _laplacian(C, dx, faces, scratch)
             - _reaction(C, vmax, uptake_km)) * free
    result.imbalance = float(np.abs(final).max())
    result.field = C
    return result
