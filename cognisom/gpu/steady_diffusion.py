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
#: Relative residual reduction asked of the inner CG solve. Much looser than
#: the outer Newton target, on purpose: Newton absorbs an inexact step, so
#: solving each linear system to more precision than the nonlinear iteration
#: needs is wasted. Measured on a 100^3 grid, the cost halves between 1e-6 and
#: 1e-3 -- 136 CG iterations and 661 ms against 68 and 338 -- while the solved
#: depletion agrees to about 1e-5 of itself. Below that it over-loosens: 1e-2
#: saves no iterations and costs an extra Newton step.
#:
#: What this gives up is exactness, and the amount is worth stating. Driven to
#: 1e-12 the solver reproduces an independently assembled sparse Newton solve
#: to 2e-16, which is how the method is verified; at the default it agrees to
#: around 1e-5 of the depletion, which is far inside the uncertainty on the
#: consumption rates feeding it. The verification tests pass ``cg_tol``
#: explicitly so that the exactness claim is tested at the setting it
#: describes and the speed claim at the setting that ships.
CG_TOLERANCE = 1e-3

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
        # Accumulated in place rather than through np.pad. Padding allocates
        # and copies a whole extra grid on EVERY call, which at 100^3 float64
        # is 8.5 MB per CG iteration and was 46% of the solve. Mirroring the
        # edge value is the same as adding the edge slice to itself, so the
        # zero-flux condition needs no ghost storage at all.
        np.multiply(field, -6.0, out=out)
        for axis in range(3):
            lo_dst = [slice(None)] * 3
            lo_src = [slice(None)] * 3
            lo_dst[axis] = slice(1, None)
            lo_src[axis] = slice(0, -1)
            out[tuple(lo_dst)] += field[tuple(lo_src)]

            hi_dst = [slice(None)] * 3
            hi_src = [slice(None)] * 3
            hi_dst[axis] = slice(0, -1)
            hi_src[axis] = slice(1, None)
            out[tuple(hi_dst)] += field[tuple(hi_src)]

            # The two faces, where the mirrored neighbour is the voxel itself.
            face = [slice(None)] * 3
            face[axis] = 0
            out[tuple(face)] += field[tuple(face)]
            face[axis] = -1
            out[tuple(face)] += field[tuple(face)]
        out *= inv
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


def krogh_exchange(length_per_volume: Any, radius_um: float,
                   resolution_um: float, diffusion_coeff: float) -> Any:
    """Exchange conductance for a sub-voxel vessel, in 1/s.

    A vessel marked as whole voxels is exactly one voxel wide, so its physical
    diameter is the grid resolution: halving the grid halves every capillary
    and quarters the vessel volume fraction, and the answer moves with it
    (see tests/test_a_vessel_is_one_voxel_wide.py). The radius has to enter
    the model through something other than how many voxels the vessel fills.

    Here it enters through the conductance. Inside a voxel the field around a
    cylinder is the Krogh solution, radial and logarithmic, so the steady flux
    per unit vessel length from a surface at radius r to a far field at radius
    R is

        Q = 2 pi D (C_blood - C) / ln(R / r)

    Taking R as the voxel's own half-width -- the distance at which the
    voxel-average concentration is representative -- and multiplying by the
    vessel length per unit volume gives a conductance per unit volume:

        beta = 2 pi D * lambda / ln(dx / (2 r))

    which depends on the grid only logarithmically, where the whole-voxel
    condition depended on it linearly. As r approaches dx/2 the logarithm
    vanishes and beta diverges, which is the Dirichlet limit: a vessel that
    fills its voxel is pinned, exactly as before.

    Args:
        length_per_volume: vessel length per unit volume, in 1/um^2. For a
            square lattice of vessels spaced S apart this is 1/S^2 averaged
            over the tissue, concentrated in the voxels the vessels pass
            through.
        radius_um: the vessel's PHYSICAL radius. A capillary is 2.5-4 um.
        resolution_um: voxel edge.
        diffusion_coeff: D, in um^2/s.

    Returns:
        Conductance in 1/s, matching the shape of ``length_per_volume``.
    """
    if radius_um <= 0:
        raise ValueError(f"radius_um must be positive, got {radius_um}")
    if resolution_um <= 0:
        raise ValueError(
            f"resolution_um must be positive, got {resolution_um}")
    half = 0.5 * resolution_um
    if radius_um >= half:
        raise ValueError(
            f"a vessel of radius {radius_um} um does not fit inside a "
            f"{resolution_um} um voxel (half-width {half} um). At that size "
            "the whole-voxel Dirichlet condition IS the right model -- pass "
            "the mask as `held` instead"
        )
    return (2.0 * np.pi * diffusion_coeff
            * np.asarray(length_per_volume, dtype=np.float64)
            / np.log(half / radius_um))


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
    # The Jacobi preconditioner folded together with the free mask, so the
    # inner loop applies one multiply instead of two. Held once: neither
    # changes across iterations.
    precond = np.where(diag > 0, 1.0 / diag, 1.0) * free
    z = r * precond
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
        # In place: each of these allocated a full grid per iteration, and at
        # 100^3 float64 that is 8 MB a time for arrays thrown away at once.
        x += alpha * p
        r -= alpha * Ap
        np.multiply(r, precond, out=z)
        rz_new = float(np.vdot(r, z).real)
        if rz_new <= target:
            break
        p *= rz_new / rz
        p += z
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
    exchange: Any = 0.0,
    exchange_value: float = 0.0,
    faces: str = INSULATED,
    tol: float = 1e-2,
    cg_tol: float = CG_TOLERANCE,
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
        held: boolean mask of Dirichlet voxels -- whole-voxel capillaries.
            Marking a voxel makes the capillary exactly one voxel wide, which
            ties the vessel's physical diameter to the grid resolution; see
            ``exchange`` for the sub-voxel alternative.
        exchange: per-voxel exchange conductance in 1/s, scalar or field,
            adding a distributed source ``exchange * (exchange_value - C)``.
            This is the sub-voxel vasculature: the vessel's radius enters
            through the conductance rather than through how many voxels it
            occupies, so refining the grid no longer thins the capillary.
            See ``krogh_exchange``.
        exchange_value: the concentration the exchange drives towards -- the
            level the blood carries.
        held_value: the value to hold them at. Defaults to the starting field's
            value at those voxels, so a mask with the arterial level already
            written needs nothing further.
        faces: ``INSULATED`` (default, zero flux) or ``ARTERIAL``.
        tol: convergence target for the Newton correction, as a fraction of
            the DEPLETION rather than of the field. See the module docstring.
        cg_tol: relative residual reduction for each inner linear solve; see
            :data:`CG_TOLERANCE` for what the default trades away.
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

    beta = np.asarray(exchange, dtype=np.float64)
    if beta.ndim:
        if beta.shape != C.shape:
            raise ValueError(
                f"exchange shape {beta.shape} does not match field shape "
                f"{C.shape}")
        if (beta < 0).any():
            raise ValueError("exchange must be non-negative")
        beta = beta * free      # a held voxel is already a boundary
    elif float(beta) < 0:
        raise ValueError("exchange must be non-negative")
    blood = float(exchange_value)

    scratch = np.empty_like(C)
    result = SteadyResult(field=C)

    for step in range(1, max_newton + 1):
        residual = (D * _laplacian(C, dx, faces, scratch)
                    - _reaction(C, vmax, uptake_km)
                    + beta * (blood - C)) * free
        # d/dC of the terms moved to the left: R'(C) from the sink, and beta
        # from the exchange. Both are non-negative, so both improve the
        # conditioning of -J rather than threatening its definiteness.
        slope = (_reaction_slope(C, vmax, uptake_km) + beta) * free

        # -J = -(D L - diag(slope)) = -D L + diag(slope), SPD.
        # Scratch for the operator, reused across every CG iteration. As
        # written with plain expressions this allocated five full grids per
        # iteration -- 40 MB a time at 100^3 float64 -- for arrays discarded
        # immediately. The search direction is already zero on held voxels
        # (the preconditioner carries the free mask), so masking the input
        # again is redundant; the output is still masked, which keeps the
        # residual clean for the vdot that follows.
        op_lap = np.empty_like(C)
        op_tmp = np.empty_like(C)

        # Bound as default arguments, not captured: an augmented assignment
        # to a closed-over name makes it local, so `lap *= -D` on a captured
        # buffer raises UnboundLocalError rather than writing through.
        def apply_op(x, _slope=slope, lap=op_lap, tmp=op_tmp):
            _laplacian(x, dx, faces, lap)
            lap *= -D
            np.multiply(_slope, x, out=tmp)
            lap += tmp
            lap *= free
            return lap

        delta, cg_used = _cg(
            apply_op, residual, free, slope + D * 6.0 / (dx * dx),
            tol=cg_tol, max_iter=max_cg)
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
             - _reaction(C, vmax, uptake_km)
             + beta * (blood - C)) * free
    result.imbalance = float(np.abs(final).max())
    result.field = C
    return result
