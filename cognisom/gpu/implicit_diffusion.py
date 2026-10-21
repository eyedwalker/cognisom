"""Unconditionally stable implicit diffusion (ADI), because explicit cannot work here.

WHY THIS EXISTS
---------------
The tissue engine advances diffusion with explicit Euler and a 5-substep
budget (``TissueScaleConfig.diffusion_substeps``). That is only stable because
of a unit bug: ``step()`` is handed ``dt`` in **hours** while the diffusion
coefficients are in **um^2/s**, so each step advances 0.01 *seconds* of
diffusion instead of 0.01 hours -- 3600x too slow in time, which for oxygen is
a diffusion length sqrt(3600) = 60x shorter than it should be.

Correcting that conversion makes the explicit scheme unusable rather than
merely slow. At the shipped configuration (oxygen D = 2000 um^2/s, dx = 10 um,
dt = 0.01 h = 36 s):

    D*dt/dx^2 = 720,   and explicit Euler in 3-D requires <= 1/6

so stability needs 4320 substeps per engine step instead of 5 -- an 864x cost
increase, about 1286 h of wall clock for a 24 h run at the 1M-cell preset
against the engine's own 2 h cap (``max_runtime_hours``). There is no amount
of kernel optimisation that closes an 864x gap; the time integrator has to
change.

An implicit scheme is unconditionally stable, so one step covers the whole
engine dt. Measured on this grid: **0.26 h** for the same 24 h run.

THE SCHEME
----------
Locally one-dimensional (LOD) backward Euler, three directional solves per
step::

    (I - r*Lx + k*dt/3) C*   = C^n
    (I - r*Ly + k*dt/3) C**  = C*
    (I - r*Lz + k*dt/3) C^n+1 = C**

with r = D*dt/dx^2 and Lx the 1-D three-point Laplacian. Each solve is
tridiagonal with *constant* coefficients, so the Thomas factorisation is
computed once per (length, r, k*dt) and reused for every line in the sweep --
that is what makes an implicit method affordable here.

ACCURACY, MEASURED
------------------
* **1-D: exact.** The fixed point reproduces the exact discrete steady state to
  0.000000% at every dt tried, including dt = 5 s where r = 100 (600x the
  explicit limit). This is not an accident of the test: folding the uptake onto
  the tridiagonal diagonal makes the scheme's fixed point *identical* to the
  steady-state linear system, since (I - rL + k*dt)C = C rearranges to
  (D/dx^2)*L*C = k*C.
* **3-D: first order in dt**, from the directional splitting -- 23.2% / 5.8% /
  1.5% at k*dt = 0.5 / 0.1 / 0.025 against an exact sparse solve of the
  discrete 3-D operator.
* **At the engine's actual operating point** (dt = 36 s, k = q/C_art ~ 0.714/h
  as ``spatial_module`` defines it): **0.062%** with a single step. The large r
  does not hurt; the splitting error scales with k*dt, which is 0.0071 there.
* Free diffusion reproduces the analytic second moment <x^2> = 2*D*t to
  0.000%, and conserves mass to 1e-8.
* Stable and non-negative at r up to 1e6, six million times the explicit limit.

WHY UPTAKE IS DIVIDED BY THE NUMBER OF *ACTIVE* SWEEPS
------------------------------------------------------
An axis with fewer than 3 cells has no interior to solve, so its sweep is
skipped. Dividing k*dt by a hard-coded 3 then applies only a fraction of the
uptake: in a (61,1,1) column it applied one third, which showed up as a 640%
steady-state error with an effective lambda of sqrt(3)*lambda. The divisor has
to be the number of sweeps that actually run.

Two alternatives were tried and rejected, both because of the Dirichlet
boundary rather than the interior arithmetic. Applying uptake as a separate
factor after the sweeps -- whether 1/(1+k*dt) or the exact exp(-k*dt) -- scales
the boundary-adjacent interior point while its pinned neighbour keeps its fixed
value, so the interior effectively sees a boundary of C_b*(1-k*dt). For a cosh
profile that propagates to the centre as an error of exactly k*dt, which is
what both variants measured.

BOUNDARIES
----------
Dirichlet: the two boundary planes on each axis hold their current values and
are never written. That matches what the explicit solver does -- it writes the
interior only -- so vessel voxels and domain faces behave as before. Callers
that pin vessels re-pin them after each step.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

try:  # numba is a hard requirement for the fast path, not for correctness
    from numba import njit, prange
    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised by the fallback tests
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        def wrap(fn):
            return fn
        return wrap(args[0]) if args and callable(args[0]) else wrap

    prange = range  # type: ignore[assignment]


__all__ = ["adi_step", "thomas_coefficients", "have_numba"]


def have_numba() -> bool:
    """Whether the compiled sweeps are available (they are ~45x the NumPy path)."""
    return _HAVE_NUMBA


def thomas_coefficients(m: int, r: float, kterm: float, dtype) -> Tuple[Any, Any]:
    """Forward-elimination coefficients for the constant tridiagonal.

    The system is (-r, 1 + 2r + kterm, -r) on every line, so these depend only
    on the line length and the two scalars -- hence computed once and cached.

    Returned in ``dtype`` deliberately. Keeping them float64 against a float32
    field promotes every element of the inner loop and measured *slower* than
    float64 throughout (191 ms vs 132 ms per 500^3 step).
    """
    diag = 1.0 + 2.0 * r + kterm
    cp = np.empty(m, dtype=dtype)
    den = np.empty(m, dtype=dtype)
    den[0] = diag
    cp[0] = -r / diag
    for i in range(1, m):
        d = diag + r * float(cp[i - 1])
        den[i] = d
        cp[i] = -r / d
    return cp, den


@njit(parallel=True, cache=True, fastmath=False)
def _sweep0(C, r, cp, den, j0, j1, k0, k1):
    """Axis 0. Parallel over j so the inner k run stays contiguous."""
    nx, ny, nz = C.shape
    m = nx - 2
    for j in prange(j0, j1):
        for kk in range(k0, k1):
            b0 = C[1, j, kk] + r * C[0, j, kk]
            if m == 1:
                # A single interior row is both first and last, so BOTH
                # Dirichlet terms fold into it and the loop below never runs.
                b0 += r * C[nx - 1, j, kk]
            C[1, j, kk] = b0 / den[0]
            for i in range(1, m):
                b = C[i + 1, j, kk]
                if i == m - 1:
                    b += r * C[nx - 1, j, kk]
                C[i + 1, j, kk] = (b + r * C[i, j, kk]) / den[i]
            for i in range(m - 2, -1, -1):
                C[i + 1, j, kk] -= cp[i] * C[i + 2, j, kk]


@njit(parallel=True, cache=True, fastmath=False)
def _sweep1(C, r, cp, den, i0, i1, k0, k1):
    """Axis 1. Parallel over i; inner k contiguous."""
    nx, ny, nz = C.shape
    m = ny - 2
    for i in prange(i0, i1):
        for kk in range(k0, k1):
            b0 = C[i, 1, kk] + r * C[i, 0, kk]
            if m == 1:
                b0 += r * C[i, ny - 1, kk]
            C[i, 1, kk] = b0 / den[0]
            for j in range(1, m):
                b = C[i, j + 1, kk]
                if j == m - 1:
                    b += r * C[i, ny - 1, kk]
                C[i, j + 1, kk] = (b + r * C[i, j, kk]) / den[j]
            for j in range(m - 2, -1, -1):
                C[i, j + 1, kk] -= cp[j] * C[i, j + 2, kk]


@njit(parallel=True, cache=True, fastmath=False)
def _sweep2(C, r, cp, den, i0, i1, j0, j1):
    """Axis 2, which is contiguous: the best access pattern of the three."""
    nx, ny, nz = C.shape
    m = nz - 2
    for i in prange(i0, i1):
        for j in range(j0, j1):
            b0 = C[i, j, 1] + r * C[i, j, 0]
            if m == 1:
                b0 += r * C[i, j, nz - 1]
            C[i, j, 1] = b0 / den[0]
            for kk in range(1, m):
                b = C[i, j, kk + 1]
                if kk == m - 1:
                    b += r * C[i, j, nz - 1]
                C[i, j, kk + 1] = (b + r * C[i, j, kk]) / den[kk]
            for kk in range(m - 2, -1, -1):
                C[i, j, kk + 1] -= cp[kk] * C[i, j, kk + 2]


def _other_axis_bounds(C: np.ndarray, axis: int) -> Tuple[int, int, int, int]:
    """Loop bounds over the two axes that are NOT being solved.

    Only triple-interior voxels may evolve: a voxel on any domain face is a
    boundary condition, which is how the explicit solver behaves -- it writes
    ``lap[1:-1, 1:-1, 1:-1]`` and leaves every face alone. Sweeping an axis
    over its full extent instead would let the axis-1 and axis-2 sweeps
    overwrite the axis-0 faces, so a pinned vessel or arterial face would not
    stay pinned.

    An axis thinner than 3 has no interior at all. Such an axis is iterated
    over its whole extent so that a genuinely 1-D or slab domain still
    diffuses along the axes that do have room; restricting it to an empty
    interior would make the whole array inert.
    """
    others = [ax for ax in (0, 1, 2) if ax != axis]
    bounds: List[int] = []
    for ax in others:
        n = C.shape[ax]
        bounds.extend((1, n - 1) if n >= 3 else (0, n))
    return bounds[0], bounds[1], bounds[2], bounds[3]


@njit(parallel=True, cache=True)
def _clamp_bounded(C, i0, i1, j0, j1, k0, k1):
    for i in prange(i0, i1):
        for j in range(j0, j1):
            for kk in range(k0, k1):
                if C[i, j, kk] < 0.0:
                    C[i, j, kk] = 0.0


def _interior_slices(C) -> Tuple[slice, slice, slice]:
    """The region a step may write. Faces are boundary conditions."""
    out: List[slice] = []
    for ax in (0, 1, 2):
        n = C.shape[ax]
        out.append(slice(1, n - 1) if n >= 3 else slice(0, n))
    return out[0], out[1], out[2]


def _clamp_interior(C) -> None:
    """Clamp only the writable interior, so a pinned face is never altered.

    The explicit solver clamps the whole array, including its faces. Doing the
    same here would decay a pinned vessel or arterial face by the clamp alone
    and, with uptake applied as a post-factor, by (1 - k*dt) each step.
    """
    si, sj, sk = _interior_slices(C)
    if si.stop <= si.start or sj.stop <= sj.start or sk.stop <= sk.start:
        return
    if _HAVE_NUMBA and C.flags["C_CONTIGUOUS"]:
        _clamp_bounded(C, si.start, si.stop, sj.start, sj.stop, sk.start, sk.stop)
    else:
        sub = C[si, sj, sk]
        np.maximum(sub, 0.0, out=sub)
        C[si, sj, sk] = sub


def _sweep_numpy(C: np.ndarray, axis: int, r: float, cp, den) -> np.ndarray:
    """Reference sweep, used when numba is unavailable.

    Bit-identical to the compiled kernels on every shape tested, including the
    degenerate ones -- see ``tests/test_implicit_diffusion.py``. It is ~45x
    slower at 500^3 because the transposes dominate, so it exists for
    correctness and portability rather than for production use.
    """
    n = C.shape[axis]
    if n < 3:
        return C
    a0, a1, b0_, b1_ = _other_axis_bounds(C, axis)
    others = [ax for ax in (0, 1, 2) if ax != axis]
    sel: List[Any] = [slice(None)] * 3
    sel[others[0]] = slice(a0, a1)
    sel[others[1]] = slice(b0_, b1_)
    view = C[tuple(sel)]
    if view.size == 0:
        return C

    A = np.moveaxis(view, axis, -1)
    shape = A.shape
    flat = np.ascontiguousarray(A).reshape(-1, n)
    m = n - 2
    rr = C.dtype.type(r)

    b = flat[:, 1:-1].copy()
    b[:, 0] += rr * flat[:, 0]
    b[:, -1] += rr * flat[:, -1]

    dp = np.empty_like(b)
    dp[:, 0] = b[:, 0] / den[0]
    for i in range(1, m):
        dp[:, i] = (b[:, i] + rr * dp[:, i - 1]) / den[i]
    x = np.empty_like(b)
    x[:, m - 1] = dp[:, m - 1]
    for i in range(m - 2, -1, -1):
        x[:, i] = dp[:, i] - cp[i] * x[:, i + 1]

    flat[:, 1:-1] = x
    C[tuple(sel)] = np.moveaxis(flat.reshape(shape), -1, axis)
    return C


@njit(parallel=True, cache=True, fastmath=False)
def _apply_uptake_field(C, K, dt, i0, i1, j0, j1, k0, k1):
    for i in prange(i0, i1):
        for j in range(j0, j1):
            for kk in range(k0, k1):
                C[i, j, kk] = C[i, j, kk] / (1.0 + K[i, j, kk] * dt)


@njit(parallel=True, cache=True, fastmath=False)
def _apply_uptake_field_mm(C, K, dt, km, ref, i0, i1, j0, j1, k0, k1):
    for i in prange(i0, i1):
        for j in range(j0, j1):
            for kk in range(k0, k1):
                c0 = C[i, j, kk]
                b = km - c0 + K[i, j, kk] * ref * dt
                q = c0 * km
                disc = np.sqrt(b * b + 4.0 * q)
                # Both branches are the same positive root. Taking
                # (-b + disc)/2 when b > 0 subtracts two nearly equal
                # numbers and loses the answer; the conjugate form does
                # not. Measured 1.8e-4 relative error at large k*dt
                # before this split.
                if b >= 0.0:
                    denom = b + disc
                    C[i, j, kk] = 2.0 * q / denom if denom > 0.0 else 0.0
                else:
                    C[i, j, kk] = 0.5 * (-b + disc)


def _uptake_field_step(C, uptake, dt_seconds,
                       km: float = 0.0, ref: float = 1.0) -> None:
    """Backward Euler on dC/dt = -k(x)*C, interior only, in place.

    A per-voxel rate cannot go on the tridiagonal diagonal without making the
    coefficients vary line by line, which would defeat the cached
    factorisation that makes this solver affordable. So a spatially varying
    uptake is operator-split instead, and that costs accuracy the scalar path
    does not pay: the scalar version folds k onto the diagonal and is EXACT at
    steady state in 1-D, while this one carries the usual reaction-diffusion
    splitting error of order k*dt.

    Measured against an exact sparse solve of the discrete operator, the error
    is roughly k*dt: ~2.5% at k*dt = 0.025, ~0.7% at the engine's operating
    point (k = 0.714/h, dt = 36 s, k*dt = 0.0071). Interior only, so pinned
    vessel and arterial voxels keep their values -- scaling those was what
    produced an error of exactly k*dt at the centre of a cosh profile.

    With ``km > 0`` the sink saturates instead: dC/dt = -Vmax*C/(km + C) with
    Vmax = k(x)*ref, so ``uptake`` keeps its meaning as the first-order rate
    AT the reference concentration and the configured magnitude is unchanged
    at ``C = ref``. Backward Euler on that ODE is a quadratic in C_new with
    one positive root, written out in closed form -- unconditionally stable,
    never negative, and exact in the km -> 0 limit, where it reduces to
    max(0, C - Vmax*dt).

    Why this matters more than the rate does: a first-order sink consumes
    MOST where the concentration is HIGHEST, which is the voxels next to a
    vessel, and least in the tissue between vessels. It decays exponentially
    with no front, so no distance from a vessel is ever anoxic while its
    neighbours are fed -- the Krogh profile cannot form at any k. Respiration
    is not first-order: cytochrome c oxidase runs at a near-constant rate down
    to about 1 mmHg, two orders below arterial pO2, so over the whole
    physiological range a cell consumes oxygen at a rate set by its own demand
    and not by the supply. That constant draw is what makes a sharp hypoxic
    boundary and an anoxic core.
    """
    si, sj, sk = _interior_slices(C)
    if si.stop <= si.start or sj.stop <= sj.start or sk.stop <= sk.start:
        return
    contiguous = (_HAVE_NUMBA and C.flags["C_CONTIGUOUS"]
                  and uptake.flags["C_CONTIGUOUS"])
    if km > 0.0:
        if contiguous:
            _apply_uptake_field_mm(C, uptake, float(dt_seconds), float(km),
                                   float(ref),
                                   si.start, si.stop, sj.start, sj.stop,
                                   sk.start, sk.stop)
        else:
            c0 = C[si, sj, sk]
            b = km - c0 + uptake[si, sj, sk] * ref * dt_seconds
            q = c0 * km
            disc = np.sqrt(b * b + 4.0 * q)
            denom = b + disc
            C[si, sj, sk] = np.where(
                b >= 0.0,
                2.0 * q / np.where(denom > 0.0, denom, 1.0),
                0.5 * (-b + disc),
            )
        return
    if contiguous:
        _apply_uptake_field(C, uptake, float(dt_seconds),
                            si.start, si.stop, sj.start, sj.stop,
                            sk.start, sk.stop)
    else:
        sub = C[si, sj, sk]
        C[si, sj, sk] = sub / (1.0 + uptake[si, sj, sk] * dt_seconds)


def adi_step(
    field: np.ndarray,
    diffusion_coeff: float,
    dt_seconds: float,
    resolution_um: float,
    uptake_k: Any = 0.0,
    uptake_km: float = 0.0,
    uptake_ref: float = 1.0,
    cache: Optional[Dict] = None,
    use_numba: Optional[bool] = None,
) -> np.ndarray:
    """Advance ``field`` by one unconditionally stable implicit step.

    Args:
        field: 3-D concentration array. Modified in place on the numba path.
        diffusion_coeff: D in um^2/s.
        dt_seconds: the step, in **seconds**. The engine's ``cfg.dt`` is in
            hours; multiply by 3600 before calling. Passing hours here is the
            unit bug this module's docstring describes.
        resolution_um: voxel edge length in um.
        uptake_km: half-saturation concentration for a saturating sink, in the
            field's own units. 0 (the default) keeps the first-order sink.
            Only honoured when ``uptake_k`` is a field; see
            ``_uptake_field_step`` for why a saturating sink is what a Krogh
            profile needs.
        uptake_ref: the concentration at which the saturating sink draws at
            exactly ``uptake_k``, so that Vmax = uptake_k * uptake_ref.
            Normally the arterial value.
        uptake_k: first-order uptake in 1/s, as ``spatial_module.add_uptake``
            defines it (k = q_volumetric / C_arterial). Zeroth-order sinks
            cannot be used in a steady-state solve; see that module.

            A float goes onto the tridiagonal diagonal and is exact at steady
            state in 1-D. An array of the field's shape -- which is what cell
            consumption actually looks like, since cells are discrete -- is
            operator-split instead, because a varying diagonal would force the
            Thomas factorisation to be recomputed per line. That split costs
            an error of order k*dt (~0.7% at this engine's operating point);
            see ``_uptake_field_step``.
        cache: reusable Thomas-coefficient cache. Pass the same dict across
            steps -- recomputing the factorisation every step is pure waste
            since the coefficients only depend on (length, r, k*dt).
        use_numba: force the compiled or the NumPy path. Defaults to compiled
            when numba is importable.

    Returns:
        The updated field (the same object on the numba path).
    """
    if field.ndim != 3:
        raise ValueError(f"adi_step expects a 3-D field, got {field.ndim}-D")
    if dt_seconds < 0:
        raise ValueError(f"dt_seconds must be non-negative, got {dt_seconds}")
    if cache is None:
        cache = {}
    if use_numba is None:
        use_numba = _HAVE_NUMBA

    uptake_is_field = isinstance(uptake_k, np.ndarray)
    if uptake_is_field:
        if uptake_k.shape != field.shape:
            raise ValueError(
                f"uptake_k array shape {uptake_k.shape} does not match the "
                f"field shape {field.shape}"
            )
        if (uptake_k < 0).any():
            raise ValueError("uptake_k must be non-negative")
        scalar_k = 0.0
    else:
        if uptake_k < 0:
            raise ValueError(f"uptake_k must be non-negative, got {uptake_k}")
        scalar_k = float(uptake_k)

    r = diffusion_coeff * dt_seconds / (resolution_um * resolution_um)
    active = [ax for ax in (0, 1, 2) if field.shape[ax] >= 3]

    if not active:
        # Nothing to diffuse: no axis has an interior. Only uptake applies, and
        # it goes on implicitly so it stays stable at any dt.
        out = np.array(field, copy=True)
        if uptake_is_field:
            _uptake_field_step(out, uptake_k, dt_seconds,
                           uptake_km, uptake_ref)
        else:
            out /= (1.0 + scalar_k * dt_seconds)
        _clamp_interior(out)
        return out

    # The divisor is the number of sweeps that RUN, not 3 -- see the module
    # docstring; using 3 here under-applies uptake on degenerate grids.
    kterm = scalar_k * dt_seconds / len(active)

    if use_numba and _HAVE_NUMBA:
        C = field if field.flags["C_CONTIGUOUS"] else np.ascontiguousarray(field)
        for ax in active:
            m = C.shape[ax] - 2
            key = (m, float(r), float(kterm), C.dtype.str)
            if key not in cache:
                cache[key] = thomas_coefficients(m, r, kterm, C.dtype.type)
            cp, den = cache[key]
            a0, a1, b0, b1 = _other_axis_bounds(C, ax)
            if a1 <= a0 or b1 <= b0:
                continue
            (_sweep0, _sweep1, _sweep2)[ax](
                C, C.dtype.type(r), cp, den, a0, a1, b0, b1)
        if uptake_is_field:
            _uptake_field_step(C, uptake_k, dt_seconds,
                               uptake_km, uptake_ref)
        _clamp_interior(C)
        if C is not field:
            field[...] = C
            return field
        return C

    out = np.ascontiguousarray(field)
    for ax in active:
        m = out.shape[ax] - 2
        key = (m, float(r), float(kterm), out.dtype.str)
        if key not in cache:
            cache[key] = thomas_coefficients(m, r, kterm, out.dtype.type)
        cp, den = cache[key]
        out = np.ascontiguousarray(_sweep_numpy(out, ax, r, cp, den))
    if uptake_is_field:
        _uptake_field_step(out, uptake_k, dt_seconds,
                           uptake_km, uptake_ref)
    _clamp_interior(out)
    return out
