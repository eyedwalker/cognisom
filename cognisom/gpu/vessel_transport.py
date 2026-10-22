"""Blood that runs out of oxygen on its way through the tissue.

Every vessel in this model was a Dirichlet boundary pinned at the arterial
level along its whole length -- an infinite reservoir. That is defensible
while the vasculature is dense. At the configured 60 um spacing the tissue
draws 9% of the oxygen the blood carries past it, so an error in the other
91% is invisible.

It stops being defensible exactly where this platform's central claim lives.
Measured prostate carcinoma sits at a median 2.4 mmHg against 30.0 mmHg in
the same patient's own periprostatic muscle -- Movsas et al., Am J Clin Oncol
2001;24(5):458-61, **PMID 11586096**, Eppendorf microelectrode, 11,944
readings in 59 patients, tissue type the only significant predictor of nine
covariates (p < 0.0001). Normal PROSTATE is a third value and much closer to
the tumour: 5.2 mmHg, from Milosevic's 44 patients as reported in that
paper's discussion. Healthy prostate is itself hypoxic by the 10 mmHg line.

Reaching 2.4 mmHg by thinning a constant-pO2 vasculature requires about 77%
of the oxygen flowing past each capillary to be extracted -- level with
maximally working muscle, against 25-30% at rest, and sustained rather than
transient.

**That figure was once 137%, and the difference was an error in this
module's favour.** Vessels were pinned at 84 mmHg, the top of tissue
physioxia, which is not a tension any capillary carries; Wack et al. (Med
Phys 2016;43(7):4113, **PMID 27370131**) reset theirs to 40 mmHg while
fitting a Michaelis-Menten sink of this same form to pimonidazole staining
on maps of PERFUSED vessels. At the corrected supply the tissue needs far
less thinning -- 96 um of spacing rather than 235 -- so the demand on each
vessel falls with it and the budget stops being impossible. The strong claim
this module was built on was therefore partly an artefact of a category
error. What survives is weaker and still sufficient: a constant vessel
reaches the measurement only at an extraction physiology rarely sustains,
and the figure swings from 64% to 92% with the assumed vessel diameter,
which the whole-voxel model cannot set independently of its grid.

So tissue oxygenation cannot be set by geometry alone, and the missing piece
is the term Krogh had and this model dropped: **blood depletes as it flows**.
Tissue beside the outlet of a capillary can be anoxic while sitting 10 um
from a vessel. That is how a tumour is hypoxic at high microvessel density --
and it is why microvessel density is the wrong observable to calibrate
against. The prostate pathology literature cannot agree whether MVD even
rises in cancer (2.02x higher, PMID 1283896; not significant and lower by
vessel area, PMID 23069258) and both results can stand while oxygenation is
set by perfusion instead.

The oxygen a capillary carries is almost all bound to haemoglobin -- about
9.1 "atmosphere-fraction" units against 0.11 dissolved, an eighty-fold
amplification, which is the whole reason haemoglobin exists and the reason
this cannot be modelled as a linear reservoir. Saturation comes from
Severinghaus's modified Hill equation, which fits the standard human
dissociation curve to +/-0.0055 saturation across the full range:
Severinghaus, J Appl Physiol 1979;46(3):599-602, doi
10.1152/jappl.1979.46.3.599.

The coupling here is **conservative by construction**. Rather than assume a
wall conductance, each iteration reads the oxygen actually leaving a vessel
voxel from the discrete Laplacian of the solved tissue field, and debits the
blood by exactly that. So the oxygen the tissue receives and the oxygen the
blood loses are the same number to machine precision, and the extraction
fraction is a measured output rather than an assumption. It cannot quietly
exceed 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Any, Optional, Tuple

import numpy as np

from cognisom.gpu.steady_diffusion import INSULATED, solve_steady_state

#: Severinghaus 1979 equation 1, S = 1 / (A / (p^3 + B p) + 1), p in mmHg.
SEVERINGHAUS_A = 23400.0
SEVERINGHAUS_B = 150.0

ATMOSPHERE_MMHG = 760.0

#: Oxygen solubility in plasma and tissue near 37 C, in mM per mmHg -- the
#: Krogh solubility. One unit of "fraction of an atmosphere", the unit the
#: diffusion fields are carried in, is this times 760.
SOLUBILITY_MM_PER_MMHG = 1.3e-3

#: Haemoglobin's carrying capacity: 1.34 mL O2 per gram of saturated Hb at
#: 150 g/L of whole blood. Expressed in the field's own units below.
HB_ML_O2_PER_G = 1.34
HB_G_PER_L = 150.0
MOLAR_GAS_ML = 22400.0

#: A capillary's calibre and the speed of blood in it. 0.5 mm/s is the
#: classic intravital red-cell velocity; a tumour's PERFUSED vessels run at
#: close to normal speed (0.91 vs 1.01 mm/s in pancreas, PMID 10867458), so
#: the tumour-versus-normal difference this model needs is not in the
#: velocity of flowing blood but in how much of the vasculature flows.
CAPILLARY_RADIUS_UM = 3.0
CAPILLARY_VELOCITY_UM_PER_S = 500.0

#: Oxygen tension of blood arriving at the tissue, in mmHg.
ARTERIAL_PO2_MMHG = 95.0


def _hb_capacity_fractions() -> float:
    """Haemoglobin's capacity in units of atmosphere-fraction.

    Derived rather than pinned so the provenance stays visible: it is about
    9.1, against 0.11 for the dissolved oxygen the tissue field carries.
    """
    bound_mm = HB_ML_O2_PER_G * HB_G_PER_L / MOLAR_GAS_ML * 1e3
    dissolved_per_fraction_mm = SOLUBILITY_MM_PER_MMHG * ATMOSPHERE_MMHG
    return bound_mm / dissolved_per_fraction_mm


#: About 9.08. See :func:`_hb_capacity_fractions`.
HB_CAPACITY_FRACTIONS = _hb_capacity_fractions()


def saturation(po2_mmhg: Any) -> Any:
    """Haemoglobin saturation at this oxygen tension, 0 to 1.

    Severinghaus 1979 equation 1. Its own P50 is 26.9 mmHg, within the
    measured 26-27.
    """
    p = np.clip(np.asarray(po2_mmhg, dtype=np.float64), 0.0, None)
    denom = p ** 3 + SEVERINGHAUS_B * p
    with np.errstate(divide="ignore", invalid="ignore"):
        s = 1.0 / (SEVERINGHAUS_A / denom + 1.0)
    return np.where(denom > 0.0, s, 0.0)


def blood_content(po2_mmhg: Any) -> Any:
    """Total oxygen carried by blood at this tension, in field units.

    Dissolved plus haemoglobin-bound, in the same atmosphere-fraction units
    the tissue field uses, so the two can be debited against each other
    without a conversion at the boundary.
    """
    p = np.clip(np.asarray(po2_mmhg, dtype=np.float64), 0.0, None)
    return p / ATMOSPHERE_MMHG + HB_CAPACITY_FRACTIONS * saturation(p)


def po2_from_content(content: Any, iterations: int = 60) -> Any:
    """Invert :func:`blood_content`.

    Bisection, not Severinghaus's equation 2: that inverts the saturation
    alone, and what has to be inverted here is saturation plus the dissolved
    term. Content is strictly increasing in tension, so bisection is
    unconditionally safe, and there are only ever a few thousand vessel
    voxels to do it for.
    """
    target = np.clip(np.asarray(content, dtype=np.float64), 0.0, None)
    lo = np.zeros_like(target)
    hi = np.full_like(target, 1000.0)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        too_low = blood_content(mid) < target
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)
    return 0.5 * (lo + hi)


def capillary_flow_um3_per_s(radius_um: float = CAPILLARY_RADIUS_UM,
                            velocity_um_per_s: float
                            = CAPILLARY_VELOCITY_UM_PER_S) -> float:
    """Volumetric flow through one capillary, in um^3/s."""
    return float(np.pi * radius_um ** 2 * velocity_um_per_s)


@dataclass
class PerfusedResult:
    """A solved field, plus what the blood had to give up to produce it."""

    field: Any
    blood_po2: Any
    converged: bool
    outer_iterations: int = 0
    #: Fraction of the arterial oxygen content extracted, per vessel. Above
    #: 1 is impossible; the solver reports it rather than hiding it.
    extraction: Any = _field(default_factory=lambda: np.zeros(0))
    #: Fraction of vessel length whose blood is exhausted, per vessel.
    exhausted: Any = _field(default_factory=lambda: np.zeros(0))

    @property
    def max_extraction(self) -> float:
        return float(self.extraction.max()) if self.extraction.size else 0.0

    def describe(self) -> str:
        return (f"perfused steady state: {self.outer_iterations} outer "
                f"iterations, max extraction {self.max_extraction:.1%}, "
                f"{'converged' if self.converged else 'NOT CONVERGED'}")


def vessel_outflux(field: Any, mask: Any, diffusion_coeff: float,
                   resolution_um: float) -> Any:
    """Oxygen leaving each vessel voxel per second, in fraction um^3/s.

    Read from the discrete Laplacian of the solved field rather than from an
    assumed wall conductance, which is what makes the coupling conservative:
    the tissue gains exactly what the blood loses. For a finite-volume cell
    the flux across one face is ``D * (C_here - C_there) / dx * dx^2``, so
    the total is ``D * dx`` times the sum of differences to the six
    neighbours.
    """
    C = np.asarray(field, dtype=np.float64)
    total = np.zeros_like(C)
    for axis in range(C.ndim):
        for shift in (-1, 1):
            neighbour = np.roll(C, shift, axis=axis)
            # A domain face has no neighbour outside it; zero flux there
            # matches the INSULATED boundary the tissue solve used.
            edge = [slice(None)] * C.ndim
            edge[axis] = 0 if shift == 1 else -1
            diff = C - neighbour
            diff[tuple(edge)] = 0.0
            total += diff
    return np.where(mask, total * diffusion_coeff * resolution_um, 0.0)


def neighbour_tension(field: Any, mask: Any) -> Any:
    """Mean tension of the non-vessel voxels adjacent to each vessel voxel.

    The equilibrium a vessel's blood can at most reach where it is drawing
    oxygen FROM the tissue rather than giving it up. See :func:`march_blood`.
    """
    C = np.asarray(field, dtype=np.float64)
    free = (~np.asarray(mask, dtype=bool)).astype(np.float64)
    total = np.zeros_like(C)
    count = np.zeros_like(C)
    for axis in range(C.ndim):
        for shift in (-1, 1):
            edge = [slice(None)] * C.ndim
            edge[axis] = 0 if shift == 1 else -1
            weight = np.roll(free, shift, axis=axis)
            value = np.roll(C * free, shift, axis=axis)
            weight[tuple(edge)] = 0.0
            value[tuple(edge)] = 0.0
            total += value
            count += weight
    return np.where(count > 0.0, total / np.maximum(count, 1e-30), C)


def march_blood(outflux: Any, mask: Any, *, axis: int = 2,
                inlet_po2_mmhg: float = ARTERIAL_PO2_MMHG,
                flow_um3_per_s: Optional[float] = None,
                neighbour_po2: Any = None) -> Tuple[Any, Any, Any]:
    """Integrate blood oxygen downstream along every vessel.

    Vessels are lines of voxels along ``axis``. Blood enters at index 0 with
    :data:`ARTERIAL_PO2_MMHG` and loses, at each voxel, exactly the oxygen
    the tissue took from it there.

    One bound is load-bearing. Where the blood has been spent it is colder
    than the tissue beside it and the flux reverses -- a stagnant capillary
    takes oxygen back up, which is real, and is part of why sluggish
    perfusion causes hypoxia. But that reverse flux divided by a small flow
    is a large change in content, so without a bound the tension overshoots
    the tissue it is equilibrating with and the iteration runs away: at a
    tenth of normal velocity it reached the inversion's 1000 mmHg ceiling and
    reported a negative extraction. Diffusion stops at equality, so blood
    leaving a voxel can be no hotter than whichever is warmer, the blood that
    arrived or the tissue it passed. ``neighbour_po2`` supplies the latter;
    without it the bound falls back to the inlet, which still prevents the
    runaway but lets the blood rewarm further than the tissue could manage.

    Returns the blood tension as a field (in atmosphere-fraction, so it can
    be handed straight back as ``held_value``), the extraction fraction per
    vessel, and the fraction of each vessel's length whose blood is spent.
    """
    flow = capillary_flow_um3_per_s() if flow_um3_per_s is None else flow_um3_per_s
    if flow <= 0:
        raise ValueError(f"flow must be positive, got {flow}")

    out = np.moveaxis(np.asarray(outflux, dtype=np.float64), axis, -1)
    m = np.moveaxis(np.asarray(mask), axis, -1)
    # A vessel is a column that is masked along its whole length.
    columns = m.all(axis=-1)
    n_along = out.shape[-1]

    inlet_content = float(blood_content(inlet_po2_mmhg))
    blood = np.zeros_like(out)
    blood[...] = inlet_po2_mmhg / ATMOSPHERE_MMHG

    taken = out[columns]                      # (n_vessels, n_along)
    if neighbour_po2 is None:
        ceiling = np.full(taken.shape, inlet_po2_mmhg)
    else:
        near = np.moveaxis(np.asarray(neighbour_po2, dtype=np.float64), axis, -1)
        ceiling = near[columns] * ATMOSPHERE_MMHG

    content = np.full(taken.shape[0], inlet_content)
    upstream = np.full(taken.shape[0], inlet_po2_mmhg)
    spent = np.zeros(taken.shape[0])
    profile = np.empty_like(taken)
    for k in range(n_along):
        tension = np.minimum(po2_from_content(content),
                             np.maximum(upstream, ceiling[:, k]))
        profile[:, k] = tension / ATMOSPHERE_MMHG
        content = np.clip(blood_content(tension) - taken[:, k] / flow, 0.0, None)
        upstream = tension
        spent += (content <= 0.0)
    blood[columns] = profile

    extraction = (inlet_content - content) / inlet_content
    return (np.moveaxis(blood, -1, axis), extraction, spent / n_along)


def solve_perfused_steady(field: Any, diffusion_coeff: float,
                          resolution_um: float, *, vessels: Any,
                          uptake_k: Any = 0.0, uptake_km: float = 0.0,
                          uptake_ref: float = 1.0, axis: int = 2,
                          inlet_po2_mmhg: float = ARTERIAL_PO2_MMHG,
                          flow_um3_per_s: Optional[float] = None,
                          faces: str = INSULATED, max_outer: int = 60,
                          outer_tol: float = 1e-4, relaxation: float = 0.5,
                          **kwargs: Any) -> PerfusedResult:
    """Steady state with vessels that run down as the tissue draws on them.

    A Picard iteration over two solves that each take the other's answer:
    the tissue field given the blood tension along each vessel, and the blood
    tension given what the tissue drew. Both are monotone in the other, so it
    converges from either side; the loop stops when the blood tension stops
    moving.
    """
    C = np.array(field, dtype=np.float64)
    mask = np.asarray(vessels, dtype=bool)
    blood = np.where(mask, inlet_po2_mmhg / ATMOSPHERE_MMHG, C)
    extraction = np.zeros(0)
    exhausted = np.zeros(0)
    converged = False
    taken = 0

    for taken in range(1, max_outer + 1):
        result = solve_steady_state(
            C, diffusion_coeff, resolution_um, uptake_k=uptake_k,
            uptake_km=uptake_km, uptake_ref=uptake_ref, held=mask,
            held_value=blood, faces=faces, **kwargs)
        C = result.field
        flux = vessel_outflux(C, mask, diffusion_coeff, resolution_um)
        updated, extraction, exhausted = march_blood(
            flux, mask, axis=axis, inlet_po2_mmhg=inlet_po2_mmhg,
            flow_um3_per_s=flow_um3_per_s,
            neighbour_po2=neighbour_tension(C, mask))
        # Under-relaxed: the two solves are strongly coupled once the blood
        # is nearly spent, where a full step oscillates instead of settling.
        updated = blood + relaxation * (updated - blood)
        shift = float(np.abs(updated[mask] - blood[mask]).max()) if mask.any() else 0.0
        blood = updated
        if shift < outer_tol * (inlet_po2_mmhg / ATMOSPHERE_MMHG):
            converged = True
            break

    C[mask] = blood[mask]
    return PerfusedResult(field=C, blood_po2=blood, converged=converged,
                          outer_iterations=taken, extraction=extraction,
                          exhausted=exhausted)
