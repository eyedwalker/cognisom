"""
Macrophage M1/M2 polarization as a simulated state (Upgrade 8)
===============================================================

Before this module, M1/M2 existed in the platform only as a
*transcriptomic marker score* over input expression data
(``genomics/cell2sentence._score_polarization_markers``, aggregated
into ``ImmuneClassification.m1_m2_ratio``). Nothing in the simulation
had a polarization state: ``ImmuneCell`` had no field for it and every
macrophage killed with the same flat probability as every NK cell.
The two halves never met. This module is the simulated half.

Sign convention
---------------
polarization in [-1, +1], **-1 = M1, +1 = M2**. This matches
``cell2sentence._score_polarization_markers`` exactly, which returns
"score: -1 = pure M1, +1 = pure M2" and labels at +/- 0.3. The same
cutoffs are reused in ``polarization_label`` so a simulated macrophage
and a classified one are described in the same words.

The immunology
--------------
The M1/M2 axis (Mills CD et al., J Immunol 2000;164:6166-73;
Mantovani A et al., Trends Immunol 2002;23:549-55; Mosser DM &
Edwards JP, Nat Rev Immunol 2008;8:958-69):

  * IFN-gamma drives classical (M1) activation: TNF-alpha, IL-1beta,
    IL-6, IL-12, NO, and tumoricidal activity.
  * IL-10 and TGF-beta drive alternative (M2) activation: IL-10 and
    TGF-beta of their own, arginase-1, tissue remodelling, and
    tumour promotion (Qian BZ & Pollard JW, Cell 2010;141:39-51).
  * Hypoxia and tumour-derived lactate push tumour-associated
    macrophages toward the M2/pro-angiogenic end (Colegio OR et al.,
    Nature 2014;513:559-63; Henze AT & Mazzone M, J Clin Invest
    2016;126:3672-9).

The cytokine outputs returned by ``cytokine_output`` are exactly the
``cytokines_secreting`` fields already curated on the "Macrophage M1"
and "Macrophage M2" rows of the entity library (M1 -> TNFa / IL1b /
IL6 / IL12 / NO; M2 -> IL10 / TGFb), so the simulated cell and the
library vocabulary agree.

M2 macrophages suppress T cells by expressing arginase-1, depleting
extracellular L-arginine, which downregulates CD3-zeta and blocks
T-cell proliferation (Rodriguez PC et al., Cancer Res 2004;64:5839-49;
Bronte V & Zanovello P, Nat Rev Immunol 2005;5:641-54; Munder M,
Br J Pharmacol 2009;158:638-51). ``arginase_attenuation`` is that
suppression, shaped to compose multiplicatively with the exhaustion
and sympathetic gates already on the T-cell kill path.

What is calibration and what is measurement
-------------------------------------------
Everything numeric here is a **calibration**, declared as such:

  * ``DEFAULT_TAU_HOURS = 24.0`` -- M1<->M2 repolarization in vitro
    takes roughly a day. At the 0.5-4 h durations the demo scripts
    run, a macrophage therefore moves only a few percent of the way
    to its target. That is the honest behaviour of a mechanism with
    this time constant, not a bug: if the mechanism needs to be
    visible in a demo, the run duration is what should change.
  * ``DEFAULT_KILL_SPAN = 0.5`` -- how much more tumoricidal a pure
    M1 is than an unpolarized M0. The direction is well established;
    the magnitude is a choice.
  * ``DEFAULT_ARGINASE_MAX_SUPPRESSION = 0.6`` -- the T-cell function
    an entirely M2-surrounded T cell loses.
  * Equal weighting of the M1 and M2 drives.

NO initial polarization is inferred here. ``initial_polarization``
takes an M2 fraction only when a caller has one, and returns the
unpolarized state otherwise. The platform currently has no measured
M1/M2 ratio for any patient -- the only runtime path to
``CellStateClassifier`` ends at ``generate_synthetic_expression()``,
which plants the M2 markers it then rediscovers -- so seeding a
simulated polarization from it would be a round trip over its own
labels. ``ImmuneModule`` reports the source as unmeasured accordingly.

This module computes local cytokine levels from *cell counts*, in the
manner of ``engine/py/spatial/ecm_barrier.ecm_density_at``, and runs
no diffusion solver. The claim is the gating -- that a patient's
myeloid compartment modulates innate killing and T-cell function --
not the concentration field.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional


#: Pure phenotype endpoints on the polarization axis.
M1 = -1.0
M2 = +1.0
M0 = 0.0

#: Label cutoffs, identical to
#: cell2sentence._score_polarization_markers.
M1_LABEL_CUTOFF: float = -0.3
M2_LABEL_CUTOFF: float = +0.3

#: Relaxation time constant toward the signal-determined target.
DEFAULT_TAU_HOURS: float = 24.0

#: Span of the polarization-dependent kill factor: a pure M1 kills
#: (1 + span) x the unpolarized rate, a pure M2 (1 - span) x.
DEFAULT_KILL_SPAN: float = 0.5

#: T-cell function lost when fully surrounded by M2 macrophages.
DEFAULT_ARGINASE_MAX_SUPPRESSION: float = 0.6

#: Radius over which a secreted cytokine is taken to act, in um.
#: Oyler-Yaniv A et al., Immunity 2017;46:609-20 measured an
#: effective diffusion-consumption range for IL-2 / IFN-gamma in
#: lymphoid tissue on the order of tens to ~150 um; 30 um is a few
#: cell diameters and is the conservative end of that range.
DEFAULT_CYTOKINE_RADIUS_UM: float = 30.0

#: Number of secreting cells inside the radius at which the local
#: signal saturates. A calibration: the point of this function is
#: the gating, not the concentration.
DEFAULT_CYTOKINE_SATURATION_COUNT: float = 3.0

#: Oxygen fraction treated as normoxic. Matches the 0.21 default on
#: CellState.oxygen in modules/cellular_module.py.
NORMOXIC_OXYGEN: float = 0.21


def _clamp01(x: float) -> float:
    if x is None:
        return 0.0
    x = float(x)
    if math.isnan(x):
        return 0.0
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _clamp_polarization(x: float) -> float:
    if x is None:
        return M0
    x = float(x)
    if math.isnan(x):
        return M0
    return M1 if x < M1 else (M2 if x > M2 else x)


def polarization_label(polarization: float) -> str:
    """'M1', 'M2' or 'mixed', on cell2sentence's cutoffs."""
    p = _clamp_polarization(polarization)
    if p < M1_LABEL_CUTOFF:
        return "M1"
    if p > M2_LABEL_CUTOFF:
        return "M2"
    return "mixed"


def local_signal(count: float, saturation_count: float = DEFAULT_CYTOKINE_SATURATION_COUNT) -> float:
    """Saturating map from a count of secreting cells to [0, 1].

    ``min(1, count / saturation_count)``. Linear then flat: the
    simplest monotone saturating shape, and the one that makes the
    zero case exactly zero (no secreting cell nearby means no signal,
    with no epsilon left over).
    """
    if saturation_count is None or float(saturation_count) <= 0.0:
        return 1.0 if float(count) > 0.0 else 0.0
    return _clamp01(float(count) / float(saturation_count))


def hypoxia_signal(oxygen: float, normoxic: float = NORMOXIC_OXYGEN) -> float:
    """Hypoxia drive in [0, 1] from an oxygen fraction.

    ``1 - oxygen / normoxic``, clamped. 0 at or above normoxia, 1 at
    anoxia.
    """
    if normoxic is None or float(normoxic) <= 0.0:
        return 0.0
    if oxygen is None:
        return 0.0
    o = float(oxygen)
    if math.isnan(o):
        return 0.0
    return _clamp01(1.0 - o / float(normoxic))


@dataclass(frozen=True)
class PolarizationDrive:
    """Decomposition of one polarization step, for logging."""
    ifn_gamma: float
    il10: float
    tgf_beta: float
    hypoxia: float
    m1_drive: float
    m2_drive: float
    target: float
    previous: float
    polarization: float
    tau_hours: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "ifn_gamma": self.ifn_gamma,
            "il10": self.il10,
            "tgf_beta": self.tgf_beta,
            "hypoxia": self.hypoxia,
            "m1_drive": self.m1_drive,
            "m2_drive": self.m2_drive,
            "target": self.target,
            "previous": self.previous,
            "polarization": self.polarization,
            "tau_hours": self.tau_hours,
        }


def polarization_target(
    ifn_gamma: float = 0.0,
    il10: float = 0.0,
    tgf_beta: float = 0.0,
    hypoxia: float = 0.0,
    m1_weight: float = 1.0,
    m2_weight: float = 1.0,
) -> float:
    """Signal-determined polarization set point in [-1, +1].

        m1_drive = m1_weight * ifn_gamma                 (saturating)
        m2_drive = m2_weight * (il10 + tgf_beta + hypoxia) (saturating)
        target   = m2_drive - m1_drive

    Each drive saturates at 1 so the target cannot leave [-1, +1]
    without any extra clamping, and the difference form means a cell
    receiving both an M1 and an M2 signal of equal strength sits at
    M0 rather than at an arbitrary tie-break.

    Equal weighting of the two arms is a declared choice, not a
    measurement. IFN-gamma plus LPS is the canonical M1 stimulus and
    is arguably stronger than any single M2 input; callers that want
    to say so raise ``m1_weight``.
    """
    m1_drive = _clamp01(float(m1_weight) * _clamp01(ifn_gamma))
    m2_drive = _clamp01(
        float(m2_weight)
        * (_clamp01(il10) + _clamp01(tgf_beta) + _clamp01(hypoxia))
    )
    return _clamp_polarization(m2_drive - m1_drive)


def polarization_step(
    current: float,
    target: float,
    dt_hours: float,
    tau_hours: float = DEFAULT_TAU_HOURS,
) -> float:
    """One first-order relaxation step toward ``target``.

        p' = p + (target - p) * dt / tau

    Explicit Euler on dp/dt = (target - p) / tau. The step is clamped
    so that dt >= tau lands exactly on the target rather than
    overshooting -- the module is stepped at dt = 0.01 h against a
    tau of 24 h in every shipped configuration, but a caller running
    coarse steps should not get an oscillation out of it.

    At tau = 24 h and dt = 0.01 h a single step closes 1/2400 of the
    gap, so a 2 h run moves a macrophage about 8% of the way to its
    target. That is the mechanism's real timescale, not a defect.
    """
    p = _clamp_polarization(current)
    t = _clamp_polarization(target)
    dt = float(dt_hours)
    tau = float(tau_hours)
    if dt <= 0.0:
        return p
    if tau <= 0.0 or dt >= tau:
        return t
    return _clamp_polarization(p + (t - p) * (dt / tau))


def kill_factor(
    polarization: float,
    span: float = DEFAULT_KILL_SPAN,
) -> float:
    """Multiplier on a macrophage's baseline kill probability.

        1 - polarization * span

    1.0 at M0, (1 + span) at pure M1, (1 - span) at pure M2, clamped
    at 0. Being exactly 1.0 at M0 is what makes an unpolarized
    macrophage behave exactly as it did before this module existed.
    """
    p = _clamp_polarization(polarization)
    f = 1.0 - p * float(span)
    return f if f > 0.0 else 0.0


def cytokine_output(polarization: float) -> Dict[str, float]:
    """Relative cytokine secretion of a macrophage at this
    polarization, in [0, 1] per species.

    The M1 and M2 species are the ones curated on the entity
    library's "Macrophage M1" and "Macrophage M2" rows. An M0
    macrophage secretes nothing: the scale is "how far toward this
    phenotype", not an absolute rate, and the platform has no
    half-life or diffusion coefficient for any of these species to
    turn one into the other.
    """
    p = _clamp_polarization(polarization)
    m1 = -p if p < 0.0 else 0.0
    m2 = p if p > 0.0 else 0.0
    return {
        # M1 (classically activated), library row "Macrophage M1"
        "TNF_alpha": m1,
        "IL1_beta": m1,
        "IL6": m1,
        "IL12": m1,
        "NO": m1,
        # M2 (alternatively activated), library row "Macrophage M2"
        "IL10": m2,
        "TGF_beta1": m2,
        "ARG1": m2,
    }


def arginase_attenuation(
    m2_burden: float,
    max_suppression: float = DEFAULT_ARGINASE_MAX_SUPPRESSION,
) -> float:
    """Fraction of T-cell kill function retained next to M2
    macrophages.

        1 - m2_burden * max_suppression

    Returns a multiplier in [1 - max_suppression, 1.0], exactly 1.0
    at ``m2_burden = 0`` -- so a T cell with no M2 macrophage nearby
    is untouched, and a configuration with no macrophages at all
    (which is every immune configuration in the test suite, and the
    flagship closed-loop configuration) is a provable no-op.

    Same shape as ``sympathetic.sympathetic_attenuation``, on purpose:
    both are final multiplicative gates on the T-cell kill path and
    they compose by multiplication.

    ``m2_burden`` is the local M2 macrophage density in [0, 1], from
    ``local_signal`` over a count of nearby M2-polarized macrophages
    weighted by how M2 each one is.
    """
    b = _clamp01(m2_burden)
    m = _clamp01(max_suppression)
    return 1.0 - b * m


def initial_polarization(m2_fraction: Optional[float]) -> float:
    """Seed polarization from a population M2 fraction, or M0.

    ``None`` -- the honest default, and what
    ``twin_config.to_simulation_params()`` emits for an unmeasured
    immune profile -- returns M0.

    A population fraction constrains the compartment *mean*, not any
    individual cell, so a fraction f seeds every macrophage at the
    mean ``2f - 1`` and lets the per-cell drive diverge them from
    there. f = 0.5 gives exactly M0, so a balanced compartment and an
    unmeasured one start from the same place by construction.
    """
    if m2_fraction is None:
        return M0
    f = _clamp01(m2_fraction)
    return _clamp_polarization(2.0 * f - 1.0)
