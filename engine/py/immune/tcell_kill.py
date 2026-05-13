"""
T-cell kill probability
=======================

Stage four of the closed-loop neoantigen presentation pipeline
(Upgrade 2). Converts a TCR-pMHC affinity into a per-encounter kill
probability after weighing it against MHC-I surface density and the
costimulation environment (CD80/CD86, IL-2, checkpoint state).

The kill rule is intentionally a small, transparent combination of
three multiplicands so the simulation behaviour is interpretable and
amenable to sensitivity analysis. Production-grade replacements -- e.g.,
agent-based cytotoxic-synapse models or experimentally-fit Hill curves
-- can drop in by reimplementing ``kill_probability`` without touching
the immune module that calls it.

Inputs
------
affinity         in [0, 1] -- from TCR repertoire matching
mhc_level        in [0, 1] -- fraction of MHC-I surface coverage
                   (0 = checkpoint inhibitor downregulation,
                    1 = normal expression)
costimulation    in [0, 1] -- CD28-CD80/86 signal strength
                   (0 = anergic state, 1 = full costim)
checkpoint_block in [0, 1] -- relief from PD-1 / CTLA-4 inhibition
                   (0 = no checkpoint therapy, 1 = full block).
                   Default 0.0 -- so the baseline reflects an
                   immune-suppressive tumor microenvironment.

The product (affinity * mhc_level * costimulation) lands in [0, 1];
checkpoint_block additively relieves a hard PD-1 ceiling. The final
probability passes through a Hill curve with adjustable threshold and
slope so the rule can express "all-or-nothing" or "graded" regimes.

Patent claim surface: see module-level docstring of
``tcr_repertoire.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


# Default Hill curve parameters. Threshold 0.3 means a combined signal
# of 0.3 yields kill probability 0.5; slope 4 sharpens the transition.
DEFAULT_HILL_THRESHOLD: float = 0.3
DEFAULT_HILL_SLOPE: float = 4.0


@dataclass(frozen=True)
class KillOutcome:
    """Decomposition of a kill-probability computation for logging.

    The closed-loop test inspects these fields to verify the kill
    decision is grounded in MHC display + affinity, not in an unrelated
    heuristic.
    """
    affinity: float
    mhc_level: float
    costimulation: float
    checkpoint_block: float
    combined_signal: float   # the value fed into the Hill curve
    kill_probability: float
    is_exhausted: bool = False
    exhaustion_multiplier: float = 1.0


def kill_probability(
    affinity: float,
    mhc_level: float,
    costimulation: float,
    checkpoint_block: float = 0.0,
    hill_threshold: float = DEFAULT_HILL_THRESHOLD,
    hill_slope: float = DEFAULT_HILL_SLOPE,
    is_exhausted: bool = False,
    exhaustion_multiplier: float = 0.1,
) -> float:
    """Probability that this TCR-pMHC encounter results in cell death.

    All inputs are clamped to [0, 1]. The combined signal is

        signal = affinity * mhc_level * costimulation
                 + (1 - costimulation) * checkpoint_block * 0.5     [precursor only]

    The second (checkpoint-rescue) term is GATED on the engaging T
    cell being a PD-1-lo precursor; exhausted PD-1-hi clones are not
    rescuable by ICB (Dolina et al. 2021, lecture slide 52). Without
    this gate, the formula would imply ICB rescues every dysfunctional
    T cell, which is the rule-based-only error the Stage-D exhaustion
    model is specifically built to correct.

    Exhausted clones additionally get their entire kill probability
    multiplied by ``exhaustion_multiplier`` (default 0.1): they retain
    residual cytotoxicity but at a fraction of the precursor pool.

    The final probability is

        p = signal^slope / (signal^slope + threshold^slope)

    which is a Hill function with EC50 = threshold, optionally scaled
    down for exhausted clones.
    """
    a = _clamp01(affinity)
    m = _clamp01(mhc_level)
    c = _clamp01(costimulation)
    b = _clamp01(checkpoint_block)
    mult = _clamp01(exhaustion_multiplier)

    if is_exhausted:
        # No checkpoint rescue for exhausted T cells -- they remain
        # locked at the PD-1-hi dysfunctional state regardless of
        # how much PD-1 is blocked at the surface.
        signal = a * m * c
    else:
        signal = a * m * c + (1.0 - c) * b * 0.5
    signal = max(0.0, min(1.0, signal))

    if signal <= 0:
        return 0.0
    slope = max(0.1, hill_slope)
    thr = max(1e-6, hill_threshold)
    num = signal ** slope
    den = num + thr ** slope
    p = float(num / den)
    if is_exhausted:
        p *= mult
    return max(0.0, min(1.0, p))


def kill_outcome(
    affinity: float,
    mhc_level: float,
    costimulation: float,
    checkpoint_block: float = 0.0,
    hill_threshold: float = DEFAULT_HILL_THRESHOLD,
    hill_slope: float = DEFAULT_HILL_SLOPE,
    is_exhausted: bool = False,
    exhaustion_multiplier: float = 0.1,
) -> KillOutcome:
    """Same as ``kill_probability`` but returns the full decomposition."""
    a = _clamp01(affinity)
    m = _clamp01(mhc_level)
    c = _clamp01(costimulation)
    b = _clamp01(checkpoint_block)
    mult = _clamp01(exhaustion_multiplier)
    if is_exhausted:
        signal = a * m * c
    else:
        signal = a * m * c + (1.0 - c) * b * 0.5
    signal = max(0.0, min(1.0, signal))
    p = kill_probability(
        a, m, c, b,
        hill_threshold=hill_threshold,
        hill_slope=hill_slope,
        is_exhausted=is_exhausted,
        exhaustion_multiplier=exhaustion_multiplier,
    )
    return KillOutcome(
        affinity=a,
        mhc_level=m,
        costimulation=c,
        checkpoint_block=b,
        combined_signal=signal,
        kill_probability=p,
        is_exhausted=is_exhausted,
        exhaustion_multiplier=mult,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _clamp01(x: float) -> float:
    if x is None or math.isnan(x):
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)
