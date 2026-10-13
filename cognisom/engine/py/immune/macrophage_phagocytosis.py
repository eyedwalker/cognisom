"""
Macrophage phagocytosis: the PD-1 "don't eat me" gate
=====================================================

Macrophage target recognition in the simulation used to be::

    return True

and the kill that followed used the same flat ``self.kill_probability``
as NK cells. Measured: macrophage-only runs at target MHC-I 0.0, 0.4 and
1.0 gave identical kills and identical activations. The macrophage arm
had no sensitivity to anything at all.

What this module adds is one real, cited gate on a field the simulation
already computes. Tumour-associated macrophages express PD-1, their PD-1
expression rises with tumour progression, and it correlates *negatively*
with their phagocytic potency; blocking PD-1 or PD-L1 in vivo increases
macrophage phagocytosis, reduces tumour growth and extends survival in a
macrophage-dependent manner (Gordon et al., Nature 2017, 545:495-499).
So tumour PD-L1 is a don't-eat-me signal in its own right, alongside the
canonical CD47-SIRPalpha axis (Jaiswal et al., Cell 2009, 138:271-285;
Willingham et al., PNAS 2012, 109:6662-6667), which this simulation has
no CD47 state to model.

``CellState.pdl1_expression`` is already driven two ways -- adaptively by
IFN-gamma from activated T cells engaging the cell (Type I TME) and
intrinsically by oncogene status at transformation (Type III TME, Teng
et al., Cancer Res 2015) -- so the gate reads a quantity with real
provenance, and it is exactly 0.0 on a cell that neither route has
touched. That makes the gate a measurable no-op on the existing
baseline: with PD-L1 at 0 the phagocytic probability is the configured
baseline to the bit.

It also repairs a user-facing defect. The dashboard's "Immunotherapy"
preset modelled anti-PD-1 by raising ``kill_probability`` to 0.95 -- a
parameter only NK cells and macrophages ever read -- and never set
``checkpoint_block``, the actual ICB knob in ``tcell_kill``. So the
anti-PD-1 scenario boosted innate killing and left the T-cell loop
untouched. With this gate in place, ``checkpoint_block`` relieves PD-1
on BOTH arms from one knob, which is what blocking one receptor does.

Declared parameter
------------------
``DEFAULT_PD1_MAX_SUPPRESSION`` is the fraction of phagocytic capacity
lost at full PD-L1 with no blockade. Gordon 2017 establishes that the
effect exists, its sign, and that blockade reverses it; it does not give
a per-encounter multiplier, so the magnitude is a declared, exposed
parameter and the tests assert the direction, the rescue and the
PD-L1 = 0 identity rather than this number. Setting it to 0.0 restores
the previous behaviour exactly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


#: Fraction of phagocytic capacity lost at PD-L1 = 1 with no checkpoint
#: blockade. Declared parameter -- see the module docstring. Matches the
#: convention of ``sympathetic.DEFAULT_MAX_SUPPRESSION`` so the two
#: multiplicative gates read the same way.
DEFAULT_PD1_MAX_SUPPRESSION: float = 0.7


@dataclass(frozen=True)
class PhagocytosisOutcome:
    """Decomposition of one macrophage-target engagement."""

    pdl1_expression: float
    checkpoint_block: float
    max_suppression: float
    effective_signal: float       # pdl1 * (1 - checkpoint_block)
    function_retained: float      # multiplier in [1 - max_suppression, 1]
    baseline: float
    probability: float


def pd1_attenuation(
    pdl1_expression: float,
    checkpoint_block: float = 0.0,
    max_suppression: float = DEFAULT_PD1_MAX_SUPPRESSION,
) -> float:
    """Fraction of phagocytic function retained under PD-1 engagement.

        effective = pdl1 * (1 - checkpoint_block)
        retained  = 1 - effective * max_suppression

    so PD-L1 = 0 gives 1.0 exactly, and full blockade gives 1.0 exactly
    at any PD-L1. Same shape as ``sympathetic_attenuation``: both are
    final multiplicative gates on an already-resolved probability.
    """
    p = _clamp01(pdl1_expression)
    b = _clamp01(checkpoint_block)
    m = _clamp01(max_suppression)
    return 1.0 - p * (1.0 - b) * m


def phagocytosis_probability(
    baseline: float,
    pdl1_expression: float,
    checkpoint_block: float = 0.0,
    max_suppression: float = DEFAULT_PD1_MAX_SUPPRESSION,
) -> float:
    """Per-encounter probability that a macrophage clears this target."""
    return _clamp01(baseline) * pd1_attenuation(
        pdl1_expression, checkpoint_block, max_suppression,
    )


def phagocytosis_outcome(
    baseline: float,
    pdl1_expression: float,
    checkpoint_block: float = 0.0,
    max_suppression: float = DEFAULT_PD1_MAX_SUPPRESSION,
) -> PhagocytosisOutcome:
    """Same as ``phagocytosis_probability`` with the full decomposition."""
    p = _clamp01(pdl1_expression)
    b = _clamp01(checkpoint_block)
    m = _clamp01(max_suppression)
    base = _clamp01(baseline)
    effective = p * (1.0 - b)
    retained = 1.0 - effective * m
    return PhagocytosisOutcome(
        pdl1_expression=p,
        checkpoint_block=b,
        max_suppression=m,
        effective_signal=effective,
        function_retained=retained,
        baseline=base,
        probability=base * retained,
    )


def _clamp01(x: float) -> float:
    if x is None:
        return 0.0
    v = float(x)
    if math.isnan(v):
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v
