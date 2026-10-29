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
checkpoint_block relieves PD-1 suppression of the COSTIMULATION arm,
multiplicatively, so it can only amplify a signal that already exists.
The final probability passes through a Hill curve with adjustable
threshold and slope so the rule can express "all-or-nothing" or
"graded" regimes.

Why blockade multiplies costimulation instead of adding a floor
---------------------------------------------------------------
It used to add one::

    signal = a * m * c + (1 - c) * checkpoint_block * 0.5

That term reads no affinity and no MHC-I, so blockade paid out against
a cell the T cell cannot see: at affinity 0.95 and costimulation 0.6,
full blockade took a cell at MHC-I 0.20 from p 0.020 to 0.545 (26.7x)
while an intact cell moved 0.929 to 0.977 (1.1x). The escape penalty
p(mhc1=0.2)/p(mhc1=1.0) inverted from 0.022 to 0.558 -- therapy
rewarded MHC-I loss. A cell at MHC-I 0.0 still scored 0.165, and the
only thing standing between that and a kill was the separate 0.05
recognition gate in ImmuneModule._recognize. The term was also largest
at ZERO costimulation, so past checkpoint_block 0.551 suppressing CD28
RAISED the kill signal, which is why a Treg acting on costimulation
projected as -27.4% in Baseline but +29.5% under blockade.

The literature says the opposite on both counts:

* Hui E et al., Science 2017;355(6332):1428-1433 (PMID 28280247). In a
  biochemical reconstitution system PD-1-recruited Shp2 preferentially
  dephosphorylates the coreceptor CD28 rather than the TCR, and the
  authors' conclusion is the operative sentence here: "PD-1 suppresses
  T cell function primarily by inactivating CD28 signaling." PD-1 acts
  ON the costimulation arm, so relieving it restores CD28.
* Kamphorst AO et al., Science 2017;355(6332):1423-1427 (PMID
  28280249), titled "Rescue of exhausted CD8 T cells by PD-1-targeted
  therapies is CD28-dependent". CD28 is required cell-intrinsically
  for CD8 proliferation after PD-1 blockade; "the CD28/B7
  costimulatory pathway is essential for effective PD-1 therapy", and
  "CD8 T cells proliferating in blood after PD-1 therapy of lung
  cancer patients were predominantly CD28-positive". Rescue therefore
  scales WITH available costimulation and must vanish as costimulation
  does -- which an additive floor cannot express.

What blockade deliberately does NOT read here: the target cell's
``pdl1_expression``. Klement JD et al., Cancer Cell 2023;41(3):620-636.e9
(PMID 36917954) deleted Cd274, which encodes PD-L1, in 4T1 and CT26.
Quoting the open preprint of that study (bioRxiv 2021.06.18.449028,
because the published text is paywalled): "tPD-L1 expression conferred
no protection against ... cytotoxicity in the 4T1 or CT26 at early or
late time points", and "This phenomenon was not rescuable by
pretreatment with IFNgamma to elevate surface PD-L1 expression" --
which is exactly the adaptive induction path this engine models
(ImmuneModule adds 0.5 to the target's pdl1_expression per
recognition, citing IFN-gamma). Primary tumour growth was likewise
unchanged: "no difference was observed in primary tumor growth,
supporting our in vitro cytotoxicity results". Tumour PD-L1 in that
study acts through MYELOID PD-1, not by shielding the tumour cell from
a CTL. So the macrophage path reads pdl1_expression
(macrophage_phagocytosis, Gordon 2017) and the T-cell path must not;
the asymmetry is the finding, not an oversight.

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

#: Fold-increase in the effective CD28 costimulation signal at full PD-1
#: blockade: c_eff = c * (1 + PD1_CD28_RELIEF * checkpoint_block), capped
#: at 1.0 because nothing can signal above full costimulation.
#:
#: ESTIMATE, range 0.5-2.0. Hui 2017 (PMID 28280247) establishes that
#: PD-1-Shp2 dephosphorylates CD28 and that blockade therefore restores
#: it, and Kamphorst 2017 (PMID 28280249) that the restoration is
#: CD28-dependent, but neither prints a fold-change in CD28 signal
#: strength that this scalar could be set from. 1.0 is the round
#: statement "full blockade at most doubles the CD28 signal"; at the
#: shipped costimulation of 0.6 the cap binds, so full blockade restores
#: costimulation exactly to its 1.0 ceiling and the preset does not
#: depend on the precise value above 0.667.
#:
#: Being multiplicative in c is the load-bearing property, not the
#: magnitude: it makes rescue vanish as costimulation vanishes
#: (Kamphorst) and makes rescue impossible where recognition is absent,
#: since the whole signal is a * m * c_eff. Any value here leaves
#: checkpoint_block = 0 algebraically identical to no rescue at all,
#: which is every shipped preset except Immunotherapy.
PD1_CD28_RELIEF: float = 1.0


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
    #: Costimulation after PD-1 relief, i.e. what the signal was actually
    #: built from. Equals `costimulation` whenever checkpoint_block is 0
    #: or the clone is exhausted. Reported because blockade now acts here
    #: rather than as a separate additive term, so this is the only place
    #: its effect is visible in the decomposition.
    effective_costimulation: float = 0.0


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

        c_eff  = min(1, costimulation * (1 + PD1_CD28_RELIEF * block))  [precursor only]
        signal = affinity * mhc_level * c_eff

    Checkpoint rescue is GATED on the engaging T cell being a PD-1-lo
    precursor; exhausted PD-1-hi clones are not rescuable by ICB
    (Dolina et al. 2021, lecture slide 52), and Kamphorst 2017 reaches
    the same place from the CD28 side. Without this gate, the formula
    would imply ICB rescues every dysfunctional T cell, which is the
    rule-based-only error the Stage-D exhaustion model is specifically
    built to correct.

    Because rescue multiplies costimulation rather than adding to the
    signal, it cannot manufacture a kill where there is nothing to
    recognise: affinity 0 or MHC-I 0 gives signal 0 at any blockade.
    See the module docstring for the three papers and for the measured
    behaviour of the additive term this replaced.

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

    signal, _ = _combined_signal(a, m, c, b, is_exhausted)

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
    signal, c_eff = _combined_signal(a, m, c, b, is_exhausted)
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
        effective_costimulation=c_eff,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _combined_signal(
    a: float,
    m: float,
    c: float,
    b: float,
    is_exhausted: bool,
    relief: float = PD1_CD28_RELIEF,
) -> tuple:
    """(signal fed to the Hill curve, effective costimulation).

    One implementation so ``kill_probability`` and ``kill_outcome``
    cannot drift apart -- they each carried their own copy of this
    arithmetic, which is how the decomposition a test reads could have
    disagreed with the probability the engine acts on.

    Exhausted clones get no PD-1 relief: their costimulation passes
    through untouched.
    """
    if is_exhausted:
        c_eff = c
    else:
        c_eff = min(1.0, c * (1.0 + max(0.0, relief) * b))
    signal = max(0.0, min(1.0, a * m * c_eff))
    return signal, c_eff


def _clamp01(x: float) -> float:
    if x is None or math.isnan(x):
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)
