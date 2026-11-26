"""From what a tumour makes to what a blood test reads.

The gap this closes
-------------------
The engine's PSA was the sum of an intracellular pool over the cancer
cells. A clinic never measures that. It measures SERUM PSA: protein the
cells secreted, sitting in blood, cleared with its own half-life. Every
clinical number this platform would like to be compared with -- PSA
fall of 50% or more, time to PSA nadir, time to PSA progression, PSA
doubling time -- is defined on the serum quantity, so without this step
the engine's headline output could not be set beside any of them.
``ar_pathway_parameters`` records the same gap from the other side: the
intracellular degradation rate is not the serum half-life, and "the
mapping from the modelled pool to a serum measurement is undefined".
This module defines it.

What is measured
----------------
Stamey 1987 (N Engl J Med 317:909, PMID 2442609, PubMed abstract read
directly on 2026-11-26): "After radical prostatectomy for cancer, PSA
routinely fell to undetectable levels, with a half-life of 2.2 days."
That is clearance with the source removed at a stroke, which is exactly
the quantity a one-compartment clearance rate needs. The same abstract
says serum PSA "was proportional to the estimated volume of the tumor",
which is the other half of the model: secretion in proportion to what
the tumour holds.

The model, and what it does not know
------------------------------------
One compartment, first-order clearance, secretion proportional to the
tumour's PSA:

    dS/dt = k_clear * (source - S),   k_clear = ln 2 / half-life

written so that S settles AT the source. The secretion rate constant
and the volume of distribution are both unknown here, and both only set
an absolute scale, so they are absorbed: S is in the source's units and
is meaningful only RELATIVE TO ITS OWN STARTING VALUE. This yields a
fold change, never ng/mL. Every clinical criterion that is a ratio (a
50% fall, a 25% rise from nadir, a doubling time) can be read off it;
any criterion with an absolute threshold (PCWG2's "and at least 2 ng/mL
above nadir") cannot, and callers must say so.

The pool starts equal to the source: a tumour that existed before the
run began is assumed to have been secreting long enough to be in
balance. A run that started the pool at zero would show serum PSA
rising for its first week and call it biology.

What this does NOT model: PSA from benign prostate, free against
complexed PSA (which clear at different rates), a secretion rate that
changes with differentiation state, or leakage that depends on tissue
architecture. The filter is slow against a day and fast against a
month: it delays and smooths a 12-week curve by a few days and changes
none of its conclusions. What it changes is the KIND of quantity the
engine reports.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

#: Serum PSA elimination half-life, days. Abstract-level sourcing: the
#: abstract gives the mean only, not the spread or the number of men.
SERUM_PSA_HALF_LIFE_DAYS: float = 2.2
SERUM_PSA_SOURCE = (
    "Stamey 1987, PMID 2442609 (PubMed abstract, read directly): after "
    "radical prostatectomy serum PSA fell with a half-life of 2.2 days"
)
SERUM_PSA_STATUS = "sourced"


def advance_serum(serum: float, source: float, dt_hours: float,
                  half_life_days: Optional[float]) -> float:
    """The serum pool after ``dt_hours`` with the source held constant.

    The exact solution of ``dS/dt = k (source - S)`` over the step, not
    an Euler step, so the result does not depend on how finely the run
    is stepped and cannot overshoot the source however long the step.

    ``half_life_days=None`` means no clearance lag at all: serum IS the
    source, the reading the engine gave before this module existed.
    """
    if half_life_days is None:
        return float(source)
    if half_life_days <= 0.0:
        raise ValueError("a half-life must be positive; None means the "
                         "serum pool tracks the source with no lag")
    if dt_hours <= 0.0:
        return float(serum)
    k = math.log(2.0) / (float(half_life_days) * 24.0)
    return float(source + (serum - source) * math.exp(-k * float(dt_hours)))


def serum_report() -> Dict[str, object]:
    """What the serum step stands on, for a run to say so."""
    return {
        "half_life_days": SERUM_PSA_HALF_LIFE_DAYS,
        "status": SERUM_PSA_STATUS,
        "source": SERUM_PSA_SOURCE,
        "units": "relative to the run's own starting value; not ng/mL",
    }
