"""Every rate in the AR/PSA pathway, and what stands behind it.

WHY THIS FILE EXISTS. ``reaction_builder.build_ar_signaling`` is the one
AR pathway the engine runs: ARSignalingModule builds it on every run, so
it produces the signalling fraction that gates division, the androgen
withdrawal hazard, and the PSA the dashboard presents as "the protein a
clinic follows". Nine of its rate constants were inline literals with a
trailing comment and no provenance -- not in the entity library, so not
reachable by the library's curation report either, and not countable
anywhere. ``capabilities.py`` says only 2 of 34 library kinetic
constants survive re-checking; these nine were not even in that 34.

WHAT THIS CHANGES AND WHAT IT DOES NOT. Every ESTIMATE here is the
value that was already running. The one SOURCED rate replaced its
literal only after measuring that no outcome moved. What it buys is that each number now states the
biological quantity it implies, which a domain reader can check without
reading the code: a first-order rate of 0.2 /h IS a 3.5 h half-life, and
saying so turns an opaque literal into a falsifiable claim. Several of
these are almost certainly wrong, and the point of writing down what
they imply is that someone can now see which.

WHY ALMOST NOTHING HERE IS MARKED 'sourced'. An audit on 2026-10-17
ran 96 agents over the library's kinetic constants and had **73% of the
citations fail independent re-retrieval** -- wrong species, wrong value,
wrong enzyme, or unretrievable. Its conclusion stands: a single agent's
literature citation is not evidence, and two agents' is not either. So
a rate is ESTIMATE until someone has read the primary record and can
quote it, the way DAMAGE_HALF_LIFE_MIN and DHT_INTRACRINE_FRACTION were
done. One has been: K_AR_DEGRADATION, whose comment quotes the PubMed
record it came from. The other ten are honest ESTIMATEs rather than
carrying citations that would not survive a check.

A NOTE ON PSA, because it is the most inviting mistake here. The
clinical serum PSA half-life (~2-3 days) is a CLEARANCE rate from
blood. ``k_psa_deg`` is degradation of the modelled intracellular pool.
They are different quantities and substituting one for the other would
be the category error this project keeps cataloguing. The mapping from
the modelled pool to a serum measurement is itself unspecified, which
is recorded below rather than papered over.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

#: Statuses, the same vocabulary kill_calibration uses, so the methods
#: page can count numbers from both in one table.
SOURCED = "sourced"
DERIVED = "derived"
ESTIMATE = "ESTIMATE"
STATUSES = (SOURCED, DERIVED, ESTIMATE)


@dataclass(frozen=True)
class PathwayParameter:
    """One rate in the AR pathway, and what stands behind it.

    ``implies`` is the biological quantity the value amounts to, stated
    so a reader can judge the number without reading the ODE. For a
    first-order rate it is filled in automatically as a half-life.
    """

    name: str
    value: float
    unit: str
    status: str
    source: str
    first_order: bool = True
    note: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r} for {self.name}")
        if not self.source:
            raise ValueError(
                f"{self.name} has no source text; an ESTIMATE must still "
                "say what it was estimated from or that it was not")

    @property
    def half_life_hours(self) -> Optional[float]:
        """The half-life a first-order rate per hour amounts to."""
        if not self.first_order or self.value <= 0:
            return None
        if not self.unit.endswith("/h"):
            return None
        return math.log(2.0) / self.value

    def as_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "value": self.value, "unit": self.unit, "status": self.status,
            "source": self.source,
        }
        half = self.half_life_hours
        if half is not None:
            out["implies"] = f"half-life {half:.2f} h"
        if self.note:
            out["note"] = self.note
        return out


_UNSOURCED = ("Inline literal carried over from the first AR pathway; "
              "no source was recorded and none has been verified since")

# ── AR protein ─────────────────────────────────────────────────────

#: Protein copies per mRNA per hour.
K_AR_TRANSLATION = PathwayParameter(
    "k_ar_tl", 20.0, "protein/mRNA/h", ESTIMATE, _UNSOURCED,
    first_order=False,
    note="Not a half-life. Sets the AR protein pool's scale, and the "
         "signalling fraction is normalised to the eugonadal steady "
         "state, so this cancels from the FRACTION while setting the "
         "absolute pool the ODE integrates.")

#: AR protein degradation. The one rate here that is SOURCED, and the
#: record of how, because the route matters as much as the number.
#: Gregory et al. 2001, Cancer Res 61:2892, PMID 11306464, PubMed
#: record read directly on 2026-11-24: "In the absence of androgen, AR
#: in androgen-dependent LNCaP cells was unstable with a degradation
#: half-time (t(1/2)) of 3 h at 37 degrees C. In contrast, AR was 2-4
#: times more stable in recurrent CWR22 tumors (t(1/2), >12 h) and
#: CWR-R1 or LNCaP-C4-2 cell lines (t(1/2), 6-7 h)."
#:
#: The value that ran before, 0.1 /h, implies 6.9 h -- the paper's
#: figure for RECURRENT lines, in an engine whose default tumour is
#: androgen-dependent. MEASURED before changing it: 0.1 against
#: ln2/3 gives identical cell counts in untreated, castrate and
#: enzalutamide ring runs at 72 h, with tumour PSA moving in the fifth
#: decimal under castration only. The complex is supply-bound with AR
#: in excess, so this rate is not load-bearing -- which is why it could
#: be corrected without moving a pinned outcome, and also why getting
#: it right buys fidelity of the AR pool rather than of any outcome.
#: One thing did move: faster turnover leaves 2.3x less free AR
#: (43,274 -> 18,729 at steady state), so the supply-bound regime is
#: shallower. A 100-fold affinity cut now shifts the complex 0.24%
#: where it shifted 0.10% -- still inert, and re-pinned with that
#: reason in test_an_antiandrogen_acts_downstream.
#:
#: Abstract-level sourcing: the full text is not in PMC, so the method
#: (the abstract does not say pulse-chase or cycloheximide) is unread.
K_AR_DEGRADATION = PathwayParameter(
    "k_ar_deg", math.log(2.0) / 3.0, "/h", SOURCED,
    "Gregory 2001, PMID 11306464 (PubMed abstract, read directly): "
    "unliganded AR half-time 3 h in androgen-dependent LNCaP at 37 C",
    note="Stands for unliganded AR in androgen-DEPENDENT cells. The "
         "same paper measures 6-7 h and >12 h in lines from recurrent "
         "tumours, so a castration-resistant run should not inherit "
         "this value; the model has no state-dependent stabilisation.")

# ── DHT and the complex ────────────────────────────────────────────

#: Interstitial DHT delivered per hour, in the ODE's units.
DHT_INPUT = PathwayParameter(
    "DHT_input", 1.0, "au/h", ESTIMATE,
    "A normalisation, not a measurement: 1.0 defines the eugonadal "
    "reference the signalling fraction is divided by",
    first_order=False,
    note="Changing it rescales the reference and the fraction together, "
         "so it is the one number here that is arbitrary BY DESIGN. "
         "Tissue DHT itself is sourced separately -- see "
         "DHT_INTRACRINE_FRACTION in cellular_module.")

K_DHT_DEGRADATION = PathwayParameter(
    "k_dht_deg", 0.2, "/h", ESTIMATE, _UNSOURCED,
    note="Implies a 3.5 h DHT half-life in the cell. This competes "
         "with the spatial DHT field, which has its own supply term, so "
         "the two together set tissue DHT.")

K_COMPLEX_DEGRADATION = PathwayParameter(
    "k_complex_deg", 0.05, "/h", ESTIMATE, _UNSOURCED,
    note="Implies a 13.9 h AR-DHT complex half-life -- deliberately "
         "longer than free AR above, which is the right direction "
         "(ligand binding stabilises the receptor) but an unsourced "
         "magnitude.")

# ── PSA ────────────────────────────────────────────────────────────

K_PSA_TRANSCRIPTION = PathwayParameter(
    "k_psa_tx", 2.0, "mRNA/h", ESTIMATE, _UNSOURCED, first_order=False,
    note="Vmax of the AR-driven KLK3 transcription term, whose Km is "
         "AR_ARE_APPARENT_KD_NM and IS sourced. An unsourced Vmax on a "
         "sourced Km sets the curve's height from nothing.")

K_PSA_MRNA_DEGRADATION = PathwayParameter(
    "k_psa_m_deg", 1.0, "/h", ESTIMATE, _UNSOURCED,
    note="Implies a 42-minute KLK3 mRNA half-life, which is short for a "
         "secreted-protein transcript and is the value most worth "
         "checking first.")

K_PSA_TRANSLATION = PathwayParameter(
    "k_psa_tl", 30.0, "protein/mRNA/h", ESTIMATE, _UNSOURCED,
    first_order=False)

K_PSA_DEGRADATION = PathwayParameter(
    "k_psa_deg", 0.2, "/h", ESTIMATE, _UNSOURCED,
    note="Implies a 3.5 h half-life for the modelled intracellular PSA "
         "pool. This is NOT the clinical serum PSA half-life of ~2-3 "
         "days, which is renal/hepatic CLEARANCE of secreted protein "
         "from blood -- a different quantity. The engine has no "
         "secretion or clearance step, so the mapping from this pool to "
         "a serum measurement is undefined; the dashboard's PSA is a "
         "relative index, not a ng/mL prediction.")

#: Fallbacks used when the entity library carries nothing. They are
#: listed because a fallback that fires is as load-bearing as a
#: literal, and these have no provenance either.
K_AR_TRANSCRIPTION_FALLBACK = PathwayParameter(
    "k_ar_tx (fallback)", 5.0, "mRNA/h", ESTIMATE,
    "Fallback when the AR entity carries no "
    "transcription_rate_mrna_per_hour", first_order=False)

K_AR_MRNA_DEGRADATION_FALLBACK = PathwayParameter(
    "k_ar_m_deg (fallback)", 0.5, "/h", ESTIMATE,
    "Fallback when the AR entity carries no half-life in any of the "
    "unit-suffixed names the resolver reads",
    note="Implies a 1.4 h AR mRNA half-life.")

#: Every parameter here, in the order a reader should meet them.
ALL: tuple = (
    K_AR_TRANSLATION, K_AR_DEGRADATION,
    DHT_INPUT, K_DHT_DEGRADATION, K_COMPLEX_DEGRADATION,
    K_PSA_TRANSCRIPTION, K_PSA_MRNA_DEGRADATION, K_PSA_TRANSLATION,
    K_PSA_DEGRADATION,
    K_AR_TRANSCRIPTION_FALLBACK, K_AR_MRNA_DEGRADATION_FALLBACK,
)


def pathway_report() -> Dict[str, Dict[str, object]]:
    """Every AR-pathway rate, for a run to say what it stood on.

    The same shape as ``kill_calibration.calibration_report`` so one
    table can carry both.
    """
    return {p.name: p.as_dict() for p in ALL}


def status_counts() -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for p in ALL:
        counts[p.status] = counts.get(p.status, 0) + 1
    return counts


def main() -> int:  # pragma: no cover
    import json
    print(json.dumps(pathway_report(), indent=2))
    print(json.dumps(status_counts(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
