"""
Per-contact killing, calibrated
===============================

Every number that decides whether an immune contact kills a tumour cell,
each with its source or marked ESTIMATE with a range. Before this module
NK cells and macrophages killed on 80% of contacts and that number had no
source; on the 12 h slab runs the innate arm did 92-98% of all kills and
cleared every preset but one.

T cells: additive cytotoxicity (sourced)
----------------------------------------
Weigelin et al. 2021, Nat Commun 12:5217 (PMID 34471116, PMC8410835),
imaging CTLs against solid tumour cells in vitro and in live melanoma:

* "only a minority of individual CTL-target cell contacts induced
  apoptosis at first encounter, whereas 60-70% (MEC-1/OVA, BLM) or >90%
  (B16F10/OVA, MV3) of individual conjugations were followed by target
  cell survival"; in vivo ">95% of CTL contacts were transient ... and
  nonlethal", "~5%" kill directly.
* Death integrates sublethal perforin hits from successive CTLs. Their
  Cox model: each perforin event is a unit of damage, damage decays
  exponentially with half-life 56.7 min (95% CI 33.1-112.2), and "3
  serial hits delivered with decay intervals below 50 min discriminate
  between tumor cell death or survival".
* "40% of antigen-specific but nonlethal CTL-target cell contacts showed
  CTL-associated perforin events" (B16F10/OVA).

The paper does not print its hazard coefficient, so the model here is the
threshold form of theirs, built from the two numbers it does print: a
target dies when its decayed damage reaches the damage three hits leave
when they arrive 50 min apart, 1 + 2^(-50/56.7) + 2^(-100/56.7) = 1.84.
Hits per contact are Poisson; the 40% of contacts carrying a perforin
event fixes the mean at -ln(0.6) = 0.51 for a fully engaged, cognate
contact (their OT-I/OVA system). A check that was not fitted: that
Poisson puts 9.4% of single contacts at two or more hits, i.e. lethal on
their own -- against the paper's "<10%" for B16F10 and MV3 in vitro.

The T-cell Hill rule (affinity x MHC-I x costimulation, then the
exhaustion / sympathetic / arginase / lactate gates) now scales the HIT
RATE rather than being a kill probability: at its ceiling it used to give
0.99 per contact, ten times the solid-tumour figure.

NK cells (ESTIMATE)
-------------------
No per-contact kill fraction for primary NK cells against a solid tumour
cell was found in open full text. Srpan et al. 2018, J Cell Biol
(PMID 29967280, PMC6122987): 48 +/- 17% of NK-target interactions ended
in death, against rituximab-coated Daudi B-lymphoma cells -- an
antibody-opsonised blood-cancer target, so an upper anchor. Weigelin's
CTLs killed a blood-cancer-like target on 30-40% of contacts and solid
melanoma lines on <10%, a factor of ~3-4; applied to 0.48 that gives
~0.13-0.16. 0.15, range 0.05-0.48. Prager 2019 (PMID 31270246), Choi &
Mitchison 2013 (PMID 23576740) and Vanherberghen 2013 (PMID 23287857)
report serial-killing kinetics and per-NK-cell kill counts, not a
per-contact fraction.

Macrophages (ESTIMATE)
----------------------
Tumour cells display CD47, which engages SIRP-alpha on macrophages and
blocks phagocytosis unless an antibody opsonises the target or blocks
CD47 (Majeti 2009, PMID 19632179; Willingham 2012, PMID 22451913). The
control-arm phagocytosis those papers measured is shown only in figures,
so the magnitude is not sourced here: 0.05 per contact, range 0-0.2.
The PD-1 gate (Gordon 2017) and polarization factor still compose on top.

Immune composition (population prior)
-------------------------------------
TCGA PRAD, CIBERSORT relative fractions from Thorsson et al. 2018,
Immunity 48:812 (PMID 29628290), file
TCGA.Kallisto.fullIDs.cibersort.relative.tsv (GDC PanCanAtlas). Over 558
primary prostate tumours, medians among the three lineages the engine
models: CD8 T 0.306, NK 0.103, macrophage 0.579; macrophages are M2
0.703 / M1 0.186 (medians of per-tumour ratios). The 195 tumours with a
significant deconvolution (p < 0.05) agree within 0.02. These are
estimates from bulk RNA deconvolution, not counts, and a population
prior, not any patient's measurement.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class KillParameter:
    """A number on the kill path, and what stands behind it."""
    value: float
    low: float
    high: float
    status: str          # 'sourced', 'derived' or 'ESTIMATE'
    source: str

    def as_dict(self) -> Dict[str, object]:
        return {'value': self.value, 'range': (self.low, self.high),
                'status': self.status, 'source': self.source}


_WEIGELIN = "Weigelin 2021, PMID 34471116"

#: Damage half-life of one perforin hit, minutes.
DAMAGE_HALF_LIFE_MIN = KillParameter(
    56.7, 33.1, 112.2, 'sourced', f"{_WEIGELIN} (Cox model, 95% CI)")

#: Interval below which three hits discriminate death, minutes.
HIT_INTERVAL_MIN = 50.0


def lethal_damage(half_life_min: float = DAMAGE_HALF_LIFE_MIN.value,
                  interval_min: float = HIT_INTERVAL_MIN) -> float:
    """Decayed damage three hits leave when they arrive interval_min apart."""
    return 1.0 + sum(2.0 ** (-k * interval_min / half_life_min) for k in (1, 2))


#: Damage at which a T-cell target dies.
LETHAL_DAMAGE = KillParameter(
    lethal_damage(), lethal_damage(DAMAGE_HALF_LIFE_MIN.low),
    lethal_damage(DAMAGE_HALF_LIFE_MIN.high), 'derived',
    f"{_WEIGELIN}: 3 hits <50 min apart, at the half-life and its CI")

#: Mean perforin hits per fully engaged cognate CTL contact.
HITS_PER_CONTACT = KillParameter(
    -math.log(1.0 - 0.40), -math.log(1.0 - 0.25), -math.log(1.0 - 0.40),
    'derived', f"{_WEIGELIN}: perforin events in 40% of antigen-specific "
               "contacts (low end: NLS leakage in 25%)")

#: Per-contact kill probability of an NK cell against a solid tumour cell.
#: This is the FLAT rate, used only by the legacy missing-self step
#: (``nk_graded_missing_self=False``). The graded path uses the ceiling
#: below, because a flat rate and a curve ceiling are different
#: quantities and one constant cannot be both.
NK_KILL_PER_CONTACT = KillParameter(
    0.15, 0.05, 0.48, 'ESTIMATE',
    "Srpan 2018 (PMID 29967280) 0.48 on opsonised lymphoma, scaled by "
    f"the solid/blood-target ratio in {_WEIGELIN}")

#: Ceiling of the GRADED NK curve: the per-contact rate against a
#: maximally stressed, class-I-null target. Not the same quantity as
#: NK_KILL_PER_CONTACT above, which is the rate against a TYPICAL solid
#: target -- and that difference is the whole reason this constant
#: exists. Turning one flat rate into a curve requires choosing where on
#: the curve that rate sits, and nothing measures that, so this is a
#: CALIBRATION CHOICE and is labelled as one.
#:
#: Why 0.48. It is the top of NK_KILL_PER_CONTACT's own range -- Srpan
#: 2018's opsonised-lymphoma figure, i.e. a maximally activated target,
#: which is what a ceiling means. Having picked it for that reason, the
#: curve then passes through **0.139** at the MHC-I the shipped presets
#: actually occupy (0.356 mean, 1 mutation), which reproduces the 0.15
#: flat rate they were calibrated against. That agreement is a
#: coincidence worth stating rather than a derivation: it is why
#: removing the 0.4 cliff does NOT silently re-potentiate the NK arm.
#:
#: Measured, 12 h slab, seeds 11/12/13, NK kills per run:
#:     step 0.4      Baseline [9, 9, 12]    Immunotherapy [22, 17, 17]
#:     graded @0.48  Baseline [7, 9,  9]    Immunotherapy [18, 17, 13]
#:     graded @0.15  Baseline [1, 3,  4]    Immunotherapy [ 6,  4,  9]
#: The last row is what anchoring the ceiling at the typical rate would
#: have cost: a ~3x cut to the arm that does most of the killing,
#: bundled invisibly into a correctness fix.
NK_GRADED_CEILING = KillParameter(
    0.48, 0.15, 0.48, 'calibration-choice',
    "top of NK_KILL_PER_CONTACT's range (Srpan 2018 PMID 29967280, "
    "opsonised lymphoma = maximal activation); chosen so the graded "
    "curve reproduces the flat 0.15 at the MHC-I the presets occupy")

_PRAGER = "Prager 2019, PMID 31270246"
_BHAT = "Bhat & Watzl 2007, PMID 17389917"

#: Targets one NK cell kills before it is spent.
#:
#: Until 2026-11-12 nothing limited this. Each kill cost 2-4 of 63
#: granules, a reserve that is never the binding constraint, so an NK cell
#: beside a nodule of targets killed for as long as the run lasted.
#: Measured on the demo patient (section, Baseline, real division clock):
#: 6 NK cells made 51 of 80 kills in 30 h, 6.8 kills per cell per day,
#: and the tumour was cleared in every arm. That, not the clock, is why no
#: longer run could show a course of disease.
#:
#: What the imaging says (both read in full text):
#:   Prager 2019 tracked 347 IL-2-activated primary human NK cells against
#:   adherent HeLa-CD48 targets for 15-17 h: "an average of 2.53
#:   kills/NK cell" (2.05 on CD95-knockout targets), "the majority of NK
#:   cells killing three or fewer targets", serial killers reaching six.
#:   Resting NK cells: "a maximum of four".
#:   Bhat & Watzl 2007, bulk chromium release: "a single IL-2 activated NK
#:   cell can kill four 221 target cells in a 16 hour assay", which the
#:   authors call the maximal potential and a possible overestimate; by
#:   microscopy "many NK cells only killed once or twice", a few up to 6.
#:
#: These are IL-2-ACTIVATED cells in a dish, so this is the generous end:
#: an NK cell inside a prostate tumour is not known to do better.
#: Vanherberghen 2013 (PMID 23287857, abstract only) found about half of
#: NK cells kill nothing at all; that is NOT applied here, because the
#: 2.53 average may already be over killers only and applying both could
#: count the same heterogeneity twice.
NK_SERIAL_KILLS_MEAN = KillParameter(
    2.53, 2.05, 4.0, 'sourced',
    f"{_PRAGER}: average 2.53 kills per IL-2-activated NK cell over "
    f"15-17 h (2.05 on CD95-KO targets); high end {_BHAT}, 4 in 16 h")

#: The most targets any one NK cell was seen to kill in either study.
NK_SERIAL_KILLS_MAX = 6

#: Hours a spent NK cell takes to kill again.
#:
#: Bhat & Watzl: spent ("exhausted") NK cells survive with reduced
#: perforin and granzyme B, and "the killing ability of these cells was
#: augmented in response to IL-2 treatment on day 2"; Prager restates it
#: as "restored within 2 d by IL-2 or IL-15 treatment". Both are in a
#: dish, away from targets, WITH added cytokine. Nothing measures
#: recovery inside a tumour, where it may be slower or not happen, so 48 h
#: is the fastest the evidence allows and is labelled an estimate.
NK_RECOVERY_HOURS = KillParameter(
    48.0, 48.0, float('inf'), 'ESTIMATE',
    f"{_BHAT}: restored by day 2 with IL-2, after separation from "
    "targets; no in-tissue measurement, so this is the fast end")


def draw_serial_kill_capacity(rng_poisson,
                              mean: float = NK_SERIAL_KILLS_MEAN.value,
                              most: int = NK_SERIAL_KILLS_MAX) -> int:
    """How many targets one NK cell will kill before it is spent.

    At least one, a Poisson spread about the measured mean, never more
    than the most ever observed. With the defaults about 80% of cells
    draw three or fewer, which is the paper's "majority ... three or
    fewer", and the cap moves the mean by under 1%.
    """
    return int(min(most, 1 + rng_poisson(max(0.0, mean - 1.0))))


#: Per-contact phagocytic kill probability of a macrophage.
MACROPHAGE_KILL_PER_CONTACT = KillParameter(
    0.05, 0.0, 0.2, 'ESTIMATE',
    "CD47-SIRPa blocks phagocytosis of unopsonised tumour cells "
    "(Majeti 2009 PMID 19632179; Willingham 2012 PMID 22451913); "
    "magnitude not sourced")

_THORSSON = "Thorsson 2018 (PMID 29628290), TCGA PRAD CIBERSORT, n=558"

#: Share of CD8 T / NK / macrophage among those three, prostate tumours.
#: Medians, so they sum to 0.988; normalise before use.
PRAD_IMMUNE_SHARE: Dict[str, float] = {
    'n_t_cells': 0.306, 'n_nk_cells': 0.103, 'n_macrophages': 0.579,
}
#: M1 / M0 / M2 share of prostate tumour macrophages. Medians of the
#: per-tumour ratios (M2 0.703, M1 0.186); M0 is the remainder.
PRAD_MACROPHAGE_POLARIZATION: Dict[str, float] = {
    'M1': 0.186, 'M0': 0.111, 'M2': 0.703,
}
PRAD_SOURCE = _THORSSON


def split_immune(total: int, share: Optional[Dict[str, float]] = None
                 ) -> Dict[str, int]:
    """Split ``total`` immune cells by ``share``, largest remainder."""
    share = share or PRAD_IMMUNE_SHARE
    norm = sum(share.values())
    exact = {k: total * v / norm for k, v in share.items()}
    out = {k: int(math.floor(x)) for k, x in exact.items()}
    for k in sorted(exact, key=lambda k: exact[k] - out[k], reverse=True):
        if sum(out.values()) >= total:
            break
        out[k] += 1
    return out


def damage_now(damage: float, since_h: float, now_h: float,
               half_life_min: float = DAMAGE_HALF_LIFE_MIN.value) -> float:
    """Damage left at ``now_h`` from ``damage`` recorded at ``since_h``."""
    elapsed_min = max(0.0, (now_h - since_h) * 60.0)
    return damage * 2.0 ** (-elapsed_min / half_life_min)


def contact_hits(hill_probability: float, rng_poisson,
                 hits_per_contact: float = HITS_PER_CONTACT.value) -> Tuple[float, int]:
    """(mean, drawn) perforin hits for one T-cell contact.

    ``hill_probability`` is the T-cell rule's output in [0, 1]; at 1 the
    contact is the fully engaged cognate one Weigelin imaged.
    """
    mean = hits_per_contact * max(0.0, min(1.0, hill_probability))
    return mean, int(rng_poisson(mean)) if mean > 0.0 else 0


def calibration_report() -> Dict[str, Dict[str, object]]:
    """Every kill-path number, for a run to say what it stood on."""
    return {
        'tcell_hits_per_contact': HITS_PER_CONTACT.as_dict(),
        'tcell_damage_half_life_min': DAMAGE_HALF_LIFE_MIN.as_dict(),
        'tcell_lethal_damage': LETHAL_DAMAGE.as_dict(),
        'nk_kill_per_contact': NK_KILL_PER_CONTACT.as_dict(),
        'nk_graded_ceiling': NK_GRADED_CEILING.as_dict(),
        'macrophage_kill_per_contact': MACROPHAGE_KILL_PER_CONTACT.as_dict(),
        'nk_serial_kills_mean': NK_SERIAL_KILLS_MEAN.as_dict(),
        'nk_recovery_hours': NK_RECOVERY_HOURS.as_dict(),
    }
