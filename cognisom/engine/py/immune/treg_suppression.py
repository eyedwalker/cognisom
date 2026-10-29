"""
Regulatory T cells suppress the cytolytic machinery
===================================================

``treg_fraction`` has been emitted by ``twin_config.to_simulation_params``
and read by nothing since it was written. The comment at
``immune_module.py`` line 481 said so outright: "They stay producer-only,
and that is a known gap rather than an oversight." This module supplies
the missing CONSUMER, and it is deliberately NOT a fourth copy of the
arginase / sympathetic / lactate gate shape applied to a different cell
name.

What is and is not connected, stated exactly
--------------------------------------------
Half the gap is closed, and saying "the gap is closed" would be the
repository's own dominant defect class -- a capability implemented,
tested, and never passed in -- dressed up as a fix. Measured by grep,
2026-10-28:

* CONNECTED. ``ImmuneModule`` reads ``treg_fraction`` and
  ``t_cell_initial_exhaustion`` from its config, and a config key
  reaches it from a run through ``EngineRunner(overrides={'immune':
  {'treg_fraction': ...}})``, which the tests exercise. So a run CAN
  have a Treg load, and ``get_state`` reports the realised multiplier.
* NOT CONNECTED. No shipped preset sets either key (``engine_runner``'s
  SCENARIOS do not, and ``_to_slab`` builds the immune dict from
  SCENARIOS plus ``_IMMUNE_KEYS``), no dashboard control writes them,
  and ``DigitalTwinConfig.to_simulation_params()`` -- the producer --
  has exactly two non-test callers, both in
  ``dashboard/_pages/28_digital_twin.py``: a ``st.download_button`` and
  a ``st.json`` expander. ``PatientEngineInputs.module_configs()``, the
  one path that injects patient data into module configs, emits neither
  key. So there is no route from a PATIENT's measured profile to a run,
  and every run that can be launched from the dashboard today still has
  ``treg_mult == 1.0`` and zero pre-exhausted clones.

The missing link is one line -- merging
``twin.to_simulation_params()['immune']`` into ``EngineRunner``'s immune
config where ``patient`` is handled -- and it was not written here
because nothing in this platform MEASURES a Treg fraction for any
patient (see :func:`treg_occupancy`), so the merge would carry None
today and the test for it would assert the no-op. The honest state is a
consumer waiting on a producer that has no data, not a closed loop.

What was measured first, and why it decided the design
------------------------------------------------------
53,388 T-cell contacts were instrumented across 27 slab runs (Baseline,
Aggressive Tumor and Immunotherapy, 24 h, shipped and all-T-cell
infiltrates) before any of this was written, to find out whether the
T-cell kill path has any room left for another suppressor at all. The
raw rows and both analyses are in the phase-3 scratch directory. What
they say:

* The path is NOT at a floor. The smallest per-contact kill probability
  anywhere in 53,388 contacts is 0.007775, and NOT ONE contact sits
  below 0.001. So the "it will be invisible because everything is already
  zero" failure mode does not apply here.
* The ARGINASE gate, by contrast, IS at its floor wherever the shipped
  presets put macrophages: multiplier mean 0.4057 against a floor of
  0.400 in Baseline, with the local M2 burden already exactly 1.0 on
  98.39% of contacts (Immunotherapy 99.43%). That is what ruled MDSCs
  out -- see "What was NOT built" below.
* Exhaustion never fires. The exhaustion multiplier was exactly 1.0 at
  every one of the 53,388 contacts, because the 10-day clock
  (EXHAUSTION_ONSET_HOURS) cannot elapse in a 24 h run.
* Sympathetic and lactate were exactly 1.0 at every contact, in every
  preset, confirming both are inert by default.

The mechanism, and why it is not the costimulation input
--------------------------------------------------------
The engine's kill rule already takes a costimulation input, so lowering
effective costimulation looked like the way to express a Treg without a
new multiplier. The measurement killed that option, and the reason is a
defect in the existing rule rather than anything about Tregs:

``tcell_kill.kill_probability`` adds an ICB-rescue term
``(1 - costimulation) * checkpoint_block * 0.5``, which is largest when
costimulation is ZERO. Since ``affinity * mhc1`` measures ~0.28 in a
real run and the rescue slope is 0.5, the combined signal starts RISING
as costimulation falls once ``checkpoint_block > 0.56``. Projected on
the measured contacts, routing a Treg through costimulation gives
-27.4% expected perforin hits in Baseline (checkpoint_block 0) but
+29.5% in the Immunotherapy preset (checkpoint_block 1.0): under
blockade the model would make an anergic T cell kill BETTER than a
costimulated one, and a Treg would read as a pro-immune intervention.

That is backwards, and the literature says which way round it should be:
PD-1 "suppresses T cell function primarily by inactivating CD28
signaling" (Hui 2017, PMID 28280247) and "Rescue of exhausted CD8 T
cells by PD-1-targeted therapies is CD28-dependent", with "a
cell-intrinsic requirement of CD28 for CD8 T cell proliferation after
PD-1 blockade" (Kamphorst 2017, PMID 28280249). ICB rescue should scale
WITH costimulation, not with its absence. Rewriting that term changes
the flagship Immunotherapy preset and is not this change; it is
recorded here as a found defect, and the costimulation route stays
unused until it is fixed.

So this module uses the mechanism the primary Treg paper actually
measured. Chen 2005 (PMID 15623559) found Tregs abolish CD8-mediated
tumour rejection by suppressing cytotoxicity SPECIFICALLY -- "CD8 cells
expand to the same extent and produce similar levels of IFN-gamma in
the presence or absence of Ag-specific Treg" -- and that the suppression
requires TGF-beta signalling in the CD8 cell, since a dominant-negative
TGF-beta receptor made the CD8 cells resistant with "unimpaired
cytotoxicity". The downstream target is named by Thomas & Massague 2005
(PMID 16286245): "TGF-beta acts on cytotoxic T lymphocytes (CTLs) to
specifically inhibit the expression of five cytolytic gene products --
namely, perforin, granzyme A, granzyme B, Fas ligand, and interferon
gamma".

PERFORIN is the quantity this engine's T-cell kill path counts. A
contact delivers Poisson(hits_per_contact x Hill) perforin hits and the
target dies when decayed damage reaches 1.84 (Weigelin 2021, see
kill_calibration.py). A Treg therefore acts on the hit rate -- the
granule content a contact can deliver -- and not on recognition, which
is what Chen measured being left intact.

Why it does not share a gate with anything, and why TAM TGF-beta is
not wired into it
-------------------------------------------------------------------
Each suppressor on this path now uses a distinct, named mechanism:
arginase = extracellular L-arginine supply (Rodriguez 2004); sympathetic
= beta2AR / cAMP / PKA; lactate = acidosis; exhaustion = the PD-1-hi
transcriptional state; this one = TGF-beta repression of cytolytic gene
expression. None of them is another's mechanism under a second name.

M2 macrophages in this engine already secrete TGF_beta1 in
``macrophage_polarization.cytokine_output``, and it would be easy to
feed that into this term as well. It is deliberately not fed in. The
M2 compartment already suppresses this exact path through arginase, at
its measured FLOOR (0.400) on 98.39% of shipped Baseline contacts;
adding a second M2-density-driven multiplier would charge one cell
population twice for its suppression, and the measurement says the
second charge would land where there is no room left anyway. The
polarization arm makes the same call for the same reason -- see the
"One IL-10 / TGF-beta term, not two" comment at the
``polarization_target`` call site.

What was NOT built: MDSCs
-------------------------
MDSCs suppress T cells by depleting extracellular L-arginine --
"Arginase production by MDSC depletes arginine from the tumor
microenvironment and impairs T-cell signal transduction and function"
(Ochoa 2007, PMID 17255300; arginase activity raised 6-8 fold in the
PBMC of 117 RCC patients and confined to the CD11b+CD14- subset). That
is the SAME mechanism, from the SAME group's work, as the arginase gate
already on this path: ``arginase_attenuation`` cites Rodriguez 2004
(PMID 15313928), which is the paper showing tumour-associated myeloid
cells "deplete extracellular L-Arg in vitro" via cationic amino acid
transporter 2B. Zea 2005 (PMID 15833831) and Munder 2006 (PMID 16709924)
put the same arginase in granulocytic cells.

An MDSC therefore has to CONTRIBUTE TO the existing arginine burden,
not multiply a second gate. Adding it to that shared sum was projected
exactly on the measured contacts, and it cannot matter in a shipped
preset, because the burden is already saturated:

    preset / arm          contacts at burden 1.0   +1 MDSC    +3 MDSC
    Baseline / shipped            98.39%            -0.29%    -0.60%
    Immunotherapy / shipped       99.43%            -0.09%    -0.11%
    Aggressive / shipped          70.93%            -9.34%   -14.36%

and in Baseline the T-cell arm does 9 of 128 kills, so -0.60% of its
hit rate is roughly -0.04% of the run's killing. Unmeasurable. The
honest figure for a SEPARATE MDSC gate at the same strength as the
arginase floor would have been -60%, which is precisely why
double-counting one mechanism under two cell names is tempting: it
manufactures a large effect out of one paper's single measurement. No
MDSC code was written. The configuration where an MDSC WOULD matter is
a myeloid-excluded tumour -- the measured all-T-cell arms have burden
0.0 at every contact and one MDSC-equivalent there is -20.0% -- and no
preset ships that.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

_CHEN = "Chen 2005, PMID 15623559"
_THOMAS = "Thomas & Massague 2005, PMID 16286245"


@dataclass(frozen=True)
class TregParameter:
    """A number in the Treg suppression model, and what stands behind it."""
    value: float
    low: float
    high: float
    status: str          # 'sourced', 'derived' or 'ESTIMATE'
    source: str

    def as_dict(self) -> Dict[str, object]:
        return {'value': self.value, 'range': (self.low, self.high),
                'status': self.status, 'source': self.source}


#: Treg-per-effector ratio at which cytolytic function is gone.
#:
#: DERIVED, not sourced, and the distinction is the whole point. Chen
#: 2005 measured the RATIO directly in the tumour-draining lymph node on
#: day 7 -- "we found 2.8 +/- 1.2 HA-specific Treg per HA-specific CD8
#: cell on day 7" -- and at that ratio "The presence of HA-specific CD4
#: Treg abolished the specific cytotoxic activity of HA-specific CD8
#: cells (P < 0.0001)". Both quotes are real. But the design is
#: PRESENCE/ABSENCE: it shows cytotoxicity abolished AT a ratio that
#: happened to be 2.8, not that abolition REQUIRES 2.8. Saturation could
#: sit anywhere in (0, 2.8]; nothing in the literature read locates it,
#: which is the same two-point problem TREG_SUPPRESSION_SHAPE records
#: for the curve between the points. Reading 2.8 as the threshold is
#: this model's choice, and it is the CONSERVATIVE one: it pushes full
#: suppression out to treg_fraction 0.737, so a realistic tumour load of
#: 0.2-0.3 retains 0.911-0.847.
#:
#: The bracket therefore is NOT the paper's SD alone. 1.6-4.0 is the
#: observed ratio +/- its SD, which is an uncertainty on the OBSERVATION;
#: the low end of 0.5 is an ESTIMATE of how early saturation could
#: actually occur -- one Treg per two effectors -- and it is there
#: because the SD cannot bound a threshold the design never measured.
#: At ratio_full 0.5 a treg_fraction of 0.3 retains 0.143 instead of
#: 0.847, so the direction the two-point design leaves open is the one
#: that makes this lever large.
#:
#: PROVENANCE. The 2.8 +/- 1.2 is BODY TEXT and is not re-retrievable
#: from PubMed or PMC: efetch on 15623559 returns an abstract that
#: confirms the mechanism verbatim but contains no ratio, efetch
#: db=pmc id=PMC544311 answers "The publisher of this article does not
#: allow downloading of the full text in XML form", and the Europe PMC
#: fullTextXML endpoint errors. The quote came from the article PDF.
#: It is also a mouse, antigen-specific, draining-lymph-node ratio;
#: applying it to a human tumour's Treg-to-effector ratio is an
#: assumption, not the paper's claim.
TREG_RATIO_FULL_SUPPRESSION = TregParameter(
    2.8, 0.5, 4.0, 'derived',
    f"{_CHEN}: 2.8 +/- 1.2 Ag-specific Treg per Ag-specific CD8 in the "
    "tumour-draining LN on day 7, the ratio at which specific "
    "cytotoxicity was abolished (P < 0.0001) -- body text, not in the "
    "abstract and the full text is not open access, so not "
    "re-retrievable from PubMed/PMC. DERIVED because the experiment is "
    "presence/absence: 2.8 is where abolition was OBSERVED, not where "
    "it begins. Bracket: 1.6-4.0 is the paper's own SD on the observed "
    "ratio, 0.5 is an ESTIMATE for how early saturation could occur "
    "(one Treg per two effectors), since a two-point design cannot "
    "locate a threshold")

#: Fraction of cytolytic capacity lost at and above that ratio.
#:
#: 1.0, because "abolished" is what the paper reports, and the
#: mechanism named by Thomas & Massague is repression of perforin and
#: granzyme EXPRESSION -- a cell with no perforin delivers no hits. The
#: magnitude itself is a figure-level readout of in-vivo specific lysis
#: (Chen 2005 Fig. 4), not a printed number, so the ceiling is sourced
#: in direction and completeness but not to two digits.
TREG_MAX_CYTOLYTIC_LOSS = TregParameter(
    1.0, 0.5, 1.0, 'sourced',
    f"{_CHEN}: specific cytotoxic activity abolished, P < 0.0001 "
    f"(Fig. 4, a figure-level readout); mechanism {_THOMAS}: TGF-beta "
    "inhibits expression of perforin, granzyme A, granzyme B, FasL and "
    "IFN-gamma. Low end 0.5 allows for the shape being read off a figure")

#: Shape between no Tregs and the full-suppression ratio. ESTIMATE:
#: Chen 2005 measured TWO points, Tregs absent and Tregs at 2.8:1, so
#: nothing in the literature constrains the curve between them. Linear
#: in the ratio, saturating at 1.0, is the simplest monotone shape that
#: is exactly 0 at zero Tregs -- the same choice, for the same reason,
#: as local_signal, lactate_attenuation and sympathetic_attenuation.
TREG_SUPPRESSION_SHAPE = 'linear-saturating (ESTIMATE: two measured points)'


def treg_effector_ratio(treg_fraction: float) -> float:
    """Tregs per non-regulatory T cell, from a Treg fraction of T cells.

    ``treg_fraction`` is what ``twin_config`` emits, and what
    ``cell_state_classifier.ImmuneComposition.treg_ratio`` computes:
    ``regulatory_t_cells / (cd8 + cd4 + regulatory)``. So the ratio of
    Tregs to everything else in the T-cell compartment is
    ``f / (1 - f)``.

    That denominator includes CD4 helpers, while Chen's ratio is per
    CD8. Dividing by the whole non-Treg pool therefore UNDERSTATES the
    Treg-to-CD8 ratio whenever helpers are present, which makes this
    the conservative direction: it reports less suppression, not more.
    """
    f = _clamp01(treg_fraction)
    if f >= 1.0:
        return float('inf')
    return f / (1.0 - f)


def treg_occupancy(
    treg_fraction: Optional[float],
    ratio_full: float = TREG_RATIO_FULL_SUPPRESSION.value,
) -> float:
    """How far toward full cytolytic suppression this Treg load is, [0, 1].

    ``None`` -- which is what ``twin_config`` emits for an unmeasured
    immune profile, and the honest default -- returns 0.0, exactly as
    0.0 does. Nothing in this platform measures a Treg fraction for any
    patient today, so the default has to be the no-op.
    """
    if treg_fraction is None:
        return 0.0
    r = treg_effector_ratio(treg_fraction)
    if ratio_full is None or float(ratio_full) <= 0.0:
        return 1.0 if r > 0.0 else 0.0
    if r == float('inf'):
        return 1.0
    return min(1.0, r / float(ratio_full))


def treg_cytolytic_attenuation(
    treg_fraction: Optional[float],
    max_loss: float = TREG_MAX_CYTOLYTIC_LOSS.value,
    ratio_full: float = TREG_RATIO_FULL_SUPPRESSION.value,
) -> float:
    """Fraction of cytolytic capacity a T cell retains at this Treg load.

        1 - occupancy(treg_fraction) * max_loss

    Returns a multiplier in ``[1 - max_loss, 1.0]``, and EXACTLY 1.0 at
    ``treg_fraction`` 0.0 or ``None`` -- so every configuration that
    does not name a Treg fraction, which is every preset and every test
    in this repository, is a provable no-op.

    It scales the perforin HIT RATE of a contact, not the probability
    that the contact is recognised or engaged: Chen 2005 measured CD8
    expansion and IFN-gamma production unchanged and cytotoxicity
    abolished, and Thomas & Massague named perforin and the granzymes
    as the repressed gene products. It is therefore NOT rescuable by
    checkpoint blockade, because TGF-beta acts on the CD8 cell
    downstream of PD-1 -- which makes it distinguishable from PD-L1
    escape by exactly the same test logic that distinguishes MHC-I loss
    from PD-L1 gain.

    Size of this lever, which is the multiplier itself and nothing more.
    Expected perforin hits are EXACTLY linear in it -- immune_module
    multiplies the Hill probability by this factor once, and
    ``kill_calibration.contact_hits`` has mean
    ``hits_per_contact * hill_probability`` -- so the reduction in
    expected hits is ``1 - multiplier`` at every contact, whatever the
    contact's inputs were:

      treg_fraction   0.05    0.1    0.2    0.3    0.5   0.737
      hits            -1.9%  -4.0%  -8.9% -15.3% -35.7%  -100%

    The classifier's own HIGH_TREG threshold of 0.3 lands at -15.3%.
    (CORRECTED 2026-10-28. This table previously read -6.2 / -12.8 /
    -27.4 / -44.0 / -80.1% and attributed -44% to treg_fraction 0.3.
    Those are the figures for the COSTIMULATION route rejected above --
    recomputing it at a representative contact, affinity 0.95, MHC-I
    0.30, costimulation 0.6 scaled by this occupancy, gives -6.7 /
    -13.7 / -29.1 / -46.0 / -81.5%, because the Hill is quartic below
    its EC50 and so multiplies the input by an elasticity of 3.66. The
    implemented term has no such amplification: it is a plain factor on
    p. The repository's own pins always agreed with the corrected row --
    mean p_kill 0.08636 -> 0.07314 at f = 0.3, kills 69 -> 59, perforin
    hits 339 -> 316 -- so the table was the only thing that was wrong.)
    """
    occ = treg_occupancy(treg_fraction, ratio_full=ratio_full)
    m = _clamp01(max_loss)
    return 1.0 - occ * m


def calibration_report(treg_fraction: Optional[float] = None
                       ) -> Dict[str, object]:
    """Every number in this model, for a run to say what it stood on."""
    out: Dict[str, object] = {
        'treg_ratio_full_suppression':
            TREG_RATIO_FULL_SUPPRESSION.as_dict(),
        'treg_max_cytolytic_loss': TREG_MAX_CYTOLYTIC_LOSS.as_dict(),
        'shape': TREG_SUPPRESSION_SHAPE,
        'mechanism': (
            'TGF-beta repression of perforin / granzyme expression '
            f'({_THOMAS}); Treg dependence on TGF-betaR in the CD8 cell '
            f'({_CHEN}). Acts on the perforin hit rate, not on '
            'recognition, and is not ICB-rescuable.'),
        'not_built': (
            'MDSC. Its mechanism is arginine depletion (Ochoa 2007, '
            'PMID 17255300), which is the arginase gate already on this '
            'path (Rodriguez 2004, PMID 15313928), so it would have to '
            'share that gate rather than add one -- and the shared '
            'burden is already 1.0 on 98.39% of shipped Baseline '
            'contacts, so one MDSC-equivalent moves the T-cell hit rate '
            '-0.29%. Not implemented.'),
    }
    if treg_fraction is not None:
        out['treg_fraction'] = float(treg_fraction)
        out['treg_per_effector'] = treg_effector_ratio(treg_fraction)
        out['occupancy'] = treg_occupancy(treg_fraction)
        out['cytolytic_retained'] = treg_cytolytic_attenuation(treg_fraction)
    return out


def _clamp01(x: Optional[float]) -> float:
    if x is None:
        return 0.0
    v = float(x)
    if v != v:          # NaN
        return 0.0
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
