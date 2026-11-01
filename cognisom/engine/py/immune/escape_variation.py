"""
Heritable immune-escape variation, calibrated
=============================================

An established tumour is by definition one that escaped. Before this
module nothing in the engine could escape, because nothing varied:
``CellularModule`` seeded EVERY cancer cell at ``mhc1_expression = 0.3``
(three sites) and ``_divide_cell`` copied ``mhc1_expression``,
``mhc1_baseline`` and ``pdl1_expression`` to the daughter exactly. With
no heritable variation immune pressure has nothing to select on, so
escape could only be imposed by hand, never observed. This module holds
the per-division probabilities and magnitudes that give it something,
each with its source or marked ESTIMATE with a range and a reason.

The two routes are not interchangeable
--------------------------------------
``ImmuneModule._recognize`` returns ``(False, None)`` when
``mhc1_expression < 0.05``: a cell that has lost presentation is
INVISIBLE to T cells and checkpoint blockade cannot rescue it, because
blockade acts on killing a cell that IS recognised. So:

* PARTIAL loss -- one HLA ALLELE gone, the rest of the panel still
  presenting. The cell stays recognised, at a lower signal, and blockade
  still works. McGranahan 2017's own title is "Allele-Specific HLA Loss
  and Immune Escape in Lung Cancer Evolution", which is the quantity
  implemented here, and it is the route the measured rate below belongs
  to (de Nooij's assay selects for loss of ONE allele, HLA-A2).
* TOTAL loss -- no surface class I at all (B2M loss of function being
  the canonical route). The cell drops out of the T-cell loop entirely.
  No per-division rate for this exists in the literature; see below.

Modelling them as one event with one magnitude would have collapsed
exactly the distinction a later acceptance test turns on.

ALLELE, not HAPLOTYPE, and the difference is a factor of three
--------------------------------------------------------------
Until 2026-10-28 this event was NAMED a haplotype loss
(``TRAIT_HAPLOTYPE_LOSS``, ``_lose_one_haplotype``) while being
IMPLEMENTED as the loss of one allele drawn from the panel, and the
magnitude was pinned to the implementation. A class I haplotype carries
one HLA-A, one -B and one -C allele in cis, so a true haplotype LOH on a
6-allele genotype removes 3 of 6 -- surface fraction 0.5, and the
repertoire filter would drop peptides on three loci. One allele of six
leaves 5/6. Three times the surface cost, and a different set of
peptides.

The name was wrong, not the code, so the name moved: the trait is
``TRAIT_ALLELE_LOSS`` = 'mhc1_allele_loss' and the rate is
:data:`MHC1_ALLELE_LOSS_PER_DIVISION`. The implemented magnitude,
(n-1)/n, is a PER-ALLELE figure and must not be quoted as a haplotype's.
Modelling cis phase instead would mean grouping ``hla_alleles`` by
parental chromosome and dropping all of one group, and the engine's
panels carry no phase information at all -- the default is three alleles
(HLA-A*02:01 / A*24:02 / B*07:02), which is not even one full
haplotype -- so the single-allele draw stands as the conservative
stand-in and is now labelled as one.

The rate: five reported values, four papers, two laboratories
-------------------------------------------------------------
A per-cell-per-generation rate for loss of one HLA class I allele has
been reported five times, and the values agree to within a factor of
14 (:data:`MHC1_LOSS_RATE_MEASUREMENTS`):

  5.0e-7  Pious 1980, PMID 6969452      RESTATED, not measured: "which
                                        we previously estimated at
                                        about 5 x 10(-7) per cell per
                                        generation". What that paper
                                        measures is the <10%
                                        contribution of extended
                                        lesions, in pseudodiploid T5-1
  7.9e-7  de Nooij-van Dalen 1999,      HLA-A2, line R83-4915
          PMID 10029664
  8.0e-7  Pious & Soderland 1977,       HLA-B27, diploid lymphoid line
          PMID 70075
  4.1e-6  de Nooij-van Dalen 1998,      HLA-A2, TK6
          PMID 9443039
  6.9e-6  de Nooij-van Dalen 1998,      HLA-A2, WI-L2-NS
          PMID 9443039

:data:`MHC1_ALLELE_LOSS_PER_DIVISION` is their geometric mean,
1.55e-6 per cell per division, bracketed by the extremes. The 5-10x
spread between the 1998 and 1999 papers is the same group and the same
assay, and they attribute it to donor genetic background and
transformation status rather than to method -- which is the reason the
range is the range.

The aggregation is NOT five independent determinations, and that is
worth stating because the geometric mean is quoted as if it were. Four
papers, two groups: Pious contributes the 1977 measurement and the 1980
restatement of the same group's earlier estimate, and de Nooij-van Dalen
contributes 3 of the 5 values -- including both values above 1e-6, which
is what carries the mean up. Dropping the restated 5.0e-7 entirely gives
2.06e-6 (+33%), inside the same 5e-7 to 6.9e-6 bracket, so the headline
number is insensitive to the one non-independent entry; its
independence, not its value, is the caveat.

What the rate is NOT. (1) All five numbers come from non-malignant human
lymphoblastoid or lymphoid lines in vitro. Tumours carry elevated
chromosomal instability, so for a cancer cell this is a LOWER bound; no
measurement in a tumour line or in vivo was found. (2) It is the rate of
losing ONE ALLELE, leaving the rest of the panel presenting -- not the
rate of total MHC-I loss, and not the rate of losing a whole haplotype
(three class I alleles in cis), which nothing read here measures.
(3) The mechanism differs by background: 84%
of de Nooij's mutants showed 6p LOH (mitotic recombination 33-50%, or
chromosome loss with duplication of the remaining chromosome 34-40%),
whereas Pious 1980 found extended lesions of 0.8 cM or longer contribute
<10% of the rate in a near-diploid line. The same rate is reached by
different mechanisms, so it must not be called a 6p-LOH rate without
naming the paper. (4) Four of the five papers have no PMCID and are not
open access: every rate here comes from the PubMed abstract via efetch,
re-retrievable by any skeptic, but confidence intervals, generation
counts and the number of independent cultures behind each rate could not
be checked.

Why the prevalence figures are a target and not a rate
------------------------------------------------------
McGranahan 2017 (PMID 29107330, TRACERx) finds HLA LOH in 40% of NSCLCs,
subclonal in 13/17 lung adenocarcinomas and 9/17 squamous carcinomas,
and in an independent cohort in 17/37 (46%) tumours, subclonal in 11/17
(65%) of those. Turning that into a per-division rate would need the
tumour cell number at sampling, the subclone detection threshold and the
variant's selection coefficient, under an assumption of neutrality --
and the paper's own evidence contradicts neutrality: "The focal nature
of HLA LOH alterations, their subclonal frequencies, enrichment in
metastatic sites, and occurrence as parallel events suggests that HLA
LOH is an immune escape mechanism that is subject to strong
microenvironmental selection pressures later in tumor evolution."
Focal HLA LOH occurs more often than chance predicts (p < 0.001), and in
four tumours the SAME allele was lost on distinct phylogenetic branches.
A prevalence produced by selection cannot be read back as a mutation
supply. So the prevalence is the calibration TARGET and never the rate.

The scale problem, stated before anyone is surprised by it
----------------------------------------------------------
Only a CANCER daughter draws. ``CellularModule._maybe_escape`` returns
None for every other cell type, so the denominator of every expectation
below is cancer divisions and NOT total divisions -- a distinction worth
14x on the slab and total on the ring, and one these notes previously
got wrong in both directions.

This arithmetic is ours, from the cited rate, and it is simple enough to
check. At mu = 1.55e-6 one expected event needs 1/mu = 6.5e5 cancer
divisions. A macroscopic 1e9-cell tumour has had ~1e9 of them, so ~1,550
events (500-6,900 across the bracket): mutation supply is abundant and
prevalence is governed by selection and drift, not by mu -- which is
also why recurrent independent loss of the same allele is expected.
But at this repo's preset scale the de novo route is SILENT. De novo
escape will not be seen in a short preset run at the literature rate,
and that is the correct behaviour, not a bug. Two consequences:

* the item's own premise is the way out -- an established tumour has
  already escaped, so a founder fraction can be seeded from the
  prevalence data (:data:`FOUNDER_HLA_LOH_FRACTION`, default 0.0, see
  below) rather than waited for;
* the per-division mechanism is implemented anyway at the sourced rate,
  so it is correct and scales with n_cells. A run that must show de novo
  escape scales mu DELIBERATELY and labels the scaling, with the
  division count above as the reason. mu is never quietly raised to make
  a demo work.

Measured on the shipped presets, 2026-10-28, counting divisions BY
PARENT TYPE:

  run (24 h, founder 0.5)   total div   CANCER div   expected   observed
  ring Baseline  seed 12345        35            0    0            0
  slab Baseline  seed 1 (PWM)     849           56    8.7e-5       0
  slab Baseline  seed 1 (flurry)  846           53    8.2e-5       0
  slab Aggressive seeds 1-3    ~930-980      226-238  3.5e-4 to    0
                                                      3.7e-4

The ring preset's cancer cells do not divide at all in 24 h, so a run on
the ring NEVER REACHES the draw; the quoted "974 divisions expect 1.5e-3
events" was 18x the true expectation and the ring's was not 5.4e-5 but
exactly zero. The conclusion is unchanged and stronger: zero observed de
novo events is correct, and at 56 cancer divisions a run would need
~11,000x more of them for one expected event.

And is the seeded variation then SELECTED? Not resolvably, at the
population, in the shipped infiltrate
-------------------------------------------------------------------------
This is the item's own acceptance claim ("surviving cells are measurably
enriched for escape traits compared with the starting population"), it is
empirical, and the shipped presets do NOT show it. 12 seeds of the
Baseline slab, 24 h, half the 80 founders seeded with one-allele loss,
against a drift control with the immune module removed. delta is the
escaped fraction among survivors minus the escaped fraction among
founders:

  immune removed (drift)   +0.0062  sd 0.0093  95% CI [+0.0003, +0.0121]
  shipped infiltrate       +0.0043  sd 0.0253  95% CI [-0.0118, +0.0203]
  difference               -0.0020 +- 0.0078,  t = -0.26

"Not shown" is the honest verdict and "false" is not, because this
design cannot tell the predicted effect from zero. The contact-level
kill data below predicts the size of the population shift: 4.82 escaped
cells spared over 12 runs, 0.40 per run against ~115 surviving cancer
cells, i.e. +0.0035 in the escaped share -- a THIRD of the +-0.0078
standard error on the contrast, whose 95% interval [-0.018, +0.014]
contains both 0 and +0.0035. Resolving +0.0035 at 2 sigma needs the se
below 0.00175, which at the shipped arm's measured per-run sd of 0.0253
is ~240 seeds per arm, 20x this design. The noise floor is dominated by
the TREATMENT arm, not the control: 0.0253 against the drift arm's
0.0093, so immune killing is what makes the measurement noisy.

The mechanism is not the null, though, and that is the actionable part.
Joining each of the 234 kills to the phenotype of the cell killed, against
the escaped fraction of the living tumour at that instant (E):

  T cell       23 kills   O=1    E=12.0   O/E 0.08   z = -4.6
  NK cell     124 kills   O=67   E=62.6   O/E 1.07   z = +0.8
  macrophage   87 kills   O=45   E=43.2   O/E 1.04   z = +0.4

The T-cell arm selects FOR escape overwhelmingly and does under a tenth
of the killing; the other two cannot see the trait (macrophage
phagocytosis reads PD-L1 and polarization, never MHC-I; the NK legacy
step at 0.4 puts 0.30 and 0.20 on the same side of itself). So the null
is DILUTION. Replace the same total infiltrate with all T cells and the
claim passes decisively: +0.0617 (sd 0.0424), which is +0.0554 +- 0.0125
above drift, t = 4.4, with O/E 0.63 at z = -7.2 over 302 kills.

Three consequences for anyone reading these rates:

* the limit on observable selection here is the T/NK/macrophage kill mix
  (realism item 2's territory), NOT any number in this module. Raising mu
  would not help; raising the T-cell share of kills would.

  SCOPE, because this was measured on two presets and there are three.
  Baseline and Aggressive Tumor are the arms above, and in both the
  T-cell share of kills is under a tenth (23 of 234, 15 of 224). The
  IMMUNOTHERAPY preset was NOT in the 96-run design, and it is the
  counter-example that bounds the claim. Measured 2026-10-28 on the
  Immunotherapy slab, 24 h, dt 0.05, founder 0.5, seeds 1-3, PWM
  scorer:

    seed   T / NK / mac kills   escaped killed   O / E    delta
      1      31 / 19 / 15          35 of 65       1.17   -0.0417
      2      33 / 18 / 10          31 of 61       0.99   +0.0021
      3      36 / 12 / 17          36 of 65       1.15   -0.0671

  T cells are 100 of 191 kills, 52% -- the regime this module says
  would make selection detectable -- and escaped cells are not spared
  at all: 102 of 191 kills against a time-matched expectation of 92.5,
  O/E 1.10, with the escaped share of survivors FALLING (mean delta
  -0.036). The reason is the next bullet: Immunotherapy runs
  ``checkpoint_block`` 1.0. "Raise the T-cell share" therefore helps
  only WITHOUT blockade. These three seeds are not pinned in the
  suite; the 12-seed Baseline and Aggressive arms are.
* BLOCKADE NOW SEPARATES THE ROUTES, but through two lineages rather
  than one rule. The item's done-when was that an MHC-I-loss cell
  resist checkpoint blockade while a PD-L1-gain cell is rescued by it,
  so blockade identifies the escape route.

  When this module was written it did not. The ICB term was
  ``(1 - costimulation) * block * 0.5``, containing neither affinity
  nor the target's MHC-I, so at affinity 0.95 and costimulation 0.6
  full blockade took an escaped cell at MHC-I 0.20 from p 0.0204 to
  0.5455 (26.7x) against an intact cell's 0.9287 to 0.9775 (1.1x): the
  escape PENALTY went from 0.022 to 0.558 of the intact cell, so
  therapy helped the escaped cell more. Only TOTAL loss resisted
  blockade, and only because ``_recognize`` hard-gates below 0.05 --
  the rule itself returned p = 0.165 at MHC-I 0.0.

  That term has since been rewritten (see ``tcell_kill``): blockade
  relieves PD-1 suppression of CD28 multiplicatively, so the signal is
  ``affinity * mhc1 * c_eff`` and cannot exist without recognition. The
  escape penalty under full blockade is now 0.140, p at MHC-I 0.0 is
  0.0 rather than 0.165, and ``checkpoint_block = 0`` is bit-identical
  to the old rule -- so the 12-seed Baseline and Aggressive numbers
  above are unaffected, since neither preset sets it.

  The PD-L1 half resolved the other way, and not by code: Klement 2023
  (PMID 36917954) deleted Cd274 in 4T1 and CT26 and found tumour PD-L1
  "conferred no protection against ... cytotoxicity", unrescuable by
  IFN-gamma pretreatment. So a PD-L1-gain cell is NOT supposed to be
  rescued on the T-cell path, and PD-L1 correctly gates only the
  MACROPHAGE path (measured: pdl1 1.0 goes 0.015 -> 0.050 under
  blockade, pdl1 0.0 unchanged at 0.050). The heritable PD-L1 rate here
  is still 0.0, so that half is demonstrated only on a hand-set
  phenotype.
* the NK step is a LIVE sign error, not a latent one. This previously
  read "both phenotypes sit below 0.4 in a shipped run ... 0 of 55 at or
  above the step at 23 of 24 hourly samples", and the measurement it
  cited says otherwise. Sampling the escaped-founder Baseline slab run
  every hour, seed 1:

    scorer      samples with an intact cell at/above 0.4   max intact
    PWM fallback          4 of 24  (h 17.3-20.3, 1 cell)      0.4010
    real MHCflurry       16 of 24  (h 8.3 on, 2-3 cells)      0.4252

  Under the predictor CI and production use, intact cancer cells were
  over the missing-self step for two thirds of the run and 2 of them
  ended the run there. Escaped cells never crossed it under either
  scorer (max 0.2653 / 0.2835). That is exactly the counter-selecting
  regime: an intact cell leaves NK's view while its escaped neighbour
  stays in it, so NK kills the escaped one -- in a 12 h arena at full
  IFN drive, 187 escaped and 0 intact. Escape becomes a liability,
  which is backwards, and it is an artefact of an unsourced step
  function. Grading NK recognition BY MHC-I instead of stepping at an
  unsourced 0.4 would remove it.

  MOSTLY STOPPED, and now SCORER-DEPENDENT, re-measured after
  division_time_normal became 90 h. Fewer normal cells means fewer
  kills (23 -> 15), which means less IFN-gamma and less bystander
  MHC-I induction, so the shipped run sits near the step instead of
  over it:

    scorer        max intact MHC-I    samples at/above 0.4
    MHCflurry           0.3885               0 of 24   (was 16)
    PWM fallback        0.4093               2 of 24   (was 4)

  So under CI and production the shipped Baseline slab no longer
  enters the regime; under the fallback it still just does. Whether it
  fires now depends on which binding predictor is installed, which is
  not a fact about NK cells -- that dependence is itself the reason
  the unsourced step is the wrong shape.

  This is NOT a fix. The step function still inverts selection wherever
  it is crossed; the shipped preset merely stopped reaching it under
  one scorer, by 0.0115 of MHC-I. Anything that raises IFN drive --
  more immune cells, more kills, a denser infiltrate -- puts it back.
  Grading NK recognition BY MHC-I instead of stepping at an unsourced
  0.4 remains the fix, and the margin is small enough that leaving it
  on the watch list means watching.

tests/test_escape_is_selected_not_imposed.py holds all of it.

Total loss: no rate exists, and the two-hit structure is why
-----------------------------------------------------------
For the T-cell-invisible phenotype only frequencies exist. Sade-Feldman
2017 (PMID 29070816): point mutations, deletions or LOH in B2M in 29.4%
of patients with progressing disease, B2M LOH enriched threefold in
non-responders (~30%) against responders (~10%), and "loss of both
copies of B2M is found only in non-responders". Zaretsky 2016
(PMID 27433843): one of four patients, a truncating B2M mutation with
"loss of surface expression of major histocompatibility complex class
I" -- a case count at n=4, not a proportion.

Biallelic inactivation needs two hits. At a per-allele rate of the order
measured above, total loss arising from an intact cell in one division
is ~mu^2 = 2.4e-12, which is never; arising from a cell that has ALREADY
lost one allele it is ~mu. That ordering is the whole structure here,
and it is consistent with B2M loss showing up mainly under therapy
pressure in both papers. The arithmetic is OURS, not either paper's, so
:data:`MHC1_SECOND_HIT_PER_DIVISION` is an ESTIMATE: the same bracket as
the first hit, on the assumption that the remaining locus fails by the
same mechanism, which nobody has measured.

The magnitude is the patient's, not a constant
----------------------------------------------
McGranahan measures how much of the neoantigen repertoire an allele
loss actually removes, and it is patient-specific: "In one extreme
example, tumor CRUK0020, a lung adenocarcinoma, we observed a total of
1,220 mutations predicted to yield neoantigens, of which 92% were
predicted to bind to lost HLA alleles." 92% is explicitly the extreme,
not a typical value, and the cohort distribution is in Figure 4F, which
is a figure and was not read. So no constant is assigned: the engine
already carries the patient's own neoantigens on the patient's own HLA
panel, so a loss event drops the presentations bound to the lost allele
and the size of the repertoire hit is computed from the patient. The
only number is the surface one -- losing 1 of n equally expressed panel
alleles leaves (n-1)/n of surface class I, which is arithmetic, not a
measurement, and is labelled 'derived'.

Measured, and it is not a small effect. On the engine's default panel
(HLA-A*02:01 / A*24:02 / B*07:02) with the real MHCflurry predictor,
KRAS G12D binds A*02:01 and nothing else: a cell that loses A*02:01
displays NOTHING and is invisible to T cells at surface MHC-I 0.2,
while a cell that loses either other allele keeps its whole repertoire
and is untouched. The same event is therefore total escape or none at
all depending on which allele went -- CRUK0020's 92% taken to its
limit for this patient. Under the position-weight-matrix fallback the
same mutation binds all three alleles and no loss goes dark, which is a
useful reminder that this coupling is only as good as the binding
predictor behind it.

The allele lost is drawn UNIFORMLY from the panel. Biasing the draw
toward the allele that carries the most neoantigens would impose the
very selection this module exists to let the immune module discover, and
a uniform supply under selection is exactly what produces McGranahan's
parallel evolution.

PD-L1: predominantly inducible, so the heritable component is ZERO
------------------------------------------------------------------
Heritable PD-L1 gain is real but rare, and the literature gives
prevalences in established tumours, never a per-division rate:

* PD-L1 (CD274) amplification in 843 of 118,187 tumours, 0.7%
  (Goodman 2018, PMID 29902298) -- and "PDL1 amplification did not
  always correlate with high-positive PD-L1 expression by
  immunohistochemical analysis", so even the magnitude does not follow;
* 9p24.1 amplification in 1.0% of 5,399 breast cancers, 5.1% in
  triple-negative (Gupta 2019, PMID 30576871);
* 3'-UTR-disrupting structural variants: adult T-cell leukaemia/lymphoma
  27%, DLBCL 8%, stomach adenocarcinoma 2% (Kataoka 2016, PMID
  27281199), which "invariably lead to a marked elevation of aberrant
  PD-L1 transcripts" -- a direction, no magnitude.

None is prostate; none is a rate; and the engine already models the
dominant route, which is signalling state: adaptive IFN-gamma-driven
induction at contact, plus oncogene-driven intrinsic PD-L1
(``_INTRINSIC_PDL1_INDUCERS``: PIK3CA, MYC, EGFR), both of which the
daughter already inherits. So :data:`PDL1_HERITABLE_GAIN_PER_DIVISION`
is 0.0: the honest answer is that no heritable PD-L1 rate was found, not
an invented one. The knob exists and is inert by default; any non-zero
value a caller sets is that caller's ESTIMATE, and
:data:`PDL1_HERITABLE_GAIN_MAGNITUDE` has a direction from Kataoka and
no measured size.

Founder escape variants
-----------------------
:data:`FOUNDER_HLA_LOH_FRACTION` is 0.0 by default, and that default is
a statement, not a shrug. McGranahan gives the fraction of TUMOURS with
HLA LOH (40%) and the fraction of those in which it is subclonal (65%);
the fraction of CELLS carrying it inside one tumour is not in anything
read here, and it is the quantity a founder fraction needs. A non-zero
value is therefore the caller's ESTIMATE against the prevalence target,
and the report says so.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple


@dataclass(frozen=True)
class EscapeParameter:
    """A number on the escape path, and what stands behind it."""
    value: float
    low: float
    high: float
    status: str          # 'sourced', 'derived' or 'ESTIMATE'
    source: str

    def as_dict(self) -> Dict[str, object]:
        return {'value': self.value, 'range': (self.low, self.high),
                'status': self.status, 'source': self.source}


#: Trait names recorded on a cell. Strings, because they are read by the
#: dashboard and written into run reports; the set is closed.
TRAIT_ALLELE_LOSS = 'mhc1_allele_loss'
TRAIT_TOTAL_LOSS = 'mhc1_total_loss'
TRAIT_PDL1_GAIN = 'pdl1_heritable_gain'
ESCAPE_TRAITS: Tuple[str, ...] = (
    TRAIT_ALLELE_LOSS, TRAIT_TOTAL_LOSS, TRAIT_PDL1_GAIN)

#: Surface MHC-I below which ``ImmuneModule._recognize`` returns
#: (False, None) for a T cell, whatever the cell is displaying. MIRRORED
#: from the literal in that method, not imported, because the immune
#: module must not depend on this one; tests/test_escape_can_be_inherited
#: pins the two together by BEHAVIOUR, so a change to either is caught.
T_CELL_RECOGNITION_MHC1_FLOOR = 0.05

#: Every per-cell-per-generation rate of HLA class I allele loss found,
#: as (rate, PMID, what was reported). The point value below is their
#: geometric mean, so the aggregation is checkable -- but it is FIVE
#: REPORTED VALUES FROM FOUR PAPERS AND TWO LABORATORIES, not five
#: independent determinations: the 1980 entry restates its own group's
#: earlier estimate rather than measuring a new rate, and
#: de Nooij-van Dalen contributes 3 of the 5 (both values above 1e-6
#: among them). Dropping the restated entry gives 2.06e-6 instead of
#: 1.55e-6, +33%, inside the same bracket.
MHC1_LOSS_RATE_MEASUREMENTS: Tuple[Tuple[float, str, str], ...] = (
    (5.0e-7, '6969452',
     "Pious 1980: PREVIOUSLY ESTIMATED, restated in this paper -- "
     "'...the overall rate of spontaneous variation for HLA alleles, "
     "which we previously estimated at about 5 x 10(-7) per cell per "
     "generation'. What the paper measures is that extended lesions of "
     "0.8 cM or longer contribute <10% of that rate, in pseudodiploid "
     "T5-1. Same group as PMID 70075"),
    (7.9e-7, '10029664',
     "de Nooij-van Dalen 1999: HLA-A2 loss, R83-4915"),
    (8.0e-7, '70075',
     "Pious & Soderland 1977: HLA-B27 loss, diploid lymphoid line"),
    (4.1e-6, '9443039',
     "de Nooij-van Dalen 1998: HLA-A2 loss, TK6"),
    (6.9e-6, '9443039',
     "de Nooij-van Dalen 1998: HLA-A2 loss, WI-L2-NS"),
)


def _geometric_mean(values: Sequence[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


_RATES = tuple(rate for rate, _pmid, _what in MHC1_LOSS_RATE_MEASUREMENTS)

#: Per-division probability that a daughter loses one HLA class I
#: ALLELE: the partial route, still recognised, blockade still effective.
#: One allele, not one haplotype -- see the module docstring.
MHC1_ALLELE_LOSS_PER_DIVISION = EscapeParameter(
    _geometric_mean(_RATES), min(_RATES), max(_RATES), 'derived',
    "geometric mean of 5 reported per-cell-per-generation rates of HLA "
    "class I ALLELE loss (PMIDs 6969452, 10029664, 70075, 9443039 x2) "
    "-- 4 papers, 2 laboratories, and the 1980 value restates that "
    "group's own earlier estimate, so this is not an average of five "
    "independent determinations; without it the mean is 2.06e-6. All in "
    "non-malignant human lymphoid lines in vitro, so a LOWER bound for "
    "a tumour cell -- no tumour-line or in vivo rate found")

#: Per-division probability that a daughter ALREADY carrying one-allele
#: loss loses presentation altogether: the total route, T-cell-invisible,
#: not rescuable by checkpoint blockade.
MHC1_SECOND_HIT_PER_DIVISION = EscapeParameter(
    MHC1_ALLELE_LOSS_PER_DIVISION.value,
    MHC1_ALLELE_LOSS_PER_DIVISION.low,
    MHC1_ALLELE_LOSS_PER_DIVISION.high, 'ESTIMATE',
    "no per-division rate for B2M loss of function or for total MHC-I "
    "loss exists in the literature read (Sade-Feldman 2017 PMID "
    "29070816 and Zaretsky 2016 PMID 27433843 give frequencies and a "
    "case count). Set equal to the first-hit bracket on the assumption "
    "that the remaining locus fails by the same mechanism, which makes "
    "total loss a TWO-HIT event -- ~mu^2 = 2.4e-12 per division from an "
    "intact cell, ~mu from one already hemizygous. That ordering is "
    "ours, not either paper's; it is why B2M loss appears under therapy "
    "pressure rather than in treatment-naive tumours")

#: Surface class I left after losing 1 of n equally expressed panel
#: alleles. Arithmetic over the engine's HLA panel, not a measurement:
#: see surviving_surface_fraction. It is a PER-ALLELE magnitude. A whole
#: haplotype carries an A, a B and a C allele in cis, so haplotype LOH
#: on a 6-allele genotype would leave 3/6 = 0.5, not 5/6; this number
#: must not be quoted for a haplotype.
SURFACE_FRACTION_PER_ALLELE_LOST = EscapeParameter(
    2.0 / 3.0, 0.5, 5.0 / 6.0, 'derived',
    "(n-1)/n per ALLELE lost, for the n alleles on the panel; 2/3 for "
    "the engine's 3-allele default panel, 5/6 for a full 6-allele class "
    "I genotype, 0.5 for a 2-allele panel. NOT a haplotype's magnitude: "
    "a haplotype is 3 of 6 alleles in cis and would leave 0.5 on a full "
    "genotype. The REPERTOIRE hit is not this number either: it is "
    "computed per patient from the peptides bound to the lost allele "
    "(McGranahan 2017 PMID 29107330, 'Allele-Specific HLA Loss and "
    "Immune Escape in Lung Cancer Evolution', whose extreme case "
    "CRUK0020 lost 92% of predicted neoantigens to its lost alleles)")

#: Per-division probability of a heritable PD-L1 gain. ZERO -- see the
#: module docstring: every PD-L1 genomic figure found is a prevalence in
#: established tumours, none a rate, none prostate, and the engine
#: already carries the dominant inducible routes.
PDL1_HERITABLE_GAIN_PER_DIVISION = EscapeParameter(
    0.0, 0.0, 0.0, 'ESTIMATE',
    "no per-division rate exists. PD-L1 amplification is 0.7% of "
    "118,187 tumours (Goodman 2018, PMID 29902298), 9p24.1 "
    "amplification 1.0% of 5,399 breast cancers (Gupta 2019, PMID "
    "30576871), 3'-UTR structural variants 27%/8%/2% in ATL/DLBCL/"
    "stomach (Kataoka 2016, PMID 27281199) -- all prevalences in "
    "established tumours. PD-L1 here is predominantly inducible "
    "signalling state (adaptive IFN-gamma induction plus oncogene-driven "
    "intrinsic PD-L1), which the daughter already inherits, so the "
    "heritable component is left at zero rather than invented")

#: How much a heritable PD-L1 gain raises pdl1_expression, if a caller
#: turns the rate above on. Inert by default.
PDL1_HERITABLE_GAIN_MAGNITUDE = EscapeParameter(
    0.5, 0.0, 1.0, 'ESTIMATE',
    "direction only: 3'-UTR disruption gives 'a marked elevation of "
    "aberrant PD-L1 transcripts' (Kataoka 2016, PMID 27281199) and "
    "amplification 'did not always correlate with high-positive PD-L1 "
    "expression' (Goodman 2018, PMID 29902298). No magnitude measured; "
    "inert while the rate is 0.0")

#: Fraction of FOUNDING cancer cells seeded already carrying one-allele
#: loss. Zero by default and deliberately so.
#:
#: The range is [0.0, 1.0] and was [0.0, 0.40] until 2026-10-28. 0.40 is
#: a per-TUMOUR prevalence and this is a per-CELL fraction: a clonal
#: event puts 1.0 of the cells there, not 0.40, so the sentence that
#: defended the old bound ("a fully clonal event would put every cell
#: there") argued against it. Every headline measurement in realism
#: item 1 uses 0.5, which the old bracket called out of range while
#: calibration_report reported only that it differed from the point
#: value.
FOUNDER_HLA_LOH_FRACTION = EscapeParameter(
    0.0, 0.0, 1.0, 'ESTIMATE',
    "McGranahan 2017 (PMID 29107330) gives HLA LOH in 40% of NSCLCs "
    "(46% in a second cohort) and subclonal in 65% of those -- "
    "fractions of TUMOURS, under selection. The fraction of CELLS "
    "carrying it inside one tumour is the quantity a founder fraction "
    "needs and was not found. The range is the whole unit interval "
    "because that is what is unconstrained: a clonal event puts every "
    "cell there and a subclonal one puts an unmeasured share there. Any "
    "non-zero value set here is the caller's ESTIMATE against the 40% "
    "per-tumour prevalence as a target")


def surviving_surface_fraction(n_alleles: int) -> float:
    """Surface class I left after losing one of ``n_alleles``.

    Arithmetic, not a measurement: equally expressed alleles, one gone.
    A single-allele panel returns 0.0, which is total loss -- the caller
    must treat that as the total route, not the partial one, because a
    cell with no presenting allele left is the T-cell-invisible
    phenotype whatever the mechanism was called.
    """
    n = max(1, int(n_alleles))
    return (n - 1) / n


def expected_events(n_divisions: float,
                    rate: float = MHC1_ALLELE_LOSS_PER_DIVISION.value
                    ) -> float:
    """Expected de novo loss events over ``n_divisions``.

    ``n_divisions`` must be CANCER divisions. Only a cancer daughter
    draws -- ``CellularModule._maybe_escape`` returns None for any other
    cell type -- so passing a run's total division count overstates the
    expectation by whatever fraction of its divisions were normal, which
    on the shipped slab is 14x and on the shipped ring is infinite (the
    ring's cancer cells do not divide in 24 h).

    The number that says whether a run can show escape at all. The
    24 h Baseline slab makes 53-56 cancer divisions of ~849 -> 8.6e-5;
    a 3,000-cell tumour doubling once -> 4.6e-3; a 1e9-cell tumour's
    history -> ~1,550.
    """
    return float(n_divisions) * float(rate)


def divisions_per_expected_event(
        rate: float = MHC1_ALLELE_LOSS_PER_DIVISION.value) -> float:
    """1/rate: the CANCER divisions one expected event needs. 6.5e5 at
    the sourced rate, 1.4e5 to 2.0e6 across its bracket -- against the
    53-56 a 24 h Baseline slab run supplies, a factor of ~11,000."""
    return 1.0 / float(rate)


def calibration_report(
        mhc1_loss: Optional[float] = None,
        second_hit: Optional[float] = None,
        pdl1_gain: Optional[float] = None,
        founder_fraction: Optional[float] = None,
        n_alleles: Optional[int] = None,
) -> Dict[str, Dict[str, object]]:
    """Every escape-path number, for a run to say what it stood on.

    Pass the values a run is actually using and each entry reports
    ``configured`` alongside the calibrated value, so a run that scaled
    mu to make escape visible cannot report the literature rate. Pass
    ``n_alleles`` for the same reason on the surface fraction: the
    calibrated 2/3 is the engine's default 3-allele panel, and a run on
    a real 6-allele patient genotype uses 5/6.

    ``out_of_range`` is reported alongside, True when the configured
    value is outside the parameter's own declared bracket. Without it a
    run could sit outside every number's stated range and say only that
    it differed from the point value, which is a much weaker statement:
    the brackets exist so that a run outside one has to admit it.
    """
    entries = {
        'mhc1_allele_loss_per_division': (
            MHC1_ALLELE_LOSS_PER_DIVISION, mhc1_loss),
        'mhc1_second_hit_per_division': (
            MHC1_SECOND_HIT_PER_DIVISION, second_hit),
        'surface_fraction_per_allele_lost': (
            SURFACE_FRACTION_PER_ALLELE_LOST,
            None if n_alleles is None
            else surviving_surface_fraction(n_alleles)),
        'pdl1_heritable_gain_per_division': (
            PDL1_HERITABLE_GAIN_PER_DIVISION, pdl1_gain),
        'pdl1_heritable_gain_magnitude': (
            PDL1_HERITABLE_GAIN_MAGNITUDE, None),
        'founder_hla_loh_fraction': (
            FOUNDER_HLA_LOH_FRACTION, founder_fraction),
    }
    report: Dict[str, Dict[str, object]] = {}
    for name, (param, configured) in entries.items():
        entry = param.as_dict()
        if configured is not None:
            entry['configured'] = float(configured)
            entry['scaled_from_calibration'] = (
                float(configured) != param.value)
            entry['out_of_range'] = not (
                param.low <= float(configured) <= param.high)
        report[name] = entry
    return report
