"""
MHC-I loading: score peptides against patient HLA alleles
=========================================================

Stage two of the closed-loop neoantigen presentation pipeline
(Upgrade 2). Consumes Peptide objects from
``engine.py.molecular.peptidome`` and returns MHCPresentation records
that downstream TCR matching and T-cell kill modules can score.

This module deliberately *reuses* the position-weight-matrix binding
scorer at cognisom/genomics/neoantigen_predictor.py:325 rather than
duplicating it. The neoantigen predictor was written for the clinical
neoantigen path (HLA -> peptide ranking for vaccine candidates); the
same scoring is what the cellular simulation needs for per-cell MHC-I
display. Keeping a single source of truth means MHCflurry-based or
NetMHCpan-based upgrades land in one place.

Provenance flow:
    MUTATION_OCCURRED  --[peptidome]-->  PEPTIDE_GENERATED
        --[mhc_loading]-->  PEPTIDE_PRESENTED  ...

Each MHCPresentation carries enough provenance to reconstruct the
causal chain back to the originating mutation when the event trace
fires CELL_KILLED_BY_TCELL.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from cognisom.engine.py.molecular.peptidome import Peptide


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MHCPresentation:
    """A peptide successfully loaded onto an MHC-I molecule.

    peptide
        The originating Peptide object (provenance: source_gene,
        mutation_label, anchor position).
    hla_allele
        The HLA-I allele on which the peptide is presented
        (e.g., "HLA-A*02:01").
    ic50_nm
        Predicted IC50 in nanomolar. Lower = stronger binding.
    binding_level
        Categorical band: "strong" (< 50 nM), "weak" (< 500 nM),
        or "non-binder" (>= 500 nM).
    presentation_score
        Composite score in [0, 1] used for ranking and for downstream
        kill-probability calculations. Combines binding affinity
        (sigmoidally mapped from IC50) with cleavage score.
    """

    peptide: Peptide
    hla_allele: str
    ic50_nm: float
    binding_level: str
    presentation_score: float

    @property
    def is_strong_binder(self) -> bool:
        return self.binding_level == "strong"

    @property
    def is_weak_binder(self) -> bool:
        return self.binding_level == "weak"

    @property
    def is_neoantigen(self) -> bool:
        """True if this presentation carries a mutated peptide."""
        return self.peptide.is_mutant


# ---------------------------------------------------------------------------
# Scorer wrapper
# ---------------------------------------------------------------------------

class MHCLoader:
    """Score peptides against a patient's HLA-I alleles.

    The PWM scorer underneath is the same one the neoantigen predictor
    uses for clinical vaccine prioritization, so the immune simulation
    and the clinical pipeline never drift apart on what counts as a
    binder. The wrapper exists to:
      * accept Peptide objects (typed inputs, provenance preserved),
      * return MHCPresentation records with a composite presentation
        score that incorporates cleavage probability, and
      * surface a single ``score_all`` entrypoint usable from the
        cellular module's per-step update.

    Strong / weak binder thresholds follow the conventional NetMHCpan
    cutoffs (50 nM strong, 500 nM weak); these are stored on the
    instance so the cellular module can override them for a specific
    simulation regime (e.g., immunotherapy ON / OFF).
    """

    def __init__(
        self,
        strong_threshold_nm: float = 50.0,
        weak_threshold_nm: float = 500.0,
    ):
        from cognisom.genomics.neoantigen_predictor import NeoantigenPredictor
        # NeoantigenPredictor() preseeds a BUILTIN_PROTEINS cache and does
        # not touch the network unless an unseen gene is requested. We
        # only use its private scoring helpers, so init cost is trivial.
        self._scorer = NeoantigenPredictor()
        self.strong_threshold_nm = strong_threshold_nm
        self.weak_threshold_nm = weak_threshold_nm

    # ----- public API --------------------------------------------------

    def score_peptide(
        self,
        peptide: Peptide,
        hla_alleles: Sequence[str],
    ) -> List[MHCPresentation]:
        """Score a single peptide against every supplied HLA allele.

        Returns one MHCPresentation per allele (including non-binders).
        Callers that only want presentable peptides should filter with
        ``presentation.binding_level != 'non-binder'``.
        """
        out: List[MHCPresentation] = []
        for allele in hla_alleles:
            ic50 = self._scorer._predict_binding(peptide.sequence, allele)
            level = self._classify(ic50)
            score = self._composite_score(ic50, peptide.cleavage_score)
            out.append(MHCPresentation(
                peptide=peptide,
                hla_allele=_canonical_allele(allele),
                ic50_nm=ic50,
                binding_level=level,
                presentation_score=score,
            ))
        return out

    def score_all(
        self,
        peptides: Iterable[Peptide],
        hla_alleles: Sequence[str],
        include_non_binders: bool = False,
    ) -> List[MHCPresentation]:
        """Score every (peptide, allele) pair. Default drops non-binders.

        This is the entrypoint the cellular module calls once per cell
        per update step to populate ``mhc1_displayed_peptides``.
        """
        peptides = list(peptides)
        # One batched call for every pair first. MHCflurry is a neural
        # network whose per-invocation overhead dwarfs the marginal cost of
        # another peptide, so scoring pairs one at a time is the difference
        # between seconds and minutes -- the neoantigen path already learned
        # this (NeoantigenPredictor._warm_binding_cache), but this loop,
        # which is the one the cellular module runs on every mutation,
        # still asked pair by pair. It blew the 120 s timeout on the
        # nine-module fixture in CI, where the real predictor runs. A miss
        # just falls through to the per-pair path below.
        self._scorer._warm_binding_cache(
            [p.sequence for p in peptides], hla_alleles,
        )
        out: List[MHCPresentation] = []
        for pep in peptides:
            for pres in self.score_peptide(pep, hla_alleles):
                if include_non_binders or pres.binding_level != "non-binder":
                    out.append(pres)
        return out

    def best_per_peptide(
        self,
        presentations: Iterable[MHCPresentation],
    ) -> Dict[str, MHCPresentation]:
        """Reduce a presentation list to the strongest allele per peptide.

        Keyed by peptide sequence + mutation label so the same sequence
        emerging from different mutations stays distinct.
        """
        best: Dict[str, MHCPresentation] = {}
        for p in presentations:
            key = f"{p.peptide.sequence}|{p.peptide.mutation_label or ''}"
            cur = best.get(key)
            if cur is None or p.ic50_nm < cur.ic50_nm:
                best[key] = p
        return best

    # ----- internals ---------------------------------------------------

    def _classify(self, ic50_nm: float) -> str:
        if ic50_nm < self.strong_threshold_nm:
            return "strong"
        if ic50_nm < self.weak_threshold_nm:
            return "weak"
        return "non-binder"

    @staticmethod
    def _composite_score(ic50_nm: float, cleavage_score: float) -> float:
        """Combine IC50 and cleavage into a [0, 1] presentation score.

        Affinity component maps IC50 -> [0, 1] via a smooth logistic
        centered on the weak-binder threshold (500 nM). Cleavage acts
        as a multiplicative modifier (the proteasome has to produce the
        peptide before MHC can load it).
        """
        # Logistic: ic50=50 nM -> ~0.91, ic50=500 nM -> ~0.5,
        # ic50=5000 nM -> ~0.09, ic50=50000 nM -> ~0.01
        import math
        affinity_component = 1.0 / (1.0 + math.exp(
            (math.log10(max(ic50_nm, 1.0)) - math.log10(500.0)) * 4.6
        ))
        return float(affinity_component * cleavage_score)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ─── how much of a cell's MHC-I carries ONE peptide ──────────────────
#
# WHY THIS EXISTS. The T-cell kill signal is
# ``affinity * mhc_level * costimulation`` through a Hill curve with
# EC50 0.3 and slope 4. Under cognate specificity the affinity is a
# fixed value for an assigned pair, so before this, a T cell targeting a
# peptide that binds at 12 nM and one targeting a peptide that barely
# binds at 12,000 nM killed identically. Predicted binding contributed
# nothing to a simulated target ranking.
#
# WHY NOT GRADE THE AFFINITY. Measured 2026-11-27: above the 0.7
# recognition threshold the Hill curve is saturated -- affinity 0.70
# gives kill probability 0.967 and 0.95 gives 0.990 -- so grading inside
# the recognised band changes nothing, and grading below it flips
# ``is_recognized`` off and makes the peptide invisible instead of
# weakly presented. That is a cliff inside the range it decides, the
# same defect already corrected once in the NK path. Binding strength
# belongs where it acts: in how much of the cell's class I actually
# carries this peptide. Occupancy multiplies into the same signal and
# leaves recognition to the cognate assignment.
#
# THE MODEL. Single-site (Langmuir) occupancy, which is the standard
# form for a peptide competing for a binding groove:
#
#     theta(IC50) = 1 / (1 + IC50 / [P])
#
# reported relative to a strong binder, so a peptide at the
# conventional 50 nM strong-binder cut sits at 1.0 and the current
# behaviour is preserved there:
#
#     occupancy = min(1, theta(IC50) / theta(STRONG_CUT))
#
# PROVENANCE. The DIRECTION is sourced: in the TESLA consortium's
# benchmark, peptides T cells recognised had significantly stronger MHC
# binding affinity (Wells 2020, Cell 183:818, PMID 33038342, abstract
# read directly 2026-11-27), and counting their Table S4 directly gives
# 25 of 224 peptides validated below 50 nM measured affinity (11.2%)
# against 37 of 608 overall (6.1%). The FUNCTIONAL FORM is the standard
# single-site isotherm. The MAGNITUDE is an ESTIMATE: the effective
# peptide concentration below is the one free parameter and no source
# was found for it. It is a single named knob so an external test can
# discipline it; do not tune it to make a simulation come out well.
#
# WHAT IT CANNOT DO. The normalisation caps a tighter-than-50 nM binder
# at 1.0, so this does not distinguish 1 nM from 50 nM. Both saturate,
# which is the point of the strong-binder cut, but it means the model is
# blind to differences among strong binders.

#: Effective peptide concentration, in nM. ESTIMATE -- the one free
#: parameter of the occupancy model, and the only unsourced number in
#: it. Set equal to the strong-binder cut so a peptide at that cut sits
#: at half-maximal single-site occupancy.
PEPTIDE_CONCENTRATION_NM: float = 50.0

#: The affinity an occupancy of 1.0 refers to: the conventional
#: NetMHCpan strong-binder cut, already used throughout this module.
OCCUPANCY_REFERENCE_IC50_NM: float = 50.0

OCCUPANCY_STATUS = "ESTIMATE"

OCCUPANCY_SOURCE = (
    "single-site (Langmuir) occupancy normalised to the 50 nM "
    "strong-binder cut; direction sourced from Wells 2020, Cell "
    "183:818, PMID 33038342 (recognised peptides had significantly "
    "stronger MHC binding; Table S4 gives 11.2% validated below 50 nM "
    "against a 6.1% base rate), but the effective peptide "
    "concentration is an ESTIMATE with no source"
)


def occupancy_from_ic50(
    ic50_nm: Optional[float],
    peptide_concentration_nm: float = PEPTIDE_CONCENTRATION_NM,
    reference_ic50_nm: float = OCCUPANCY_REFERENCE_IC50_NM,
) -> float:
    """Share of a cell's MHC-I carrying this peptide, relative to a strong binder.

    Returns 1.0 when ``ic50_nm`` is None, so a presentation with no
    predicted affinity behaves exactly as it did before this existed
    rather than being silently penalised for a missing measurement.
    """
    if ic50_nm is None:
        return 1.0
    ic50 = float(ic50_nm)
    if ic50 <= 0.0:
        return 1.0
    p = max(1e-9, float(peptide_concentration_nm))
    theta = 1.0 / (1.0 + ic50 / p)
    theta_ref = 1.0 / (1.0 + max(0.0, float(reference_ic50_nm)) / p)
    if theta_ref <= 0.0:
        return 1.0
    return max(0.0, min(1.0, theta / theta_ref))


def occupancy_report() -> Dict[str, object]:
    """What the occupancy step contributes, for the methods page."""
    return {
        "quantity": "share of a cell's MHC-I carrying one peptide",
        "model": "single-site (Langmuir), normalised to a strong binder",
        "peptide_concentration_nm": PEPTIDE_CONCENTRATION_NM,
        "reference_ic50_nm": OCCUPANCY_REFERENCE_IC50_NM,
        "status": OCCUPANCY_STATUS,
        "source": OCCUPANCY_SOURCE,
        "cannot_distinguish": (
            "affinities tighter than the reference cut: they all "
            "saturate at 1.0"),
    }


def _canonical_allele(allele: str) -> str:
    """Normalize allele string to 'HLA-A*02:01' form, matching the
    PWM table at cognisom.genomics.hla_typer.HLA_BINDING_PROPERTIES.
    """
    if allele.startswith("HLA-"):
        return allele
    return f"HLA-{allele}"
