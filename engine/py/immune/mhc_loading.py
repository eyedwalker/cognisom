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

from engine.py.molecular.peptidome import Peptide


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

def _canonical_allele(allele: str) -> str:
    """Normalize allele string to 'HLA-A*02:01' form, matching the
    PWM table at cognisom.genomics.hla_typer.HLA_BINDING_PROPERTIES.
    """
    if allele.startswith("HLA-"):
        return allele
    return f"HLA-{allele}"
