"""
Cell State Classifier
=====================

Classify tumor immune microenvironment (TIME) from cell state predictions.
Combines Cell2Sentence output with marker-based scoring to classify:
- Overall immune landscape (hot/cold/excluded/suppressed)
- T-cell exhaustion profile
- Macrophage polarization balance
- Immune cell composition
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .cell2sentence import CellStatePrediction

logger = logging.getLogger(__name__)


@dataclass
class ImmuneComposition:
    """Quantified immune cell composition."""
    cd8_t_cells: int = 0
    cd4_t_cells: int = 0
    regulatory_t_cells: int = 0
    nk_cells: int = 0
    macrophages: int = 0
    dendritic_cells: int = 0
    b_cells: int = 0
    other_immune: int = 0
    epithelial: int = 0
    cancer_cells: int = 0
    stromal: int = 0
    total: int = 0

    @property
    def immune_fraction(self) -> float:
        immune = (self.cd8_t_cells + self.cd4_t_cells + self.regulatory_t_cells +
                  self.nk_cells + self.macrophages + self.dendritic_cells +
                  self.b_cells + self.other_immune)
        return immune / self.total if self.total > 0 else 0.0

    @property
    def t_cell_fraction(self) -> float:
        t_cells = self.cd8_t_cells + self.cd4_t_cells + self.regulatory_t_cells
        return t_cells / self.total if self.total > 0 else 0.0

    @property
    def treg_ratio(self) -> float:
        """Ratio of Tregs to total T cells. High → immunosuppression."""
        total_t = self.cd8_t_cells + self.cd4_t_cells + self.regulatory_t_cells
        return self.regulatory_t_cells / total_t if total_t > 0 else 0.0


@dataclass
class ImmuneClassification:
    """Full immune microenvironment classification."""
    # Overall
    immune_score: str  # "hot", "cold", "excluded", "suppressed"
    immune_score_confidence: float

    # Composition
    composition: ImmuneComposition

    # T-cell exhaustion profile
    mean_exhaustion: float  # 0-1
    exhausted_fraction: float  # Fraction with score > 0.6
    progenitor_exhausted_fraction: float  # Score 0.3-0.6

    # Macrophage polarization
    m1_fraction: float
    m2_fraction: float
    m1_m2_ratio: float

    # Derived
    immunotherapy_responsive: bool
    checkpoint_blockade_likely_effective: bool
    combination_therapy_needed: bool

    # Per-cell predictions
    predictions: List[CellStatePrediction] = field(default_factory=list)

    def summary(self) -> Dict:
        return {
            "immune_score": self.immune_score,
            "confidence": round(self.immune_score_confidence, 2),
            "immune_fraction": round(self.composition.immune_fraction, 3),
            "t_cell_fraction": round(self.composition.t_cell_fraction, 3),
            "treg_ratio": round(self.composition.treg_ratio, 3),
            "mean_exhaustion": round(self.mean_exhaustion, 3),
            "exhausted_fraction": round(self.exhausted_fraction, 3),
            "m1_m2_ratio": round(self.m1_m2_ratio, 2),
            "immunotherapy_responsive": self.immunotherapy_responsive,
            "checkpoint_blockade_likely_effective": self.checkpoint_blockade_likely_effective,
            "combination_therapy_needed": self.combination_therapy_needed,
        }


class CellStateClassifier:
    """Classify tumor immune microenvironment from cell state predictions.

    Takes Cell2Sentence predictions (or marker-based heuristic results)
    and classifies the overall immune landscape.

    Example:
        from cognisom.genomics.cell2sentence import Cell2SentenceModel
        from cognisom.genomics.expression_ranker import ExpressionRanker

        ranker = ExpressionRanker()
        model = Cell2SentenceModel()
        model.load()

        sentences = ranker.rank_adata(adata)
        predictions = model.batch_predict(sentences)

        classifier = CellStateClassifier()
        result = classifier.classify(predictions)
        print(f"Immune score: {result.immune_score}")
        print(f"Immunotherapy responsive: {result.immunotherapy_responsive}")
    """

    # Thresholds for immune scoring
    HOT_IMMUNE_FRACTION = 0.15  # >15% immune cells = inflamed
    COLD_IMMUNE_FRACTION = 0.05  # <5% immune cells = desert
    HIGH_EXHAUSTION = 0.6  # Exhaustion score threshold
    HIGH_TREG = 0.3  # Treg ratio threshold for suppression

    def classify(self, predictions: List[CellStatePrediction],
                 tmb: float = 0.0,
                 msi_status: str = "unknown") -> ImmuneClassification:
        """Classify immune microenvironment from cell predictions.

        Args:
            predictions: List of CellStatePrediction from Cell2Sentence.
            tmb: Tumor mutational burden (variants/Mb) — from genomic profile.
            msi_status: MSI status — from genomic profile.

        Returns:
            ImmuneClassification with scores and recommendations.
        """
        # Count cell types
        composition = self._count_composition(predictions)

        # T-cell exhaustion stats
        t_cell_preds = [
            p for p in predictions
            if "t cell" in p.predicted_cell_type.lower() and
               p.predicted_cell_type.lower() != "regulatory t cell"
        ]
        exhaustion_scores = [
            p.exhaustion_score for p in t_cell_preds
            if p.exhaustion_score is not None
        ]

        mean_exhaustion = float(np.mean(exhaustion_scores)) if exhaustion_scores else 0.0
        exhausted_frac = (
            sum(1 for s in exhaustion_scores if s > self.HIGH_EXHAUSTION)
            / len(exhaustion_scores) if exhaustion_scores else 0.0
        )
        progenitor_frac = (
            sum(1 for s in exhaustion_scores if 0.3 < s <= self.HIGH_EXHAUSTION)
            / len(exhaustion_scores) if exhaustion_scores else 0.0
        )

        # Macrophage polarization
        mac_preds = [
            p for p in predictions
            if "macrophage" in p.predicted_cell_type.lower()
        ]
        m1_count = sum(1 for p in mac_preds if p.polarization == "M1")
        m2_count = sum(1 for p in mac_preds if p.polarization == "M2")
        total_mac = len(mac_preds) or 1
        m1_frac = m1_count / total_mac
        m2_frac = m2_count / total_mac
        m1_m2_ratio = m1_count / m2_count if m2_count > 0 else float("inf") if m1_count > 0 else 1.0

        # Classify immune landscape
        immune_score, confidence = self._classify_landscape(
            composition, mean_exhaustion, m1_m2_ratio, tmb, msi_status
        )

        # Determine therapy responsiveness
        immunotherapy_responsive = (
            immune_score == "hot" or
            tmb >= 10.0 or
            msi_status == "MSI-H"
        )

        checkpoint_effective = (
            immunotherapy_responsive and
            exhausted_frac > 0.2  # Need exhausted T-cells to re-activate
        )

        combination_needed = (
            immune_score in ("cold", "excluded", "suppressed") or
            (exhausted_frac > 0.5 and composition.treg_ratio > self.HIGH_TREG)
        )

        return ImmuneClassification(
            immune_score=immune_score,
            immune_score_confidence=confidence,
            composition=composition,
            mean_exhaustion=mean_exhaustion,
            exhausted_fraction=exhausted_frac,
            progenitor_exhausted_fraction=progenitor_frac,
            m1_fraction=m1_frac,
            m2_fraction=m2_frac,
            m1_m2_ratio=m1_m2_ratio,
            immunotherapy_responsive=immunotherapy_responsive,
            checkpoint_blockade_likely_effective=checkpoint_effective,
            combination_therapy_needed=combination_needed,
            predictions=predictions,
        )

    def _count_composition(self, predictions: List[CellStatePrediction]) -> ImmuneComposition:
        """Count cell types from predictions."""
        comp = ImmuneComposition(total=len(predictions))

        type_map = {
            "cd8+ t cell": "cd8_t_cells",
            "cd4+ t cell": "cd4_t_cells",
            "t cell": "cd4_t_cells",  # Default T cell
            "regulatory t cell": "regulatory_t_cells",
            "nk cell": "nk_cells",
            "macrophage": "macrophages",
            "monocyte": "macrophages",
            "dendritic cell": "dendritic_cells",
            "b cell": "b_cells",
            "plasma cell": "b_cells",
            "epithelial": "epithelial",
            "cancer epithelial": "cancer_cells",
            "fibroblast": "stromal",
            "endothelial": "stromal",
        }

        for pred in predictions:
            ct = pred.predicted_cell_type.lower()
            attr = type_map.get(ct, "other_immune")
            setattr(comp, attr, getattr(comp, attr) + 1)

        return comp

    def _classify_landscape(self, composition: ImmuneComposition,
                            mean_exhaustion: float,
                            m1_m2_ratio: float,
                            tmb: float,
                            msi_status: str) -> tuple:
        """Classify as hot/cold/excluded/suppressed."""
        immune_frac = composition.immune_fraction
        treg_ratio = composition.treg_ratio

        # Hot: high immune infiltration, moderate exhaustion, high TMB
        if immune_frac > self.HOT_IMMUNE_FRACTION:
            if treg_ratio > self.HIGH_TREG or mean_exhaustion > 0.7:
                return "suppressed", 0.7
            return "hot", 0.8

        # Cold: very low immune cells
        if immune_frac < self.COLD_IMMUNE_FRACTION:
            return "cold", 0.8

        # Between cold and hot
        if treg_ratio > self.HIGH_TREG:
            return "suppressed", 0.6

        # Moderate infiltration but not reaching hot
        return "excluded", 0.5
