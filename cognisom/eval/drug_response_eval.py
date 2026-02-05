"""
Drug Response Evaluator (Phase 5)
=================================

Evaluate drug candidate predictions against experimental data.
Validates ranking quality and dose-response accuracy.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import metrics

log = logging.getLogger(__name__)


@dataclass
class DrugCandidate:
    """A drug candidate with predicted and actual responses."""
    drug_id: str = ""
    drug_name: str = ""
    predicted_ic50: float = 0.0       # Predicted IC50 (uM)
    actual_ic50: float = 0.0          # Actual IC50 (uM)
    predicted_efficacy: float = 0.0   # Predicted max efficacy (%)
    actual_efficacy: float = 0.0      # Actual max efficacy (%)
    predicted_rank: int = 0           # Rank by prediction
    actual_rank: int = 0              # Rank by actual
    dose_response_predicted: List[Tuple[float, float]] = field(default_factory=list)
    dose_response_actual: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class DrugEvalReport:
    """Drug response evaluation report."""
    timestamp: float = 0.0
    candidates_evaluated: int = 0
    ranking_metrics: Dict[str, float] = field(default_factory=dict)
    ic50_metrics: Dict[str, float] = field(default_factory=dict)
    dose_response_metrics: Dict[str, float] = field(default_factory=dict)
    candidates: List[DrugCandidate] = field(default_factory=list)
    top_k_hits: Dict[int, float] = field(default_factory=dict)  # Precision@K
    recommendations: List[str] = field(default_factory=list)
    elapsed_sec: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    @property
    def overall_score(self) -> float:
        """Overall evaluation score (0-100)."""
        scores = []
        if "spearman" in self.ranking_metrics:
            scores.append(self.ranking_metrics["spearman"] * 50 + 50)  # Scale to 0-100
        if "within_2fold" in self.ic50_metrics:
            scores.append(self.ic50_metrics["within_2fold"])
        if "correlation" in self.dose_response_metrics:
            scores.append(self.dose_response_metrics["correlation"] * 50 + 50)
        return sum(scores) / len(scores) if scores else 0.0

    def summary(self) -> str:
        lines = [
            f"Drug Response Evaluation (Score: {self.overall_score:.1f}/100)",
            f"=" * 50,
            f"Candidates evaluated: {self.candidates_evaluated}",
            "",
            "Ranking Quality:",
            f"  Spearman correlation: {self.ranking_metrics.get('spearman', 0):.3f}",
            f"  NDCG@10: {self.ranking_metrics.get('ndcg_10', 0):.3f}",
            f"  Concordance index: {self.ranking_metrics.get('c_index', 0):.3f}",
            "",
            "IC50 Prediction:",
            f"  Within 2-fold: {self.ic50_metrics.get('within_2fold', 0):.1f}%",
            f"  Within 3-fold: {self.ic50_metrics.get('within_3fold', 0):.1f}%",
            f"  Correlation: {self.ic50_metrics.get('correlation', 0):.3f}",
            "",
            "Top-K Hits:",
        ]
        for k, precision in sorted(self.top_k_hits.items()):
            lines.append(f"  Precision@{k}: {precision:.1f}%")

        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in self.recommendations[:3]:
                lines.append(f"  - {rec}")

        return "\n".join(lines)


class DrugResponseEvaluator:
    """Evaluates drug candidate predictions.

    Compares predicted drug responses against experimental data
    to validate ranking quality and dose-response accuracy.
    """

    # Activity threshold for binary classification
    ACTIVE_THRESHOLD_UM = 10.0  # IC50 < 10 uM considered active

    def __init__(
        self,
        ground_truth_dir: str = "data/drug_responses",
        results_dir: str = "data/eval_results"
    ) -> None:
        """Initialize the evaluator.

        Args:
            ground_truth_dir: Directory with ground truth data
            results_dir: Directory to save results
        """
        self._truth_dir = Path(ground_truth_dir)
        self._results_dir = Path(results_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)

        self._ground_truth: Dict[str, Dict] = {}
        self._load_ground_truth()

    def evaluate_candidates(
        self,
        predictions: List[Dict],
        ground_truth: Optional[List[Dict]] = None
    ) -> DrugEvalReport:
        """Evaluate drug candidate predictions.

        Args:
            predictions: List of predicted drug responses
                [{"drug_id": "...", "ic50": ..., "efficacy": ..., "dose_response": [...]}]
            ground_truth: Optional ground truth (uses loaded data if None)

        Returns:
            Evaluation report
        """
        t0 = time.time()
        report = DrugEvalReport()

        # Match predictions to ground truth
        candidates = self._match_candidates(predictions, ground_truth)
        report.candidates = candidates
        report.candidates_evaluated = len(candidates)

        if len(candidates) < 2:
            report.recommendations.append("Need at least 2 candidates for evaluation")
            return report

        # Extract arrays for metrics
        pred_ic50 = np.array([c.predicted_ic50 for c in candidates])
        actual_ic50 = np.array([c.actual_ic50 for c in candidates])
        pred_efficacy = np.array([c.predicted_efficacy for c in candidates])
        actual_efficacy = np.array([c.actual_efficacy for c in candidates])

        # Ranking metrics
        report.ranking_metrics = self._evaluate_ranking(pred_ic50, actual_ic50)

        # IC50 prediction metrics
        report.ic50_metrics = self._evaluate_ic50(pred_ic50, actual_ic50)

        # Dose-response metrics (if available)
        report.dose_response_metrics = self._evaluate_dose_response(candidates)

        # Top-K precision (for virtual screening)
        binary_active = (actual_ic50 < self.ACTIVE_THRESHOLD_UM).astype(int)
        for k in [5, 10, 20, 50]:
            if k <= len(candidates):
                report.top_k_hits[k] = metrics.precision_at_k(
                    binary_active, -pred_ic50, k  # Negative because lower IC50 is better
                ) * 100

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        report.elapsed_sec = time.time() - t0
        self._save_report(report)

        return report

    def evaluate_dose_response_curve(
        self,
        predicted_curve: List[Tuple[float, float]],
        actual_curve: List[Tuple[float, float]]
    ) -> Dict[str, float]:
        """Evaluate a single dose-response curve.

        Args:
            predicted_curve: [(dose, response), ...] predicted
            actual_curve: [(dose, response), ...] actual

        Returns:
            Dictionary of curve comparison metrics
        """
        if not predicted_curve or not actual_curve:
            return {}

        # Align curves to same doses
        pred_doses = np.array([p[0] for p in predicted_curve])
        pred_responses = np.array([p[1] for p in predicted_curve])
        actual_doses = np.array([p[0] for p in actual_curve])
        actual_responses = np.array([p[1] for p in actual_curve])

        # Interpolate to common doses
        common_doses = np.union1d(pred_doses, actual_doses)
        pred_interp = np.interp(common_doses, pred_doses, pred_responses)
        actual_interp = np.interp(common_doses, actual_doses, actual_responses)

        return {
            "mae": metrics.mean_absolute_error(actual_interp, pred_interp),
            "rmse": metrics.mean_squared_error(actual_interp, pred_interp, squared=False),
            "correlation": metrics.correlation_coefficient(actual_interp, pred_interp),
            "r_squared": metrics.r_squared(actual_interp, pred_interp),
        }

    def evaluate_virtual_screen(
        self,
        predictions: List[Dict],
        actives: List[str],
        fraction: float = 0.01
    ) -> Dict[str, float]:
        """Evaluate virtual screening performance.

        Args:
            predictions: Predicted scores [{"drug_id": ..., "score": ...}]
            actives: List of active drug IDs
            fraction: Top fraction to consider

        Returns:
            Virtual screening metrics
        """
        active_set = set(actives)
        drug_ids = [p["drug_id"] for p in predictions]
        scores = [p.get("score", -p.get("ic50", 0)) for p in predictions]

        # Binary labels
        y_true = np.array([1 if d in active_set else 0 for d in drug_ids])
        y_pred = np.array(scores)

        return {
            "enrichment_factor": metrics.enrichment_factor(y_true, y_pred, fraction),
            "auc_roc": metrics.auc_roc(y_true, y_pred),
            "auc_prc": metrics.auc_prc(y_true, y_pred),
            "precision_1pct": metrics.precision_at_k(
                y_true, y_pred, max(1, int(len(y_true) * 0.01))
            ) * 100,
        }

    # ── Internal Methods ────────────────────────────────────────────────

    def _load_ground_truth(self) -> None:
        """Load ground truth drug response data."""
        # Built-in prostate cancer drug data
        self._ground_truth = {
            "enzalutamide": {
                "ic50": 0.5, "efficacy": 85,
                "dose_response": [(0.001, 100), (0.01, 95), (0.1, 70), (1, 40), (10, 15)]
            },
            "abiraterone": {
                "ic50": 1.2, "efficacy": 80,
                "dose_response": [(0.001, 100), (0.01, 90), (0.1, 65), (1, 35), (10, 20)]
            },
            "docetaxel": {
                "ic50": 0.01, "efficacy": 90,
                "dose_response": [(0.0001, 100), (0.001, 85), (0.01, 50), (0.1, 20), (1, 5)]
            },
            "cabazitaxel": {
                "ic50": 0.005, "efficacy": 88,
                "dose_response": [(0.0001, 100), (0.001, 80), (0.01, 45), (0.1, 15)]
            },
            "olaparib": {
                "ic50": 2.5, "efficacy": 70,
                "dose_response": [(0.01, 100), (0.1, 90), (1, 60), (10, 30)]
            },
            "pembrolizumab": {
                "ic50": 5.0, "efficacy": 60,  # Checkpoint inhibitor, different mechanism
            },
        }

        # Load additional from files
        if self._truth_dir.exists():
            for file_path in self._truth_dir.glob("*.json"):
                try:
                    data = json.loads(file_path.read_text())
                    self._ground_truth.update(data)
                except Exception as e:
                    log.warning("Failed to load %s: %s", file_path, e)

    def _match_candidates(
        self,
        predictions: List[Dict],
        ground_truth: Optional[List[Dict]]
    ) -> List[DrugCandidate]:
        """Match predictions to ground truth."""
        candidates = []

        # Use provided ground truth or loaded data
        truth_lookup = {}
        if ground_truth:
            for gt in ground_truth:
                drug_id = gt.get("drug_id", gt.get("drug_name", ""))
                truth_lookup[drug_id.lower()] = gt
        else:
            truth_lookup = {k.lower(): v for k, v in self._ground_truth.items()}

        for pred in predictions:
            drug_id = pred.get("drug_id", pred.get("drug_name", ""))
            drug_key = drug_id.lower()

            if drug_key not in truth_lookup:
                continue

            actual = truth_lookup[drug_key]

            candidate = DrugCandidate(
                drug_id=drug_id,
                drug_name=pred.get("drug_name", drug_id),
                predicted_ic50=pred.get("ic50", pred.get("predicted_ic50", 0)),
                actual_ic50=actual.get("ic50", actual.get("actual_ic50", 0)),
                predicted_efficacy=pred.get("efficacy", pred.get("predicted_efficacy", 0)),
                actual_efficacy=actual.get("efficacy", actual.get("actual_efficacy", 0)),
                dose_response_predicted=pred.get("dose_response", []),
                dose_response_actual=actual.get("dose_response", []),
            )
            candidates.append(candidate)

        # Assign ranks
        candidates_by_pred = sorted(candidates, key=lambda c: c.predicted_ic50)
        candidates_by_actual = sorted(candidates, key=lambda c: c.actual_ic50)

        for i, c in enumerate(candidates_by_pred):
            c.predicted_rank = i + 1
        for i, c in enumerate(candidates_by_actual):
            c.actual_rank = i + 1

        return candidates

    def _evaluate_ranking(
        self,
        pred_ic50: np.ndarray,
        actual_ic50: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate ranking quality."""
        # For IC50, lower is better, so we negate for ranking
        pred_scores = -pred_ic50
        actual_scores = -actual_ic50

        return {
            "spearman": metrics.spearman_correlation(actual_ic50, pred_ic50),
            "c_index": metrics.concordance_index(actual_scores, pred_scores),
            "ndcg_10": metrics.ndcg_score(actual_scores, pred_scores, k=10),
            "ndcg_all": metrics.ndcg_score(actual_scores, pred_scores),
        }

    def _evaluate_ic50(
        self,
        pred_ic50: np.ndarray,
        actual_ic50: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate IC50 prediction accuracy."""
        # Use log scale for IC50
        log_pred = np.log10(np.maximum(pred_ic50, 1e-10))
        log_actual = np.log10(np.maximum(actual_ic50, 1e-10))

        return {
            "correlation": metrics.correlation_coefficient(log_actual, log_pred),
            "r_squared": metrics.r_squared(log_actual, log_pred),
            "mae_log": metrics.mean_absolute_error(log_actual, log_pred),
            "within_2fold": metrics.percent_within_fold(actual_ic50, pred_ic50, fold=2.0),
            "within_3fold": metrics.percent_within_fold(actual_ic50, pred_ic50, fold=3.0),
            "within_5fold": metrics.percent_within_fold(actual_ic50, pred_ic50, fold=5.0),
        }

    def _evaluate_dose_response(
        self,
        candidates: List[DrugCandidate]
    ) -> Dict[str, float]:
        """Evaluate dose-response curve accuracy."""
        correlations = []
        maes = []

        for c in candidates:
            if c.dose_response_predicted and c.dose_response_actual:
                curve_metrics = self.evaluate_dose_response_curve(
                    c.dose_response_predicted,
                    c.dose_response_actual
                )
                if "correlation" in curve_metrics:
                    correlations.append(curve_metrics["correlation"])
                    maes.append(curve_metrics["mae"])

        if not correlations:
            return {}

        return {
            "correlation": float(np.mean(correlations)),
            "mae": float(np.mean(maes)),
            "n_curves_evaluated": len(correlations),
        }

    def _generate_recommendations(self, report: DrugEvalReport) -> List[str]:
        """Generate recommendations."""
        recommendations = []

        spearman = report.ranking_metrics.get("spearman", 0)
        within_2fold = report.ic50_metrics.get("within_2fold", 0)

        if spearman < 0.5:
            recommendations.append(
                f"Ranking correlation low ({spearman:.2f}) - "
                "review molecular descriptors and training data"
            )

        if within_2fold < 50:
            recommendations.append(
                f"Only {within_2fold:.0f}% predictions within 2-fold - "
                "consider model recalibration"
            )

        if report.top_k_hits.get(10, 0) < 50:
            recommendations.append(
                "Low precision in top-10 - virtual screening may miss actives"
            )

        return recommendations

    def _save_report(self, report: DrugEvalReport) -> None:
        """Save report to file."""
        filename = f"drug_eval_{int(report.timestamp)}.json"
        filepath = self._results_dir / filename

        data = {
            "timestamp": report.timestamp,
            "candidates_evaluated": report.candidates_evaluated,
            "overall_score": report.overall_score,
            "ranking_metrics": report.ranking_metrics,
            "ic50_metrics": report.ic50_metrics,
            "top_k_hits": report.top_k_hits,
            "recommendations": report.recommendations,
        }

        filepath.write_text(json.dumps(data, indent=2))
