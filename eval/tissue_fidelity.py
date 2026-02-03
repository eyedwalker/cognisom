"""
Tissue Fidelity Evaluator (Phase 5)
===================================

Compare digital twin simulations to histology and imaging data
to measure how accurately the simulation reproduces tissue structure.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import metrics

log = logging.getLogger(__name__)


@dataclass
class SpatialMetrics:
    """Spatial distribution metrics."""
    cell_density: float = 0.0           # cells per mm^2
    density_correlation: float = 0.0    # Correlation with reference
    cluster_count: int = 0
    mean_cluster_size: float = 0.0
    nearest_neighbor_dist: float = 0.0
    ripley_k_deviation: float = 0.0     # Deviation from expected Ripley's K


@dataclass
class CompositionMetrics:
    """Tissue composition metrics."""
    cell_type_distribution: Dict[str, float] = field(default_factory=dict)
    distribution_error: float = 0.0     # KL divergence from reference
    ar_positive_fraction: float = 0.0
    ki67_positive_fraction: float = 0.0
    necrotic_fraction: float = 0.0
    immune_infiltrate: float = 0.0


@dataclass
class MorphologyMetrics:
    """Morphological metrics."""
    mean_cell_diameter: float = 0.0
    cell_diameter_std: float = 0.0
    circularity: float = 0.0            # Mean cell circularity
    aspect_ratio: float = 0.0           # Mean aspect ratio
    gland_area_fraction: float = 0.0    # For prostate tissue


@dataclass
class FidelityReport:
    """Tissue fidelity evaluation report."""
    timestamp: float = 0.0
    reference_source: str = ""          # e.g., "TCGA-PRAD-001"
    spatial_metrics: SpatialMetrics = field(default_factory=SpatialMetrics)
    composition_metrics: CompositionMetrics = field(default_factory=CompositionMetrics)
    morphology_metrics: MorphologyMetrics = field(default_factory=MorphologyMetrics)
    overall_fidelity: float = 0.0       # 0-100
    grade: str = ""                     # A-F
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    elapsed_sec: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    def summary(self) -> str:
        lines = [
            f"Tissue Fidelity Report (Score: {self.overall_fidelity:.1f}/100, Grade: {self.grade})",
            f"=" * 60,
            f"Reference: {self.reference_source}",
            "",
            "Spatial Metrics:",
            f"  Cell density: {self.spatial_metrics.cell_density:.0f} cells/mm²",
            f"  Density correlation: {self.spatial_metrics.density_correlation:.3f}",
            f"  Cluster count: {self.spatial_metrics.cluster_count}",
            "",
            "Composition:",
            f"  Distribution error (KL): {self.composition_metrics.distribution_error:.4f}",
            f"  AR+ fraction: {self.composition_metrics.ar_positive_fraction:.1%}",
            f"  Ki67+ fraction: {self.composition_metrics.ki67_positive_fraction:.1%}",
            f"  Immune infiltrate: {self.composition_metrics.immune_infiltrate:.1%}",
            "",
            "Morphology:",
            f"  Mean cell diameter: {self.morphology_metrics.mean_cell_diameter:.1f} µm",
            f"  Gland area fraction: {self.morphology_metrics.gland_area_fraction:.1%}",
        ]

        if self.issues:
            lines.append("")
            lines.append("Issues:")
            for issue in self.issues[:5]:
                lines.append(f"  - {issue}")

        return "\n".join(lines)


class TissueFidelityEvaluator:
    """Evaluates simulation fidelity against histology data.

    Compares spatial distributions, cell compositions, and morphology
    to reference tissue samples.
    """

    # Reference values for prostate tissue (from literature)
    PROSTATE_REFERENCE = {
        "cell_density": 2500,           # cells per mm^2
        "cell_types": {
            "luminal": 0.60,
            "basal": 0.15,
            "stromal": 0.20,
            "immune": 0.05,
        },
        "ar_positive": 0.75,
        "ki67_positive": 0.15,
        "mean_cell_diameter": 15.0,     # µm
        "gland_area_fraction": 0.40,
    }

    def __init__(
        self,
        reference_dir: str = "data/histology",
        results_dir: str = "data/eval_results"
    ) -> None:
        """Initialize the evaluator.

        Args:
            reference_dir: Directory with reference histology data
            results_dir: Directory to save results
        """
        self._reference_dir = Path(reference_dir)
        self._results_dir = Path(results_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        simulation_state: Dict,
        reference: Optional[Dict] = None,
        reference_name: str = "prostate_default"
    ) -> FidelityReport:
        """Evaluate simulation fidelity against reference.

        Args:
            simulation_state: Current simulation state
                {"cells": [...], "dimensions": [x, y, z], ...}
            reference: Reference tissue data (uses defaults if None)
            reference_name: Name of reference for reporting

        Returns:
            Fidelity report
        """
        t0 = time.time()
        report = FidelityReport(reference_source=reference_name)

        ref = reference or self.PROSTATE_REFERENCE

        # Evaluate spatial distribution
        report.spatial_metrics = self._evaluate_spatial(simulation_state, ref)

        # Evaluate composition
        report.composition_metrics = self._evaluate_composition(simulation_state, ref)

        # Evaluate morphology
        report.morphology_metrics = self._evaluate_morphology(simulation_state, ref)

        # Calculate overall fidelity score
        report.overall_fidelity = self._calculate_overall_score(report)
        report.grade = self._score_to_grade(report.overall_fidelity)

        # Generate recommendations
        report.issues, report.recommendations = self._analyze_issues(report, ref)

        report.elapsed_sec = time.time() - t0
        return report

    def evaluate_from_image(
        self,
        simulation_state: Dict,
        image_path: str,
        segmentation_path: Optional[str] = None
    ) -> FidelityReport:
        """Evaluate against histology image.

        Args:
            simulation_state: Simulation state
            image_path: Path to histology image
            segmentation_path: Optional cell segmentation mask

        Returns:
            Fidelity report
        """
        # Extract features from histology image
        reference = self._extract_image_features(image_path, segmentation_path)
        return self.evaluate(simulation_state, reference, reference_name=image_path)

    # ── Spatial Evaluation ──────────────────────────────────────────────

    def _evaluate_spatial(
        self,
        state: Dict,
        reference: Dict
    ) -> SpatialMetrics:
        """Evaluate spatial distribution."""
        metrics_out = SpatialMetrics()

        cells = state.get("cells", [])
        dimensions = state.get("dimensions", [1000, 1000, 100])  # µm

        if not cells:
            return metrics_out

        # Calculate cell density
        area_mm2 = (dimensions[0] * dimensions[1]) / 1e6
        metrics_out.cell_density = len(cells) / max(area_mm2, 0.001)

        # Compare to reference density
        ref_density = reference.get("cell_density", 2500)
        density_ratio = metrics_out.cell_density / ref_density
        metrics_out.density_correlation = 1.0 - abs(1.0 - density_ratio)

        # Analyze clustering
        positions = np.array([c.get("position", [0, 0, 0])[:2] for c in cells])
        if len(positions) > 1:
            metrics_out.cluster_count, metrics_out.mean_cluster_size = \
                self._analyze_clusters(positions)
            metrics_out.nearest_neighbor_dist = self._mean_nearest_neighbor(positions)
            metrics_out.ripley_k_deviation = self._ripley_k_deviation(
                positions, dimensions[:2]
            )

        return metrics_out

    def _analyze_clusters(
        self,
        positions: np.ndarray,
        eps: float = 50.0
    ) -> Tuple[int, float]:
        """Analyze cell clusters using DBSCAN."""
        try:
            from sklearn.cluster import DBSCAN

            clustering = DBSCAN(eps=eps, min_samples=5).fit(positions)
            labels = clustering.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            if n_clusters > 0:
                cluster_sizes = [np.sum(labels == i) for i in range(n_clusters)]
                mean_size = np.mean(cluster_sizes)
            else:
                mean_size = 0.0

            return n_clusters, mean_size

        except ImportError:
            return 0, 0.0

    def _mean_nearest_neighbor(self, positions: np.ndarray) -> float:
        """Calculate mean nearest neighbor distance."""
        from scipy.spatial import distance

        if len(positions) < 2:
            return 0.0

        dists = distance.cdist(positions, positions)
        np.fill_diagonal(dists, np.inf)
        nn_dists = dists.min(axis=1)

        return float(np.mean(nn_dists))

    def _ripley_k_deviation(
        self,
        positions: np.ndarray,
        bounds: List[float],
        r: float = 100.0
    ) -> float:
        """Calculate deviation from expected Ripley's K function."""
        n = len(positions)
        if n < 2:
            return 0.0

        area = bounds[0] * bounds[1]

        # Count pairs within distance r
        from scipy.spatial import distance
        dists = distance.pdist(positions)
        pairs_within_r = np.sum(dists <= r)

        # Observed K
        k_observed = (area / (n * (n - 1))) * 2 * pairs_within_r

        # Expected K for random distribution
        k_expected = np.pi * r ** 2

        return abs(k_observed - k_expected) / k_expected

    # ── Composition Evaluation ──────────────────────────────────────────

    def _evaluate_composition(
        self,
        state: Dict,
        reference: Dict
    ) -> CompositionMetrics:
        """Evaluate tissue composition."""
        metrics_out = CompositionMetrics()

        cells = state.get("cells", [])
        if not cells:
            return metrics_out

        # Count cell types
        type_counts: Dict[str, int] = {}
        ar_positive = 0
        ki67_positive = 0
        necrotic = 0
        immune = 0

        for cell in cells:
            cell_type = cell.get("type", cell.get("cell_type", "unknown"))
            type_counts[cell_type] = type_counts.get(cell_type, 0) + 1

            if cell.get("ar_positive", False):
                ar_positive += 1
            if cell.get("ki67_positive", cell.get("dividing", False)):
                ki67_positive += 1
            if cell.get("state") == "necrotic":
                necrotic += 1
            if cell_type in ("immune", "tcd8", "macrophage", "nk"):
                immune += 1

        total = len(cells)
        metrics_out.cell_type_distribution = {
            k: v / total for k, v in type_counts.items()
        }
        metrics_out.ar_positive_fraction = ar_positive / total
        metrics_out.ki67_positive_fraction = ki67_positive / total
        metrics_out.necrotic_fraction = necrotic / total
        metrics_out.immune_infiltrate = immune / total

        # Calculate KL divergence from reference
        ref_dist = reference.get("cell_types", {})
        if ref_dist:
            metrics_out.distribution_error = self._kl_divergence(
                metrics_out.cell_type_distribution,
                ref_dist
            )

        return metrics_out

    def _kl_divergence(self, p: Dict[str, float], q: Dict[str, float]) -> float:
        """Calculate KL divergence between distributions."""
        all_keys = set(p.keys()) | set(q.keys())
        epsilon = 1e-10

        kl = 0.0
        for key in all_keys:
            p_val = p.get(key, epsilon)
            q_val = q.get(key, epsilon)
            if p_val > 0:
                kl += p_val * np.log(p_val / q_val)

        return float(kl)

    # ── Morphology Evaluation ───────────────────────────────────────────

    def _evaluate_morphology(
        self,
        state: Dict,
        reference: Dict
    ) -> MorphologyMetrics:
        """Evaluate cell morphology."""
        metrics_out = MorphologyMetrics()

        cells = state.get("cells", [])
        if not cells:
            return metrics_out

        # Extract radii/diameters
        diameters = []
        for cell in cells:
            radius = cell.get("radius", 7.5)
            diameters.append(radius * 2)

        metrics_out.mean_cell_diameter = float(np.mean(diameters))
        metrics_out.cell_diameter_std = float(np.std(diameters))

        # Circularity and aspect ratio (if available)
        circularities = [c.get("circularity", 1.0) for c in cells]
        aspect_ratios = [c.get("aspect_ratio", 1.0) for c in cells]

        metrics_out.circularity = float(np.mean(circularities))
        metrics_out.aspect_ratio = float(np.mean(aspect_ratios))

        # Gland area (estimated from luminal cells)
        luminal_count = sum(
            1 for c in cells
            if c.get("type", c.get("cell_type", "")) == "luminal"
        )
        if luminal_count > 0 and len(cells) > 0:
            # Rough estimate
            metrics_out.gland_area_fraction = luminal_count / len(cells) * 0.7

        return metrics_out

    # ── Overall Score ───────────────────────────────────────────────────

    def _calculate_overall_score(self, report: FidelityReport) -> float:
        """Calculate overall fidelity score (0-100)."""
        scores = []

        # Spatial score (40% weight)
        spatial = report.spatial_metrics
        spatial_score = (
            spatial.density_correlation * 50 +
            max(0, 50 - spatial.ripley_k_deviation * 50)
        )
        scores.append(spatial_score * 0.4)

        # Composition score (40% weight)
        composition = report.composition_metrics
        kl_score = max(0, 100 - composition.distribution_error * 100)
        scores.append(kl_score * 0.4)

        # Morphology score (20% weight)
        morphology = report.morphology_metrics
        ref_diameter = self.PROSTATE_REFERENCE.get("mean_cell_diameter", 15)
        diameter_error = abs(morphology.mean_cell_diameter - ref_diameter) / ref_diameter
        morph_score = max(0, 100 - diameter_error * 100)
        scores.append(morph_score * 0.2)

        return sum(scores)

    def _score_to_grade(self, score: float) -> str:
        """Convert score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def _analyze_issues(
        self,
        report: FidelityReport,
        reference: Dict
    ) -> Tuple[List[str], List[str]]:
        """Analyze issues and generate recommendations."""
        issues = []
        recommendations = []

        # Check density
        ref_density = reference.get("cell_density", 2500)
        if report.spatial_metrics.cell_density < ref_density * 0.5:
            issues.append("Cell density significantly below reference")
            recommendations.append("Increase initial cell seeding or growth rate")
        elif report.spatial_metrics.cell_density > ref_density * 1.5:
            issues.append("Cell density significantly above reference")
            recommendations.append("Increase cell death rate or reduce proliferation")

        # Check composition
        if report.composition_metrics.distribution_error > 0.5:
            issues.append("Cell type distribution differs significantly from reference")
            recommendations.append("Adjust cell type initialization ratios")

        # Check AR expression
        ref_ar = reference.get("ar_positive", 0.75)
        if abs(report.composition_metrics.ar_positive_fraction - ref_ar) > 0.2:
            issues.append(f"AR+ fraction ({report.composition_metrics.ar_positive_fraction:.0%}) "
                        f"differs from reference ({ref_ar:.0%})")

        # Check morphology
        ref_diameter = reference.get("mean_cell_diameter", 15)
        if abs(report.morphology_metrics.mean_cell_diameter - ref_diameter) > 5:
            issues.append("Mean cell diameter outside expected range")
            recommendations.append("Adjust cell size parameters")

        return issues, recommendations

    def _extract_image_features(
        self,
        image_path: str,
        segmentation_path: Optional[str]
    ) -> Dict:
        """Extract features from histology image."""
        # This would use image analysis tools in production
        # For now, return default reference
        log.info("Image feature extraction not implemented, using defaults")
        return self.PROSTATE_REFERENCE
