"""Publication-quality figure generation from simulation run artifacts.

Generates matplotlib figures with consistent scientific styling:
serif fonts, labeled axes with units, proper legends, and
colorblind-safe palettes.  Outputs PNG for dashboards and PDF
for LaTeX manuscripts.

Standard per-run figures:
    1. Cell populations over time
    2. Immune dynamics
    3. Oxygen & metabolism
    4. Metastasis & lymphatic events
    5. 3D tissue scatter (final state)
    6. Benchmark comparison overlay

Multi-run comparison figures:
    7. Overlay time-series
    8. Parameter sensitivity
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

# Colorblind-safe palette (Okabe-Ito)
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "cyan": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
}

COLOR_CYCLE = list(COLORS.values())


def _configure_style():
    """Apply publication-quality matplotlib style."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


class FigureGenerator:
    """Generate publication-quality figures from simulation artifacts."""

    def __init__(self):
        try:
            _configure_style()
        except ImportError:
            log.warning("matplotlib not available — figure generation disabled")

    def generate_all(self, run_id: str, artifact_store) -> List[str]:
        """Generate all standard figures for a run.

        Returns list of relative paths to generated figure files.
        """
        ts = artifact_store.load_time_series(run_id)
        if not ts or "time" not in ts:
            log.warning("No time series data for run %s, skipping figures", run_id)
            return []

        generated = []

        try:
            path = self._fig_cell_populations(run_id, ts, artifact_store)
            if path:
                generated.append(path)
        except Exception as e:
            log.warning("Cell populations figure failed: %s", e)

        try:
            path = self._fig_immune_dynamics(run_id, ts, artifact_store)
            if path:
                generated.append(path)
        except Exception as e:
            log.warning("Immune dynamics figure failed: %s", e)

        try:
            path = self._fig_oxygen_metabolism(run_id, ts, artifact_store)
            if path:
                generated.append(path)
        except Exception as e:
            log.warning("Oxygen metabolism figure failed: %s", e)

        try:
            path = self._fig_metastasis(run_id, ts, artifact_store)
            if path:
                generated.append(path)
        except Exception as e:
            log.warning("Metastasis figure failed: %s", e)

        try:
            path = self._fig_3d_tissue(run_id, artifact_store)
            if path:
                generated.append(path)
        except Exception as e:
            log.warning("3D tissue figure failed: %s", e)

        try:
            path = self._fig_event_accumulation(run_id, ts, artifact_store)
            if path:
                generated.append(path)
        except Exception as e:
            log.warning("Event accumulation figure failed: %s", e)

        log.info("Generated %d figures for run %s", len(generated), run_id)
        return generated

    # ── Standard Figures ─────────────────────────────────────────────

    def _fig_cell_populations(self, run_id, ts, artifact_store) -> Optional[str]:
        """Fig 1: Cell populations over time."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.5))
        t = ts.get("time", [])

        ax.plot(t, ts.get("n_cancer", []), color=COLORS["red"],
                linewidth=2, label="Cancer cells")
        ax.plot(t, ts.get("n_normal", []), color=COLORS["blue"],
                linewidth=2, label="Normal cells")
        ax.plot(t, ts.get("n_cells", []), color=COLORS["black"],
                linewidth=1.5, linestyle="--", label="Total cells")

        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Cell Count")
        ax.set_title("Cell Populations Over Time")
        ax.legend(loc="best")
        fig.tight_layout()

        return artifact_store.save_figure(run_id, fig, "fig_cell_populations")

    def _fig_immune_dynamics(self, run_id, ts, artifact_store) -> Optional[str]:
        """Fig 2: Immune cell dynamics and kill events."""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        t = ts.get("time", [])

        # Left: immune cell counts
        ax1.plot(t, ts.get("n_t_cells", []), color=COLORS["blue"],
                 linewidth=2, label="T cells")
        ax1.plot(t, ts.get("n_nk_cells", []), color=COLORS["green"],
                 linewidth=2, label="NK cells")
        ax1.plot(t, ts.get("n_macrophages", []), color=COLORS["orange"],
                 linewidth=2, label="Macrophages")
        ax1.plot(t, ts.get("n_activated", []), color=COLORS["red"],
                 linewidth=1.5, linestyle=":", label="Activated")
        ax1.set_xlabel("Time (hours)")
        ax1.set_ylabel("Cell Count")
        ax1.set_title("Immune Cell Dynamics")
        ax1.legend(loc="best")

        # Right: cumulative kills
        ax2.plot(t, ts.get("total_kills", []), color=COLORS["red"],
                 linewidth=2)
        ax2.set_xlabel("Time (hours)")
        ax2.set_ylabel("Cumulative Kills")
        ax2.set_title("Immune-Mediated Cancer Cell Death")
        ax2.fill_between(t, 0, ts.get("total_kills", []),
                         color=COLORS["red"], alpha=0.1)

        fig.tight_layout()
        return artifact_store.save_figure(run_id, fig, "fig_immune_dynamics")

    def _fig_oxygen_metabolism(self, run_id, ts, artifact_store) -> Optional[str]:
        """Fig 3: Oxygen and metabolic dynamics."""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        t = ts.get("time", [])

        # Left: O2 and glucose
        ax1.plot(t, ts.get("avg_cell_O2", []), color=COLORS["blue"],
                 linewidth=2, label="Avg O₂")
        ax1.plot(t, ts.get("avg_cell_glucose", []), color=COLORS["orange"],
                 linewidth=2, label="Avg Glucose")
        ax1.set_xlabel("Time (hours)")
        ax1.set_ylabel("Concentration (normalized)")
        ax1.set_title("Oxygen & Glucose Levels")
        ax1.legend(loc="best")

        # Right: hypoxic regions
        ax2.plot(t, ts.get("hypoxic_regions", []), color=COLORS["red"],
                 linewidth=2)
        ax2.set_xlabel("Time (hours)")
        ax2.set_ylabel("Count")
        ax2.set_title("Hypoxic Regions")
        ax2.fill_between(t, 0, ts.get("hypoxic_regions", []),
                         color=COLORS["red"], alpha=0.1)

        fig.tight_layout()
        return artifact_store.save_figure(run_id, fig, "fig_oxygen_metabolism")

    def _fig_metastasis(self, run_id, ts, artifact_store) -> Optional[str]:
        """Fig 4: Metastasis and lymphatic transport."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.5))
        t = ts.get("time", [])

        ax.plot(t, ts.get("total_metastases", []), color=COLORS["red"],
                linewidth=2, label="Metastases")
        ax.plot(t, ts.get("cancer_in_vessels", []), color=COLORS["orange"],
                linewidth=2, label="Cancer in vessels")
        ax.plot(t, ts.get("immune_in_vessels", []), color=COLORS["blue"],
                linewidth=2, label="Immune in vessels")

        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Count")
        ax.set_title("Metastasis & Lymphatic Transport")
        ax.legend(loc="best")
        fig.tight_layout()

        return artifact_store.save_figure(run_id, fig, "fig_metastasis")

    def _fig_3d_tissue(self, run_id, artifact_store) -> Optional[str]:
        """Fig 5: 3D scatter of final tissue state."""
        import matplotlib.pyplot as plt

        snapshots = artifact_store.load_cell_snapshots(run_id)
        if not snapshots:
            return None

        last = snapshots[-1]
        positions = last.get("cell_positions")
        cell_types = last.get("cell_types", [])

        if positions is None or len(positions) == 0:
            return None

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Color by cell type
        type_colors = {"cancer": COLORS["red"], "normal": COLORS["blue"]}
        for ct, color in type_colors.items():
            mask = np.array([t == ct for t in cell_types])
            if mask.any():
                pts = positions[mask]
                ax.scatter(
                    pts[:, 0], pts[:, 1], pts[:, 2],
                    c=color, s=15, alpha=0.6, label=ct.capitalize(),
                )

        # Immune cells
        immune_pos = last.get("immune_positions")
        if immune_pos is not None and len(immune_pos) > 0:
            ax.scatter(
                immune_pos[:, 0], immune_pos[:, 1], immune_pos[:, 2],
                c=COLORS["green"], s=20, marker="^", alpha=0.8, label="Immune",
            )

        ax.set_xlabel("X (μm)")
        ax.set_ylabel("Y (μm)")
        ax.set_zlabel("Z (μm)")
        ax.set_title("3D Tissue State (Final)")
        ax.legend(loc="upper left")
        fig.tight_layout()

        return artifact_store.save_figure(run_id, fig, "fig_3d_tissue")

    def _fig_event_accumulation(self, run_id, ts, artifact_store) -> Optional[str]:
        """Fig 6: Cumulative event counts."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.5))
        t = ts.get("time", [])

        metrics = [
            ("total_divisions", "Divisions", COLORS["blue"]),
            ("total_deaths", "Deaths", COLORS["red"]),
            ("total_transformations", "Transformations", COLORS["orange"]),
            ("total_kills", "Immune Kills", COLORS["green"]),
            ("total_metastases", "Metastases", COLORS["purple"]),
        ]

        for key, label, color in metrics:
            values = ts.get(key, [])
            if values and max(v for v in values if isinstance(v, (int, float))) > 0:
                ax.plot(t, values, color=color, linewidth=2, label=label)

        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Cumulative Count")
        ax.set_title("Event Accumulation")
        ax.legend(loc="best")
        fig.tight_layout()

        return artifact_store.save_figure(run_id, fig, "fig_event_accumulation")

    # ── Comparison Figures ───────────────────────────────────────────

    def generate_comparison(
        self,
        run_ids: List[str],
        run_names: List[str],
        artifact_store,
        metric_key: str = "n_cancer",
        title: str = "Comparison",
        ylabel: str = "Count",
        output_run_id: Optional[str] = None,
    ) -> Optional[str]:
        """Generate overlay time-series comparison across multiple runs.

        Saves figure to the first run's directory (or output_run_id).
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))

        for i, (run_id, name) in enumerate(zip(run_ids, run_names)):
            ts = artifact_store.load_time_series(run_id)
            if not ts:
                continue
            t = ts.get("time", [])
            values = ts.get(metric_key, [])
            color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
            ax.plot(t, values, color=color, linewidth=2, label=name)

        ax.set_xlabel("Time (hours)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best")
        fig.tight_layout()

        save_id = output_run_id or run_ids[0]
        return artifact_store.save_figure(save_id, fig, f"fig_comparison_{metric_key}")

    def generate_parameter_sensitivity(
        self,
        run_ids: List[str],
        param_values: List[float],
        metric_key: str,
        artifact_store,
        param_name: str = "Parameter",
        metric_name: str = "Metric",
        output_run_id: Optional[str] = None,
    ) -> Optional[str]:
        """Generate parameter sensitivity bar chart.

        Shows how a final metric varies across parameter values.
        """
        import matplotlib.pyplot as plt

        final_values = []
        for run_id in run_ids:
            ts = artifact_store.load_time_series(run_id)
            if ts and metric_key in ts:
                vals = ts[metric_key]
                final_values.append(vals[-1] if vals else 0)
            else:
                final_values.append(0)

        fig, ax = plt.subplots(figsize=(8, 5))
        x = range(len(param_values))
        bars = ax.bar(x, final_values, color=COLORS["blue"], alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in param_values])
        ax.set_xlabel(param_name)
        ax.set_ylabel(f"Final {metric_name}")
        ax.set_title(f"Sensitivity: {metric_name} vs {param_name}")
        fig.tight_layout()

        save_id = output_run_id or run_ids[0]
        return artifact_store.save_figure(save_id, fig, f"fig_sensitivity_{metric_key}")

    def generate_benchmark_overlay(
        self,
        run_id: str,
        artifact_store,
    ) -> Optional[str]:
        """Generate benchmark comparison overlay from eval report.

        Plots simulated vs observed for each benchmark comparison.
        """
        import matplotlib.pyplot as plt

        report = artifact_store.load_eval_report(run_id)
        accuracy = report.get("accuracy", {})
        comparisons = accuracy.get("comparisons", [])

        if not comparisons:
            return None

        n = len(comparisons)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if n == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, comp in enumerate(comparisons):
            if i >= len(axes):
                break
            ax = axes[i]
            observed = comp.get("observed_values", [])
            simulated = comp.get("simulated_values", [])
            time_pts = comp.get("time_points", list(range(len(observed))))

            ax.plot(time_pts, observed, "o-", color=COLORS["black"],
                    linewidth=1.5, markersize=4, label="Observed")
            ax.plot(time_pts, simulated, "s--", color=COLORS["blue"],
                    linewidth=1.5, markersize=4, label="Simulated")
            ax.set_title(comp.get("benchmark_name", f"Benchmark {i+1}"), fontsize=10)
            ax.legend(fontsize=8)

            # Show R² in corner
            r2 = comp.get("r_squared")
            if r2 is not None:
                ax.text(0.95, 0.05, f"R²={r2:.3f}", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # Hide unused axes
        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Benchmark Comparisons", fontsize=13, y=1.02)
        fig.tight_layout()

        return artifact_store.save_figure(run_id, fig, "fig_benchmark_comparison")
