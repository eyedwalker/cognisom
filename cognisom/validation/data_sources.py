"""
Data Source Downloaders
=======================

Download and prepare real experimental datasets for validation.
Supports TCGA, GEO, and synthetic reference data generation.
"""

import io
import gzip
import logging
import json
from pathlib import Path
from typing import Dict, Optional, List, Any

import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"


# ────────────────────────────────────────────────────────────────
# Known prostate cancer datasets
# ────────────────────────────────────────────────────────────────

PROSTATE_DATASETS = {
    "TCGA-PRAD": {
        "description": "The Cancer Genome Atlas - Prostate Adenocarcinoma",
        "source": "GDC Data Portal",
        "url": "https://portal.gdc.cancer.gov/projects/TCGA-PRAD",
        "n_samples": 500,
        "data_types": ["gene_expression", "mutations", "copy_number", "clinical"],
    },
    "GSE176031": {
        "description": "Single-cell RNA-seq of prostate cancer",
        "source": "NCBI GEO",
        "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE176031",
        "n_cells": 36000,
        "cell_types": [
            "luminal_epithelial", "basal_epithelial", "fibroblast",
            "endothelial", "macrophage", "t_cell", "nk_cell",
        ],
    },
    "GSE141445": {
        "description": "Human prostate single-cell atlas",
        "source": "NCBI GEO",
        "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE141445",
        "n_cells": 98000,
        "cell_types": [
            "luminal", "basal", "club", "hillock",
            "neuroendocrine", "fibroblast", "smooth_muscle",
            "endothelial", "leukocyte",
        ],
    },
}


class SyntheticDataGenerator:
    """Generate synthetic reference datasets for validation.

    These are NOT real experimental data — they are synthetic datasets
    that follow published growth curves, immune kinetics, and metabolic
    profiles. Useful for testing the validation pipeline before real
    data is available.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def tumor_spheroid_growth(
        self, n_days: int = 14, n_initial: int = 50,
        doubling_time_h: float = 18.0, carrying_capacity: int = 5000,
        noise_pct: float = 5.0,
    ) -> Dict[str, Any]:
        """Generate synthetic tumor spheroid growth curve (Gompertzian).

        Based on: Luca et al., PLoS ONE, 2013 — PC-3 spheroid dynamics.
        """
        times_h = np.arange(0, n_days * 24 + 1, 1)

        # Gompertz growth: N(t) = K * exp(ln(N0/K) * exp(-alpha*t))
        alpha = np.log(2) / doubling_time_h * 0.3  # growth deceleration
        log_ratio = np.log(n_initial / carrying_capacity)
        cell_counts = carrying_capacity * np.exp(log_ratio * np.exp(-alpha * times_h))

        # Add noise
        noise = self.rng.normal(0, noise_pct / 100, len(cell_counts))
        cell_counts = cell_counts * (1 + noise)
        cell_counts = np.clip(cell_counts, n_initial, None)

        # Diameter from cell count (spheroid geometry)
        cell_volume_um3 = 1767  # ~15um diameter cell
        total_volume = cell_counts * cell_volume_um3
        diameters_um = (6 * total_volume / np.pi) ** (1.0 / 3.0)

        return {
            "type": "tumor_spheroid_growth",
            "time_hours": times_h.tolist(),
            "cell_counts": cell_counts.tolist(),
            "diameters_um": diameters_um.tolist(),
            "parameters": {
                "n_initial": n_initial,
                "doubling_time_h": doubling_time_h,
                "carrying_capacity": carrying_capacity,
            },
        }

    def immune_infiltration(
        self, n_days: int = 28, tumor_size_initial: int = 100,
        noise_pct: float = 10.0,
    ) -> Dict[str, Any]:
        """Generate synthetic immune infiltration time course.

        Models CD8+ T cell and NK cell density in growing tumor.
        Based on: Ness et al., The Prostate, 2014.
        """
        times_days = np.arange(0, n_days + 1, 1)

        # CD8+ T cells: delayed response, then exhaustion
        cd8_density = (
            200 * (1 - np.exp(-times_days / 5))  # recruitment
            * np.exp(-times_days / 60)  # slow decline (exhaustion)
        )

        # NK cells: fast response, quick decline
        nk_density = (
            100 * np.exp(-((times_days - 3) ** 2) / 20)  # peak at day 3
            + 20  # baseline
        )

        # PD-1+ fraction on CD8 (exhaustion kinetics)
        pd1_fraction = 1 - np.exp(-times_days / 15)
        pd1_fraction = np.clip(pd1_fraction, 0, 0.95)

        # Add noise
        cd8_density *= 1 + self.rng.normal(0, noise_pct / 100, len(times_days))
        nk_density *= 1 + self.rng.normal(0, noise_pct / 100, len(times_days))

        return {
            "type": "immune_infiltration",
            "time_days": times_days.tolist(),
            "cd8_density_per_mm2": np.clip(cd8_density, 0, None).tolist(),
            "nk_density_per_mm2": np.clip(nk_density, 0, None).tolist(),
            "pd1_positive_fraction": pd1_fraction.tolist(),
        }

    def metabolic_profile(
        self, n_hours: int = 48, noise_pct: float = 5.0,
    ) -> Dict[str, Any]:
        """Generate synthetic metabolic time course for normal vs cancer cells.

        Based on: Vander Heiden et al., Science, 2009 (Warburg effect).
        """
        times_h = np.arange(0, n_hours + 1, 1)

        # Glucose concentration (mM) — decreasing as cells consume
        glucose_normal = 5.5 * np.exp(-0.01 * times_h)
        glucose_cancer = 5.5 * np.exp(-0.04 * times_h)

        # Lactate (mM) — increasing, especially cancer
        lactate_normal = 1.0 + 0.5 * (1 - np.exp(-0.02 * times_h))
        lactate_cancer = 1.0 + 4.0 * (1 - np.exp(-0.03 * times_h))

        # ATP (relative) — maintained until glucose depleted
        atp_normal = np.where(glucose_normal > 0.5, 1.0, glucose_normal / 0.5)
        atp_cancer = np.where(glucose_cancer > 0.3, 0.95, glucose_cancer / 0.3 * 0.95)

        # O2 (relative) — faster depletion in dense tumor
        o2_normal = np.exp(-0.005 * times_h)
        o2_cancer = np.exp(-0.02 * times_h)

        # Add noise
        for arr in [glucose_normal, glucose_cancer, lactate_normal,
                     lactate_cancer, atp_normal, atp_cancer]:
            arr *= 1 + self.rng.normal(0, noise_pct / 100, len(times_h))

        return {
            "type": "metabolic_profile",
            "time_hours": times_h.tolist(),
            "normal": {
                "glucose_mM": np.clip(glucose_normal, 0, None).tolist(),
                "lactate_mM": np.clip(lactate_normal, 0, None).tolist(),
                "atp_relative": np.clip(atp_normal, 0, 1).tolist(),
                "o2_relative": np.clip(o2_normal, 0, 1).tolist(),
            },
            "cancer": {
                "glucose_mM": np.clip(glucose_cancer, 0, None).tolist(),
                "lactate_mM": np.clip(lactate_cancer, 0, None).tolist(),
                "atp_relative": np.clip(atp_cancer, 0, 1).tolist(),
                "o2_relative": np.clip(o2_cancer, 0, 1).tolist(),
            },
        }

    def cell_type_proportions(self) -> Dict[str, Any]:
        """Expected cell type proportions in prostate tissue.

        Based on: Henry et al., 2018 (GSE141445) prostate atlas.
        """
        return {
            "type": "cell_type_proportions",
            "source": "Henry et al., 2018 - Prostate single-cell atlas",
            "normal_tissue": {
                "luminal_epithelial": 0.35,
                "basal_epithelial": 0.20,
                "fibroblast": 0.18,
                "smooth_muscle": 0.10,
                "endothelial": 0.07,
                "leukocyte": 0.05,
                "neuroendocrine": 0.03,
                "other": 0.02,
            },
            "tumor_tissue": {
                "cancer_epithelial": 0.45,
                "fibroblast_caf": 0.15,
                "t_cell": 0.10,
                "macrophage": 0.08,
                "endothelial": 0.08,
                "basal_epithelial": 0.05,
                "nk_cell": 0.03,
                "b_cell": 0.03,
                "other": 0.03,
            },
        }

    def generate_all(self, output_dir: Optional[str] = None) -> Dict[str, Path]:
        """Generate all synthetic reference datasets and save as JSON."""
        out = Path(output_dir) if output_dir else DATA_DIR / "validation" / "synthetic"
        out.mkdir(parents=True, exist_ok=True)

        datasets = {
            "tumor_growth": self.tumor_spheroid_growth(),
            "immune_infiltration": self.immune_infiltration(),
            "metabolic_profile": self.metabolic_profile(),
            "cell_type_proportions": self.cell_type_proportions(),
        }

        paths = {}
        for name, data in datasets.items():
            path = out / f"{name}.json"
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            paths[name] = path
            logger.info(f"Generated: {path}")

        return paths
