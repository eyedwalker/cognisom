"""
Single-Cell Bridge
==================

Converts extracted cell archetypes into cognisom simulation initial conditions.
This is the critical translation layer between real scRNA-seq data and the
simulation engine.

Produces:
    - CellState objects with realistic positions, types, and metabolism
    - ImmuneCell objects for immune archetypes
    - SpatialField configurations based on tissue structure
    - Gene expression parameters for the molecular module
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .archetypes import CellArchetype

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Complete initial conditions for a cognisom simulation,
    derived from scRNA-seq archetypes."""

    # Cells to create
    cells: List[Dict] = field(default_factory=list)
    # Immune cells to create
    immune_cells: List[Dict] = field(default_factory=list)
    # Spatial field configuration
    spatial_config: Dict = field(default_factory=dict)
    # Module parameter overrides
    module_params: Dict = field(default_factory=dict)
    # Source archetype info
    source_info: Dict = field(default_factory=dict)


class SingleCellBridge:
    """Convert scRNA-seq archetypes to simulation initial conditions.

    Takes the output of ArchetypeExtractor and produces configuration
    that can initialize a cognisom simulation with biologically realistic
    cell populations, proportions, and spatial arrangements.

    Example:
        from cognisom.ingestion import ScRNALoader, ScRNAPreprocessor
        from cognisom.ingestion import ArchetypeExtractor, SingleCellBridge

        # Load and process
        loader = ScRNALoader()
        adata = loader.from_cellxgene(tissue="prostate gland")
        adata = ScRNAPreprocessor().run(adata)
        archetypes = ArchetypeExtractor().extract(adata)

        # Bridge to simulation
        bridge = SingleCellBridge()
        config = bridge.create_config(archetypes, total_cells=500)

        # Use config to initialize simulation
        engine = SimulationEngine(...)
        bridge.apply_config(engine, config)
    """

    # Tissue geometry defaults (prostate acinus cross-section)
    DEFAULT_GEOMETRY = {
        "domain_size": (200.0, 200.0, 100.0),  # um
        "lumen_center": (100.0, 100.0, 50.0),
        "lumen_radius": 30.0,  # um
        "epithelial_thickness": 20.0,  # um
        "stromal_zone_start": 50.0,  # um from center
        "vessel_positions": [
            (160.0, 100.0, 50.0),
            (40.0, 100.0, 50.0),
            (100.0, 160.0, 50.0),
            (100.0, 40.0, 50.0),
        ],
    }

    def create_config(self, archetypes: List[CellArchetype],
                      total_cells: int = 500,
                      geometry: Optional[Dict] = None,
                      include_cancer: bool = False) -> SimulationConfig:
        """Create simulation configuration from archetypes.

        Args:
            archetypes: List of CellArchetype from ArchetypeExtractor.
            total_cells: Total number of cells to place in simulation.
            geometry: Tissue geometry overrides.
            include_cancer: Whether to include cancer archetypes.

        Returns:
            SimulationConfig ready to initialize a simulation.
        """
        geo = {**self.DEFAULT_GEOMETRY, **(geometry or {})}

        # Filter archetypes
        valid = [a for a in archetypes if include_cancer or a.simulation_type != "cancer"]
        if not valid:
            raise ValueError("No valid archetypes after filtering")

        # Recalculate proportions
        total_src = sum(a.cell_count for a in valid)
        proportions = {a.name: a.cell_count / total_src for a in valid}

        config = SimulationConfig()
        config.source_info = {
            "n_archetypes": len(valid),
            "total_source_cells": total_src,
            "proportions": proportions,
        }

        # Generate cells for each archetype
        for archetype in valid:
            n_cells = max(1, int(total_cells * proportions[archetype.name]))

            if archetype.simulation_type == "immune":
                self._place_immune_cells(
                    config, archetype, n_cells, geo
                )
            else:
                self._place_tissue_cells(
                    config, archetype, n_cells, geo
                )

        # Configure spatial fields based on tissue structure
        config.spatial_config = self._build_spatial_config(geo, valid)

        # Configure module parameters
        config.module_params = self._build_module_params(valid)

        logger.info(
            f"Created config: {len(config.cells)} tissue cells, "
            f"{len(config.immune_cells)} immune cells"
        )
        return config

    def _place_tissue_cells(self, config: SimulationConfig,
                            archetype: CellArchetype, n_cells: int,
                            geo: Dict):
        """Place tissue cells with biologically plausible spatial arrangement."""
        center = np.array(geo["lumen_center"])
        lumen_r = geo["lumen_radius"]
        epi_thick = geo["epithelial_thickness"]
        domain = geo["domain_size"]

        for i in range(n_cells):
            if archetype.name in ("luminal_epithelial", "basal_epithelial"):
                # Epithelial cells: ring around lumen
                angle = np.random.uniform(0, 2 * np.pi)
                if archetype.name == "luminal_epithelial":
                    # Inner layer (closer to lumen)
                    r = lumen_r + np.random.uniform(2, epi_thick * 0.5)
                else:
                    # Outer layer (basal)
                    r = lumen_r + np.random.uniform(epi_thick * 0.5, epi_thick)
                x = center[0] + r * np.cos(angle)
                y = center[1] + r * np.sin(angle)
                z = center[2] + np.random.uniform(-10, 10)

            elif archetype.name == "endothelial":
                # Near blood vessels
                vessel = geo["vessel_positions"][i % len(geo["vessel_positions"])]
                x = vessel[0] + np.random.uniform(-5, 5)
                y = vessel[1] + np.random.uniform(-5, 5)
                z = vessel[2] + np.random.uniform(-5, 5)

            elif archetype.name in ("stromal_fibroblast", "smooth_muscle"):
                # Stromal zone (outside epithelium)
                angle = np.random.uniform(0, 2 * np.pi)
                r = geo["stromal_zone_start"] + np.random.uniform(10, 40)
                x = center[0] + r * np.cos(angle)
                y = center[1] + r * np.sin(angle)
                z = center[2] + np.random.uniform(-20, 20)

            elif archetype.name == "cancer_epithelial":
                # Cancer cluster (focal)
                cancer_center = center + np.array([20, 20, 0])
                x = cancer_center[0] + np.random.normal(0, 10)
                y = cancer_center[1] + np.random.normal(0, 10)
                z = cancer_center[2] + np.random.normal(0, 5)

            else:
                # Default: random within domain
                x = np.random.uniform(10, domain[0] - 10)
                y = np.random.uniform(10, domain[1] - 10)
                z = np.random.uniform(10, domain[2] - 10)

            # Clamp to domain
            x = np.clip(x, 0, domain[0])
            y = np.clip(y, 0, domain[1])
            z = np.clip(z, 0, domain[2])

            cell_config = {
                "position": [float(x), float(y), float(z)],
                "cell_type": archetype.simulation_type,
                "archetype": archetype.name,
                "oxygen": 0.21,
                "glucose": 5.0,
                "mhc1_expression": 0.3 if archetype.simulation_type == "cancer" else 1.0,
                "mutations": archetype.params.get("mutations", []),
            }
            config.cells.append(cell_config)

    def _place_immune_cells(self, config: SimulationConfig,
                            archetype: CellArchetype, n_cells: int,
                            geo: Dict):
        """Place immune cells in tissue or near vessels."""
        center = np.array(geo["lumen_center"])
        domain = geo["domain_size"]
        vessels = geo["vessel_positions"]

        for i in range(n_cells):
            # Immune cells patrol near vessels or through stroma
            if np.random.random() < 0.3:
                # Near a vessel (recently extravasated)
                vessel = vessels[i % len(vessels)]
                x = vessel[0] + np.random.uniform(-15, 15)
                y = vessel[1] + np.random.uniform(-15, 15)
                z = vessel[2] + np.random.uniform(-10, 10)
            else:
                # Patrolling through stroma
                angle = np.random.uniform(0, 2 * np.pi)
                r = np.random.uniform(30, 80)
                x = center[0] + r * np.cos(angle)
                y = center[1] + r * np.sin(angle)
                z = center[2] + np.random.uniform(-20, 20)

            x = np.clip(x, 0, domain[0])
            y = np.clip(y, 0, domain[1])
            z = np.clip(z, 0, domain[2])

            immune_config = {
                "position": [float(x), float(y), float(z)],
                "cell_type": archetype.params.get("immune_subtype", "T_cell"),
                "archetype": archetype.name,
                "speed": archetype.params.get("speed", 10.0),
                "detection_radius": archetype.params.get("detection_radius", 10.0),
            }
            config.immune_cells.append(immune_config)

    def _build_spatial_config(self, geo: Dict,
                              archetypes: List[CellArchetype]) -> Dict:
        """Build spatial field configuration from tissue geometry."""
        return {
            "grid_size": (40, 40, 20),
            "resolution": 5.0,  # um per voxel
            "fields": {
                "oxygen": {
                    "diffusion_coeff": 2000.0,  # um^2/s
                    "initial_value": 0.21,
                    "sources": [
                        {"position": v, "rate": 0.5}
                        for v in geo["vessel_positions"]
                    ],
                },
                "glucose": {
                    "diffusion_coeff": 600.0,
                    "initial_value": 5.0,
                    "sources": [
                        {"position": v, "rate": 0.3}
                        for v in geo["vessel_positions"]
                    ],
                },
                "cytokine": {
                    "diffusion_coeff": 100.0,
                    "initial_value": 0.0,
                    "decay_rate": 0.01,
                },
            },
        }

    def _build_module_params(self, archetypes: List[CellArchetype]) -> Dict:
        """Build module parameter overrides from archetype data."""
        # Calculate weighted averages for metabolic parameters
        tissue = [a for a in archetypes if a.simulation_type != "immune"]
        immune = [a for a in archetypes if a.simulation_type == "immune"]

        total_tissue = sum(a.cell_count for a in tissue) or 1
        total_immune = sum(a.cell_count for a in immune) or 1

        immune_proportions = {}
        for a in immune:
            subtype = a.params.get("immune_subtype", "T_cell")
            immune_proportions[subtype] = a.cell_count / total_immune

        return {
            "cellular": {
                "tissue_composition": {
                    a.name: a.cell_count / total_tissue for a in tissue
                },
            },
            "immune": {
                "immune_composition": immune_proportions,
                "total_immune_fraction": total_immune / (total_tissue + total_immune),
            },
            "molecular": {
                "marker_genes": {
                    a.name: a.top_genes[:5] for a in archetypes
                },
            },
        }

    def apply_config(self, simulation_engine, config: SimulationConfig):
        """Apply a SimulationConfig to a running simulation engine.

        Clears existing cells and reinitializes from the config.

        Args:
            simulation_engine: cognisom SimulationEngine instance.
            config: SimulationConfig from create_config().
        """
        # Apply tissue cells to cellular module
        cellular = simulation_engine.get_module("cellular")
        if cellular:
            cellular.cells.clear()
            cellular.next_cell_id = 0
            for cell_cfg in config.cells:
                cell_id = cellular.add_cell(
                    position=cell_cfg["position"],
                    cell_type=cell_cfg["cell_type"],
                )
                cell = cellular.cells[cell_id]
                cell.oxygen = cell_cfg.get("oxygen", 0.21)
                cell.glucose = cell_cfg.get("glucose", 5.0)
                cell.mhc1_expression = cell_cfg.get("mhc1_expression", 1.0)
                cell.mutations = cell_cfg.get("mutations", [])

        # Apply immune cells to immune module
        immune = simulation_engine.get_module("immune")
        if immune:
            immune.immune_cells.clear()
            immune.next_immune_id = 0
            for ic_cfg in config.immune_cells:
                immune_id = immune.next_immune_id
                immune.next_immune_id += 1
                from modules.immune_module import ImmuneCell
                ic = ImmuneCell(
                    cell_id=immune_id,
                    position=np.array(ic_cfg["position"], dtype=np.float32),
                    cell_type=ic_cfg["cell_type"],
                    speed=ic_cfg.get("speed", 10.0),
                    detection_radius=ic_cfg.get("detection_radius", 10.0),
                )
                immune.immune_cells[immune_id] = ic

        logger.info(
            f"Applied config: {len(config.cells)} tissue cells, "
            f"{len(config.immune_cells)} immune cells"
        )
