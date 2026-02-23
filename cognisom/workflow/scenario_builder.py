"""
Scenario Builder
=================

High-level API for constructing SimulationScenarios from enriched entities
in the EntityStore. Provides both a generic builder and preset scenario
builders for common simulation setups.

Usage:
    store = EntityStore()
    builder = ScenarioBuilder(store)

    # Generic
    scenario = builder.build_scenario(
        name="My AR Simulation",
        scenario_type="drug_response",
        gene_names=["AR", "TP53", "PTEN"],
        drug_names=["Enzalutamide"],
        duration_hours=72.0,
    )

    # Preset
    scenario = builder.build_prostate_cancer_scenario()
    scenario = builder.build_immunotherapy_scenario("Pembrolizumab")
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from cognisom.library.models import (
    BioEntity,
    Drug,
    Gene,
    ParameterSet,
    SimulationScenario,
)
from cognisom.library.store import EntityStore

log = logging.getLogger(__name__)


class ScenarioBuilder:
    """Build SimulationScenarios from enriched entities in EntityStore.

    The builder looks up entities by name, resolves their IDs, and
    optionally auto-generates ParameterSets based on entity properties.
    """

    def __init__(self, store: EntityStore):
        self._store = store

    # ── Generic Builder ────────────────────────────────────────────────

    def build_scenario(
        self,
        name: str,
        scenario_type: str = "baseline",
        gene_names: Optional[List[str]] = None,
        drug_names: Optional[List[str]] = None,
        cell_type_names: Optional[List[str]] = None,
        parameter_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        duration_hours: float = 48.0,
        dt: float = 0.01,
        grid_shape: Optional[List[int]] = None,
        resolution_um: float = 10.0,
        initial_cell_counts: Optional[Dict[str, int]] = None,
        initial_field_values: Optional[Dict[str, float]] = None,
        auto_generate_params: bool = True,
    ) -> SimulationScenario:
        """Build a SimulationScenario from entity names.

        Args:
            name: Human-readable scenario name.
            scenario_type: Category (baseline, drug_response, immune_challenge, etc.)
            gene_names: Gene symbols to include (looked up in EntityStore).
            drug_names: Drug names to include.
            cell_type_names: Cell type names to include.
            parameter_overrides: Manual per-module parameter overrides.
            duration_hours: Simulation duration.
            dt: Time step in hours.
            grid_shape: Spatial grid dimensions [x, y, z].
            resolution_um: Voxel size in micrometers.
            initial_cell_counts: Starting cell counts by type.
            initial_field_values: Starting field concentrations.
            auto_generate_params: If True, create ParameterSets from entity data.

        Returns:
            A SimulationScenario entity persisted in the store.
        """
        # Resolve entity names → IDs
        gene_ids = self._resolve_names(gene_names or [], "gene")
        drug_ids = self._resolve_names(drug_names or [], "drug")
        cell_type_ids = self._resolve_names(cell_type_names or [], "cell_type")

        # Auto-generate ParameterSets
        param_set_ids = []
        if auto_generate_params:
            param_set_ids = self._generate_parameter_sets(
                gene_ids, drug_ids, cell_type_ids,
                context=scenario_type,
                overrides=parameter_overrides,
            )

        # Build scenario entity
        scenario = SimulationScenario(
            name=name,
            display_name=name,
            description=f"{scenario_type} scenario with "
                        f"{len(gene_ids)} genes, {len(drug_ids)} drugs",
            scenario_type=scenario_type,
            duration_hours=duration_hours,
            time_step_hours=dt,
            grid_shape=grid_shape or [200, 200, 100],
            resolution_um=resolution_um,
            cell_type_ids=cell_type_ids,
            gene_ids=gene_ids,
            drug_ids=drug_ids,
            parameter_set_ids=param_set_ids,
            initial_cell_counts=initial_cell_counts or {},
            initial_field_values=initial_field_values or {},
            source="scenario_builder",
            tags=[scenario_type],
        )

        self._store.add_entity(scenario)
        log.info(
            "Built scenario '%s': %d genes, %d drugs, %d param sets",
            name, len(gene_ids), len(drug_ids), len(param_set_ids),
        )
        return scenario

    # ── Preset Builders ────────────────────────────────────────────────

    def build_prostate_cancer_scenario(
        self,
        duration_hours: float = 72.0,
    ) -> SimulationScenario:
        """Prostate cancer baseline with AR signaling, key genes, and drugs."""
        return self.build_scenario(
            name="Prostate Cancer Baseline",
            scenario_type="baseline",
            gene_names=[
                "AR", "TP53", "PTEN", "MYC", "RB1", "BRCA1", "BRCA2",
                "ERG", "TMPRSS2", "FOXA1", "CDK12", "SPOP", "KMT2D",
                "PIK3CA", "AKT1", "CDKN2A", "MDM2",
            ],
            drug_names=["Enzalutamide", "Abiraterone", "Docetaxel"],
            initial_cell_counts={
                "normal": 80,
                "cancer": 15,
                "t_cell": 12,
                "nk_cell": 8,
                "macrophage": 6,
            },
            initial_field_values={
                "oxygen": 0.21,
                "glucose": 5.0,
            },
            duration_hours=duration_hours,
        )

    def build_immune_challenge_scenario(
        self,
        pathogen_type: str = "viral",
        duration_hours: float = 96.0,
    ) -> SimulationScenario:
        """Immune response to pathogen challenge."""
        return self.build_scenario(
            name=f"Immune Challenge ({pathogen_type})",
            scenario_type="immune_challenge",
            gene_names=[
                # Innate
                "TLR3", "TLR7", "TLR9", "MYD88", "IRF3", "IRF7",
                "IFNB1", "IFNG", "TNF", "IL6", "IL1B",
                # Adaptive
                "CD3E", "CD4", "CD8A", "CD28", "PDCD1",
                "CD19", "IGHM", "IGHG1",
                # Antigen presentation
                "HLA-A", "HLA-DRA", "TAP1", "B2M",
            ],
            initial_cell_counts={
                "normal": 100,
                "t_cell": 20,
                "nk_cell": 15,
                "macrophage": 10,
            },
            parameter_overrides={
                "immune": {
                    "n_t_cells": 20,
                    "n_nk_cells": 15,
                    "n_macrophages": 10,
                    "kill_probability": 0.85,
                },
            },
            duration_hours=duration_hours,
        )

    def build_immunotherapy_scenario(
        self,
        drug_name: str = "Pembrolizumab",
        duration_hours: float = 72.0,
    ) -> SimulationScenario:
        """Tumor + immunotherapy drug simulation."""
        return self.build_scenario(
            name=f"Immunotherapy ({drug_name})",
            scenario_type="immunotherapy",
            gene_names=[
                "AR", "TP53", "PTEN", "MYC",
                "PDCD1", "CD274", "CTLA4", "LAG3", "HAVCR2",
                "CD8A", "CD4", "IFNG", "TNF", "PRF1", "GZMB",
            ],
            drug_names=[drug_name],
            initial_cell_counts={
                "normal": 70,
                "cancer": 20,
                "t_cell": 25,
                "nk_cell": 12,
                "macrophage": 8,
            },
            duration_hours=duration_hours,
        )

    def build_drug_response_scenario(
        self,
        drug_name: str,
        gene_context: Optional[List[str]] = None,
        duration_hours: float = 48.0,
    ) -> SimulationScenario:
        """Drug response with optional gene context."""
        genes = gene_context or ["AR", "TP53", "PTEN"]
        return self.build_scenario(
            name=f"Drug Response ({drug_name})",
            scenario_type="drug_response",
            gene_names=genes,
            drug_names=[drug_name],
            initial_cell_counts={
                "normal": 80,
                "cancer": 15,
            },
            duration_hours=duration_hours,
        )

    def build_hypoxia_scenario(
        self,
        duration_hours: float = 48.0,
    ) -> SimulationScenario:
        """Hypoxic tumor microenvironment."""
        return self.build_scenario(
            name="Hypoxia",
            scenario_type="hypoxia",
            gene_names=[
                "HIF1A", "VEGFA", "LDHA", "PDK1", "SLC2A1",
                "TP53", "PTEN", "MYC",
            ],
            initial_cell_counts={
                "normal": 60,
                "cancer": 25,
                "t_cell": 8,
                "nk_cell": 5,
                "macrophage": 4,
            },
            initial_field_values={
                "oxygen": 0.05,  # Hypoxic
                "glucose": 3.0,
            },
            parameter_overrides={
                "vascular": {"n_capillaries": 3, "exchange_rate": 0.3},
            },
            duration_hours=duration_hours,
        )

    # ── Internal Helpers ───────────────────────────────────────────────

    def _resolve_names(
        self, names: List[str], entity_type: str,
    ) -> List[str]:
        """Look up entity names in store, return list of entity IDs."""
        ids = []
        for name in names:
            entity = self._store.find_entity_by_name(name, entity_type)
            if entity:
                ids.append(entity.entity_id)
            else:
                log.debug(
                    "Entity '%s' (type=%s) not in store — skipping",
                    name, entity_type,
                )
        return ids

    def _generate_parameter_sets(
        self,
        gene_ids: List[str],
        drug_ids: List[str],
        cell_type_ids: List[str],
        context: str,
        overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[str]:
        """Auto-generate ParameterSets based on entity properties.

        Creates one ParameterSet per module that has relevant parameters
        derived from the scenario's entities.
        """
        from cognisom.workflow.parameter_bridge import (
            ParameterBridge,
            BASELINE_DEFAULTS,
            DRUG_CLASS_PARAM_MAP,
            DRUG_TARGET_PARAM_MAP,
            GENE_SPECIFIC_PARAM_MAP,
        )

        param_set_ids = []

        # Collect drug-derived params
        drug_params: Dict[str, Dict[str, Any]] = {}
        for drug_id in drug_ids:
            entity = self._store.get_entity(drug_id)
            if not isinstance(entity, Drug):
                continue
            drug_class = (entity.drug_class or "").lower().replace("-", "_").replace(" ", "_")

            for target in (entity.targets or []):
                target_upper = target.upper()
                if target_upper in DRUG_TARGET_PARAM_MAP:
                    for mod, params in DRUG_TARGET_PARAM_MAP[target_upper].items():
                        drug_params.setdefault(mod, {}).update(params)

            if drug_class in DRUG_CLASS_PARAM_MAP:
                for mod, params in DRUG_CLASS_PARAM_MAP[drug_class].items():
                    drug_params.setdefault(mod, {}).update(params)

        # Collect gene-derived params
        gene_params: Dict[str, Dict[str, Any]] = {}
        for gene_id in gene_ids:
            entity = self._store.get_entity(gene_id)
            if not isinstance(entity, Gene):
                continue
            symbol = (entity.symbol or entity.name or "").upper()
            if symbol in GENE_SPECIFIC_PARAM_MAP:
                for mod, params in GENE_SPECIFIC_PARAM_MAP[symbol].items():
                    gene_params.setdefault(mod, {}).update(params)

        # Merge with overrides
        all_params: Dict[str, Dict[str, Any]] = {}
        for source in [drug_params, gene_params]:
            for mod, params in source.items():
                all_params.setdefault(mod, {}).update(params)
        if overrides:
            for mod, params in overrides.items():
                all_params.setdefault(mod, {}).update(params)

        # Create a ParameterSet for each module with parameters
        for module, params in all_params.items():
            if not params:
                continue

            ps = ParameterSet(
                name=f"{context}_{module}_params",
                display_name=f"{context.replace('_', ' ').title()} — {module}",
                description=f"Auto-generated parameters for {module} module in {context} context",
                context=context,
                module=module,
                parameters={k: float(v) if isinstance(v, (int, float)) else v
                            for k, v in params.items()
                            if isinstance(v, (int, float))},
                source="scenario_builder",
                tags=["auto_generated", context],
            )

            # Store string params separately (system, method, etc.)
            # These go into parameters as well but aren't floats
            for k, v in params.items():
                if isinstance(v, str):
                    ps.parameters[k] = v  # type: ignore

            self._store.add_entity(ps)
            param_set_ids.append(ps.entity_id)

            log.debug(
                "Generated ParameterSet '%s' for module=%s: %s",
                ps.name, module, params,
            )

        return param_set_ids

    # ── Utilities ──────────────────────────────────────────────────────

    def list_available_genes(self, limit: int = 100) -> List[str]:
        """List gene symbols available in the store."""
        genes, _ = self._store.search(entity_type="gene", limit=limit)
        return [
            getattr(g, "symbol", None) or g.name
            for g in genes
            if isinstance(g, Gene)
        ]

    def list_available_drugs(self, limit: int = 100) -> List[str]:
        """List drug names available in the store."""
        drugs, _ = self._store.search(entity_type="drug", limit=limit)
        return [d.name for d in drugs]
