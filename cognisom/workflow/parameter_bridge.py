"""
Parameter Bridge
=================

Resolves entity-driven SimulationScenarios into concrete module config dicts
that EngineRunner can consume. This is the critical translation layer between
the enriched entity database and the simulation engine.

Resolution priority (lowest → highest):
    1. Baseline defaults (from SCENARIOS["Baseline"])
    2. ParameterSet entities (grouped by module field)
    3. Gene-derived parameters (oncogene/tumor_suppressor → rate overrides)
    4. Drug-derived parameters (mechanism → inhibition constants)
    5. Scenario initial_cell_counts / initial_field_values

Usage:
    from cognisom.workflow.parameter_bridge import ParameterBridge

    bridge = ParameterBridge(store)
    module_configs = bridge.resolve_scenario(scenario)
    # module_configs is Dict[str, Dict[str, Any]] ready for EngineRunner
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from cognisom.library.models import (
    BioEntity,
    CellTypeEntity,
    Drug,
    Gene,
    ParameterSet,
    Protein,
    SimulationScenario,
)
from cognisom.library.store import EntityStore

log = logging.getLogger(__name__)


# ── Module Parameter Schema ────────────────────────────────────────────
# Documents valid parameter names per module for validation and debugging.

MODULE_PARAM_SCHEMA: Dict[str, Dict[str, type]] = {
    "cellular": {
        "n_normal_cells": int, "n_cancer_cells": int,
        "division_time_normal": float, "division_time_cancer": float,
        "glucose_consumption": float, "glucose_consumption_cancer": float,
    },
    "immune": {
        "n_t_cells": int, "n_nk_cells": int, "n_macrophages": int,
        "patrol_speed": float, "kill_probability": float,
    },
    "vascular": {
        "n_capillaries": int, "arterial_O2": float,
        "arterial_glucose": float, "exchange_rate": float,
    },
    "lymphatic": {
        "n_vessels": int, "drainage_rate": float,
        "metastasis_probability": float, "immune_trafficking_rate": float,
    },
    "molecular": {
        "transcription_rate": float, "translation_rate": float,
        "exosome_release_rate": float, "mutation_rate": float,
    },
    "spatial": {
        "grid_size": tuple, "resolution": float, "update_interval": float,
    },
    "epigenetic": {
        "methylation_rate": float, "demethylation_rate": float,
        "histone_mod_rate": float, "cancer_hypermethylation": bool,
        "hypoxia_methylation": bool,
    },
    "circadian": {
        "coupling_strength": float, "amplitude": float, "enable_gating": bool,
    },
    "morphogen": {
        "tissue_size": tuple, "enable_fate_determination": bool,
    },
    "ode": {
        "system": str, "n_cells": int, "method": str,
        "rtol": float, "atol": float, "heterogeneity": float,
    },
    "bngl": {
        "model": str, "method": str, "max_species": int,
        "ode_rtol": float, "ode_atol": float,
    },
    "hybrid": {
        "system": str, "n_cells": int, "threshold": float,
        "repartition_interval": int, "ode_method": str, "heterogeneity": float,
    },
    "smoldyn": {
        "system": str, "domain_size": tuple, "max_particles": int,
        "boundary": str, "species_counts": dict, "diffusion_coefficients": dict,
    },
}

# Baseline defaults (mirrors engine_runner.SCENARIOS["Baseline"])
BASELINE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "cellular": {"n_normal_cells": 80, "n_cancer_cells": 10},
    "immune": {"n_t_cells": 12, "n_nk_cells": 8, "n_macrophages": 6},
    "vascular": {"n_capillaries": 8},
    "lymphatic": {"n_vessels": 4, "metastasis_probability": 0.001},
    "molecular": {},
    "spatial": {},
    "epigenetic": {},
    "circadian": {},
    "morphogen": {},
}


# ── Drug-to-Parameter Mapping ──────────────────────────────────────────

# Maps (drug_class, mechanism_keyword) → module parameter overrides.
# Fallback: uses drug_class alone if mechanism doesn't match.

DRUG_CLASS_PARAM_MAP: Dict[str, Dict[str, Dict[str, Any]]] = {
    "anti-androgen": {
        "ode": {"system": "ar_signaling", "k_bind": 10.0},  # reduced from 100
        "cellular": {"division_time_cancer": 18.0},  # slowed growth
    },
    "chemotherapy": {
        "cellular": {"division_time_cancer": 24.0, "glucose_consumption_cancer": 0.8},
    },
    "immunotherapy": {
        "immune": {"n_t_cells": 30, "kill_probability": 0.95},
    },
    "checkpoint_inhibitor": {
        "immune": {"n_t_cells": 30, "n_nk_cells": 15, "kill_probability": 0.95},
    },
    "parp_inhibitor": {
        "cellular": {"division_time_cancer": 20.0},
        "molecular": {"mutation_rate": 0.01},
    },
    "kinase_inhibitor": {
        "molecular": {"transcription_rate": 0.5},
        "cellular": {"division_time_cancer": 16.0},
    },
    "anti-cytokine": {
        "immune": {"kill_probability": 0.6},
    },
    "tlr_agonist": {
        "immune": {"n_macrophages": 12, "kill_probability": 0.9},
    },
}

# Target-based overrides: if drug.targets contains these genes, apply these params
DRUG_TARGET_PARAM_MAP: Dict[str, Dict[str, Dict[str, Any]]] = {
    "AR": {
        "ode": {"system": "ar_signaling", "k_bind": 10.0},
    },
    "PDCD1": {  # PD-1
        "immune": {"n_t_cells": 30, "kill_probability": 0.95},
    },
    "CD274": {  # PD-L1
        "immune": {"n_t_cells": 25, "kill_probability": 0.92},
    },
    "CTLA4": {
        "immune": {"n_t_cells": 28, "kill_probability": 0.90},
    },
    "BRCA1": {
        "molecular": {"mutation_rate": 0.005},
    },
    "BRCA2": {
        "molecular": {"mutation_rate": 0.005},
    },
    "VEGFA": {
        "vascular": {"n_capillaries": 4, "exchange_rate": 0.5},
    },
    "ERBB2": {  # HER2
        "cellular": {"division_time_cancer": 8.0},
        "molecular": {"transcription_rate": 2.0},
    },
}


# ── Gene-to-Parameter Mapping ──────────────────────────────────────────

GENE_TYPE_PARAM_MAP: Dict[str, Dict[str, Dict[str, Any]]] = {
    "oncogene": {
        "molecular": {"transcription_rate": 1.5},
        "cellular": {"division_time_cancer": 10.0},
    },
    "tumor_suppressor": {
        # When a tumor suppressor is listed in a scenario, it may be
        # mutated/lost — model this as reduced constraint on growth
        "cellular": {"division_time_cancer": 8.0},
        "molecular": {"mutation_rate": 0.005},
    },
}

# Specific gene overrides (highest priority gene-level mapping)
GENE_SPECIFIC_PARAM_MAP: Dict[str, Dict[str, Dict[str, Any]]] = {
    "AR": {
        "ode": {"system": "ar_signaling"},
    },
    "TP53": {
        "cellular": {"division_time_cancer": 8.0},
        "molecular": {"mutation_rate": 0.008},
    },
    "MYC": {
        "molecular": {"transcription_rate": 2.0},
        "cellular": {"division_time_cancer": 6.0},
    },
    "PTEN": {
        "cellular": {"division_time_cancer": 9.0},
    },
    "RB1": {
        "cellular": {"division_time_cancer": 9.0},
    },
    "EGFR": {
        "molecular": {"transcription_rate": 1.8},
        "cellular": {"division_time_cancer": 8.0},
    },
    "KRAS": {
        "molecular": {"transcription_rate": 1.5},
        "cellular": {"division_time_cancer": 7.0},
    },
}


# ── ParameterBridge ────────────────────────────────────────────────────

class ParameterBridge:
    """Resolves entity-driven SimulationScenarios into simulation module configs.

    This bridge connects the enriched entity database to the simulation engine
    by translating ParameterSet entities, Gene properties, Drug mechanisms,
    and initial conditions into the flat config dicts that each simulation
    module expects.
    """

    def __init__(self, store: EntityStore):
        self._store = store

    def resolve_scenario(
        self,
        scenario: SimulationScenario,
        include_defaults: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """Resolve a SimulationScenario into module_configs for EngineRunner.

        Args:
            scenario: A SimulationScenario entity with parameter_set_ids,
                      gene_ids, drug_ids, cell_type_ids, initial conditions.
            include_defaults: If True, start from BASELINE_DEFAULTS.

        Returns:
            Dict mapping module name → parameter dict.
            Ready to pass as ``overrides`` to EngineRunner.
        """
        layers: List[Dict[str, Dict[str, Any]]] = []

        # Layer 1: Baseline defaults
        if include_defaults:
            layers.append(deepcopy(BASELINE_DEFAULTS))

        # Layer 2: ParameterSet entities
        if scenario.parameter_set_ids:
            ps_configs = self._resolve_parameter_sets(scenario.parameter_set_ids)
            if ps_configs:
                layers.append(ps_configs)

        # Layer 3: Gene-derived parameters
        if scenario.gene_ids:
            gene_configs = self._map_genes_to_params(scenario.gene_ids)
            if gene_configs:
                layers.append(gene_configs)

        # Layer 4: Drug-derived parameters
        if scenario.drug_ids:
            drug_configs = self._map_drugs_to_params(scenario.drug_ids)
            if drug_configs:
                layers.append(drug_configs)

        # Layer 5: Scenario initial conditions
        condition_configs = self._map_initial_conditions(
            scenario.initial_cell_counts,
            scenario.initial_field_values,
        )
        if condition_configs:
            layers.append(condition_configs)

        # Merge all layers (later layers override earlier)
        result = self._merge_configs(*layers)

        # Apply grid/resolution from scenario if set
        if scenario.grid_shape and scenario.grid_shape != [200, 200, 100]:
            result.setdefault("spatial", {})["grid_size"] = tuple(scenario.grid_shape)
        if scenario.resolution_um and scenario.resolution_um != 10.0:
            result.setdefault("spatial", {})["resolution"] = scenario.resolution_um

        log.info(
            "Resolved scenario %s: %d modules configured",
            scenario.name or scenario.entity_id,
            sum(1 for v in result.values() if v),
        )
        return result

    # ── Layer resolvers ────────────────────────────────────────────────

    def _resolve_parameter_sets(
        self, param_set_ids: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Load ParameterSet entities and group by module."""
        configs: Dict[str, Dict[str, Any]] = {}

        for ps_id in param_set_ids:
            entity = self._store.get_entity(ps_id)
            if not isinstance(entity, ParameterSet):
                log.warning("ParameterSet %s not found or wrong type", ps_id)
                continue

            # Validate ranges
            violations = entity.validate_ranges()
            if violations:
                log.warning(
                    "ParameterSet %s (%s) has range violations: %s",
                    ps_id, entity.name, violations,
                )

            module = entity.module or "ode"  # default to ODE if unspecified
            configs.setdefault(module, {}).update(entity.parameters)

            log.debug(
                "Loaded ParameterSet %s for module=%s: %d params",
                entity.name, module, len(entity.parameters),
            )

        return configs

    def _map_drugs_to_params(
        self, drug_ids: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Convert Drug entities to simulation parameters."""
        configs: Dict[str, Dict[str, Any]] = {}

        for drug_id in drug_ids:
            entity = self._store.get_entity(drug_id)
            if not isinstance(entity, Drug):
                log.warning("Drug entity %s not found", drug_id)
                continue

            drug_name = entity.name
            drug_class = (entity.drug_class or "").lower().replace("-", "_").replace(" ", "_")
            mechanism = (entity.mechanism or "").lower()

            # Strategy 1: Target-based mapping (highest specificity)
            for target in (entity.targets or []):
                target_upper = target.upper()
                if target_upper in DRUG_TARGET_PARAM_MAP:
                    target_params = DRUG_TARGET_PARAM_MAP[target_upper]
                    for mod, params in target_params.items():
                        configs.setdefault(mod, {}).update(params)
                    log.debug("Drug %s: target %s → %s", drug_name, target_upper, target_params)

            # Strategy 2: Drug class mapping
            if drug_class in DRUG_CLASS_PARAM_MAP:
                class_params = DRUG_CLASS_PARAM_MAP[drug_class]
                for mod, params in class_params.items():
                    configs.setdefault(mod, {}).update(params)
                log.debug("Drug %s: class=%s → %s", drug_name, drug_class, class_params)

            # Strategy 3: Infer checkpoint inhibitor from mechanism text
            elif "checkpoint" in mechanism or "pd-1" in mechanism or "pd-l1" in mechanism:
                for mod, params in DRUG_CLASS_PARAM_MAP.get("checkpoint_inhibitor", {}).items():
                    configs.setdefault(mod, {}).update(params)

        return configs

    def _map_genes_to_params(
        self, gene_ids: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Convert Gene entities to simulation parameters."""
        configs: Dict[str, Dict[str, Any]] = {}

        # Collect gene properties
        oncogene_count = 0
        suppressor_count = 0
        has_ar = False

        for gene_id in gene_ids:
            entity = self._store.get_entity(gene_id)
            if not isinstance(entity, Gene):
                continue

            symbol = (entity.symbol or entity.name or "").upper()
            gene_type = (entity.gene_type or "").lower()

            # Specific gene overrides
            if symbol in GENE_SPECIFIC_PARAM_MAP:
                for mod, params in GENE_SPECIFIC_PARAM_MAP[symbol].items():
                    configs.setdefault(mod, {}).update(params)

            if symbol == "AR":
                has_ar = True

            # Count gene types for aggregate effects
            if "oncogene" in gene_type:
                oncogene_count += 1
            elif "tumor_suppressor" in gene_type or "suppressor" in gene_type:
                suppressor_count += 1

        # Aggregate gene-type effects (scaled by count)
        if oncogene_count >= 3:
            configs.setdefault("molecular", {}).setdefault("transcription_rate", 1.5)
            configs.setdefault("cellular", {}).setdefault("division_time_cancer", 8.0)
        elif oncogene_count >= 1:
            configs.setdefault("molecular", {}).setdefault("transcription_rate", 1.2)

        if suppressor_count >= 2:
            configs.setdefault("molecular", {}).setdefault("mutation_rate", 0.005)

        # If AR pathway genes present, default to AR signaling ODE system
        if has_ar:
            configs.setdefault("ode", {}).setdefault("system", "ar_signaling")

        return configs

    def _map_initial_conditions(
        self,
        initial_cell_counts: Dict[str, int],
        initial_field_values: Dict[str, float],
    ) -> Dict[str, Dict[str, Any]]:
        """Map scenario initial conditions to module params."""
        configs: Dict[str, Dict[str, Any]] = {}

        if initial_cell_counts:
            cellular = {}
            immune = {}
            for cell_type, count in initial_cell_counts.items():
                ct_lower = cell_type.lower().replace(" ", "_")
                if ct_lower in ("normal", "normal_cells", "epithelial"):
                    cellular["n_normal_cells"] = count
                elif ct_lower in ("cancer", "cancer_cells", "tumor"):
                    cellular["n_cancer_cells"] = count
                elif ct_lower in ("t_cell", "t_cells", "cd8_t_cell"):
                    immune["n_t_cells"] = count
                elif ct_lower in ("nk_cell", "nk_cells", "nk"):
                    immune["n_nk_cells"] = count
                elif ct_lower in ("macrophage", "macrophages"):
                    immune["n_macrophages"] = count

            if cellular:
                configs["cellular"] = cellular
            if immune:
                configs["immune"] = immune

        if initial_field_values:
            vascular = {}
            for field_name, value in initial_field_values.items():
                fn_lower = field_name.lower()
                if fn_lower in ("oxygen", "o2"):
                    vascular["arterial_O2"] = value
                elif fn_lower in ("glucose",):
                    vascular["arterial_glucose"] = value

            if vascular:
                configs["vascular"] = vascular

        return configs

    # ── Merge utility ──────────────────────────────────────────────────

    @staticmethod
    def _merge_configs(
        *config_dicts: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Merge multiple module config dicts. Later values override earlier."""
        result: Dict[str, Dict[str, Any]] = {}

        for cfg in config_dicts:
            for module, params in cfg.items():
                if module not in result:
                    result[module] = {}
                result[module].update(params)

        return result

    # ── Validation ─────────────────────────────────────────────────────

    @staticmethod
    def validate_config(
        module_configs: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        """Validate parameter names against known module schemas.

        Returns list of warning strings (empty = all valid).
        """
        warnings = []
        for module, params in module_configs.items():
            if module not in MODULE_PARAM_SCHEMA:
                continue  # Unknown modules are OK (forward compat)
            known = MODULE_PARAM_SCHEMA[module]
            for param_name in params:
                if param_name not in known:
                    # Not necessarily an error — modules accept extra params
                    # via their config dict. Log for debugging.
                    warnings.append(
                        f"Module '{module}': param '{param_name}' not in known schema"
                    )
        return warnings

    # ── Convenience ────────────────────────────────────────────────────

    def infer_enabled_modules(
        self,
        scenario: SimulationScenario,
        module_configs: Dict[str, Dict[str, Any]],
    ) -> Dict[str, bool]:
        """Infer which modules should be enabled based on scenario content."""
        enabled = {
            "cellular": True,
            "immune": True,
            "vascular": True,
            "lymphatic": True,
            "molecular": True,
            "spatial": True,
            "epigenetic": True,
            "circadian": True,
            "morphogen": True,
        }

        # Enable ODE module if ODE params present
        if "ode" in module_configs and module_configs["ode"]:
            enabled["ode"] = True

        # Enable BNGL module if BNGL params present
        if "bngl" in module_configs and module_configs["bngl"]:
            enabled["bngl"] = True

        # Enable hybrid if hybrid params present
        if "hybrid" in module_configs and module_configs["hybrid"]:
            enabled["hybrid"] = True

        # Enable smoldyn if smoldyn params present
        if "smoldyn" in module_configs and module_configs["smoldyn"]:
            enabled["smoldyn"] = True

        return enabled
