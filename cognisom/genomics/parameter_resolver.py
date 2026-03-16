"""
Simulation Parameter Resolver
===============================

Bridges the Entity Library and the Simulation Engine by resolving
simulation parameters from entity definitions at runtime.

Instead of hardcoded treatment profiles and gene effects, the resolver
queries the entity library for physics_params, interacts_with, and
other simulation-relevant fields.

This means:
- Adding a drug entity → automatically available in treatment simulator
- Editing entity physics_params → changes simulation behavior
- No code changes needed to add or modify treatments

Usage:
    from cognisom.library.store import EntityStore
    from cognisom.genomics.parameter_resolver import SimulationParameterResolver

    store = EntityStore()
    resolver = SimulationParameterResolver(store)

    # Get all available treatment profiles
    profiles = resolver.get_all_treatment_profiles()

    # Get gene-specific immune effects
    pten_effect = resolver.get_gene_immune_effect("PTEN")
"""

import logging
from typing import Any, Dict, List, Optional

from cognisom.library.store import EntityStore

log = logging.getLogger(__name__)


class SimulationParameterResolver:
    """Resolves simulation parameters from the entity library.

    Replaces hardcoded dicts in treatment_simulator.py, twin_config.py,
    and tissue_config.py with entity-driven lookups.
    """

    def __init__(self, store: Optional[EntityStore] = None):
        if store is None:
            store = EntityStore()
        self.store = store
        self._profile_cache: Optional[Dict] = None
        self._gene_cache: Dict[str, Dict] = {}

    # ── Treatment Profiles ──────────────────────────────────────────

    def get_treatment_profile(self, drug_name: str) -> Optional[Dict]:
        """Build a treatment profile from a drug entity.

        Returns dict compatible with TREATMENT_PROFILES format in
        treatment_simulator.py, or None if drug not found.
        """
        drug = self.store.find_entity_by_name(drug_name, "drug")
        if not drug:
            return None

        props = drug.to_dict().get("properties", {})
        pp = drug.physics_params if hasattr(drug, "physics_params") else {}

        # Determine drug class and target from entity data
        drug_class = props.get("drug_class", "unknown")
        targets = [i.get("target", "") for i in (drug.interacts_with or [])]
        mechanism = drug.description[:200] if drug.description else ""

        # Map drug_class to treatment class labels
        class_mapping = {
            "anti-androgen": "AR antagonist",
            "anti_androgen": "AR antagonist",
            "checkpoint_inhibitor": _infer_checkpoint_class(targets),
            "PARP_inhibitor": "PARP inhibitor",
            "parp_inhibitor": "PARP inhibitor",
            "chemotherapy": "chemotherapy",
            "PI3K_pathway": "PI3K/AKT inhibitor",
            "radiotherapy": "radiotherapy",
            "JAK_inhibitor": "JAK inhibitor",
            "anti_TNF": "anti-TNF",
            "anti_cd20": "anti-CD20",
            "cytokine_therapy": "cytokine therapy",
            "neoantigen_vaccine": "neoantigen vaccine",
        }
        treatment_class = class_mapping.get(drug_class, drug_class)

        # Build profile from entity physics_params
        profile = {
            "name": drug.display_name or drug.name,
            "class": treatment_class,
            "target": ", ".join(targets[:3]) if targets else "unknown",
            "mechanism": mechanism,
            # Simulation-specific parameters (from physics_params)
            "exhaustion_reversal": pp.get("exhaustion_reversal", 0.0),
            "treg_effect": pp.get("treg_effect", 0.0),
            "irae_base_risk": pp.get("irae_base_risk", 0.1),
            "effect_onset_days": pp.get("effect_onset_days", 14),
            "effect_ramp_days": pp.get("effect_ramp_days", 21),
            # Biomarker requirements
            "requires_tmb_high": pp.get("requires_tmb_high", False),
            "requires_msi_h": pp.get("requires_msi_h", False),
            "requires_dna_repair_defect": pp.get("requires_dna_repair_defect", False),
            "requires_ar_sensitivity": pp.get("requires_ar_sensitivity", False),
            "requires_neoantigens": pp.get("requires_neoantigens", False),
            # Best immune context
            "best_for": pp.get("best_for", ["any"]),
            # Source tracking
            "_source": "entity_library",
            "_entity_id": drug.entity_id if hasattr(drug, "entity_id") else "",
        }

        return profile

    def get_all_treatment_profiles(self) -> Dict[str, Dict]:
        """Load all drug treatment profiles from entity library.

        Returns dict keyed by normalized drug name, compatible with
        TREATMENT_PROFILES format in treatment_simulator.py.
        """
        if self._profile_cache is not None:
            return self._profile_cache

        drugs, _ = self.store.search(entity_type="drug", limit=100)
        profiles = {}

        for drug in drugs:
            pp = drug.physics_params if hasattr(drug, "physics_params") else {}
            # Only include drugs that have simulation parameters
            if pp.get("exhaustion_reversal") is not None or \
               pp.get("irae_base_risk") is not None or \
               pp.get("effect_onset_days") is not None:
                profile = self.get_treatment_profile(drug.name)
                if profile:
                    key = _normalize_drug_key(drug.name)
                    profiles[key] = profile

        self._profile_cache = profiles
        log.info("Loaded %d treatment profiles from entity library", len(profiles))
        return profiles

    # ── Gene Immune Effects ─────────────────────────────────────────

    def get_gene_immune_effect(self, gene_name: str) -> Dict[str, float]:
        """Get immune-related effects of a gene mutation.

        Returns dict with:
        - mhc1_downregulation_effect: how much MHC-I is reduced (0-1)
        - tumor_growth_multiplier: growth rate multiplier (1.0 = no change)
        - pd_l1_effect: PD-L1 expression change
        """
        if gene_name in self._gene_cache:
            return self._gene_cache[gene_name]

        gene = self.store.find_entity_by_name(gene_name, "gene")
        if not gene:
            return {"mhc1_downregulation_effect": 0.0, "tumor_growth_multiplier": 1.0}

        pp = gene.physics_params if hasattr(gene, "physics_params") else {}

        effect = {
            "mhc1_downregulation_effect": pp.get("mhc1_downregulation_effect", 0.0),
            "tumor_growth_multiplier": pp.get("tumor_growth_multiplier", 1.0),
            "pd_l1_effect": pp.get("pd_l1_effect", 0.0),
            "pi3k_pathway_activation": pp.get("pi3k_pathway_activation", 0.0),
            "_source": "entity_library",
        }

        self._gene_cache[gene_name] = effect
        return effect

    def get_gene_effects_for_mutations(
        self, mutated_genes: List[str]
    ) -> Dict[str, float]:
        """Aggregate immune effects across all mutated genes.

        Returns combined MHC-I downregulation, tumor growth multiplier,
        and other effects.
        """
        total_mhc1_down = 0.0
        growth_multiplier = 1.0

        for gene_name in mutated_genes:
            effect = self.get_gene_immune_effect(gene_name)
            total_mhc1_down += effect.get("mhc1_downregulation_effect", 0.0)
            growth_multiplier *= effect.get("tumor_growth_multiplier", 1.0)

        return {
            "mhc1_downregulation": min(1.0, total_mhc1_down),
            "tumor_growth_multiplier": growth_multiplier,
            "_source": "entity_library",
            "_genes": mutated_genes,
        }

    # ── Cell Type Parameters ────────────────────────────────────────

    def get_cell_type_params(self, cell_name: str) -> Dict[str, Any]:
        """Get simulation parameters for a cell type.

        Returns division time, migration speed, oxygen consumption, etc.
        from the cell type entity in the library.
        """
        # Search both cell_type and immune_cell
        cell = self.store.find_entity_by_name(cell_name, "cell_type")
        if not cell:
            cell = self.store.find_entity_by_name(cell_name, "immune_cell")
        if not cell:
            return {}

        pp = cell.physics_params if hasattr(cell, "physics_params") else {}
        return {
            "division_time_hours": pp.get("division_time_hours", 24.0),
            "apoptosis_rate_per_hour": pp.get("apoptosis_rate_per_hour", 0.001),
            "migration_speed_um_per_min": pp.get("migration_speed_um_per_min", 0.5),
            "oxygen_consumption_rate": pp.get("oxygen_consumption_rate", 0.05),
            "glucose_consumption_rate": pp.get("glucose_consumption_rate", 0.02),
            "detection_radius_um": pp.get("detection_radius_um", 0.0),
            "kill_radius_um": pp.get("kill_radius_um", 0.0),
            "kill_probability": pp.get("kill_probability", 0.0),
            "exhaustion_rate": pp.get("exhaustion_rate", 0.0),
            "_source": "entity_library",
        }

    # ── Metabolite Parameters ───────────────────────────────────────

    def get_metabolite_params(self, metabolite_name: str) -> Dict[str, float]:
        """Get diffusion and concentration parameters for a metabolite."""
        met = self.store.find_entity_by_name(metabolite_name, "metabolite")
        if not met:
            return {}

        pp = met.physics_params if hasattr(met, "physics_params") else {}
        return {
            "diffusion_coefficient_m2_per_s": pp.get("diffusion_coefficient_m2_per_s", 1e-9),
            "normal_concentration_mM": pp.get("normal_concentration_mM", 5.0),
            "critical_low_mM": pp.get("critical_low_mM", 0.5),
            "_source": "entity_library",
        }

    # ── Cache Management ────────────────────────────────────────────

    def clear_cache(self):
        """Clear all cached lookups (call after entity modifications)."""
        self._profile_cache = None
        self._gene_cache.clear()


def _normalize_drug_key(name: str) -> str:
    """Normalize drug name to a dict key."""
    return (name.lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_"))


def _infer_checkpoint_class(targets: List[str]) -> str:
    """Infer checkpoint class from target list."""
    targets_lower = [t.lower() for t in targets]
    if any("pd" in t or "pdcd1" in t or "cd274" in t for t in targets_lower):
        return "anti-PD-1"
    if any("ctla" in t for t in targets_lower):
        return "anti-CTLA-4"
    if any("lag" in t for t in targets_lower):
        return "anti-LAG-3"
    return "checkpoint inhibitor"
