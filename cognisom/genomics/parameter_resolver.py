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
from typing import Any, Dict, List, Optional, Tuple

from cognisom.library.store import EntityStore
from cognisom.library.parameters import (
    PARAMETER_SPECS, SIMULATION_GRADE, ParameterValue, Provenance, get_parameter,
)

log = logging.getLogger(__name__)

# (entity name, field) pairs already reported, so a resolver used inside a
# loop warns once per missing number rather than once per call.
_MISSING_REPORTED: set = set()


# (entity, entity type, field) refusals already reported by sourced_value.
_REFUSED_REPORTED: set = set()


def reset_missing_log() -> None:
    """Forget which absences have been reported. For tests."""
    _MISSING_REPORTED.clear()
    _REFUSED_REPORTED.clear()


_SHARED: Optional["SimulationParameterResolver"] = None


def shared_resolver() -> Optional["SimulationParameterResolver"]:
    """One resolver per process, or None if the library cannot be opened.

    Simulation modules are constructed many times over a run and a test
    suite; opening the store is cheap after the first time but not free, and
    a module must never fail to construct because the library is missing --
    it falls back to its defaults and says so.
    """
    global _SHARED
    if _SHARED is None:
        try:
            _SHARED = SimulationParameterResolver(EntityStore())
        except Exception as exc:                        # pragma: no cover
            log.warning("entity library unavailable (%s); simulation "
                        "modules will use their built-in defaults", exc)
            return None
    return _SHARED


GENE_EFFECT_FIELDS = (
    "mhc1_downregulation_effect", "tumor_growth_multiplier",
    "pd_l1_effect", "pi3k_pathway_activation",
)

CELL_PARAM_FIELDS = (
    "division_time_hours", "apoptosis_rate_per_hour",
    "migration_speed_um_per_min", "oxygen_consumption_rate",
    "glucose_consumption_rate", "detection_radius_um",
    "kill_radius_um", "kill_probability", "exhaustion_rate",
)

#: Fields only a cytotoxic-effector row carries. Used to break the
#: cell_type / immune_cell name collision toward the row that can answer.
CYTOTOXIC_FIELDS = ("kill_probability", "detection_radius_um", "exhaustion_rate")

METABOLITE_PARAM_FIELDS = (
    "diffusion_coefficient_m2_per_s", "normal_concentration_mM",
    "critical_low_mM",
)


def _resolve(entity, field: str) -> Optional[float]:
    """The entity's value for `field`, or None if it has none.

    Goes through `library.parameters.get_parameter` rather than reading
    `physics_params` directly, because that is the only reader which can
    tell a curated value from a type-level default stamped in by
    enrichment. It never substitutes a literal: a plausible number
    standing in for an absent one is the defect this function removes.

    A value genuinely stored as 0.0 is returned as 0.0, so a measured
    zero and an absent field stay distinguishable in both directions.
    """
    pv = get_parameter(entity, field)
    if pv is None:
        key = (getattr(entity, "name", "?"), field)
        if key not in _MISSING_REPORTED:
            _MISSING_REPORTED.add(key)
            log.warning(
                "%s has no value for %r; omitting it rather than "
                "substituting a default", key[0], field,
            )
        return None
    return pv.value


def _provenance_of(entity, fields) -> str:
    """Worst provenance among the fields actually present on `entity`.

    Today this returns ':placeholder' for every entity in the library --
    0 of 294 entities carry a provenance sidecar and 665 of 665
    parameters resolve as PLACEHOLDER. That label is the point:
    `_source: entity_library` claimed a curation that exists nowhere yet.
    """
    levels = [
        pv.provenance for pv in (get_parameter(entity, f) for f in fields)
        if pv is not None
    ]
    if not levels:
        return "absent"
    if Provenance.PLACEHOLDER in levels:
        return "entity_library:placeholder"
    return "entity_library:curated"


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

    def get_gene_immune_effect(self, gene_name: str) -> Dict[str, Any]:
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
            # The old miss-return was the worst case in this file:
            # `tumor_growth_multiplier: 1.0` for an unknown gene is not a
            # neutral placeholder, it is a positive assertion that the
            # mutation does not affect growth.
            #
            # DEBUG, not WARNING: an exome-wide VCF reaches here for
            # thousands of genes and the library holds 20. The aggregate in
            # `get_gene_effects_for_mutations` warns once with a count,
            # which is the readable form of the same fact.
            log.debug(
                "No gene entity named %r in the library; reporting no "
                "effects rather than neutral ones", gene_name,
            )
            self._gene_cache[gene_name] = {"_source": "absent"}
            return self._gene_cache[gene_name]

        # Keys are omitted rather than set to None, so existing
        # `.get(key, default)` callers keep working and a caller that wants
        # to know whether a value exists can ask with `in`.
        effect = {
            f: v for f, v in (
                (f, _resolve(gene, f)) for f in GENE_EFFECT_FIELDS
            ) if v is not None
        }
        effect["_source"] = _provenance_of(gene, GENE_EFFECT_FIELDS)

        self._gene_cache[gene_name] = effect
        return effect

    def get_gene_effects_for_mutations(
        self, mutated_genes: List[str]
    ) -> Dict[str, Any]:
        """Aggregate immune effects across all mutated genes.

        Returns combined MHC-I downregulation, tumor growth multiplier,
        and other effects.
        """
        # An empty gene list is a DIFFERENT absence: it means annotation
        # ran and found nothing to report, so zero effect is the correct
        # answer. Only genes that exist but carry no value are "unknown".
        if not mutated_genes:
            return {
                "mhc1_downregulation": 0.0,
                "tumor_growth_multiplier": 1.0,
                "_source": "no_mutations",
                "_genes": [],
                "_contributing_genes": [],
                "_unresolved_genes": [],
            }

        total_mhc1_down = 0.0
        growth_multiplier = 1.0
        contributing: List[str] = []
        unresolved: List[str] = []

        for gene_name in mutated_genes:
            effect = self.get_gene_immune_effect(gene_name)
            mhc1 = effect.get("mhc1_downregulation_effect")
            growth = effect.get("tumor_growth_multiplier")
            if mhc1 is None and growth is None:
                unresolved.append(gene_name)
                continue
            contributing.append(gene_name)
            total_mhc1_down += mhc1 or 0.0
            growth_multiplier *= 1.0 if growth is None else growth

        if unresolved:
            # One line per call, with a count, instead of one per gene.
            log.warning(
                "%d of %d mutated genes carry no immune annotation in the "
                "library (first few: %s); their contribution is reported as "
                "absent, not as zero",
                len(unresolved), len(mutated_genes), unresolved[:5],
            )

        return {
            # None, not 0.0, when no mutated gene carried a value: a tumor
            # whose genes are all uncurated is not a tumor with intact
            # MHC-I. The library holds 20 prostate genes, so a patient from
            # another tissue routinely lands here.
            "mhc1_downregulation": (
                min(1.0, total_mhc1_down) if contributing else None
            ),
            "tumor_growth_multiplier": growth_multiplier if contributing else None,
            # The aggregate inherits the provenance of the rows that
            # contributed, so a consumer can see that the number rests on
            # uncurated stamps rather than on curation.
            "_source": (
                _provenance_of(
                    self.store.find_entity_by_name(contributing[0], "gene"),
                    GENE_EFFECT_FIELDS,
                ) if contributing else "absent"
            ),
            "_genes": list(mutated_genes),
            # What makes the aggregate auditable: without these, a 37-gene
            # panel with one hit and a 1-gene panel with one hit produce
            # indistinguishable output.
            "_contributing_genes": contributing,
            "_unresolved_genes": unresolved,
        }

    # ── Cell Type Parameters ────────────────────────────────────────

    def get_cell_type_params(self, cell_name: str) -> Dict[str, Any]:
        """Get simulation parameters for a cell type.

        Returns division time, migration speed, oxygen consumption, etc.
        from the cell type entity in the library.
        """
        # A name can exist as BOTH a cell_type and an immune_cell row
        # ("NK cell" is the only such collision in the shipped library).
        # Querying cell_type first reported an NK cell with no
        # kill_probability, and the hardcoded 0.0 below then presented it as
        # a cell that cannot kill anything. Prefer the row that actually
        # carries cytotoxic parameters, and record both which row answered
        # and that the name was ambiguous.
        as_immune = self.store.find_entity_by_name(cell_name, "immune_cell")
        as_cell = self.store.find_entity_by_name(cell_name, "cell_type")
        cytotoxic = as_immune is not None and any(
            get_parameter(as_immune, f) is not None for f in CYTOTOXIC_FIELDS
        )
        cell = as_immune if cytotoxic else (as_cell or as_immune)
        if cell is None:
            log.warning(
                "No cell_type or immune_cell entity named %r in the "
                "library; returning no parameters", cell_name,
            )
            return {}

        params = {
            f: v for f, v in (
                (f, _resolve(cell, f)) for f in CELL_PARAM_FIELDS
            ) if v is not None
        }
        params["_entity_type"] = cell.entity_type.value
        if as_immune is not None and as_cell is not None:
            other = as_cell if cell is as_immune else as_immune
            params["_shadowed_by"] = other.entity_type.value
            key = (cell_name, "__ambiguous_entity_type__")
            if key not in _MISSING_REPORTED:
                _MISSING_REPORTED.add(key)
                log.warning(
                    "%r exists as both an immune_cell and a cell_type "
                    "entity; answering from the %s row. Resolve the "
                    "duplicate in the library.",
                    cell_name, cell.entity_type.value,
                )
        params["_source"] = _provenance_of(cell, CELL_PARAM_FIELDS)
        return params

    # ── Metabolite Parameters ───────────────────────────────────────

    def get_metabolite_params(self, metabolite_name: str) -> Dict[str, Any]:
        """Get diffusion and concentration parameters for a metabolite."""
        met = self.store.find_entity_by_name(metabolite_name, "metabolite")
        if not met:
            log.warning(
                "No metabolite entity named %r in the library; returning "
                "no parameters", metabolite_name,
            )
            return {}

        # NOTE what this does NOT fix: all 12 metabolites carry the same
        # stamped 1e-9 m^2/s and 5.0 mM, so they still return those --
        # now correctly labelled placeholder. The label is the deliverable
        # here; the values still need curation.
        params = {
            f: v for f, v in (
                (f, _resolve(met, f)) for f in METABOLITE_PARAM_FIELDS
            ) if v is not None
        }
        params["_source"] = _provenance_of(met, METABOLITE_PARAM_FIELDS)
        return params

    # ── Single values for running code ─────────────────────────────

    def sourced_value(self, entity_name: str, entity_type: str,
                      field: str) -> Tuple[Optional[ParameterValue], str]:
        """One parameter, only if it is evidence a simulation may consume.

        This is the accessor for wiring a library value into running code,
        and it is deliberately stricter than :func:`_resolve`, which returns
        any stored number. Returns ``(value, note)``: the full
        ParameterValue -- unit, grade and citation intact, so the caller
        can report where its number came from -- with a note of the form
        ``"measured:PMID:20689600"``; or ``None`` with the reason it was
        refused. A value is refused when:

        - the parameter is a model construct. A construct is calibrated
          against an observable, never sourced, and must not reach running
          code through this path however it is graded;
        - the entity is not in the library under that exact name and type;
        - the parameter is absent;
        - the value is a PLACEHOLDER: a type-level stamp from enrichment is
          not a measurement of this cell, and reading it would let, say,
          the cell_type ``Macrophage`` row's 0.5 um/min slow every simulated
          macrophage tenfold on a number of unknown origin;
        - its unit differs from the spec's. A unit is converted where it is
          stored, never assumed where it is read, so a mismatch is a defect
          in the library row and is logged as one -- but it does not stop a
          simulation that has a default to fall back on.

        Each refusal is logged once per process.
        """
        spec = PARAMETER_SPECS.get(field)

        def refuse(reason: str, level: int = logging.INFO):
            key = (entity_name, entity_type, field)
            if key not in _REFUSED_REPORTED:
                _REFUSED_REPORTED.add(key)
                log.log(level, "%s %s not used from the library: %s",
                        entity_name, field, reason)
            return None, reason

        if spec is not None and spec.model_construct:
            return refuse("model construct: calibrated, not sourced")
        entity = self.store.find_entity_by_name(entity_name, entity_type)
        if entity is None:
            return refuse(f"no {entity_type} named {entity_name!r}")
        pv = get_parameter(entity, field)
        if pv is None:
            return refuse("absent")
        if pv.provenance not in SIMULATION_GRADE:
            return refuse(f"{pv.provenance.value}: not evidence")
        if spec is not None and pv.unit != spec.unit:
            return refuse(f"stored in {pv.unit!r}, the spec is {spec.unit!r}",
                          logging.WARNING)
        return pv, f"{pv.provenance.value}:{pv.citation or pv.source}"

    # ── Cache Management ────────────────────────────────────────────

    def clear_cache(self):
        """Clear all cached lookups (call after entity modifications)."""
        self._profile_cache = None
        self._gene_cache.clear()
        _MISSING_REPORTED.clear()


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
