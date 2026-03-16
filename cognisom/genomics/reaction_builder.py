"""
Reaction Network Builder
=========================

Converts entity interaction networks into ODE systems that can be
simulated by the GPU ODE solver. This is Layer 3 of the entity-driven
simulation integration.

Given a set of biological entities with their interacts_with relationships
and physics_params, this module:

1. Identifies species (molecules/proteins that change over time)
2. Builds reaction equations from interaction types
3. Extracts rate constants from entity physics_params
4. Generates a complete ODESystem compatible with cognisom's solver

Supported reaction types:
- Binding: A + B ⇌ AB (reversible, from kd_nm)
- Activation: A activates B (Michaelis-Menten, from km/kcat)
- Inhibition: A inhibits B (competitive, from ki)
- Phosphorylation: Enzyme phosphorylates substrate (enzymatic, from kcat/km)
- Catalysis: Enzyme catalyzes substrate conversion
- Production/degradation: constitutive synthesis and turnover

Usage:
    from cognisom.genomics.reaction_builder import ReactionNetworkBuilder
    from cognisom.library.store import EntityStore

    store = EntityStore()
    builder = ReactionNetworkBuilder(store)

    # Build AR signaling from entity interactions
    system = builder.build_pathway("AR Signaling")

    # Or build from specific entities
    system = builder.build_from_entities(["AR", "DHT", "PSA", "FOXA1"])
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from cognisom.library.store import EntityStore

log = logging.getLogger(__name__)


@dataclass
class Species:
    """A molecular species in the ODE system."""
    name: str
    initial_value: float = 0.0
    entity_name: str = ""  # Source entity


@dataclass
class Reaction:
    """A biochemical reaction with rate law."""
    name: str
    reaction_type: str  # "binding", "activation", "inhibition", etc.
    reactants: List[str] = field(default_factory=list)
    products: List[str] = field(default_factory=list)
    parameters: Dict[str, float] = field(default_factory=dict)
    source_entity: str = ""
    target_entity: str = ""


@dataclass
class EntityODESystem:
    """Complete ODE system built from entity interactions.

    Compatible with cognisom's ODESystem format for the GPU solver.
    """
    n_species: int = 0
    species_names: List[str] = field(default_factory=list)
    species_initial: List[float] = field(default_factory=list)
    reactions: List[Reaction] = field(default_factory=list)
    parameters: Dict[str, float] = field(default_factory=dict)
    _species_index: Dict[str, int] = field(default_factory=dict)

    def get_rhs_func(self) -> Callable:
        """Generate the right-hand-side function for the ODE solver."""
        idx = self._species_index
        reactions = self.reactions
        params = self.parameters
        n = self.n_species

        def rhs(t: float, y: np.ndarray, p: Dict) -> np.ndarray:
            # Use passed params or stored params
            pp = p if p else params
            dydt = np.zeros_like(y)

            for rxn in reactions:
                if rxn.reaction_type == "transcription":
                    # mRNA production: d[mRNA]/dt += k_tx
                    product = rxn.products[0]
                    k = pp.get(rxn.parameters.get("rate_key", ""), 1.0)
                    if product in idx:
                        dydt[..., idx[product]] += k

                elif rxn.reaction_type == "degradation":
                    # Degradation: d[X]/dt -= k_deg * [X]
                    species = rxn.reactants[0]
                    k = pp.get(rxn.parameters.get("rate_key", ""), 0.1)
                    if species in idx:
                        dydt[..., idx[species]] -= k * y[..., idx[species]]

                elif rxn.reaction_type == "translation":
                    # Protein from mRNA: d[Protein]/dt += k_tl * [mRNA]
                    mrna = rxn.reactants[0]
                    protein = rxn.products[0]
                    k = pp.get(rxn.parameters.get("rate_key", ""), 10.0)
                    if mrna in idx and protein in idx:
                        dydt[..., idx[protein]] += k * y[..., idx[mrna]]

                elif rxn.reaction_type == "binding":
                    # A + B ⇌ AB: d[A]/dt -= k_on*[A]*[B] - k_off*[AB]
                    a, b = rxn.reactants[0], rxn.reactants[1]
                    ab = rxn.products[0]
                    k_on = pp.get(rxn.parameters.get("k_on_key", ""), 100.0)
                    k_off = pp.get(rxn.parameters.get("k_off_key", ""), 10.0)
                    if a in idx and b in idx and ab in idx:
                        binding_flux = k_on * y[..., idx[a]] * y[..., idx[b]]
                        unbinding_flux = k_off * y[..., idx[ab]]
                        dydt[..., idx[a]] -= binding_flux - unbinding_flux
                        dydt[..., idx[b]] -= binding_flux - unbinding_flux
                        dydt[..., idx[ab]] += binding_flux - unbinding_flux

                elif rxn.reaction_type == "michaelis_menten":
                    # Enzyme activates product: d[P]/dt += Vmax*[E]*[S]/(Km+[S])
                    enzyme = rxn.reactants[0]
                    product = rxn.products[0]
                    vmax = pp.get(rxn.parameters.get("vmax_key", ""), 2.0)
                    km = pp.get(rxn.parameters.get("km_key", ""), 10.0)
                    if enzyme in idx and product in idx:
                        e_conc = y[..., idx[enzyme]]
                        dydt[..., idx[product]] += vmax * e_conc / (km + e_conc)

                elif rxn.reaction_type == "inhibition":
                    # Inhibitor reduces target: d[T]/dt -= k_inh * [I] * [T] / (Ki + [I])
                    inhibitor = rxn.reactants[0]
                    target = rxn.products[0]
                    k_inh = pp.get(rxn.parameters.get("k_inh_key", ""), 0.5)
                    ki = pp.get(rxn.parameters.get("ki_key", ""), 10.0)
                    if inhibitor in idx and target in idx:
                        i_conc = y[..., idx[inhibitor]]
                        dydt[..., idx[target]] -= k_inh * i_conc * y[..., idx[target]] / (ki + i_conc)

                elif rxn.reaction_type == "constant_input":
                    # Constant production: d[X]/dt += k_input
                    species = rxn.products[0]
                    k = pp.get(rxn.parameters.get("rate_key", ""), 1.0)
                    if species in idx:
                        dydt[..., idx[species]] += k

            return dydt

        return rhs

    def to_ode_system(self):
        """Convert to format compatible with cognisom's ODESystem."""
        from cognisom.gpu.ode_solver import ODESystem
        return ODESystem(
            n_species=self.n_species,
            species_names=self.species_names,
            rhs_func=self.get_rhs_func(),
            parameters=self.parameters,
            stiff=True,
        )


class ReactionNetworkBuilder:
    """Build ODE systems from entity library interactions.

    Example:
        builder = ReactionNetworkBuilder(EntityStore())

        # Build AR signaling pathway from entities
        system = builder.build_ar_signaling()

        # Or build custom system from named entities
        system = builder.build_from_entity_names(["AR", "DHT", "PSA"])
    """

    def __init__(self, store: EntityStore):
        self.store = store

    def build_ar_signaling(self) -> EntityODESystem:
        """Build AR signaling pathway ODE system from entity library.

        Replaces the hardcoded ar_signaling_pathway() in ode_solver.py
        with entity-driven parameters.

        Species: AR_mRNA, AR, DHT, AR_DHT, PSA_mRNA, PSA
        """
        system = EntityODESystem()

        # Get entity parameters
        ar = self.store.find_entity_by_name("AR", "gene")
        ar_pp = (ar.physics_params if ar and hasattr(ar, "physics_params") else {}) or {}

        # Species
        species_list = [
            Species("AR_mRNA", 0.0, "AR"),
            Species("AR", 0.0, "AR"),
            Species("DHT", 1.0, "DHT"),
            Species("AR_DHT", 0.0, "AR"),
            Species("PSA_mRNA", 0.0, "PSA"),
            Species("PSA", 0.0, "PSA"),
        ]

        system.n_species = len(species_list)
        system.species_names = [s.name for s in species_list]
        system.species_initial = [s.initial_value for s in species_list]
        system._species_index = {s.name: i for i, s in enumerate(species_list)}

        # Parameters — from entity physics_params with fallbacks
        system.parameters = {
            # AR transcription/translation/degradation
            "k_ar_tx": ar_pp.get("transcription_rate_mrna_per_hour", 5.0),
            "k_ar_m_deg": 0.693 / ar_pp.get("half_life_hours", 3.0) if ar_pp.get("half_life_hours") else 0.5,
            "k_ar_tl": 20.0,  # translation rate
            "k_ar_deg": 0.1,  # protein degradation

            # AR-DHT binding — from entity interacts_with
            "k_bind": _get_binding_rate(ar, "DHT", default_on=100.0),
            "k_unbind": _get_unbinding_rate(ar, "DHT", default_off=10.0),

            # DHT dynamics
            "DHT_input": 1.0,
            "k_dht_deg": 0.2,
            "k_complex_deg": 0.05,

            # PSA dynamics
            "k_psa_tx": 2.0,
            "K_psa": ar_pp.get("dna_binding_kd_nm", 10.0),  # PSA induction Km
            "k_psa_m_deg": 1.0,
            "k_psa_tl": 30.0,
            "k_psa_deg": 0.2,
        }

        # Reactions
        system.reactions = [
            Reaction("AR_transcription", "transcription", [], ["AR_mRNA"],
                     {"rate_key": "k_ar_tx"}, source_entity="AR"),
            Reaction("AR_mRNA_degradation", "degradation", ["AR_mRNA"], [],
                     {"rate_key": "k_ar_m_deg"}, source_entity="AR"),
            Reaction("AR_translation", "translation", ["AR_mRNA"], ["AR"],
                     {"rate_key": "k_ar_tl"}, source_entity="AR"),
            Reaction("AR_degradation", "degradation", ["AR"], [],
                     {"rate_key": "k_ar_deg"}, source_entity="AR"),
            Reaction("AR_DHT_binding", "binding", ["AR", "DHT"], ["AR_DHT"],
                     {"k_on_key": "k_bind", "k_off_key": "k_unbind"},
                     source_entity="AR", target_entity="DHT"),
            Reaction("DHT_input", "constant_input", [], ["DHT"],
                     {"rate_key": "DHT_input"}, source_entity="DHT"),
            Reaction("DHT_degradation", "degradation", ["DHT"], [],
                     {"rate_key": "k_dht_deg"}, source_entity="DHT"),
            Reaction("AR_DHT_degradation", "degradation", ["AR_DHT"], [],
                     {"rate_key": "k_complex_deg"}, source_entity="AR"),
            Reaction("PSA_transcription", "michaelis_menten", ["AR_DHT"], ["PSA_mRNA"],
                     {"vmax_key": "k_psa_tx", "km_key": "K_psa"},
                     source_entity="AR_DHT", target_entity="PSA"),
            Reaction("PSA_mRNA_degradation", "degradation", ["PSA_mRNA"], [],
                     {"rate_key": "k_psa_m_deg"}, source_entity="PSA"),
            Reaction("PSA_translation", "translation", ["PSA_mRNA"], ["PSA"],
                     {"rate_key": "k_psa_tl"}, source_entity="PSA"),
            Reaction("PSA_degradation", "degradation", ["PSA"], [],
                     {"rate_key": "k_psa_deg"}, source_entity="PSA"),
        ]

        log.info(
            "Built AR signaling ODE: %d species, %d reactions, %d parameters "
            "(entity-driven: AR kd=%.1f nM, half-life=%.1f hr)",
            system.n_species, len(system.reactions), len(system.parameters),
            ar_pp.get("dna_binding_kd_nm", 0), ar_pp.get("half_life_hours", 0),
        )
        return system

    def build_from_entity_names(self, entity_names: List[str]) -> EntityODESystem:
        """Build a custom ODE system from named entities and their interactions.

        Automatically discovers species, reactions, and parameters from
        entity physics_params and interacts_with fields.
        """
        system = EntityODESystem()
        entities = []
        for name in entity_names:
            e = self.store.find_entity_by_name(name, None)
            if e:
                entities.append(e)

        if not entities:
            log.warning("No entities found for: %s", entity_names)
            return system

        # Collect species (each entity = one species)
        species_map = {}
        for i, e in enumerate(entities):
            species_map[e.name] = i
            pp = e.physics_params if hasattr(e, "physics_params") else {}
            system.species_names.append(e.name)
            system.species_initial.append(pp.get("initial_concentration", 0.0))

        system.n_species = len(species_map)
        system._species_index = species_map

        # Build reactions from interacts_with
        rxn_count = 0
        for e in entities:
            pp = e.physics_params if hasattr(e, "physics_params") else {}
            interactions = e.interacts_with if hasattr(e, "interacts_with") else []

            # Add degradation for this species
            half_life = pp.get("half_life_hours", pp.get("half_life_min", 0))
            if half_life and half_life > 0:
                if "min" in str(pp.get("half_life_min", "")):
                    k_deg = 0.693 / (half_life / 60.0)
                else:
                    k_deg = 0.693 / half_life
                deg_key = f"k_deg_{e.name}"
                system.parameters[deg_key] = k_deg
                system.reactions.append(Reaction(
                    f"{e.name}_degradation", "degradation", [e.name], [],
                    {"rate_key": deg_key}, source_entity=e.name))
                rxn_count += 1

            # Add reactions from interactions
            for inter in (interactions or []):
                target = inter.get("target", "")
                itype = inter.get("type", "")
                kd = inter.get("kd_nm", inter.get("kd_um", 0))

                if target not in species_map:
                    continue  # Target not in our system

                if itype == "binds_to" and kd:
                    # Reversible binding
                    complex_name = f"{e.name}_{target}"
                    if complex_name not in species_map:
                        species_map[complex_name] = len(system.species_names)
                        system.species_names.append(complex_name)
                        system.species_initial.append(0.0)
                        system.n_species += 1

                    k_on_key = f"k_on_{e.name}_{target}"
                    k_off_key = f"k_off_{e.name}_{target}"
                    system.parameters[k_on_key] = 1.0 / max(0.01, kd)  # Approximate
                    system.parameters[k_off_key] = 1.0

                    system.reactions.append(Reaction(
                        f"{e.name}_{target}_binding", "binding",
                        [e.name, target], [complex_name],
                        {"k_on_key": k_on_key, "k_off_key": k_off_key},
                        source_entity=e.name, target_entity=target))
                    rxn_count += 1

                elif itype == "activates":
                    vmax_key = f"vmax_{e.name}_to_{target}"
                    km_key = f"km_{e.name}_to_{target}"
                    system.parameters[vmax_key] = pp.get("kinase_kcat_per_s", 2.0)
                    system.parameters[km_key] = kd if kd else 10.0

                    system.reactions.append(Reaction(
                        f"{e.name}_activates_{target}", "michaelis_menten",
                        [e.name], [target],
                        {"vmax_key": vmax_key, "km_key": km_key},
                        source_entity=e.name, target_entity=target))
                    rxn_count += 1

                elif itype == "inhibits":
                    k_inh_key = f"k_inh_{e.name}_on_{target}"
                    ki_key = f"ki_{e.name}_on_{target}"
                    system.parameters[k_inh_key] = 0.5
                    system.parameters[ki_key] = kd if kd else 10.0

                    system.reactions.append(Reaction(
                        f"{e.name}_inhibits_{target}", "inhibition",
                        [e.name], [target],
                        {"k_inh_key": k_inh_key, "ki_key": ki_key},
                        source_entity=e.name, target_entity=target))
                    rxn_count += 1

        system._species_index = species_map
        log.info(
            "Built custom ODE: %d species, %d reactions from %d entities",
            system.n_species, rxn_count, len(entities))
        return system


def _get_binding_rate(entity, target_name: str, default_on: float = 100.0) -> float:
    """Extract binding on-rate from entity interacts_with."""
    if not entity or not hasattr(entity, "interacts_with"):
        return default_on
    for inter in (entity.interacts_with or []):
        if inter.get("target", "") == target_name and inter.get("type") == "binds_to":
            kd = inter.get("kd_nm", 0)
            if kd and kd > 0:
                # Approximate k_on from Kd: k_on ≈ 1/Kd (simplified)
                return 1.0 / kd * 10.0  # Scale factor
    return default_on


def _get_unbinding_rate(entity, target_name: str, default_off: float = 10.0) -> float:
    """Extract unbinding rate from entity interacts_with."""
    if not entity or not hasattr(entity, "interacts_with"):
        return default_off
    for inter in (entity.interacts_with or []):
        if inter.get("target", "") == target_name:
            kd = inter.get("kd_nm", 0)
            if kd and kd > 0:
                return kd  # k_off ∝ Kd
    return default_off
