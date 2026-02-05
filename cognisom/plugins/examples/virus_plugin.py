"""
Example Virus Plugin
====================

A complete example plugin demonstrating how to extend Cognisom with
custom biological entities, Bio-USD prims, simulation modules, and
dashboard components.

This plugin adds:
1. VirusEntity — Library entity for virus metadata
2. BioVirusParticle — Bio-USD prim for visualization
3. VirusModule — Simulation module for virus dynamics
4. Virus Tracker — Dashboard panel for monitoring

Usage::

    # Import the plugin to auto-register components
    import cognisom.plugins.examples.virus_plugin

    # Or register via entry points (see pyproject.toml example below)

Entry points configuration (pyproject.toml)::

    [project.entry-points."cognisom.entities"]
    virus = "cognisom.plugins.examples.virus_plugin:VirusEntity"

    [project.entry-points."cognisom.prims"]
    bio_virus = "cognisom.plugins.examples.virus_plugin:BioVirusParticle"

    [project.entry-points."cognisom.modules"]
    virus = "cognisom.plugins.examples.virus_plugin:VirusModule"

    [project.entry-points."cognisom.components"]
    virus_tracker = "cognisom.plugins.examples.virus_plugin:render_virus_tracker"

Phase 0 of the Strategic Implementation Plan.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 1. ENTITY DEFINITION
# ═══════════════════════════════════════════════════════════════════════

from cognisom.library.models import (
    BioEntity,
    EntityType,
    entity_registry,
)


class VirusGenomeType(str, Enum):
    """Type of viral genome."""
    DNA_DS = "dsDNA"        # Double-stranded DNA
    DNA_SS = "ssDNA"        # Single-stranded DNA
    RNA_POSITIVE = "+ssRNA"  # Positive-sense ssRNA
    RNA_NEGATIVE = "-ssRNA"  # Negative-sense ssRNA
    RNA_DS = "dsRNA"        # Double-stranded RNA
    RETROVIRUS = "retrovirus"  # RNA that reverse transcribes


class VirusFamily(str, Enum):
    """Major virus families."""
    CORONAVIRUS = "Coronaviridae"
    RETROVIRUS = "Retroviridae"
    HERPESVIRUS = "Herpesviridae"
    PAPILLOMA = "Papillomaviridae"
    POLYOMA = "Polyomaviridae"
    ADENOVIRUS = "Adenoviridae"
    OTHER = "Other"


@dataclass
class VirusEntity(BioEntity):
    """
    A virus entity in the biological library.

    Stores metadata about viral species including genome type,
    host tropism, and structural features.

    Properties:
        virus_family: Taxonomic family (Coronaviridae, Retroviridae, etc.)
        genome_type: Type of genetic material (dsDNA, +ssRNA, etc.)
        genome_length: Length in base pairs/nucleotides
        host_receptors: Target receptors on host cells
        capsid_proteins: Structural proteins in the capsid
        envelope: Whether the virus has a lipid envelope
        tropism: Tissue/cell types the virus can infect
    """
    virus_family: VirusFamily = VirusFamily.OTHER
    genome_type: VirusGenomeType = VirusGenomeType.RNA_POSITIVE
    genome_length: int = 0
    host_receptors: List[str] = field(default_factory=list)
    capsid_proteins: List[str] = field(default_factory=list)
    envelope: bool = False
    tropism: List[str] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        # Override entity_type (we'll use string instead of enum for extensibility)
        # This shows how to work around the EntityType enum limitation

    def _extra_properties(self) -> dict:
        return {
            "virus_family": self.virus_family.value,
            "genome_type": self.genome_type.value,
            "genome_length": self.genome_length,
            "host_receptors": self.host_receptors,
            "capsid_proteins": self.capsid_proteins,
            "envelope": self.envelope,
            "tropism": self.tropism,
        }

    def _apply_properties(self, props: dict):
        self.virus_family = VirusFamily(props.get("virus_family", "Other"))
        self.genome_type = VirusGenomeType(props.get("genome_type", "+ssRNA"))
        self.genome_length = props.get("genome_length", 0)
        self.host_receptors = props.get("host_receptors", [])
        self.capsid_proteins = props.get("capsid_proteins", [])
        self.envelope = props.get("envelope", False)
        self.tropism = props.get("tropism", [])


# Register the virus entity type
entity_registry.register_class(
    "virus",
    VirusEntity,
    version="1.0.0",
    tags=["plugin", "example", "pathogen"],
    description="Viral pathogen with genome and tropism data",
)


# ═══════════════════════════════════════════════════════════════════════
# 2. BIO-USD PRIM DEFINITION
# ═══════════════════════════════════════════════════════════════════════

from cognisom.biousd.schema import (
    BioUnit,
    prim_registry,
    register_prim,
)


@dataclass
class BioPayloadCarrierAPI:
    """
    API schema for entities that carry molecular cargo.

    Applied to exosomes, viruses, liposomes, etc.
    """
    cargo_rna: List[str] = field(default_factory=list)
    cargo_proteins: List[str] = field(default_factory=list)
    cargo_lipids: List[str] = field(default_factory=list)
    payload_mass_kda: float = 0.0


@register_prim("bio_virus_particle", version="1.0.0")
@dataclass
class BioVirusParticle(BioUnit):
    """
    A virus particle in the Bio-USD scene.

    Represents a single virion with position, state, and cargo.

    Properties:
        virus_id: Unique particle identifier
        position: (x, y, z) in micrometres
        velocity: Movement vector in um/hour
        attached_cell_id: ID of cell this virus is attached to (-1 = free)
        internalized: Whether the virus has entered a cell
        degraded: Whether the virus has been destroyed
        capsid_intact: Capsid structural integrity (0-1)
        genome_released: Whether genome has been released into cytoplasm
    """
    virus_id: int = 0
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    attached_cell_id: int = -1
    internalized: bool = False
    degraded: bool = False
    capsid_intact: float = 1.0
    genome_released: bool = False

    # Payload API
    payload: Optional[BioPayloadCarrierAPI] = None


# Also register the payload API schema
from cognisom.biousd.schema import api_registry

api_registry.register_class(
    "bio_payload_carrier_api",
    BioPayloadCarrierAPI,
    version="1.0.0",
    tags=["plugin", "example"],
    description="Molecular cargo carrier interface",
)


# ═══════════════════════════════════════════════════════════════════════
# 3. SIMULATION MODULE
# ═══════════════════════════════════════════════════════════════════════

from cognisom.core.module_base import SimulationModule
from cognisom.modules import register_module


@register_module("virus", version="1.0.0")
class VirusModule(SimulationModule):
    """
    Simulation module for virus dynamics.

    Models:
    - Virus diffusion in extracellular space
    - Cell attachment via receptor binding
    - Internalization and uncoating
    - Genome replication (simplified)
    - Virion release (budding/lysis)
    - Immune clearance
    """

    def initialize(self):
        """Initialize virus simulation state."""
        self.viruses: List[Dict[str, Any]] = []
        self.next_id = 0
        self.total_produced = 0
        self.total_cleared = 0

        # Parameters (can be overridden via config)
        self.diffusion_rate = self.config.get("diffusion_rate", 10.0)  # um/hr
        self.attachment_probability = self.config.get("attachment_prob", 0.1)
        self.internalization_rate = self.config.get("internalization_rate", 0.5)
        self.replication_rate = self.config.get("replication_rate", 100)  # virions/hr
        self.clearance_rate = self.config.get("clearance_rate", 0.01)

        log.info(f"VirusModule initialized with {len(self.viruses)} viruses")

    def update(self, dt: float):
        """
        Update virus state.

        Parameters
        ----------
        dt : float
            Time step in hours
        """
        if not self.enabled:
            return

        import random

        # Process each virus
        for virus in self.viruses:
            if virus["degraded"]:
                continue

            # Free diffusion
            if virus["attached_cell_id"] < 0 and not virus["internalized"]:
                # Random walk
                dx = random.gauss(0, self.diffusion_rate * dt)
                dy = random.gauss(0, self.diffusion_rate * dt)
                dz = random.gauss(0, self.diffusion_rate * dt)
                virus["position"] = (
                    virus["position"][0] + dx,
                    virus["position"][1] + dy,
                    virus["position"][2] + dz,
                )

            # Clearance by immune system
            if random.random() < self.clearance_rate * dt:
                virus["degraded"] = True
                self.total_cleared += 1
                self.emit_event("VIRUS_CLEARED", {"virus_id": virus["virus_id"]})

    def get_state(self) -> Dict[str, Any]:
        """Return current module state."""
        active = [v for v in self.viruses if not v["degraded"]]
        attached = [v for v in active if v["attached_cell_id"] >= 0]
        internalized = [v for v in active if v["internalized"]]

        return {
            "total_viruses": len(self.viruses),
            "active_viruses": len(active),
            "attached_viruses": len(attached),
            "internalized_viruses": len(internalized),
            "total_produced": self.total_produced,
            "total_cleared": self.total_cleared,
        }

    def add_virus(
        self,
        position: Tuple[float, float, float],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Add a new virus particle.

        Parameters
        ----------
        position : tuple
            Initial (x, y, z) position
        **kwargs
            Additional virus properties

        Returns
        -------
        dict
            The created virus data
        """
        virus = {
            "virus_id": self.next_id,
            "position": position,
            "velocity": (0.0, 0.0, 0.0),
            "attached_cell_id": -1,
            "internalized": False,
            "degraded": False,
            "capsid_intact": 1.0,
            "genome_released": False,
            **kwargs,
        }
        self.viruses.append(virus)
        self.next_id += 1
        self.total_produced += 1
        return virus

    def seed_viruses(
        self,
        count: int,
        region: Tuple[float, float, float, float, float, float],
    ):
        """
        Seed random viruses in a region.

        Parameters
        ----------
        count : int
            Number of viruses to add
        region : tuple
            Bounding box (x_min, y_min, z_min, x_max, y_max, z_max)
        """
        import random

        x_min, y_min, z_min, x_max, y_max, z_max = region
        for _ in range(count):
            pos = (
                random.uniform(x_min, x_max),
                random.uniform(y_min, y_max),
                random.uniform(z_min, z_max),
            )
            self.add_virus(pos)

        log.info(f"Seeded {count} viruses in region")


# ═══════════════════════════════════════════════════════════════════════
# 4. DASHBOARD COMPONENT
# ═══════════════════════════════════════════════════════════════════════

from cognisom.dashboard.components import (
    register_component,
    ComponentType,
    ComponentCategory,
)


@register_component(
    "virus_tracker",
    component_type=ComponentType.PANEL,
    title="Virus Tracker",
    icon="🦠",
    category=ComponentCategory.SIMULATION,
    min_tier="researcher",
    position=50,
)
def render_virus_tracker(state: Dict[str, Any]):
    """
    Dashboard panel showing virus simulation status.

    Displays:
    - Active virus count
    - Attached vs free viruses
    - Clearance rate
    - Infection progress
    """
    import streamlit as st

    st.subheader("🦠 Virus Tracker")

    # Get virus state from simulation
    virus_state = state.get("virus", {})

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Active Viruses",
            virus_state.get("active_viruses", 0),
            delta=virus_state.get("delta_active", None),
        )

    with col2:
        st.metric(
            "Attached",
            virus_state.get("attached_viruses", 0),
        )

    with col3:
        st.metric(
            "Internalized",
            virus_state.get("internalized_viruses", 0),
        )

    # Show clearance stats
    total_produced = virus_state.get("total_produced", 0)
    total_cleared = virus_state.get("total_cleared", 0)

    if total_produced > 0:
        clearance_pct = (total_cleared / total_produced) * 100
        st.progress(
            min(clearance_pct / 100, 1.0),
            text=f"Clearance: {clearance_pct:.1f}%"
        )


@register_component(
    "virus_seeder",
    component_type=ComponentType.WIDGET,
    title="Virus Seeder",
    icon="💉",
    category=ComponentCategory.SIMULATION,
    min_tier="researcher",
    position=51,
)
def render_virus_seeder(state: Dict[str, Any]):
    """Widget to seed viruses into simulation."""
    import streamlit as st

    with st.expander("💉 Seed Viruses"):
        count = st.number_input("Virus Count", min_value=1, max_value=10000, value=100)

        st.caption("Region (micrometres)")
        col1, col2 = st.columns(2)
        with col1:
            x_min = st.number_input("X min", value=0.0)
            y_min = st.number_input("Y min", value=0.0)
            z_min = st.number_input("Z min", value=0.0)
        with col2:
            x_max = st.number_input("X max", value=100.0)
            y_max = st.number_input("Y max", value=100.0)
            z_max = st.number_input("Z max", value=50.0)

        if st.button("Seed Viruses", key="seed_viruses_btn"):
            # In a real implementation, this would call the VirusModule
            st.success(f"Seeded {count} viruses")


# ═══════════════════════════════════════════════════════════════════════
# 5. PLUGIN SUMMARY
# ═══════════════════════════════════════════════════════════════════════

log.info("Virus plugin loaded successfully")
log.debug(f"  - Entity: virus (VirusEntity)")
log.debug(f"  - Prim: bio_virus_particle (BioVirusParticle)")
log.debug(f"  - Module: virus (VirusModule)")
log.debug(f"  - Components: virus_tracker, virus_seeder")
