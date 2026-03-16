"""
Page 14 — Biological Entity Library
====================================

Research-grade catalog of biological entities: genes, proteins, drugs,
immune cells, cytokines, pathways, and their interactions. Each entity
includes PhD-level descriptions, physics parameters, interaction networks,
and Bio-USD visualization mapping.
"""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
from cognisom.dashboard.page_config import safe_set_page_config

safe_set_page_config(page_title="Entity Library | Cognisom", page_icon="\U0001f4da", layout="wide")

from cognisom.auth.middleware import streamlit_page_gate
user = streamlit_page_gate("14_entity_library")

import json
import time
import pandas as pd
from cognisom.library.store import EntityStore
from cognisom.library.models import (
    BioEntity, EntityType, EntityStatus, RelationshipType, Relationship,
    ENTITY_CLASS_MAP,
)


@st.cache_resource
def get_store():
    return EntityStore()

store = get_store()

# ── Page Header ──────────────────────────────────────────────────────

st.title("Biological Entity Library")

# Stats row at top — always visible
stats = store.stats()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Entities", stats["total_entities"])
m2.metric("Relationships", stats["total_relationships"])
m3.metric("Entity Types", len(stats["by_type"]))
m4.metric("Tags", len(store.get_all_tags()))

st.divider()

# ── Search Bar (always visible) ──────────────────────────────────────

search_col, type_col, status_col = st.columns([3, 1, 1])
with search_col:
    search_query = st.text_input(
        "Search entities",
        placeholder="Search genes, proteins, drugs, pathways, immune cells...",
        key="entity_search",
        label_visibility="collapsed",
    )
with type_col:
    type_options = ["All Types"] + sorted(stats.get("by_type", {}).keys())
    selected_type = st.selectbox("Type", type_options, label_visibility="collapsed")
with status_col:
    selected_status = st.selectbox(
        "Status", ["active", "all", "deprecated", "review"],
        label_visibility="collapsed",
    )

# ── Run Search ───────────────────────────────────────────────────────

etype_filter = selected_type if selected_type != "All Types" else None
status_filter = selected_status if selected_status != "all" else ""
entities, total = store.search(
    query=search_query,
    entity_type=etype_filter,
    status=status_filter,
    limit=100,
)

st.caption(f"Showing {len(entities)} of {total} entities")

# ── Type Distribution Bar ────────────────────────────────────────────

if not search_query and stats["by_type"]:
    sorted_types = sorted(stats["by_type"].items(), key=lambda x: -x[1])
    type_cols = st.columns(min(len(sorted_types), 9))
    for i, (etype, count) in enumerate(sorted_types[:9]):
        with type_cols[i]:
            st.caption(f"**{etype}**\n{count}")

st.divider()

# ── Entity List with Inline Detail ───────────────────────────────────

# Color mapping for entity types
TYPE_COLORS = {
    "gene": "#818cf8", "protein": "#4ade80", "drug": "#fb923c",
    "metabolite": "#c084fc", "cell_type": "#f472b6", "pathway": "#38bdf8",
    "mutation": "#f87171", "receptor": "#fbbf24", "tissue_type": "#2dd4bf",
    "organ": "#f472b6", "immune_cell": "#34d399", "cytokine": "#fbbf24",
    "virus": "#e879f9", "bacterium": "#a3e635", "antibody": "#60a5fa",
    "complement": "#f59e0b", "prr": "#a78bfa", "mhc": "#fb7185",
    "adhesion_molecule": "#86efac",
}

if entities:
    for entity in entities:
        etype = entity.entity_type.value
        color = TYPE_COLORS.get(etype, "#9ca3af")

        # Entity card
        with st.expander(
            f"**{entity.display_name}** — _{etype}_",
            expanded=False,
        ):
            # Top row: type badge + description
            st.markdown(
                f'<span style="background: {color}22; color: {color}; '
                f'padding: 2px 10px; border-radius: 4px; font-size: 0.75rem; '
                f'font-weight: 600; text-transform: uppercase;">{etype}</span>'
                f'&nbsp;&nbsp;<span style="opacity: 0.5; font-size: 0.8rem;">'
                f'ID: {entity.entity_id}</span>',
                unsafe_allow_html=True,
            )

            # Description (the research-grade content)
            if entity.description:
                st.markdown(entity.description)
            else:
                st.caption("No description available.")

            # Details in columns
            col_left, col_right = st.columns(2)

            with col_left:
                # Physics parameters
                if hasattr(entity, "physics_params") and entity.physics_params:
                    st.markdown("**Physics / Simulation Parameters**")
                    for k, v in entity.physics_params.items():
                        label = k.replace("_", " ").title()
                        st.markdown(f"- {label}: `{v}`")

                # Compartments
                if hasattr(entity, "compartments") and entity.compartments:
                    st.markdown("**Subcellular Localization**")
                    st.markdown(", ".join(
                        f"`{c}`" for c in entity.compartments
                    ))

                # Interactions
                if hasattr(entity, "interacts_with") and entity.interacts_with:
                    st.markdown("**Molecular Interactions**")
                    for inter in entity.interacts_with[:8]:
                        target = inter.get("target", "?")
                        itype = inter.get("type", "?")
                        note = inter.get("note", "")
                        kd = inter.get("kd_nm", "")
                        detail = f" (Kd: {kd} nM)" if kd else (f" — {note}" if note else "")
                        st.markdown(f"- **{itype}** → {target}{detail}")

            with col_right:
                # External IDs
                if entity.external_ids:
                    st.markdown("**External Identifiers**")
                    for k, v in entity.external_ids.items():
                        st.markdown(f"- {k}: `{v}`")

                # Bio-USD mapping
                if hasattr(entity, "usd_prim_type") and entity.usd_prim_type:
                    st.markdown("**Bio-USD Visualization**")
                    st.markdown(f"- Prim Type: `{entity.usd_prim_type}`")
                    if hasattr(entity, "mesh_type") and entity.mesh_type:
                        st.markdown(f"- Mesh: `{entity.mesh_type}`")
                    if hasattr(entity, "scale_um") and entity.scale_um:
                        st.markdown(f"- Scale: `{entity.scale_um}` um")
                    if hasattr(entity, "color_rgb") and entity.color_rgb:
                        r, g, b = [int(c * 255) for c in entity.color_rgb[:3]]
                        st.markdown(
                            f'- Color: <span style="background: rgb({r},{g},{b}); '
                            f'padding: 2px 12px; border-radius: 3px;">&nbsp;</span> '
                            f'`[{entity.color_rgb[0]:.1f}, {entity.color_rgb[1]:.1f}, {entity.color_rgb[2]:.1f}]`',
                            unsafe_allow_html=True,
                        )

                # Tags
                if entity.tags:
                    st.markdown("**Tags**")
                    st.markdown(" ".join(
                        f'`{t}`' for t in entity.tags
                    ))

                # Relationships from store
                rels = store.get_relationships(entity.entity_id)
                if rels:
                    st.markdown(f"**Relationships ({len(rels)})**")
                    for rel in rels[:8]:
                        direction = "→" if rel.source_id == entity.entity_id else "←"
                        other_id = rel.target_id if rel.source_id == entity.entity_id else rel.source_id
                        other = store.get_entity(other_id)
                        other_name = other.display_name if other else other_id[:8]
                        st.markdown(
                            f"- {direction} **{rel.rel_type.value}** {direction} "
                            f"_{other_name}_ "
                            f'<span style="opacity:0.5">({rel.confidence:.0%})</span>',
                            unsafe_allow_html=True,
                        )

                # Type-specific properties
                data = entity.to_dict()
                props = data.get("properties", {})
                if props:
                    shown = 0
                    for k, v in props.items():
                        if k in ("symbol", "full_name", "gene_type", "chromosome",
                                 "map_location", "cytokine_family", "receptor",
                                 "signaling_pathway", "pro_inflammatory",
                                 "molecular_weight_kda", "gene_symbol",
                                 "virus_family", "genome_type", "genome_size_kb",
                                 "immune_type", "subtype", "drug_class",
                                 "mechanism_of_action"):
                            if v:
                                if shown == 0:
                                    st.markdown("**Properties**")
                                label = k.replace("_", " ").title()
                                st.markdown(f"- {label}: `{v}`")
                                shown += 1

            # Source link
            if entity.source_url:
                st.markdown(f"[View on source database ↗]({entity.source_url})")

else:
    if search_query:
        st.info(f"No entities found matching '{search_query}'")
    elif stats["total_entities"] == 0:
        st.warning("Entity library is empty. It should auto-seed on restart.")

st.divider()

# ── Manage Section (collapsed) ───────────────────────────────────────

with st.expander("Manage Entities", expanded=False):
    from cognisom.auth.models import UserRole
    can_curate = user.role in (UserRole.ADMIN, UserRole.ORG_ADMIN)

    if not can_curate:
        st.info("Admin or Org Admin role required to manage entities.")
    else:
        manage_tab1, manage_tab2, manage_tab3 = st.tabs([
            "Add Entity", "Add Relationship", "Import / Export"
        ])

        with manage_tab1:
            with st.form("add_entity_form"):
                etype = st.selectbox("Entity Type", [t.value for t in EntityType])
                name = st.text_input("Name *")
                display = st.text_input("Display Name")
                description = st.text_area("Description")
                tags = st.text_input("Tags (comma-separated)")
                submitted = st.form_submit_button("Create Entity", type="primary")

            if submitted and name:
                klass = ENTITY_CLASS_MAP.get(etype, BioEntity)
                entity = klass(
                    name=name,
                    display_name=display or name,
                    description=description,
                    entity_type=EntityType(etype),
                    tags=[t.strip() for t in tags.split(",") if t.strip()],
                    source="user",
                    created_by=user.username,
                )
                if store.add_entity(entity):
                    st.success(f"Created: **{name}** ({entity.entity_id})")
                    st.cache_resource.clear()
                else:
                    st.error("Failed to create entity")

        with manage_tab2:
            with st.form("add_rel_form"):
                src_id = st.text_input("Source entity ID")
                tgt_id = st.text_input("Target entity ID")
                rel_type = st.selectbox("Relationship", [rt.value for rt in RelationshipType])
                confidence = st.slider("Confidence", 0.0, 1.0, 0.9)
                evidence = st.text_input("Evidence")
                submitted = st.form_submit_button("Create Relationship", type="primary")

            if submitted and src_id and tgt_id:
                rel = Relationship(
                    source_id=src_id, target_id=tgt_id,
                    rel_type=RelationshipType(rel_type),
                    confidence=confidence, evidence=evidence,
                )
                if store.add_relationship(rel):
                    st.success("Relationship created")
                else:
                    st.error("Failed")

        with manage_tab3:
            col_exp, col_imp = st.columns(2)
            with col_exp:
                st.markdown("**Export**")
                if st.button("Export as JSON"):
                    all_entities, _ = store.search(limit=10000, status="")
                    export = [e.to_dict() for e in all_entities]
                    st.download_button(
                        "Download", data=json.dumps(export, indent=2),
                        file_name="cognisom_entity_library.json",
                        mime="application/json",
                    )
            with col_imp:
                st.markdown("**Import**")
                uploaded = st.file_uploader("Upload JSON", type=["json"])
                if uploaded:
                    try:
                        data = json.loads(uploaded.read())
                        if isinstance(data, list):
                            imported = sum(1 for item in data if store.add_entity(BioEntity.from_dict(item)))
                            st.success(f"Imported {imported} entities")
                            st.cache_resource.clear()
                    except Exception as e:
                        st.error(f"Import failed: {e}")
