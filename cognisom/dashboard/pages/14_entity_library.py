"""
Page 14 — Biological Entity Library
====================================

Browse, search, and manage the biological entity catalog.
Features: full-text search, faceted filtering, entity detail view,
relationship graph, curation interface, import/export.
"""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st

st.set_page_config(page_title="Entity Library | Cognisom", page_icon="📚", layout="wide")

from cognisom.auth.middleware import streamlit_page_gate
user = streamlit_page_gate("14_entity_library")

import json
import time
from cognisom.library.store import EntityStore
from cognisom.library.models import (
    BioEntity, EntityType, EntityStatus, RelationshipType, Relationship,
    ENTITY_CLASS_MAP,
)

# ── Initialize store ─────────────────────────────────────────────────

@st.cache_resource
def get_store():
    return EntityStore()

store = get_store()

# ── Custom CSS ───────────────────────────────────────────────────────

st.markdown("""
<style>
.entity-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    transition: all 0.2s ease;
}
.entity-card:hover {
    border-color: rgba(0,200,200,0.3);
    background: rgba(255,255,255,0.06);
}
.entity-type-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.type-gene { background: rgba(99,102,241,0.2); color: #818cf8; }
.type-protein { background: rgba(34,197,94,0.2); color: #4ade80; }
.type-drug { background: rgba(249,115,22,0.2); color: #fb923c; }
.type-metabolite { background: rgba(168,85,247,0.2); color: #c084fc; }
.type-cell_type { background: rgba(236,72,153,0.2); color: #f472b6; }
.type-pathway { background: rgba(14,165,233,0.2); color: #38bdf8; }
.type-mutation { background: rgba(239,68,68,0.2); color: #f87171; }
.type-receptor { background: rgba(251,191,36,0.2); color: #fbbf24; }
.type-tissue_type { background: rgba(20,184,166,0.2); color: #2dd4bf; }
.type-organ { background: rgba(244,114,182,0.2); color: #f472b6; }
.type-ligand { background: rgba(132,204,22,0.2); color: #a3e635; }
.stat-number {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
}
.rel-badge {
    display: inline-block;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 0.65rem;
    background: rgba(255,255,255,0.08);
    margin: 1px 2px;
}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────

st.title("Biological Entity Library")
st.caption("Browse, search, and manage the biological entity catalog — genes, proteins, drugs, pathways, and more.")

# ── Tabs ─────────────────────────────────────────────────────────────

tab_browse, tab_detail, tab_graph, tab_manage, tab_import = st.tabs([
    "Browse & Search", "Entity Detail", "Relationship Graph", "Manage", "Import / Seed",
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1: Browse & Search
# ══════════════════════════════════════════════════════════════════════

with tab_browse:
    # Stats row
    stats = store.stats()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Entities", stats["total_entities"])
    with col2:
        st.metric("Relationships", stats["total_relationships"])
    with col3:
        st.metric("Entity Types", len(stats["by_type"]))
    with col4:
        all_tags = store.get_all_tags()
        st.metric("Tags", len(all_tags))

    st.divider()

    # Search bar
    search_col, filter_col1, filter_col2 = st.columns([3, 1, 1])
    with search_col:
        search_query = st.text_input(
            "Search entities",
            placeholder="Search by name, description, synonyms...",
            key="entity_search",
        )
    with filter_col1:
        type_options = ["All"] + [t.value for t in EntityType]
        selected_type = st.selectbox("Entity Type", type_options)
    with filter_col2:
        status_options = ["active", "deprecated", "review", ""]
        selected_status = st.selectbox("Status", status_options, index=0)

    # Run search
    etype_filter = selected_type if selected_type != "All" else None
    entities, total = store.search(
        query=search_query,
        entity_type=etype_filter,
        status=selected_status,
        limit=50,
    )

    st.caption(f"Showing {len(entities)} of {total} entities")

    # Results grid
    if entities:
        for entity in entities:
            etype = entity.entity_type.value
            type_class = f"type-{etype}"

            with st.container():
                cols = st.columns([0.15, 0.6, 0.25])
                with cols[0]:
                    st.markdown(
                        f'<span class="entity-type-badge {type_class}">{etype}</span>',
                        unsafe_allow_html=True,
                    )
                with cols[1]:
                    st.markdown(f"**{entity.display_name}**")
                    if entity.description:
                        st.caption(entity.description[:150] + ("..." if len(entity.description) > 150 else ""))
                with cols[2]:
                    # Show external IDs if available
                    ext_ids = entity.external_ids
                    if ext_ids:
                        id_parts = [f"`{k}:{v}`" for k, v in list(ext_ids.items())[:2]]
                        st.caption(" | ".join(id_parts))
                    # Button to view detail
                    if st.button("View", key=f"view_{entity.entity_id}", type="secondary"):
                        st.session_state["detail_entity_id"] = entity.entity_id
                        st.rerun()
                st.divider()
    else:
        if search_query:
            st.info(f"No entities found matching '{search_query}'")
        elif stats["total_entities"] == 0:
            st.info("The entity library is empty. Go to the **Import / Seed** tab to populate it.")

    # Type distribution
    if stats["by_type"]:
        st.subheader("Entity Distribution")
        type_counts = stats["by_type"]
        cols = st.columns(min(len(type_counts), 6))
        for i, (etype, count) in enumerate(sorted(type_counts.items(), key=lambda x: -x[1])):
            with cols[i % len(cols)]:
                st.markdown(
                    f'<div style="text-align:center">'
                    f'<div class="stat-number">{count}</div>'
                    f'<span class="entity-type-badge type-{etype}">{etype}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

# ══════════════════════════════════════════════════════════════════════
# TAB 2: Entity Detail
# ══════════════════════════════════════════════════════════════════════

with tab_detail:
    detail_id = st.session_state.get("detail_entity_id", "")

    if not detail_id:
        # Let user enter an entity ID or select from search
        detail_id = st.text_input("Enter entity ID to view details", key="manual_detail_id")

    if detail_id:
        entity = store.get_entity(detail_id)
        if entity is None:
            st.error(f"Entity '{detail_id}' not found")
        else:
            etype = entity.entity_type.value
            type_class = f"type-{etype}"

            # Header
            header_col1, header_col2 = st.columns([3, 1])
            with header_col1:
                st.markdown(
                    f'<span class="entity-type-badge {type_class}" style="font-size:0.85rem; padding:4px 12px;">'
                    f'{etype}</span>',
                    unsafe_allow_html=True,
                )
                st.title(entity.display_name)
            with header_col2:
                st.caption(f"ID: `{entity.entity_id}`")
                st.caption(f"Status: **{entity.status.value}**")
                st.caption(f"Source: {entity.source}")

            # Description
            if entity.description:
                st.markdown(entity.description)

            st.divider()

            # Properties columns
            prop_col1, prop_col2 = st.columns(2)

            with prop_col1:
                st.subheader("Identifiers")
                if entity.external_ids:
                    for k, v in entity.external_ids.items():
                        st.markdown(f"- **{k}**: `{v}`")
                else:
                    st.caption("No external identifiers")

                if entity.ontology_ids:
                    st.subheader("Ontology IDs")
                    for oid in entity.ontology_ids[:10]:
                        st.markdown(f"- `{oid}`")

                if entity.synonyms:
                    st.subheader("Synonyms")
                    st.write(", ".join(entity.synonyms))

            with prop_col2:
                st.subheader("Type-Specific Properties")
                data = entity.to_dict()
                props = data.get("properties", {})
                if props:
                    for k, v in props.items():
                        if isinstance(v, list):
                            if v:
                                st.markdown(f"**{k}**: {', '.join(str(x) for x in v[:10])}")
                        elif v:
                            st.markdown(f"**{k}**: {v}")
                else:
                    st.caption("No additional properties")

                if entity.tags:
                    st.subheader("Tags")
                    tag_html = " ".join(
                        f'<span class="rel-badge">{t}</span>' for t in entity.tags
                    )
                    st.markdown(tag_html, unsafe_allow_html=True)

            # Relationships
            st.divider()
            st.subheader("Relationships")
            rels = store.get_relationships(detail_id)
            if rels:
                for rel in rels:
                    direction = "outgoing" if rel.source_id == detail_id else "incoming"
                    other_id = rel.target_id if direction == "outgoing" else rel.source_id
                    other = store.get_entity(other_id)
                    other_name = other.display_name if other else other_id

                    arrow = "→" if direction == "outgoing" else "←"
                    st.markdown(
                        f"- {arrow} **{rel.rel_type.value}** {arrow} "
                        f"*{other_name}* (confidence: {rel.confidence:.0%})"
                    )
            else:
                st.caption("No relationships found for this entity")

            # Source URL
            if entity.source_url:
                st.divider()
                st.markdown(f"[View on source database]({entity.source_url})")

            # Audit log
            with st.expander("Change History"):
                audit = store.get_audit_log(detail_id, limit=10)
                if audit:
                    for entry in audit:
                        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(entry["changed_at"]))
                        st.caption(f"{ts} — **{entry['action']}** by {entry['changed_by']}")
                else:
                    st.caption("No audit records")
    else:
        st.info("Select an entity from the Browse tab or enter an entity ID above.")

# ══════════════════════════════════════════════════════════════════════
# TAB 3: Relationship Graph
# ══════════════════════════════════════════════════════════════════════

with tab_graph:
    st.subheader("Entity Relationship Graph")
    st.caption("Interactive visualization of entity relationships")

    graph_col1, graph_col2 = st.columns([1, 3])

    with graph_col1:
        graph_type = st.selectbox(
            "Relationship type",
            ["All"] + [rt.value for rt in RelationshipType],
            key="graph_rel_type",
        )
        graph_limit = st.slider("Max relationships", 10, 200, 50, key="graph_limit")

    with graph_col2:
        # Get relationships
        if graph_type == "All":
            # Get a sample of all relationships
            all_rels = []
            for rt in RelationshipType:
                rels = store.get_relationships_by_type(rt.value, limit=graph_limit // len(RelationshipType) + 1)
                all_rels.extend(rels)
            all_rels = all_rels[:graph_limit]
        else:
            all_rels = store.get_relationships_by_type(graph_type, limit=graph_limit)

        if all_rels:
            # Build adjacency data for display
            nodes = set()
            edges = []
            for rel in all_rels:
                src = store.get_entity(rel.source_id)
                tgt = store.get_entity(rel.target_id)
                src_name = src.display_name if src else rel.source_id[:8]
                tgt_name = tgt.display_name if tgt else rel.target_id[:8]
                nodes.add(src_name)
                nodes.add(tgt_name)
                edges.append((src_name, rel.rel_type.value, tgt_name))

            # Display as a table (Streamlit doesn't have native graph viz)
            st.markdown(f"**{len(nodes)} nodes, {len(edges)} edges**")

            # Format as edge list table
            import pandas as pd
            df = pd.DataFrame(edges, columns=["Source", "Relationship", "Target"])
            st.dataframe(df, use_container_width=True, height=400)

            # Node list
            with st.expander(f"All nodes ({len(nodes)})"):
                st.write(sorted(nodes))
        else:
            st.info("No relationships found. Seed the library first.")

# ══════════════════════════════════════════════════════════════════════
# TAB 4: Manage (Curation)
# ══════════════════════════════════════════════════════════════════════

with tab_manage:
    st.subheader("Entity Curation")

    from cognisom.auth.models import UserRole
    can_curate = user.role in (UserRole.ADMIN, UserRole.ORG_ADMIN)

    if not can_curate:
        st.warning("You need Admin or Org Admin role to manage entities.")
    else:
        manage_action = st.radio(
            "Action",
            ["Add Entity", "Edit Entity", "Add Relationship", "Delete Entity"],
            horizontal=True,
        )

        if manage_action == "Add Entity":
            st.markdown("### Add a new entity")
            with st.form("add_entity_form"):
                etype = st.selectbox("Entity Type", [t.value for t in EntityType])
                name = st.text_input("Name *")
                display = st.text_input("Display Name (optional)")
                description = st.text_area("Description")
                synonyms = st.text_input("Synonyms (comma-separated)")
                tags = st.text_input("Tags (comma-separated)")
                source = st.text_input("Source", value="manual")
                submitted = st.form_submit_button("Create Entity", type="primary")

            if submitted and name:
                klass = ENTITY_CLASS_MAP.get(etype, BioEntity)
                entity = klass(
                    name=name,
                    display_name=display or name,
                    description=description,
                    entity_type=EntityType(etype),
                    synonyms=[s.strip() for s in synonyms.split(",") if s.strip()],
                    tags=[t.strip() for t in tags.split(",") if t.strip()],
                    source=source,
                    created_by=user.username,
                )
                if store.add_entity(entity):
                    st.success(f"Created entity: **{name}** ({entity.entity_id})")
                else:
                    st.error("Failed to create entity (may already exist)")

        elif manage_action == "Edit Entity":
            st.markdown("### Edit an existing entity")
            edit_id = st.text_input("Entity ID to edit")
            if edit_id:
                entity = store.get_entity(edit_id)
                if entity:
                    with st.form("edit_entity_form"):
                        new_name = st.text_input("Name", value=entity.name)
                        new_display = st.text_input("Display Name", value=entity.display_name)
                        new_desc = st.text_area("Description", value=entity.description)
                        new_status = st.selectbox(
                            "Status",
                            [s.value for s in EntityStatus],
                            index=[s.value for s in EntityStatus].index(entity.status.value),
                        )
                        new_tags = st.text_input("Tags", value=", ".join(entity.tags))
                        submitted = st.form_submit_button("Update", type="primary")

                    if submitted:
                        entity.name = new_name
                        entity.display_name = new_display
                        entity.description = new_desc
                        entity.status = EntityStatus(new_status)
                        entity.tags = [t.strip() for t in new_tags.split(",") if t.strip()]
                        if store.update_entity(entity, changed_by=user.username):
                            st.success("Entity updated")
                        else:
                            st.error("Failed to update entity")
                else:
                    st.error(f"Entity '{edit_id}' not found")

        elif manage_action == "Add Relationship":
            st.markdown("### Add a relationship between entities")
            with st.form("add_rel_form"):
                src_id = st.text_input("Source entity ID")
                tgt_id = st.text_input("Target entity ID")
                rel_type = st.selectbox("Relationship type", [rt.value for rt in RelationshipType])
                confidence = st.slider("Confidence", 0.0, 1.0, 1.0)
                evidence = st.text_input("Evidence / source")
                submitted = st.form_submit_button("Create Relationship", type="primary")

            if submitted and src_id and tgt_id:
                rel = Relationship(
                    source_id=src_id,
                    target_id=tgt_id,
                    rel_type=RelationshipType(rel_type),
                    confidence=confidence,
                    evidence=evidence,
                )
                if store.add_relationship(rel):
                    st.success("Relationship created")
                else:
                    st.error("Failed to create relationship")

        elif manage_action == "Delete Entity":
            st.markdown("### Deprecate / delete an entity")
            del_id = st.text_input("Entity ID to deprecate")
            if del_id:
                entity = store.get_entity(del_id)
                if entity:
                    st.markdown(f"**{entity.display_name}** ({entity.entity_type.value})")
                    if st.button("Deprecate Entity", type="primary"):
                        if store.delete_entity(del_id, changed_by=user.username):
                            st.success("Entity deprecated")
                        else:
                            st.error("Failed to deprecate")
                else:
                    st.warning("Entity not found")

# ══════════════════════════════════════════════════════════════════════
# TAB 5: Import / Seed
# ══════════════════════════════════════════════════════════════════════

with tab_import:
    st.subheader("Import & Seed Data")

    import_col1, import_col2 = st.columns(2)

    with import_col1:
        st.markdown("### Seed Prostate Cancer Catalog")
        st.markdown(
            "Populate the library with a curated set of ~100 biological entities "
            "relevant to prostate cancer research: genes, proteins, drugs, pathways, "
            "cell types, metabolites, mutations, and their relationships."
        )
        fetch_remote = st.checkbox(
            "Fetch from NCBI/UniProt (slower but richer)",
            value=False,
        )
        if st.button("Seed Catalog", type="primary"):
            with st.spinner("Seeding entity library..."):
                from cognisom.library.seed_data import seed_prostate_cancer_catalog
                counts = seed_prostate_cancer_catalog(store, fetch_remote=fetch_remote)
                st.success("Seed complete!")
                for k, v in counts.items():
                    if v > 0:
                        st.write(f"- {k}: **{v}**")
                st.cache_resource.clear()
                st.rerun()

    with import_col2:
        st.markdown("### Load Individual Gene")
        st.markdown("Fetch a gene and its protein from NCBI Gene / UniProt.")
        gene_name = st.text_input("Gene symbol", placeholder="e.g. TP53, BRCA1, EGFR")
        gene_type = st.selectbox(
            "Gene type",
            ["", "oncogene", "tumor_suppressor", "housekeeping", "signaling"],
            key="import_gene_type",
        )
        if st.button("Load Gene + Protein") and gene_name:
            with st.spinner(f"Loading {gene_name}..."):
                from cognisom.library.loaders import EntityLoader
                loader = EntityLoader(store)
                gene, protein = loader.load_gene_protein_pair(gene_name, gene_type)
                if gene:
                    st.success(f"Gene loaded: **{gene.name}** ({gene.entity_id})")
                if protein:
                    st.success(f"Protein loaded: **{protein.name}** ({protein.entity_id})")
                if not gene and not protein:
                    st.error(f"Could not load '{gene_name}' from NCBI/UniProt")
                st.cache_resource.clear()

        st.divider()

        st.markdown("### Export Library")
        if st.button("Export as JSON"):
            all_entities, _ = store.search(limit=10000, status="")
            export = [e.to_dict() for e in all_entities]
            st.download_button(
                "Download JSON",
                data=json.dumps(export, indent=2),
                file_name="cognisom_entity_library.json",
                mime="application/json",
            )

        st.markdown("### Import from JSON")
        uploaded = st.file_uploader("Upload entity JSON file", type=["json"])
        if uploaded:
            try:
                data = json.loads(uploaded.read())
                if isinstance(data, list):
                    imported = 0
                    for item in data:
                        entity = BioEntity.from_dict(item)
                        if store.add_entity(entity):
                            imported += 1
                    st.success(f"Imported {imported} entities")
                    st.cache_resource.clear()
                else:
                    st.error("Expected a JSON array of entities")
            except Exception as e:
                st.error(f"Import failed: {e}")

# Footer
from cognisom.dashboard.footer import render_footer
render_footer()
