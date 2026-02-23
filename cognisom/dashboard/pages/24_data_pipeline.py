"""
Page 24 — Data Pipeline
========================

Bulk import, multi-source enrichment, progress monitoring,
and data quality metrics for the entity database.
"""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st

st.set_page_config(page_title="Data Pipeline | Cognisom", page_icon="🔬", layout="wide")

from cognisom.auth.middleware import streamlit_page_gate
user = streamlit_page_gate("24_data_pipeline")

import time
from cognisom.library.store import EntityStore
from cognisom.library.bulk_import import BulkImporter
from cognisom.library.gene_sets import IMPORT_SETS


@st.cache_resource
def get_store() -> EntityStore:
    return EntityStore()


store = get_store()
importer = BulkImporter(store)

# ── Header ───────────────────────────────────────────────────────────

st.title("Data Pipeline")
st.markdown("Bulk import genes, drugs, and pathways from public databases with multi-source enrichment.")

# Quick stats
stats = store.stats()
c1, c2, c3 = st.columns(3)
c1.metric("Total Entities", f"{stats['total_entities']:,}")
c2.metric("Total Relationships", f"{stats['total_relationships']:,}")
c3.metric("Entity Types", len(stats.get("by_type", {})))

st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────

tab_seed, tab_custom, tab_enrich, tab_status = st.tabs([
    "Quick Seed", "Custom Import", "Enrichment", "Status & Quality"
])

# ═══════════════════════════════════════════════════════════════════════
# Tab 1: Quick Seed
# ═══════════════════════════════════════════════════════════════════════

with tab_seed:
    st.subheader("One-Click Import")
    st.markdown("Select a curated gene/drug set to populate the entity database.")

    for name, preset in IMPORT_SETS.items():
        genes = preset.get("genes", [])
        drugs = preset.get("drugs", [])
        pathways = preset.get("pathways", [])

        with st.expander(f"{name}", expanded=False):
            st.markdown(preset.get("description", ""))
            st.caption(f"{len(genes)} genes | {len(drugs)} drugs | {len(pathways)} pathways")

            if genes:
                st.markdown("**Genes:** " + ", ".join(genes[:20]) + ("..." if len(genes) > 20 else ""))
            if drugs:
                st.markdown("**Drugs:** " + ", ".join(drugs[:10]) + ("..." if len(drugs) > 10 else ""))

            btn_key = f"import_{name.replace(' ', '_').replace('(', '').replace(')', '')}"
            if st.button(f"Import {name}", key=btn_key, type="primary"):
                progress = st.progress(0)
                status = st.empty()
                log_area = st.empty()
                logs = []

                def seed_progress(msg):
                    logs.append(msg)
                    status.markdown(f"**{msg}**")
                    # Estimate progress from message
                    if "[" in msg and "/" in msg:
                        try:
                            parts = msg.split("[")[1].split("]")[0].split("/")
                            current = int(parts[0])
                            total_items = int(parts[1].split("]")[0].split(")")[0])
                            progress.progress(min(current / max(total_items, 1), 1.0))
                        except (IndexError, ValueError):
                            pass
                    log_area.text("\n".join(logs[-10:]))

                t0 = time.time()
                report = importer.import_preset(name, progress_callback=seed_progress)
                elapsed = time.time() - t0

                progress.progress(1.0)
                status.empty()

                st.success(
                    f"Import complete in {elapsed:.1f}s: "
                    f"{report.entities_created} created, "
                    f"{report.entities_updated} updated, "
                    f"{report.relationships_created} relationships"
                )

                if report.errors:
                    with st.expander(f"{len(report.errors)} errors", expanded=False):
                        for err in report.errors[:50]:
                            st.text(err)

                st.rerun()

# ═══════════════════════════════════════════════════════════════════════
# Tab 2: Custom Import
# ═══════════════════════════════════════════════════════════════════════

with tab_custom:
    st.subheader("Custom Import")

    import_type = st.radio(
        "Import type",
        ["Gene List", "Drug List", "KEGG Pathway", "STRING Network"],
        horizontal=True,
    )

    if import_type == "Gene List":
        st.markdown("Paste gene symbols (one per line or comma-separated).")
        gene_text = st.text_area(
            "Gene symbols",
            placeholder="TP53\nBRCA1\nPTEN\nAR\nMYC",
            height=200,
            key="custom_genes",
        )
        enrich_toggle = st.toggle("Full enrichment (slower, richer data)", value=True, key="gene_enrich")
        custom_tags = st.text_input("Tags (comma-separated)", value="", key="gene_tags")

        if st.button("Import Genes", type="primary", key="import_genes_btn"):
            # Parse gene list
            symbols = []
            for line in gene_text.strip().split("\n"):
                for sym in line.split(","):
                    sym = sym.strip().upper()
                    if sym and len(sym) < 20:
                        symbols.append(sym)

            if not symbols:
                st.error("No valid gene symbols found.")
            else:
                tags = [t.strip() for t in custom_tags.split(",") if t.strip()] if custom_tags else None
                progress = st.progress(0)
                status = st.empty()

                def gene_progress(msg):
                    status.markdown(f"**{msg}**")
                    if "[" in msg and "/" in msg:
                        try:
                            parts = msg.split("[")[1].split("/")
                            current = int(parts[0])
                            total_items = int(parts[1].split("]")[0])
                            progress.progress(min(current / max(total_items, 1), 1.0))
                        except (IndexError, ValueError):
                            pass

                report = importer.import_gene_list(
                    symbols, enrich=enrich_toggle, tags=tags,
                    progress_callback=gene_progress,
                )

                progress.progress(1.0)
                st.success(
                    f"Done: {report.entities_created} created, "
                    f"{report.entities_updated} updated, "
                    f"{report.relationships_created} relationships, "
                    f"{report.duration_seconds:.1f}s"
                )
                if report.errors:
                    with st.expander(f"{len(report.errors)} errors"):
                        for err in report.errors[:50]:
                            st.text(err)

    elif import_type == "Drug List":
        st.markdown("Paste drug names (one per line).")
        drug_text = st.text_area(
            "Drug names",
            placeholder="Enzalutamide\nOlaparib\nPembrolizumab",
            height=200,
            key="custom_drugs",
        )

        if st.button("Import Drugs", type="primary", key="import_drugs_btn"):
            names = [n.strip() for n in drug_text.strip().split("\n") if n.strip()]
            if not names:
                st.error("No valid drug names found.")
            else:
                progress = st.progress(0)
                status = st.empty()

                def drug_progress(msg):
                    status.markdown(f"**{msg}**")

                report = importer.import_drug_list(names, progress_callback=drug_progress)

                progress.progress(1.0)
                st.success(
                    f"Done: {report.entities_created} created, "
                    f"{report.duration_seconds:.1f}s"
                )

    elif import_type == "KEGG Pathway":
        st.markdown("Enter a KEGG pathway ID to import the pathway and all its gene members.")
        pathway_id = st.text_input(
            "KEGG Pathway ID",
            placeholder="hsa05215",
            key="kegg_pw_id",
        )
        import_pw_genes = st.toggle("Import all pathway genes", value=True, key="pw_genes")

        if st.button("Import Pathway", type="primary", key="import_pw_btn"):
            if not pathway_id.strip():
                st.error("Enter a KEGG pathway ID.")
            else:
                progress = st.progress(0)
                status = st.empty()

                def pw_progress(msg):
                    status.markdown(f"**{msg}**")

                report = importer.import_kegg_pathway(
                    pathway_id.strip(),
                    import_genes=import_pw_genes,
                    progress_callback=pw_progress,
                )

                progress.progress(1.0)
                st.success(
                    f"Done: {report.entities_created} created, "
                    f"{report.relationships_created} relationships"
                )

    elif import_type == "STRING Network":
        st.markdown("Enter seed genes to import their STRING interaction network.")
        string_genes = st.text_input(
            "Seed genes (comma-separated)",
            placeholder="AR, TP53, PTEN, BRCA1",
            key="string_genes",
        )
        score_thresh = st.slider("Minimum STRING score", 400, 950, 700, step=50, key="string_score")
        import_partners = st.toggle("Import interaction partners", value=True, key="string_partners")

        if st.button("Import Network", type="primary", key="import_string_btn"):
            seeds = [s.strip().upper() for s in string_genes.split(",") if s.strip()]
            if not seeds:
                st.error("Enter at least one seed gene.")
            else:
                progress = st.progress(0)
                status = st.empty()

                def string_progress(msg):
                    status.markdown(f"**{msg}**")

                report = importer.import_string_network(
                    seeds,
                    score_threshold=score_thresh,
                    import_partners=import_partners,
                    progress_callback=string_progress,
                )

                progress.progress(1.0)
                st.success(
                    f"Done: {report.entities_created} created, "
                    f"{report.relationships_created} relationships"
                )

# ═══════════════════════════════════════════════════════════════════════
# Tab 3: Enrichment
# ═══════════════════════════════════════════════════════════════════════

with tab_enrich:
    st.subheader("Enrichment")
    st.markdown("Re-enrich existing entities with latest data from external APIs.")

    e1, e2, e3 = st.columns(3)

    with e1:
        gene_count = stats.get("by_type", {}).get("gene", 0)
        st.metric("Genes in DB", gene_count)
        if st.button("Enrich All Genes", type="primary", disabled=gene_count == 0, key="enrich_genes"):
            progress = st.progress(0)
            status = st.empty()

            def enrich_gene_progress(msg):
                status.markdown(f"**{msg}**")

            report = importer.enricher.enrich_all_genes(enrich_gene_progress)
            progress.progress(1.0)
            st.success(f"Enriched: {report.entities_updated} updated in {report.duration_seconds:.1f}s")

    with e2:
        drug_count = stats.get("by_type", {}).get("drug", 0)
        st.metric("Drugs in DB", drug_count)
        if st.button("Enrich All Drugs", type="primary", disabled=drug_count == 0, key="enrich_drugs"):
            progress = st.progress(0)
            status = st.empty()

            def enrich_drug_progress(msg):
                status.markdown(f"**{msg}**")

            report = importer.enricher.enrich_all_drugs(enrich_drug_progress)
            progress.progress(1.0)
            st.success(f"Enriched: {report.entities_updated} updated in {report.duration_seconds:.1f}s")

    with e3:
        st.metric("Relationships", stats.get("total_relationships", 0))
        if st.button("Build Cross-References", type="primary", key="build_xrefs"):
            status = st.empty()

            def xref_progress(msg):
                status.markdown(f"**{msg}**")

            count = importer.build_cross_references(xref_progress)
            st.success(f"Created {count} cross-reference relationships")

    st.divider()

    # Enrichment audit
    st.subheader("Data Quality Audit")
    quality = importer.get_data_quality_report()

    if quality["total_entities"] == 0:
        st.info("No entities in database. Use Quick Seed or Custom Import to populate.")
    else:
        qcols = st.columns(4)
        for i, (metric_name, pct) in enumerate(quality.get("quality", {}).items()):
            col = qcols[i % 4]
            label = metric_name.replace("_", " ").title()
            col.metric(label, f"{pct:.0%}")

# ═══════════════════════════════════════════════════════════════════════
# Tab 4: Status & Quality
# ═══════════════════════════════════════════════════════════════════════

with tab_status:
    st.subheader("Database Status")

    # Entity counts by type
    by_type = stats.get("by_type", {})
    if by_type:
        import plotly.express as px

        type_data = [{"Entity Type": k, "Count": v} for k, v in sorted(by_type.items(), key=lambda x: -x[1])]
        fig = px.bar(
            type_data,
            x="Entity Type",
            y="Count",
            title="Entity Counts by Type",
            color="Count",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No entities in database yet.")

    st.divider()

    # Relationship summary
    st.subheader("Relationship Summary")
    rel_count = stats.get("total_relationships", 0)
    if rel_count > 0:
        from cognisom.library.models import RelationshipType as RT
        rel_types_to_check = [
            RT.ENCODES, RT.TARGETS, RT.BINDS_TO, RT.PART_OF,
            RT.REGULATES, RT.MUTATED_IN, RT.INHIBITS, RT.ACTIVATES,
        ]
        rel_data = []
        for rt in rel_types_to_check:
            rels = store.get_relationships_by_type(rt.value, limit=1000)
            if rels:
                rel_data.append({"Relationship": rt.value, "Count": len(rels)})

        if rel_data:
            fig2 = px.bar(
                rel_data,
                x="Relationship",
                y="Count",
                title="Relationship Counts by Type",
                color="Count",
                color_continuous_scale="Plasma",
            )
            fig2.update_layout(height=300)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No relationships yet.")

    st.divider()

    # External API health
    st.subheader("External API Health")
    if st.button("Test API Connectivity", key="test_apis"):
        import requests

        apis = {
            "NCBI E-utilities": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi?retmode=json",
            "UniProt": "https://rest.uniprot.org/uniprotkb/search?query=TP53&size=1&format=json",
            "PubChem": "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/aspirin/cids/JSON",
            "KEGG": "https://rest.kegg.jp/info/pathway",
            "STRING": "https://string-db.org/api/json/version",
            "Reactome": "https://reactome.org/ContentService/data/database/version",
            "RCSB PDB": "https://data.rcsb.org/rest/v1/core/entry/1TUP",
            "AlphaFold": "https://alphafold.ebi.ac.uk/api/prediction/P04637",
        }

        results = []
        api_progress = st.progress(0)
        for i, (name, url) in enumerate(apis.items()):
            try:
                t0 = time.time()
                resp = requests.get(url, timeout=10, headers={"User-Agent": "Cognisom/1.0"})
                ms = (time.time() - t0) * 1000
                results.append({
                    "API": name,
                    "Status": f"{resp.status_code}",
                    "Latency": f"{ms:.0f}ms",
                })
            except Exception as e:
                results.append({
                    "API": name,
                    "Status": f"Error: {str(e)[:40]}",
                    "Latency": "N/A",
                })
            api_progress.progress((i + 1) / len(apis))

        st.dataframe(results, use_container_width=True, hide_index=True)

# Footer
from cognisom.dashboard.footer import render_footer
render_footer()
