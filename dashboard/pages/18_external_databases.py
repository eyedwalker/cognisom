"""
External Databases Browser
==========================

Browse and import data from external biological databases:
- KEGG: Pathways and gene-pathway relationships
- PubChem: Compound properties and structures
- STRING: Protein-protein interaction networks
- Reactome: Pathway enrichment analysis
"""

import streamlit as st
import plotly.graph_objects as go
import networkx as nx
from typing import List

st.set_page_config(
    page_title="External Databases | Cognisom",
    page_icon="🌐",
    layout="wide",
)

# Auth gate
try:
    from cognisom.auth.middleware import streamlit_page_gate
    user = streamlit_page_gate(required_tier="researcher")
except Exception:
    user = None

st.title("🌐 External Databases")
st.markdown("Browse and import from KEGG, PubChem, STRING, Reactome, and more")

# ═══════════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ KEGG Pathways", "🧪 PubChem", "🕸️ STRING Networks", "⚗️ Reactome", "🔍 Unified Search"
])

# ─────────────────────────────────────────────────────────────────────────
# Tab 1: KEGG Pathways
# ─────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("KEGG Pathways")
    st.markdown("Search and browse KEGG pathway maps and gene-pathway relationships.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Search Pathways")

        query = st.text_input("Search KEGG", "prostate cancer", key="kegg_query")
        organism = st.selectbox(
            "Organism",
            ["hsa (Human)", "mmu (Mouse)", "rno (Rat)"],
            format_func=lambda x: x.split(" ")[1].strip("()")
        )
        organism_code = organism.split()[0]

        if st.button("🔍 Search KEGG", key="kegg_search"):
            with st.spinner("Searching KEGG..."):
                try:
                    from cognisom.library.external_sources import KEGGClient

                    kegg = KEGGClient()
                    pathways = kegg.search_pathways(query, organism=organism_code)

                    if pathways:
                        st.session_state.kegg_results = pathways
                        st.success(f"Found {len(pathways)} pathways")
                    else:
                        st.warning("No pathways found")
                        st.session_state.kegg_results = []

                except Exception as e:
                    st.error(f"Search failed: {e}")

        st.divider()

        # Browse all pathways
        if st.button("📋 List All Human Pathways"):
            with st.spinner("Loading pathways..."):
                try:
                    from cognisom.library.external_sources import KEGGClient
                    kegg = KEGGClient()
                    pathways = kegg.list_human_pathways()
                    st.session_state.kegg_results = pathways
                    st.success(f"Loaded {len(pathways)} pathways")
                except Exception as e:
                    st.error(f"Failed: {e}")

    with col2:
        st.markdown("### Results")

        if "kegg_results" in st.session_state and st.session_state.kegg_results:
            pathways = st.session_state.kegg_results

            # Display as table
            pathway_data = [
                {"ID": p.pathway_id, "Name": p.name[:60]}
                for p in pathways[:50]
            ]
            st.dataframe(pathway_data, use_container_width=True)

            # Select pathway for details
            selected_id = st.selectbox(
                "Select pathway for details",
                [p.pathway_id for p in pathways[:50]],
                format_func=lambda x: next((p.name for p in pathways if p.pathway_id == x), x)
            )

            if st.button("📖 Get Pathway Details"):
                with st.spinner("Loading pathway..."):
                    try:
                        from cognisom.library.external_sources import KEGGClient
                        kegg = KEGGClient()
                        pathway = kegg.get_pathway(selected_id)

                        if pathway:
                            st.markdown(f"**{pathway.name}**")
                            st.markdown(f"ID: `{pathway.pathway_id}`")
                            if pathway.description:
                                st.markdown(f"_{pathway.description}_")
                            st.markdown(f"**Genes:** {len(pathway.genes)}")

                            if pathway.genes:
                                with st.expander("View Genes"):
                                    st.write(", ".join(pathway.genes[:100]))

                            st.markdown(f"[View on KEGG]({pathway.url})")

                            # Import to entity library
                            if st.button("📥 Import to Entity Library"):
                                st.info("Import functionality - would add pathway and genes to library")

                    except Exception as e:
                        st.error(f"Failed: {e}")
        else:
            st.info("Search for pathways to see results")

# ─────────────────────────────────────────────────────────────────────────
# Tab 2: PubChem
# ─────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("PubChem Compounds")
    st.markdown("Search for small molecules, drugs, and their properties.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Search Compounds")

        compound_query = st.text_input("Compound Name", "enzalutamide", key="pubchem_query")
        limit = st.slider("Max Results", 1, 20, 5)

        if st.button("🔍 Search PubChem", key="pubchem_search"):
            with st.spinner("Searching PubChem..."):
                try:
                    from cognisom.library.external_sources import PubChemClient

                    pubchem = PubChemClient()
                    compounds = pubchem.search_compounds(compound_query, limit=limit)

                    if compounds:
                        st.session_state.pubchem_results = compounds
                        st.success(f"Found {len(compounds)} compounds")
                    else:
                        st.warning("No compounds found")

                except Exception as e:
                    st.error(f"Search failed: {e}")

        st.divider()

        # Quick lookups
        st.markdown("### Quick Lookup")
        quick_compounds = [
            "Enzalutamide", "Docetaxel", "Abiraterone",
            "Olaparib", "Pembrolizumab", "Cabazitaxel"
        ]

        for comp in quick_compounds[:3]:
            if st.button(comp, key=f"quick_{comp}"):
                try:
                    from cognisom.library.external_sources import PubChemClient
                    pubchem = PubChemClient()
                    result = pubchem.get_compound(comp)
                    if result:
                        st.session_state.pubchem_results = [result]
                        st.success(f"Found {comp}")
                except:
                    pass

    with col2:
        st.markdown("### Compound Details")

        if "pubchem_results" in st.session_state and st.session_state.pubchem_results:
            compounds = st.session_state.pubchem_results

            for compound in compounds:
                with st.expander(f"**{compound.name or f'CID {compound.cid}'}**", expanded=True):
                    col_c1, col_c2 = st.columns([1, 1])

                    with col_c1:
                        # Structure image
                        from cognisom.library.external_sources import PubChemClient
                        img_url = PubChemClient().get_compound_image_url(compound.cid, size=200)
                        st.image(img_url, caption="2D Structure", width=200)

                    with col_c2:
                        st.markdown(f"**CID:** {compound.cid}")
                        st.markdown(f"**Formula:** {compound.molecular_formula}")
                        st.markdown(f"**MW:** {compound.molecular_weight:.2f} g/mol")

                        if compound.xlogp is not None:
                            st.markdown(f"**XLogP:** {compound.xlogp:.2f}")
                        st.markdown(f"**TPSA:** {compound.tpsa:.1f} Å²")
                        st.markdown(f"**H-Bond Donors:** {compound.hbd}")
                        st.markdown(f"**H-Bond Acceptors:** {compound.hba}")

                    # SMILES
                    if compound.smiles:
                        st.code(compound.smiles, language=None)

                    st.markdown(f"[View on PubChem]({compound.url})")

                    # Lipinski's Rule of Five
                    ro5_pass = (
                        compound.molecular_weight <= 500 and
                        (compound.xlogp or 0) <= 5 and
                        compound.hbd <= 5 and
                        compound.hba <= 10
                    )
                    if ro5_pass:
                        st.success("✓ Passes Lipinski's Rule of Five")
                    else:
                        st.warning("⚠ Violates Lipinski's Rule of Five")

        else:
            st.info("Search for compounds to see results")

# ─────────────────────────────────────────────────────────────────────────
# Tab 3: STRING Networks
# ─────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("STRING Protein Interactions")
    st.markdown("Explore protein-protein interaction networks.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Input Proteins")

        proteins_input = st.text_area(
            "Protein/Gene Names (one per line)",
            "AR\nTP53\nPTEN\nBRCA1\nBRCA2\nERG",
            height=150
        )

        proteins = [p.strip() for p in proteins_input.split("\n") if p.strip()]

        score_threshold = st.slider(
            "Minimum Confidence Score",
            0, 1000, 400,
            help="STRING combined score (0-1000). Higher = more confident"
        )

        if st.button("🕸️ Get Interactions", key="string_search"):
            with st.spinner("Querying STRING..."):
                try:
                    from cognisom.library.external_sources import STRINGClient

                    string = STRINGClient()
                    interactions = string.get_interactions(proteins, score_threshold=score_threshold)

                    if interactions:
                        st.session_state.string_results = interactions
                        st.session_state.string_proteins = proteins
                        st.success(f"Found {len(interactions)} interactions")
                    else:
                        st.warning("No interactions found")

                except Exception as e:
                    st.error(f"Query failed: {e}")

        st.divider()

        # Presets
        st.markdown("### Presets")
        presets = {
            "AR Signaling": ["AR", "HSP90AA1", "FKBP5", "KLK3", "TMPRSS2", "NKX3-1"],
            "DNA Repair": ["BRCA1", "BRCA2", "ATM", "ATR", "RAD51", "PARP1"],
            "PI3K/AKT": ["PIK3CA", "PIK3R1", "AKT1", "PTEN", "MTOR", "RPS6KB1"],
        }

        for name, genes in presets.items():
            if st.button(name, key=f"preset_{name}"):
                st.session_state.string_proteins = genes

    with col2:
        st.markdown("### Interaction Network")

        if "string_results" in st.session_state and st.session_state.string_results:
            interactions = st.session_state.string_results
            proteins = st.session_state.get("string_proteins", [])

            # Build network visualization
            G = nx.Graph()

            # Add nodes
            nodes_in_network = set()
            for interaction in interactions:
                nodes_in_network.add(interaction.gene_a)
                nodes_in_network.add(interaction.gene_b)
                G.add_edge(
                    interaction.gene_a,
                    interaction.gene_b,
                    weight=interaction.combined_score / 1000
                )

            if G.number_of_nodes() > 0:
                # Create layout
                pos = nx.spring_layout(G, seed=42)

                # Create edge trace
                edge_x = []
                edge_y = []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=1, color='#888'),
                    hoverinfo='none',
                    mode='lines'
                )

                # Create node trace
                node_x = [pos[node][0] for node in G.nodes()]
                node_y = [pos[node][1] for node in G.nodes()]
                node_text = list(G.nodes())
                node_degree = [G.degree(node) for node in G.nodes()]

                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    hoverinfo='text',
                    text=node_text,
                    textposition="top center",
                    marker=dict(
                        size=[10 + d*3 for d in node_degree],
                        color=node_degree,
                        colorscale='YlGnBu',
                        colorbar=dict(title="Connections"),
                        line=dict(width=2, color='white')
                    )
                )

                fig = go.Figure(
                    data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f"STRING Network ({len(G.nodes())} proteins, {len(G.edges())} interactions)",
                        showlegend=False,
                        hovermode='closest',
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=500,
                    )
                )

                st.plotly_chart(fig, use_container_width=True)

            # Interaction table
            st.markdown("### Interactions")
            interaction_data = [
                {
                    "Protein A": i.gene_a,
                    "Protein B": i.gene_b,
                    "Score": i.combined_score,
                    "Experimental": i.experimental_score,
                }
                for i in interactions[:30]
            ]
            st.dataframe(interaction_data, use_container_width=True)

            # Get network image from STRING
            if proteins:
                from cognisom.library.external_sources import STRINGClient
                string = STRINGClient()
                img_url = string.get_network_image(proteins[:10])
                st.markdown(f"[View interactive network on STRING]({img_url})")

        else:
            st.info("Enter proteins and click 'Get Interactions'")

# ─────────────────────────────────────────────────────────────────────────
# Tab 4: Reactome
# ─────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Reactome Pathway Analysis")
    st.markdown("Pathway enrichment analysis using Reactome.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Input Genes")

        genes_input = st.text_area(
            "Gene Symbols (one per line)",
            "AR\nTP53\nPTEN\nBRCA1\nPIK3CA\nMYC\nRB1",
            height=150,
            key="reactome_genes"
        )

        genes = [g.strip() for g in genes_input.split("\n") if g.strip()]

        st.markdown(f"**{len(genes)} genes entered**")

        if st.button("⚗️ Run Pathway Analysis", key="reactome_analyze"):
            with st.spinner("Analyzing with Reactome..."):
                try:
                    from cognisom.library.external_sources import ReactomeClient

                    reactome = ReactomeClient()
                    results = reactome.analyze_genes(genes)

                    if results:
                        st.session_state.reactome_results = results
                        st.success(f"Found {len(results)} enriched pathways")
                    else:
                        st.warning("No significant pathways found")

                except Exception as e:
                    st.error(f"Analysis failed: {e}")

        st.divider()

        st.markdown("### Search Pathways")
        reactome_query = st.text_input("Search Reactome", "apoptosis")

        if st.button("🔍 Search"):
            try:
                from cognisom.library.external_sources import ReactomeClient
                reactome = ReactomeClient()
                pathways = reactome.search_pathways(reactome_query)
                if pathways:
                    st.session_state.reactome_search = pathways
                    st.success(f"Found {len(pathways)} pathways")
            except Exception as e:
                st.error(f"Search failed: {e}")

    with col2:
        st.markdown("### Enrichment Results")

        if "reactome_results" in st.session_state and st.session_state.reactome_results:
            results = st.session_state.reactome_results

            # Top pathways
            top_pathways = results[:15]

            # Bar chart
            fig = go.Figure(data=[go.Bar(
                y=[p["name"][:40] for p in reversed(top_pathways)],
                x=[-np.log10(p["fdr"]) for p in reversed(top_pathways)],
                orientation='h',
                marker_color='purple',
            )])

            import numpy as np

            fig.update_layout(
                title="Top Enriched Pathways",
                xaxis_title="-log10(FDR)",
                height=400,
                margin=dict(l=200),
            )

            st.plotly_chart(fig, use_container_width=True)

            # Table
            st.markdown("### All Results")
            pathway_data = [
                {
                    "Pathway": p["name"][:50],
                    "FDR": f"{p['fdr']:.2e}",
                    "Genes Found": p["found_genes"],
                    "Total Genes": p["total_genes"],
                }
                for p in results[:30]
            ]
            st.dataframe(pathway_data, use_container_width=True)

        elif "reactome_search" in st.session_state and st.session_state.reactome_search:
            pathways = st.session_state.reactome_search

            for pw in pathways[:10]:
                with st.expander(pw.name):
                    st.markdown(f"**ID:** `{pw.stable_id}`")
                    st.markdown(f"**Species:** {pw.species}")
                    if pw.is_disease:
                        st.warning("Disease pathway")

                    # Diagram link
                    from cognisom.library.external_sources import ReactomeClient
                    diagram_url = ReactomeClient().get_pathway_diagram_url(pw.stable_id)
                    st.markdown(f"[View Diagram]({diagram_url})")

        else:
            st.info("Enter genes and run pathway analysis")

# ─────────────────────────────────────────────────────────────────────────
# Tab 5: Unified Search
# ─────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Unified Search")
    st.markdown("Search across all databases simultaneously.")

    unified_query = st.text_input("Search All Databases", "androgen receptor", key="unified_query")

    if st.button("🔍 Search All", type="primary"):
        with st.spinner("Searching all databases..."):
            try:
                from cognisom.library.external_sources import get_data_manager

                manager = get_data_manager()
                results = manager.search_all(unified_query)

                st.session_state.unified_results = results

            except Exception as e:
                st.error(f"Search failed: {e}")

    if "unified_results" in st.session_state:
        results = st.session_state.unified_results

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### KEGG Pathways")
            for pw in results.get("kegg_pathways", []):
                st.markdown(f"- [{pw.name}]({pw.url})")
            if not results.get("kegg_pathways"):
                st.info("No KEGG results")

        with col2:
            st.markdown("### PubChem Compounds")
            for comp in results.get("pubchem_compounds", []):
                st.markdown(f"- [{comp.name}]({comp.url})")
                st.caption(f"MW: {comp.molecular_weight:.1f}")
            if not results.get("pubchem_compounds"):
                st.info("No PubChem results")

        with col3:
            st.markdown("### Reactome Pathways")
            for pw in results.get("reactome_pathways", []):
                st.markdown(f"- {pw.name}")
                st.caption(f"ID: {pw.stable_id}")
            if not results.get("reactome_pathways"):
                st.info("No Reactome results")

# Footer
st.divider()
st.caption("Cognisom External Databases | KEGG, PubChem, STRING, Reactome | eyentelligence inc.")
