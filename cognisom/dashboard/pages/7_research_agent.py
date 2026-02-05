"""Research Agent – interactive gene investigation, mutation analysis & drug target exploration."""

import sys
import json
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st

st.set_page_config(page_title="Research Agent | Cognisom", page_icon="🔬", layout="wide")

from cognisom.auth.middleware import streamlit_page_gate
user = streamlit_page_gate("7_research_agent")

from cognisom.agent import ResearchAgent

# ── Initialise agent ────────────────────────────────────────────────


@st.cache_resource
def get_agent() -> ResearchAgent:
    return ResearchAgent()


agent = get_agent()

# ── Page header ─────────────────────────────────────────────────────

st.title("Research Agent")
st.markdown(
    "Interactive tool-based agent for **gene investigation**, **mutation analysis**, "
    "**drug target exploration**, and **custom queries** — powered by public databases + NVIDIA NIMs."
)

# ── Session state defaults ──────────────────────────────────────────

if "agent_results" not in st.session_state:
    st.session_state.agent_results = {}

# ── Tabs ────────────────────────────────────────────────────────────

tab_gene, tab_mutation, tab_drug, tab_custom = st.tabs([
    "Gene Investigation",
    "Mutation Analysis",
    "Drug Target Explorer",
    "Custom Query",
])

# ====================================================================
# TAB 1: Gene Investigation
# ====================================================================

with tab_gene:
    st.subheader("Gene Investigation")
    st.markdown(
        "Enter a gene symbol to run a full investigation: "
        "NCBI Gene → UniProt → PDB structures → cBioPortal mutations → PubMed literature."
    )

    col_a, col_b = st.columns([2, 1])
    with col_a:
        gene_input = st.text_input("Gene symbol", value="TP53", key="gene_inv_input")
    with col_b:
        study_input = st.text_input(
            "cBioPortal study",
            value="prad_tcga",
            key="gene_inv_study",
            help="Default: prostate adenocarcinoma (TCGA)",
        )

    if st.button("Investigate Gene", key="btn_gene_inv", type="primary"):
        progress_bar = st.progress(0.0, text="Starting investigation…")

        def _gene_progress(label: str, current: int, total: int) -> None:
            frac = current / max(total, 1)
            progress_bar.progress(frac, text=label)

        with st.spinner("Running 5-step investigation pipeline…"):
            wf = agent.investigate_gene(gene_input, study=study_input, progress_cb=_gene_progress)

        progress_bar.progress(1.0, text="Complete!")
        st.session_state.agent_results["gene_investigation"] = wf

    # Display results
    wf = st.session_state.agent_results.get("gene_investigation")
    if wf:
        st.markdown(f"### Results for **{wf.workflow_name}**")
        st.caption(f"{wf.steps_completed}/{wf.steps_total} steps completed")

        if wf.errors:
            for e in wf.errors:
                st.warning(e)

        # Gene info
        gi = wf.results.get("gene_info")
        if gi and gi.success and isinstance(gi.data, dict) and "gene_id" in gi.data:
            d = gi.data
            st.markdown("#### Gene Info")
            c1, c2, c3 = st.columns(3)
            c1.metric("Symbol", d.get("symbol", ""))
            c2.metric("Chromosome", d.get("chromosome", ""))
            c3.metric("Type", d.get("gene_type", ""))
            st.markdown(f"**{d.get('full_name', '')}**")
            if d.get("aliases"):
                st.caption(f"Aliases: {d['aliases']}")
            if d.get("summary"):
                with st.expander("Gene summary"):
                    st.write(d["summary"])

        # Protein info
        pi = wf.results.get("protein_info")
        if pi and pi.success and isinstance(pi.data, dict) and "accession" in pi.data:
            d = pi.data
            st.markdown("#### Protein")
            c1, c2, c3 = st.columns(3)
            c1.metric("UniProt", d.get("accession", ""))
            c2.metric("Length", d.get("length", 0))
            c3.metric("GO terms", len(d.get("go_terms", [])))
            if d.get("function"):
                with st.expander("Function"):
                    st.write(d["function"])
            if d.get("sequence"):
                with st.expander("Sequence preview"):
                    st.code(d["sequence"], language=None)

        # PDB structures
        sr = wf.results.get("structures")
        if sr and sr.success and isinstance(sr.data, list) and sr.data:
            st.markdown("#### 3D Structures")
            for s in sr.data[:5]:
                if isinstance(s, dict) and "pdb_id" in s:
                    st.markdown(
                        f"**[{s['pdb_id']}](https://www.rcsb.org/structure/{s['pdb_id']})** — "
                        f"{s.get('title', '')[:100]}  \n"
                        f"Method: {s.get('method', 'N/A')} · Resolution: {s.get('resolution', 'N/A')} Å"
                    )

        # Cancer mutations
        cm = wf.results.get("cancer_mutations")
        if cm and cm.success and isinstance(cm.data, dict):
            d = cm.data
            st.markdown("#### Cancer Mutations")
            c1, c2 = st.columns(2)
            c1.metric("Study", d.get("study_name", "")[:40])
            c2.metric("Total mutations", d.get("total_mutations", 0))
            muts = d.get("mutation_types", [])
            if muts:
                import plotly.express as px
                import pandas as pd

                df = pd.DataFrame(muts)
                fig = px.bar(df, x="type", y="count", title="Mutation Types", color="type")
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # Literature
        lit = wf.results.get("literature")
        if lit and lit.success and isinstance(lit.data, list):
            st.markdown("#### Recent Literature")
            for paper in lit.data[:5]:
                if isinstance(paper, dict):
                    st.markdown(f"- **{paper.get('title', '')}** ({paper.get('year', '')})")
                    st.caption(
                        f"  {', '.join(paper.get('authors', [])[:3])} · "
                        f"[PubMed]({paper.get('url', '')})"
                    )

        # Export
        if wf.results:
            export_data = {}
            for k, r in wf.results.items():
                export_data[k] = {"success": r.success, "data": r.data, "error": r.error}
            st.download_button(
                "Export investigation as JSON",
                data=json.dumps(export_data, indent=2, default=str),
                file_name=f"gene_investigation_{gene_input}.json",
                mime="application/json",
                key="dl_gene_inv",
            )

# ====================================================================
# TAB 2: Mutation Analysis
# ====================================================================

with tab_mutation:
    st.subheader("Mutation Analysis")
    st.markdown(
        "Compare wild-type vs mutant protein sequences using **ESM2 embeddings** "
        "and optionally score DNA-level impact with **Evo2**."
    )

    gene_mut = st.text_input("Gene symbol (for context)", value="TP53", key="mut_gene")

    col1, col2 = st.columns(2)
    with col1:
        wt_seq = st.text_area(
            "Wild-type protein sequence",
            height=120,
            placeholder="MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPS…",
            key="mut_wt_seq",
        )
    with col2:
        mut_seq = st.text_area(
            "Mutant protein sequence",
            height=120,
            placeholder="MEEPQSDPSVEPPLSQETFSDLWKLLHENNVLSPLPS…",
            key="mut_mut_seq",
        )

    with st.expander("Optional: DNA sequences (for Evo2 scoring)"):
        dna_wt = st.text_area("Wild-type DNA", height=80, key="mut_dna_wt")
        dna_mut = st.text_area("Mutant DNA", height=80, key="mut_dna_mut")

    if st.button("Analyse Mutation", key="btn_mut", type="primary"):
        if not wt_seq or not mut_seq:
            st.error("Both wild-type and mutant sequences are required.")
        else:
            progress_bar = st.progress(0.0, text="Starting analysis…")

            def _mut_progress(label, cur, tot):
                progress_bar.progress(cur / max(tot, 1), text=label)

            with st.spinner("Running mutation analysis pipeline…"):
                wf = agent.mutation_analysis(
                    gene_mut, wt_seq, mut_seq,
                    dna_wt=dna_wt, dna_mut=dna_mut,
                    progress_cb=_mut_progress,
                )
            progress_bar.progress(1.0, text="Complete!")
            st.session_state.agent_results["mutation_analysis"] = wf

    wf = st.session_state.agent_results.get("mutation_analysis")
    if wf:
        st.markdown(f"### Results: {wf.workflow_name}")

        if wf.errors:
            for e in wf.errors:
                st.warning(e)

        # Embedding comparison
        emb = wf.results.get("embedding_comparison")
        if emb and emb.success and isinstance(emb.data, dict):
            comp = emb.data.get("comparison", {})
            if comp:
                st.markdown("#### Embedding Comparison (ESM2)")
                c1, c2 = st.columns(2)
                c1.metric("Cosine similarity", f"{comp.get('cosine_similarity', 0):.4f}")
                c2.metric("Embedding dimension", emb.data.get("embedding_dimension", ""))

        # Mutation impact
        mi = wf.results.get("mutation_impact")
        if mi and mi.success and isinstance(mi.data, dict):
            st.markdown("#### Mutation Impact")
            prot = mi.data.get("protein_level", {})
            if prot and "error" not in prot:
                c1, c2, c3 = st.columns(3)
                c1.metric("Cosine similarity", f"{prot.get('cosine_similarity', 0):.4f}")
                c2.metric("Euclidean distance", f"{prot.get('euclidean_distance', 0):.4f}")
                c3.metric("Interpretation", prot.get("interpretation", ""))

            dna = mi.data.get("dna_level", {})
            if dna and "error" not in dna:
                st.markdown("**DNA-level (Evo2)**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Position", dna.get("position", ""))
                c2.metric("Change", f"{dna.get('wt_base', '')} → {dna.get('mut_base', '')}")
                c3.metric("Log-likelihood ratio", f"{dna.get('log_likelihood_ratio', 0):.4f}")

        # Literature
        lit = wf.results.get("literature")
        if lit and lit.success and isinstance(lit.data, list):
            st.markdown("#### Related Literature")
            for p in lit.data:
                if isinstance(p, dict):
                    st.markdown(f"- **{p.get('title', '')}** [{p.get('year', '')}]({p.get('url', '')})")

# ====================================================================
# TAB 3: Drug Target Explorer
# ====================================================================

with tab_drug:
    st.subheader("Drug Target Explorer")
    st.markdown(
        "Explore a gene as a drug target: fetch protein info, find structures, "
        "generate candidate molecules with **GenMol**, and dock with **DiffDock**."
    )

    col1, col2 = st.columns(2)
    with col1:
        drug_gene = st.text_input("Target gene", value="AR", key="drug_gene")
    with col2:
        seed_smiles = st.text_input(
            "Seed SMILES (optional)",
            placeholder="CC1=CC2=C(C=C1)NC(=O)C2",
            key="drug_smiles",
            help="Provide a starting molecule for fragment-based generation",
        )

    num_mols = st.slider("Molecules to generate", 5, 50, 10, key="drug_n_mols")

    if st.button("Explore Target", key="btn_drug", type="primary"):
        progress_bar = st.progress(0.0, text="Starting exploration…")

        def _drug_progress(label, cur, tot):
            progress_bar.progress(cur / max(tot, 1), text=label)

        with st.spinner("Running drug target pipeline…"):
            wf = agent.drug_target_exploration(
                drug_gene,
                seed_smiles=seed_smiles,
                num_molecules=num_mols,
                progress_cb=_drug_progress,
            )
        progress_bar.progress(1.0, text="Complete!")
        st.session_state.agent_results["drug_target"] = wf

    wf = st.session_state.agent_results.get("drug_target")
    if wf:
        st.markdown(f"### Results: {wf.workflow_name}")

        if wf.errors:
            for e in wf.errors:
                st.warning(e)

        # Protein info
        pi = wf.results.get("protein_info")
        if pi and pi.success and isinstance(pi.data, dict) and "accession" in pi.data:
            d = pi.data
            st.markdown("#### Target Protein")
            c1, c2 = st.columns(2)
            c1.metric("UniProt", d.get("accession", ""))
            c2.metric("Length", d.get("length", 0))
            if d.get("function"):
                with st.expander("Function"):
                    st.write(d["function"])

        # Structures
        sr = wf.results.get("structures")
        if sr and sr.success and isinstance(sr.data, list):
            st.markdown("#### Known Structures")
            for s in sr.data[:3]:
                if isinstance(s, dict) and "pdb_id" in s:
                    st.markdown(
                        f"**[{s['pdb_id']}](https://www.rcsb.org/structure/{s['pdb_id']})** — "
                        f"{s.get('title', '')[:80]}"
                    )

        # Generated molecules
        gm = wf.results.get("generated_molecules")
        if gm and gm.success and isinstance(gm.data, list) and gm.data:
            st.markdown("#### Generated Candidate Molecules")
            import pandas as pd
            df = pd.DataFrame(gm.data)
            st.dataframe(df, use_container_width=True)

        # Docking
        dock = wf.results.get("docking")
        if dock and dock.success and isinstance(dock.data, dict):
            st.markdown("#### Docking Results (DiffDock)")
            c1, c2 = st.columns(2)
            c1.metric("Poses", dock.data.get("num_poses", 0))
            c2.metric("Best confidence", f"{dock.data.get('best_confidence', 0):.3f}")

        # Export
        if wf.results:
            export_data = {}
            for k, r in wf.results.items():
                export_data[k] = {"success": r.success, "data": r.data, "error": r.error}
            st.download_button(
                "Export exploration as JSON",
                data=json.dumps(export_data, indent=2, default=str),
                file_name=f"drug_target_{drug_gene}.json",
                mime="application/json",
                key="dl_drug",
            )

# ====================================================================
# TAB 4: Custom Query
# ====================================================================

with tab_custom:
    st.subheader("Custom Query")
    st.markdown("Run any individual tool directly. Select a tool and fill in the parameters.")

    tools_list = agent.list_tools()
    tool_names = [t["name"] for t in tools_list]
    tool_descs = {t["name"]: t["description"] for t in tools_list}

    selected_tool = st.selectbox("Tool", tool_names, key="custom_tool_select")
    st.caption(tool_descs.get(selected_tool, ""))

    # Get parameter info from the tool
    tool_obj = agent.registry.get(selected_tool)
    params_spec = getattr(tool_obj, "parameters", {}) if tool_obj else {}

    # Dynamic parameter inputs
    param_values = {}
    for pname, pdesc in params_spec.items():
        param_values[pname] = st.text_input(
            f"{pname}",
            key=f"custom_{selected_tool}_{pname}",
            help=pdesc,
        )

    if st.button("Run Tool", key="btn_custom", type="primary"):
        # Filter empty params
        kwargs = {k: v for k, v in param_values.items() if v}
        # Convert numeric-looking values
        for k, v in kwargs.items():
            if v.isdigit():
                kwargs[k] = int(v)

        with st.spinner(f"Running {selected_tool}…"):
            result = agent.run_tool(selected_tool, **kwargs)

        st.session_state.agent_results["custom_query"] = result

    result = st.session_state.agent_results.get("custom_query")
    if result:
        if result.success:
            st.success(f"{result.tool_name} completed in {result.elapsed_sec}s")
            st.json(result.data if isinstance(result.data, (dict, list)) else {"result": str(result.data)})
        else:
            st.error(f"{result.tool_name} failed: {result.error}")

        st.download_button(
            "Export result as JSON",
            data=json.dumps({"tool": result.tool_name, "success": result.success, "data": result.data, "error": result.error}, indent=2, default=str),
            file_name=f"tool_result_{result.tool_name}.json",
            mime="application/json",
            key="dl_custom",
        )

# Footer
from cognisom.dashboard.footer import render_footer
render_footer()
