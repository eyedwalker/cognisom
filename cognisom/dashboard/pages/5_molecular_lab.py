"""Molecular Lab - Interactive protein/molecule structure explorer."""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Molecular Lab | Cognisom", page_icon="🧬", layout="wide")

from cognisom.auth.middleware import streamlit_page_gate
user = streamlit_page_gate("5_molecular_lab")

st.title("Molecular Lab")
st.markdown(
    "Interactive molecular and protein structure explorer. "
    "Browse RCSB PDB structures, draw molecules from SMILES, and explore in 3D."
)

tab_protein, tab_molstar, tab_molecule, tab_targets, tab_structure_pred, tab_embeddings, tab_dna, tab_blast = st.tabs([
    "Protein Viewer", "Mol* Viewer", "Molecule Drawer", "Prostate Cancer Targets",
    "Structure Prediction", "Protein Embeddings", "DNA Analysis", "BLAST Search",
])

# ══════════════════════════════════════════════════════════════════════
# PROTEIN VIEWER
# ══════════════════════════════════════════════════════════════════════

with tab_protein:
    st.subheader("3D Protein Structure Viewer")
    st.markdown("Enter a PDB ID to fetch and visualize any protein structure from the RCSB database.")

    p1, p2 = st.columns([1, 3])

    with p1:
        pdb_id = st.text_input("PDB ID", value="2AM9",
                                help="4-character PDB code (e.g., 2AM9, 1XOW, 5T35)")

        st.markdown("**Quick access:**")
        quick_pdbs = {
            "2AM9": "AR Ligand Binding Domain",
            "1XOW": "PSA (KLK3) + substrate",
            "5T35": "PI3K alpha",
            "4ERJ": "PARP1 catalytic domain",
            "5IUS": "PD-L1 + antibody",
            "3LN1": "AR + enzalutamide",
            "6MXY": "CDK4/6",
            "1A52": "p53 DNA binding domain",
        }
        for pid, desc in quick_pdbs.items():
            if st.button(f"{pid}: {desc}", key=f"pdb_{pid}", use_container_width=True):
                pdb_id = pid

        st.markdown("---")
        style = st.selectbox("Render style",
                              ["cartoon", "stick", "cartoon+stick", "sphere", "line"],
                              key="pv_style")
        color = st.selectbox("Color scheme",
                              ["spectrum", "chain", "ssType"],
                              key="pv_color")
        surface = st.checkbox("Surface overlay", key="pv_surface")
        spin = st.checkbox("Auto-rotate", key="pv_spin")

    with p2:
        if pdb_id:
            from cognisom.dashboard.mol_viz import fetch_pdb, protein_viewer_html, pdb_stats

            with st.spinner(f"Fetching {pdb_id.upper()} from RCSB..."):
                pdb_data = fetch_pdb(pdb_id)

            if pdb_data:
                stats = pdb_stats(pdb_data)
                ms1, ms2, ms3, ms4 = st.columns(4)
                ms1.metric("PDB ID", pdb_id.upper())
                ms2.metric("Chains", stats["chains"])
                ms3.metric("Residues", stats["residues"])
                ms4.metric("Atoms", f"{stats['atoms']:,}")

                html = protein_viewer_html(
                    pdb_data, width=750, height=550,
                    style=style, color=color,
                    surface=surface, spin=spin)
                components.html(html, height=570, width=770)

                with st.expander("PDB header (first 30 lines)"):
                    header_lines = [l for l in pdb_data.splitlines()[:30]
                                    if l.startswith(("HEADER", "TITLE", "COMPND",
                                                     "SOURCE", "EXPDTA", "REMARK"))]
                    st.code("\n".join(header_lines), language=None)
            else:
                st.error(f"Could not fetch PDB: {pdb_id}")


# ══════════════════════════════════════════════════════════════════════
# MOL* (MOLSTAR) VIEWER
# ══════════════════════════════════════════════════════════════════════

with tab_molstar:
    st.subheader("Mol* (Molstar) Protein Viewer")
    st.markdown(
        "Full-featured molecular viewer powered by [Mol*](https://molstar.org). "
        "Supports PDB, mmCIF, AlphaFold models, and more."
    )

    ms_col1, ms_col2 = st.columns([1, 3])

    with ms_col1:
        molstar_source = st.radio(
            "Source",
            ["RCSB PDB", "AlphaFold DB", "Custom PDB ID"],
            key="molstar_source",
        )

        if molstar_source == "RCSB PDB":
            molstar_id = st.text_input("PDB ID", value="2AM9", key="molstar_pdb_id")
            molstar_url = f"https://www.ebi.ac.uk/pdbe/entry/view/{molstar_id.upper()}"
            embed_url = (
                f"https://molstar.org/viewer/"
                f"?pdb-url=https://files.rcsb.org/download/{molstar_id.upper()}.cif"
                f"&hide-controls=1"
            )
        elif molstar_source == "AlphaFold DB":
            uniprot_id = st.text_input(
                "UniProt ID", value="P10275",
                help="e.g. P10275 (Androgen Receptor), P07288 (PSA/KLK3)",
                key="molstar_uniprot_id",
            )
            st.markdown("**Quick access:**")
            af_targets = {
                "P10275": "Androgen Receptor",
                "P07288": "PSA (KLK3)",
                "P42336": "PI3K alpha",
                "Q07817": "BCL2",
                "P04637": "p53",
                "O14727": "APAF1",
            }
            for uid, desc in af_targets.items():
                if st.button(f"{uid}: {desc}", key=f"af_{uid}", use_container_width=True):
                    uniprot_id = uid
            embed_url = (
                f"https://molstar.org/viewer/"
                f"?pdb-url=https://alphafold.ebi.ac.uk/files/"
                f"AF-{uniprot_id.upper()}-F1-model_v4.cif"
                f"&hide-controls=1"
            )
        else:
            custom_pdb = st.text_input("PDB ID", value="5T35", key="molstar_custom")
            embed_url = (
                f"https://molstar.org/viewer/"
                f"?pdb-url=https://files.rcsb.org/download/{custom_pdb.upper()}.cif"
                f"&hide-controls=1"
            )

        st.markdown("---")
        viewer_height = st.slider("Viewer height", 400, 800, 600, 50, key="molstar_height")

    with ms_col2:
        # Documentation
        with st.expander("📖 How to use Mol* viewer", expanded=False):
            st.markdown("""
            **What this does:**
            Interactive 3D molecular viewer for proteins and other biomolecules.

            **Sources:**
            - **RCSB PDB**: Experimentally determined structures (X-ray, cryo-EM, NMR)
            - **AlphaFold DB**: AI-predicted structures for most human proteins

            **Controls:**
            - Left-click + drag: Rotate
            - Scroll: Zoom
            - Right-click + drag: Pan
            - Click atoms: Select and get info
            """)

        # Embed Mol* via iframe with better attributes
        molstar_html = f"""
        <iframe
            src="{embed_url}"
            width="100%"
            height="{viewer_height}px"
            style="border: 1px solid #444; border-radius: 8px; background: #1a1a2e;"
            allow="fullscreen"
            loading="lazy"
            referrerpolicy="no-referrer"
            sandbox="allow-scripts allow-same-origin allow-popups allow-forms"
        ></iframe>
        <p style="color: #888; font-size: 12px; margin-top: 5px;">
            If the viewer doesn't load, try refreshing or
            <a href="{embed_url}" target="_blank" style="color: #ff6b6b;">open in new tab</a>.
        </p>
        """
        components.html(molstar_html, height=viewer_height + 50)

        st.caption(
            "Mol* is developed by PDBe and RCSB PDB. "
            "Supports interactive selection, measurement, superposition, and export."
        )


# ══════════════════════════════════════════════════════════════════════
# MOLECULE DRAWER
# ══════════════════════════════════════════════════════════════════════

with tab_molecule:
    st.subheader("2D/3D Molecule Visualization")
    st.markdown("Enter a SMILES string to render the molecular structure and compute properties.")

    known_molecules = {
        "Enzalutamide (AR antagonist)":
            "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
        "Abiraterone (CYP17 inhibitor)":
            "CC(=O)OC1CCC2(C3CCC4=CC(=O)CCC4(C3CCC12C)C)C",
        "Docetaxel (taxane)":
            "CC1=CC2=C(CC1OC(=O)C3=CC=CC=C3NC(=O)OC(C)(C)C)C(=O)C(C2O)(OC(=O)C4=CC=CC=C4)C",
        "Olaparib (PARP inhibitor)":
            "C1CC1C(=O)N2CCN(CC2)C(=O)C3=C(C=CC(=C3)F)CC4=NNC(=O)C5=CC=CC=C54",
        "Pembrolizumab fragment":
            "CC(C)CC(NC(=O)C(CC1=CC=CC=C1)NC(=O)C(CC(=O)O)NC(=O)C(CO)NC(=O)C)C(=O)O",
        "Caffeine":
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Aspirin":
            "CC(=O)OC1=CC=CC=C1C(=O)O",
    }

    mol_choice = st.selectbox("Select a known molecule, or enter custom SMILES below:",
                               ["Custom SMILES"] + list(known_molecules.keys()))

    if mol_choice == "Custom SMILES":
        smiles_input = st.text_input("SMILES", value="c1ccccc1", help="Enter any valid SMILES string")
    else:
        smiles_input = known_molecules[mol_choice]
        st.code(smiles_input, language=None)

    if smiles_input:
        from cognisom.dashboard.mol_viz import (
            smiles_to_image, compute_properties, lipinski_check, smiles_to_3d_sdf
        )

        mv1, mv2 = st.columns([1, 1])

        with mv1:
            st.markdown("**2D Structure**")
            img = smiles_to_image(smiles_input, size=(500, 400))
            if img:
                st.image(img, width=500)
            else:
                st.error("Invalid SMILES - could not parse molecule")

        with mv2:
            props = compute_properties(smiles_input)
            if props:
                st.markdown("**Molecular Properties**")
                pp1, pp2 = st.columns(2)
                with pp1:
                    st.metric("Molecular Weight", f"{props['MW']} Da")
                    st.metric("LogP", props["LogP"])
                    st.metric("QED", props["QED"])
                    st.metric("Formula", props["Formula"])
                with pp2:
                    st.metric("HB Donors", props["HBD"])
                    st.metric("HB Acceptors", props["HBA"])
                    st.metric("TPSA", f"{props['TPSA']} A^2")
                    st.metric("Rot. Bonds", props["RotBonds"])

                st.markdown("---")
                lip = lipinski_check(props)
                if lip["pass"]:
                    st.success(f"Lipinski Rule of Five: **PASS** ({lip['violations']} violations)")
                else:
                    st.error(f"Lipinski Rule of Five: **FAIL** ({lip['violations']} violations)")
                for rule, ok in lip["checks"].items():
                    st.markdown(f"- {'Pass' if ok else '**FAIL**'}: {rule}")

        # PubChem lookup
        st.markdown("---")
        st.markdown("**PubChem Database Lookup**")
        if st.button("Search PubChem", key="btn_pubchem_lookup"):
            with st.spinner("Querying PubChem…"):
                try:
                    from cognisom.ncbi.pubchem import search_compound, similarity_search
                    pc = search_compound(smiles_input, search_type="smiles")
                    if pc:
                        pc_c1, pc_c2, pc_c3, pc_c4 = st.columns(4)
                        pc_c1.metric("PubChem CID", pc.cid)
                        pc_c2.metric("Name", pc.name[:30] if pc.name else "—")
                        pc_c3.metric("Formula", pc.molecular_formula)
                        pc_c4.metric("InChIKey", pc.inchi_key[:14] + "…" if pc.inchi_key else "—")
                        st.caption(f"[View on PubChem](https://pubchem.ncbi.nlm.nih.gov/compound/{pc.cid})")

                        with st.expander("Similar compounds in PubChem"):
                            similar = similarity_search(smiles_input, threshold=0.8, max_results=5)
                            if similar:
                                import pandas as pd
                                sim_df = pd.DataFrame(similar)
                                st.dataframe(sim_df, use_container_width=True)
                            else:
                                st.info("No similar compounds found (threshold: 80%).")
                    else:
                        st.info("Compound not found in PubChem.")
                except Exception as e:
                    st.warning(f"PubChem lookup failed: {e}")

        # 3D conformer
        st.markdown("---")
        st.markdown("**3D Conformer**")
        sdf = smiles_to_3d_sdf(smiles_input)
        if sdf:
            # Render with py3Dmol
            mol_3d_html = f"""
            <div id="mol3d" style="width:700px; height:400px; border:1px solid #333;
                 border-radius:8px; overflow:hidden; background:#1a1a2e;"></div>
            <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
            <script>
            (function() {{
                var viewer = $3Dmol.createViewer("mol3d", {{backgroundColor: "#1a1a2e"}});
                var sdfData = `{sdf.replace(chr(96), "").replace("$", "S")}`;
                viewer.addModel(sdfData, "sdf");
                viewer.setStyle({{}}, {{stick: {{colorscheme: "greenCarbon", radius: 0.2}}}});
                viewer.addSurface(
                    $3Dmol.SurfaceType.VDW,
                    {{opacity: 0.2, color: "cyan"}},
                    {{}}, {{}}
                );
                viewer.zoomTo();
                viewer.spin(true);
                viewer.render();
            }})();
            </script>
            """
            components.html(mol_3d_html, height=420, width=720)
        else:
            st.info("Could not generate 3D conformer for this molecule.")


# ══════════════════════════════════════════════════════════════════════
# PROSTATE CANCER TARGETS
# ══════════════════════════════════════════════════════════════════════

with tab_targets:
    st.subheader("Key Prostate Cancer Drug Targets")
    st.markdown(
        "These are the primary molecular targets in prostate cancer therapy. "
        "Click any target to view its 3D structure."
    )

    targets = [
        {
            "name": "Androgen Receptor (AR)",
            "pdb": "2AM9",
            "desc": "Primary driver of prostate cancer. Ligand binding domain shown. "
                    "Target of enzalutamide, apalutamide, darolutamide.",
            "drugs": ["Enzalutamide", "Apalutamide", "Darolutamide"],
            "pathway": "AR signaling",
        },
        {
            "name": "PSA / KLK3",
            "pdb": "1XOW",
            "desc": "Prostate-specific antigen. Serine protease used as biomarker. "
                    "Expressed by luminal epithelial cells.",
            "drugs": ["Biomarker (not drugged)"],
            "pathway": "Secretion",
        },
        {
            "name": "PI3K alpha",
            "pdb": "5T35",
            "desc": "PI3K/AKT/mTOR pathway kinase. Frequently mutated in CRPC. "
                    "Target of alpelisib.",
            "drugs": ["Alpelisib", "Copanlisib"],
            "pathway": "PI3K/AKT/mTOR",
        },
        {
            "name": "PARP1",
            "pdb": "4ERJ",
            "desc": "DNA damage repair enzyme. Synthetic lethal with BRCA1/2 mutations. "
                    "Target of olaparib, rucaparib.",
            "drugs": ["Olaparib", "Rucaparib", "Talazoparib"],
            "pathway": "DNA repair",
        },
        {
            "name": "PD-L1",
            "pdb": "5IUS",
            "desc": "Immune checkpoint ligand. Enables tumor immune evasion. "
                    "Target of pembrolizumab, atezolizumab.",
            "drugs": ["Pembrolizumab", "Atezolizumab"],
            "pathway": "Immune checkpoint",
        },
        {
            "name": "p53 (TP53)",
            "pdb": "1A52",
            "desc": "Tumor suppressor. Most commonly mutated gene in cancer. "
                    "DNA binding domain shown.",
            "drugs": ["APR-246 (eprenetapopt)"],
            "pathway": "Cell cycle / apoptosis",
        },
    ]

    for t in targets:
        with st.expander(f"**{t['name']}** | PDB: {t['pdb']} | {t['pathway']}"):
            tc1, tc2 = st.columns([2, 1])

            with tc2:
                st.markdown(f"**Description**: {t['desc']}")
                st.markdown(f"**Drugs**: {', '.join(t['drugs'])}")
                st.markdown(f"**Pathway**: {t['pathway']}")
                st.markdown(f"**PDB**: [{t['pdb']}](https://www.rcsb.org/structure/{t['pdb']})")

            with tc1:
                from cognisom.dashboard.mol_viz import fetch_pdb, protein_viewer_html, pdb_stats

                pdb_data = fetch_pdb(t["pdb"])
                if pdb_data:
                    stats = pdb_stats(pdb_data)
                    st.caption(f"{stats['chains']} chains | {stats['residues']} residues | {stats['atoms']:,} atoms")

                    html = protein_viewer_html(pdb_data, width=550, height=400,
                                                style="cartoon", color="spectrum",
                                                surface=False, spin=False)
                    components.html(html, height=420, width=570)
                else:
                    st.warning(f"Could not fetch {t['pdb']}")


# ══════════════════════════════════════════════════════════════════════
# STRUCTURE PREDICTION (OpenFold3 / Boltz-2)
# ══════════════════════════════════════════════════════════════════════

with tab_structure_pred:
    st.subheader("AI Structure Prediction")
    st.markdown(
        "Predict 3D protein structure from amino acid sequence using "
        "**OpenFold3** or **Boltz-2** (via NVIDIA NIMs). Optionally runs MSA-Search first."
    )

    # Documentation expander
    with st.expander("📖 How to use & where to get sequences", expanded=False):
        st.markdown("""
        **What this does:**
        Predicts the 3D structure of a protein from its amino acid sequence using AI (similar to AlphaFold).

        **When to use:**
        - You have a novel protein sequence without a known structure
        - You want to model mutant proteins
        - You need a structure for docking studies

        **Where to get amino acid sequences:**
        - [UniProt](https://www.uniprot.org) - Search by gene name (e.g., TP53, AR, BRCA1)
        - [NCBI Protein](https://www.ncbi.nlm.nih.gov/protein) - Search by accession
        - [Ensembl](https://www.ensembl.org) - Human genome browser

        **Input format:**
        Single-letter amino acid codes, e.g.: `MKTAYIAKQRQISFVKSHFSRQDILDLWIYHT...`
        """)

    sp_method = st.radio("Method", ["OpenFold3", "Boltz-2"], horizontal=True, key="sp_method")

    # Example sequences
    st.markdown("**Quick-load example sequences:**")
    example_proteins = {
        "p53 (tumor suppressor)": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD",
        "Insulin (hormone)": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
        "GFP (fluorescent protein)": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK",
    }
    ex_col1, ex_col2, ex_col3 = st.columns(3)
    seq_to_use = ""
    for i, (name, seq) in enumerate(example_proteins.items()):
        col = [ex_col1, ex_col2, ex_col3][i % 3]
        with col:
            if st.button(name, key=f"ex_prot_{i}", use_container_width=True):
                seq_to_use = seq

    sp_seq = st.text_area(
        "Amino acid sequence",
        height=120,
        value=seq_to_use,
        placeholder="MKTAYIAKQRQISFVKSHFSRQDILDLWIYHT…",
        key="sp_seq",
        help="Single-letter amino acid codes. Max ~1000 residues.",
    )

    if st.button("Predict Structure", key="btn_sp", type="primary"):
        if not sp_seq.strip():
            st.error("Enter an amino acid sequence.")
        else:
            with st.spinner(f"Running {sp_method} structure prediction (this may take a while)…"):
                try:
                    from cognisom.agent import ResearchAgent
                    agent = ResearchAgent()
                    result = agent.run_tool(
                        "structure_prediction",
                        sequence=sp_seq.strip().replace("\n", "").replace(" ", ""),
                        method="boltz2" if sp_method == "Boltz-2" else "openfold3",
                    )
                    if result.success:
                        st.success(f"Structure predicted in {result.elapsed_sec}s")
                        d = result.data
                        c1, c2 = st.columns(2)
                        c1.metric("Method", d.get("method", sp_method))
                        c2.metric("Confidence", f"{d.get('plddt', d.get('confidence', 'N/A'))}")
                        st.text_area("Structure data (preview)", value=d.get("structure_data", "")[:2000], height=200)
                    else:
                        st.error(f"Prediction failed: {result.error}")
                except Exception as e:
                    st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════
# PROTEIN EMBEDDINGS (ESM2-650M)
# ══════════════════════════════════════════════════════════════════════

with tab_embeddings:
    st.subheader("Protein Embeddings (ESM2-650M)")
    st.markdown(
        "Embed protein sequences using Meta's **ESM2-650M** model. "
        "Compare two sequences or score a mutation's impact."
    )

    # Documentation
    with st.expander("📖 How to use & what embeddings mean", expanded=False):
        st.markdown("""
        **What this does:**
        Converts a protein sequence into a numerical vector (embedding) that captures its biological properties.

        **When to use:**
        - **Compare proteins**: Find how similar two proteins are functionally
        - **Score mutations**: See if a mutation significantly changes the protein
        - **Clustering**: Group similar proteins together
        - **ML input**: Use embeddings as features for machine learning

        **Output:**
        - **Cosine similarity** near 1.0 = very similar proteins
        - **Cosine similarity** near 0 = unrelated proteins

        **Example applications:**
        - Compare wild-type vs mutant p53
        - Find proteins similar to a drug target
        - Analyze evolutionary relationships
        """)

    emb_mode = st.radio("Mode", ["Single embedding", "Compare two sequences"], horizontal=True, key="emb_mode")

    # Example sequences for comparison
    st.markdown("**Example: p53 wild-type vs mutant (R175H)**")
    p53_wt = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
    p53_r175h = p53_wt[:174] + "H" + p53_wt[175:]  # R175H mutation

    ex_btn_col1, ex_btn_col2 = st.columns(2)
    with ex_btn_col1:
        if st.button("Load p53 WT", key="emb_p53_wt", use_container_width=True):
            st.session_state["emb_seq1_val"] = p53_wt
    with ex_btn_col2:
        if st.button("Load p53 WT + R175H mutant", key="emb_p53_both", use_container_width=True):
            st.session_state["emb_seq1_val"] = p53_wt
            st.session_state["emb_seq2_val"] = p53_r175h

    emb_seq1 = st.text_area("Sequence 1", height=100, key="emb_seq1",
                             value=st.session_state.get("emb_seq1_val", ""),
                             placeholder="MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPS…")

    emb_seq2 = ""
    if emb_mode == "Compare two sequences":
        emb_seq2 = st.text_area("Sequence 2", height=100, key="emb_seq2",
                                 placeholder="MEEPQSDPSVEPPLSQETFSDLWKLLHENNVLSPLPS…")

    if st.button("Compute Embedding", key="btn_emb", type="primary"):
        seq1 = emb_seq1.strip().replace("\n", "").replace(" ", "")
        seq2 = emb_seq2.strip().replace("\n", "").replace(" ", "") if emb_seq2 else ""
        if not seq1:
            st.error("Enter at least one sequence.")
        else:
            with st.spinner("Computing ESM2 embedding…"):
                try:
                    from cognisom.agent import ResearchAgent
                    agent = ResearchAgent()
                    result = agent.run_tool("protein_embedding", sequence=seq1, compare_to=seq2)
                    if result.success:
                        st.success(f"Embedding computed in {result.elapsed_sec}s")
                        d = result.data
                        c1, c2 = st.columns(2)
                        c1.metric("Sequence length", d.get("sequence_length", ""))
                        c2.metric("Embedding dim", d.get("embedding_dimension", ""))

                        comp = d.get("comparison")
                        if comp:
                            st.markdown("#### Comparison")
                            c1, c2 = st.columns(2)
                            c1.metric("Cosine similarity", f"{comp.get('cosine_similarity', 0):.4f}")
                            c2.metric("Seq 2 length", comp.get("sequence_2_length", ""))
                    else:
                        st.error(f"Failed: {result.error}")
                except Exception as e:
                    st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════
# DNA ANALYSIS (Evo2)
# ══════════════════════════════════════════════════════════════════════

with tab_dna:
    st.subheader("DNA Sequence Analysis (Evo2)")
    st.markdown(
        "Analyse DNA sequences using the **Evo2-40B** genomic foundation model. "
        "Generate DNA or score mutations at the nucleotide level."
    )

    # Documentation
    with st.expander("📖 How to use & where to get DNA sequences", expanded=False):
        st.markdown("""
        **What this does:**
        - **Generate DNA**: Extend a seed sequence with biologically plausible DNA
        - **Score mutation**: Evaluate if a nucleotide change is likely deleterious

        **When to use:**
        - Design synthetic promoters or regulatory elements
        - Predict effects of SNPs (Single Nucleotide Polymorphisms)
        - Generate sequences for synthetic biology

        **Where to get DNA sequences:**
        - [NCBI Gene](https://www.ncbi.nlm.nih.gov/gene) - Search by gene name
        - [Ensembl](https://www.ensembl.org) - Human genome browser
        - [UCSC Genome Browser](https://genome.ucsc.edu) - Get specific regions

        **Input format:**
        Standard nucleotides: A, T, G, C (uppercase)
        """)

    dna_mode = st.radio("Mode", ["Generate DNA", "Score mutation"], horizontal=True, key="dna_mode")

    # Example DNA sequences
    example_dna = {
        "TP53 promoter (partial)": "ATGGAGGAGCCGCAGTCAGATCCTAGCGTCGAGCCCCCTCTGAGTCAGGAAACATTTTCAGACCTATGGAAACTACTTCCTGAAAACAACGTTCTGTCC",
        "AR exon 1 (partial)": "ATGGAAGTGCAGTTAGGGCTGGGAAGGGTCTACCCTCGGCCGCCGTCCAAGACCTACCGAGGAGCTTTCCAGAATCTGTTCCAGAGCGTGCGCGAAGTG",
        "BRCA1 exon 11 (partial)": "ATGATTCTGCCCTGTGAGGGAAATAATGACCTCTATTACTGGCACTTAATATTAAATAAAAGTTTAGGCATACATTGCTAACGGACAGTATAAGGGGAA",
    }

    # Initialize session state for DNA sequence
    if "dna_loaded_seq" not in st.session_state:
        st.session_state.dna_loaded_seq = ""

    st.markdown("**Quick-load example sequences:**")
    dna_col1, dna_col2, dna_col3 = st.columns(3)
    for i, (name, seq) in enumerate(example_dna.items()):
        col = [dna_col1, dna_col2, dna_col3][i % 3]
        with col:
            if st.button(name, key=f"dna_ex_{i}", use_container_width=True):
                st.session_state.dna_loaded_seq = seq

    # ── Import DNA from file ──
    st.markdown("**Import from file:**")
    dna_file = st.file_uploader(
        "Upload a DNA sequence file",
        type=["fasta", "fa", "fna", "txt", "gb", "gbk"],
        key="dna_file_upload",
        help="Supports FASTA (.fasta, .fa, .fna), plain text (.txt), and GenBank (.gb, .gbk) formats.",
    )
    if dna_file is not None:
        raw = dna_file.read().decode("utf-8", errors="ignore")
        fname = dna_file.name.lower()
        imported_seq = ""
        if fname.endswith((".gb", ".gbk")):
            # GenBank: extract sequence from ORIGIN section
            in_origin = False
            parts = []
            for line in raw.splitlines():
                if line.startswith("ORIGIN"):
                    in_origin = True
                    continue
                if line.startswith("//"):
                    break
                if in_origin:
                    parts.append("".join(ch for ch in line if ch.isalpha()))
            imported_seq = "".join(parts).upper()
        else:
            # FASTA or plain text: skip header lines starting with '>'
            parts = []
            for line in raw.splitlines():
                stripped = line.strip()
                if stripped.startswith(">"):
                    if parts:
                        break  # stop at second record
                    continue
                parts.append(stripped)
            imported_seq = "".join(parts).upper()
        # Keep only valid nucleotides
        imported_seq = "".join(ch for ch in imported_seq if ch in "ATGCNRYSWKMBDHV")
        if imported_seq:
            st.session_state.dna_loaded_seq = imported_seq
            st.success(f"Imported {len(imported_seq)} nucleotides from **{dna_file.name}**")
        else:
            st.warning("No valid DNA nucleotides found in the uploaded file.")

    # ── Fetch from NCBI ──
    st.markdown("**Fetch from NCBI:**")
    ncbi_col1, ncbi_col2, ncbi_col3 = st.columns([2, 1, 1])
    with ncbi_col1:
        ncbi_query = st.text_input(
            "Gene name or accession",
            placeholder="e.g. BRCA1, TP53, NM_007294.4",
            key="ncbi_seq_query",
            help="Enter a gene symbol (BRCA1) or RefSeq accession (NM_007294.4).",
        )
    with ncbi_col2:
        ncbi_db = st.selectbox("Database", ["nucleotide", "protein"], key="ncbi_seq_db")
    with ncbi_col3:
        st.markdown("<br>", unsafe_allow_html=True)
        ncbi_fetch_btn = st.button("Fetch from NCBI", key="btn_ncbi_fetch", type="secondary")
    if ncbi_fetch_btn and ncbi_query.strip():
        with st.spinner(f"Fetching {ncbi_query} from NCBI {ncbi_db}…"):
            try:
                from cognisom.ncbi import fetch_sequence
                seq_result = fetch_sequence(ncbi_query.strip(), db=ncbi_db)
                if seq_result:
                    st.session_state.dna_loaded_seq = seq_result.sequence
                    st.success(f"Fetched **{seq_result.accession}** — {seq_result.length:,} {'nt' if ncbi_db == 'nucleotide' else 'aa'}")
                    with st.expander("Sequence details"):
                        st.caption(seq_result.description)
                        st.code(seq_result.sequence[:500] + ("…" if len(seq_result.sequence) > 500 else ""), language=None)
                else:
                    st.warning(f"No sequences found for '{ncbi_query}' in NCBI {ncbi_db}.")
            except Exception as e:
                st.error(f"NCBI fetch error: {e}")

    if dna_mode == "Generate DNA":
        dna_prompt = st.text_area(
            "DNA prompt sequence",
            height=100,
            value=st.session_state.dna_loaded_seq,
            placeholder="ATGCGATCGATCGATCG…",
            key="dna_prompt",
            help="Seed sequence; the model will extend it.",
        )
        dna_len = st.slider("Tokens to generate", 10, 500, 100, key="dna_gen_len")

        if st.button("Generate", key="btn_dna_gen", type="primary"):
            if not dna_prompt.strip():
                st.error("Enter a DNA seed sequence.")
            else:
                with st.spinner("Running Evo2 generation…"):
                    try:
                        from cognisom.nims import Evo2Client
                        client = Evo2Client()
                        result = client.generate(dna_prompt.strip(), num_tokens=dna_len)
                        st.success("Generated!")
                        st.code(result.generated_sequence, language=None)
                        st.caption(f"Length: {len(result.generated_sequence)} nt")
                    except Exception as e:
                        st.error(f"Error: {e}")

    else:  # Score mutation
        dna_wt = st.text_area("Wild-type DNA", height=80, key="dna_wt_score",
                               value=st.session_state.dna_loaded_seq,
                               placeholder="ATGCGATCGATCG…")
        mut_col1, mut_col2 = st.columns(2)
        with mut_col1:
            mut_pos = st.number_input("Mutation position (0-indexed)", min_value=0, value=0, key="dna_mut_pos")
        with mut_col2:
            mut_base = st.selectbox("Mutant base", ["A", "T", "G", "C"], key="dna_mut_base")

        if st.button("Score Mutation", key="btn_dna_score", type="primary"):
            seq = dna_wt.strip().replace("\n", "").replace(" ", "")
            if not seq:
                st.error("Enter a DNA sequence.")
            elif mut_pos >= len(seq):
                st.error(f"Position {mut_pos} is out of range (sequence length {len(seq)}).")
            else:
                with st.spinner("Scoring with Evo2…"):
                    try:
                        from cognisom.nims import Evo2Client
                        client = Evo2Client()
                        result = client.score_mutation(seq, mut_pos, mut_base)
                        st.success("Scored!")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Position", mut_pos)
                        c2.metric("Change", f"{seq[mut_pos]} → {mut_base}")
                        c3.metric("Log-likelihood ratio", f"{result.get('log_likelihood_ratio', 0):.4f}")
                    except Exception as e:
                        st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════════════
# BLAST SEARCH
# ══════════════════════════════════════════════════════════════════════

with tab_blast:
    st.subheader("NCBI BLAST Search")
    st.markdown(
        "Search sequence databases using **BLAST** (Basic Local Alignment Search Tool). "
        "Find similar DNA or protein sequences across NCBI databases."
    )

    blast_program = st.selectbox(
        "Program",
        ["blastn", "blastp", "blastx", "tblastn"],
        help="blastn: DNA→DNA, blastp: Protein→Protein, blastx: DNA→Protein, tblastn: Protein→DNA",
        key="blast_program",
    )

    blast_db_options = {
        "blastn": ["core_nt", "nt", "refseq_rna", "refseq_genomic"],
        "blastp": ["swissprot", "nr", "refseq_protein", "pdb"],
        "blastx": ["swissprot", "nr", "refseq_protein"],
        "tblastn": ["core_nt", "nt", "refseq_rna"],
    }
    blast_db = st.selectbox(
        "Database", blast_db_options.get(blast_program, ["nr"]),
        key="blast_db",
    )

    blast_seq = st.text_area(
        "Query sequence",
        height=120,
        placeholder="Paste a DNA or protein sequence (FASTA or plain)…",
        key="blast_query_seq",
    )

    blast_col1, blast_col2 = st.columns(2)
    with blast_col1:
        blast_evalue = st.select_slider("E-value threshold", [0.001, 0.01, 0.1, 1, 10, 100], value=10, key="blast_evalue")
    with blast_col2:
        blast_hits = st.slider("Max hits", 10, 100, 50, key="blast_max_hits")

    if st.button("Run BLAST", key="btn_blast_run", type="primary"):
        seq = blast_seq.strip()
        if not seq:
            st.error("Enter a query sequence.")
        else:
            try:
                from cognisom.ncbi.blast import submit_blast, wait_for_blast, get_blast_results

                progress = st.progress(0, text="Submitting BLAST search…")
                rid = submit_blast(seq, program=blast_program, database=blast_db,
                                   expect=blast_evalue, hitlist_size=blast_hits)
                st.info(f"BLAST job submitted (RID: `{rid}`). Waiting for results…")

                def _blast_progress(status, elapsed):
                    pct = min(int(elapsed / 120 * 100), 95)
                    progress.progress(pct, text=f"Status: {status} ({elapsed:.0f}s elapsed)")

                final_status = wait_for_blast(rid, timeout=300, poll_interval=15,
                                              callback=_blast_progress)

                if final_status == "READY":
                    progress.progress(100, text="Retrieving results…")
                    result = get_blast_results(rid, max_hits=blast_hits)

                    if result.hits:
                        st.success(f"Found **{len(result.hits)}** hits")
                        import pandas as pd
                        df = pd.DataFrame([
                            {
                                "Accession": h.accession,
                                "Identity %": f"{h.identity_pct:.1f}",
                                "E-value": f"{h.evalue:.2e}",
                                "Score": f"{h.score:.0f}",
                                "Length": h.length,
                            }
                            for h in result.hits
                        ])
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.warning("BLAST completed but no hits found. Try a longer sequence or less stringent E-value.")
                elif final_status == "TIMEOUT":
                    st.warning("BLAST search timed out (>5 min). Try a shorter sequence or smaller database.")
                else:
                    st.error(f"BLAST search failed with status: {final_status}")
            except Exception as e:
                st.error(f"BLAST error: {e}")

# Footer
from cognisom.dashboard.footer import render_footer
render_footer()
