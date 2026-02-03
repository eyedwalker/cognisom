"""Discovery page - NIM drug discovery pipeline with real molecular visualization."""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
import streamlit.components.v1 as components
import os
import numpy as np

st.set_page_config(page_title="Discovery | Cognisom", page_icon="🧬", layout="wide")

from cognisom.auth.middleware import streamlit_page_gate
user = streamlit_page_gate("2_discovery")


def _load_api_key():
    key = os.environ.get("NVIDIA_API_KEY", "")
    if not key:
        env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("NVIDIA_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
    return key


# ── Page ─────────────────────────────────────────────────────────────

st.title("NIM Drug Discovery Pipeline")
st.markdown(
    "Generate drug candidates with **NVIDIA MolMIM**, visualize molecular structures, "
    "design protein binders with **RFdiffusion**, and optimize sequences with **ProteinMPNN**."
)

api_key = _load_api_key()
if not api_key or api_key.startswith("your-"):
    st.warning("NVIDIA API key not configured. Set `NVIDIA_API_KEY` in `.env`.")
    st.stop()

os.environ["NVIDIA_API_KEY"] = api_key

st.divider()

# ── Configuration ────────────────────────────────────────────────────

st.subheader("Pipeline Configuration")
cfg1, cfg2 = st.columns(2)

with cfg1:
    scaffold = st.text_input(
        "Seed SMILES (drug scaffold)",
        value="CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
        help="Enzalutamide scaffold (prostate cancer AR antagonist)")
    target = st.selectbox("Target pathway",
                          ["AR (Androgen Receptor)", "PI3K/AKT", "PARP", "PD-1/PD-L1"])

with cfg2:
    generator = st.radio(
        "Molecule generator",
        ["MolMIM", "GenMol"],
        help="MolMIM: interpolation-based. GenMol: fragment-based (newer).",
        horizontal=True,
    )
    num_molecules = st.slider("Molecules to generate", 1, 20, 5)
    run_protein = st.checkbox("Run RFdiffusion protein binder design", value=True,
                              help="Design a protein binder for the AR ligand binding domain")
    run_sequence = st.checkbox("Run ProteinMPNN sequence optimization", value=True,
                               help="Optimize the designed binder sequence")

# Show scaffold molecule
st.markdown("**Seed molecule (Enzalutamide scaffold):**")
from cognisom.dashboard.mol_viz import smiles_to_image, compute_properties, lipinski_check
scaffold_img = smiles_to_image(scaffold, size=(500, 350))
if scaffold_img:
    st.image(scaffold_img, width=500)
    props = compute_properties(scaffold)
    if props:
        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("MW", f"{props['MW']}")
        p2.metric("LogP", f"{props['LogP']}")
        p3.metric("QED", f"{props['QED']}")
        p4.metric("HBD / HBA", f"{props['HBD']} / {props['HBA']}")
        p5.metric("Formula", props["Formula"])

st.divider()

run_btn = st.button("Run Discovery Pipeline", type="primary", use_container_width=True)

# ── Pipeline execution ───────────────────────────────────────────────

if run_btn or st.session_state.get("discovery_ran"):
    if run_btn:
        # ── Step 1: MolMIM ──
        gen_label = "GenMol" if generator == "GenMol" else "MolMIM"
        with st.spinner(f"Step 1: Generating molecules with {gen_label}..."):
            try:
                if generator == "GenMol":
                    from cognisom.nims.genmol import GenMolClient
                    client = GenMolClient()
                    molecules = client.generate(smiles=scaffold, num_molecules=num_molecules)
                else:
                    from cognisom.nims.molmim import MolMIMClient
                    client = MolMIMClient()
                    molecules = client.generate(scaffold, num_molecules=num_molecules)
                st.session_state["molmim_results"] = molecules
            except Exception as e:
                st.error(f"MolMIM failed: {e}")
                st.stop()

        # ── Step 2: Drug Bridge ──
        with st.spinner("Step 2: Converting to drug candidates..."):
            try:
                from cognisom.bridge.drug_bridge import DrugBridge
                bridge = DrugBridge()
                candidates = bridge.convert_molecules(molecules)
                st.session_state["drug_candidates"] = candidates
            except Exception as e:
                st.error(f"Drug bridge failed: {e}")
                st.stop()

        # ── Step 3: RFdiffusion (optional) ──
        if run_protein:
            with st.spinner("Step 3: Designing protein binder with RFdiffusion..."):
                try:
                    from cognisom.nims.rfdiffusion import RFdiffusionClient
                    rf_client = RFdiffusionClient()
                    # Use AR ligand binding domain (PDB: 2AM9)
                    from cognisom.dashboard.mol_viz import fetch_pdb
                    ar_pdb = fetch_pdb("2AM9")
                    if ar_pdb:
                        binder = rf_client.design_binder(ar_pdb,
                                                         target_residues="A710-730",
                                                         binder_length="60")
                        st.session_state["binder_result"] = binder
                        st.session_state["target_pdb"] = ar_pdb
                    else:
                        st.warning("Could not fetch AR structure (PDB: 2AM9)")
                except Exception as e:
                    st.warning(f"RFdiffusion: {e}")

        # ── Step 4: ProteinMPNN (optional) ──
        if run_sequence and st.session_state.get("binder_result"):
            with st.spinner("Step 4: Optimizing sequence with ProteinMPNN..."):
                try:
                    from cognisom.nims.proteinmpnn import ProteinMPNNClient
                    mpnn_client = ProteinMPNNClient()
                    binder = st.session_state["binder_result"]
                    sequences = mpnn_client.design_for_binder(binder.pdb_data, num_sequences=3)
                    st.session_state["designed_sequences"] = sequences
                except Exception as e:
                    st.warning(f"ProteinMPNN: {e}")

        st.session_state["discovery_ran"] = True

    molecules = st.session_state.get("molmim_results", [])
    candidates = st.session_state.get("drug_candidates", [])
    binder = st.session_state.get("binder_result")
    target_pdb = st.session_state.get("target_pdb")
    sequences = st.session_state.get("designed_sequences", [])

    # ══════════════════════════════════════════════════════════════════
    # MOLECULE VISUALIZATION
    # ══════════════════════════════════════════════════════════════════

    if molecules:
        st.divider()
        st.subheader("Generated Molecules")

        # Molecule grid image
        from cognisom.dashboard.mol_viz import molecule_grid
        smiles_list = [m.smiles for m in molecules]
        labels = [f"MOL-{i+1} (QED={m.score:.2f})" for i, m in enumerate(molecules)]
        grid_img = molecule_grid(smiles_list, labels=labels,
                                 mols_per_row=min(len(molecules), 4))
        if grid_img:
            st.image(grid_img, use_container_width=True,
                     caption="Generated molecule structures (RDKit 2D rendering)")

        # Individual molecule detail
        st.markdown("---")
        st.markdown("**Molecule Details** (click to expand)")

        for i, mol in enumerate(molecules):
            with st.expander(f"MOL-{i+1:03d} | QED = {mol.score:.3f}"):
                det1, det2 = st.columns([1, 2])

                with det1:
                    img = smiles_to_image(mol.smiles, size=(400, 300))
                    if img:
                        st.image(img, width=400)

                with det2:
                    props = compute_properties(mol.smiles)
                    if props:
                        st.markdown("**Molecular Properties**")
                        pc1, pc2, pc3, pc4 = st.columns(4)
                        pc1.metric("MW", f"{props['MW']}")
                        pc2.metric("LogP", f"{props['LogP']}")
                        pc3.metric("QED", f"{props['QED']}")
                        pc4.metric("TPSA", f"{props['TPSA']}")

                        pc5, pc6, pc7, pc8 = st.columns(4)
                        pc5.metric("HB Donors", props["HBD"])
                        pc6.metric("HB Acceptors", props["HBA"])
                        pc7.metric("Rot. Bonds", props["RotBonds"])
                        pc8.metric("Heavy Atoms", props["Atoms"])

                        lip = lipinski_check(props)
                        st.markdown(f"**Lipinski Ro5**: {'PASS' if lip['pass'] else 'FAIL'} "
                                    f"({lip['violations']} violations)")
                        for rule, ok in lip["checks"].items():
                            st.markdown(f"- {'Pass' if ok else 'FAIL'}: {rule}")

                    st.markdown("**SMILES**")
                    st.code(mol.smiles, language=None)

        # Property comparison chart
        import plotly.graph_objects as go

        st.markdown("---")
        st.subheader("Molecule Property Comparison")

        all_props = []
        for mol in molecules:
            p = compute_properties(mol.smiles)
            if p:
                all_props.append(p)

        if all_props:
            categories = ["QED", "MW_norm", "LogP_norm", "TPSA_norm", "HBD_norm", "RotBonds_norm"]
            cat_labels = ["QED", "MW", "LogP", "TPSA", "HBD", "Rot. Bonds"]

            fig_radar = go.Figure()
            for i, (mol, p) in enumerate(zip(molecules, all_props)):
                vals = [
                    p["QED"],
                    min(p["MW"] / 500, 1.0),
                    min(abs(p["LogP"]) / 5, 1.0),
                    min(p["TPSA"] / 140, 1.0),
                    min(p["HBD"] / 5, 1.0),
                    min(p["RotBonds"] / 10, 1.0),
                ]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]], theta=cat_labels + [cat_labels[0]],
                    fill="toself", name=f"MOL-{i+1}", opacity=0.5))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Molecular property radar (normalized)", height=450)
            st.plotly_chart(fig_radar, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════
    # DRUG CANDIDATE PARAMETERS
    # ══════════════════════════════════════════════════════════════════

    if candidates:
        st.divider()
        st.subheader("Simulation-Ready Drug Parameters")

        import plotly.express as px

        cand_data = []
        for i, c in enumerate(candidates):
            sel = c.cancer_kill_rate / max(c.normal_toxicity, 1e-6)
            cand_data.append({
                "Drug": f"DRUG-{i+1:03d}",
                "Cancer Kill Rate": c.cancer_kill_rate,
                "Normal Toxicity": c.normal_toxicity,
                "Selectivity": sel,
                "Diffusion (um2/s)": c.diffusion_coefficient,
                "Half-life (h)": c.half_life,
                "Immune Modulation": c.immune_modulation,
            })

        st.dataframe(cand_data, use_container_width=True, hide_index=True)

        # Selectivity bar chart
        fig_sel = px.bar(
            x=[d["Drug"] for d in cand_data],
            y=[d["Selectivity"] for d in cand_data],
            color=[d["Selectivity"] for d in cand_data],
            color_continuous_scale="RdYlGn",
            labels={"x": "Drug Candidate", "y": "Selectivity (kill/toxicity ratio)"},
            title="Drug selectivity index (higher = better therapeutic window)")
        fig_sel.update_layout(height=350)
        st.plotly_chart(fig_sel, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════
    # 3D PROTEIN STRUCTURE
    # ══════════════════════════════════════════════════════════════════

    if target_pdb:
        st.divider()
        st.subheader("Androgen Receptor - Ligand Binding Domain (PDB: 2AM9)")
        st.markdown(
            "The AR-LBD is the primary drug target in prostate cancer. "
            "Enzalutamide and other antiandrogens bind to this domain."
        )

        from cognisom.dashboard.mol_viz import protein_viewer_html, pdb_stats

        stats = pdb_stats(target_pdb)
        s1, s2, s3 = st.columns(3)
        s1.metric("Chains", stats["chains"])
        s2.metric("Residues", stats["residues"])
        s3.metric("Atoms", f"{stats['atoms']:,}")

        # Interactive 3D viewer
        v_col1, v_col2 = st.columns([3, 1])
        with v_col2:
            viz_style = st.selectbox("Style", ["cartoon", "stick", "cartoon+stick", "sphere"])
            viz_color = st.selectbox("Color", ["spectrum", "chain", "ssType"])
            show_surface = st.checkbox("Show surface", value=False)
            spin = st.checkbox("Spin", value=False)

        with v_col1:
            viewer_html = protein_viewer_html(
                target_pdb, width=700, height=500,
                style=viz_style, color=viz_color,
                surface=show_surface, spin=spin)
            components.html(viewer_html, height=520, width=720)

    # ── Designed binder ──
    if binder:
        st.divider()
        st.subheader("Designed Protein Binder (RFdiffusion)")
        st.markdown("RFdiffusion designed a de novo protein binder targeting the AR-LBD.")

        from cognisom.dashboard.mol_viz import protein_viewer_html, pdb_stats

        b_stats = pdb_stats(binder.pdb_data)
        b1, b2, b3 = st.columns(3)
        b1.metric("Binder Length", f"{binder.binder_length} residues")
        b2.metric("Chains", b_stats["chains"])
        b3.metric("Atoms", f"{b_stats['atoms']:,}")

        binder_html = protein_viewer_html(
            binder.pdb_data, width=700, height=500,
            style="cartoon", color="chain", surface=True)
        components.html(binder_html, height=520, width=720)

    # ── Designed sequences ──
    if sequences:
        st.divider()
        st.subheader("Optimized Sequences (ProteinMPNN)")
        st.markdown("ProteinMPNN optimized the amino acid sequence of the designed binder.")

        for i, seq in enumerate(sequences):
            with st.expander(f"Sequence {i+1} | Score = {seq.score:.3f} | Recovery = {seq.recovery:.1%}"):
                st.code(seq.sequence, language=None)
                seq_len = len(seq.sequence)
                st.markdown(f"**Length**: {seq_len} residues | "
                            f"**Score**: {seq.score:.4f} | "
                            f"**Recovery**: {seq.recovery:.1%}")

    # ══════════════════════════════════════════════════════════════════
    # PIPELINE STATUS
    # ══════════════════════════════════════════════════════════════════

    st.divider()
    st.subheader("Pipeline Status")
    steps = [
        ("MolMIM", "Molecule Generation", len(molecules) > 0),
        ("DrugBridge", "Pharmacological Conversion", len(candidates) > 0),
        ("RFdiffusion", "Protein Binder Design", binder is not None),
        ("ProteinMPNN", "Sequence Optimization", len(sequences) > 0),
        ("DiffDock", "Molecular Docking", False),
    ]
    for name, desc, done in steps:
        st.markdown(f"- **{name}** ({desc}): `{'COMPLETE' if done else 'PENDING'}`")

else:
    st.info("Configure parameters above and click **Run Discovery Pipeline** to generate "
            "molecules and visualize structures.")
