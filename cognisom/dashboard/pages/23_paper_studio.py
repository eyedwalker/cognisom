"""Paper Studio — compose and export scientific manuscripts.

Three tabs:
    1. Compose — write abstract, intro, discussion; preview auto-generated sections
    2. Figures & Tables — gallery, select, reorder, caption
    3. Export — generate LaTeX, compile PDF, download bundle
"""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st

st.set_page_config(page_title="Paper Studio | Cognisom", page_icon="📝", layout="wide")

from cognisom.auth.middleware import streamlit_page_gate
user = streamlit_page_gate("23_paper_studio")

from cognisom.library.store import EntityStore
from cognisom.library.models import EntityType, ResearchProject, SimulationRun
from cognisom.workflow.artifact_store import ArtifactStore
from cognisom.workflow.project_manager import ProjectManager
from cognisom.workflow.paper_generator import PaperGenerator


# ── Shared state ──────────────────────────────────────────────────────

@st.cache_resource
def get_store():
    return EntityStore()

@st.cache_resource
def get_artifact_store():
    return ArtifactStore()

@st.cache_resource
def get_project_manager():
    return ProjectManager(get_store())

@st.cache_resource
def get_paper_generator():
    return PaperGenerator(get_artifact_store(), get_store())


store = get_store()
artifacts = get_artifact_store()
proj_mgr = get_project_manager()
paper_gen = get_paper_generator()


# ── Page header ───────────────────────────────────────────────────────

st.title("Paper Studio")
st.markdown(
    "Compose a scientific manuscript from your research project's simulation results. "
    "Write your narrative, review auto-generated figures and tables, and export "
    "as a LaTeX bundle ready for journal submission."
)


# ── Project selector ──────────────────────────────────────────────────

projects = proj_mgr.list_projects()
if not projects:
    st.warning(
        "No research projects found. Create a project in the **Researcher Workflow** page first."
    )
    from cognisom.dashboard.footer import render_footer
    render_footer()
    st.stop()

project_options = {f"{p.title} ({p.entity_id[:8]})": p.entity_id for p in projects}
selected_label = st.selectbox("Select Research Project", list(project_options.keys()))
project_id = project_options[selected_label]
project = proj_mgr.get_project(project_id)

if not project:
    st.error("Project not found.")
    st.stop()

# Show project summary
summary = proj_mgr.get_project_summary(project_id)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Runs", summary.get("n_runs", 0))
col2.metric("Completed", summary.get("n_completed", 0))
col3.metric("Paper Status", project.paper_status.title())
col4.metric("Citations", len(project.bibtex_entries))


# ── Tabs ──────────────────────────────────────────────────────────────

tab_compose, tab_figures, tab_export = st.tabs([
    "Compose", "Figures & Tables", "Export"
])


# ═══════════════════════════════════════════════════════════════════════
# TAB 1: Compose
# ═══════════════════════════════════════════════════════════════════════

with tab_compose:
    st.subheader("Manuscript Sections")
    st.caption(
        "Write your abstract, introduction, and discussion below. "
        "Methods and Results sections are auto-generated from simulation data."
    )

    authors = st.text_input(
        "Authors",
        value=st.session_state.get("paper_authors", ""),
        key="paper_authors_input",
        placeholder="e.g., Walker, D. and Smith, J.",
    )
    st.session_state["paper_authors"] = authors

    # Abstract
    abstract = st.text_area(
        "Abstract",
        value=project.abstract,
        height=150,
        key="paper_abstract",
        placeholder="Summarize the study objectives, methods, key results, and conclusions.",
    )

    # Introduction
    introduction = st.text_area(
        "Introduction",
        value=project.introduction,
        height=200,
        key="paper_intro",
        placeholder="Background context, research question, and study motivation.",
    )

    # Discussion
    discussion = st.text_area(
        "Discussion",
        value=project.discussion,
        height=200,
        key="paper_discussion",
        placeholder="Interpretation of results, comparison to prior work, limitations, and future directions.",
    )

    col_save, col_preview = st.columns(2)

    with col_save:
        if st.button("Save Draft", type="primary", key="save_draft_btn"):
            proj_mgr.update_text(
                project_id,
                abstract=abstract,
                introduction=introduction,
                discussion=discussion,
            )
            st.success("Draft saved.")

    with col_preview:
        if st.button("Preview Auto-Generated Sections", key="preview_auto_btn"):
            # Generate manuscript to get preview
            tex_path = paper_gen.generate_manuscript(
                project_id, authors=authors,
            )
            if tex_path:
                preview = paper_gen.get_tex_preview(project_id)
                if preview:
                    # Show just Methods + Results sections
                    methods_start = preview.find("\\section{Methods}")
                    results_end = preview.find("\\section{Discussion}")
                    if methods_start > 0 and results_end > methods_start:
                        section = preview[methods_start:results_end]
                        st.code(section, language="latex")
                    else:
                        st.code(preview[:3000], language="latex")
                else:
                    st.warning("Could not generate preview.")
            else:
                st.warning("Manuscript generation failed. Ensure completed runs are linked.")

    # Citation manager
    st.markdown("---")
    st.subheader("Citations")

    if project.bibtex_entries:
        for key, entry in project.bibtex_entries.items():
            with st.expander(f"\\cite{{{key}}}"):
                st.code(entry, language="bibtex")
                if st.button(f"Remove", key=f"rm_cite_{key}"):
                    proj_mgr.remove_citation(project_id, key)
                    st.rerun()

    new_cite_key = st.text_input("Citation Key", key="new_cite_key", placeholder="e.g., smith2024")
    new_cite_bib = st.text_area(
        "BibTeX Entry", key="new_cite_bib", height=100,
        placeholder="@article{smith2024,\n  title={...},\n  author={...},\n  year={2024},\n  journal={...}\n}",
    )
    if st.button("Add Citation", key="add_cite_btn"):
        if new_cite_key and new_cite_bib:
            proj_mgr.add_citation(project_id, new_cite_key, new_cite_bib)
            st.success(f"Added \\cite{{{new_cite_key}}}")
            st.rerun()
        else:
            st.error("Both key and BibTeX entry are required.")


# ═══════════════════════════════════════════════════════════════════════
# TAB 2: Figures & Tables
# ═══════════════════════════════════════════════════════════════════════

with tab_figures:
    st.subheader("Available Figures")

    figs = paper_gen.get_available_figures(project_id)

    if not figs:
        st.info(
            "No figures found. Run simulations first, and figures will be "
            "auto-generated in the run artifacts directory."
        )
    else:
        # Gallery view
        cols_per_row = 3
        for i in range(0, len(figs), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(figs):
                    fig_info = figs[idx]
                    with col:
                        try:
                            st.image(
                                fig_info["path"],
                                caption=f"{fig_info['name']} ({fig_info['run_name']})",
                                use_container_width=True,
                            )
                        except Exception:
                            st.write(f"**{fig_info['name']}**")
                            st.caption(f"Run: {fig_info['run_name']}")
                        size_kb = fig_info["size_bytes"] / 1024
                        st.caption(f"{size_kb:.0f} KB")

    # Metrics comparison table
    st.markdown("---")
    st.subheader("Metrics Table Preview")

    completed_runs = []
    for run_id in project.run_ids:
        entity = store.get_entity(run_id)
        if isinstance(entity, SimulationRun) and entity.run_status == "completed":
            completed_runs.append(entity)

    if completed_runs:
        import pandas as pd
        metrics_data = {}
        for run in completed_runs:
            col_name = run.name[:20]
            metrics_data[col_name] = run.final_metrics

        df = pd.DataFrame(metrics_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No completed runs in this project.")


# ═══════════════════════════════════════════════════════════════════════
# TAB 3: Export
# ═══════════════════════════════════════════════════════════════════════

with tab_export:
    st.subheader("Generate & Export")

    col_gen, col_status = st.columns([2, 1])

    with col_gen:
        if st.button("Generate LaTeX Manuscript", type="primary", key="gen_tex_btn"):
            with st.spinner("Generating manuscript..."):
                # Save latest text first
                proj_mgr.update_text(
                    project_id,
                    abstract=st.session_state.get("paper_abstract", project.abstract),
                    introduction=st.session_state.get("paper_intro", project.introduction),
                    discussion=st.session_state.get("paper_discussion", project.discussion),
                )

                tex_path = paper_gen.generate_manuscript(
                    project_id,
                    authors=st.session_state.get("paper_authors", ""),
                )

            if tex_path:
                st.success(f"Manuscript generated.")
                st.session_state["tex_generated"] = True
                st.session_state["tex_path"] = tex_path
            else:
                st.error("Manuscript generation failed. Check that runs are completed.")

    with col_status:
        paper_dir = artifacts.paper_dir(project_id)
        tex_exists = (paper_dir / "manuscript.tex").exists()
        bib_exists = (paper_dir / "manuscript.bib").exists()
        pdf_exists = (paper_dir / "manuscript.pdf").exists()
        zip_exists = (paper_dir / "bundle.zip").exists()

        st.write(f"**manuscript.tex**: {'Ready' if tex_exists else 'Not generated'}")
        st.write(f"**manuscript.bib**: {'Ready' if bib_exists else 'Not generated'}")
        st.write(f"**manuscript.pdf**: {'Ready' if pdf_exists else 'Not compiled'}")
        st.write(f"**bundle.zip**: {'Ready' if zip_exists else 'Not created'}")

    # Preview LaTeX source
    if tex_exists:
        st.markdown("---")
        with st.expander("Preview LaTeX Source"):
            preview = paper_gen.get_tex_preview(project_id)
            if preview:
                st.code(preview, language="latex")

    # Compile PDF
    st.markdown("---")
    col_pdf, col_zip = st.columns(2)

    with col_pdf:
        if tex_exists:
            if st.button("Compile PDF", key="compile_pdf_btn"):
                with st.spinner("Compiling PDF (requires pdflatex)..."):
                    pdf_path = paper_gen.compile_pdf(str(paper_dir / "manuscript.tex"))
                if pdf_path:
                    st.success("PDF compiled successfully.")
                else:
                    st.warning(
                        "PDF compilation unavailable. Install texlive: "
                        "`apt-get install texlive-latex-base texlive-latex-extra texlive-bibtex-extra`"
                    )
        else:
            st.info("Generate the manuscript first.")

    with col_zip:
        if tex_exists:
            if st.button("Create ZIP Bundle", key="create_zip_btn"):
                zip_path = artifacts.create_zip_bundle(project_id)
                if zip_path:
                    st.success("Bundle created.")
                    st.session_state["zip_path"] = zip_path

    # Download buttons
    st.markdown("---")
    st.subheader("Downloads")

    download_cols = st.columns(4)

    with download_cols[0]:
        if tex_exists:
            tex_content = (paper_dir / "manuscript.tex").read_text()
            st.download_button(
                "Download .tex",
                tex_content,
                "manuscript.tex",
                "text/x-latex",
            )

    with download_cols[1]:
        if bib_exists:
            bib_content = (paper_dir / "manuscript.bib").read_text()
            st.download_button(
                "Download .bib",
                bib_content,
                "manuscript.bib",
                "text/x-bibtex",
            )

    with download_cols[2]:
        if pdf_exists:
            pdf_content = (paper_dir / "manuscript.pdf").read_bytes()
            st.download_button(
                "Download PDF",
                pdf_content,
                "manuscript.pdf",
                "application/pdf",
            )

    with download_cols[3]:
        zip_file = paper_dir / "bundle.zip"
        if zip_file.exists():
            zip_content = zip_file.read_bytes()
            st.download_button(
                "Download ZIP Bundle",
                zip_content,
                f"{project.title[:30]}_bundle.zip",
                "application/zip",
            )


# ── Footer ────────────────────────────────────────────────────────────

from cognisom.dashboard.footer import render_footer
render_footer()
