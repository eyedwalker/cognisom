"""Scientific paper generator — Jinja2 LaTeX manuscripts from simulation results.

Generates complete LaTeX manuscript bundles ready for journal submission:
- Main .tex file from Jinja2 template
- .bib file with built-in + user citations
- Publication-quality figures (PDF)
- Parameter and metrics tables
- Reproducibility section with full provenance
- ZIP bundle for submission

Usage::

    gen = PaperGenerator(artifact_store, entity_store)
    tex_path = gen.generate_manuscript(project_id)
    pdf_path = gen.compile_pdf(tex_path)  # if pdflatex available
    zip_path = artifact_store.create_zip_bundle(project_id)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"


def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters in text."""
    if not text:
        return ""
    replacements = [
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


class PaperGenerator:
    """Generate complete scientific manuscripts from simulation results."""

    def __init__(self, artifact_store, store):
        self._artifacts = artifact_store
        self._store = store
        self._env = self._create_jinja_env()

    def _create_jinja_env(self):
        """Create Jinja2 environment with LaTeX-safe delimiters."""
        try:
            import jinja2
        except ImportError:
            log.error("jinja2 not installed. Install with: pip install jinja2")
            return None

        return jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(TEMPLATE_DIR)),
            block_start_string="\\BLOCK{",
            block_end_string="}",
            variable_start_string="\\VAR{",
            variable_end_string="}",
            comment_start_string="\\#{",
            comment_end_string="}",
            line_statement_prefix="%%",
            line_comment_prefix="%#",
            trim_blocks=True,
            autoescape=False,
        )

    # ── Main Entry Point ─────────────────────────────────────────────

    def generate_manuscript(
        self,
        project_id: str,
        template: str = "manuscript",
        authors: str = "",
    ) -> Optional[str]:
        """Generate a complete LaTeX manuscript bundle.

        Parameters
        ----------
        project_id
            ID of the ResearchProject entity.
        template
            Template name (without .tex.j2 extension).
        authors
            Author line for the manuscript.

        Returns
        -------
        str or None
            Path to generated manuscript.tex, or None on failure.
        """
        if self._env is None:
            return None

        from cognisom.library.models import ResearchProject, SimulationRun

        # Load project
        project = self._store.get_entity(project_id)
        if not isinstance(project, ResearchProject):
            log.error("Project %s not found", project_id)
            return None

        # Load all linked runs
        runs_data = []
        for run_id in project.run_ids:
            entity = self._store.get_entity(run_id)
            if isinstance(entity, SimulationRun) and entity.run_status == "completed":
                runs_data.append(self._build_run_context(entity))

        if not runs_data:
            log.error("No completed runs in project %s", project_id)
            return None

        # Build template context
        context = self._build_context(project, runs_data, authors)

        # Render template
        try:
            tmpl = self._env.get_template(f"{template}.tex.j2")
            rendered = tmpl.render(**context)
        except Exception as e:
            log.error("Template rendering failed: %s", e)
            return None

        # Write output files
        paper_dir = self._artifacts.paper_dir(project_id)

        # Main .tex file
        tex_path = paper_dir / "manuscript.tex"
        tex_path.write_text(rendered)

        # .bib file
        bib_path = paper_dir / "manuscript.bib"
        self._write_bibtex(project, bib_path)

        # Copy figures
        self._copy_figures(project, runs_data, paper_dir)

        log.info("Generated manuscript at %s", tex_path)
        return str(tex_path)

    # ── Context Building ─────────────────────────────────────────────

    def _build_run_context(self, run) -> Dict[str, Any]:
        """Build template context for a single run."""
        # Enabled modules list
        enabled = [k for k, v in run.modules_enabled.items() if v] if run.modules_enabled else []
        modules_list = ", ".join(enabled) if enabled else "all modules"

        # Metrics text
        m = run.final_metrics
        metrics_parts = []
        if "final_cancer" in m:
            metrics_parts.append(f"Final cancer cell count: {m['final_cancer']}")
        if "total_kills" in m:
            metrics_parts.append(f"Total immune-mediated kills: {m['total_kills']}")
        if "total_metastases" in m:
            metrics_parts.append(f"Metastatic events: {m['total_metastases']}")
        metrics_text = ". ".join(metrics_parts) + "." if metrics_parts else ""

        return {
            "run_id": run.entity_id,
            "name": _latex_escape(run.name),
            "short_name": _latex_escape(run.name[:20]),
            "duration_hours": run.duration_hours,
            "dt": run.dt,
            "random_seed": run.random_seed,
            "modules_list": modules_list,
            "accuracy_grade": run.accuracy_grade,
            "fidelity_grade": run.fidelity_grade,
            "metrics_text": metrics_text,
            "final_metrics": run.final_metrics,
            "hardware_info": run.hardware_info,
            "software_versions": run.software_versions,
            "git_commit": run.git_commit,
        }

    def _build_context(
        self, project, runs_data: List[Dict], authors: str
    ) -> Dict[str, Any]:
        """Build the full template context."""
        import time

        # Gather reproducibility info from first run
        first = runs_data[0] if runs_data else {}
        sw = first.get("software_versions", {})
        hw = first.get("hardware_info", {})

        # Package versions string
        pkg_parts = []
        for pkg in ["numpy", "scipy", "matplotlib"]:
            if pkg in sw:
                pkg_parts.append(f"{pkg} {sw[pkg]}")
        package_versions = ", ".join(pkg_parts) if pkg_parts else ""

        # Hardware description
        hw_parts = []
        if hw.get("gpu"):
            hw_parts.append(f"GPU: {hw['gpu']}")
        if hw.get("cpu"):
            hw_parts.append(f"CPU: {hw['cpu']}")
        if hw.get("cpu_count"):
            hw_parts.append(f"{hw['cpu_count']} cores")
        hardware_description = "; ".join(hw_parts) if hw_parts else "Not recorded"

        # Build metrics comparison table
        metrics_table = self._build_metrics_table(runs_data)

        # Build parameter table
        parameter_table = self._build_parameter_table(runs_data)

        # Collect figures
        figures = self._collect_figure_refs(project, runs_data)

        return {
            "title": _latex_escape(project.title or "Simulation Study"),
            "authors": _latex_escape(authors or ""),
            "date": time.strftime("%B %d, %Y"),
            "abstract": project.abstract or "Abstract not yet provided.",
            "introduction": project.introduction or "Introduction not yet provided.",
            "discussion": project.discussion or "Discussion not yet provided.",
            "runs": runs_data,
            "figures": figures,
            "metrics_table": metrics_table,
            "parameter_table": parameter_table,
            "software_version": sw.get("cognisom", "dev"),
            "git_commit": first.get("git_commit", "unknown"),
            "python_version": sw.get("python", ""),
            "package_versions": package_versions,
            "hardware_description": hardware_description,
        }

    def _build_metrics_table(self, runs_data: List[Dict]) -> List[Dict]:
        """Build metrics comparison table rows."""
        if not runs_data:
            return []

        metric_keys = [
            ("final_cancer", "Final Cancer Cells"),
            ("final_normal", "Final Normal Cells"),
            ("final_immune", "Final Immune Cells"),
            ("total_kills", "Total Immune Kills"),
            ("total_divisions", "Total Divisions"),
            ("total_deaths", "Total Deaths"),
            ("total_metastases", "Metastases"),
        ]

        rows = []
        for key, label in metric_keys:
            values = []
            has_data = False
            for rd in runs_data:
                val = rd.get("final_metrics", {}).get(key)
                if val is not None:
                    has_data = True
                    values.append(str(val))
                else:
                    values.append("--")
            if has_data:
                rows.append({"metric": label, "run_values": values})

        return rows

    def _build_parameter_table(self, runs_data: List[Dict]) -> List[Dict]:
        """Build parameter table from run configs."""
        rows = []
        for rd in runs_data:
            rows.append({
                "param": "Duration (hours)",
                "value": str(rd.get("duration_hours", "")),
                "run_name": rd.get("short_name", ""),
            })
            rows.append({
                "param": "Time step (hours)",
                "value": str(rd.get("dt", "")),
                "run_name": rd.get("short_name", ""),
            })
            rows.append({
                "param": "Random seed",
                "value": str(rd.get("random_seed", "")),
                "run_name": rd.get("short_name", ""),
            })
        return rows

    # ── Figures ──────────────────────────────────────────────────────

    def _collect_figure_refs(self, project, runs_data: List[Dict]) -> List[Dict]:
        """Collect figure references for the manuscript."""
        figures = []
        figure_labels = {
            "fig_cell_populations": "Cell population dynamics over simulation time.",
            "fig_immune_dynamics": "Immune cell dynamics and cancer cell killing.",
            "fig_oxygen_metabolism": "Oxygen and glucose concentration dynamics.",
            "fig_metastasis": "Metastatic events and lymphatic transport.",
            "fig_3d_tissue": "Three-dimensional tissue state at simulation end.",
            "fig_event_accumulation": "Cumulative simulation events over time.",
            "fig_benchmark_comparison": "Comparison of simulated values against published benchmarks.",
        }

        for rd in runs_data:
            run_id = rd["run_id"]
            figs_dir = self._artifacts.figures_dir(run_id)
            if figs_dir.exists():
                for fig_file in sorted(figs_dir.glob("*.png")):
                    stem = fig_file.stem
                    caption = figure_labels.get(stem, f"{stem} ({rd['name']})")
                    if len(runs_data) > 1:
                        caption += f" [{rd['short_name']}]"
                    figures.append({
                        "path": f"figures/{fig_file.name}",
                        "caption": _latex_escape(caption),
                        "label": f"{stem}_{run_id[:6]}",
                        "width": "0.85\\textwidth",
                        "source_path": str(fig_file),
                    })

        return figures

    def _copy_figures(self, project, runs_data: List[Dict], paper_dir: Path):
        """Copy run figures to the paper directory."""
        figs_dst = paper_dir / "figures"
        figs_dst.mkdir(parents=True, exist_ok=True)

        for rd in runs_data:
            run_id = rd["run_id"]
            figs_src = self._artifacts.figures_dir(run_id)
            if figs_src.exists():
                for fig_file in figs_src.glob("*.png"):
                    dst = figs_dst / fig_file.name
                    if not dst.exists():
                        shutil.copy2(fig_file, dst)

    # ── BibTeX ───────────────────────────────────────────────────────

    def _write_bibtex(self, project, bib_path: Path):
        """Write combined BibTeX file (built-in + project citations)."""
        # Start with built-in references
        builtin_bib = TEMPLATE_DIR / "bibtex_base.bib"
        content = ""
        if builtin_bib.exists():
            content = builtin_bib.read_text() + "\n\n"

        # Add project-specific citations
        if project.bibtex_entries:
            content += "% Project-specific citations\n\n"
            for key, entry in project.bibtex_entries.items():
                content += entry.strip() + "\n\n"

        bib_path.write_text(content)

    # ── PDF Compilation ──────────────────────────────────────────────

    def compile_pdf(self, tex_path: str) -> Optional[str]:
        """Compile LaTeX to PDF using pdflatex.

        Runs pdflatex twice (for references) and bibtex once.
        Returns path to PDF if successful, None otherwise.
        """
        tex_path = Path(tex_path)
        if not tex_path.exists():
            return None

        work_dir = tex_path.parent
        stem = tex_path.stem

        try:
            # Check if pdflatex is available
            subprocess.run(
                ["pdflatex", "--version"],
                capture_output=True, timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            log.info("pdflatex not available — skipping PDF compilation")
            return None

        try:
            # First pass
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", stem + ".tex"],
                cwd=work_dir, capture_output=True, timeout=60,
            )

            # BibTeX
            subprocess.run(
                ["bibtex", stem],
                cwd=work_dir, capture_output=True, timeout=30,
            )

            # Second pass
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", stem + ".tex"],
                cwd=work_dir, capture_output=True, timeout=60,
            )

            # Third pass (resolve all cross-references)
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", stem + ".tex"],
                cwd=work_dir, capture_output=True, timeout=60,
            )

            pdf_path = work_dir / f"{stem}.pdf"
            if pdf_path.exists():
                log.info("PDF compiled: %s", pdf_path)
                return str(pdf_path)
            else:
                log.warning("PDF compilation did not produce output")
                return None

        except subprocess.TimeoutExpired:
            log.warning("PDF compilation timed out")
            return None
        except Exception as e:
            log.warning("PDF compilation failed: %s", e)
            return None

    # ── Preview ──────────────────────────────────────────────────────

    def get_tex_preview(self, project_id: str) -> str:
        """Get the rendered LaTeX source for preview."""
        paper_dir = self._artifacts.paper_dir(project_id)
        tex_path = paper_dir / "manuscript.tex"
        if tex_path.exists():
            return tex_path.read_text()
        return ""

    def get_available_figures(self, project_id: str) -> List[Dict]:
        """List all figures available for the manuscript."""
        from cognisom.library.models import ResearchProject, SimulationRun

        project = self._store.get_entity(project_id)
        if not isinstance(project, ResearchProject):
            return []

        figures = []
        for run_id in project.run_ids:
            run = self._store.get_entity(run_id)
            if not isinstance(run, SimulationRun):
                continue

            figs_dir = self._artifacts.figures_dir(run_id)
            if figs_dir.exists():
                for fig_file in sorted(figs_dir.glob("*.png")):
                    figures.append({
                        "path": str(fig_file),
                        "name": fig_file.stem,
                        "run_id": run_id,
                        "run_name": run.name,
                        "size_bytes": fig_file.stat().st_size,
                    })

        return figures
