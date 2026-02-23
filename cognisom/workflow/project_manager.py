"""Project Manager — group simulation runs into research projects.

Projects organize runs for comparison, parameter sweeps, and paper
generation.  A project links to a set of runs, designates a baseline,
and stores researcher-authored text sections for the manuscript.

Usage::

    mgr = ProjectManager(store)
    project = mgr.create_project(
        title="Enzalutamide Response Study",
        hypothesis="AR pathway inhibition reduces tumor growth",
    )
    mgr.add_run(project.entity_id, run_id)
    mgr.set_baseline(project.entity_id, run_id)
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from cognisom.library.models import (
    EntityType,
    RelationshipType,
    Relationship,
    ResearchProject,
    SimulationRun,
)
from cognisom.library.store import EntityStore

log = logging.getLogger(__name__)


class ProjectManager:
    """Manage research projects and their linked simulation runs."""

    def __init__(self, store: EntityStore):
        self._store = store

    # ── Create / Update ──────────────────────────────────────────────

    def create_project(
        self,
        title: str,
        hypothesis: str = "",
        methodology: str = "",
    ) -> ResearchProject:
        """Create a new research project."""
        project = ResearchProject(
            entity_id=str(uuid.uuid4()),
            name=title,
            entity_type=EntityType.RESEARCH_PROJECT,
            title=title,
            hypothesis=hypothesis,
            methodology=methodology,
            created_by="researcher",
        )
        self._store.add_entity(project)
        log.info("Created project %s: %s", project.entity_id, title)
        return project

    def update_project(self, project: ResearchProject) -> bool:
        """Save changes to a project."""
        return self._store.update_entity(project, changed_by="researcher")

    def get_project(self, project_id: str) -> Optional[ResearchProject]:
        """Get a project by ID."""
        entity = self._store.get_entity(project_id)
        if isinstance(entity, ResearchProject):
            return entity
        return None

    def list_projects(self, limit: int = 50) -> List[ResearchProject]:
        """List all research projects."""
        results, _ = self._store.search(
            query="",
            entity_type=EntityType.RESEARCH_PROJECT.value,
            limit=limit,
        )
        projects = [r for r in results if isinstance(r, ResearchProject)]
        projects.sort(key=lambda p: p.created_at, reverse=True)
        return projects

    def delete_project(self, project_id: str) -> bool:
        """Delete a project (does not delete linked runs)."""
        return self._store.hard_delete_entity(project_id)

    # ── Run Management ───────────────────────────────────────────────

    def add_run(self, project_id: str, run_id: str) -> bool:
        """Add a simulation run to the project."""
        project = self.get_project(project_id)
        if not project:
            return False

        if run_id not in project.run_ids:
            project.run_ids.append(run_id)
            self._store.update_entity(project, changed_by="researcher")

            # Update the run's project_id
            run_entity = self._store.get_entity(run_id)
            if isinstance(run_entity, SimulationRun):
                run_entity.project_id = project_id
                self._store.update_entity(run_entity, changed_by="researcher")

            # Create relationship
            self._store.add_relationship(Relationship(
                source_id=run_id,
                target_id=project_id,
                rel_type=RelationshipType.BELONGS_TO,
            ))
            return True
        return False

    def remove_run(self, project_id: str, run_id: str) -> bool:
        """Remove a simulation run from the project."""
        project = self.get_project(project_id)
        if not project:
            return False

        if run_id in project.run_ids:
            project.run_ids.remove(run_id)
            if project.baseline_run_id == run_id:
                project.baseline_run_id = ""
            self._store.update_entity(project, changed_by="researcher")
            return True
        return False

    def set_baseline(self, project_id: str, run_id: str) -> bool:
        """Set a run as the baseline/control for comparison."""
        project = self.get_project(project_id)
        if not project:
            return False

        if run_id in project.run_ids:
            project.baseline_run_id = run_id
            self._store.update_entity(project, changed_by="researcher")
            return True
        return False

    # ── Paper Content ────────────────────────────────────────────────

    def update_text(
        self,
        project_id: str,
        abstract: Optional[str] = None,
        introduction: Optional[str] = None,
        discussion: Optional[str] = None,
    ) -> bool:
        """Update the researcher-authored text sections."""
        project = self.get_project(project_id)
        if not project:
            return False

        if abstract is not None:
            project.abstract = abstract
        if introduction is not None:
            project.introduction = introduction
        if discussion is not None:
            project.discussion = discussion

        return self._store.update_entity(project, changed_by="researcher")

    def add_citation(self, project_id: str, key: str, bibtex: str) -> bool:
        """Add a BibTeX citation entry to the project."""
        project = self.get_project(project_id)
        if not project:
            return False

        project.bibtex_entries[key] = bibtex
        return self._store.update_entity(project, changed_by="researcher")

    def remove_citation(self, project_id: str, key: str) -> bool:
        """Remove a BibTeX citation entry."""
        project = self.get_project(project_id)
        if not project:
            return False

        if key in project.bibtex_entries:
            del project.bibtex_entries[key]
            return self._store.update_entity(project, changed_by="researcher")
        return False

    # ── Parameter Sweep ──────────────────────────────────────────────

    def create_sweep_runs(
        self,
        project_id: str,
        scenario_id: str,
        parameter_module: str,
        parameter_name: str,
        values: List[float],
        run_manager,
    ) -> List[SimulationRun]:
        """Create a set of runs varying a single parameter.

        Parameters
        ----------
        project_id
            Project to add runs to.
        scenario_id
            Base scenario to use.
        parameter_module
            Module containing the parameter (e.g., "cellular").
        parameter_name
            Parameter to vary (e.g., "division_time_cancer").
        values
            List of values to sweep.
        run_manager
            RunManager instance for creating runs.

        Returns
        -------
        List[SimulationRun]
            Created (pending) runs.
        """
        runs = []
        for val in values:
            overrides = {parameter_module: {parameter_name: val}}
            run = run_manager.create_run(
                scenario_id=scenario_id,
                overrides=overrides,
                run_name=f"{parameter_name}={val}",
                project_id=project_id,
            )
            self.add_run(project_id, run.entity_id)
            runs.append(run)

        log.info(
            "Created %d sweep runs for %s.%s in project %s",
            len(runs), parameter_module, parameter_name, project_id,
        )
        return runs

    def get_project_summary(self, project_id: str) -> Dict[str, Any]:
        """Get a summary of a project and its runs."""
        project = self.get_project(project_id)
        if not project:
            return {}

        runs = []
        for run_id in project.run_ids:
            entity = self._store.get_entity(run_id)
            if isinstance(entity, SimulationRun):
                runs.append({
                    "run_id": entity.entity_id,
                    "name": entity.name,
                    "run_status": entity.run_status,
                    "accuracy_grade": entity.accuracy_grade,
                    "fidelity_grade": entity.fidelity_grade,
                    "elapsed_seconds": entity.elapsed_seconds,
                    "final_metrics": entity.final_metrics,
                })

        return {
            "project_id": project.entity_id,
            "title": project.title,
            "hypothesis": project.hypothesis,
            "paper_status": project.paper_status,
            "n_runs": len(runs),
            "n_completed": sum(1 for r in runs if r["run_status"] == "completed"),
            "baseline_run_id": project.baseline_run_id,
            "runs": runs,
        }
