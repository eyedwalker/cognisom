"""Researcher workflow — run management, artifact storage, and paper generation.

Provides the complete researcher pipeline:
    Scenario → Run → Persist → Evaluate → Compare → Publish

Classes:
    ArtifactStore      — Filesystem storage for simulation artifacts
    RunManager         — Create, execute, and persist simulation runs
    ProjectManager     — Group runs into research projects
    FigureGenerator    — Publication-quality matplotlib figures
    PaperGenerator     — Jinja2 LaTeX manuscript generation
"""

from cognisom.workflow.artifact_store import ArtifactStore
from cognisom.workflow.run_manager import RunManager
from cognisom.workflow.project_manager import ProjectManager
from cognisom.workflow.figure_generator import FigureGenerator
from cognisom.workflow.paper_generator import PaperGenerator

__all__ = [
    "ArtifactStore",
    "RunManager",
    "ProjectManager",
    "FigureGenerator",
    "PaperGenerator",
]
