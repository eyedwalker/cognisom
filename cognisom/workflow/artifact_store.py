"""Filesystem storage for simulation run artifacts.

Manages reading and writing large artifacts that don't belong in the
database: time-series CSVs, compressed cell snapshots, event logs,
evaluation reports, and generated figures.

Directory layout::

    <base_dir>/
        runs/<run_id>/
            timeseries.csv
            cell_snapshots.npz
            events.jsonl
            eval_report.json
            figures/
                fig_cell_populations.png
                ...
        papers/<project_id>/
            manuscript.tex
            manuscript.bib
            figures/
            tables/
            bundle.zip
"""

from __future__ import annotations

import csv
import io
import json
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)


class ArtifactStore:
    """Filesystem read/write for simulation artifacts."""

    def __init__(self, base_dir: str = "/app/data"):
        self._base = Path(base_dir)
        self._runs_dir = self._base / "runs"
        self._papers_dir = self._base / "papers"

    # ── Directory helpers ────────────────────────────────────────────

    def run_dir(self, run_id: str) -> Path:
        """Get (and create) the directory for a run's artifacts."""
        d = self._runs_dir / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def figures_dir(self, run_id: str) -> Path:
        """Get (and create) the figures subdirectory for a run."""
        d = self.run_dir(run_id) / "figures"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def paper_dir(self, project_id: str) -> Path:
        """Get (and create) the directory for a paper's artifacts."""
        d = self._papers_dir / project_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def paper_figures_dir(self, project_id: str) -> Path:
        d = self.paper_dir(project_id) / "figures"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def paper_tables_dir(self, project_id: str) -> Path:
        d = self.paper_dir(project_id) / "tables"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── Time Series ──────────────────────────────────────────────────

    def save_time_series(
        self, run_id: str, data: Dict[str, list], filename: str = "timeseries.csv"
    ) -> str:
        """Save time-series data as CSV.

        Parameters
        ----------
        data : dict
            Column name -> list of values. All lists must be same length.

        Returns
        -------
        str
            Relative path of saved file.
        """
        path = self.run_dir(run_id) / filename
        if not data:
            path.write_text("")
            return filename

        columns = list(data.keys())
        n_rows = len(next(iter(data.values())))

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for i in range(n_rows):
                writer.writerow([data[col][i] if i < len(data[col]) else "" for col in columns])

        log.info("Saved time series (%d rows, %d cols) to %s", n_rows, len(columns), path)
        return filename

    def load_time_series(self, run_id: str, filename: str = "timeseries.csv") -> Dict[str, list]:
        """Load time-series CSV back into a dict of column -> values."""
        path = self.run_dir(run_id) / filename
        if not path.exists():
            return {}

        result: Dict[str, list] = {}
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for col in reader.fieldnames or []:
                result[col] = []
            for row in reader:
                for col in result:
                    val = row.get(col, "")
                    try:
                        result[col].append(float(val))
                    except (ValueError, TypeError):
                        result[col].append(val)
        return result

    # ── Cell Snapshots ───────────────────────────────────────────────

    def save_cell_snapshots(
        self,
        run_id: str,
        snapshots: List[Dict[str, Any]],
        filename: str = "cell_snapshots.npz",
    ) -> str:
        """Save cell snapshot list as compressed numpy archive.

        Each snapshot is a dict with numpy arrays (positions, radii, etc.)
        and scalar metadata (time, step).  Arrays are stored with keys like
        ``snap_0_positions``, ``snap_0_radii``, etc.  Metadata is stored
        in a JSON sidecar.
        """
        path = self.run_dir(run_id) / filename
        meta_path = path.with_suffix(".meta.json")

        arrays = {}
        metadata = []
        for i, snap in enumerate(snapshots):
            meta_entry = {}
            for key, value in snap.items():
                if isinstance(value, np.ndarray):
                    arrays[f"snap_{i}_{key}"] = value
                else:
                    meta_entry[key] = value
            metadata.append(meta_entry)

        np.savez_compressed(path, **arrays)
        meta_path.write_text(json.dumps(metadata, default=str))

        log.info("Saved %d cell snapshots to %s", len(snapshots), path)
        return filename

    def load_cell_snapshots(
        self, run_id: str, filename: str = "cell_snapshots.npz"
    ) -> List[Dict[str, Any]]:
        """Load cell snapshots from compressed numpy archive."""
        path = self.run_dir(run_id) / filename
        meta_path = path.with_suffix(".meta.json")

        if not path.exists():
            return []

        data = np.load(path, allow_pickle=False)
        metadata = []
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())

        # Reconstruct snapshot list
        # Determine how many snapshots
        snap_indices = set()
        for key in data.files:
            parts = key.split("_", 2)
            if len(parts) >= 2 and parts[0] == "snap":
                snap_indices.add(int(parts[1]))

        snapshots = []
        for i in sorted(snap_indices):
            snap: Dict[str, Any] = {}
            # Restore metadata
            if i < len(metadata):
                snap.update(metadata[i])
            # Restore arrays
            prefix = f"snap_{i}_"
            for key in data.files:
                if key.startswith(prefix):
                    field_name = key[len(prefix):]
                    snap[field_name] = data[key]
            snapshots.append(snap)

        return snapshots

    # ── Events ───────────────────────────────────────────────────────

    def save_events(
        self, run_id: str, events: List[dict], filename: str = "events.jsonl"
    ) -> str:
        """Save event log as newline-delimited JSON."""
        path = self.run_dir(run_id) / filename
        with open(path, "w") as f:
            for event in events:
                f.write(json.dumps(event, default=str) + "\n")
        log.info("Saved %d events to %s", len(events), path)
        return filename

    def load_events(self, run_id: str, filename: str = "events.jsonl") -> List[dict]:
        """Load event log from JSONL file."""
        path = self.run_dir(run_id) / filename
        if not path.exists():
            return []
        events = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events

    # ── Evaluation Reports ───────────────────────────────────────────

    def save_eval_report(
        self, run_id: str, report: dict, filename: str = "eval_report.json"
    ) -> str:
        """Save evaluation report as JSON."""
        path = self.run_dir(run_id) / filename
        path.write_text(json.dumps(report, indent=2, default=str))
        log.info("Saved eval report to %s", path)
        return filename

    def load_eval_report(self, run_id: str, filename: str = "eval_report.json") -> dict:
        """Load evaluation report from JSON."""
        path = self.run_dir(run_id) / filename
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    # ── Figures ──────────────────────────────────────────────────────

    def save_figure(self, run_id: str, fig, name: str, dpi: int = 150) -> str:
        """Save a matplotlib figure to the run's figures directory.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
        name : str
            Filename without extension (e.g., ``fig_cell_populations``).
        dpi : int
            Resolution for rasterized output.

        Returns
        -------
        str
            Relative path from run dir (e.g., ``figures/fig_cell_populations.png``).
        """
        figs_dir = self.figures_dir(run_id)
        png_path = figs_dir / f"{name}.png"
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        import matplotlib.pyplot as plt
        plt.close(fig)
        return f"figures/{name}.png"

    def save_paper_figure(self, project_id: str, fig, name: str) -> str:
        """Save a publication-quality figure as PDF for LaTeX inclusion."""
        figs_dir = self.paper_figures_dir(project_id)
        pdf_path = figs_dir / f"{name}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        import matplotlib.pyplot as plt
        plt.close(fig)
        return f"figures/{name}.pdf"

    # ── Listing & Cleanup ────────────────────────────────────────────

    def list_runs(self) -> List[str]:
        """List all run IDs that have artifact directories."""
        if not self._runs_dir.exists():
            return []
        return sorted(d.name for d in self._runs_dir.iterdir() if d.is_dir())

    def get_artifact_sizes(self, run_id: str) -> Dict[str, int]:
        """Get file sizes for all artifacts in a run."""
        d = self._runs_dir / run_id
        if not d.exists():
            return {}
        sizes = {}
        for f in d.rglob("*"):
            if f.is_file():
                rel = str(f.relative_to(d))
                sizes[rel] = f.stat().st_size
        return sizes

    def cleanup_run(self, run_id: str) -> bool:
        """Remove all artifacts for a run."""
        d = self._runs_dir / run_id
        if d.exists():
            shutil.rmtree(d)
            log.info("Cleaned up artifacts for run %s", run_id)
            return True
        return False

    def cleanup_paper(self, project_id: str) -> bool:
        """Remove all paper artifacts for a project."""
        d = self._papers_dir / project_id
        if d.exists():
            shutil.rmtree(d)
            return True
        return False

    def create_zip_bundle(self, project_id: str) -> Optional[str]:
        """Create a ZIP archive of all paper artifacts."""
        paper_d = self._papers_dir / project_id
        if not paper_d.exists():
            return None

        zip_path = paper_d / "bundle.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in paper_d.rglob("*"):
                if f.is_file() and f.name != "bundle.zip":
                    zf.write(f, f.relative_to(paper_d))

        log.info("Created paper bundle: %s", zip_path)
        return str(zip_path)
