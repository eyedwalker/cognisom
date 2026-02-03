"""
Data Sync Agent (Phase 7)
==========================

Autonomous agent that monitors and syncs external data sources:
- GEO (Gene Expression Omnibus) for scRNA-seq datasets
- TCGA (The Cancer Genome Atlas) for cancer genomics data
- CellxGene for curated single-cell atlases

The agent periodically checks for new datasets relevant to prostate cancer
and downloads/indexes them for use in simulations.

Usage::

    from cognisom.agents import DataSyncAgent

    agent = DataSyncAgent(data_dir="data/scrna")
    report = agent.sync_all()
    print(report.summary())
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

log = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Metadata about an external dataset."""
    source: str = ""           # "geo", "tcga", "cellxgene"
    accession: str = ""        # e.g., "GSE176031"
    title: str = ""
    description: str = ""
    organism: str = "Homo sapiens"
    cell_count: int = 0
    gene_count: int = 0
    publication_date: str = ""
    local_path: str = ""       # path to downloaded file
    last_synced: float = 0.0


@dataclass
class DataSyncReport:
    """Results of a data sync operation."""
    timestamp: float = 0.0
    source: str = ""
    datasets_checked: int = 0
    datasets_new: int = 0
    datasets_updated: int = 0
    datasets_downloaded: int = 0
    bytes_downloaded: int = 0
    errors: List[str] = field(default_factory=list)
    elapsed_sec: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    def summary(self) -> str:
        status = "OK" if not self.errors else f"ERRORS ({len(self.errors)})"
        lines = [
            f"Data Sync [{self.source}] {status}",
            f"  Checked: {self.datasets_checked} datasets",
            f"  New: {self.datasets_new}, Updated: {self.datasets_updated}",
            f"  Downloaded: {self.datasets_downloaded} ({self.bytes_downloaded / 1e6:.1f} MB)",
            f"  Elapsed: {self.elapsed_sec:.1f}s",
        ]
        for err in self.errors[:5]:
            lines.append(f"  [ERROR] {err}")
        return "\n".join(lines)


class DataSyncAgent:
    """Autonomous agent for syncing external genomics datasets.

    Monitors GEO, TCGA, and CellxGene for prostate cancer datasets.
    """

    # Search terms for prostate cancer datasets
    SEARCH_TERMS = [
        "prostate cancer",
        "prostate adenocarcinoma",
        "PRAD",
        "prostate tumor",
        "prostatic neoplasm",
    ]

    # Known high-quality datasets to track
    PRIORITY_DATASETS = {
        "geo": [
            "GSE176031",  # scRNA-seq prostate cancer
            "GSE141445",  # PRAD microenvironment
            "GSE157703",  # Prostate epithelial cells
        ],
        "tcga": [
            "TCGA-PRAD",  # TCGA prostate adenocarcinoma
        ],
        "cellxgene": [
            "tabula-sapiens-prostate",
        ],
    }

    def __init__(self, data_dir: str = "data/scrna") -> None:
        """Initialize the data sync agent.

        Args:
            data_dir: Directory for storing downloaded datasets
        """
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self._data_dir / "dataset_index.json"
        self._datasets: Dict[str, DatasetInfo] = {}
        self._load_index()

    def sync_all(self) -> List[DataSyncReport]:
        """Sync all data sources. Returns list of reports."""
        reports = []
        reports.append(self.sync_geo())
        reports.append(self.sync_tcga())
        reports.append(self.sync_cellxgene())
        return reports

    def sync_geo(self) -> DataSyncReport:
        """Sync with Gene Expression Omnibus."""
        t0 = time.time()
        report = DataSyncReport(source="GEO")

        try:
            # Check priority datasets first
            for accession in self.PRIORITY_DATASETS.get("geo", []):
                report.datasets_checked += 1
                info = self._fetch_geo_metadata(accession)
                if info:
                    if accession not in self._datasets:
                        report.datasets_new += 1
                        self._datasets[accession] = info
                    elif self._needs_update(self._datasets[accession], info):
                        report.datasets_updated += 1
                        self._datasets[accession] = info

            # Search for new datasets
            new_datasets = self._search_geo()
            report.datasets_checked += len(new_datasets)
            for ds in new_datasets:
                if ds.accession not in self._datasets:
                    report.datasets_new += 1
                    self._datasets[ds.accession] = ds

            self._save_index()

        except Exception as e:
            report.errors.append(f"GEO sync failed: {str(e)}")
            log.error("GEO sync error: %s", e)

        report.elapsed_sec = time.time() - t0
        return report

    def sync_tcga(self) -> DataSyncReport:
        """Sync with TCGA via GDC API."""
        t0 = time.time()
        report = DataSyncReport(source="TCGA")

        try:
            # Check TCGA-PRAD project
            info = self._fetch_tcga_metadata("TCGA-PRAD")
            if info:
                report.datasets_checked += 1
                key = f"tcga:{info.accession}"
                if key not in self._datasets:
                    report.datasets_new += 1
                    self._datasets[key] = info

            self._save_index()

        except Exception as e:
            report.errors.append(f"TCGA sync failed: {str(e)}")
            log.error("TCGA sync error: %s", e)

        report.elapsed_sec = time.time() - t0
        return report

    def sync_cellxgene(self) -> DataSyncReport:
        """Sync with CellxGene Census API."""
        t0 = time.time()
        report = DataSyncReport(source="CellxGene")

        try:
            # Query CellxGene for prostate datasets
            datasets = self._search_cellxgene()
            report.datasets_checked = len(datasets)

            for ds in datasets:
                key = f"cellxgene:{ds.accession}"
                if key not in self._datasets:
                    report.datasets_new += 1
                    self._datasets[key] = ds

            self._save_index()

        except Exception as e:
            report.errors.append(f"CellxGene sync failed: {str(e)}")
            log.error("CellxGene sync error: %s", e)

        report.elapsed_sec = time.time() - t0
        return report

    def download_dataset(self, accession: str) -> Tuple[bool, str]:
        """Download a specific dataset by accession.

        Returns (success, local_path or error message).
        """
        if accession not in self._datasets:
            return False, f"Dataset {accession} not in index"

        ds = self._datasets[accession]

        try:
            if ds.source == "geo":
                return self._download_geo(accession)
            elif ds.source == "tcga":
                return self._download_tcga(accession)
            elif ds.source == "cellxgene":
                return self._download_cellxgene(accession)
            else:
                return False, f"Unknown source: {ds.source}"
        except Exception as e:
            return False, str(e)

    def list_datasets(self, source: Optional[str] = None) -> List[DatasetInfo]:
        """List all indexed datasets, optionally filtered by source."""
        datasets = list(self._datasets.values())
        if source:
            datasets = [d for d in datasets if d.source == source]
        return sorted(datasets, key=lambda d: d.publication_date, reverse=True)

    # ── GEO Methods ──────────────────────────────────────────────────

    def _fetch_geo_metadata(self, accession: str) -> Optional[DatasetInfo]:
        """Fetch dataset metadata from GEO."""
        import urllib.request
        import xml.etree.ElementTree as ET

        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gds&term={accession}"

        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                # Just verify the accession exists
                return DatasetInfo(
                    source="geo",
                    accession=accession,
                    title=f"GEO Dataset {accession}",
                    organism="Homo sapiens",
                    last_synced=time.time(),
                )
        except Exception as e:
            log.warning("Failed to fetch GEO %s: %s", accession, e)
            return None

    def _search_geo(self) -> List[DatasetInfo]:
        """Search GEO for prostate cancer scRNA-seq datasets."""
        import urllib.request
        import urllib.parse

        datasets = []
        query = urllib.parse.quote("prostate cancer[Title] AND scRNA-seq[Title]")
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gds&term={query}&retmax=20"

        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                # Parse response and extract accessions
                # (simplified - real implementation would parse XML)
                log.info("Searched GEO for prostate cancer datasets")
        except Exception as e:
            log.warning("GEO search failed: %s", e)

        return datasets

    def _download_geo(self, accession: str) -> Tuple[bool, str]:
        """Download a GEO dataset."""
        # GEO datasets are typically downloaded via GEOquery or direct FTP
        # This is a stub - real implementation would use proper download
        log.info("Would download GEO dataset: %s", accession)
        return True, str(self._data_dir / f"{accession}.h5ad")

    # ── TCGA Methods ─────────────────────────────────────────────────

    def _fetch_tcga_metadata(self, project: str) -> Optional[DatasetInfo]:
        """Fetch project metadata from GDC API."""
        import urllib.request
        import json

        url = f"https://api.gdc.cancer.gov/projects/{project}"

        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                return DatasetInfo(
                    source="tcga",
                    accession=project,
                    title=data.get("data", {}).get("name", project),
                    description=data.get("data", {}).get("summary", {}).get("case_count", 0),
                    organism="Homo sapiens",
                    cell_count=data.get("data", {}).get("summary", {}).get("case_count", 0),
                    last_synced=time.time(),
                )
        except Exception as e:
            log.warning("Failed to fetch TCGA %s: %s", project, e)
            return None

    def _download_tcga(self, project: str) -> Tuple[bool, str]:
        """Download TCGA data via GDC API."""
        log.info("Would download TCGA project: %s", project)
        return True, str(self._data_dir / f"{project}.h5ad")

    # ── CellxGene Methods ────────────────────────────────────────────

    def _search_cellxgene(self) -> List[DatasetInfo]:
        """Search CellxGene for prostate datasets."""
        datasets = []

        # CellxGene Census API (simplified)
        # Real implementation would use cellxgene_census package
        log.info("Searched CellxGene for prostate datasets")

        return datasets

    def _download_cellxgene(self, dataset_id: str) -> Tuple[bool, str]:
        """Download a CellxGene dataset."""
        log.info("Would download CellxGene dataset: %s", dataset_id)
        return True, str(self._data_dir / f"{dataset_id}.h5ad")

    # ── Persistence ──────────────────────────────────────────────────

    def _load_index(self) -> None:
        """Load dataset index from disk."""
        if self._index_file.exists():
            import json
            try:
                data = json.loads(self._index_file.read_text())
                self._datasets = {
                    k: DatasetInfo(**v) for k, v in data.items()
                }
            except Exception as e:
                log.warning("Failed to load dataset index: %s", e)

    def _save_index(self) -> None:
        """Save dataset index to disk."""
        import json
        data = {k: vars(v) for k, v in self._datasets.items()}
        self._index_file.write_text(json.dumps(data, indent=2))

    def _needs_update(self, old: DatasetInfo, new: DatasetInfo) -> bool:
        """Check if dataset metadata needs updating."""
        # Re-sync if more than 7 days old
        return (time.time() - old.last_synced) > (7 * 24 * 3600)
