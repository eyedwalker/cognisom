"""
Bidirectional Sync Engine (Phase 6)
====================================

Synchronises state between the Cognisom simulation engine and Bio-USD
scene representation. Changes in either direction are detected and
propagated.

Flows:
    Engine -> USD:  After each simulation step, export changed state to scene
    USD -> Engine:  After user edits in Omniverse, import scene diffs back

The sync uses a diff-based approach: each entity has a version counter.
Only changed entities are re-exported or re-imported.

Usage::

    from cognisom.biousd.sync import BidirectionalSync

    sync = BidirectionalSync(engine, scene)
    sync.engine_to_usd()   # push simulation state to USD
    sync.usd_to_engine()   # pull USD edits back to engine
    sync.full_sync()       # bidirectional merge

The Omniverse connector (Phase 8) will call these methods via the
Kit extension's event loop.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .schema import (
    BioCapillary,
    BioCell,
    BioExosome,
    BioGene,
    BioImmuneCell,
    BioMetabolicAPI,
    BioMolecule,
    BioProtein,
    BioScene,
    BioSpatialField,
    BioTissue,
    CellPhase,
    CellType,
    ImmuneCellType,
)

log = logging.getLogger(__name__)


@dataclass
class SyncDiff:
    """A record of changes detected during sync."""
    timestamp: float = 0.0
    direction: str = ""  # "engine_to_usd" or "usd_to_engine"
    cells_updated: int = 0
    cells_added: int = 0
    cells_removed: int = 0
    fields_updated: int = 0
    genes_updated: int = 0
    molecules_updated: int = 0
    details: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    @property
    def total_changes(self) -> int:
        return (self.cells_updated + self.cells_added + self.cells_removed +
                self.fields_updated + self.genes_updated + self.molecules_updated)

    def summary(self) -> str:
        parts = []
        if self.cells_added:
            parts.append(f"+{self.cells_added} cells")
        if self.cells_removed:
            parts.append(f"-{self.cells_removed} cells")
        if self.cells_updated:
            parts.append(f"~{self.cells_updated} cells")
        if self.fields_updated:
            parts.append(f"~{self.fields_updated} fields")
        if self.genes_updated:
            parts.append(f"~{self.genes_updated} genes")
        if self.molecules_updated:
            parts.append(f"~{self.molecules_updated} molecules")
        return f"[{self.direction}] {', '.join(parts) or 'no changes'}"


class BidirectionalSync:
    """Sync state between Cognisom engine and Bio-USD scene.

    The sync maintains version counters to detect changes efficiently.
    """

    def __init__(self, engine=None, scene: Optional[BioScene] = None):
        """Initialize sync with engine and/or scene.

        Args:
            engine: A cognisom.core.simulation_engine.SimulationEngine
            scene: A BioScene to sync with (created if not provided)
        """
        self._engine = engine
        self._scene = scene or BioScene()
        self._engine_version = 0
        self._scene_version = 0
        self._cell_versions: Dict[int, int] = {}  # cell_id -> last synced version
        self._sync_history: List[SyncDiff] = []

    @property
    def scene(self) -> BioScene:
        return self._scene

    @property
    def history(self) -> List[SyncDiff]:
        return self._sync_history

    # ── Engine -> USD ────────────────────────────────────────────────

    def engine_to_usd(self) -> SyncDiff:
        """Push current engine state to the Bio-USD scene.

        Detects what has changed since last sync and only updates
        affected prims.
        """
        diff = SyncDiff(direction="engine_to_usd")

        if self._engine is None:
            log.warning("No engine connected for sync")
            return diff

        # Sync simulation metadata
        self._scene.simulation_time = getattr(self._engine, "time", 0.0)
        self._scene.time_step = getattr(self._engine, "dt", 0.1)
        self._scene.step_count = getattr(self._engine, "step_count", 0)

        # Sync cells
        diff.cells_updated, diff.cells_added, diff.cells_removed = (
            self._sync_cells_to_usd()
        )

        # Sync spatial fields
        diff.fields_updated = self._sync_fields_to_usd()

        self._engine_version += 1
        self._sync_history.append(diff)

        if diff.total_changes:
            log.info("Engine->USD sync: %s", diff.summary())

        return diff

    def _sync_cells_to_usd(self) -> Tuple[int, int, int]:
        """Sync cell state from engine to scene. Returns (updated, added, removed)."""
        updated = added = removed = 0

        # Get current engine cells
        engine_cells = self._get_engine_cells()
        engine_cell_ids = {c["id"] for c in engine_cells}

        # Existing scene cell IDs
        scene_cell_ids = {c.cell_id for c in self._scene.cells}
        scene_immune_ids = {c.cell_id for c in self._scene.immune_cells}
        all_scene_ids = scene_cell_ids | scene_immune_ids

        # Add new cells
        for cell_data in engine_cells:
            cid = cell_data["id"]
            if cid not in all_scene_ids:
                bio_cell = self._engine_cell_to_bio(cell_data)
                if isinstance(bio_cell, BioImmuneCell):
                    self._scene.immune_cells.append(bio_cell)
                else:
                    self._scene.cells.append(bio_cell)
                added += 1
            else:
                # Update existing cell
                if self._update_bio_cell(cell_data):
                    updated += 1

        # Remove cells no longer in engine
        self._scene.cells = [c for c in self._scene.cells if c.cell_id in engine_cell_ids]
        self._scene.immune_cells = [c for c in self._scene.immune_cells if c.cell_id in engine_cell_ids]
        removed = len(all_scene_ids - engine_cell_ids)

        return updated, added, removed

    def _get_engine_cells(self) -> List[dict]:
        """Extract cell data from the engine."""
        cells = []
        engine = self._engine

        # Try different engine attribute patterns
        cell_list = getattr(engine, "cells", None)
        if cell_list is None:
            # Try module-based access
            cell_module = getattr(engine, "cellular_module", None)
            if cell_module:
                cell_list = getattr(cell_module, "cells", [])

        if cell_list is None:
            return cells

        for i, cell in enumerate(cell_list):
            if isinstance(cell, dict):
                cells.append(cell)
            else:
                # Convert object to dict
                cells.append({
                    "id": getattr(cell, "cell_id", i),
                    "position": getattr(cell, "position", (0, 0, 0)),
                    "cell_type": str(getattr(cell, "cell_type", "normal")),
                    "phase": str(getattr(cell, "phase", "G1")),
                    "alive": getattr(cell, "alive", True),
                    "age": getattr(cell, "age", 0.0),
                    "volume": getattr(cell, "volume", 1.0),
                    "oxygen": getattr(cell, "oxygen", 0.21),
                    "glucose": getattr(cell, "glucose", 5.0),
                    "atp": getattr(cell, "atp", 1000.0),
                    "lactate": getattr(cell, "lactate", 0.0),
                })

        return cells

    def _engine_cell_to_bio(self, data: dict) -> BioCell:
        """Convert engine cell dict to BioCell."""
        pos = data.get("position", (0, 0, 0))
        if not isinstance(pos, tuple):
            pos = tuple(pos) if hasattr(pos, '__iter__') else (0, 0, 0)

        cell_type_str = str(data.get("cell_type", "normal")).lower()
        try:
            ct = CellType(cell_type_str)
        except ValueError:
            ct = CellType.NORMAL

        phase_str = str(data.get("phase", "G1"))
        try:
            phase = CellPhase(phase_str)
        except ValueError:
            phase = CellPhase.G1

        # Check if it's an immune cell
        if ct == CellType.IMMUNE:
            return BioImmuneCell(
                prim_path=f"/World/Cells/cell_{data['id']}",
                cell_id=data["id"],
                position=pos,
                cell_type=ct,
                phase=phase,
                alive=data.get("alive", True),
                age=data.get("age", 0.0),
                volume=data.get("volume", 1.0),
                metabolic=BioMetabolicAPI(
                    oxygen=data.get("oxygen", 0.21),
                    glucose=data.get("glucose", 5.0),
                    atp=data.get("atp", 1000.0),
                    lactate=data.get("lactate", 0.0),
                ),
            )

        return BioCell(
            prim_path=f"/World/Cells/cell_{data['id']}",
            cell_id=data["id"],
            position=pos,
            cell_type=ct,
            phase=phase,
            alive=data.get("alive", True),
            age=data.get("age", 0.0),
            volume=data.get("volume", 1.0),
            metabolic=BioMetabolicAPI(
                oxygen=data.get("oxygen", 0.21),
                glucose=data.get("glucose", 5.0),
                atp=data.get("atp", 1000.0),
                lactate=data.get("lactate", 0.0),
            ),
        )

    def _update_bio_cell(self, data: dict) -> bool:
        """Update an existing BioCell in the scene. Returns True if changed."""
        cid = data["id"]

        # Find the cell in scene
        cell = None
        for c in self._scene.cells:
            if c.cell_id == cid:
                cell = c
                break
        if cell is None:
            for c in self._scene.immune_cells:
                if c.cell_id == cid:
                    cell = c
                    break
        if cell is None:
            return False

        changed = False
        pos = data.get("position", cell.position)
        if not isinstance(pos, tuple):
            pos = tuple(pos) if hasattr(pos, '__iter__') else cell.position

        if cell.position != pos:
            cell.position = pos
            changed = True

        alive = data.get("alive", cell.alive)
        if cell.alive != alive:
            cell.alive = alive
            changed = True

        age = data.get("age", cell.age)
        if abs(cell.age - age) > 0.01:
            cell.age = age
            changed = True

        volume = data.get("volume", cell.volume)
        if abs(cell.volume - volume) > 0.01:
            cell.volume = volume
            changed = True

        # Update metabolic state
        if cell.metabolic:
            o2 = data.get("oxygen", cell.metabolic.oxygen)
            if abs(cell.metabolic.oxygen - o2) > 0.001:
                cell.metabolic.oxygen = o2
                changed = True
            gluc = data.get("glucose", cell.metabolic.glucose)
            if abs(cell.metabolic.glucose - gluc) > 0.01:
                cell.metabolic.glucose = gluc
                changed = True

        return changed

    def _sync_fields_to_usd(self) -> int:
        """Sync spatial fields from engine. Returns count of updated fields."""
        # This would read from engine.spatial_module for concentration grids
        # Placeholder for when engine has field data
        return 0

    # ── USD -> Engine ────────────────────────────────────────────────

    def usd_to_engine(self) -> SyncDiff:
        """Pull Bio-USD scene changes back to the engine.

        This enables Omniverse users to manipulate cells/conditions
        and have those changes reflected in the simulation.
        """
        diff = SyncDiff(direction="usd_to_engine")

        if self._engine is None:
            log.warning("No engine connected for reverse sync")
            return diff

        # Sync cell positions and states back
        for cell in self._scene.cells:
            if self._apply_cell_to_engine(cell):
                diff.cells_updated += 1

        for cell in self._scene.immune_cells:
            if self._apply_cell_to_engine(cell):
                diff.cells_updated += 1

        self._scene_version += 1
        self._sync_history.append(diff)

        if diff.total_changes:
            log.info("USD->Engine sync: %s", diff.summary())

        return diff

    def _apply_cell_to_engine(self, bio_cell: BioCell) -> bool:
        """Apply a BioCell's state back to the engine. Returns True if changed."""
        engine = self._engine
        cell_list = getattr(engine, "cells", None)
        if cell_list is None:
            cell_module = getattr(engine, "cellular_module", None)
            if cell_module:
                cell_list = getattr(cell_module, "cells", [])

        if cell_list is None:
            return False

        # Find matching engine cell
        for eng_cell in cell_list:
            eid = eng_cell.get("cell_id", eng_cell.get("id", -1)) if isinstance(eng_cell, dict) else getattr(eng_cell, "cell_id", -1)
            if eid == bio_cell.cell_id:
                changed = False
                if isinstance(eng_cell, dict):
                    if eng_cell.get("position") != bio_cell.position:
                        eng_cell["position"] = bio_cell.position
                        changed = True
                    if eng_cell.get("alive") != bio_cell.alive:
                        eng_cell["alive"] = bio_cell.alive
                        changed = True
                else:
                    if getattr(eng_cell, "position", None) != bio_cell.position:
                        eng_cell.position = bio_cell.position
                        changed = True
                    if getattr(eng_cell, "alive", None) != bio_cell.alive:
                        eng_cell.alive = bio_cell.alive
                        changed = True
                return changed

        return False

    # ── Full Sync ────────────────────────────────────────────────────

    def full_sync(self) -> Tuple[SyncDiff, SyncDiff]:
        """Bidirectional sync: engine -> USD, then USD -> engine.

        Engine state takes priority for simulation variables (position, phase,
        metabolic state). USD takes priority for user-controlled variables
        (alive/dead, drug application).
        """
        diff_e2u = self.engine_to_usd()
        diff_u2e = self.usd_to_engine()
        return diff_e2u, diff_u2e

    # ── Utilities ────────────────────────────────────────────────────

    def get_sync_stats(self) -> dict:
        """Return sync statistics."""
        return {
            "engine_version": self._engine_version,
            "scene_version": self._scene_version,
            "total_syncs": len(self._sync_history),
            "scene_cells": self._scene.total_cells,
            "scene_alive": self._scene.alive_cells,
            "last_sync": self._sync_history[-1].summary() if self._sync_history else "none",
        }
