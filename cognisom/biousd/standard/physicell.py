"""Read a PhysiCell / MultiCellDS digital snapshot and write it as Bio-USD.

PhysiCell (Ghaffarizadeh et al. 2018, PLoS Comput Biol 14:e1005991) writes
each output as a MultiCellDS XML file plus MATLAB v4 matrices; the cells
matrix has one column per cell and one row per variable, and the XML's
``<labels>`` table says which row is which. This module reads exactly
that -- the label table and the ``<filename>`` it names -- and hands the
population to :func:`~.write.write_cells`. Nothing is assumed about row
order: a renamed or reordered variable table is read correctly or
refused, never silently misread.

Verified against a PhysiCell 1.14.1 snapshot (MultiCellDS version 2) in
the test data; the numeric cycle, phase and death codes below are taken
from ``core/PhysiCell_constants.cpp`` in the PhysiCell source, not
remembered.

What maps, and what does not
----------------------------
``bio:cell:id``       ``ID``
``position``          ``position`` (microns; Bio-USD is micrometres)
``scales``            diameter from ``total_volume`` (cubic microns)
``bio:cell:volume``   ``total_volume`` over the median living volume of
                      the same ``cell_type`` -- the vocabulary's
                      "relative to the type's resting volume", with the
                      median standing in for a resting volume PhysiCell
                      does not record
``bio:cell:alive``    ``dead == 0`` and the phase is not a death phase
``bio:cell:phase``    from ``current_phase`` through the table below;
                      composite phases (G0G1, G2M) and the Ki67 and
                      "live" models carry less than the vocabulary asks
                      and are mapped to their nearest token, counted in
                      the report as ``phase_coarse``
``bio:cell:type``     ``cell_type`` is an integer in the snapshot; the
                      names live in the PhysiCell settings file, not the
                      output, so the caller supplies ``types={0: "normal",
                      1: "cancer"}``. Unmapped ids are written ``normal``
                      and counted.
``bio:cell:age``      not asserted (0). PhysiCell records time in the
                      current phase, not time since division.
``bio:cell:ontologyTerm``  from the type token (``curie_for``), or the
                      caller's ``terms={1: "NCIT:C4863"}``.

The microenvironment matrices (oxygen and the like) are not read.
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# ── PhysiCell codes (core/PhysiCell_constants.cpp) ─────────────────

CYCLE_MODELS: Dict[int, str] = {
    0: "advanced_Ki67", 1: "basic_Ki67", 2: "flow_cytometry",
    3: "live_apoptotic", 4: "total_cells", 5: "live",
    6: "flow_cytometry_separated", 7: "cycling_quiescent",
    100: "apoptosis_death", 101: "necrosis_death", 102: "autophagy_death",
    9999: "custom",
}

#: PhysiCell phase code -> (Bio-USD phase token, exact?). ``exact`` is
#: False where PhysiCell's phase is coarser than the token.
PHASE_TOKENS: Dict[int, tuple] = {
    0: ("S", False),    # Ki67_positive_premitotic: cycling, phase unknown
    1: ("G1", False),   # Ki67_positive_postmitotic
    2: ("S", False),    # Ki67_positive
    3: ("G0", True),    # Ki67_negative: quiescent
    4: ("G1", False),   # G0G1_phase (composite)
    5: ("G0", True),
    6: ("G1", True), 7: ("G1", True), 8: ("G1", True), 9: ("G1", True),
    10: ("S", True),
    11: ("G2", False),  # G2M_phase (composite)
    12: ("G2", True), 13: ("M", True),
    14: ("G1", False),  # live: the model has no phases at all
    15: ("G1", True), 16: ("G1", True),   # G1pm, G1ps
    17: ("G1", False),  # cycling
    18: ("G0", True),   # quiescent
}
#: Death phases start at 100: apoptotic, necrotic_swelling, necrotic_lysed,
#: necrotic, debris. A cell in any of them is not alive.
FIRST_DEATH_PHASE = 100

DEFAULT_TYPE = "normal"


@dataclass
class Snapshot:
    """One MultiCellDS cell population, as read."""

    xml_path: str
    cells_path: str
    time_min: Optional[float]
    software: str
    labels: Dict[str, slice]          # variable name -> rows in the matrix
    units: Dict[str, str]
    matrix: np.ndarray                # (n_variables, n_cells)

    @property
    def n(self) -> int:
        return int(self.matrix.shape[1])

    def column(self, name: str) -> np.ndarray:
        """The variable's values, (n,) or (k, n) for a vector variable."""
        rows = self.labels[name]
        block = self.matrix[rows]
        return block[0] if block.shape[0] == 1 else block


@dataclass
class ConvertReport:
    path: str
    cells: int = 0
    alive: int = 0
    types_seen: Dict[int, int] = field(default_factory=dict)
    types_unmapped: List[int] = field(default_factory=list)
    phase_coarse: int = 0
    phase_unknown: int = 0
    primvars: List[str] = field(default_factory=list)
    time_min: Optional[float] = None

    def describe(self) -> str:
        unmapped = (f"; cell_type ids with no name: {sorted(self.types_unmapped)} "
                    f"(written as '{DEFAULT_TYPE}')" if self.types_unmapped else "")
        return (f"{self.cells} cells ({self.alive} alive) at t = {self.time_min} min -> {self.path}; "
                f"{self.phase_coarse} phases mapped to their nearest token, "
                f"{self.phase_unknown} unknown{unmapped}")


# ── reading ────────────────────────────────────────────────────────

def _find_simplified_data(root: ET.Element) -> ET.Element:
    for node in root.iter("simplified_data"):
        if node.find("labels") is not None:
            return node
    raise ValueError("no <simplified_data> with a <labels> table: not a PhysiCell "
                     "cell snapshot")


def read_snapshot(xml_path: str) -> Snapshot:
    """Read the cell population named by a MultiCellDS snapshot file."""
    from scipy.io import loadmat

    tree = ET.parse(xml_path)
    root = tree.getroot()
    if root.tag != "MultiCellDS":
        raise ValueError(f"{xml_path!r}: root element is <{root.tag}>, not <MultiCellDS>")
    software = (root.findtext("metadata/software/name") or "").strip()
    time_text = root.findtext("metadata/current_time")
    time_min = float(time_text) if time_text else None

    data = _find_simplified_data(root)
    labels: Dict[str, slice] = {}
    units: Dict[str, str] = {}
    for label in data.find("labels").findall("label"):
        name = (label.text or "").strip()
        index = int(label.get("index"))
        size = int(label.get("size", "1"))
        if name in labels:
            # PhysiCell 1.14 writes `elapsed_time_in_phase` twice (indices
            # 8 and 25); the first is the cycle's, the second the death
            # model's. Keep the first and name the second.
            name = f"{name}@{index}"
        labels[name] = slice(index, index + size)
        units[name] = label.get("units", "")
    filename = (data.findtext("filename") or "").strip()
    if not filename:
        raise ValueError("the <simplified_data> block names no <filename>")
    cells_path = os.path.join(os.path.dirname(os.path.abspath(xml_path)), filename)
    if not os.path.exists(cells_path):
        raise FileNotFoundError(f"{filename!r} named by {xml_path!r} is not beside it")

    mat = loadmat(cells_path)
    arrays = [v for k, v in mat.items() if not k.startswith("__")]
    if len(arrays) != 1:
        raise ValueError(f"{cells_path!r} holds {len(arrays)} matrices; expected one")
    matrix = np.asarray(arrays[0], dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{cells_path!r}: matrix is {matrix.ndim}-D")
    needed = max(s.stop for s in labels.values())
    if matrix.shape[0] < needed:
        raise ValueError(f"{cells_path!r} has {matrix.shape[0]} rows but the label table "
                         f"addresses {needed}")
    for required in ("ID", "position", "total_volume", "cell_type"):
        if required not in labels:
            raise ValueError(f"label table lacks {required!r}")
    return Snapshot(xml_path=xml_path, cells_path=cells_path, time_min=time_min,
                    software=software, labels=labels, units=units, matrix=matrix)


# ── converting ─────────────────────────────────────────────────────

def cells_from_snapshot(snap: Snapshot, *,
                        types: Optional[Dict[int, str]] = None,
                        terms: Optional[Dict[int, str]] = None,
                        report: Optional[ConvertReport] = None) -> List[Dict[str, Any]]:
    """Per-cell dicts in the shape :func:`~.write.write_cells` reads."""
    from .terms import curie_for
    from .vocabulary import CELL_PHASES, CELL_TYPES

    types = dict(types or {})
    terms = dict(terms or {})
    report = report or ConvertReport(path="")

    ids = snap.column("ID").astype(int)
    pos = np.asarray(snap.column("position"), dtype=float)       # (3, n)
    vol = np.asarray(snap.column("total_volume"), dtype=float)
    ctype = snap.column("cell_type").astype(int)
    dead = (snap.column("dead").astype(int) if "dead" in snap.labels
            else np.zeros(snap.n, dtype=int))
    phase = (snap.column("current_phase").astype(int) if "current_phase" in snap.labels
             else np.full(snap.n, -1))
    alive = (dead == 0) & (phase < FIRST_DEATH_PHASE)

    # Resting volume per type: the median living volume of that type.
    resting: Dict[int, float] = {}
    for t in np.unique(ctype):
        live_v = vol[(ctype == t) & alive]
        resting[int(t)] = float(np.median(live_v if live_v.size else vol[ctype == t]))

    cells: List[Dict[str, Any]] = []
    for i in range(snap.n):
        t = int(ctype[i])
        report.types_seen[t] = report.types_seen.get(t, 0) + 1
        token = types.get(t)
        if token is None or token not in CELL_TYPES:
            if t not in report.types_unmapped:
                report.types_unmapped.append(t)
            token = DEFAULT_TYPE
        p = int(phase[i])
        if p in PHASE_TOKENS:
            phase_token, exact = PHASE_TOKENS[p]
            if not exact:
                report.phase_coarse += 1
        elif p >= FIRST_DEATH_PHASE:
            phase_token, exact = "G0", True   # a dead cell has left the cycle
        else:
            phase_token = "G1"
            report.phase_unknown += 1
        assert phase_token in CELL_PHASES
        term = terms.get(t) or curie_for(None, cell_type=token) or ""
        diameter = float(np.cbrt(6.0 * vol[i] / np.pi))
        cells.append({
            "cell_id": int(ids[i]),
            "position": (float(pos[0, i]), float(pos[1, i]), float(pos[2, i])),
            "cell_type": token,
            "ontology_term": term,
            "phase": phase_token,
            "age": 0.0,
            "alive": bool(alive[i]),
            "volume": float(vol[i] / resting[t]) if resting[t] > 0 else 1.0,
            "scale_um": diameter,
        })
    report.cells = len(cells)
    report.alive = int(alive.sum())
    report.time_min = snap.time_min
    return cells


def convert(xml_path: str, out_path: str, *,
            types: Optional[Dict[int, str]] = None,
            terms: Optional[Dict[int, str]] = None,
            up_axis: str = "Y") -> ConvertReport:
    """PhysiCell snapshot in, conforming Bio-USD stage out."""
    from .write import write_cells

    snap = read_snapshot(xml_path)
    report = ConvertReport(path=out_path)
    cells = cells_from_snapshot(snap, types=types, terms=terms, report=report)
    written = write_cells(out_path, cells, up_axis=up_axis)
    report.primvars = list(written.primvars)
    return report


# ── command line ───────────────────────────────────────────────────

def _parse_map(items: Sequence[str], what: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for item in items or ():
        if "=" not in item:
            raise SystemExit(f"--{what} expects N=value, got {item!r}")
        k, v = item.split("=", 1)
        out[int(k)] = v.strip()
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="biousd-from-physicell",
        description="Write a PhysiCell / MultiCellDS cell snapshot as a Bio-USD stage.")
    parser.add_argument("xml", help="the snapshot's MultiCellDS XML file")
    parser.add_argument("out", help="output stage (.usda or .usdc)")
    parser.add_argument("--type", action="append", default=[], metavar="N=TOKEN",
                        help="name a PhysiCell cell_type id, e.g. --type 1=cancer")
    parser.add_argument("--term", action="append", default=[], metavar="N=CURIE",
                        help="ontology term for a cell_type id, e.g. --term 1=NCIT:C4863")
    parser.add_argument("--up-axis", default="Y", choices=("Y", "Z"))
    args = parser.parse_args(argv)
    try:
        report = convert(args.xml, args.out, types=_parse_map(args.type, "type"),
                         terms=_parse_map(args.term, "term"), up_axis=args.up_axis)
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(report.describe())
    return 0


if __name__ == "__main__":        # pragma: no cover
    raise SystemExit(main())
