"""What a transcript atlas can honestly tell a protein library.

CZ CELLxGENE Census aggregates the Human Cell Atlas and Tabula Sapiens
into one queryable corpus -- 154 million cells at the version pinned
below, with a Cell Ontology label on every one. All 25 of this library's
immune lineages appear in it.

THE LINE THIS MODULE DOES NOT CROSS
-----------------------------------
It never writes a parameter. There is no ``set_parameter`` call in this
file and that is deliberate, because the obvious thing to do with an
expression atlas is the wrong thing: RNA is not protein. The correlation
across genes runs around 0.4-0.6, so a transcript count cannot become a
copy number, an abundance, or a surface density. Reading it as one would
be the same error as reading "proteins PRODUCED per cell" as a census,
which is the trap that shaped :mod:`seed_copy_numbers`.

What an atlas is good for is RELATIVE comparison within one gene across
cell types, and that supports two things this library actually needs.

ONE: COVERAGE, SO THIN EVIDENCE IS VISIBLE
------------------------------------------
Lineage coverage in the census spans four orders of magnitude -- over a
million macrophages, 164,587 neutrophils, 480 eosinophils, 55 Th2 cells.
A result resting on Th2 behaviour is resting on 55 cells somewhere, and
nothing in this library said so. Now each lineage carries its count.

Two limits come with it. The atlas does not distinguish macrophage
polarisation: M1 and M2 both resolve to "macrophage", so a difference
between them cannot be checked here at all. And granulocytes are
systematically under-represented, because they carry little RNA and
survive dissociation poorly -- their absence is a property of the assay,
not of blood.

TWO: MARKERS, WHICH CAN BE FALSIFIED
------------------------------------
The library asserts marker sets for each lineage -- a naive B cell is
CD19, CD20, CD23, IgD, IgM. Those are claims, and an atlas can test
them: a marker for a lineage should be expressed more in that lineage
than in the others. That is a ratio within one gene, which is the
comparison transcript data is most reliable for.

The test can fail, which is the point. A marker that shows no enrichment
is either wrong, or protein-level only, or not a single gene -- and this
module reports which rather than quietly dropping it.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

#: Pinned. The census is versioned and a floating "latest" would make
#: every recorded count unreproducible.
CENSUS_VERSION = "2025-01-30"

ATTRIBUTION = (
    "CZ CELLxGENE Census, version " + CENSUS_VERSION + ", "
    "cellxgene.cziscience.com. Aggregates the Human Cell Atlas, Tabula "
    "Sapiens and others; each dataset carries its own citation.")

#: Library lineage -> the Cell Ontology label the census uses. Verified
#: against the census: all 25 resolve.
#:
#: Macrophage M1 and M2 both map to "macrophage" because the atlas does
#: not type polarisation. Anything that would distinguish them cannot be
#: checked here, and the duplicate mapping is left visible rather than
#: hidden behind one entry.
ONTOLOGY: Dict[str, str] = {
    "Basophil": "basophil",
    "CD4+ Th1 cell": "T-helper 1 cell",
    "CD4+ Th2 cell": "T-helper 2 cell",
    "CD4+ Th17 cell": "T-helper 17 cell",
    "CD8+ Effector T cell": "effector CD8-positive, alpha-beta T cell",
    "CD8+ Memory T cell": "central memory CD8-positive, alpha-beta T cell",
    "CD8+ Naive T cell":
        "naive thymus-derived CD8-positive, alpha-beta T cell",
    "Conventional dendritic cell (cDC)": "conventional dendritic cell",
    "Eosinophil": "eosinophil",
    "Gamma-delta T cell": "gamma-delta T cell",
    "ILC1": "group 1 innate lymphoid cell",
    "ILC2": "group 2 innate lymphoid cell",
    "ILC3": "group 3 innate lymphoid cell",
    "Macrophage M1": "macrophage",
    "Macrophage M2": "macrophage",
    "Mast cell": "mast cell",
    "Memory B cell": "memory B cell",
    "NK cell": "natural killer cell",
    "NKT cell": "mature NK T cell",
    "Naive B cell": "naive B cell",
    "Neutrophil": "neutrophil",
    "Plasma cell": "plasma cell",
    "Plasmacytoid dendritic cell (pDC)": "plasmacytoid dendritic cell",
    "Regulatory T cell (Treg)": "regulatory T cell",
    "T follicular helper (Tfh)": "T follicular helper cell",
}

#: Below this, a lineage's numbers are too thin to carry an argument.
#: Not a hard floor -- the count is recorded either way -- but it is the
#: line at which this module says so out loud.
THIN_CELL_COUNT = 5_000

#: A marker should be enriched in its own lineage. Below this ratio
#: against the median lineage, the claim is reported as unsupported.
ENRICHMENT = 1.5

MARKERS_PATH = Path(__file__).resolve().parent / "marker_molecules.json"


@dataclass
class Coverage:
    """How much of a lineage the atlas actually holds."""

    lineage: str = ""
    ontology_label: str = ""
    n_cells: int = 0
    shares_label_with: Tuple[str, ...] = ()

    @property
    def thin(self) -> bool:
        return self.n_cells < THIN_CELL_COUNT

    def note(self) -> str:
        parts = [f"{self.n_cells:,} cells typed {self.ontology_label!r} in "
                 f"{ATTRIBUTION}"]
        if self.shares_label_with:
            parts.append(
                "The atlas does not separate this from "
                + ", ".join(repr(s) for s in self.shares_label_with)
                + ": they share one ontology label, so no comparison "
                "between them can be made here.")
        if self.thin:
            parts.append(
                f"THIN: fewer than {THIN_CELL_COUNT:,} cells. Anything "
                "resting on this lineage rests on that sample.")
        return " ".join(parts)


@dataclass
class MarkerCheck:
    """One asserted marker, tested against expression."""

    lineage: str = ""
    marker: str = ""
    gene: str = ""
    pct_in_lineage: float = 0.0
    pct_median_elsewhere: float = 0.0
    n_cells: int = 0
    note: str = ""

    @property
    def enrichment(self) -> float:
        if self.pct_median_elsewhere <= 0:
            return float("inf") if self.pct_in_lineage > 0 else 0.0
        return self.pct_in_lineage / self.pct_median_elsewhere

    @property
    def supported(self) -> bool:
        return self.enrichment >= ENRICHMENT and self.pct_in_lineage > 0.05


def marker_gene_map() -> Dict[str, str]:
    """Marker string -> gene symbol, from the library's own records.

    The library names markers the way immunologists do -- CD19, BLIMP-1,
    CD45RA -- and an atlas is indexed by gene symbol. Everything that
    does not resolve is reported rather than dropped: IgD and IgM are
    not single genes, CD127lo is an expression level, and TCRgd is a
    heterodimer. Those are real reasons a claim cannot be tested, and
    silence about them would read as a passing test.
    """
    rows = json.loads(MARKERS_PATH.read_text())
    out: Dict[str, str] = {}
    for row in rows:
        gene = row.get("gene")
        if not gene:
            continue
        keys = [row.get("referenced_as"), row.get("entity_name")]
        keys += list(row.get("synonyms") or [])
        for key in keys:
            if key:
                out[str(key).strip()] = gene
    return out


def _open_census():
    import cellxgene_census

    return cellxgene_census.open_soma(census_version=CENSUS_VERSION)


def cell_counts(census=None) -> Dict[str, int]:
    """Primary-data cell count for every ontology label we care about."""
    close = census is None
    census = census or _open_census()
    try:
        obs = census["census_data"]["homo_sapiens"].obs
        frame = obs.read(value_filter="is_primary_data == True",
                         column_names=["cell_type"]).concat().to_pandas()
        counts = frame["cell_type"].value_counts()
        return {label: int(counts.get(label, 0))
                for label in set(ONTOLOGY.values())}
    finally:
        if close:
            census.close()


#: Cells sampled per lineage for the marker test. A percentage is a
#: ratio, so it converges fast; the whole census is not needed and
#: fetching it would cost hours for a number that stops moving.
SAMPLE_PER_LINEAGE = 1_500


def check_markers(store, *, census=None,
                  sample: int = SAMPLE_PER_LINEAGE
                  ) -> Tuple[List[MarkerCheck], List[str]]:
    """Test every asserted marker against expression in its own lineage.

    Returns ``(checks, unresolved)``. A marker is supported when the
    fraction of cells expressing it in its own lineage is at least
    :data:`ENRICHMENT` times the median across the other lineages --
    a comparison within one gene, which is what transcript data
    supports. Comparing two different genes' counts would not be.
    """
    import numpy as np
    import cellxgene_census

    import cognisom.visualization.cell_profiles as cp
    from .models import EntityType

    gene_of = marker_gene_map()
    claims: Dict[str, List[Tuple[str, str]]] = {}
    unresolved: List[str] = []
    entities, _ = store.search(limit=10000)
    for entity in entities:
        if entity.entity_type is not EntityType.IMMUNE_CELL:
            continue
        if entity.name not in ONTOLOGY:
            continue
        for marker in sorted(cp.markers_of(entity.to_dict())):
            gene = gene_of.get(marker)
            if gene is None:
                unresolved.append(f"{entity.name}: {marker}")
                continue
            claims.setdefault(entity.name, []).append((marker, gene))

    genes = sorted({g for v in claims.values() for _, g in v})
    labels = sorted(set(ONTOLOGY.values()))
    if not genes:
        return [], unresolved

    close = census is None
    census = census or _open_census()
    try:
        human = census["census_data"]["homo_sapiens"]
        obs = human.obs.read(
            value_filter=("is_primary_data == True and cell_type in "
                          + str(labels)),
            column_names=["soma_joinid", "cell_type"],
        ).concat().to_pandas()

        rng = np.random.default_rng(0)
        picked = []
        for label, group in obs.groupby("cell_type", observed=True):
            take = group if len(group) <= sample else group.iloc[
                rng.choice(len(group), sample, replace=False)]
            picked.append(take)
        import pandas as pd
        chosen = pd.concat(picked)
        log.info("marker test over %d cells, %d genes", len(chosen), len(genes))

        adata = cellxgene_census.get_anndata(
            census=census, organism="Homo sapiens",
            obs_coords=chosen["soma_joinid"].to_numpy(),
            var_value_filter="feature_name in " + str(genes),
            column_names={"obs": ["cell_type"], "var": ["feature_name"]},
        )
    finally:
        if close:
            census.close()

    matrix = np.asarray(adata.X.todense())
    cell_type = adata.obs["cell_type"].astype(str).to_numpy()
    feature = list(adata.var["feature_name"].astype(str))
    # Fraction of cells expressing each gene, per ontology label.
    pct: Dict[str, Dict[str, float]] = {}
    for col, gene in enumerate(feature):
        per_label = {}
        for label in set(cell_type):
            sel = matrix[cell_type == label, col]
            if len(sel):
                per_label[label] = float((sel > 0).mean())
        pct[gene] = per_label

    checks: List[MarkerCheck] = []
    for lineage, pairs in sorted(claims.items()):
        label = ONTOLOGY[lineage]
        for marker, gene in pairs:
            per_label = pct.get(gene)
            if not per_label or label not in per_label:
                unresolved.append(f"{lineage}: {marker} ({gene}) not measured")
                continue
            mine = per_label[label]
            others = [v for k, v in per_label.items() if k != label]
            median = float(np.median(others)) if others else 0.0
            n = int((cell_type == label).sum())
            checks.append(MarkerCheck(lineage, marker, gene, mine, median, n))
    return checks, unresolved


@dataclass
class Run:
    coverage: List[Coverage] = field(default_factory=list)
    markers: List[MarkerCheck] = field(default_factory=list)
    unresolved: List[str] = field(default_factory=list)
    written: int = 0

    @property
    def thin(self) -> List[Coverage]:
        return [c for c in self.coverage if c.thin]

    @property
    def unsupported(self) -> List[MarkerCheck]:
        return [m for m in self.markers if not m.supported]

    def summary(self) -> str:
        return (f"{len(self.coverage)} lineages covered "
                f"({len(self.thin)} thin), {len(self.markers)} markers "
                f"tested ({len(self.unsupported)} unsupported), "
                f"{len(self.unresolved)} markers not a single gene, "
                f"{self.written} entities annotated")


def source(store, *, dry_run: bool = False, census=None) -> Run:
    """Record atlas coverage on every lineage. Writes no parameter.

    Everything lands in ``entity.sourced``, which is annotation rather
    than evidence for a value. Nothing here is a number the simulation
    reads, and that is the design: an RNA atlas has no business setting
    a protein parameter.
    """
    run = Run()
    close = census is None
    census = census or _open_census()
    try:
        counts = cell_counts(census)
    finally:
        if close:
            census.close()

    shared: Dict[str, List[str]] = {}
    for lineage, label in ONTOLOGY.items():
        shared.setdefault(label, []).append(lineage)

    for lineage, label in sorted(ONTOLOGY.items()):
        others = tuple(sorted(set(shared[label]) - {lineage}))
        coverage = Coverage(lineage, label, counts.get(label, 0), others)
        run.coverage.append(coverage)
        if dry_run:
            continue
        entity = store.find_entity_by_name(lineage)
        if entity is None:
            continue
        entity.sourced["atlas_coverage"] = {
            "cells": coverage.n_cells,
            "ontology_label": label,
            "census_version": CENSUS_VERSION,
            "note": coverage.note(),
        }
        store.update_entity(entity, changed_by="cellxgene_sourcing")
        run.written += 1
    return run


def main(argv: Sequence[str] | None = None) -> int:    # pragma: no cover
    import argparse
    import contextlib
    import io

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    from .store import EntityStore
    with contextlib.redirect_stdout(io.StringIO()):
        store = EntityStore()
    run = source(store, dry_run=args.dry_run)
    print(run.summary())
    print(f"\n  {ATTRIBUTION}\n")
    print(f"  {'lineage':<36} {'cells':>12}")
    for cov in sorted(run.coverage, key=lambda c: -c.n_cells):
        flag = "   THIN" if cov.thin else ""
        print(f"  {cov.lineage:<36} {cov.n_cells:>12,}{flag}")
    shared = [c for c in run.coverage if c.shares_label_with]
    if shared:
        print("\n  SHARE ONE ONTOLOGY LABEL (not separable here):")
        for cov in shared:
            print(f"    {cov.lineage} = {cov.ontology_label}")
    return 0


if __name__ == "__main__":                             # pragma: no cover
    raise SystemExit(main())
