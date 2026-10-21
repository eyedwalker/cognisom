"""How much of each protein a cell actually contains.

The library knows what molecules a cell carries and not how many of
them. That is the gap between a diagram and a cell: what makes a
Goodsell illustration read as alive is not the shapes, it is the
DENSITY -- every molecule to scale, packed as densely as it really is.
Cytoplasm is roughly 300 mg/mL of protein, effectively solid, and this
renderer draws a few hundred features in a mostly empty sphere.

PaxDb is the measurement. Integrated mass-spectrometry abundances, 419
human datasets, **CC BY 4.0** and free for commercial use with
attribution.

FIVE OF ITS DATASETS ARE THIS PLATFORM'S CELLS
----------------------------------------------
Kim et al., Nature 2014 -- the human draft proteome -- separated
peripheral blood into B cells, CD4 T cells, CD8 T cells, monocytes and
NK cells, each measured on its own. Those are five of the library's own
immune lineages, measured rather than inferred from a tissue average,
and they cover 148 of the 216 entities here that carry a gene symbol.

ppm, NOT COPIES, AND THE DIFFERENCE MATTERS
-------------------------------------------
PaxDb reports parts per million: of every million protein molecules in
the cell, how many are this one. Each file sums to 1,000,000, which is
how you can tell it is a mole fraction and not a concentration.

Copies per cell would be ppm times the total number of protein
molecules in that cell, and **this library cannot cite that total for
any of its cell types**. Rather than multiply by a number from memory,
the ppm is stored as what it is. It is also enough for the thing it is
wanted for: relative crowding needs mole fractions and molecular
volumes, both of which are now here, and only ABSOLUTE density needs
the total.

AN ABUNDANCE IS NOT A PROPERTY OF A PROTEIN
-------------------------------------------
Perforin is 2,907 ppm in an NK cell and far less in a B cell. So this
cannot be a scalar on the protein entity the way a molecular weight is;
it is a function of cell type, and it is stored as one -- a mapping from
cell type to ppm, in ``sourced``, where it survives whatever the entity
class is.

WHAT ABSENCE MEANS HERE
-----------------------
68 of the library's genes appear in none of the five datasets, and they
are mostly secreted chemokines -- CCL2, CCL19, CCL21, CXCL1, CXCL8 --
and low-abundance surface receptors. A secreted protein leaves the cell
pellet that was digested, and a receptor at a few hundred copies falls
under the detection floor of spectral counting. **Absence from PaxDb is
a detection limit, not a zero**, and nothing here records it as one.
"""
from __future__ import annotations

import logging
import os
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

#: Required by CC BY 4.0, stored on every value this module writes.
ATTRIBUTION = (
    "PaxDb (pax-db.org) v6.1, CC BY 4.0. Abundances from Kim et al., "
    "Nature 2014 (doi:10.1038/nature13302), human draft proteome, "
    "spectral counting on sorted peripheral blood populations.")

#: Where to get the data, because it is not in the repository: 28 MB of
#: abundances and 17 MB of identifier mappings are a download, not an
#: asset.
#:
#:     curl -A 'your-tool/1.0' -o 9606.zip \\
#:       https://pax-db.org/downloads/latest/datasets/9606.zip
#:     unzip 9606.zip -d <somewhere>
#:
#: then point ``--data-dir`` at the ``9606`` directory inside it.
DOWNLOAD = "https://pax-db.org/downloads/latest/datasets/9606.zip"

#: Below this many spectra, two proteins landing on the same count is
#: ordinary granularity rather than evidence of shared peptides. Set
#: from the data: counts in the single digits are heavily degenerate,
#: and every real family tie observed here -- the actins at 11,313, the
#: H4 cluster at 1,712, the H2B cluster at 2,119 -- is far above it.
MIN_SHARED_SPECTRA = 50

#: A shared peptide pool is a property of the SEQUENCES, so it holds in
#: every dataset where both proteins were seen -- not in some of them.
#: Measured over these five, the distribution is bimodal and leaves no
#: room to argue: 56,126 gene pairs tie in exactly one dataset, 450 in
#: two, 42 in three, 38 in four, and 428 in all five. The all-five
#: population is unmistakable on inspection -- ACTB/ACTG1, the three
#: calmodulin genes that encode an identical protein, the H2A and H4
#: histone clusters, the tubulins -- and everything below it is
#: coincidence at a coarse granularity.
#:
#: So the rule is not a threshold at all: the tie must hold EVERY time
#: both were measured.
REQUIRE_TIE_IN_EVERY_DATASET = True

#: PaxDb dataset -> the library cell type it measured.
#:
#: Only the five sorted blood populations. The tissue datasets -- spleen,
#: lymph node -- are mixtures, and attributing a lymph node's proteome to
#: any one of the cells in it would be the kind of inference this module
#: exists to avoid.
DATASETS: Dict[str, Tuple[str, ...]] = {
    "9606-Adult_CD4Tcells_Kim_2014_SEQUEST.txt": (
        "CD4+ T cell", "CD4+ Th1 cell", "CD4+ Th2 cell", "CD4+ Th17 cell",
        "Regulatory T cell (Treg)", "T follicular helper (Tfh)"),
    "9606-Adult_CD8Tcells_Kim_2014_SEQUEST.txt": (
        "CD8+ T cell", "CD8+ Effector T cell", "CD8+ Memory T cell",
        "CD8+ Naive T cell", "Exhausted CD8+ T cell"),
    "9606-Adult_NKcells_Kim_2014_SEQUEST.txt": ("NK cell", "NKT cell"),
    "9606-Adult_Monocytes_Kim_2014_SEQUEST.txt": (
        "Monocyte", "Macrophage", "Macrophage M1", "Macrophage M2"),
    "9606-Adult_Bcells_Kim_2014_SEQUEST.txt": (
        "Naive B cell", "Memory B cell", "Plasma cell"),
}

#: The population actually measured, for the record. A CD4 dataset is a
#: bulk CD4 sort, so attributing it to a Th17 is an assumption and the
#: assumption is named rather than hidden.
MEASURED_POPULATION: Dict[str, str] = {
    "9606-Adult_CD4Tcells_Kim_2014_SEQUEST.txt": "bulk CD4+ T cells",
    "9606-Adult_CD8Tcells_Kim_2014_SEQUEST.txt": "bulk CD8+ T cells",
    "9606-Adult_NKcells_Kim_2014_SEQUEST.txt": "sorted NK cells",
    "9606-Adult_Monocytes_Kim_2014_SEQUEST.txt": "peripheral blood monocytes",
    "9606-Adult_Bcells_Kim_2014_SEQUEST.txt": "peripheral blood B cells",
}


@dataclass
class Dataset:
    """One PaxDb abundance file."""

    filename: str = ""
    organ: str = ""
    name: str = ""
    coverage: float = 0.0
    #: ``GENE -> ppm``
    ppm: Dict[str, float] = field(default_factory=dict)
    #: ``GENE -> the other genes built from the identical spectra``.
    #:
    #: Spectral counting assigns a peptide to every protein it could
    #: have come from, so a family whose members are 90% identical gets
    #: one measurement reported several times. The signature is an
    #: identical RAW SPECTRAL COUNT, not an identical ppm: ppm ties are
    #: mostly coincidence, because the values are coarse -- 92% of the
    #: proteins under 5 ppm share a value with something.
    #:
    #: On the count the picture is unambiguous. ACTB and ACTG1 both
    #: report 11,313 spectra in NK cells; fourteen histone H4 genes all
    #: report 1,712. Meanwhile PCBP1 and ACTA2 happen to land on the
    #: same 1,533 ppm from 381 and 4,505 spectra respectively, which is
    #: a coincidence and is correctly left alone.
    shared: Dict[str, Tuple[str, ...]] = field(default_factory=dict)

    @property
    def total_ppm(self) -> float:
        return sum(self.ppm.values())


def read_dataset(path: str) -> Dataset:
    """Parse one abundance file, header and all."""
    out = Dataset(filename=os.path.basename(path))
    counts: Dict[str, int] = {}
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#"):
                if ":" in line:
                    key, _, value = line[1:].partition(":")
                    key, value = key.strip().lower(), value.strip()
                    if key == "organ":
                        out.organ = value
                    elif key == "name":
                        out.name = value
                    elif key == "coverage":
                        try:
                            out.coverage = float(value)
                        except ValueError:
                            pass
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or not parts[0].strip():
                continue
            gene = parts[0].strip().upper()
            try:
                value = float(parts[2])
            except ValueError:
                continue
            count = 0
            if len(parts) > 3:
                try:
                    count = int(float(parts[3]))
                except ValueError:
                    count = 0
            # A gene can appear under several ENSPs; keep the largest,
            # which is the dominant isoform by abundance.
            if value > out.ppm.get(gene, 0.0):
                out.ppm[gene] = value
                counts[gene] = count

    # An identical spectral count is one measurement reported twice.
    by_count: Dict[int, List[str]] = {}
    for gene, count in counts.items():
        if count >= MIN_SHARED_SPECTRA:
            by_count.setdefault(count, []).append(gene)
    for genes in by_count.values():
        if len(genes) < 2:
            continue
        for gene in genes:
            out.shared[gene] = tuple(sorted(g for g in genes if g != gene))
    return out


@dataclass
class Run:
    datasets: List[Dataset] = field(default_factory=list)
    #: entity name -> {cell type: ppm}
    abundances: Dict[str, Dict[str, float]] = field(default_factory=dict)
    missing_datasets: List[str] = field(default_factory=list)
    #: Library genes that appear in none of the five.
    undetected: List[str] = field(default_factory=list)
    #: entity name -> the genes it shares a peptide pool with.
    shared_pool: Dict[str, set] = field(default_factory=dict)
    #: entity -> partner -> ``(ties, datasets where both were seen)``.
    _tie_counts: Dict[str, Dict[str, Tuple[int, int]]] = field(
        default_factory=dict)
    written: int = 0

    def summary(self) -> str:
        return (f"{len(self.datasets)} datasets read, "
                f"{len(self.abundances)} molecules with an abundance, "
                f"{self.written} written, "
                f"{len(self.undetected)} library genes undetected")


def _library_genes(store) -> Dict[str, str]:
    """``GENE -> entity name`` for everything with a gene symbol."""
    import json

    out: Dict[str, str] = {}
    for blob, in store._conn.execute("SELECT data_json FROM entities"):
        data = json.loads(blob)
        gene = ((data.get("properties") or {}).get("gene_symbol") or "").strip()
        if gene:
            out.setdefault(gene.upper(), data["name"])
    return out


def source(store, data_dir: str, *, dry_run: bool = False) -> Run:
    """Read the five sorted-blood datasets onto the library's molecules."""
    run = Run()
    genes = _library_genes(store)
    seen: set = set()

    for filename, cell_types in DATASETS.items():
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            run.missing_datasets.append(filename)
            continue
        dataset = read_dataset(path)
        run.datasets.append(dataset)
        population = MEASURED_POPULATION.get(filename, dataset.organ)
        for gene, ppm in dataset.ppm.items():
            entity_name = genes.get(gene)
            if entity_name is None:
                continue
            seen.add(gene)
            per_cell = run.abundances.setdefault(entity_name, {})
            for cell_type in cell_types:
                per_cell[cell_type] = ppm
            for cell_type in cell_types:
                per_cell.setdefault("_measured_in", {})[cell_type] = population
            pair = run._tie_counts.setdefault(entity_name, {})
            tied = set(dataset.shared.get(gene, ()))
            # Count both the ties and the chances to tie, so a pair can
            # be asked whether it tied EVERY time rather than often.
            for partner in tied:
                hits, seen_both = pair.get(partner, (0, 0))
                pair[partner] = (hits + 1, seen_both + 1)
            for partner in list(pair):
                if partner in tied or partner not in dataset.ppm:
                    continue
                hits, seen_both = pair[partner]
                pair[partner] = (hits, seen_both + 1)

    run.undetected = sorted(set(genes) - seen)

    # A tie in ONE dataset is granularity; the same two proteins tying
    # in two or more is one measurement reported twice. ACTB and ACTG1
    # tie in all five. A gene that happened to land on another's count
    # once does not repeat, which is what separated 144 false positives
    # from the handful that are real.
    for entity_name, partners in run._tie_counts.items():
        always = {partner for partner, (hits, seen_both) in partners.items()
                  if hits >= 2 and hits == seen_both}
        if always:
            run.shared_pool[entity_name] = always

    if dry_run:
        return run

    for entity_name, per_cell in run.abundances.items():
        entity = store.find_entity_by_name(entity_name)
        if entity is None:
            continue
        measured_in = per_cell.pop("_measured_in", {}) or {}
        entity.sourced["abundance_ppm"] = {
            k: round(v, 2) for k, v in sorted(per_cell.items())}
        entity.sourced["abundance_source"] = ATTRIBUTION
        shared = sorted(run.shared_pool.get(entity_name, ()))
        # Cleared when there is nothing to say, so a re-run with a
        # better rule removes what the previous rule wrongly wrote. A
        # pass that only ever ADDS leaves its own mistakes behind: the
        # first version of this detector flagged 144 of 148 molecules,
        # and the fix looked like it had not worked because the stale
        # keys were still sitting in the database.
        entity.sourced.pop("abundance_shared_with", None)
        if shared:
            # Not this protein's abundance on its own. Recorded rather
            # than dropped, because the number is still the family's
            # and a reader who knows that can use it.
            entity.sourced["abundance_shared_with"] = shared
        entity.sourced["abundance_measured_in"] = dict(sorted(
            measured_in.items()))
        entity.sourced["abundance_note"] = (
            "Parts per million of total protein MOLECULES. Not copies "
            "per cell: that would be this number times the total protein "
            "molecules in the cell, which this library cannot cite for "
            "any of its cell types. abundance_measured_in names the "
            "population actually sorted for each entry -- a Th17's value "
            "is the bulk CD4 measurement, which is an assumption and is "
            "recorded as one rather than hidden.")
        store.update_entity(entity, changed_by="paxdb_sourcing")
        run.written += 1
    return run


def main(argv: Sequence[str] | None = None) -> int:     # pragma: no cover
    import argparse
    import contextlib
    import io

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True,
                        help="the 9606 directory from " + DOWNLOAD)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    from .store import EntityStore
    with contextlib.redirect_stdout(io.StringIO()):
        store = EntityStore()
    run = source(store, args.data_dir, dry_run=args.dry_run)
    print(run.summary())
    print(f"\nattribution: {ATTRIBUTION}")

    for dataset in run.datasets:
        # PaxDb labels three of these CELL_LINE in its organ header; the
        # population actually sorted is the useful label.
        population = MEASURED_POPULATION.get(dataset.filename, dataset.organ)
        print(f"  {population:<26} {len(dataset.ppm):>6,} proteins  "
              f"sums to {dataset.total_ppm:>10,.0f} ppm  "
              f"coverage {dataset.coverage:g}%  "
              f"{len(dataset.shared):>4} in shared pools")
    if run.missing_datasets:
        print(f"\nMISSING FILES ({len(run.missing_datasets)}):")
        for name in run.missing_datasets:
            print("   ", name)
    if run.abundances:
        print("\nMOST ABUNDANT OF OURS, by any cell type:")
        ranked = sorted(
            ((max(v for k, v in per.items() if not k.startswith("_")), name, per)
             for name, per in run.abundances.items()), reverse=True)[:10]
        for ppm, name, per in ranked:
            where = max(((v, k) for k, v in per.items()
                         if not k.startswith("_")))[1]
            print(f"   {ppm:>9,.0f} ppm  {name[:42]:<44} in {where}")
    print(f"\nUNDETECTED ({len(run.undetected)}) -- a detection limit, "
          "not a zero:")
    print("   ", ", ".join(run.undetected[:20]))
    return 0


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(main())
