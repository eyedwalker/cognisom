#!/usr/bin/env python3
"""
Sprint 2 Module Demo: Reference-Genome + Per-Cell Delta Architecture
====================================================================

Patent-evidence demo for Upgrade 1 (USC 101 anchor). Drives the full
stack -- SimulationEngine, MolecularModule, ReferenceGenome,
CellGenomeView -- end-to-end with ~20 cells and induces a multi-
generation transformation. Prints the two facts the patent
specification relies on:

    A) Reference-identity invariant
       Every cell view holds the SAME ReferenceGenome object
       (id(view.reference) is identical across cells). No cell ever
       copies the reference genome.

    B) Per-cell sparse delta count
       Each cell's per-cell memory cost is O(deltas), not O(genome).
       Unmutated cells carry 0 deltas. Mutated cells carry one delta
       per substitution. Daughter cells inherit parent deltas via
       fork() and can diverge with subsequent mutations.

The existing cancer_transmission_demo.py at examples/molecular/ uses
raw Gene/Exosome objects (SimplifiedCell) and does NOT exercise the
MolecularModule refactor. This file is the canonical end-to-end
evidence for the Sprint 2 architecture and is referenced from
DECISIONS.md.
"""

import sys
import os

# Allow running directly from examples/patent_evidence/
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from core import SimulationEngine, SimulationConfig
from core.event_bus import EventTypes
from modules.molecular_module import MolecularModule


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def assert_shared_reference(molecular: MolecularModule) -> None:
    """Patent claim A: every view points at THE shared ReferenceGenome."""
    ref_id = id(molecular.reference_genome)
    mismatches = []
    for cell_id, view in molecular.cell_views.items():
        if id(view.reference) != ref_id:
            mismatches.append(cell_id)
    if mismatches:
        raise AssertionError(
            f"Reference-identity broken for cells {mismatches}; "
            "per-cell views must share the module's ReferenceGenome object."
        )
    print(f"  [A] Reference-identity: id={ref_id} shared across "
          f"{len(molecular.cell_views)} cells. PASS.")


def print_delta_table(molecular: MolecularModule, label: str) -> None:
    """Patent claim B: print per-cell delta counts."""
    print(f"\n  [B] Per-cell delta counts -- {label}")
    print(f"      {'cell':>6} {'#deltas':>8}  {'oncogene_flags':<24} deltas")
    total_deltas = 0
    for cell_id in sorted(molecular.cell_views.keys()):
        view = molecular.cell_views[cell_id]
        n = view.n_deltas()
        total_deltas += n
        flags = ",".join(sorted(molecular.cell_oncogene_flags.get(cell_id, set()))) or "-"
        delta_repr = ", ".join(
            f"{d.gene_name}:{d.mutation_id or f'pos{d.position}'}"
            for d in view.deltas()
        ) or "-"
        print(f"      {cell_id:>6} {n:>8}  {flags:<24} {delta_repr}")
    n_cells = len(molecular.cell_views)
    print(f"      total deltas across {n_cells} cells: {total_deltas}")
    print(f"      avg deltas/cell: {total_deltas / n_cells:.3f}")


def reference_memory_summary(molecular: MolecularModule) -> None:
    """Show what the reference genome carries -- shared cost."""
    ref = molecular.reference_genome
    print()
    print("  Shared reference genome (paid once for the entire population):")
    for name in sorted(ref.gene_names()):
        print(f"      {name:<8} {ref.length(name)} bp")
    print(f"      total: {ref.total_bases()} bp shared")


def main() -> int:
    banner("Sprint 2 Module Demo  --  Upgrade 1 patent evidence")
    print("Population: 20 cells  |  3 reference genes  |  multi-generation lineage")

    # --- Build engine + register molecular module ----------------------------
    engine = SimulationEngine(SimulationConfig(dt=0.05, duration=1.0, use_gpu=False))
    engine.register_module("molecular", MolecularModule, {
        "transcription_rate": 0.4,
        "exosome_release_rate": 0.0,  # silence exosome chatter for this demo
        "mutation_rate": 0.0,         # we drive mutations explicitly, not stochastically
    })
    engine.initialize()
    molecular: MolecularModule = engine.modules["molecular"]

    # --- Founder generation: 20 unmutated cells ------------------------------
    N_FOUNDERS = 20
    for cid in range(N_FOUNDERS):
        molecular.add_cell(cid)

    banner("Generation 0 (founders, no mutations)")
    reference_memory_summary(molecular)
    assert_shared_reference(molecular)
    print_delta_table(molecular, "founders")

    # --- Generation 1: introduce mutations in 4 of the 20 founders ----------
    # Use one mutation per gene to exercise all three reference CDSes
    # (KRAS, BRAF, TP53) and the classifier path inside introduce_mutation.
    banner("Generation 1: introduce hotspot mutations in 4 founders")
    mutations_to_apply = [
        (2,  "KRAS", "G12D"),
        (5,  "KRAS", "G12V"),
        (11, "BRAF", "V600E"),
        (17, "TP53", "R175H"),
    ]
    for cell_id, gene, mut_name in mutations_to_apply:
        mut = molecular.introduce_mutation(cell_id, gene, mut_name)
        assert mut is not None, f"introduce_mutation({cell_id},{gene},{mut_name}) returned None"
        print(f"  cell {cell_id:>2}  {gene} {mut_name}  ->  "
              f"category={mut.effect.category:<10} "
              f"impact={mut.effect.impact_score:.2f}  "
              f"aa_change={mut.effect.aa_change}  "
              f"oncogenic={mut.oncogenic}")

    assert_shared_reference(molecular)
    print_delta_table(molecular, "after Gen-1 mutations")

    # --- Generation 2: divide 2 mutated cells (the daughters inherit deltas)
    # On CELL_DIVIDED, MolecularModule.on_cell_divided forks the parent view,
    # carrying its delta log to the daughter; oncogene flags are also inherited.
    banner("Generation 2: cells 2 (KRAS G12D) and 11 (BRAF V600E) divide")
    daughter_a = 100  # daughter of cell 2
    daughter_b = 101  # daughter of cell 11
    engine.event_bus.emit(EventTypes.CELL_DIVIDED, {
        "cell_id": 2, "daughter_id": daughter_a,
        "position": [0.0, 0.0, 0.0],
    })
    engine.event_bus.emit(EventTypes.CELL_DIVIDED, {
        "cell_id": 11, "daughter_id": daughter_b,
        "position": [0.0, 0.0, 0.0],
    })
    # Force a single engine step to flush events through the bus.
    engine.step()

    print("  After fork:")
    print(f"    cell  2  deltas={molecular.cell_views[2].n_deltas()}  "
          f"flags={sorted(molecular.cell_oncogene_flags[2])}")
    print(f"    cell {daughter_a}  deltas={molecular.cell_views[daughter_a].n_deltas()}  "
          f"flags={sorted(molecular.cell_oncogene_flags[daughter_a])}  "
          "(daughter inherits parent deltas)")
    print(f"    cell 11  deltas={molecular.cell_views[11].n_deltas()}  "
          f"flags={sorted(molecular.cell_oncogene_flags[11])}")
    print(f"    cell {daughter_b}  deltas={molecular.cell_views[daughter_b].n_deltas()}  "
          f"flags={sorted(molecular.cell_oncogene_flags[daughter_b])}")

    # --- Generation 3: divergence  ------------------------------------------
    # Mutate ONE daughter further: it should now carry strictly more deltas
    # than its parent, proving the fork() snapshot semantics.
    banner("Generation 3: daughter 100 acquires a second hit (TP53 R248W)")
    parent_deltas_before = molecular.cell_views[2].n_deltas()
    mut = molecular.introduce_mutation(daughter_a, "TP53", "R248W")
    assert mut is not None
    print(f"  daughter {daughter_a}: now has {molecular.cell_views[daughter_a].n_deltas()} deltas; "
          f"parent (cell 2) still has {molecular.cell_views[2].n_deltas()} -- divergence proven.")
    assert molecular.cell_views[2].n_deltas() == parent_deltas_before, (
        "Parent gained a delta when daughter mutated -- views are not independent."
    )

    # --- Run a few engine steps so transcription fires through CellGenomeView
    banner("Run engine: 0.5h of transcription via CellGenomeView")
    engine.run(duration=0.5)

    # --- Final state --------------------------------------------------------
    banner("Final state")
    state = molecular.get_state()
    print(f"  cells tracked:           {state['n_cells_tracked']}")
    print(f"  total transcriptions:    {state['total_transcriptions']}")
    print(f"  total mutations applied: {state['total_mutations']}")
    print(f"  total deltas summed:     "
          f"{sum(v.n_deltas() for v in molecular.cell_views.values())}")
    assert_shared_reference(molecular)
    print_delta_table(molecular, "end of run")

    # Sanity: a mutated cell's mRNA should carry the mutation; an unmutated
    # cell's mRNA for the same gene should not.
    banner("mRNA mutation propagation spot-check")
    view_mut = molecular.cell_views[2]
    view_wt = molecular.cell_views[0]
    mrna_mut = molecular._build_mrna_from_view(view_mut, "KRAS", 2)
    mrna_wt = molecular._build_mrna_from_view(view_wt, "KRAS", 0)
    print(f"  cell  2 (KRAS G12D)  mRNA mutations: {[m.name for m in mrna_mut.mutations]}")
    print(f"  cell  0 (wild type)  mRNA mutations: {[m.name for m in mrna_wt.mutations]}")
    assert any(m.name == "G12D" for m in mrna_mut.mutations), \
        "Materialized mRNA for cell 2 missing the G12D delta"
    assert not mrna_wt.mutations, \
        "Wild-type cell mRNA should carry zero mutations"
    print("  PASS -- mutations propagate per-cell through CellGenomeView.")

    banner("Demo complete -- Upgrade 1 patent claims A and B demonstrated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
