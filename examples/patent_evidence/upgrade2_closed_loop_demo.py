#!/usr/bin/env python3
"""
Upgrade 2 Closed-Loop Demo: Neoantigen presentation, end-to-end
================================================================

Patent-evidence demo for Upgrade 2 (USC 101 anchor). Drives the full
stack -- SimulationEngine + MolecularModule + CellularModule +
ImmuneModule -- against three canonical cancer driver mutations and
prints the resulting event trace at human-readable resolution. The
patent-evidence claims this demo demonstrates:

  CLAIM A (full closed loop, sensitivity).
    For a mutation whose neoantigen peptides are presentable on the
    patient's HLA panel, the trace fires in causal order:
        MUTATION_OCCURRED -> PEPTIDE_GENERATED -> PEPTIDE_PRESENTED
            -> IMMUNE_ACTIVATED -> CELL_KILLED_BY_TCELL
    and the kill event carries enough provenance (peptide, mutation,
    gene, HLA, TCR, affinity) to reconstruct the causal chain.

  CLAIM B (HLA restriction, specificity).
    For a mutation whose neoantigen peptides do NOT bind the patient's
    HLA panel, the trace stops at PEPTIDE_GENERATED. The pipeline
    correctly predicts that this patient cannot mount a T-cell-mediated
    response against that mutation. This is a USC 101 differentiator
    vs. uniform "predict immune response" prior art -- the pipeline
    captures the clinical reality that HLA mismatch makes some
    mutations invisible to the patient's T cells (e.g., BRAF V600E
    is famously hard for immunotherapy across HLA-A/B/C panels and
    is treated with the small-molecule inhibitor vemurafenib instead).

  CLAIM C (no neoantigen = no T-cell kill).
    A wild-type cancer cell with no driver mutation generates no
    peptides and is not T-cell-killed. The closed loop only fires
    when a mutation is present.

End-to-end test at tests/test_closed_loop_neoantigen.py asserts the
trace ordering programmatically; this demo prints the same evidence in
a form attorneys / patent examiners can read.

Parallel to examples/patent_evidence/sprint2_module_demo.py, which is
the Upgrade-1 evidence file (reference-genome + per-cell-delta).
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

# Silence TensorFlow / Keras chatter from MHCflurry so the patent-
# evidence trace is the only thing in the output. Must be set BEFORE
# tensorflow / mhcflurry is imported (which happens transitively via
# MHCLoader -> NeoantigenPredictor).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:  # Best-effort: silence keras progress bars too
    import tensorflow as _tf
    _tf.get_logger().setLevel("ERROR")
    _tf.keras.utils.disable_interactive_logging()
except Exception:
    pass

from core import SimulationConfig, SimulationEngine
from core.event_bus import EventTypes
from modules.cellular_module import CellularModule
from modules.immune_module import ImmuneModule
from modules.molecular_module import MolecularModule


# Canonical driver mutations to exercise. Each entry is
# (gene, mutation_name, expected_aa_change). All three are in the
# curated ONCOGENIC_SUBSTITUTIONS table; KRAS prefix carries codon 12,
# TP53 reaches R248, BRAF reaches V600.
MUTATIONS: List[Tuple[str, str, str]] = [
    ("KRAS", "G12D", "G12D"),
    ("BRAF", "V600E", "V600E"),
    ("TP53", "R248W", "R248W"),
]


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _build_engine() -> SimulationEngine:
    """Three-module engine tuned for trace visibility.

    - One T cell per cancer cell so detection is deterministic.
    - TCR recognition threshold 0.0 so any displayed peptide is
      recognized; the demo is for the *chain*, not for studying
      threshold sensitivity (covered by unit tests).
    - High costimulation + zero checkpoint block -> Hill kill prob
      stays well above the EC50 once affinity x mhc clears it.
    """
    engine = SimulationEngine(SimulationConfig(
        # 5 steps is enough -- the T cell detects + kills on step 1 or 2
        # at this configuration; keeping the run short caps the
        # mhcflurry overhead at the one-time per-mutation peptide
        # scoring.
        dt=0.01, duration=0.05, use_gpu=False,
    ))
    engine.register_module("molecular", MolecularModule, {
        "transcription_rate": 0.0,
        "exosome_release_rate": 0.0,
        "mutation_rate": 0.0,
    })
    engine.register_module("cellular", CellularModule, {
        "n_normal_cells": 0,
        "n_cancer_cells": 0,
        "hla_alleles": ["HLA-A*02:01", "HLA-A*24:02", "HLA-B*07:02"],
        "max_displayed_per_mutation": 4,
    })
    engine.register_module("immune", ImmuneModule, {
        "n_t_cells": 4, "n_nk_cells": 0, "n_macrophages": 0,
        "tcr_recognition_threshold": 0.0,
        "tcr_repertoire_size": 500,
        "tcr_seed": 0,
        "costimulation": 1.0,
        "checkpoint_block": 0.0,
    })
    engine.initialize()
    return engine


def _spawn_cancer_cell(
    engine: SimulationEngine, position: List[float], cell_index: int
) -> int:
    """Add a cancer cell, register it with the molecular module, and
    park the next-available T cell on top of it. Returns the cancer
    cell's id."""
    cellular: CellularModule = engine.modules["cellular"]
    molecular: MolecularModule = engine.modules["molecular"]
    immune: ImmuneModule = engine.modules["immune"]

    cancer_id = cellular.add_cell(position=position, cell_type="cancer")
    cellular.cells[cancer_id].mhc1_expression = 0.9
    molecular.add_cell(cancer_id)

    # Colocate the i-th T cell with this cancer cell.
    t_ids = list(immune.immune_cells.keys())
    t_cell = immune.immune_cells[t_ids[cell_index]]
    t_cell.position = np.array(position, dtype=np.float32)
    return cancer_id


def _events_for_cell(log, cancer_id: int) -> List[Tuple[str, dict]]:
    """Filter the event log to only events relating to a cell_id."""
    out: List[Tuple[str, dict]] = []
    for evt, data in log:
        if not isinstance(data, dict):
            continue
        if data.get("cell_id") == cancer_id:
            out.append((evt, data))
    return out


def _print_displayed_peptides(cellular: CellularModule, cancer_id: int) -> None:
    cell = cellular.cells.get(cancer_id)
    if cell is None:
        # Cell was killed and removed -- displayed peptides were last
        # observable at the moment of kill; the log carries them via
        # the CELL_KILLED_BY_TCELL provenance fields.
        print("    (cell removed; see CELL_KILLED_BY_TCELL for the recognized peptide)")
        return
    if not cell.mhc1_displayed_peptides:
        print("    (no peptides displayed)")
        return
    for pres in cell.mhc1_displayed_peptides:
        print(f"    {pres.peptide.sequence:<12} on {pres.hla_allele:<14} "
              f"IC50={pres.ic50_nm:>8.1f} nM  "
              f"binding={pres.binding_level:<10} "
              f"mut={pres.peptide.mutation_label}")


def _print_trace_for_cell(log, cancer_id: int) -> None:
    """Print the causal chain for a single cancer cell."""
    events = _events_for_cell(log, cancer_id)
    if not events:
        print("    (no events for this cell)")
        return
    targets = (
        EventTypes.MUTATION_OCCURRED,
        EventTypes.PEPTIDE_GENERATED,
        EventTypes.PEPTIDE_PRESENTED,
        EventTypes.IMMUNE_ACTIVATED,
        EventTypes.CELL_KILLED_BY_TCELL,
    )
    for evt, data in events:
        if evt not in targets:
            continue
        if evt == EventTypes.MUTATION_OCCURRED:
            print(f"    [1] MUTATION_OCCURRED  gene={data.get('gene')}  "
                  f"mut={data.get('mutation')}  aa={data.get('aa_change')}  "
                  f"impact={data.get('impact_score'):.2f}")
        elif evt == EventTypes.PEPTIDE_GENERATED:
            print(f"    [2] PEPTIDE_GENERATED  gene={data.get('gene')}  "
                  f"n={data.get('n_peptides')}")
        elif evt == EventTypes.PEPTIDE_PRESENTED:
            print(f"    [3] PEPTIDE_PRESENTED  {data.get('peptide')}  "
                  f"on {data.get('hla_allele')}  IC50={data.get('ic50_nm'):.1f} nM  "
                  f"({data.get('binding_level')})")
        elif evt == EventTypes.IMMUNE_ACTIVATED:
            tcr_id = data.get("tcr_id")
            if tcr_id is None:
                continue  # NK / macrophage activation, not the closed loop
            print(f"    [4] IMMUNE_ACTIVATED   TCR clone={tcr_id}  "
                  f"affinity={data.get('affinity'):.3f}")
        elif evt == EventTypes.CELL_KILLED_BY_TCELL:
            print(f"    [5] CELL_KILLED_BY_TCELL  "
                  f"peptide={data.get('peptide')}  "
                  f"on {data.get('hla_allele')}  "
                  f"TCR={data.get('tcr_id')}  "
                  f"affinity={data.get('affinity'):.3f}  "
                  f"gene={data.get('source_gene')}  "
                  f"mut={data.get('mutation')}")


def _assert_chain_consistent(log, cancer_id: int, gene: str, mut_name: str) -> str:
    """Patent-evidence assertion: classify the trace and verify
    internal consistency.

    Returns a label:
        "full"            -- claim A: full closed loop fired in order
        "hla_restricted"  -- claim B: peptides generated but none
                              bound the HLA panel; no kill
    Raises AssertionError on any inconsistency (e.g., a kill event
    without a matching presentation, or out-of-order trace).
    """
    events = _events_for_cell(log, cancer_id)
    types = [evt for evt, _ in events]

    must_have = (EventTypes.MUTATION_OCCURRED, EventTypes.PEPTIDE_GENERATED)
    for r in must_have:
        assert r in types, (
            f"trace for cell {cancer_id} ({gene} {mut_name}) missing {r!r}; "
            f"every mutation must reach the peptidome stage. events: {types}"
        )

    has_presented = EventTypes.PEPTIDE_PRESENTED in types
    has_killed = EventTypes.CELL_KILLED_BY_TCELL in types

    if has_killed:
        assert has_presented, (
            f"cell {cancer_id} ({gene} {mut_name}) was T-cell-killed but "
            "PEPTIDE_PRESENTED never fired; closed-loop invariant violated"
        )

    if has_presented:
        # Claim A: presentation fired, so the kill must follow under
        # this demo's tuning (recognition threshold 0.0, high costim).
        assert has_killed, (
            f"cell {cancer_id} ({gene} {mut_name}) had presented peptides "
            "but no CELL_KILLED_BY_TCELL event; the demo is configured to "
            "make recognition + kill probability essentially certain "
            "(threshold=0.0, costim=1.0). Investigate the engine config "
            "or numpy.random seed."
        )
        ordered = [
            EventTypes.MUTATION_OCCURRED,
            EventTypes.PEPTIDE_GENERATED,
            EventTypes.PEPTIDE_PRESENTED,
            EventTypes.CELL_KILLED_BY_TCELL,
        ]
        first_idx = [types.index(r) for r in ordered]
        assert first_idx == sorted(first_idx), (
            f"trace for cell {cancer_id} ({gene} {mut_name}) out of order: "
            f"{dict(zip(ordered, first_idx))}"
        )
        return "full"

    # Claim B: HLA restriction. No presentation => no kill.
    assert not has_killed, (
        f"cell {cancer_id} ({gene} {mut_name}) was T-cell-killed without "
        "any displayed peptide; closed-loop invariant violated"
    )
    return "hla_restricted"


def main() -> int:
    banner("Upgrade 2 Closed-Loop Demo  --  USC 101 patent evidence")
    print("Pipeline: MolecularModule -> CellularModule -> ImmuneModule")
    print("Trace:    MUTATION_OCCURRED -> PEPTIDE_GENERATED -> "
          "PEPTIDE_PRESENTED -> IMMUNE_ACTIVATED -> CELL_KILLED_BY_TCELL")

    # Reproducible per-step kill rolls inside immune.update.
    np.random.seed(0)

    engine = _build_engine()
    molecular: MolecularModule = engine.modules["molecular"]
    cellular: CellularModule = engine.modules["cellular"]
    immune: ImmuneModule = engine.modules["immune"]
    cellular.set_molecular_module(molecular)
    immune.set_cellular_module(cellular)

    # ---- Spawn 3 mutated cancer cells + 1 wild-type control ---------------
    cancer_ids_by_mut: Dict[Tuple[str, str], int] = {}
    for i, (gene, mut_name, _) in enumerate(MUTATIONS):
        cancer_ids_by_mut[(gene, mut_name)] = _spawn_cancer_cell(
            engine,
            position=[100.0 + 20.0 * i, 100.0, 50.0],
            cell_index=i,
        )

    # Wild-type control: cancer cell with no mutation, T cell colocated.
    wt_id = _spawn_cancer_cell(
        engine, position=[100.0 + 20.0 * 3, 100.0, 50.0], cell_index=3,
    )

    banner("Driving mutations")
    for gene, mut_name, expected_aa in MUTATIONS:
        cid = cancer_ids_by_mut[(gene, mut_name)]
        mut = molecular.introduce_mutation(cid, gene, mut_name)
        assert mut is not None, f"introduce_mutation failed for {gene} {mut_name}"
        assert mut.effect.aa_change == expected_aa, (
            f"aa_change drift: {gene} {mut_name} -> {mut.effect.aa_change} "
            f"(expected {expected_aa})"
        )
        print(f"  cell {cid:>3}  {gene:<5} {mut_name:<6}  "
              f"category={mut.effect.category:<10} "
              f"impact={mut.effect.impact_score:.2f}  "
              f"aa_change={mut.effect.aa_change}  "
              f"WT residue: {mut.effect.wild_type_aa} "
              f"mutant: {mut.effect.mutant_aa}")

    banner("Run engine: 5 steps, T cells colocated with targets")
    engine.run(duration=0.05)

    # ---- Per-mutation trace + provenance --------------------------------
    log = list(engine.event_bus.event_log)
    outcomes: Dict[Tuple[str, str], str] = {}
    for gene, mut_name, _ in MUTATIONS:
        cid = cancer_ids_by_mut[(gene, mut_name)]
        banner(f"Cell {cid}: {gene} {mut_name} -- closed-loop trace")
        print("  Displayed peptides at end of run:")
        _print_displayed_peptides(cellular, cid)
        print()
        print("  Event trace (causal order):")
        _print_trace_for_cell(log, cid)
        label = _assert_chain_consistent(log, cid, gene, mut_name)
        outcomes[(gene, mut_name)] = label
        if label == "full":
            print(f"  CLASSIFICATION: full closed loop (claim A) -- "
                  f"presented + killed via TCR")
        else:
            print(f"  CLASSIFICATION: HLA-restricted (claim B) -- "
                  f"peptides generated but none bind HLA-A*02:01 / "
                  f"A*24:02 / B*07:02; pipeline correctly predicts no "
                  f"T-cell-mediated response in this patient")

    # ---- Negative control: WT cancer cell should NOT be T-cell-killed ----
    banner(f"Cell {wt_id}: wild-type control (no mutation introduced)")
    wt_events = _events_for_cell(log, wt_id)
    wt_types = [evt for evt, _ in wt_events]
    print(f"  events on this cell: {wt_types or '(none)'}")
    assert EventTypes.PEPTIDE_GENERATED not in wt_types, (
        "WT cell unexpectedly generated peptides -- no mutation was driven"
    )
    assert EventTypes.CELL_KILLED_BY_TCELL not in wt_types, (
        "WT cell was T-cell-killed; closed loop should not fire without "
        "a displayed neoantigen"
    )
    print("  PASS: closed loop did not fire for the wild-type cell.")

    # ---- Summary --------------------------------------------------------
    banner("Summary")
    full = [k for k, v in outcomes.items() if v == "full"]
    restricted = [k for k, v in outcomes.items() if v == "hla_restricted"]
    print(f"  full closed loop (claim A):  "
          f"{', '.join(f'{g} {m}' for g, m in full) if full else '(none)'}")
    print(f"  HLA-restricted (claim B):    "
          f"{', '.join(f'{g} {m}' for g, m in restricted) if restricted else '(none)'}")
    print(f"  total peptides generated:    {cellular.total_peptides_generated}")
    print(f"  total peptides presented:    {cellular.total_peptides_presented}")
    print(f"  total TCR recognitions:      {immune.total_tcr_recognitions}")
    print(f"  total T-cell kills:          {immune.total_tcell_kills}")
    print(f"  total NK / macrophage kills: "
          f"{immune.total_kills - immune.total_tcell_kills}")

    # At least one full-loop case is required for the demo to be
    # evidence of claim A; the demo's KRAS G12D path produces this.
    assert full, (
        "no full closed-loop case observed -- the demo must show at least "
        "one mutation that drives the entire MUTATION -> ... -> "
        "CELL_KILLED_BY_TCELL chain. Check the HLA panel covers KRAS G12D."
    )

    banner("Demo complete -- Upgrade 2 closed-loop trace verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
