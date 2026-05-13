"""
End-to-end VCF round-trip test (patent-evidence anchor).

Closes the pre-filing gap "we never actually read DNA in any test".
The closed-loop test at tests/test_closed_loop_neoantigen.py drives
the pipeline via direct ``molecular.introduce_mutation(cell_id, gene,
mutation_label)`` calls -- it skips the VCF ingest stage. This test
starts from raw VCF text (the synthetic prostate-cancer corpus at
cognisom/genomics/synthetic_vcf.py), parses it via the production
VCFParser, filters down to variants on curated genes (KRAS, BRAF,
TP53), drives the simulation for each, and asserts:

  1. Provenance flows VCF -> Variant.protein_change -> MutationEffect
     -> CELL_KILLED_BY_TCELL.source_gene + .mutation. The kill event
     in the engine event log can be traced back to the originating
     VCF row.

  2. At least one VCF-derived mutation drives the full closed loop
     (sensitivity claim). With the demo HLA panel covering A*02:01,
     the KRAS G12V row at chr12:25398284 is the natural candidate;
     other rows (TP53 R175H / R248W, BRAF V600E) typically come back
     as HLA-restricted, which is also valid patent evidence.

  3. Non-coding / off-target VCF rows (intronic, intergenic, genes
     not in the curated library) cause no spurious peptide events.
     This is the specificity counterpart to claim 2.

Together (1) + (2) + (3) demonstrate that cognisom's molecular layer
correctly bridges from the standard VCF interchange format (the
output of any production variant caller, GATK / DeepVariant /
Mutect2 / Parabricks) into the simulation -- without requiring the
caller to know about cognisom-specific gene/mutation tuples.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

# Silence keras/tensorflow chatter from MHCflurry transitive imports.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.genomics.synthetic_vcf import SYNTHETIC_PROSTATE_VCF
from cognisom.genomics.vcf_parser import VCFParser, Variant
from core import SimulationConfig, SimulationEngine
from core.event_bus import EventTypes
from engine.py.molecular.nucleic_acids import Gene
from modules.cellular_module import CellularModule
from modules.immune_module import ImmuneModule
from modules.molecular_module import MolecularModule


# Genes the curated reference genome knows about. Variants outside
# this set are not drivable through the simulation (silent skip).
SUPPORTED_GENES = ("KRAS", "TP53", "BRAF")


# ---------------------------------------------------------------------------
# VCF -> driver-tuple extraction
# ---------------------------------------------------------------------------

_AA_CHANGE_PATTERN = re.compile(r"p\.([A-Z])(\d+)([A-Z*])")


def _aa_change_to_mutation_label(protein_change: str) -> Optional[str]:
    """Convert 'p.G12V' -> 'G12V'. Returns None for unparseable or
    non-missense protein changes (e.g., frameshifts, '*' stops)."""
    if not protein_change:
        return None
    m = _AA_CHANGE_PATTERN.match(protein_change)
    if not m:
        return None
    wt, pos, mut = m.group(1), m.group(2), m.group(3)
    if mut == "*":
        return None  # nonsense; molecular_module curates missense hotspots only
    return f"{wt}{pos}{mut}"


def extract_drivable_mutations(
    variants: List[Variant],
) -> List[Tuple[Variant, str, str]]:
    """Filter VCF variants to those drivable through the simulation.

    Returns a list of (variant, gene, mutation_label) tuples where:
      - variant.gene is in SUPPORTED_GENES
      - mutation_label can be parsed from variant.protein_change
      - the (gene, mutation_label) is in Gene.ONCOGENIC_SUBSTITUTIONS
        (i.e., the curated hotspot table the molecular module honors)

    Variants that fail any check are silently dropped. This mirrors
    how a clinical pipeline would behave -- a VCF carries thousands
    of variants; only a curated subset drives the simulation.
    """
    out: List[Tuple[Variant, str, str]] = []
    for v in variants:
        if v.gene not in SUPPORTED_GENES:
            continue
        label = _aa_change_to_mutation_label(v.protein_change or "")
        if label is None:
            continue
        gene_table = Gene.ONCOGENIC_SUBSTITUTIONS.get(v.gene, {})
        if label not in gene_table:
            continue
        out.append((v, v.gene, label))
    return out


# ---------------------------------------------------------------------------
# Engine fixture
# ---------------------------------------------------------------------------

def _build_engine(n_drivers: int) -> SimulationEngine:
    """Three-module engine sized to one cancer cell + one T cell per
    drivable mutation. Tuning matches test_closed_loop_neoantigen.py
    so the trace is deterministic and the kill probability stays high
    once affinity x mhc clears the EC50."""
    engine = SimulationEngine(SimulationConfig(
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
        "n_t_cells": max(n_drivers, 1),
        "n_nk_cells": 0,
        "n_macrophages": 0,
        "tcr_recognition_threshold": 0.0,
        "tcr_repertoire_size": 500,
        "tcr_seed": 0,
        "costimulation": 1.0,
        "checkpoint_block": 0.0,
    })
    engine.initialize()
    return engine


# ---------------------------------------------------------------------------
# VCF parsing
# ---------------------------------------------------------------------------

def test_vcf_parses_and_carries_protein_change():
    """Sanity check on the VCF parser: it must populate variant.gene
    and variant.protein_change for the prostate-cancer corpus."""
    parser = VCFParser()
    variants = parser.parse_text(SYNTHETIC_PROSTATE_VCF)

    assert len(variants) >= 40, (
        f"synthetic VCF parsed to only {len(variants)} variants; "
        "expected ~50"
    )

    # Every supported-gene variant must carry a protein_change so the
    # downstream pipeline has something to anchor on.
    for v in variants:
        if v.gene in SUPPORTED_GENES:
            assert v.protein_change is not None, (
                f"VCF row for {v.gene} at {v.chrom}:{v.pos} has no "
                "protein_change annotation; cognisom cannot drive the "
                "simulation without one"
            )


def test_vcf_extract_drivable_mutations_finds_canonical_hotspots():
    """Extraction must surface KRAS G12V, TP53 R175H, TP53 R248W,
    and BRAF V600E from the corpus -- these are the four canonical
    hotspots that exist in both the VCF and the curated
    ONCOGENIC_SUBSTITUTIONS table."""
    parser = VCFParser()
    variants = parser.parse_text(SYNTHETIC_PROSTATE_VCF)
    drivable = extract_drivable_mutations(variants)
    labels = {(gene, label) for _, gene, label in drivable}

    for expected in [
        ("KRAS", "G12V"),
        ("TP53", "R175H"),
        ("TP53", "R248W"),
        ("BRAF", "V600E"),
    ]:
        assert expected in labels, (
            f"drivable mutation {expected} missing from VCF extraction; "
            f"got {labels}"
        )


def test_vcf_non_coding_rows_are_silently_dropped():
    """Intronic / intergenic VCF rows must not surface as drivable.
    They have no .gene field and their consequence is non-coding."""
    parser = VCFParser()
    variants = parser.parse_text(SYNTHETIC_PROSTATE_VCF)
    drivable = extract_drivable_mutations(variants)
    drivable_chroms = {v.chrom: v.pos for v, _, _ in drivable}

    # The intronic / intergenic rows live on chr1 27100000, chr2 48010000,
    # chr3 120000000, etc. None of them should make it into the
    # drivable set -- they have no GENE annotation in the VCF.
    for v in variants:
        if v.consequence in ("intronic", "intergenic"):
            assert (v.chrom, v.pos) not in [
                (vv.chrom, vv.pos) for vv, _, _ in drivable
            ]


# ---------------------------------------------------------------------------
# End-to-end: VCF -> simulation -> kill event with provenance
# ---------------------------------------------------------------------------

def test_vcf_round_trip_drives_closed_loop_with_provenance():
    """Patent-evidence anchor: a real VCF, parsed and pushed through
    the cognisom pipeline, produces at least one CELL_KILLED_BY_TCELL
    event whose provenance fields recover the originating VCF row's
    gene and mutation label."""
    np.random.seed(0)

    parser = VCFParser()
    variants = parser.parse_text(SYNTHETIC_PROSTATE_VCF)
    drivable = extract_drivable_mutations(variants)
    assert drivable, "no drivable mutations extracted from synthetic VCF"

    engine = _build_engine(n_drivers=len(drivable))
    molecular: MolecularModule = engine.modules["molecular"]
    cellular: CellularModule = engine.modules["cellular"]
    immune: ImmuneModule = engine.modules["immune"]
    cellular.set_molecular_module(molecular)
    immune.set_cellular_module(cellular)

    # Spawn one cancer cell per drivable mutation, colocate one T
    # cell on each, then drive each mutation through the molecular
    # module. Track the VCF row that originated each cell so the
    # kill-event provenance can be cross-checked.
    cell_to_vcf: Dict[int, Tuple[str, int, str, str]] = {}
    t_ids = list(immune.immune_cells.keys())
    for i, (variant, gene, label) in enumerate(drivable):
        position = [100.0 + 20.0 * i, 100.0, 50.0]
        cancer_id = cellular.add_cell(position=position, cell_type="cancer")
        cellular.cells[cancer_id].mhc1_expression = 0.9
        molecular.add_cell(cancer_id)
        # Colocate one T cell with each target.
        if i < len(t_ids):
            immune.immune_cells[t_ids[i]].position = np.array(
                position, dtype=np.float32
            )
        cell_to_vcf[cancer_id] = (variant.chrom, variant.pos, gene, label)
        mut = molecular.introduce_mutation(cancer_id, gene, label)
        assert mut is not None, (
            f"molecular.introduce_mutation failed for {gene} {label} "
            f"(VCF row {variant.chrom}:{variant.pos})"
        )

    engine.run(duration=0.05)

    log = list(engine.event_bus.event_log)
    kills = [
        data for evt, data in log
        if evt == EventTypes.CELL_KILLED_BY_TCELL
    ]
    assert kills, (
        "no CELL_KILLED_BY_TCELL event in log; the VCF round-trip "
        "should drive at least one mutation through the full closed "
        "loop (KRAS G12V on HLA-A*02:01 is the expected positive "
        "case under this HLA panel)"
    )

    # For every kill, the provenance must match the VCF row that
    # originated the cancer cell.
    for kill in kills:
        cancer_id = kill["cell_id"]
        assert cancer_id in cell_to_vcf, (
            f"kill event references cell_id {cancer_id} not in the "
            f"VCF-to-cell map; provenance broken"
        )
        chrom, pos, vcf_gene, vcf_label = cell_to_vcf[cancer_id]
        assert kill["source_gene"] == vcf_gene, (
            f"kill provenance source_gene={kill['source_gene']!r} "
            f"disagrees with VCF row {chrom}:{pos} gene={vcf_gene!r}"
        )
        assert kill["mutation"] == vcf_label, (
            f"kill provenance mutation={kill['mutation']!r} "
            f"disagrees with VCF row {chrom}:{pos} mutation={vcf_label!r}"
        )


def test_vcf_with_no_curated_drivers_produces_no_kills():
    """Negative control: a VCF whose driver mutations all map to genes
    outside SUPPORTED_GENES (or are not in the curated hotspot table)
    must produce zero peptide events. This is the specificity
    counterpart to the main round-trip test -- ensures we are not
    silently fabricating neoantigens for unsupported genes."""
    np.random.seed(0)

    # Strip the supported-gene rows out of the synthetic VCF and keep
    # only the unrelated ones (AR, PTEN, BRCA2, etc.). Quickest way:
    # parse, filter, regenerate a minimal VCF text.
    parser = VCFParser()
    variants = parser.parse_text(SYNTHETIC_PROSTATE_VCF)
    unsupported = [v for v in variants if v.gene not in SUPPORTED_GENES]
    assert unsupported, "synthetic VCF unexpectedly has no unsupported rows"

    drivable = extract_drivable_mutations(unsupported)
    # By construction this list must be empty.
    assert drivable == [], (
        f"unsupported VCF rows leaked into drivable set: "
        f"{[(g, l) for _, g, l in drivable]}"
    )

    # Build the engine, attempt to drive a few "mutations" that the
    # molecular module does not curate, and assert no peptide events.
    engine = _build_engine(n_drivers=2)
    molecular: MolecularModule = engine.modules["molecular"]
    cellular: CellularModule = engine.modules["cellular"]
    immune: ImmuneModule = engine.modules["immune"]
    cellular.set_molecular_module(molecular)
    immune.set_cellular_module(cellular)

    for i, v in enumerate(unsupported[:2]):
        cancer_id = cellular.add_cell(
            position=[100.0 + 20.0 * i, 100.0, 50.0],
            cell_type="cancer",
        )
        cellular.cells[cancer_id].mhc1_expression = 0.9
        molecular.add_cell(cancer_id)
        # introduce_mutation on a non-curated (gene, label) returns None
        label = _aa_change_to_mutation_label(v.protein_change or "")
        result = molecular.introduce_mutation(cancer_id, v.gene or "", label or "")
        assert result is None, (
            f"molecular.introduce_mutation accepted unsupported "
            f"({v.gene}, {label}); the curated table should reject it"
        )

    engine.run(duration=0.05)

    log = list(engine.event_bus.event_log)
    types = [evt for evt, _ in log]
    assert EventTypes.PEPTIDE_GENERATED not in types, (
        "PEPTIDE_GENERATED fired for an unsupported VCF row; the "
        "pipeline must not fabricate neoantigens for genes outside "
        "the curated reference"
    )
    assert EventTypes.CELL_KILLED_BY_TCELL not in types, (
        "CELL_KILLED_BY_TCELL fired without a preceding mutation event"
    )
