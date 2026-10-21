"""How much of each protein a cell contains, measured.

The library knew what molecules a cell carries and not how many. That
is the gap between a diagram and a cell: what makes a Goodsell
illustration read as alive is DENSITY, and cytoplasm is roughly 300
mg/mL of protein -- effectively solid -- against a few hundred features
in a mostly empty sphere.

Kim et al., Nature 2014 sorted peripheral blood into B cells, CD4 T
cells, CD8 T cells, monocytes and NK cells and measured each. Those are
five of this library's own lineages, and they cover 148 of the 216
entities that carry a gene symbol. PaxDb is CC BY 4.0.

ppm, NOT COPIES
---------------
Each file sums to 1,000,000, which is how you can tell it is a mole
fraction. Copies per cell would be ppm times the total protein molecules
in that cell, and this library cannot cite that total for any of its
cell types -- so the ppm is stored as what it is rather than multiplied
by a number from memory.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.library import paxdb_sourcing as px        # noqa: E402


@pytest.fixture(scope="module")
def store():
    with contextlib.redirect_stdout(io.StringIO()):
        from cognisom.library.store import EntityStore
        return EntityStore()


def _file(tmp_path, rows, organ="NK_CELL"):
    """A PaxDb-shaped abundance file. rows are (gene, ppm, count)."""
    lines = [f"#name: test", f"#organ: {organ}", "#coverage: 42",
             "#", "#gene_name\tstring_external_id\tabundance\traw_spectral_count"]
    for i, (gene, ppm, count) in enumerate(rows, 1):
        lines.append(f"{gene}\t9606.ENSP{i:08d}\t{ppm}\t{count}")
    path = tmp_path / "9606-test.txt"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


# ── parsing ──────────────────────────────────────────────────────────

def test_the_header_and_the_table_are_both_read(tmp_path):
    path = _file(tmp_path, [("PRF1", 2907, 1307), ("GZMA", 1474, 600)])
    dataset = px.read_dataset(path)
    assert dataset.organ == "NK_CELL"
    assert dataset.coverage == 42
    assert dataset.ppm["PRF1"] == 2907
    assert len(dataset.ppm) == 2


def test_a_gene_under_several_ids_keeps_its_largest(tmp_path):
    """A gene appears under several ENSPs; the dominant isoform is the
    one worth having."""
    path = _file(tmp_path, [("PRF1", 100, 40), ("PRF1", 2907, 1307)])
    assert px.read_dataset(path).ppm["PRF1"] == 2907


# ── the shared peptide pool ──────────────────────────────────────────

def test_an_identical_spectral_count_marks_a_shared_pool(tmp_path):
    """Spectral counting assigns a peptide to every protein it could
    have come from, so a family 90% identical gets one measurement
    reported several times. ACTB and ACTG1 both report 11,313 spectra."""
    path = _file(tmp_path, [("ACTB", 13827, 11313), ("ACTG1", 13827, 11313),
                            ("PRF1", 2907, 1307)])
    dataset = px.read_dataset(path)
    assert dataset.shared["ACTB"] == ("ACTG1",)
    assert dataset.shared["ACTG1"] == ("ACTB",)
    assert "PRF1" not in dataset.shared


def test_a_ppm_tie_alone_is_not_evidence(tmp_path):
    """ppm ties are mostly coincidence -- the values are coarse, and 92%
    of the proteins under 5 ppm share a value with something. PCBP1 and
    ACTA2 both land on 1,533 ppm from 381 and 4,505 spectra."""
    path = _file(tmp_path, [("PCBP1", 1533, 381), ("ACTA2", 1533, 4505)])
    assert px.read_dataset(path).shared == {}


def test_a_low_count_tie_is_granularity_not_a_family(tmp_path):
    """PTEN is measured on 2 spectra in CD8 cells and 1,273 other
    proteins share that count. None of them is a PTEN paralogue."""
    rows = [("PTEN", 3.03, 2)] + [(f"GENE{i}", 3.03, 2) for i in range(20)]
    assert px.read_dataset(_file(tmp_path, rows)).shared == {}
    assert px.MIN_SHARED_SPECTRA >= 50


def test_the_rule_is_every_dataset_not_most_of_them(store):
    """A shared pool is a property of the SEQUENCES, so it holds
    wherever both were seen. Measured over the five: 56,126 pairs tie in
    exactly one dataset, 450 in two, 42 in three, 38 in four and 428 in
    all five -- and the all-five group is ACTB/ACTG1, the calmodulins,
    the histone clusters."""
    assert px.REQUIRE_TIE_IN_EVERY_DATASET is True


def test_none_of_our_molecules_is_in_a_shared_pool(store):
    """A true negative, not a missing feature: this library holds CD
    markers, cytokines and signalling proteins, none of which is a
    member of a near-identical family. The detector fires on ACTB; it
    has nothing to say about CD8A."""
    flagged = []
    for blob, in store._conn.execute("SELECT data_json FROM entities"):
        props = json.loads(blob).get("properties") or {}
        if props.get("abundance_shared_with"):
            flagged.append(json.loads(blob)["name"])
    assert not flagged, flagged


# ── what the live pass stored ────────────────────────────────────────

def test_the_library_knows_how_much_of_each_protein(store):
    stored = 0
    for blob, in store._conn.execute("SELECT data_json FROM entities"):
        if (json.loads(blob).get("properties") or {}).get("abundance_ppm"):
            stored += 1
    assert stored >= 140, f"only {stored} molecules carry an abundance"


def test_an_abundance_is_per_cell_type_not_per_protein(store):
    """Perforin is 2,907 ppm in an NK cell and 39 in a B cell. It cannot
    be a scalar the way a molecular weight is."""
    entity = store.find_entity_by_name("Perforin (PRF1)")
    ppm = (entity.sourced or {}).get("abundance_ppm") or {}
    assert ppm.get("NK cell") == pytest.approx(2907, abs=1)
    assert ppm.get("CD8+ T cell", 0) > ppm.get("CD4+ T cell", 0)
    assert ppm.get("Naive B cell", 1e9) < ppm.get("NK cell")


def test_a_subset_value_says_which_bulk_sort_it_came_from(store):
    """A Th17's entry is the bulk CD4 measurement. That is an
    assumption, and it is recorded as one rather than hidden."""
    entity = store.find_entity_by_name("Perforin (PRF1)")
    where = (entity.sourced or {}).get("abundance_measured_in") or {}
    assert where.get("CD4+ Th17 cell") == "bulk CD4+ T cells"
    assert where.get("NK cell") == "sorted NK cells"


def test_it_does_not_claim_copies_per_cell(store):
    """The total protein molecules in a lymphocyte is not a number this
    library can cite, so the ppm is not multiplied by one."""
    entity = store.find_entity_by_name("Perforin (PRF1)")
    note = (entity.sourced or {}).get("abundance_note") or ""
    assert "Not copies per cell" in note
    assert "cannot cite" in note


def test_the_attribution_is_stored(store):
    for blob, in store._conn.execute("SELECT data_json FROM entities"):
        props = json.loads(blob).get("properties") or {}
        if props.get("abundance_ppm"):
            assert "PaxDb" in (props.get("abundance_source") or "")
            assert "CC BY 4.0" in props["abundance_source"]


def test_the_most_abundant_are_in_the_right_cells(store):
    """The check that the mapping is not scrambled: MPO and CD14 should
    peak in monocytes, perforin and granzyme in NK cells."""
    def peak(name):
        entity = store.find_entity_by_name(name)
        ppm = (entity.sourced or {}).get("abundance_ppm") or {}
        return max(ppm.items(), key=lambda kv: kv[1])[0] if ppm else None

    assert peak("Myeloperoxidase (MPO)") in ("Monocyte", "Macrophage",
                                             "Macrophage M1", "Macrophage M2")
    assert peak("LPS co-receptor (CD14)") in ("Monocyte", "Macrophage",
                                              "Macrophage M1", "Macrophage M2")
    assert peak("Perforin (PRF1)") in ("NK cell", "NKT cell")
    assert peak("Granzyme A (GZMA)") in ("NK cell", "NKT cell")
