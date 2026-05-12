"""
Integration tests for MolecularModule using the new CellGenomeView
architecture (Sprint 2b).

These tests prove that the wiring between MolecularModule and the
ReferenceGenome / CellGenomeView pair preserves the observable behaviors
the cancer-transmission demo relies on:

  - Adding cells produces sparse-state cells (views, not deep Gene copies)
  - Introducing a named oncogenic mutation writes a delta + flags the cell
  - Cell division forks the view (daughter inherits, then diverges)
  - Transcription materializes the cell's actual mutated sequence
  - Exosomes carry the per-cell sequence (not the reference)
"""

import sys
import warnings
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import pytest

from modules.molecular_module import MolecularModule
from engine.py.molecular.sequence_view import CellGenomeView
from engine.py.molecular.reference_genome import ReferenceGenome


@pytest.fixture
def module():
    m = MolecularModule(config={})
    m.initialize()
    return m


# --- shape ------------------------------------------------------------------

def test_module_has_reference_genome_after_initialize(module):
    """Initialize() must populate self.reference_genome."""
    assert isinstance(module.reference_genome, ReferenceGenome)
    assert module.reference_genome.is_frozen
    assert set(module.reference_genome.gene_names()) == {"KRAS", "TP53", "BRAF"}


def test_module_no_longer_has_cell_genes_attribute(module):
    """The legacy per-cell Gene copy dict is gone."""
    assert not hasattr(module, "cell_genes")


def test_module_uses_cell_views(module):
    """Per-cell state is in module.cell_views, not in deep Gene copies."""
    module.add_cell(42)
    assert 42 in module.cell_views
    assert isinstance(module.cell_views[42], CellGenomeView)


def test_cells_share_reference_by_identity(module):
    """All cell views point at the SAME ReferenceGenome instance.
    This is the memory-architecture patent claim invariant."""
    module.add_cell(0)
    module.add_cell(1)
    module.add_cell(2)
    refs = {id(v.reference) for v in module.cell_views.values()}
    assert len(refs) == 1
    assert next(iter(refs)) == id(module.reference_genome)


# --- mutation introduction --------------------------------------------------

def test_introduce_mutation_writes_delta_to_view(module):
    """introduce_mutation must write a sparse delta to the cell's view."""
    module.add_cell(7)
    mutation = module.introduce_mutation(7, "KRAS", "G12D")
    assert mutation is not None
    view = module.cell_views[7]
    assert view.n_deltas() == 1
    delta = view.deltas()[0]
    assert delta.gene_name == "KRAS"
    assert delta.position == 34
    assert delta.new_base == "A"


def test_introduce_mutation_classifies_and_flags_oncogene(module):
    """The classifier-derived impact score determines the oncogene flag,
    not external belief."""
    module.add_cell(7)
    mutation = module.introduce_mutation(7, "KRAS", "G12D")
    assert mutation.oncogenic is True
    assert mutation.effect is not None
    assert mutation.effect.category == "missense"
    assert mutation.effect.aa_change == "G12D"
    assert "KRAS" in module.cell_oncogene_flags[7]


def test_introduce_mutation_isolates_to_target_cell(module):
    """Mutating cell 7 must not affect cell 8."""
    module.add_cell(7)
    module.add_cell(8)
    module.introduce_mutation(7, "KRAS", "G12D")
    assert module.cell_views[7].n_deltas() == 1
    assert module.cell_views[8].n_deltas() == 0
    assert "KRAS" in module.cell_oncogene_flags[7]
    assert "KRAS" not in module.cell_oncogene_flags[8]


def test_introduce_mutation_unknown_returns_none(module):
    module.add_cell(7)
    assert module.introduce_mutation(7, "KRAS", "NOT_A_MUTATION") is None


def test_introduce_mutation_unknown_gene_returns_none(module):
    module.add_cell(7)
    assert module.introduce_mutation(7, "NOT_A_GENE", "G12D") is None


def test_introduce_mutation_unknown_cell_returns_none(module):
    assert module.introduce_mutation(999, "KRAS", "G12D") is None


# --- cell division -----------------------------------------------------------

def test_on_cell_divided_forks_view(module):
    """Daughter inherits parent's deltas via fork()."""
    module.add_cell(1)
    module.introduce_mutation(1, "KRAS", "G12D")
    # Simulate division event
    module.on_cell_divided({'cell_id': 1, 'daughter_id': 2})
    # Daughter exists and carries the parent's delta
    assert 2 in module.cell_views
    assert module.cell_views[2].n_deltas() == 1
    assert module.cell_views[2].base_at("KRAS", 34) == "A"
    # Daughter inherits oncogene flags
    assert "KRAS" in module.cell_oncogene_flags[2]


def test_on_cell_divided_independent_after_fork(module):
    """After division, mutations on parent do not affect daughter and
    vice versa."""
    module.add_cell(1)
    module.introduce_mutation(1, "KRAS", "G12D")
    module.on_cell_divided({'cell_id': 1, 'daughter_id': 2})
    # Mutate parent further
    module.introduce_mutation(1, "TP53", "R175H")
    assert "TP53" in module.cell_oncogene_flags[1]
    assert "TP53" not in module.cell_oncogene_flags[2]
    # Both daughter and parent retain the KRAS delta
    assert module.cell_views[1].base_at("KRAS", 34) == "A"
    assert module.cell_views[2].base_at("KRAS", 34) == "A"
    # But TP53 R175H is only in parent
    assert module.cell_views[1].base_at("TP53", 523) == "A"
    assert module.cell_views[2].base_at("TP53", 523) == module.reference_genome.get_reference_base("TP53", 523)


def test_on_cell_divided_shares_reference_by_identity(module):
    """Patent-claim invariant: fork() shares ReferenceGenome by identity."""
    module.add_cell(1)
    module.on_cell_divided({'cell_id': 1, 'daughter_id': 2})
    assert module.cell_views[1].reference is module.cell_views[2].reference


# --- death -----------------------------------------------------------------

def test_on_cell_died_cleans_up_all_per_cell_state(module):
    module.add_cell(7)
    module.introduce_mutation(7, "KRAS", "G12D")
    module.on_cell_died({'cell_id': 7})
    assert 7 not in module.cell_views
    assert 7 not in module.cell_oncogene_flags
    assert 7 not in module.cell_mrnas


# --- exosomes --------------------------------------------------------------

def test_oncogenic_exosome_packages_mutated_mrna(module):
    """An oncogenic exosome must carry mRNA whose sequence reflects the
    cell's mutations (materialized from the view), not the reference."""
    module.add_cell(1)
    module.introduce_mutation(1, "KRAS", "G12D")
    exosome = module.create_exosome(1, oncogenic=True)
    assert exosome is not None
    # The exosome's KRAS mRNA must carry the G12D substitution. In the
    # mRNA the corresponding position is the U at codon 12 middle base
    # which is 'A' after G12D (after T->U transcription).
    kras_mrnas = [m for m in exosome.cargo.mrnas if m.from_gene == "KRAS"]
    assert kras_mrnas, "oncogenic exosome from a KRAS-G12D cell must carry KRAS mRNA"
    mrna = kras_mrnas[0]
    # mRNA position 34 should be 'A' (post-G12D substitution; mRNA U->A
    # doesn't change since the substituted base is A in DNA too)
    assert mrna.sequence[34] == "A"
    # And the mRNA mutations log should record G12D
    assert any(m.name == "G12D" for m in mrna.mutations)


def test_normal_exosome_from_non_mutated_cell_carries_reference_mrna(module):
    """A non-oncogenic exosome from a non-mutated cell carries reference mRNA."""
    module.add_cell(1)
    # No mutation. To get an mRNA in the cell pool we need a transcription
    # event; the create_exosome(oncogenic=False) path reads from cell_mrnas.
    # Easiest: directly poke a transcription via the helper.
    view = module.cell_views[1]
    mrna = module._build_mrna_from_view(view, "KRAS", 1)
    module.cell_mrnas[1].append(mrna)
    exosome = module.create_exosome(1, oncogenic=False)
    assert exosome is not None
    # Packaged mRNA must reflect the reference sequence (no deltas)
    packaged = exosome.cargo.mrnas[0]
    expected = module.reference_genome.get_reference_sequence("KRAS").replace("T", "U")
    assert packaged.sequence == expected


# --- get_state -------------------------------------------------------------

def test_get_state_reports_view_population(module):
    module.add_cell(1)
    module.add_cell(2)
    module.add_cell(3)
    state = module.get_state()
    assert state['n_cells_tracked'] == 3
    assert state['n_genes'] == 3
    assert state['total_mutations'] == 0


def test_get_state_after_mutation_increments_counter(module):
    module.add_cell(1)
    module.introduce_mutation(1, "KRAS", "G12D")
    state = module.get_state()
    assert state['total_mutations'] == 1
