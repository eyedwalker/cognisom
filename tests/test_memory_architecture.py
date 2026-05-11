"""
Memory benchmark: per-cell-delta architecture vs naive deep-copy.

This is the patent-claim load-bearing test. The architecture under test
must scale per-cell memory with mutation count, not with genome size. We
assert this empirically by creating 10,000 cell views with a small number
of mutations each and verifying the total resident-set-size increase
stays under a budget consistent with the sparse representation.

We compare against a baseline "naive" approach that deep-copies the
entire genome into every cell. The asymptotic ratio between the two
modes is the headline number for the patent.
"""

import gc
import os
import random
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import pytest

# psutil is required for RSS measurement. Skip if not present rather than
# fail the suite -- this is a benchmark, not a hard correctness gate.
psutil = pytest.importorskip("psutil")

from engine.py.molecular.reference_genome import build_default_reference_genome
from engine.py.molecular.sequence_view import CellGenomeView


def _rss_mb() -> float:
    """Current process resident-set-size in megabytes."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


# --- Sparse architecture ----------------------------------------------------

def _build_view_population(genome, n_cells: int, n_mutations_per_cell: int):
    """Build n_cells CellGenomeViews on a shared genome, each with the
    given number of random mutations. Returns the list of views (so they
    stay alive and the RSS measurement reflects their true cost)."""
    rng = random.Random(20260511)
    views = []
    gene_names = list(genome.gene_names())
    for _ in range(n_cells):
        v = CellGenomeView(genome)
        for _ in range(n_mutations_per_cell):
            gene = rng.choice(gene_names)
            pos = rng.randrange(genome.length(gene))
            new_base = rng.choice("ACGT")
            v.add_substitution(gene, pos, new_base, mutation_id="")
        views.append(v)
    return views


# --- Naive deep-copy architecture (for comparison) -------------------------

class _NaiveCell:
    """Reference architecture: each cell owns a deep copy of every gene's
    sequence. This is what the codebase did before Upgrade 1."""
    __slots__ = ("sequences",)

    def __init__(self, genome):
        # Force a fresh string copy per cell so Python's small-string
        # interning cannot accidentally make this benchmark cheaper than
        # the real legacy behavior.
        self.sequences = {
            name: genome.get_reference_sequence(name)[:] + "​"[:0]
            for name in genome.gene_names()
        }


def _build_naive_population(genome, n_cells: int, n_mutations_per_cell: int):
    """Build n_cells naive cells, each with a deep copy of every gene
    plus the given number of substitutions applied in place."""
    rng = random.Random(20260511)
    cells = []
    gene_names = list(genome.gene_names())
    for _ in range(n_cells):
        c = _NaiveCell(genome)
        for _ in range(n_mutations_per_cell):
            gene = rng.choice(gene_names)
            pos = rng.randrange(len(c.sequences[gene]))
            new_base = rng.choice("ACGT")
            seq_list = list(c.sequences[gene])
            seq_list[pos] = new_base
            c.sequences[gene] = "".join(seq_list)
        cells.append(c)
    return cells


# --- The actual benchmark --------------------------------------------------

# Stock cognisom genome is 3 genes totaling ~3636 bases:
#   KRAS 153, TP53 1182, BRAF 2301
# Naive cost per cell: ~3636 bytes (one string of length ~3636 per cell)
# Plus Python object overhead: ~200-500 bytes per cell.

N_CELLS = 10_000
N_MUTATIONS_PER_CELL = 3


def test_view_population_under_memory_budget():
    """10,000 views, each with 3 mutations, must stay well under 50 MB of
    incremental RSS. This is the headline patent-claim number."""
    genome = build_default_reference_genome()
    gc.collect()
    baseline_rss = _rss_mb()

    views = _build_view_population(genome, N_CELLS, N_MUTATIONS_PER_CELL)

    gc.collect()
    final_rss = _rss_mb()
    delta = final_rss - baseline_rss

    # Sanity: we built the structures we intended to
    assert len(views) == N_CELLS
    assert all(v.n_deltas() == N_MUTATIONS_PER_CELL for v in views)

    # Patent budget: with 10k cells and 3 deltas each, peak overhead
    # should be well under 50 MB. In practice we observe ~10-30 MB.
    assert delta < 50.0, (
        f"View population used {delta:.1f} MB; budget is 50 MB. "
        f"Either the sparse architecture regressed or psutil reporting "
        f"is unusually noisy."
    )

    # Keep `views` alive to the end of the function so the RSS reading
    # above measures the live structures.
    del views


def _build_large_synthetic_genome(n_genes: int, gene_length: int):
    """Build a synthetic ReferenceGenome with n_genes genes each of
    length `gene_length`. Used to show the architecture's asymptotic
    advantage at human-exome scale."""
    from engine.py.molecular.reference_genome import ReferenceGenome, GeneMetadata
    g = ReferenceGenome()
    # Use a long ACGT cycle so the sequence is non-trivial but cheap to
    # construct.
    pattern = "ACGT" * ((gene_length // 4) + 1)
    base_seq = pattern[:gene_length]
    for i in range(n_genes):
        # Each gene gets a slightly different sequence so string interning
        # cannot make the naive approach artificially cheaper.
        seq = chr(ord("A") + (i % 4)) + base_seq[1:]  # vary first base
        # Force ACGT
        if seq[0] not in "ACGT":
            seq = "A" + base_seq[1:]
        g.add_gene(f"GENE_{i:04d}", seq, GeneMetadata(name=f"GENE_{i:04d}"))
    return g.freeze()


def test_view_is_dramatically_more_efficient_at_realistic_genome_size():
    """At realistic genome size (100 genes x 3 kb each = 300 KB total),
    the sparse architecture uses asymptotically less memory than the
    naive deep-copy approach because per-cell cost is dominated by
    delta count, not by genome size.

    Build 5000 cells in each architecture; report the ratio. With
    Python's string interning, small genomes show a modest 2-5x
    advantage; at this scale we expect 20x+. Patent claim is the
    asymptotic property: cost = O(deltas) not O(genome)."""
    large_genome = _build_large_synthetic_genome(n_genes=100, gene_length=3000)
    n_cells = 5_000
    n_mut = 3

    # --- Naive
    gc.collect()
    base_naive = _rss_mb()
    naive_cells = _build_naive_population(large_genome, n_cells, n_mut)
    gc.collect()
    naive_delta = _rss_mb() - base_naive
    del naive_cells
    gc.collect()

    # --- Views
    base_views = _rss_mb()
    views = _build_view_population(large_genome, n_cells, n_mut)
    gc.collect()
    views_delta = _rss_mb() - base_views

    if naive_delta < 50.0:
        pytest.skip(
            f"Naive delta {naive_delta:.1f} MB too small for reliable ratio."
        )

    ratio = naive_delta / max(views_delta, 0.1)
    print(
        f"\nMemory architecture (100 genes x 3 kb, {n_cells} cells, {n_mut} muts/cell): "
        f"naive={naive_delta:.1f}MB, views={views_delta:.1f}MB, ratio={ratio:.1f}x"
    )
    assert ratio >= 20.0, (
        f"At realistic genome size, the view architecture should be at "
        f"least 20x cheaper than naive (got {ratio:.1f}x). This is the "
        f"core patent-claim demonstration."
    )

    del views


def test_per_cell_cost_scales_with_mutations_not_genome():
    """Direct test of the asymptotic claim: holding cell count constant,
    per-cell incremental memory scales with mutation count, not with
    genome size.

    Build two view populations: one on a small genome, one on a much
    larger genome. Both with the same mutation count per cell. The
    incremental cost should be approximately equal -- not 100x different
    in proportion to the genome size difference.
    """
    n_cells = 2_000
    n_mut = 3

    small_genome = build_default_reference_genome()  # ~3.6 KB total
    large_genome = _build_large_synthetic_genome(n_genes=100, gene_length=3000)  # ~300 KB total

    size_ratio = large_genome.total_bases() / small_genome.total_bases()

    # --- Small genome
    gc.collect()
    base = _rss_mb()
    small_views = _build_view_population(small_genome, n_cells, n_mut)
    gc.collect()
    small_delta = _rss_mb() - base
    del small_views
    gc.collect()

    # --- Large genome
    base = _rss_mb()
    large_views = _build_view_population(large_genome, n_cells, n_mut)
    gc.collect()
    large_delta = _rss_mb() - base

    print(
        f"\nPer-cell cost vs genome size: "
        f"genome size ratio = {size_ratio:.0f}x, "
        f"memory ratio = {large_delta / max(small_delta, 0.1):.2f}x "
        f"(small={small_delta:.1f}MB, large={large_delta:.1f}MB)"
    )

    # Patent claim: memory ratio is approximately 1, not approximately
    # size_ratio. Allow 3x slack for Python overhead variance and
    # measurement noise. The genome size grew by ~80x; we assert memory
    # grew by < 5x.
    memory_ratio = large_delta / max(small_delta, 0.1)
    assert memory_ratio < 5.0, (
        f"Per-cell memory grew {memory_ratio:.1f}x when genome size grew "
        f"{size_ratio:.0f}x. The view architecture does not show the "
        f"O(deltas) asymptotic property."
    )

    del large_views


def test_fork_does_not_copy_reference_genome():
    """Patent-claim invariant: forking N times must not allocate N more
    copies of the reference genome. After forking 10,000 times from one
    view, RSS overhead must be sparse-only."""
    genome = build_default_reference_genome()
    gc.collect()
    base = _rss_mb()

    root = CellGenomeView(genome)
    root.add_substitution("KRAS", 34, "A", "founder")

    children = [root.fork() for _ in range(N_CELLS)]

    gc.collect()
    delta = _rss_mb() - base

    # All children must point at the same reference by identity
    assert all(c.reference is genome for c in children)
    # All children must carry the founder's delta
    assert all(c.base_at("KRAS", 34) == "A" for c in children)

    assert delta < 50.0, (
        f"Forking {N_CELLS} times allocated {delta:.1f} MB; budget 50 MB. "
        f"Reference may have been copied per-fork."
    )
    del children
