"""
Memory benchmark: per-cell-delta architecture vs naive deep-copy.

This is the patent-claim load-bearing test: per-cell memory must scale
with mutation count, not with genome size.

Measured with `tracemalloc`, not resident-set size. RSS was the wrong
instrument and made this file unreliable in two directions at once. It
reports the whole process, so it moves with allocator retention, heap
fragmentation and GC timing rather than with the structures under test --
freeing the naive population did not return its pages to the OS, so the
views measured afterwards allocated into that reclaimed heap and read
arbitrarily low. The headline ratio came out anywhere from 16x to over
20x run to run, and the assertion sat at `>= 20.0`, so a green run was
partly luck. Worse, when the noise fell the other way the test called
`pytest.skip`, and a skipped benchmark silently proves nothing.

tracemalloc counts Python allocations attributable to the objects built,
so the numbers are reproducible to four decimal places across runs and
machines.

The naive baseline also has to genuinely copy. It previously tried with
`genome.get_reference_sequence(name)[:] + ""[:0]`, but CPython returns
the *same object* for both `s[:]` and `s + ""` -- so the "deep copy"
shared storage with the reference and the baseline was measuring dict
overhead rather than duplicated sequence. That understated legacy cost by
roughly 25x, which is why a genuinely large advantage kept failing a 20x
threshold. Copying for real puts the ratio near 290x at 100 genes.
"""

import gc
import random
import sys
import tracemalloc
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from cognisom.engine.py.molecular.reference_genome import build_default_reference_genome
from cognisom.engine.py.molecular.sequence_view import CellGenomeView


def _live_mb(build) -> float:
    """Megabytes allocated by `build()` and still live when it returns.

    Deliberately measures *live* bytes rather than peak: the naive builder
    churns transient lists while applying substitutions, and that garbage is
    not part of what either architecture costs to hold. Collecting before
    reading drops it, leaving the retained footprint of the structures.

    No dependency on psutil, and no dependency on how the allocator happens
    to feel about returning pages to the OS.
    """
    gc.collect()
    tracemalloc.start()
    try:
        obj = build()
        gc.collect()
        current, _peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    del obj
    gc.collect()
    return current / (1024 * 1024)


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

def _force_copy(seq: str) -> str:
    """Return a string equal to `seq` but backed by its own storage.

    `seq[:]` and `seq + ""` both return `seq` itself in CPython, so the
    obvious spellings of "copy this string" duplicate nothing. Appending a
    character and slicing it back off allocates for real: the concatenation
    builds a new object, and the slice is partial so it cannot return its
    input either.
    """
    return (seq + "X")[:-1]


class _NaiveCell:
    """Reference architecture: each cell owns a deep copy of every gene's
    sequence. This is what the codebase did before Upgrade 1."""
    __slots__ = ("sequences",)

    def __init__(self, genome):
        # A real per-cell copy. The previous spelling shared storage with
        # the reference genome, so this baseline measured dict overhead
        # instead of duplicated sequence and understated legacy cost ~25x.
        self.sequences = {
            name: _force_copy(genome.get_reference_sequence(name))
            for name in genome.gene_names()
        }

        # Guard the property this whole baseline rests on. If a future
        # CPython optimises the copy away, this fails loudly rather than
        # quietly turning the comparison back into a no-op.
        if genome.gene_names():
            first = next(iter(genome.gene_names()))
            assert self.sequences[first] is not genome.get_reference_sequence(first), (
                "naive baseline is sharing storage with the reference genome; "
                "it is not measuring deep-copy cost"
            )


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

# Stock cognisom genome is 3 genes totaling 4050 bases (authentic CDSes):
#   KRAS 567 (NM_004985.5), TP53 1182 (NM_000546.6), BRAF 2301 (NM_004333.6)
# Naive cost per cell: ~4050 bytes (one string of length ~4050 per cell)
# Plus Python object overhead: ~200-500 bytes per cell.

N_CELLS = 10_000
N_MUTATIONS_PER_CELL = 3


def test_view_population_under_memory_budget():
    """10,000 views with 3 mutations each must stay inside a tight budget."""
    genome = build_default_reference_genome()

    # Correctness of what we built is checked on its own population, so the
    # measured build holds nothing but the views themselves.
    sample = _build_view_population(genome, 100, N_MUTATIONS_PER_CELL)
    assert len(sample) == 100
    assert all(v.n_deltas() == N_MUTATIONS_PER_CELL for v in sample)
    del sample

    used = _live_mb(lambda: _build_view_population(genome, N_CELLS, N_MUTATIONS_PER_CELL))

    # Measures ~9.96 MB and is reproducible to four decimals, so the budget
    # is 2x observed rather than the old 5x -- a real regression now has to
    # clear a bar it can actually hit, instead of hiding under noise slack.
    assert used < 20.0, (
        f"View population used {used:.2f} MB; budget is 20 MB "
        f"(observed ~10 MB). The sparse architecture regressed."
    )


def _build_large_synthetic_genome(n_genes: int, gene_length: int):
    """Build a synthetic ReferenceGenome with n_genes genes each of
    length `gene_length`. Used to show the architecture's asymptotic
    advantage at human-exome scale."""
    from cognisom.engine.py.molecular.reference_genome import ReferenceGenome, GeneMetadata
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
    """The sparse architecture must cost dramatically less than deep copies,
    and the gap must *widen* with genome size -- that is the asymptotic claim.

    Asserting one magic ratio was never meaningful on its own, because the
    ratio is a function of genome size: the naive cost grows with it while
    the view cost does not. So this pins both the size of the advantage and
    its direction of travel.
    """
    n_cells = 300
    n_mut = 3

    small = _build_large_synthetic_genome(n_genes=50, gene_length=3000)
    large = _build_large_synthetic_genome(n_genes=100, gene_length=3000)

    naive_small = _live_mb(lambda: _build_naive_population(small, n_cells, n_mut))
    views_small = _live_mb(lambda: _build_view_population(small, n_cells, n_mut))
    naive_large = _live_mb(lambda: _build_naive_population(large, n_cells, n_mut))
    views_large = _live_mb(lambda: _build_view_population(large, n_cells, n_mut))

    ratio_small = naive_small / views_small
    ratio_large = naive_large / views_large
    print(
        f"\n{n_cells} cells, {n_mut} muts | "
        f"50 genes: naive={naive_small:.1f}MB views={views_small:.2f}MB ({ratio_small:.0f}x) | "
        f"100 genes: naive={naive_large:.1f}MB views={views_large:.2f}MB ({ratio_large:.0f}x)"
    )

    # Observed ~145x and ~290x. The floor sits far below both: this guards
    # against the architecture regressing, not against measurement drift,
    # because with tracemalloc there is no drift to guard against.
    assert ratio_large >= 100.0, (
        f"View architecture is only {ratio_large:.0f}x cheaper than naive at "
        f"100 genes; expected >= 100x. This is the core patent-claim number."
    )

    # Doubling the genome must roughly double the naive cost...
    naive_growth = naive_large / naive_small
    assert 1.7 <= naive_growth <= 2.3, (
        f"Naive cost grew {naive_growth:.2f}x when the genome doubled; "
        f"expected ~2x. The baseline is not tracking genome size, so the "
        f"comparison is not measuring what it claims."
    )

    # ...while leaving the view cost alone. This is the O(deltas) property.
    views_growth = views_large / views_small
    assert views_growth < 1.2, (
        f"View cost grew {views_growth:.2f}x when the genome doubled; "
        f"per-cell cost must scale with delta count, not genome size."
    )

    # And therefore the advantage widens rather than plateauing.
    assert ratio_large > ratio_small, (
        f"Advantage did not grow with genome size ({ratio_small:.0f}x -> "
        f"{ratio_large:.0f}x); the asymptotic claim does not hold."
    )


def test_per_cell_cost_scales_with_mutations_not_genome():
    """Direct test of the asymptotic claim: holding cell count constant,
    per-cell memory scales with mutation count, not with genome size.
    """
    n_cells = 2_000
    n_mut = 3

    small_genome = build_default_reference_genome()                          # ~4 KB
    large_genome = _build_large_synthetic_genome(n_genes=100, gene_length=3000)  # ~300 KB

    size_ratio = large_genome.total_bases() / small_genome.total_bases()

    small_delta = _live_mb(lambda: _build_view_population(small_genome, n_cells, n_mut))
    large_delta = _live_mb(lambda: _build_view_population(large_genome, n_cells, n_mut))
    memory_ratio = large_delta / small_delta

    print(
        f"\nPer-cell cost vs genome size: genome grew {size_ratio:.0f}x, "
        f"memory grew {memory_ratio:.3f}x "
        f"(small={small_delta:.2f}MB, large={large_delta:.2f}MB)"
    )

    # Observed 1.014x against a 74x genome. The old bound was 5.0, chosen to
    # survive RSS noise; with a deterministic instrument it can sit where the
    # claim actually lives. Anything approaching size_ratio means per-cell
    # cost has started tracking the genome.
    assert memory_ratio < 1.2, (
        f"Per-cell memory grew {memory_ratio:.2f}x when genome size grew "
        f"{size_ratio:.0f}x. The view architecture does not show the "
        f"O(deltas) asymptotic property."
    )


def test_fork_does_not_copy_reference_genome():
    """Patent-claim invariant: forking N times must not allocate N more
    copies of the reference genome."""
    genome = build_default_reference_genome()

    def fork_population():
        root = CellGenomeView(genome)
        root.add_substitution("KRAS", 34, "A", "founder")
        return [root.fork() for _ in range(N_CELLS)]

    # Identity and inheritance are correctness claims, checked separately so
    # the measured build holds only the forks.
    children = fork_population()
    assert all(c.reference is genome for c in children)
    assert all(c.base_at("KRAS", 34) == "A" for c in children)
    del children

    used = _live_mb(fork_population)

    # Observed ~3.97 MB for 10,000 forks. A per-fork genome copy would cost
    # roughly 10,000 x 4 KB = 40 MB, so this bound separates the two clearly.
    assert used < 10.0, (
        f"Forking {N_CELLS} times allocated {used:.2f} MB; budget 10 MB "
        f"(observed ~4 MB). Reference may have been copied per-fork."
    )
