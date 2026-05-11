"""
Per-cell sparse view over a shared ReferenceGenome.

A CellGenomeView is the per-cell data structure for the memory-architecture
patent claim (UPGRADES_SPEC.md Upgrade 1). It holds a reference (by
pointer, not copy) to a ReferenceGenome plus a per-cell sparse list of
SubstitutionDelta records. Per-base reads consult the delta index first,
then fall back to the reference.

Memory characteristics:
  - A reference is shared across all cells (no per-cell genome copy).
  - Per-cell overhead = (small object header) + delta index size, where
    the delta index size scales with number of mutations the cell has
    accumulated (typically << genome size).
  - Daughter-cell creation via fork() copies only the delta index, not
    the reference.

This module's correctness invariant: for any sequence of substitutions
applied to a view, the values returned by base_at(g, p) are identical to
the values you would get by deep-copying the reference sequence and
applying the same substitutions in order. Tests enforce this invariant.
"""

from __future__ import annotations

import itertools
from typing import Dict, Iterator, List, Optional, Tuple

from engine.py.molecular.reference_genome import (
    ReferenceGenome,
    SubstitutionDelta,
)


class CellGenomeView:
    """Per-cell view of the genome: shared reference + private deltas.

    The reference is held by-reference (pointer). Multiple views share
    the same ReferenceGenome instance; modifying one view's deltas does
    not affect another view.

    Construction takes an optional starting delta list (typically empty;
    nonzero only when reconstituting from serialized state).
    """

    __slots__ = ("_ref", "_delta_index", "_delta_log")

    def __init__(
        self,
        reference: ReferenceGenome,
        initial_deltas: Optional[List[SubstitutionDelta]] = None,
    ) -> None:
        if not reference.is_frozen:
            raise RuntimeError(
                "CellGenomeView requires a frozen ReferenceGenome. Call "
                "reference.freeze() before constructing views."
            )
        self._ref: ReferenceGenome = reference
        # Sparse index for O(1) per-base lookup. Keyed by (gene, position).
        self._delta_index: Dict[Tuple[str, int], SubstitutionDelta] = {}
        # Chronological log for provenance and daughter inheritance.
        self._delta_log: List[SubstitutionDelta] = []
        if initial_deltas:
            for d in initial_deltas:
                self._add_internal(d)

    # -- per-base read ------------------------------------------------------

    def base_at(self, gene_name: str, position: int) -> str:
        """Return the base this cell carries at the specified position.

        Returns the cell's delta if one exists at (gene, position),
        otherwise the reference base. O(1).
        """
        d = self._delta_index.get((gene_name, position))
        if d is not None:
            return d.new_base
        return self._ref.get_reference_base(gene_name, position)

    def codon_at(self, gene_name: str, codon_index_0based: int) -> str:
        """Return the 3-base codon at the given 0-indexed codon position
        within the gene's CDS. Applies deltas on the fly."""
        pos = codon_index_0based * 3
        return (
            self.base_at(gene_name, pos)
            + self.base_at(gene_name, pos + 1)
            + self.base_at(gene_name, pos + 2)
        )

    def iter_codons(
        self,
        gene_name: str,
        start: int = 0,
        end: Optional[int] = None,
    ) -> Iterator[str]:
        """Yield successive 3-base codons starting at base position `start`
        through `end` (exclusive). If `end` is None, runs to end of gene.

        Applies deltas as it iterates. Does NOT materialize the full
        sequence in memory.
        """
        length = self._ref.length(gene_name)
        if end is None:
            end = length
        end = min(end, length)
        for i in range(start, end - 2, 3):
            yield (
                self.base_at(gene_name, i)
                + self.base_at(gene_name, i + 1)
                + self.base_at(gene_name, i + 2)
            )

    def materialize(self, gene_name: str) -> str:
        """ESCAPE HATCH: return the full per-cell sequence as a string.

        This allocates a full copy of the sequence with deltas applied.
        Avoid in hot loops; use base_at() / iter_codons() instead. Use
        materialize() only when handing the sequence to an external
        library that requires a full string (e.g., a protein language
        model like ESM-2).
        """
        ref = self._ref.get_reference_sequence(gene_name)
        # Fast path: no deltas for this gene -> return reference directly.
        if not any(g == gene_name for (g, _) in self._delta_index):
            return ref
        # Slow path: apply deltas.
        chars = list(ref)
        for (g, pos), delta in self._delta_index.items():
            if g == gene_name:
                chars[pos] = delta.new_base
        return "".join(chars)

    # -- write -------------------------------------------------------------

    def add_substitution(
        self,
        gene_name: str,
        position: int,
        new_base: str,
        mutation_id: str = "",
    ) -> SubstitutionDelta:
        """Add a delta. Overwrites any existing delta at the same
        (gene_name, position). Returns the SubstitutionDelta added."""
        delta = SubstitutionDelta(
            gene_name=gene_name,
            position=position,
            new_base=new_base.upper().replace("U", "T"),
            mutation_id=mutation_id,
        )
        # Validate gene exists and position is in range BEFORE recording
        # to avoid corrupting the log on bad input.
        ref_len = self._ref.length(gene_name)  # raises if gene unknown
        if not (0 <= position < ref_len):
            raise IndexError(
                f"position {position} out of range for {gene_name!r} "
                f"(length {ref_len})"
            )
        self._add_internal(delta)
        return delta

    def _add_internal(self, delta: SubstitutionDelta) -> None:
        self._delta_index[(delta.gene_name, delta.position)] = delta
        self._delta_log.append(delta)

    # -- daughter creation -------------------------------------------------

    def fork(self) -> "CellGenomeView":
        """Create a daughter view that shares the reference and has a
        copy of this view's deltas.

        Subsequent mutations on either parent or daughter do NOT affect
        the other. This is the operation invoked at simulated cell
        division and is the core of the memory-architecture patent claim.
        """
        child = CellGenomeView(self._ref)
        # Copy the log; rebuild index from log to preserve chronology
        # while also having O(1) lookup.
        for d in self._delta_log:
            child._delta_index[(d.gene_name, d.position)] = d
        child._delta_log = list(self._delta_log)
        return child

    # -- provenance --------------------------------------------------------

    def deltas(self) -> Tuple[SubstitutionDelta, ...]:
        """Return all deltas this cell carries, in chronological order
        (insertion order). Useful for debugging, serialization, and
        mutation-history tracking."""
        return tuple(self._delta_log)

    def deltas_for_gene(self, gene_name: str) -> Tuple[SubstitutionDelta, ...]:
        return tuple(d for d in self._delta_log if d.gene_name == gene_name)

    def n_deltas(self) -> int:
        return len(self._delta_log)

    # -- introspection -----------------------------------------------------

    @property
    def reference(self) -> ReferenceGenome:
        """The shared ReferenceGenome backing this view. Same Python
        identity across all sibling views."""
        return self._ref

    def __repr__(self) -> str:
        return (
            f"CellGenomeView(ref id=0x{id(self._ref):x}, "
            f"n_deltas={self.n_deltas()})"
        )


# ---------------------------------------------------------------------------
# Equivalence helper for regression tests: compute the per-cell sequence
# the "naive" way (deep-copy reference and apply deltas in order) so tests
# can assert that the view's read API gives the same answer.
# ---------------------------------------------------------------------------

def materialize_naive(
    reference: ReferenceGenome,
    gene_name: str,
    deltas: List[SubstitutionDelta],
) -> str:
    """Reference implementation: deep-copy the reference sequence for the
    given gene and apply deltas in chronological order. Used by tests as
    the oracle against which the CellGenomeView's behavior is checked.
    """
    seq_list = list(reference.get_reference_sequence(gene_name))
    for d in deltas:
        if d.gene_name == gene_name:
            seq_list[d.position] = d.new_base
    return "".join(seq_list)
