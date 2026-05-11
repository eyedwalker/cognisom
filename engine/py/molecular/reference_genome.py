"""
Reference genome: the shared, immutable substrate for per-cell sparse genomic
state in the agent-based simulator.

This module implements the read-only side of the memory architecture
described in UPGRADES_SPEC.md (Upgrade 1). A single ReferenceGenome
instance holds the canonical sequence for each gene plus its metadata.
Every cell object in the simulation holds a CellGenomeView (see
sequence_view.py) that overlays a sparse list of substitutions on this
shared reference, so per-cell memory usage scales with mutation count
rather than with genome size.

This is the patent claim anchor for the memory-architecture invention:

  A method for memory-efficient agent-based simulation of cellular
  populations carrying genomic sequence state, comprising maintaining a
  canonical reference genome in a single shared memory region accessible
  to all cell objects, representing each cell's deviation from said
  reference as a sparse list of substitution records...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# SubstitutionDelta - the per-cell record of a single-base divergence
# from the reference. Plural deltas per cell make up the cell's genotype.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubstitutionDelta:
    """One sparse divergence from the reference at a specific base.

    Attributes
    ----------
    gene_name : str
        Which gene (must exist in the ReferenceGenome).
    position : int
        0-indexed base position within the gene's reference sequence.
    new_base : str
        The base this cell carries at this position ('A'/'C'/'G'/'T').
    mutation_id : str
        Provenance identifier for tracking. Multiple deltas may share a
        mutation_id (e.g., multi-base mutations would be N deltas with one
        shared id). Free-form; the simulator typically uses a short UUID.
    """
    gene_name: str
    position: int
    new_base: str
    mutation_id: str = ""

    def __post_init__(self) -> None:
        if len(self.new_base) != 1:
            raise ValueError(f"new_base must be a single character, got {self.new_base!r}")
        if self.new_base.upper() not in "ACGTU":
            raise ValueError(f"new_base must be A/C/G/T/U, got {self.new_base!r}")
        if self.position < 0:
            raise ValueError(f"position must be >= 0, got {self.position}")


# ---------------------------------------------------------------------------
# GeneMetadata - regulatory and biological annotations that are shared
# across all cells (the per-cell variable state lives in deltas).
# ---------------------------------------------------------------------------

@dataclass
class GeneMetadata:
    """Per-gene annotations shared by all cells.

    Note: variability that differs per cell (e.g., "is_oncogene after
    mutation") does NOT live here. Only annotations that are properties
    of the canonical reference itself.
    """
    name: str
    gene_type: str = "protein_coding"
    cds_start: int = 0
    cds_end: Optional[int] = None  # one-past-the-end; None means end of sequence
    is_baseline_oncogene: bool = False
    is_baseline_tumor_suppressor: bool = False
    transcription_rate: float = 1.0
    promoter_strength: float = 1.0
    # Add fields as needed; this struct is intentionally flat.


# ---------------------------------------------------------------------------
# ReferenceGenome - the canonical, immutable shared substrate.
# ---------------------------------------------------------------------------

class ReferenceGenome:
    """Immutable registry of canonical gene sequences and metadata.

    Construction is two-phase: instantiate, then add_gene() each gene,
    then call freeze() to lock the object. After freeze(), add_gene()
    raises. This prevents accidental mutation during simulation.

    Sequences are stored as plain strings (uppercase ACGT). The choice of
    str over bytes is deliberate: the codebase elsewhere uses str, and
    Python string slicing is fast for the read patterns this class
    supports. Migration to bytes is a future optimization, not a
    correctness concern.
    """

    def __init__(self) -> None:
        self._sequences: Dict[str, str] = {}
        self._metadata: Dict[str, GeneMetadata] = {}
        self._frozen: bool = False

    def add_gene(
        self,
        name: str,
        sequence: str,
        metadata: Optional[GeneMetadata] = None,
    ) -> None:
        """Add a gene to the reference. Raises if the genome is frozen or
        if the gene already exists. Sequence is normalized to uppercase
        and validated to contain only ACGT(U)."""
        if self._frozen:
            raise RuntimeError("ReferenceGenome is frozen; cannot add genes")
        if name in self._sequences:
            raise ValueError(f"gene {name!r} already registered")
        seq = sequence.upper().replace("U", "T")
        bad = set(seq) - set("ACGT")
        if bad:
            raise ValueError(f"gene {name!r} sequence contains non-ACGT chars: {bad}")
        if not seq:
            raise ValueError(f"gene {name!r} sequence is empty")
        self._sequences[name] = seq
        self._metadata[name] = metadata or GeneMetadata(name=name)

    def freeze(self) -> "ReferenceGenome":
        """Lock the genome. Returns self for chaining."""
        self._frozen = True
        return self

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    # -- read access -------------------------------------------------------

    def has_gene(self, name: str) -> bool:
        return name in self._sequences

    def gene_names(self) -> Iterable[str]:
        return tuple(self._sequences.keys())

    def get_reference_sequence(self, name: str) -> str:
        """Return the full reference sequence for a gene.

        Use sparingly during simulation - this returns the whole string.
        For per-base lookups, use get_reference_base() instead."""
        try:
            return self._sequences[name]
        except KeyError:
            raise KeyError(f"gene {name!r} not in reference genome") from None

    def get_reference_base(self, name: str, position: int) -> str:
        """Return the single base at the given 0-indexed position. O(1)."""
        try:
            seq = self._sequences[name]
        except KeyError:
            raise KeyError(f"gene {name!r} not in reference genome") from None
        if not (0 <= position < len(seq)):
            raise IndexError(
                f"position {position} out of range for {name!r} "
                f"(length {len(seq)})"
            )
        return seq[position]

    def length(self, name: str) -> int:
        try:
            return len(self._sequences[name])
        except KeyError:
            raise KeyError(f"gene {name!r} not in reference genome") from None

    def metadata(self, name: str) -> GeneMetadata:
        try:
            return self._metadata[name]
        except KeyError:
            raise KeyError(f"gene {name!r} not in reference genome") from None

    # -- introspection -----------------------------------------------------

    def total_bases(self) -> int:
        """Sum of sequence lengths across all genes. Useful for memory
        accounting in tests."""
        return sum(len(s) for s in self._sequences.values())

    def __len__(self) -> int:
        return len(self._sequences)

    def __repr__(self) -> str:
        state = "frozen" if self._frozen else "mutable"
        return (
            f"ReferenceGenome({len(self)} genes, "
            f"{self.total_bases()} total bases, {state})"
        )


# ---------------------------------------------------------------------------
# Convenience builder for cognisom's stock gene library.
# ---------------------------------------------------------------------------

def build_default_reference_genome() -> ReferenceGenome:
    """Build a ReferenceGenome populated with the cognisom stock gene
    library (KRAS, TP53, BRAF) from reference_cds.py. Returns a frozen
    instance ready for use."""
    from engine.py.molecular.reference_cds import (
        KRAS_CDS, TP53_CDS, BRAF_CDS,
    )
    g = ReferenceGenome()
    g.add_gene("KRAS", KRAS_CDS, GeneMetadata(
        name="KRAS",
        gene_type="protein_coding",
        is_baseline_oncogene=False,
    ))
    g.add_gene("TP53", TP53_CDS, GeneMetadata(
        name="TP53",
        gene_type="protein_coding",
        is_baseline_tumor_suppressor=True,
    ))
    g.add_gene("BRAF", BRAF_CDS, GeneMetadata(
        name="BRAF",
        gene_type="protein_coding",
        is_baseline_oncogene=False,
    ))
    return g.freeze()
