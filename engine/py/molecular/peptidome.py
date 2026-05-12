"""
Peptidome: protein -> peptide pool
==================================

Generates the pool of short peptides (8-11mers) that a cell could
potentially display on MHC-I given a protein sequence. Two modes:

    1. Full sliding window across an entire protein (self-peptide pool).
    2. Mutation-anchored window around a specific amino-acid substitution
       (neoantigen pool -- the primary patent-evidence path).

A *simple* proteasomal-cleavage score is attached to each peptide so
downstream code can rank-filter peptides before passing them to the MHC
loader. The scoring captures only the most widely-replicated cleavage
preferences (C-terminal hydrophobic / aromatic residues are favored;
basic residues at P1 are penalized). NetChop / MHC-flurry's processing
predictor remain the right production-grade upgrade -- see
UPGRADES_SPEC.md, Section 2, step 1 ("NetChop integration deferred").

This module is intentionally framework-free -- no dependence on
SimulationModule, event bus, or cell state. The cellular module is the
caller that bridges per-cell protein state into peptide generation.

Patent claim surface (Upgrade 2): this is the first stage of the
closed-loop neoantigen presentation pipeline. Peptide objects carry
provenance (source gene, mutation label, mutation position within the
peptide) so the downstream PEPTIDE_PRESENTED and CELL_KILLED_BY_TCELL
events can be traced back to the originating MUTATION_OCCURRED event.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple


# Default peptide-length panel for MHC-I (most production models score
# 8- through 11-mers; 9-mers dominate empirically).
DEFAULT_LENGTHS: Tuple[int, ...] = (8, 9, 10, 11)


# Simple proteasomal-cleavage preferences. These are weights applied at
# the C-terminus (P1) of the peptide, which is the cleavage site the
# proteasome carves out for MHC-I display. Numbers are not from NetChop;
# they are the qualitative trend reported in Kessler / Toes reviews:
#   * hydrophobic / aromatic at P1  -> favored
#   * positively-charged at P1      -> disfavored (chymotryptic activity)
#   * acidic / small at P1          -> neutral
_P1_PREFERRED = set("LIVFYWM")    # favored hydrophobic / aromatic
_P1_DISFAVORED = set("KR")        # basic -- disfavored at P1


@dataclass(frozen=True)
class Peptide:
    """A short peptide candidate for MHC-I display.

    sequence
        The peptide amino-acid sequence (single-letter, length 8-11).
    source_gene
        Gene of origin (e.g., "KRAS").
    length
        Convenience copy of len(sequence) -- avoids hot-path len() calls
        during MHC scoring loops.
    is_mutant
        True if this peptide carries a non-reference amino acid at the
        anchor position. Self-peptides have is_mutant=False.
    wild_type_sequence
        The corresponding wild-type peptide at the same window. Equal to
        ``sequence`` when is_mutant=False; differs at exactly one
        position when is_mutant=True.
    mutation_label
        Human-readable mutation identifier (e.g., "G12D"). None for
        self-peptides.
    anchor_position_in_peptide
        0-indexed position of the mutated residue within the peptide.
        -1 for self-peptides (no anchor).
    parent_position_1based
        1-indexed start position of this peptide within the parent
        protein. Useful for downstream visualization / provenance.
    cleavage_score
        Score in [0, 1] reflecting how plausible the proteasome would
        cut at the C-terminus of this peptide. Higher = more likely.
    """

    sequence: str
    source_gene: str
    length: int
    is_mutant: bool
    wild_type_sequence: str
    mutation_label: Optional[str]
    anchor_position_in_peptide: int
    parent_position_1based: int
    cleavage_score: float

    def __post_init__(self) -> None:
        if len(self.sequence) != self.length:
            raise ValueError(
                f"Peptide.length={self.length} disagrees with "
                f"len(sequence)={len(self.sequence)}"
            )
        if self.is_mutant and self.mutation_label is None:
            raise ValueError("mutant peptide must carry a mutation_label")
        if not self.is_mutant and self.sequence != self.wild_type_sequence:
            raise ValueError(
                "self-peptide must have sequence == wild_type_sequence"
            )

    @property
    def is_self(self) -> bool:
        return not self.is_mutant


# ---------------------------------------------------------------------------
# Cleavage scoring
# ---------------------------------------------------------------------------

def proteasomal_cleavage_score(c_terminal_aa: str) -> float:
    """Score the proteasome's preference for cutting after this residue.

    Returns a value in [0, 1]. Used to rank peptides; combined with MHC
    binding affinity downstream. A "production-grade" replacement is
    NetChop or MHCflurry's processing predictor; this implementation is
    explicit about its simplicity (see module docstring).
    """
    if c_terminal_aa in _P1_PREFERRED:
        return 0.9
    if c_terminal_aa in _P1_DISFAVORED:
        return 0.2
    return 0.5


# ---------------------------------------------------------------------------
# Public generators
# ---------------------------------------------------------------------------

def generate_peptides(
    protein_sequence: str,
    source_gene: str,
    lengths: Iterable[int] = DEFAULT_LENGTHS,
    min_cleavage_score: float = 0.0,
) -> List[Peptide]:
    """Generate all sliding-window peptides from a full protein.

    Self-peptides only. is_mutant=False on every output.
    """
    if not protein_sequence:
        return []
    out: List[Peptide] = []
    seq = protein_sequence
    n = len(seq)
    for length in lengths:
        if length <= 0 or length > n:
            continue
        for start in range(0, n - length + 1):
            pep = seq[start:start + length]
            # Reject stop codons mid-peptide -- those are truncation
            # artifacts, not displayable peptides.
            if "*" in pep:
                continue
            cleav = proteasomal_cleavage_score(pep[-1])
            if cleav < min_cleavage_score:
                continue
            out.append(Peptide(
                sequence=pep,
                source_gene=source_gene,
                length=length,
                is_mutant=False,
                wild_type_sequence=pep,
                mutation_label=None,
                anchor_position_in_peptide=-1,
                parent_position_1based=start + 1,
                cleavage_score=cleav,
            ))
    return out


def generate_neoantigen_peptides(
    wild_type_protein: str,
    mutant_position_1based: int,
    wild_type_aa: str,
    mutant_aa: str,
    source_gene: str,
    mutation_label: str,
    lengths: Iterable[int] = DEFAULT_LENGTHS,
    min_cleavage_score: float = 0.0,
) -> List[Peptide]:
    """Generate mutant peptides covering a single missense substitution.

    For each window of each requested length that spans the mutation
    position, emit the (mutant, wild-type) peptide pair as a single
    Peptide record with is_mutant=True. The mutant peptide carries the
    substituted residue; the wild_type_sequence field carries the
    unsubstituted counterpart for downstream agretopicity scoring.

    The wild-type sequence is checked at the mutation site so callers
    cannot silently desync wild_type_aa from the actual protein. If the
    declared wild_type_aa does not match the protein at this position a
    ValueError is raised -- this is a regression net for misuse from
    the cellular module.
    """
    if not wild_type_protein:
        return []
    idx = mutant_position_1based - 1  # to 0-based
    n = len(wild_type_protein)
    if idx < 0 or idx >= n:
        raise IndexError(
            f"mutant_position_1based={mutant_position_1based} out of range "
            f"for protein of length {n}"
        )
    if wild_type_protein[idx] != wild_type_aa:
        raise ValueError(
            f"wild_type_aa='{wild_type_aa}' does not match protein at "
            f"position {mutant_position_1based} (found "
            f"'{wild_type_protein[idx]}'); peptidome refuses to silently "
            "fabricate a neoantigen against a mis-specified reference."
        )

    out: List[Peptide] = []
    for length in lengths:
        if length <= 0 or length > n:
            continue
        # Windows that contain idx: start in [max(0, idx-length+1), idx]
        first_start = max(0, idx - length + 1)
        last_start = min(idx, n - length)
        for start in range(first_start, last_start + 1):
            wt_window = wild_type_protein[start:start + length]
            if "*" in wt_window:
                continue
            anchor = idx - start
            mut_window_list = list(wt_window)
            mut_window_list[anchor] = mutant_aa
            mut_window = "".join(mut_window_list)
            if "*" in mut_window:
                # Mutation introduced a stop -- handled by nonsense path,
                # not the MHC-I missense path.
                continue
            cleav = proteasomal_cleavage_score(mut_window[-1])
            if cleav < min_cleavage_score:
                continue
            out.append(Peptide(
                sequence=mut_window,
                source_gene=source_gene,
                length=length,
                is_mutant=True,
                wild_type_sequence=wt_window,
                mutation_label=mutation_label,
                anchor_position_in_peptide=anchor,
                parent_position_1based=start + 1,
                cleavage_score=cleav,
            ))
    return out


def top_k_by_cleavage(
    peptides: Sequence[Peptide], k: int
) -> List[Peptide]:
    """Return the k peptides with highest cleavage score, ties broken by
    preferring shorter peptides (smaller search space for MHC scoring).
    """
    if k <= 0 or not peptides:
        return []
    ordered = sorted(peptides, key=lambda p: (-p.cleavage_score, p.length))
    return ordered[:k]
