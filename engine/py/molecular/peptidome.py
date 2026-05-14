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
    # Upgrade 8: provenance of the mutation that produced this peptide.
    # "missense"   -- single-residue substitution (Upgrade 2 path)
    # "frameshift" -- insertion/deletion that shifts the reading frame
    # "fusion"     -- inter-gene fusion creating a chimeric protein
    # "self"       -- non-mutant self-peptide from sliding window
    mutation_type: str = "missense"

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
        if self.mutation_type not in ("missense", "frameshift", "fusion", "self"):
            raise ValueError(
                f"invalid mutation_type {self.mutation_type!r}; must be "
                "missense / frameshift / fusion / self"
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
                mutation_type="self",
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
                mutation_type="missense",
            ))
    return out


# ---------------------------------------------------------------------------
# Frameshift neoantigens (Upgrade 8)
# ---------------------------------------------------------------------------

def generate_frameshift_peptides(
    wild_type_protein: str,
    frameshift_position_1based: int,
    novel_c_terminal: str,
    source_gene: str,
    mutation_label: str,
    lengths: Iterable[int] = DEFAULT_LENGTHS,
    min_cleavage_score: float = 0.0,
) -> List[Peptide]:
    """Generate peptides covering a frameshift mutation.

    A frameshift shifts the reading frame from ``frameshift_position_1based``
    onward, producing a novel C-terminal stretch of amino acids
    (``novel_c_terminal``) until a new stop codon. Every position from
    the FS site onward carries a residue different from the WT protein,
    so the displayed peptides are *strongly* foreign -- in clinical
    practice these are the highest-quality neoantigens (especially in
    MMR-deficient / MSI-high cancers, which is why anti-PD-1 has its
    best response rates in colorectal MSI-high and Lynch syndrome).

    Returns peptides whose windows contain at least one novel residue.
    For windows that begin upstream of the FS site, the prefix is from
    the WT protein and the suffix is novel. For windows that begin at
    or after the FS site, the entire peptide is novel.

    Each output peptide carries:
        is_mutant         = True
        mutation_type     = "frameshift"
        wild_type_sequence = the WT protein's window at the same span
            (truncated and X-padded if the FS extends past the original
            protein length)
        anchor_position_in_peptide = 0-indexed position of the first
            novel residue within the peptide, or 0 if the window
            starts at or after the FS site.

    Parameters
    ----------
    wild_type_protein : str
        Full WT protein sequence (single-letter, no stops).
    frameshift_position_1based : int
        1-indexed AA position where the frame shifts.
    novel_c_terminal : str
        The new amino-acid sequence the shifted frame produces, from
        the FS position up to the new stop codon. May be longer or
        shorter than the WT region it replaces.
    source_gene, mutation_label
        Provenance fields propagated to every output peptide.
    """
    if not wild_type_protein:
        return []
    if not novel_c_terminal:
        return []
    n_wt = len(wild_type_protein)
    fs_idx = frameshift_position_1based - 1  # 0-based
    if fs_idx < 0 or fs_idx > n_wt:
        raise IndexError(
            f"frameshift_position_1based={frameshift_position_1based} "
            f"out of range for WT protein of length {n_wt}"
        )

    # Synthesize the full mutant protein: WT prefix up to fs_idx,
    # then the novel C-terminal sequence.
    mutant_protein = wild_type_protein[:fs_idx] + novel_c_terminal
    n_mut = len(mutant_protein)

    out: List[Peptide] = []
    for length in lengths:
        if length <= 0 or length > n_mut:
            continue
        # Windows that contain at least one novel residue: window must
        # END at or after the FS site. start in [max(0, fs_idx - length + 1),
        # n_mut - length].
        first_start = max(0, fs_idx - length + 1)
        last_start = n_mut - length
        for start in range(first_start, last_start + 1):
            mut_window = mutant_protein[start:start + length]
            if "*" in mut_window:
                # Window spans the new stop codon; truncate by skipping.
                continue
            cleav = proteasomal_cleavage_score(mut_window[-1])
            if cleav < min_cleavage_score:
                continue
            # Reconstruct the corresponding WT window. For positions
            # >= n_wt, the WT protein had nothing -- pad with 'X' so
            # the WT sequence is the same length as the mutant.
            wt_window_chars: List[str] = []
            for i in range(start, start + length):
                wt_window_chars.append(
                    wild_type_protein[i] if i < n_wt else "X"
                )
            wt_window = "".join(wt_window_chars)
            anchor = max(0, fs_idx - start)
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
                mutation_type="frameshift",
            ))
    return out


# ---------------------------------------------------------------------------
# Fusion neoantigens (Upgrade 8)
# ---------------------------------------------------------------------------

def generate_fusion_peptides(
    left_protein: str,
    right_protein: str,
    left_breakpoint_1based: int,
    right_breakpoint_1based: int,
    source_gene: str,
    mutation_label: str,
    lengths: Iterable[int] = DEFAULT_LENGTHS,
    min_cleavage_score: float = 0.0,
) -> List[Peptide]:
    """Generate peptides spanning a gene-fusion junction.

    A fusion creates a chimeric protein:
        left_protein[:left_breakpoint] + right_protein[right_breakpoint - 1:]
    Peptides that span the junction contain residues from both
    partners and are not present in either parent proteome, so they
    are strongly foreign even if both partner proteins individually
    are self-tolerized.

    Parameters
    ----------
    left_protein, right_protein : str
        The two partner protein sequences.
    left_breakpoint_1based : int
        1-indexed position in the LEFT partner at and after which
        residues are replaced by the right partner. The chimeric
        protein has length (left_breakpoint - 1) of left + the rest
        of right_protein.
    right_breakpoint_1based : int
        1-indexed position in the RIGHT partner where the chimera
        continues. The right partner contributes residues at and
        after this position.
    source_gene, mutation_label : str
        Provenance for the output peptides. ``source_gene`` typically
        names the fusion (e.g., "BCR-ABL1", "EML4-ALK").
    """
    if not left_protein or not right_protein:
        return []
    n_left = len(left_protein)
    n_right = len(right_protein)
    if not (1 <= left_breakpoint_1based <= n_left + 1):
        raise IndexError(
            f"left_breakpoint_1based={left_breakpoint_1based} out of "
            f"range for left protein of length {n_left}"
        )
    if not (1 <= right_breakpoint_1based <= n_right + 1):
        raise IndexError(
            f"right_breakpoint_1based={right_breakpoint_1based} out of "
            f"range for right protein of length {n_right}"
        )

    # The chimeric protein is left[:left_break-1] + right[right_break-1:].
    left_break_idx = left_breakpoint_1based - 1  # 0-based exclusive end
    right_break_idx = right_breakpoint_1based - 1  # 0-based inclusive start
    chimera = left_protein[:left_break_idx] + right_protein[right_break_idx:]
    n_ch = len(chimera)
    junction_idx_in_chimera = left_break_idx  # first right-partner residue

    out: List[Peptide] = []
    for length in lengths:
        if length <= 0 or length > n_ch:
            continue
        # Windows that contain residues from BOTH partners: start in
        # [max(0, junction_idx - length + 1), junction_idx - 1]. The
        # window's last index is start + length - 1; for both-partners
        # coverage we need start <= junction_idx - 1 AND
        # start + length - 1 >= junction_idx.
        first_start = max(0, junction_idx_in_chimera - length + 1)
        last_start = min(junction_idx_in_chimera - 1, n_ch - length)
        for start in range(first_start, last_start + 1):
            window = chimera[start:start + length]
            if "*" in window:
                continue
            cleav = proteasomal_cleavage_score(window[-1])
            if cleav < min_cleavage_score:
                continue
            # WT comparison is ill-defined for fusions; use the LEFT
            # partner extended (the cell's prior identity is closest
            # to left_protein). For positions past the left partner's
            # length, pad with 'X'.
            wt_chars: List[str] = []
            for i in range(start, start + length):
                wt_chars.append(
                    left_protein[i] if i < n_left else "X"
                )
            wt_window = "".join(wt_chars)
            anchor = junction_idx_in_chimera - start  # first right residue
            out.append(Peptide(
                sequence=window,
                source_gene=source_gene,
                length=length,
                is_mutant=True,
                wild_type_sequence=wt_window,
                mutation_label=mutation_label,
                anchor_position_in_peptide=anchor,
                parent_position_1based=start + 1,
                cleavage_score=cleav,
                mutation_type="fusion",
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
