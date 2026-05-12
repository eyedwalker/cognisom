"""
Rule-based mutation-effect classifier for single-nucleotide substitutions.

Replaces the prior hardcoded 5-entry oncogenic-mutation lookup in
nucleic_acids.py with a classifier that works on any (sequence, position,
new_base) triple. Outputs:

  - category in {synonymous, missense, nonsense, start_loss, outside_coding}
  - aa change (e.g., "G12D") for missense / nonsense
  - BLOSUM62 score for missense
  - numerical impact_score in [0, 1] for use as a downstream rate modifier

This is Stage A of UPGRADES_SPEC.md Upgrade 3. Stages B (domain-aware) and
C (ESM-2 stability) layer on top of these results.

Reason for decoupling from the Gene class: this module takes a raw coding
sequence rather than a Gene instance so that Upgrade 1 (reference-genome +
per-cell-delta architecture) can supply sequences via a view object without
changing this classifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

# Upgrade 3 Stage B: optional domain-aware impact lookup. Importing the
# lookup directly (as a module-level name aliased to _domain_at_codon)
# keeps the classifier hot-path free of module attribute resolution
# and isolates Stage B's dependency from Stage A's API.
from engine.py.molecular.protein_domains import (
    domain_at_codon as _domain_at_codon,
)


# ---------------------------------------------------------------------------
# Genetic code (standard, table 1)
# ---------------------------------------------------------------------------

_CODON_TABLE: Dict[str, str] = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

_STOP = '*'
_START = 'M'


# ---------------------------------------------------------------------------
# BLOSUM62 substitution matrix.
# Source: NCBI standard 20-AA BLOSUM62 (Henikoff & Henikoff 1992).
# Higher = more conservative substitution. Diagonal = identity.
# Range across the matrix is roughly [-4, +11].
# ---------------------------------------------------------------------------

_AA_ORDER = "ARNDCQEGHILKMFPSTWYV"

# Row/col order matches _AA_ORDER. Values are standard BLOSUM62.
_BLOSUM62_RAW = [
    [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],  # A
    [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],  # R
    [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],  # N
    [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],  # D
    [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],  # C
    [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],  # Q
    [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],  # E
    [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],  # G
    [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],  # H
    [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],  # I
    [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],  # L
    [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2],  # K
    [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],  # M
    [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1],  # F
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2],  # P
    [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],  # S
    [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0],  # T
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3],  # W
    [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],  # Y
    [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4],  # V
]

_BLOSUM62: Dict[Tuple[str, str], int] = {
    (_AA_ORDER[i], _AA_ORDER[j]): _BLOSUM62_RAW[i][j]
    for i in range(len(_AA_ORDER))
    for j in range(len(_AA_ORDER))
}


def blosum62(aa_from: str, aa_to: str) -> int:
    """Return BLOSUM62 substitution score for two amino acids.

    Raises KeyError on unknown residues (e.g., 'X', '*'); callers must
    handle stops separately.
    """
    return _BLOSUM62[(aa_from.upper(), aa_to.upper())]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MutationEffect:
    """Outcome of classifying a single-nucleotide substitution.

    category : str
        One of: synonymous, missense, nonsense, start_loss, outside_coding,
        invalid_codon. "invalid_codon" applies when the reference or mutant
        codon contains an N or partial codon (e.g., position past CDS end).

    impact_score : float in [0, 1]
        0 = no functional impact (synonymous, outside coding).
        1 = complete loss of function (early nonsense, start codon loss).
        Intermediate = scaled by BLOSUM62 for missense, by truncation
        position for nonsense.

    aa_change : str or None
        Human-readable change, e.g., "G12D". None for synonymous (no AA
        change at all) and outside_coding.

    blosum62_score : int or None
        Standard BLOSUM62 substitution score for the AA change. Only set
        for missense.

    wild_type_aa, mutant_aa : str or None
        Single-letter AAs at the affected codon. "*" denotes stop.

    codon_index : int or None
        Index (0-based) of the affected codon within the CDS.

    truncation_fraction : float or None
        For nonsense, the fraction of full protein retained before the
        premature stop, in [0, 1]. None otherwise.

    notes : str
        Free-form annotation for debugging / patent-evidence logging.
    """

    category: str
    impact_score: float
    aa_change: Optional[str]
    blosum62_score: Optional[int]
    wild_type_aa: Optional[str]
    mutant_aa: Optional[str]
    codon_index: Optional[int]
    truncation_fraction: Optional[float] = None
    notes: str = ""

    # Upgrade 3 Stage B: domain-aware impact. Populated when the caller
    # passes ``gene_name`` to classify_substitution() AND the codon
    # falls inside a curated domain in protein_domains.DOMAINS.
    # domain_name / domain_role are descriptive; domain_multiplier is
    # the factor applied to the base missense impact (1.0 means no
    # multiplier was applied -- i.e., the mutation is in a linker /
    # disordered region or the gene is not annotated).
    domain_name: Optional[str] = None
    domain_role: Optional[str] = None
    domain_multiplier: float = 1.0


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class MutationEffectClassifier:
    """Rule-based classifier for single-nucleotide substitutions.

    Usage:
        clf = MutationEffectClassifier()
        effect = clf.classify_substitution(
            coding_sequence="ATGGAC...",
            position=35,
            new_base="A",
            cds_start=0,
        )
    """

    # Mapping from BLOSUM62 score range to missense impact [0, 1].
    # The diagonal (identity) of BLOSUM62 ranges 4..11; off-diagonal -4..3.
    # Define: blosum_score >= 4 -> impact = 0 (conservative / identity);
    # blosum_score = 0 -> impact ~ 0.4 (neutral);
    # blosum_score = -4 -> impact ~ 0.75 (radical change).
    # Capped at 0.85 so that nonsense still scores higher than any missense.
    _MISSENSE_IMPACT_MAX = 0.85
    _MISSENSE_NEUTRAL_THRESHOLD = 4

    def classify_substitution(
        self,
        coding_sequence: str,
        position: int,
        new_base: str,
        cds_start: int = 0,
        gene_name: Optional[str] = None,
    ) -> MutationEffect:
        """Classify a substitution.

        Parameters
        ----------
        coding_sequence : str
            The reference coding sequence (DNA, ACGT). Use uppercase. May
            include 5' UTR before cds_start; the CDS itself must start with
            ATG and end at a stop codon (TAA/TAG/TGA).
        position : int
            Zero-based base position in coding_sequence at which the
            substitution occurs.
        new_base : str
            One of 'A', 'C', 'G', 'T' (or 'U', which is treated as T).
        cds_start : int
            Position in coding_sequence where the open reading frame begins.
            Defaults to 0. Substitutions before cds_start are classified
            outside_coding.
        gene_name : str, optional
            When supplied, the classifier looks up ``codon_index`` in
            ``protein_domains.DOMAINS`` and applies a 1.5x-4x multiplier
            to the BLOSUM62-derived missense impact (Upgrade 3 Stage B).
            The base score, multiplier, and the resolved domain are all
            recorded on the returned MutationEffect. When omitted, the
            classifier behaves exactly as Stage A.
        """
        ref = coding_sequence.upper().replace('U', 'T')
        new_base = new_base.upper().replace('U', 'T')

        # Input validation
        if not (0 <= position < len(ref)):
            return MutationEffect(
                category="invalid_position",
                impact_score=0.0,
                aa_change=None,
                blosum62_score=None,
                wild_type_aa=None,
                mutant_aa=None,
                codon_index=None,
                notes=f"position {position} outside sequence length {len(ref)}",
            )
        if new_base not in "ACGT":
            return MutationEffect(
                category="invalid_base",
                impact_score=0.0,
                aa_change=None,
                blosum62_score=None,
                wild_type_aa=None,
                mutant_aa=None,
                codon_index=None,
                notes=f"new_base {new_base!r} is not one of ACGT(U)",
            )

        # Same-base substitution: no effect
        if ref[position] == new_base:
            return MutationEffect(
                category="synonymous",
                impact_score=0.0,
                aa_change=None,
                blosum62_score=None,
                wild_type_aa=None,
                mutant_aa=None,
                codon_index=None,
                notes="substitution does not change the base",
            )

        # Outside the CDS
        if position < cds_start:
            return MutationEffect(
                category="outside_coding",
                impact_score=0.0,
                aa_change=None,
                blosum62_score=None,
                wild_type_aa=None,
                mutant_aa=None,
                codon_index=None,
                notes="mutation in 5' UTR",
            )

        # Locate the codon containing this position
        cds = ref[cds_start:]
        offset_in_cds = position - cds_start
        codon_index = offset_in_cds // 3
        base_in_codon = offset_in_cds % 3

        # Reference codon
        codon_start_in_ref = cds_start + codon_index * 3
        if codon_start_in_ref + 3 > len(ref):
            return MutationEffect(
                category="invalid_codon",
                impact_score=0.0,
                aa_change=None,
                blosum62_score=None,
                wild_type_aa=None,
                mutant_aa=None,
                codon_index=codon_index,
                notes="position past end of CDS / incomplete codon",
            )

        wt_codon = ref[codon_start_in_ref:codon_start_in_ref + 3]
        mut_codon = wt_codon[:base_in_codon] + new_base + wt_codon[base_in_codon + 1:]

        wt_aa = _CODON_TABLE.get(wt_codon, 'X')
        mut_aa = _CODON_TABLE.get(mut_codon, 'X')

        # Start codon (codon_index 0) special case
        if codon_index == 0 and wt_codon == 'ATG' and mut_codon != 'ATG':
            return MutationEffect(
                category="start_loss",
                impact_score=0.9,
                aa_change=f"M1{mut_aa}",
                blosum62_score=None,
                wild_type_aa='M',
                mutant_aa=mut_aa,
                codon_index=0,
                notes="start codon (ATG) destroyed; translation cannot initiate",
            )

        # Synonymous
        if wt_aa == mut_aa and wt_aa != _STOP:
            return MutationEffect(
                category="synonymous",
                impact_score=0.0,
                aa_change=None,
                blosum62_score=None,
                wild_type_aa=wt_aa,
                mutant_aa=mut_aa,
                codon_index=codon_index,
                notes=f"silent at codon {codon_index+1}: {wt_codon}->{mut_codon}",
            )

        # Nonsense: missense -> stop
        if wt_aa != _STOP and mut_aa == _STOP:
            # Compute fraction of protein retained
            total_codons = self._count_codons_until_stop(cds)
            truncation_fraction = (
                codon_index / total_codons if total_codons > 0 else 0.0
            )
            # Early stops are catastrophic; late stops less so.
            # impact = 1 - 0.5 * truncation_fraction, clamped to [0.5, 1.0].
            impact = max(0.5, 1.0 - 0.5 * truncation_fraction)
            return MutationEffect(
                category="nonsense",
                impact_score=impact,
                aa_change=f"{wt_aa}{codon_index+1}*",
                blosum62_score=None,
                wild_type_aa=wt_aa,
                mutant_aa='*',
                codon_index=codon_index,
                truncation_fraction=truncation_fraction,
                notes=(
                    f"premature stop at codon {codon_index+1} of "
                    f"~{total_codons} ({truncation_fraction*100:.1f}% retained)"
                ),
            )

        # Stop-loss: stop -> AA (read-through)
        if wt_aa == _STOP and mut_aa != _STOP:
            return MutationEffect(
                category="stop_loss",
                impact_score=0.7,
                aa_change=f"*{codon_index+1}{mut_aa}",
                blosum62_score=None,
                wild_type_aa='*',
                mutant_aa=mut_aa,
                codon_index=codon_index,
                notes="stop codon read-through; C-terminal extension",
            )

        # Missense
        if wt_aa != mut_aa and wt_aa != _STOP and mut_aa != _STOP:
            score = blosum62(wt_aa, mut_aa)
            base_impact = self._missense_impact_from_blosum(score)

            # Upgrade 3 Stage B: domain-aware impact. The multiplier is
            # applied to the BLOSUM-derived base score, then clamped to
            # the missense ceiling so nonsense / start-loss remain the
            # categorically more severe outcomes.
            domain = (
                _domain_at_codon(gene_name, codon_index + 1)
                if gene_name else None
            )
            multiplier = domain.impact_multiplier if domain is not None else 1.0
            if multiplier > 1.0:
                impact = min(self._MISSENSE_IMPACT_MAX, base_impact * multiplier)
            else:
                impact = base_impact
            return MutationEffect(
                category="missense",
                impact_score=impact,
                aa_change=f"{wt_aa}{codon_index+1}{mut_aa}",
                blosum62_score=score,
                wild_type_aa=wt_aa,
                mutant_aa=mut_aa,
                codon_index=codon_index,
                notes=(
                    f"missense at codon {codon_index+1}: "
                    f"{wt_codon}({wt_aa})->{mut_codon}({mut_aa})"
                    + (
                        f"; in {domain.gene}/{domain.name} ({domain.role}), "
                        f"impact {base_impact:.2f}x{multiplier:.1f}={impact:.2f}"
                        if domain is not None else ""
                    )
                ),
                domain_name=domain.name if domain is not None else None,
                domain_role=domain.role if domain is not None else None,
                domain_multiplier=multiplier,
            )

        # Fallthrough (should not occur)
        return MutationEffect(
            category="unclassified",
            impact_score=0.0,
            aa_change=None,
            blosum62_score=None,
            wild_type_aa=wt_aa,
            mutant_aa=mut_aa,
            codon_index=codon_index,
            notes="classifier fell through; please report",
        )

    # ---- helpers ---------------------------------------------------------

    @staticmethod
    def _count_codons_until_stop(cds: str) -> int:
        """Return number of codons translated before the first stop in cds.

        If no stop is reached (incomplete CDS), returns codons // 3.
        """
        n_codons = len(cds) // 3
        for i in range(n_codons):
            codon = cds[i * 3 : i * 3 + 3]
            if _CODON_TABLE.get(codon) == _STOP:
                return i
        return n_codons

    @classmethod
    def _missense_impact_from_blosum(cls, score: int) -> float:
        """Map a BLOSUM62 score to a missense impact in [0, _MISSENSE_IMPACT_MAX].

        Mapping is piecewise-linear:
            score >= 4    -> 0.0   (conservative)
            score =  0    -> 0.4   (neutral / mildly disruptive)
            score = -4    -> 0.75  (radical change)
            score < -4    -> capped at _MISSENSE_IMPACT_MAX
        """
        if score >= cls._MISSENSE_NEUTRAL_THRESHOLD:
            return 0.0
        if score >= 0:
            # Linear interpolate 0..4 -> 0.4..0.0
            return 0.4 * (cls._MISSENSE_NEUTRAL_THRESHOLD - score) / cls._MISSENSE_NEUTRAL_THRESHOLD
        # score in -inf..0 -> 0.4..0.85
        # impact = 0.4 + 0.45 * min(1, |score|/4)
        magnitude = min(1.0, abs(score) / 4.0)
        return 0.4 + 0.45 * magnitude


# ---------------------------------------------------------------------------
# Convenience: classify a "named" mutation like KRAS_G12D against a sequence.
# Kept here (rather than re-imported from nucleic_acids.py) so that this
# module has no dependency on Gene/DNA. Used as a test fixture, not a
# production API.
# ---------------------------------------------------------------------------

# Known oncogenic mutations, expressed as (codon_index_0based, expected_wt_aa,
# expected_mut_aa). Position-in-CDS is derived. This list is used by tests
# only; the runtime mutation pipeline must work on arbitrary substitutions.
KNOWN_ONCOGENIC = {
    'KRAS': {
        'G12D': (11, 'G', 'D'),  # codon 12 of KRAS (0-indexed = 11)
        'G12V': (11, 'G', 'V'),
        'G13D': (12, 'G', 'D'),
    },
    'BRAF': {
        'V600E': (599, 'V', 'E'),
    },
    'TP53': {
        'R175H': (174, 'R', 'H'),
        'R248W': (247, 'R', 'W'),
    },
}
