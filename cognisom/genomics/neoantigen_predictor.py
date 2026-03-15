"""
Neoantigen Predictor
====================

Predict tumor neoantigens by modeling peptide-MHC class I binding affinity.

For each cancer driver mutation, this module:
1. Applies the mutation to the wild-type protein sequence
2. Generates overlapping peptides around the mutation site
3. Scores binding affinity to each of the patient's HLA alleles
4. Ranks neoantigens by predicted immunogenicity

The binding affinity model uses a position-weight matrix (PWM) approach
based on known anchor residue preferences for common HLA alleles.
For clinical use, this should be replaced with MHCflurry or NetMHCpan.

References:
- NetMHCpan 4.1 (Reynisson et al., NAR 2020)
- MHCflurry 2.0 (O'Donnell et al., Cell Systems 2020)
- OmniNeo (multi-omics neoantigen pipeline)
"""

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .gene_protein_mapper import GeneProteinMapper, ProteinInfo
from .hla_typer import HLA_BINDING_PROPERTIES
from .vcf_parser import Variant

logger = logging.getLogger(__name__)

# Amino acid hydrophobicity index (Kyte-Doolittle scale)
# Used to estimate peptide properties relevant to MHC binding
HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "E": -3.5, "Q": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Amino acid classes for anchor residue matching
AA_CLASSES = {
    "hydrophobic": {"A", "V", "L", "I", "M", "F", "W", "P"},
    "aromatic": {"F", "Y", "W"},
    "positive": {"R", "K", "H"},
    "negative": {"D", "E"},
    "small": {"G", "A", "S", "T"},
    "polar": {"S", "T", "N", "Q", "C", "Y"},
}


@dataclass
class Neoantigen:
    """A predicted tumor neoantigen.

    Represents a mutant peptide predicted to bind to the patient's
    MHC class I molecules, making it a potential target for
    T-cell recognition and immunotherapy.
    """
    peptide: str                    # Mutant peptide sequence (8-11 AA)
    wild_type_peptide: str          # Corresponding wild-type peptide
    source_gene: str                # Gene containing the mutation
    mutation: str                   # Mutation string (e.g., "R248W")
    protein_change: str             # Full protein change (e.g., "p.R248W")
    mutation_position_in_peptide: int  # 0-indexed position of mutation in peptide
    peptide_length: int             # Length of peptide

    # MHC binding predictions
    best_hla_allele: str            # Best-binding HLA allele
    binding_affinity_nm: float      # Predicted IC50 in nanomolar
    percentile_rank: float          # Rank among random peptides (0-100, lower=better)
    all_allele_scores: Dict[str, float] = field(default_factory=dict)

    # Classification
    is_strong_binder: bool = False  # < 50 nM
    is_weak_binder: bool = False    # < 500 nM
    binding_level: str = "non-binder"  # "strong", "weak", "non-binder"

    # Immunogenicity features
    agretopicity: float = 0.0       # Ratio of mutant/WT binding (>1 = better)
    foreignness: float = 0.0        # Sequence dissimilarity to self-proteome
    expression_level: str = "high"  # From RNA-seq if available
    clonality: str = "clonal"       # Clonal vs subclonal mutation

    # Vaccine design
    vaccine_priority: int = 0       # 1=highest priority
    include_in_vaccine: bool = False

    def to_dict(self) -> Dict:
        """Serialize to JSON-safe dict."""
        return {
            "peptide": self.peptide,
            "wild_type_peptide": self.wild_type_peptide,
            "source_gene": self.source_gene,
            "mutation": self.mutation,
            "protein_change": self.protein_change,
            "peptide_length": self.peptide_length,
            "best_hla_allele": self.best_hla_allele,
            "binding_affinity_nm": round(self.binding_affinity_nm, 1),
            "percentile_rank": round(self.percentile_rank, 2),
            "binding_level": self.binding_level,
            "is_strong_binder": self.is_strong_binder,
            "agretopicity": round(self.agretopicity, 2),
            "foreignness": round(self.foreignness, 2),
            "vaccine_priority": self.vaccine_priority,
            "include_in_vaccine": self.include_in_vaccine,
            "all_allele_scores": {
                k: round(v, 1) for k, v in self.all_allele_scores.items()
            },
        }


class NeoantigenPredictor:
    """Predict neoantigens from cancer mutations and HLA alleles.

    Implements a simplified MHC-I binding prediction model based on
    anchor residue preferences and peptide properties. For clinical
    applications, integrate MHCflurry or NetMHCpan.

    Example:
        predictor = NeoantigenPredictor()
        neoantigens = predictor.predict(
            cancer_mutations=profile.cancer_driver_mutations,
            affected_proteins=profile.affected_proteins,
            hla_alleles=["HLA-A*02:01", "HLA-A*24:02", "HLA-B*07:02"],
        )

        for neo in neoantigens[:5]:
            print(f"{neo.source_gene} {neo.mutation}: {neo.peptide} "
                  f"→ {neo.best_hla_allele} ({neo.binding_affinity_nm:.0f} nM)")
    """

    def __init__(self):
        self.mapper = GeneProteinMapper()

    def predict(
        self,
        cancer_mutations: List[Variant],
        affected_proteins: Dict[str, ProteinInfo],
        hla_alleles: List[str],
        peptide_lengths: Optional[List[int]] = None,
        max_neoantigens: int = 50,
    ) -> List[Neoantigen]:
        """Predict neoantigens from mutations and HLA alleles.

        Args:
            cancer_mutations: Variants identified as cancer drivers.
            affected_proteins: Dict of gene → ProteinInfo with sequences.
            hla_alleles: Patient HLA alleles (e.g., ["HLA-A*02:01", ...]).
            peptide_lengths: Peptide lengths to consider (default [8, 9, 10, 11]).
            max_neoantigens: Maximum neoantigens to return.

        Returns:
            List of Neoantigen objects, sorted by binding affinity.
        """
        if peptide_lengths is None:
            peptide_lengths = [8, 9, 10, 11]

        all_neoantigens: List[Neoantigen] = []

        for variant in cancer_mutations:
            gene = variant.gene
            if not gene:
                continue

            protein = affected_proteins.get(gene)
            if not protein or not protein.sequence:
                continue

            mutation_str = variant.protein_change
            if not mutation_str:
                continue

            # Parse mutation
            clean_mut = mutation_str.replace("p.", "").strip()
            match = re.match(r"([A-Z])(\d+)([A-Z])", clean_mut)
            if not match:
                # Skip frameshifts, nonsense for now (could extend later)
                continue

            wt_aa = match.group(1)
            pos = int(match.group(2))  # 1-indexed
            mut_aa = match.group(3)

            # Generate mutant and WT peptides around the mutation site
            for pep_len in peptide_lengths:
                peptides = self._generate_peptides(
                    protein.sequence, pos, wt_aa, mut_aa, pep_len
                )

                for mut_pep, wt_pep, mut_pos_in_pep in peptides:
                    # Score binding to each HLA allele
                    allele_scores = {}
                    for allele in hla_alleles:
                        score = self._predict_binding(mut_pep, allele)
                        allele_scores[allele] = score

                    if not allele_scores:
                        continue

                    # Best allele
                    best_allele = min(allele_scores, key=allele_scores.get)
                    best_affinity = allele_scores[best_allele]

                    # Score WT peptide for agretopicity
                    wt_affinity = self._predict_binding(wt_pep, best_allele)
                    agretopicity = wt_affinity / max(best_affinity, 1.0)

                    # Foreignness: how different is the mutant from WT
                    foreignness = self._compute_foreignness(mut_pep, wt_pep)

                    # Classification
                    is_strong = best_affinity < 50
                    is_weak = best_affinity < 500
                    if is_strong:
                        level = "strong"
                    elif is_weak:
                        level = "weak"
                    else:
                        level = "non-binder"

                    # Percentile rank (approximate from affinity)
                    percentile = self._affinity_to_percentile(best_affinity)

                    neo = Neoantigen(
                        peptide=mut_pep,
                        wild_type_peptide=wt_pep,
                        source_gene=gene,
                        mutation=clean_mut,
                        protein_change=mutation_str,
                        mutation_position_in_peptide=mut_pos_in_pep,
                        peptide_length=pep_len,
                        best_hla_allele=best_allele,
                        binding_affinity_nm=best_affinity,
                        percentile_rank=percentile,
                        all_allele_scores=allele_scores,
                        is_strong_binder=is_strong,
                        is_weak_binder=is_weak,
                        binding_level=level,
                        agretopicity=agretopicity,
                        foreignness=foreignness,
                    )
                    all_neoantigens.append(neo)

        # Sort by binding affinity (lower = better)
        all_neoantigens.sort(key=lambda n: n.binding_affinity_nm)

        # Remove duplicates (same peptide, same allele)
        seen = set()
        unique = []
        for neo in all_neoantigens:
            key = (neo.peptide, neo.best_hla_allele)
            if key not in seen:
                seen.add(key)
                unique.append(neo)

        # Assign vaccine priority
        unique = unique[:max_neoantigens]
        for i, neo in enumerate(unique):
            neo.vaccine_priority = i + 1
            # Top 20 binders with agretopicity >= 1 are vaccine candidates
            neo.include_in_vaccine = (
                i < 20 and
                neo.is_weak_binder and
                neo.agretopicity >= 1.0
            )

        n_strong = sum(1 for n in unique if n.is_strong_binder)
        n_weak = sum(1 for n in unique if n.is_weak_binder and not n.is_strong_binder)
        n_vaccine = sum(1 for n in unique if n.include_in_vaccine)
        logger.info(
            f"Predicted {len(unique)} neoantigens: "
            f"{n_strong} strong, {n_weak} weak binders, "
            f"{n_vaccine} vaccine candidates"
        )
        return unique

    def _generate_peptides(
        self,
        sequence: str,
        mut_pos: int,
        wt_aa: str,
        mut_aa: str,
        peptide_length: int,
    ) -> List[Tuple[str, str, int]]:
        """Generate overlapping mutant/WT peptide pairs around a mutation.

        Args:
            sequence: Full protein sequence.
            mut_pos: 1-indexed mutation position.
            wt_aa: Wild-type amino acid.
            mut_aa: Mutant amino acid.
            peptide_length: Length of peptides to generate.

        Returns:
            List of (mutant_peptide, wt_peptide, mutation_position_in_peptide).
        """
        idx = mut_pos - 1  # Convert to 0-indexed
        seq_len = len(sequence)
        peptides = []

        # Generate all windows that contain the mutation position
        for start in range(max(0, idx - peptide_length + 1),
                           min(idx + 1, seq_len - peptide_length + 1)):
            end = start + peptide_length
            if end > seq_len:
                break

            wt_pep = sequence[start:end]
            mut_pos_in_pep = idx - start

            # Apply mutation
            pep_list = list(wt_pep)
            if mut_pos_in_pep < len(pep_list):
                pep_list[mut_pos_in_pep] = mut_aa
            mut_pep = "".join(pep_list)

            peptides.append((mut_pep, wt_pep, mut_pos_in_pep))

        return peptides

    def _predict_binding(self, peptide: str, hla_allele: str) -> float:
        """Predict peptide-MHC binding affinity (IC50 in nM).

        Uses a simplified position-weight matrix model based on
        anchor residue preferences. Lower values = stronger binding.

        For clinical use, replace with MHCflurry:
            from mhcflurry import Class1PresentationPredictor
            predictor = Class1PresentationPredictor.load()
            df = predictor.predict(peptides=[peptide], alleles=[hla_allele])

        Args:
            peptide: Peptide sequence (8-11 AA).
            hla_allele: HLA allele (e.g., "HLA-A*02:01").

        Returns:
            Predicted IC50 in nanomolar (lower = stronger binding).
        """
        # Normalize allele format
        if not hla_allele.startswith("HLA-"):
            hla_allele = f"HLA-{hla_allele}"

        props = HLA_BINDING_PROPERTIES.get(hla_allele)
        if not props:
            # Unknown allele — return moderate affinity estimate
            return 5000.0

        # Check peptide length compatibility
        if len(peptide) not in props.get("peptide_lengths", [9]):
            return 50000.0  # Very weak binding for wrong length

        # Score based on anchor residues
        score = 0.0
        anchor_positions = props.get("anchor_positions", {})

        for pos, preferred_aas in anchor_positions.items():
            aa_idx = pos - 1  # Convert to 0-indexed
            if aa_idx < 0:
                aa_idx = len(peptide) + aa_idx  # Handle C-terminal
            if aa_idx < len(peptide):
                aa = peptide[aa_idx]
                if aa in preferred_aas:
                    score += 3.0  # Strong anchor match
                elif aa in self._get_similar_aas(preferred_aas):
                    score += 1.5  # Similar amino acid
                else:
                    score -= 1.0  # Anchor mismatch

        # Peptide composition features
        hydro_score = sum(HYDROPHOBICITY.get(aa, 0) for aa in peptide) / len(peptide)
        if -1.0 < hydro_score < 2.0:
            score += 0.5  # Moderate hydrophobicity is favorable

        # Penalize charged residues at position 1
        if peptide[0] in ("D", "E", "K", "R"):
            score -= 0.5

        # Convert score to IC50 (nM)
        # Higher score → lower IC50 (better binding)
        # Base IC50 = 5000 nM, each score point halves it
        ic50 = 5000.0 * math.exp(-0.7 * score)

        # Clamp to reasonable range
        return max(1.0, min(50000.0, ic50))

    def _get_similar_aas(self, preferred: List[str]) -> set:
        """Get amino acids similar to the preferred set."""
        similar = set()
        for aa in preferred:
            for cls_name, cls_members in AA_CLASSES.items():
                if aa in cls_members:
                    similar.update(cls_members)
        return similar - set(preferred)

    def _compute_foreignness(self, mutant: str, wildtype: str) -> float:
        """Compute sequence foreignness of mutant vs wild-type peptide.

        Higher values indicate the mutant is more "foreign" to the
        immune system, which correlates with better T-cell recognition.

        Returns:
            Foreignness score (0-1).
        """
        if len(mutant) != len(wildtype):
            return 0.5

        # Count differences and weight by chemical dissimilarity
        total_diff = 0.0
        for m_aa, w_aa in zip(mutant, wildtype):
            if m_aa != w_aa:
                # Chemical dissimilarity
                m_hydro = HYDROPHOBICITY.get(m_aa, 0)
                w_hydro = HYDROPHOBICITY.get(w_aa, 0)
                diff = abs(m_hydro - w_hydro) / 9.0  # Normalize by max range
                total_diff += 0.3 + 0.7 * diff  # Base + chemical distance

        return min(1.0, total_diff)

    @staticmethod
    def _affinity_to_percentile(affinity_nm: float) -> float:
        """Convert IC50 (nM) to approximate percentile rank.

        Percentile rank represents what fraction of random peptides
        bind as well or better. Lower = more exceptional binder.
        """
        if affinity_nm < 50:
            return 0.5
        elif affinity_nm < 500:
            return 2.0
        elif affinity_nm < 5000:
            return 10.0
        else:
            return 50.0

    @staticmethod
    def get_vaccine_candidates(neoantigens: List[Neoantigen]) -> List[Neoantigen]:
        """Filter neoantigens to those suitable for vaccine inclusion.

        Selection criteria:
        - Weak or strong binder (< 500 nM)
        - Agretopicity > 1.0 (mutant binds better than WT)
        - Prioritize diversity across genes
        """
        return [n for n in neoantigens if n.include_in_vaccine]

    @staticmethod
    def summary(neoantigens: List[Neoantigen]) -> Dict:
        """Generate summary statistics for predicted neoantigens."""
        if not neoantigens:
            return {
                "total": 0, "strong_binders": 0, "weak_binders": 0,
                "vaccine_candidates": 0, "genes_with_neoantigens": [],
            }

        return {
            "total": len(neoantigens),
            "strong_binders": sum(1 for n in neoantigens if n.is_strong_binder),
            "weak_binders": sum(
                1 for n in neoantigens
                if n.is_weak_binder and not n.is_strong_binder
            ),
            "non_binders": sum(
                1 for n in neoantigens if not n.is_weak_binder
            ),
            "vaccine_candidates": sum(
                1 for n in neoantigens if n.include_in_vaccine
            ),
            "genes_with_neoantigens": sorted(set(
                n.source_gene for n in neoantigens if n.is_weak_binder
            )),
            "best_affinity_nm": min(n.binding_affinity_nm for n in neoantigens),
            "mean_agretopicity": (
                sum(n.agretopicity for n in neoantigens if n.is_weak_binder) /
                max(1, sum(1 for n in neoantigens if n.is_weak_binder))
            ),
            "alleles_with_binders": sorted(set(
                n.best_hla_allele for n in neoantigens if n.is_weak_binder
            )),
        }
