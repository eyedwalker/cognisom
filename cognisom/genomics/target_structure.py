"""
Target Structure Builder
========================

Build peptide-MHC complex 3D structures from neoantigen predictions.

Strategy:
  1. Template-first: Fetch known HLA crystal structures from RCSB PDB
  2. AF2-Multimer fallback: Predict de novo with AlphaFold2-Multimer NIM

The resulting PDB text can be rendered in 3Dmol.js (browser, no GPU)
or sent to the Kit/USD molecular viewer (RTX rendering).

Chain convention:
  A = MHC heavy chain (alpha)
  B = Beta-2-microglobulin
  C = Peptide

References:
  - RCSB PDB: https://www.rcsb.org
  - AlphaFold2-Multimer: Evans et al., bioRxiv 2021
"""

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Cache directory for downloaded/predicted structures
_CACHE_DIR = Path(os.environ.get(
    "COGNISOM_HLA_CACHE",
    "/opt/cognisom/data/hla_templates",
))


@dataclass
class PeptideMHCStructure:
    """Result of peptide-MHC structure building."""
    pdb_text: str
    method: str                       # "rcsb_template" | "alphafold2_multimer"
    template_pdb_id: Optional[str] = None
    hla_allele: str = ""
    peptide: str = ""
    wild_type_peptide: Optional[str] = None
    mutation_residue_index: int = -1  # 0-indexed position in peptide chain
    mhc_chain_ids: List[str] = field(default_factory=lambda: ["A", "B"])
    peptide_chain_id: str = "C"
    confidence: Optional[float] = None  # pLDDT (predicted) or resolution (template)
    binding_affinity_nm: float = 0.0
    source_gene: str = ""
    mutation: str = ""


# ═══════════════════════════════════════════════════════════════════════
# HLA ALLELE → PDB TEMPLATE MAPPING
# ═══════════════════════════════════════════════════════════════════════
# Curated crystal structures with peptides bound in the MHC groove.
# Selected for resolution, completeness, and allele coverage.

HLA_TEMPLATES: Dict[str, Dict] = {
    # HLA-A alleles
    "HLA-A*02:01": {"pdb_id": "1HHK", "peptide_chain": "C", "resolution": 2.50,
                     "peptide": "GILGFVFTL", "description": "Influenza M1 58-66"},
    "HLA-A*01:01": {"pdb_id": "4NQX", "peptide_chain": "C", "resolution": 1.73,
                     "peptide": "EVDPIGHLY", "description": "MAGE-A1"},
    "HLA-A*03:01": {"pdb_id": "3RL1", "peptide_chain": "C", "resolution": 1.40,
                     "peptide": "KLIETYFSK", "description": "Nef peptide"},
    "HLA-A*11:01": {"pdb_id": "1X7Q", "peptide_chain": "C", "resolution": 1.50,
                     "peptide": "AIMPARFYPK", "description": "EBV EBNA3B"},
    "HLA-A*24:02": {"pdb_id": "3I6G", "peptide_chain": "C", "resolution": 2.10,
                     "peptide": "RFPLTFGWCF", "description": "HIV Nef"},
    # HLA-B alleles
    "HLA-B*07:02": {"pdb_id": "5EO1", "peptide_chain": "C", "resolution": 1.80,
                     "peptide": "RPHERNGFTVL", "description": "CMV pp65"},
    "HLA-B*08:01": {"pdb_id": "1AGB", "peptide_chain": "C", "resolution": 2.50,
                     "peptide": "FLRGRAYGL", "description": "EBV EBNA3A"},
    "HLA-B*15:01": {"pdb_id": "1XR8", "peptide_chain": "P", "resolution": 1.50,
                     "peptide": "LEKARGSTY", "description": "Self-peptide"},
    "HLA-B*35:01": {"pdb_id": "1A1M", "peptide_chain": "C", "resolution": 2.00,
                     "peptide": "EPLPQGQLTAY", "description": "HIV Nef"},
    "HLA-B*44:02": {"pdb_id": "1M6O", "peptide_chain": "C", "resolution": 1.70,
                     "peptide": "EEFGRAFSF", "description": "ABCD3"},
    # HLA-C alleles
    "HLA-C*07:01": {"pdb_id": "5VGE", "peptide_chain": "C", "resolution": 2.30,
                     "peptide": "RYRPGTVAL", "description": "Self-peptide"},
    "HLA-C*07:02": {"pdb_id": "5VGE", "peptide_chain": "C", "resolution": 2.30,
                     "peptide": "RYRPGTVAL", "description": "Self-peptide"},
    "HLA-C*04:01": {"pdb_id": "1QQD", "peptide_chain": "C", "resolution": 2.60,
                     "peptide": "QYDDAVYKL", "description": "p53 peptide"},
}


class TargetStructureBuilder:
    """Build peptide-MHC complex structures for neoantigen visualization.

    Usage:
        builder = TargetStructureBuilder()
        structure = builder.build(neoantigen)
        # structure.pdb_text → render in 3Dmol.js or Kit
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self._cache_dir = cache_dir or _CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def build(self, neoantigen) -> PeptideMHCStructure:
        """Build peptide-MHC structure for a neoantigen.

        Strategy:
          1. Check if HLA allele has a known template in RCSB PDB
          2. If yes, fetch the template (cached) → return as-is with annotation
          3. If no, try AlphaFold2-Multimer NIM for de novo prediction
          4. If NIM unavailable, return a minimal peptide-only structure

        Args:
            neoantigen: Neoantigen dataclass from neoantigen_predictor.py

        Returns:
            PeptideMHCStructure with PDB text ready for visualization
        """
        hla = neoantigen.best_hla_allele
        peptide = neoantigen.peptide

        # Try template first
        template = self._fetch_template(hla)
        if template is not None:
            return PeptideMHCStructure(
                pdb_text=template["pdb_text"],
                method="rcsb_template",
                template_pdb_id=template["pdb_id"],
                hla_allele=hla,
                peptide=peptide,
                wild_type_peptide=neoantigen.wild_type_peptide,
                mutation_residue_index=neoantigen.mutation_position_in_peptide,
                mhc_chain_ids=["A", "B"],
                peptide_chain_id=template.get("peptide_chain", "C"),
                confidence=template.get("resolution"),
                binding_affinity_nm=neoantigen.binding_affinity_nm,
                source_gene=neoantigen.source_gene,
                mutation=neoantigen.mutation,
            )

        # Try AF2-Multimer
        af2_result = self._predict_with_af2(peptide, hla)
        if af2_result is not None:
            af2_result.wild_type_peptide = neoantigen.wild_type_peptide
            af2_result.mutation_residue_index = neoantigen.mutation_position_in_peptide
            af2_result.binding_affinity_nm = neoantigen.binding_affinity_nm
            af2_result.source_gene = neoantigen.source_gene
            af2_result.mutation = neoantigen.mutation
            return af2_result

        # Fallback: peptide-only minimal PDB
        logger.warning(
            "No template or AF2 available for %s — generating peptide-only structure",
            hla,
        )
        return self._build_peptide_only(neoantigen)

    def build_comparison(self, neoantigen) -> Tuple[PeptideMHCStructure, PeptideMHCStructure]:
        """Build both mutant and wild-type structures for comparison view."""
        mutant = self.build(neoantigen)

        # Create a temporary neoantigen-like object for the WT peptide
        class _WTNeoantigen:
            pass

        wt = _WTNeoantigen()
        wt.peptide = neoantigen.wild_type_peptide
        wt.wild_type_peptide = neoantigen.wild_type_peptide
        wt.best_hla_allele = neoantigen.best_hla_allele
        wt.binding_affinity_nm = neoantigen.all_allele_scores.get(
            neoantigen.best_hla_allele, 999.0
        )
        wt.source_gene = neoantigen.source_gene
        wt.mutation = "wild-type"
        wt.mutation_position_in_peptide = neoantigen.mutation_position_in_peptide

        wild_type = self.build(wt)
        return mutant, wild_type

    # ─────────────────────────────────────────────────────────────────
    # RCSB PDB Template Fetch
    # ─────────────────────────────────────────────────────────────────

    def _fetch_template(self, hla_allele: str) -> Optional[Dict]:
        """Fetch HLA template PDB from RCSB. Caches locally."""
        template_info = HLA_TEMPLATES.get(hla_allele)
        if template_info is None:
            # Try without the leading "HLA-" prefix
            short = hla_allele.replace("HLA-", "")
            for key, val in HLA_TEMPLATES.items():
                if key.replace("HLA-", "") == short:
                    template_info = val
                    break

        if template_info is None:
            logger.info("No PDB template for allele %s", hla_allele)
            return None

        pdb_id = template_info["pdb_id"]
        cache_path = self._cache_dir / f"{pdb_id}.pdb"

        # Check cache
        if cache_path.exists():
            pdb_text = cache_path.read_text()
            if len(pdb_text) > 100:
                return {**template_info, "pdb_text": pdb_text}

        # Fetch from RCSB
        try:
            import urllib.request

            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            logger.info("Fetching %s from RCSB PDB...", pdb_id)

            req = urllib.request.Request(url, headers={"User-Agent": "Cognisom/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                pdb_text = resp.read().decode("utf-8", errors="replace")

            if len(pdb_text) < 100:
                logger.warning("RCSB returned empty/short PDB for %s", pdb_id)
                return None

            # Cache it
            try:
                cache_path.write_text(pdb_text)
                logger.info("Cached %s to %s (%d bytes)", pdb_id, cache_path, len(pdb_text))
            except OSError:
                pass  # Cache write failure is non-fatal

            return {**template_info, "pdb_text": pdb_text}

        except Exception as e:
            logger.warning("Failed to fetch %s from RCSB: %s", pdb_id, e)
            return None

    # ─────────────────────────────────────────────────────────────────
    # AlphaFold2-Multimer NIM Prediction
    # ─────────────────────────────────────────────────────────────────

    # Beta-2-microglobulin sequence (UniProt P61769, human)
    B2M_SEQUENCE = (
        "MSRSVALAVLALLSLSGLEA"
        "IQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDW"
        "SFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM"
    )

    # Representative MHC heavy chain sequences for common HLA alleles
    # Truncated to extracellular domain (~275 AA) for faster prediction
    _MHC_SEQUENCES: Dict[str, str] = {
        "HLA-A*02:01": (
            "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYW"
            "DGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYD"
            "GKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETL"
            "QRTDAPKTHMTHHAVSDHEATLRCWALSFYPAEITLTWQRDGEDQTQDTELVETRPAGDG"
            "TFQKWAAVVVPSGQEQRYTCHVQHEGLPKPLTLRWE"
        ),
    }

    def _predict_with_af2(self, peptide: str, hla_allele: str) -> Optional[PeptideMHCStructure]:
        """Use AlphaFold2-Multimer NIM for de novo pMHC prediction."""
        try:
            from cognisom.nims.alphafold2_multimer import AlphaFold2MultimerClient
        except ImportError:
            logger.info("AlphaFold2-Multimer NIM client not available")
            return None

        # Get MHC heavy chain sequence
        mhc_seq = self._MHC_SEQUENCES.get(hla_allele)
        if mhc_seq is None:
            logger.info("No MHC sequence for %s — cannot predict", hla_allele)
            return None

        # Check cache
        cache_key = hashlib.md5(
            f"{hla_allele}:{peptide}".encode()
        ).hexdigest()[:12]
        cache_path = self._cache_dir / f"af2_{cache_key}.pdb"

        if cache_path.exists():
            pdb_text = cache_path.read_text()
            if len(pdb_text) > 100:
                logger.info("Using cached AF2 prediction for %s:%s", hla_allele, peptide)
                return PeptideMHCStructure(
                    pdb_text=pdb_text,
                    method="alphafold2_multimer",
                    hla_allele=hla_allele,
                    peptide=peptide,
                    mhc_chain_ids=["A", "B"],
                    peptide_chain_id="C",
                )

        try:
            client = AlphaFold2MultimerClient()
            sequences = [mhc_seq, self.B2M_SEQUENCE, peptide]

            logger.info(
                "Predicting pMHC complex with AF2-Multimer: %s + %s (%d + %d + %d AA)",
                hla_allele, peptide, len(mhc_seq), len(self.B2M_SEQUENCE), len(peptide),
            )
            result = client.predict_complex(sequences)

            if result and result.structure_data:
                pdb_text = result.structure_data
                confidence = None
                if result.plddt_scores:
                    confidence = sum(result.plddt_scores) / len(result.plddt_scores)

                # Cache
                try:
                    cache_path.write_text(pdb_text)
                except OSError:
                    pass

                return PeptideMHCStructure(
                    pdb_text=pdb_text,
                    method="alphafold2_multimer",
                    hla_allele=hla_allele,
                    peptide=peptide,
                    mhc_chain_ids=["A", "B"],
                    peptide_chain_id="C",
                    confidence=confidence,
                )

        except Exception as e:
            logger.warning("AF2-Multimer prediction failed: %s", e)

        return None

    # ─────────────────────────────────────────────────────────────────
    # Peptide-Only Fallback
    # ─────────────────────────────────────────────────────────────────

    def _build_peptide_only(self, neoantigen) -> PeptideMHCStructure:
        """Generate a minimal PDB with just the peptide in extended conformation.

        This is the last resort when no template or prediction is available.
        Places each residue at 3.8 A spacing along the X axis (extended chain).
        """
        peptide = neoantigen.peptide
        lines = [
            "REMARK   1 COGNISOM PEPTIDE-ONLY STRUCTURE (EXTENDED CONFORMATION)",
            f"REMARK   2 PEPTIDE: {peptide}",
            f"REMARK   3 HLA: {neoantigen.best_hla_allele}",
            f"REMARK   4 GENE: {neoantigen.source_gene} {neoantigen.mutation}",
        ]

        aa_3letter = {
            "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
            "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
            "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
            "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
        }

        atom_num = 1
        for i, aa in enumerate(peptide):
            resname = aa_3letter.get(aa, "UNK")
            resnum = i + 1
            x = i * 3.8
            y = 0.0
            z = 0.0
            # Place CA atom
            lines.append(
                f"ATOM  {atom_num:5d}  CA  {resname} C{resnum:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
            )
            atom_num += 1

        lines.append("END")

        return PeptideMHCStructure(
            pdb_text="\n".join(lines),
            method="peptide_only",
            hla_allele=neoantigen.best_hla_allele,
            peptide=peptide,
            wild_type_peptide=neoantigen.wild_type_peptide,
            mutation_residue_index=neoantigen.mutation_position_in_peptide,
            mhc_chain_ids=[],
            peptide_chain_id="C",
            confidence=None,
            binding_affinity_nm=neoantigen.binding_affinity_nm,
            source_gene=neoantigen.source_gene,
            mutation=neoantigen.mutation,
        )

    # ─────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────

    def get_available_templates(self) -> Dict[str, str]:
        """Return mapping of HLA alleles to PDB IDs for which templates exist."""
        return {
            allele: info["pdb_id"]
            for allele, info in HLA_TEMPLATES.items()
        }

    def clear_cache(self):
        """Remove all cached structures."""
        if self._cache_dir.exists():
            for f in self._cache_dir.glob("*.pdb"):
                f.unlink()
            logger.info("Cleared structure cache at %s", self._cache_dir)
