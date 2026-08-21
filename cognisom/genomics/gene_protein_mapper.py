"""
Gene-Protein Mapper
===================

Map gene symbols to UniProt protein IDs and sequences.
Uses the UniProt REST API for sequence retrieval and a built-in
cache of key prostate cancer protein sequences.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

UNIPROT_API = "https://rest.uniprot.org/uniprotkb"


@dataclass
class ProteinInfo:
    """Protein information for a gene."""
    gene: str
    uniprot_id: str
    protein_name: str
    sequence: str
    length: int
    organism: str = "Homo sapiens"
    function: str = ""
    subcellular_location: str = ""

    #: Added to a literature/VCF residue number to reach this sequence's
    #: index. Non-zero only where the numbering convention in use differs
    #: from the UniProt canonical isoform -- see NUMBERING_OFFSETS.
    numbering_offset: int = 0

    def resolve_position(self, pos: int) -> int:
        """Map a reported residue number onto this sequence's numbering."""
        return pos + self.numbering_offset

    @property
    def sequence_preview(self) -> str:
        if len(self.sequence) <= 60:
            return self.sequence
        return self.sequence[:30] + "..." + self.sequence[-30:]

    @property
    def is_partial(self) -> bool:
        """True when the stored sequence is shorter than the real protein.

        ``length`` is the true canonical length; ``sequence`` is what we
        actually hold. Several built-in entries are excerpts, so residue
        numbering beyond ``len(sequence)`` cannot be resolved and any
        mutation there must be refused rather than guessed at.
        """
        return len(self.sequence) < self.length

    @property
    def coverage(self) -> float:
        """Fraction of the real protein this sequence covers (0.0-1.0)."""
        return (len(self.sequence) / self.length) if self.length else 0.0

    def covers_position(self, pos: int) -> bool:
        """True when residue `pos` (1-based, as reported) is present.

        The reported position is bounds-checked as well as the resolved
        one: a positive offset would otherwise rescue position 0, which
        is not a residue number under any numbering convention.
        """
        if pos < 1:
            return False
        resolved = self.resolve_position(pos)
        return 1 <= resolved <= len(self.sequence)

    def residue_at(self, pos: int) -> Optional[str]:
        """Residue at a reported position, after numbering resolution."""
        if not self.covers_position(pos):
            return None
        return self.sequence[self.resolve_position(pos) - 1]


# ── Reference proteome ───────────────────────────────────────────────
#
# Canonical human Swiss-Prot sequences for the curated driver gene set,
# bundled as data/reference_proteins.fasta so the neoantigen path works
# offline and reproducibly.
#
# What was here before was a hand-typed dict of five proteins, four of
# them truncated -- AR held 173 of 919 residues, BRCA2 119 of 3418 -- and
# TP53 was a 302-residue chimera carrying the wrong residue at both of
# p53's best-known hotspots. Because `get_protein` checks this cache
# *first*, those five shadowed the working UniProt fetch below, so the
# five most clinically important genes were the only ones that never got
# a real sequence.

_DATA_DIR = Path(__file__).parent / "data"
REFERENCE_FASTA = _DATA_DIR / "reference_proteins.fasta"

# Residue numbering offsets, applied to a reported position to reach the
# canonical sequence's index.
#
# AR: UniProt P10275 canonical is 920 aa, but the numbering used
# throughout the androgen-receptor literature (and by the annotators that
# emit these calls) corresponds to a 919-aa reference -- the
# polyglutamine tract in exon 1 is polymorphic, so isoforms differ in
# length upstream of the ligand-binding domain where every hotspot sits.
# Verified against eight documented AR mutations, all of which match at
# +1 and none at 0: L701, V715, W741, H874, F876, T877, M895, R629.
#
# This is a curated per-gene fact, deliberately not a per-variant search.
# Searching for an offset that makes each variant's wild-type residue fit
# is unreliable: for a common residue several offsets match, so it
# invents agreement. When a gene is not listed here and the residue does
# not match, the mutation is refused rather than shifted.
NUMBERING_OFFSETS: Dict[str, int] = {
    "AR": 1,
}


def _load_reference_proteome() -> Dict[str, ProteinInfo]:
    """Parse the bundled FASTA into ProteinInfo records."""
    proteins: Dict[str, ProteinInfo] = {}
    if not REFERENCE_FASTA.exists():
        logger.warning(
            "Reference proteome not found at %s; every gene will require a "
            "UniProt fetch.", REFERENCE_FASTA,
        )
        return proteins

    gene = accession = name = None
    chunks: List[str] = []

    def flush():
        if gene and chunks:
            sequence = "".join(chunks)
            proteins[gene] = ProteinInfo(
                gene=gene,
                uniprot_id=accession or "",
                protein_name=name or gene,
                sequence=sequence,
                length=len(sequence),
                numbering_offset=NUMBERING_OFFSETS.get(gene, 0),
            )

    for line in REFERENCE_FASTA.read_text().splitlines():
        if line.startswith(">"):
            flush()
            header = line[1:]
            fields, _, name = header.partition(" ")
            parts = fields.split("|")
            gene = parts[0]
            accession = parts[1] if len(parts) > 1 else ""
            chunks = []
        elif line.strip():
            chunks.append(line.strip())
    flush()

    logger.debug("Loaded %d reference proteins", len(proteins))
    return proteins


BUILTIN_PROTEINS: Dict[str, ProteinInfo] = _load_reference_proteome()


class GeneProteinMapper:
    """Map gene symbols to protein sequences.

    Uses built-in cache for key cancer genes, falls back to
    UniProt REST API for other genes.

    Example:
        mapper = GeneProteinMapper()
        protein = mapper.get_protein("AR")
        print(f"{protein.protein_name}: {protein.length} AA")

        # Get mutant sequence
        mutant = mapper.apply_mutation(protein, "T877A")
        print(f"Mutant: ...{mutant.sequence[870:885]}...")
    """

    def __init__(self):
        self._cache: Dict[str, ProteinInfo] = dict(BUILTIN_PROTEINS)

    def get_protein(self, gene: str) -> Optional[ProteinInfo]:
        """Get protein info for a gene symbol.

        Checks built-in cache first, then queries UniProt.

        Args:
            gene: Gene symbol (e.g. "AR", "TP53").

        Returns:
            ProteinInfo or None if not found.
        """
        gene_upper = gene.upper()
        if gene_upper in self._cache:
            return self._cache[gene_upper]

        # Try UniProt API
        protein = self._fetch_from_uniprot(gene_upper)
        if protein:
            self._cache[gene_upper] = protein
        return protein

    def get_proteins_for_genes(self, genes: List[str]) -> Dict[str, ProteinInfo]:
        """Get protein info for multiple genes.

        Returns:
            Dict mapping gene symbols to ProteinInfo (only found genes).
        """
        results = {}
        for gene in genes:
            protein = self.get_protein(gene)
            if protein:
                results[gene.upper()] = protein
        return results

    def apply_mutation(self, protein: ProteinInfo,
                       mutation: str) -> Optional[ProteinInfo]:
        """Create a mutant protein sequence from a mutation string.

        Args:
            protein: Wild-type ProteinInfo.
            mutation: Mutation string, e.g. "T877A" (Thr877→Ala),
                     "R130*" (Arg130→Stop), "p.T877A" (with prefix).

        Returns:
            New ProteinInfo with mutant sequence, or None if invalid.
        """
        # Parse mutation string
        mutation = mutation.replace("p.", "").strip()
        match = re.match(r"([A-Z])(\d+)([A-Z*])", mutation)
        if not match:
            logger.warning(f"Cannot parse mutation: {mutation}")
            return None

        wt_aa = match.group(1)
        pos = int(match.group(2))
        mut_aa = match.group(3)

        # Validate position
        if not protein.covers_position(pos):
            detail = (
                f"sequence covers {len(protein.sequence)} of {protein.length} "
                f"residues ({protein.coverage:.0%})"
                if protein.is_partial else f"length {len(protein.sequence)}"
            )
            logger.warning(
                "Position %d out of range for %s (%s)",
                pos, protein.gene, detail,
            )
            return None

        # Validate wild-type amino acid. A mismatch means the residue
        # numbering does not line up with this sequence, so applying the
        # substitution would edit a different residue than the variant
        # describes and return a protein that does not exist. Refuse.
        idx = protein.resolve_position(pos) - 1  # 0-indexed, offset applied
        if protein.sequence[idx] != wt_aa:
            logger.warning(
                "Refusing %s%d%s on %s: reference has %s at position %d, "
                "not %s. The mutation does not match this sequence.",
                wt_aa, pos, mut_aa, protein.gene,
                protein.sequence[idx], protein.resolve_position(pos), wt_aa,
            )
            return None

        # Apply mutation
        seq_list = list(protein.sequence)
        if mut_aa == "*":
            # Nonsense mutation — truncate
            mutant_seq = protein.sequence[:idx]
        else:
            seq_list[idx] = mut_aa
            mutant_seq = "".join(seq_list)

        return ProteinInfo(
            gene=protein.gene,
            uniprot_id=protein.uniprot_id,
            protein_name=f"{protein.protein_name} ({mutation})",
            sequence=mutant_seq,
            length=len(mutant_seq),
            organism=protein.organism,
            function=protein.function,
            subcellular_location=protein.subcellular_location,
        )

    def _fetch_from_uniprot(self, gene: str) -> Optional[ProteinInfo]:
        """Fetch protein info from UniProt REST API.

        Searches for human proteins matching the gene symbol.
        """
        try:
            # Search UniProt for human protein with this gene name
            url = (
                f"{UNIPROT_API}/search?"
                f"query=gene_exact:{gene}+AND+organism_id:9606+AND+reviewed:true"
                f"&format=json&size=1"
                f"&fields=accession,protein_name,gene_names,sequence,length,"
                f"cc_function,cc_subcellular_location"
            )
            r = requests.get(url, timeout=15)
            if r.status_code != 200:
                logger.warning(f"UniProt search failed for {gene}: {r.status_code}")
                return None

            data = r.json()
            results = data.get("results", [])
            if not results:
                logger.debug(f"No UniProt results for {gene}")
                return None

            entry = results[0]
            accession = entry.get("primaryAccession", "")
            name_data = entry.get("proteinDescription", {})
            rec_name = name_data.get("recommendedName", {})
            protein_name = rec_name.get("fullName", {}).get("value", gene)

            seq_data = entry.get("sequence", {})
            sequence = seq_data.get("value", "")
            length = seq_data.get("length", len(sequence))

            # Extract function
            function = ""
            comments = entry.get("comments", [])
            for comment in comments:
                if comment.get("commentType") == "FUNCTION":
                    texts = comment.get("texts", [])
                    if texts:
                        function = texts[0].get("value", "")

            protein = ProteinInfo(
                gene=gene,
                uniprot_id=accession,
                protein_name=protein_name,
                sequence=sequence,
                length=length,
                function=function,
            )
            logger.info(f"Fetched {gene} from UniProt: {accession} ({length} AA)")
            return protein

        except Exception as e:
            logger.warning(f"UniProt fetch error for {gene}: {e}")
            return None


# Needed for apply_mutation regex
import re
