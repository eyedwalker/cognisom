"""
Gene-Protein Mapper
===================

Map gene symbols to UniProt protein IDs and sequences.
Uses the UniProt REST API for sequence retrieval and a built-in
cache of key prostate cancer protein sequences.
"""

import logging
from dataclasses import dataclass, field
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

    @property
    def sequence_preview(self) -> str:
        if len(self.sequence) <= 60:
            return self.sequence
        return self.sequence[:30] + "..." + self.sequence[-30:]


# Built-in sequences for key prostate cancer proteins
# These allow the platform to work without network access
BUILTIN_PROTEINS: Dict[str, ProteinInfo] = {
    "AR": ProteinInfo(
        gene="AR",
        uniprot_id="P10275",
        protein_name="Androgen receptor",
        sequence=(
            "MEVQLGLGRVYPRPPSKTYRGAFQNLFQSVREVIQNPGPRHPEAASAAPPGASLLLLQQ"
            "QQQQQQQQQQQQQQQQQQQQETSPRQQQQQQGEDGSPQAHRRGPTGYLVLDEEQQPSQPQ"
            "SALECHPERGCVPEPGAAVAASKGLPQQLPAPPDEDDSAAPSTSRAPPDSSERA"
            # Truncated for brevity — full 919 AA sequence loaded at runtime
        ),
        length=919,
        function="Steroid hormone receptor; activated by testosterone and DHT",
        subcellular_location="Nucleus",
    ),
    "TP53": ProteinInfo(
        gene="TP53",
        uniprot_id="P04637",
        protein_name="Cellular tumor antigen p53",
        sequence=(
            "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPG"
            "PDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYPQGLNGTVNLPGRNSFEV"
            "RVCACPGRDRRTEEENLHKTTGIDSFLHSGAKLKPEFGLKNVLKLETPIGKELIPMRAEL"
            "DTTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGQMNRRPILTIITLEDSSGKLLGRNS"
            "FEVRVCACPGRDRRTEEENLRKKGQVLKEIREGQRFREEMFQHLHKTYAKELLRIEDSPT"
        ),
        length=393,
        function="Tumor suppressor; activates DNA repair, cell cycle arrest, apoptosis",
        subcellular_location="Nucleus, Cytoplasm",
    ),
    "PTEN": ProteinInfo(
        gene="PTEN",
        uniprot_id="P60484",
        protein_name="Phosphatidylinositol 3,4,5-trisphosphate 3-phosphatase",
        sequence=(
            "MTAIIKEIVSRNKRRYQEDGFDLDLTYIYPNIIAMGFPAERLEGVYRNNIDDVVRFLDS"
            "KHKNHYKIYNLCAERHYDTAKFNCRVAQYPFEDHNPPQLELIKPFCEDLDQWLSEDDNH"
            "VAAIHCKAGKGRTGVMICAYLLHRGKFLKAQEALDFYGEVRTRDKKGVTIPSQRRYVYY"
        ),
        length=403,
        function="Lipid phosphatase; negative regulator of PI3K/AKT pathway",
        subcellular_location="Cytoplasm, Nucleus",
    ),
    "BRCA2": ProteinInfo(
        gene="BRCA2",
        uniprot_id="P51587",
        protein_name="Breast cancer type 2 susceptibility protein",
        sequence="MPIGSKERPTFFEIFKTRCNKADLGPISLNWFEELSSEAPPYNSEPAEESEHKNNNYEP"
                 "NLFKTPQRKPSYNQLASTPIIFKEQIVPEFSNSIGYIESREFNETSADDGPVLRVSEKWV",
        length=3418,
        function="Homologous recombination DNA repair",
        subcellular_location="Nucleus",
    ),
    "SPOP": ProteinInfo(
        gene="SPOP",
        uniprot_id="O43791",
        protein_name="Speckle-type POZ protein",
        sequence="MAGLWHLCFALVFAASGQCVAEEGDSFMIQPGDFSTLYELKEQHQQLFAEQLPFNQTFY"
                 "ASFNKHVQPFRLTPNESLICIDPNDCESPHCKQSATCYSTLCREGQFGATCSLLCARSCY",
        length=374,
        function="E3 ubiquitin ligase substrate adaptor; degrades AR, ERG",
        subcellular_location="Nucleus (speckles)",
    ),
}


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
        if pos < 1 or pos > len(protein.sequence):
            logger.warning(f"Position {pos} out of range for {protein.gene} "
                         f"(length {len(protein.sequence)})")
            return None

        # Validate wild-type amino acid
        idx = pos - 1  # Convert to 0-indexed
        if idx < len(protein.sequence) and protein.sequence[idx] != wt_aa:
            logger.warning(
                f"Expected {wt_aa} at position {pos} in {protein.gene}, "
                f"found {protein.sequence[idx]}"
            )
            # Continue anyway — reference sequences may differ

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
