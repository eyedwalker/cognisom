"""Resolve library entities to UniProt accessions, by gene symbol only.

WHY SYMBOL ONLY
---------------
Free-text name search against UniProt is confidently wrong for exactly the
entities this library is full of, because a ligand and its receptor share a
name. Measured, all four accepted by a substring check:

    "Androgen Receptor" -> P82094  TATA element modulatory factor
    "TNF-alpha"         -> P78536  ADAM17, the TNF-alpha CONVERTING ENZYME
    "IFN-gamma"         -> P15260  IFNGR1, the RECEPTOR
    "C3a"               -> Q16581  C3AR1, the C3a RECEPTOR

Each is a plausible top hit whose name genuinely contains the query. So nothing
here is resolved by name. A gene symbol is supplied or derived, UniProt is
asked for that exact symbol in reviewed human entries, and the entry is kept
only if the symbol it returns is the symbol that was asked for.

ON THE ALIASES
--------------
``GENE_ALIASES`` is model-asserted: these mappings are not read out of the
library, they are supplied here. That is a weaker provenance than the structure
work, where PubChem answered a name query directly, and it is recorded as such
in ``resolved_by``. The mitigation is that a wrong alias is *visible*: ask for
the wrong symbol and UniProt returns a protein whose name obviously does not
match the entity. A wrong SMILES is invisible; a wrong gene symbol is not.

``NOT_A_SINGLE_PROTEIN`` is the other half, and matters as much. An entity that
names a complex, a cleavage fragment, a gene family or a polysaccharide has no
single accession, and assigning one of its chains would be a quiet lie. Those
are refused with a reason rather than resolved.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

#: Display name -> HGNC gene symbol. Model-asserted; see the module docstring.
GENE_ALIASES: Dict[str, str] = {
    # cytokines and growth factors
    "IL-1beta": "IL1B", "IL-2": "IL2", "IL-4": "IL4", "IL-6": "IL6",
    "IL-10": "IL10", "IL-15": "IL15", "IL-17A": "IL17A", "IL-21": "IL21",
    "IL-33": "IL33", "TNF-alpha": "TNF", "IFN-beta": "IFNB1",
    "IFN-gamma": "IFNG", "TGF-beta1": "TGFB1", "VEGF-A": "VEGFA",
    "GM-CSF": "CSF2", "M-CSF": "CSF1", "SCF (Kit ligand)": "KITLG",
    "PDGF-BB": "PDGFB", "IGF-1": "IGF1",
    # receptors
    "PD-1": "PDCD1", "CTLA-4 (CD152)": "CTLA4", "LAG-3 (CD223)": "LAG3",
    "OX40 (CD134)": "TNFRSF4", "4-1BB (CD137)": "TNFRSF9",
    "VISTA (B7-H5)": "VSIR", "VEGFR2": "KDR", "PSMA": "FOLH1",
    "Androgen Receptor": "AR",
    # intracellular proteins
    "BCL-2": "BCL2", "MCL-1": "MCL1", "Caspase-3": "CASP3",
    "Caspase-8": "CASP8", "Caspase-9": "CASP9", "HIF-1alpha": "HIF1A",
    "mTOR": "MTOR", "MMP-2 (Gelatinase A)": "MMP2",
    "MMP-9 (Gelatinase B)": "MMP9", "TIMP-1": "TIMP1",
    "Cytochrome c": "CYCS", "Fibronectin": "FN1",
    # pattern-recognition receptors
    "Dectin-1": "CLEC7A", "MDA5": "IFIH1", "RIG-I": "DDX58",
    "NLRP3 inflammasome": "NLRP3",
    # complement
    "MBL": "MBL2", "Properdin": "CFP", "Factor B": "CFB",
    # antibody heavy-chain constant regions
    "IgG1": "IGHG1", "IgG2": "IGHG2", "IgG3": "IGHG3", "IgG4": "IGHG4",
    "IgA1 (secretory)": "IGHA1", "IgE": "IGHE", "IgM (pentameric)": "IGHM",
    "IgD": "IGHD",
    # adhesion
    "E-selectin (CD62E)": "SELE", "P-selectin (CD62P)": "SELP",
    "L-selectin (CD62L)": "SELL", "PSGL-1 (CD162)": "SELPLG",
    "VLA-4 (α4β1)": "ITGA4", "ICAM-1 (CD54)": "ICAM1",
    "VCAM-1 (CD106)": "VCAM1", "PECAM-1 (CD31)": "PECAM1",
    "CD31/PECAM-1": "PECAM1", "ICAM-2 (CD102)": "ICAM2",
    "JAM-A": "F11R", "MAdCAM-1": "MADCAM1",
    "VE-cadherin (CD144)": "CDH5",
    # HGNC renamed DDX58 to RIGI, so the old symbol no longer resolves.
    "RIG-I": "RIGI",
    "Factor D": "CFD",
}

#: Entities that have no single accession, and why. Resolving these to one
#: chain would be a quiet lie, so they are refused instead.
NOT_A_SINGLE_PROTEIN: Dict[str, str] = {
    "Hyaluronan": "a glycosaminoglycan, not a protein",
    "Collagen I": "a heterotrimer of COL1A1 and COL1A2",
    "Collagen IV": "a family of six alpha chains, COL4A1-A6",
    "Laminin": "a heterotrimer; the family has many chain combinations",
    "MAC (C5b-9)": "the membrane attack complex, assembled from C5b-C9",
    "cGAS-STING": "two proteins, CGAS and STING1, named as one pathway",
    "IL-12": "a heterodimer of IL12A and IL12B",
    "IL-23": "a heterodimer of IL23A and IL12B",
    "IFN-alpha": "a family of 13 genes, IFNA1-IFNA21",
    "VEGF-A splice variants": "isoforms of VEGFA, not a distinct protein",
    "LFA-1 (CD11a/CD18)": "a heterodimer of ITGAL and ITGB2",
    "MAC-1 (CD11b/CD18)": "a heterodimer of ITGAM and ITGB2",
    "HLA-DQ2.5": "a serotype spanning DQA1 and DQB1 alleles",
    "HLA-DQ8": "a serotype spanning DQA1 and DQB1 alleles",
    "C3a": "a cleavage fragment of C3, not a separate gene product",
    "C3b": "a cleavage fragment of C3, not a separate gene product",
    "C5a": "a cleavage fragment of C5, not a separate gene product",
    "C5b": "a cleavage fragment of C5, not a separate gene product",
    "C1q": "a hexamer of C1QA, C1QB and C1QC chains",
}

#: An HLA allele names a specific allele of a gene. UniProt carries the gene,
#: so the gene is recorded and the allele is kept as a separate field rather
#: than pretending the accession is allele-specific.
_HLA = re.compile(r"^(HLA-[A-Z]+\d?)\*?([\d:]+)?$")


def gene_symbol_for(name: str) -> Tuple[Optional[str], Optional[str]]:
    """Best gene symbol for a display name, and how it was arrived at.

    Returns ``(None, reason)`` when the entity is not a single protein.
    """
    if name in NOT_A_SINGLE_PROTEIN:
        return None, f"not a single protein: {NOT_A_SINGLE_PROTEIN[name]}"
    if name in GENE_ALIASES:
        return GENE_ALIASES[name], "curated_alias"

    m = _HLA.match(name.strip())
    if m:
        return m.group(1), "hla_gene_from_allele"

    # The name may already be a symbol, or carry one in parentheses.
    return None, "no symbol derived"


def candidate_symbols(name: str) -> List[str]:
    """Symbols to try directly, for names that are already gene symbols."""
    out: List[str] = []
    base = re.sub(r"\s*\([^)]*\)", "", name).strip()
    if base:
        out.append(base)
    for inner in re.findall(r"\(([^)]+)\)", name):
        out.extend(p.strip() for p in re.split(r"[/,]", inner))
    out.extend(p.strip() for p in re.split(r"[/]", base))
    seen = set()
    return [s for s in out
            if s and len(s) <= 14 and not (s in seen or seen.add(s))]


def accepts(entry: Dict, symbol: str) -> bool:
    """Whether a UniProt entry really is the gene that was asked for.

    UniProt will answer a gene query with a related entry. The returned gene
    names must contain the requested symbol, or the hit is discarded.
    """
    genes = [g.get("geneName", {}).get("value") for g in entry.get("genes", []) or []]
    return symbol in [g for g in genes if g]
