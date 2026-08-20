"""
Protein domain annotations for cancer driver genes (Upgrade 3 Stage B).

For each gene in the cancer-driver panel, this module exposes a
static list of UniProt-derived domain ranges along with a "role"
classification. The panel starts with the three genes for which
cognisom carries authentic CDSes (KRAS, TP53, BRAF) and extends to
~20 of the most-frequently-mutated cancer drivers per the TCGA
Pan-Cancer Atlas (Bailey et al., Cell 2018). Adding more genes here
broadens the classifier's domain-aware path without requiring CDS
sequences -- the classifier looks up codon ranges by gene name when
any caller supplies one, regardless of whether the gene is in the
curated reference genome.

    role             example                          multiplier
    --------------   ------------------------------   ----------
    critical         KRAS P-loop, BRAF activation     4.0
                     loop, TP53 DNA-binding core
    functional       KRAS Switch I/II, BRAF kinase    2.5
                     domain (outside activation
                     loop), TP53 tetramerization
    regulatory       BRAF CRD, TP53 TAD / regulatory  1.5
    structural       (none currently annotated)       1.5

The MutationEffectClassifier consumes these annotations to apply a
2-5x multiplier to the BLOSUM62-derived missense impact score
(UPGRADES_SPEC.md Upgrade 3 Stage B). The patent-evidence point is
that "mutations in critical functional regions are not 'mild' even
when the BLOSUM substitution looks conservative" -- this resolves a
known weakness of the rule-based-only classifier from Stage A.

Domain ranges are 1-based, inclusive on both ends, to match the
convention UniProt uses for feature tables. UniProt accessions are
recorded per domain so reviewers can audit each entry back to the
canonical source.

Sources (accessed for the patent disclosure 2026-05-12):
    KRAS  - UniProt P01116, "K-Ras isoform 4B" feature table
    TP53  - UniProt P04637, "Cellular tumor antigen p53" features
    BRAF  - UniProt P15056, "Serine/threonine-protein kinase B-raf"

Notes
-----
1. The ranges below match the UniProt feature ranges that were
   relevant for cancer hotspots (G12/G13 in KRAS, V600 in BRAF,
   R175/R248/R273 in TP53). Where a feature has multiple slightly
   different ranges across literature, the UniProt canonical range
   is used.
2. The hypervariable membrane-anchor region of KRAS (codons 165-189)
   is classified "structural" rather than "functional" because
   substitutions there impair membrane localization rather than
   catalysis directly.
3. The classifier resolves domain ambiguity (e.g., a residue in both
   the BRAF kinase domain AND the activation loop sub-region) by
   choosing the *highest-priority* domain, i.e., the one with the
   largest impact multiplier. See ``domain_at_codon``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


# Multiplier values per role. The MutationEffectClassifier multiplies
# the base missense impact by these factors, then clamps to the
# missense impact ceiling. Keeping the table here rather than in the
# classifier means both modules read from one source of truth and
# auditors only have to verify these numbers once.
ROLE_MULTIPLIER: Dict[str, float] = {
    "critical": 4.0,
    "functional": 2.5,
    "regulatory": 1.5,
    "structural": 1.5,
}


@dataclass(frozen=True)
class ProteinDomain:
    """A single annotated functional region of a protein.

    Ranges are 1-based, inclusive. ``role`` keys into ROLE_MULTIPLIER.
    """
    gene: str
    name: str
    role: str
    start_codon_1based: int
    end_codon_1based: int
    uniprot_id: str
    source: str = ""

    def __post_init__(self) -> None:
        if self.role not in ROLE_MULTIPLIER:
            raise ValueError(
                f"unknown domain role {self.role!r} for "
                f"{self.gene}/{self.name}; allowed: "
                f"{sorted(ROLE_MULTIPLIER)}"
            )
        if self.start_codon_1based < 1 or self.end_codon_1based < self.start_codon_1based:
            raise ValueError(
                f"invalid range [{self.start_codon_1based}, "
                f"{self.end_codon_1based}] for {self.gene}/{self.name}"
            )

    @property
    def impact_multiplier(self) -> float:
        return ROLE_MULTIPLIER[self.role]

    def contains_codon(self, codon_1based: int) -> bool:
        return self.start_codon_1based <= codon_1based <= self.end_codon_1based


# ---------------------------------------------------------------------------
# Curated domain tables.
# ---------------------------------------------------------------------------

_KRAS = [
    ProteinDomain(
        gene="KRAS", name="P-loop", role="critical",
        start_codon_1based=10, end_codon_1based=17,
        uniprot_id="P01116",
        source="UniProt P01116 nucleotide phosphate-binding region "
               "(G10-K17); GTPase fold P-loop covering G12/G13 hotspots",
    ),
    ProteinDomain(
        gene="KRAS", name="Switch I", role="functional",
        start_codon_1based=30, end_codon_1based=38,
        uniprot_id="P01116",
        source="UniProt P01116 region of interest; effector / GAP "
               "interaction surface",
    ),
    ProteinDomain(
        gene="KRAS", name="Switch II", role="functional",
        start_codon_1based=60, end_codon_1based=76,
        uniprot_id="P01116",
        source="UniProt P01116 region of interest; catalytic Q61 in this "
               "range",
    ),
    ProteinDomain(
        gene="KRAS", name="Hypervariable membrane anchor",
        role="structural",
        start_codon_1based=165, end_codon_1based=189,
        uniprot_id="P01116",
        source="UniProt P01116 hypervariable region; CAAX motif at "
               "C-terminus drives membrane localization",
    ),
]

_TP53 = [
    ProteinDomain(
        gene="TP53", name="Transactivation domain 1",
        role="regulatory",
        start_codon_1based=1, end_codon_1based=42,
        uniprot_id="P04637",
        source="UniProt P04637 transactivation domain TAD1; binds MDM2",
    ),
    ProteinDomain(
        gene="TP53", name="Transactivation domain 2",
        role="regulatory",
        start_codon_1based=43, end_codon_1based=63,
        uniprot_id="P04637",
        source="UniProt P04637 transactivation domain TAD2",
    ),
    ProteinDomain(
        gene="TP53", name="Proline-rich region",
        role="regulatory",
        start_codon_1based=64, end_codon_1based=92,
        uniprot_id="P04637",
        source="UniProt P04637 PRR; modulates apoptosis signalling",
    ),
    ProteinDomain(
        gene="TP53", name="DNA-binding domain",
        role="critical",
        start_codon_1based=94, end_codon_1based=312,
        uniprot_id="P04637",
        source="UniProt P04637 DBD; covers R175, G245, R248, R249, "
               "R273, R282 hotspots responsible for >80% of p53 "
               "cancer-associated missense mutations",
    ),
    ProteinDomain(
        gene="TP53", name="Tetramerization domain",
        role="functional",
        start_codon_1based=320, end_codon_1based=356,
        uniprot_id="P04637",
        source="UniProt P04637 tetramerization helix; required for "
               "p53 dimerization and full transcriptional activity",
    ),
    ProteinDomain(
        gene="TP53", name="C-terminal regulatory",
        role="regulatory",
        start_codon_1based=363, end_codon_1based=393,
        uniprot_id="P04637",
        source="UniProt P04637 C-terminal regulatory domain; "
               "post-translational modification sites",
    ),
]

_BRAF = [
    ProteinDomain(
        gene="BRAF", name="Ras-binding domain (RBD)",
        role="functional",
        start_codon_1based=156, end_codon_1based=227,
        uniprot_id="P15056",
        source="UniProt P15056 RBD; binds activated RAS-GTP",
    ),
    ProteinDomain(
        gene="BRAF", name="Cysteine-rich domain (CRD)",
        role="regulatory",
        start_codon_1based=234, end_codon_1based=280,
        uniprot_id="P15056",
        source="UniProt P15056 zinc finger / phorbol-ester binding region",
    ),
    ProteinDomain(
        gene="BRAF", name="Kinase domain",
        role="functional",
        start_codon_1based=457, end_codon_1based=717,
        uniprot_id="P15056",
        source="UniProt P15056 protein kinase domain; ATP-binding "
               "P-loop and HRD motif within this range",
    ),
    ProteinDomain(
        gene="BRAF", name="Activation loop",
        role="critical",
        start_codon_1based=594, end_codon_1based=623,
        uniprot_id="P15056",
        source="UniProt P15056 activation segment; V600E sits in the "
               "DFG+1 region and produces constitutive kinase activity",
    ),
]


_PIK3CA = [
    # UniProt P42336, p110-alpha catalytic subunit. Helical domain
    # carries the E545K hotspot; the kinase domain carries H1047R.
    ProteinDomain(
        gene="PIK3CA", name="Adapter-binding domain (ABD)",
        role="regulatory",
        start_codon_1based=16, end_codon_1based=105,
        uniprot_id="P42336",
        source="UniProt P42336 ABD; p85 regulatory subunit binding",
    ),
    ProteinDomain(
        gene="PIK3CA", name="Ras-binding domain (RBD)",
        role="functional",
        start_codon_1based=187, end_codon_1based=289,
        uniprot_id="P42336",
        source="UniProt P42336 RBD; receives Ras-GTP signal",
    ),
    ProteinDomain(
        gene="PIK3CA", name="C2 domain",
        role="structural",
        start_codon_1based=330, end_codon_1based=487,
        uniprot_id="P42336",
        source="UniProt P42336 C2 lipid-binding domain",
    ),
    ProteinDomain(
        gene="PIK3CA", name="Helical domain",
        role="critical",
        start_codon_1based=525, end_codon_1based=696,
        uniprot_id="P42336",
        source="UniProt P42336 helical domain; E542K / E545K hotspot",
    ),
    ProteinDomain(
        gene="PIK3CA", name="Kinase domain",
        role="critical",
        start_codon_1based=797, end_codon_1based=1068,
        uniprot_id="P42336",
        source="UniProt P42336 PI3/PI4-kinase domain; H1047R hotspot "
               "(activating)",
    ),
]

_PTEN = [
    # UniProt P60484, lipid phosphatase. Catalytic core is the
    # phosphatase domain; C2 supports membrane localization.
    ProteinDomain(
        gene="PTEN", name="Phosphatase domain",
        role="critical",
        start_codon_1based=14, end_codon_1based=185,
        uniprot_id="P60484",
        source="UniProt P60484 phosphatase domain; HCxxGxxR P-loop "
               "motif with catalytic C124 (essential)",
    ),
    ProteinDomain(
        gene="PTEN", name="C2 domain",
        role="functional",
        start_codon_1based=186, end_codon_1based=351,
        uniprot_id="P60484",
        source="UniProt P60484 C2 membrane-binding domain",
    ),
    ProteinDomain(
        gene="PTEN", name="C-terminal regulatory tail",
        role="regulatory",
        start_codon_1based=352, end_codon_1based=403,
        uniprot_id="P60484",
        source="UniProt P60484 PEST + PDZ-binding motif; "
               "phosphorylation cluster",
    ),
]

_EGFR = [
    # UniProt P00533, receptor tyrosine kinase. L858R hotspot is in
    # the kinase activation loop.
    ProteinDomain(
        gene="EGFR", name="Extracellular ligand-binding (L1+L2)",
        role="regulatory",
        start_codon_1based=25, end_codon_1based=480,
        uniprot_id="P00533",
        source="UniProt P00533 receptor L domains binding EGF/TGFa",
    ),
    ProteinDomain(
        gene="EGFR", name="Transmembrane domain",
        role="structural",
        start_codon_1based=646, end_codon_1based=668,
        uniprot_id="P00533",
        source="UniProt P00533 single-pass TM helix",
    ),
    ProteinDomain(
        gene="EGFR", name="Tyrosine kinase domain",
        role="functional",
        start_codon_1based=712, end_codon_1based=979,
        uniprot_id="P00533",
        source="UniProt P00533 protein kinase domain",
    ),
    ProteinDomain(
        gene="EGFR", name="Kinase activation loop",
        role="critical",
        start_codon_1based=855, end_codon_1based=874,
        uniprot_id="P00533",
        source="UniProt P00533 activation segment carrying L858R; "
               "DFG-out -> DFG-in conformational switch",
    ),
]

_NRAS = [
    # UniProt P01111. Same GTPase fold as KRAS; Q61R is the hotspot.
    ProteinDomain(
        gene="NRAS", name="P-loop",
        role="critical",
        start_codon_1based=10, end_codon_1based=17,
        uniprot_id="P01111",
        source="UniProt P01111 nucleotide-binding P-loop; G12/G13 hotspots",
    ),
    ProteinDomain(
        gene="NRAS", name="Switch I",
        role="functional",
        start_codon_1based=30, end_codon_1based=38,
        uniprot_id="P01111",
        source="UniProt P01111 effector-binding region",
    ),
    ProteinDomain(
        gene="NRAS", name="Switch II",
        role="critical",
        start_codon_1based=60, end_codon_1based=76,
        uniprot_id="P01111",
        source="UniProt P01111 catalytic switch II; Q61 hotspot for "
               "GTPase activity loss",
    ),
]

_IDH1 = [
    # UniProt O75874, cytoplasmic NADP-dependent IDH. R132H is the
    # gliomagenic hotspot producing 2-hydroxyglutarate.
    ProteinDomain(
        gene="IDH1", name="Substrate-binding domain",
        role="critical",
        start_codon_1based=94, end_codon_1based=137,
        uniprot_id="O75874",
        source="UniProt O75874 isocitrate-binding pocket; R132 forms "
               "salt bridge with isocitrate, R132H is neomorphic",
    ),
    ProteinDomain(
        gene="IDH1", name="NADP+ binding",
        role="functional",
        start_codon_1based=212, end_codon_1based=288,
        uniprot_id="O75874",
        source="UniProt O75874 Rossmann fold NADP+ binding region",
    ),
]

_IDH2 = [
    # UniProt P48735, mitochondrial. R140 and R172 are the gliomagenic
    # hotspots producing 2-HG.
    ProteinDomain(
        gene="IDH2", name="Substrate-binding domain",
        role="critical",
        start_codon_1based=124, end_codon_1based=180,
        uniprot_id="P48735",
        source="UniProt P48735 isocitrate-binding pocket; R140 and "
               "R172 are gain-of-function 2-HG hotspots",
    ),
]

_APC = [
    # UniProt P25054. Truncating mutations cluster in the mutation
    # cluster region (MCR) within the central beta-catenin-binding region.
    ProteinDomain(
        gene="APC", name="Armadillo repeats",
        role="functional",
        start_codon_1based=453, end_codon_1based=767,
        uniprot_id="P25054",
        source="UniProt P25054 ARM repeats; protein-protein scaffold",
    ),
    ProteinDomain(
        gene="APC", name="Beta-catenin binding region (MCR)",
        role="critical",
        start_codon_1based=1020, end_codon_1based=2000,
        uniprot_id="P25054",
        source="UniProt P25054 beta-catenin binding; mutation cluster "
               "region (MCR) where truncating mutations concentrate in "
               "colorectal cancer",
    ),
]

_RB1 = [
    # UniProt P06400. Pocket domains A and B; R552 sits in the pocket.
    ProteinDomain(
        gene="RB1", name="Pocket domain A",
        role="critical",
        start_codon_1based=379, end_codon_1based=572,
        uniprot_id="P06400",
        source="UniProt P06400 pocket domain A; E2F binding interface",
    ),
    ProteinDomain(
        gene="RB1", name="Pocket domain B",
        role="critical",
        start_codon_1based=646, end_codon_1based=772,
        uniprot_id="P06400",
        source="UniProt P06400 pocket domain B; cyclin-CDK binding",
    ),
]

_BRCA1 = [
    # UniProt P38398. RING + BRCT are the structured ends.
    ProteinDomain(
        gene="BRCA1", name="RING domain",
        role="critical",
        start_codon_1based=24, end_codon_1based=64,
        uniprot_id="P38398",
        source="UniProt P38398 zinc-binding RING; E3 ubiquitin ligase "
               "with BARD1",
    ),
    ProteinDomain(
        gene="BRCA1", name="BRCT repeats",
        role="critical",
        start_codon_1based=1646, end_codon_1based=1859,
        uniprot_id="P38398",
        source="UniProt P38398 tandem BRCT phospho-peptide binding "
               "modules; pathogenic missense hotspot",
    ),
]

_BRCA2 = [
    # UniProt P51587. BRC repeats bind RAD51; DBD binds ssDNA.
    ProteinDomain(
        gene="BRCA2", name="BRC repeats",
        role="critical",
        start_codon_1based=1002, end_codon_1based=2085,
        uniprot_id="P51587",
        source="UniProt P51587 eight BRC repeats binding RAD51; "
               "homologous-recombination essential",
    ),
    ProteinDomain(
        gene="BRCA2", name="DNA-binding domain",
        role="critical",
        start_codon_1based=2481, end_codon_1based=3186,
        uniprot_id="P51587",
        source="UniProt P51587 helical + OB folds binding ssDNA",
    ),
]

_ATM = [
    # UniProt Q13315. Massive kinase; FAT and kinase at C-term.
    ProteinDomain(
        gene="ATM", name="HEAT repeat region",
        role="structural",
        start_codon_1based=1, end_codon_1based=1900,
        uniprot_id="Q13315",
        source="UniProt Q13315 HEAT alpha-solenoid scaffold",
    ),
    ProteinDomain(
        gene="ATM", name="FAT domain",
        role="functional",
        start_codon_1based=1960, end_codon_1based=2566,
        uniprot_id="Q13315",
        source="UniProt Q13315 FAT regulatory domain",
    ),
    ProteinDomain(
        gene="ATM", name="Kinase domain",
        role="critical",
        start_codon_1based=2712, end_codon_1based=2962,
        uniprot_id="Q13315",
        source="UniProt Q13315 PI3K-family kinase domain; ATP binding",
    ),
]

_STK11 = [
    # UniProt Q15831 (LKB1). Kinase + regulatory C-tail.
    ProteinDomain(
        gene="STK11", name="Kinase domain",
        role="critical",
        start_codon_1based=49, end_codon_1based=309,
        uniprot_id="Q15831",
        source="UniProt Q15831 protein kinase domain; Peutz-Jeghers "
               "loss-of-function mutations cluster here",
    ),
]

_CDKN2A = [
    # UniProt P42771 (p16/INK4a). Ankyrin repeats bind CDK4/6.
    ProteinDomain(
        gene="CDKN2A", name="Ankyrin repeats",
        role="critical",
        start_codon_1based=8, end_codon_1based=132,
        uniprot_id="P42771",
        source="UniProt P42771 ankyrin repeats; CDK4/6 binding interface",
    ),
]

_FGFR3 = [
    # UniProt P22607. Three Ig-like extracellular + kinase.
    ProteinDomain(
        gene="FGFR3", name="Immunoglobulin-like II",
        role="regulatory",
        start_codon_1based=164, end_codon_1based=247,
        uniprot_id="P22607",
        source="UniProt P22607 Ig-like C2 II; FGF binding",
    ),
    ProteinDomain(
        gene="FGFR3", name="Immunoglobulin-like III",
        role="critical",
        start_codon_1based=248, end_codon_1based=359,
        uniprot_id="P22607",
        source="UniProt P22607 / InterPro IPR007110 Ig-like III; "
               "S249 / G370 hotspots in urothelial cancer and "
               "skeletal dysplasias",
    ),
    ProteinDomain(
        gene="FGFR3", name="Tyrosine kinase domain",
        role="critical",
        start_codon_1based=472, end_codon_1based=761,
        uniprot_id="P22607",
        source="UniProt P22607 kinase domain; K650 activation hotspot",
    ),
]

_MYC = [
    # UniProt P01106. Transcription factor with bHLH-LZ at C-term.
    ProteinDomain(
        gene="MYC", name="Transactivation domain",
        role="functional",
        start_codon_1based=1, end_codon_1based=143,
        uniprot_id="P01106",
        source="UniProt P01106 N-terminal transactivation; MYC boxes I-II",
    ),
    ProteinDomain(
        gene="MYC", name="bHLH-LZ DNA binding",
        role="critical",
        start_codon_1based=355, end_codon_1based=437,
        uniprot_id="P01106",
        source="UniProt P01106 basic helix-loop-helix leucine zipper; "
               "MAX heterodimerization + E-box DNA binding",
    ),
]

_CDH1 = [
    # UniProt P12830 (E-cadherin). Five extracellular cadherin repeats.
    ProteinDomain(
        gene="CDH1", name="Cadherin repeat 1 (EC1)",
        role="critical",
        start_codon_1based=158, end_codon_1based=262,
        uniprot_id="P12830",
        source="UniProt P12830 EC1; homophilic adhesion strand swap",
    ),
    ProteinDomain(
        gene="CDH1", name="Cytoplasmic catenin-binding",
        role="critical",
        start_codon_1based=735, end_codon_1based=882,
        uniprot_id="P12830",
        source="UniProt P12830 cytoplasmic tail; beta-catenin and "
               "p120-catenin binding",
    ),
]

_AR = [
    # UniProt P10275. NTD + DBD + LBD.
    ProteinDomain(
        gene="AR", name="N-terminal transactivation",
        role="regulatory",
        start_codon_1based=1, end_codon_1based=537,
        uniprot_id="P10275",
        source="UniProt P10275 NTD; AF1 + polyQ + polyG repeats",
    ),
    ProteinDomain(
        gene="AR", name="DNA-binding domain",
        role="critical",
        start_codon_1based=556, end_codon_1based=623,
        uniprot_id="P10275",
        source="UniProt P10275 DBD; two zinc fingers bind ARE half-sites",
    ),
    ProteinDomain(
        gene="AR", name="Ligand-binding domain",
        role="critical",
        start_codon_1based=688, end_codon_1based=919,
        uniprot_id="P10275",
        source="UniProt P10275 LBD; T877A and L702H hotspots "
               "broaden ligand specificity in CRPC",
    ),
]

_NF1 = [
    # UniProt P21359. GAP-related domain inactivates RAS.
    ProteinDomain(
        gene="NF1", name="Cysteine-serine-rich domain (CSRD)",
        role="functional",
        start_codon_1based=543, end_codon_1based=909,
        uniprot_id="P21359",
        source="UniProt P21359 CSRD; phosphorylation cluster",
    ),
    ProteinDomain(
        gene="NF1", name="GAP-related domain (GRD)",
        role="critical",
        start_codon_1based=1198, end_codon_1based=1530,
        uniprot_id="P21359",
        source="UniProt P21359 RasGAP-related domain; accelerates Ras-GTP "
               "hydrolysis, loss-of-function drives RAS pathway hyperactivation",
    ),
]


DOMAINS: Dict[str, List[ProteinDomain]] = {
    "KRAS": _KRAS,
    "TP53": _TP53,
    "BRAF": _BRAF,
    "PIK3CA": _PIK3CA,
    "PTEN": _PTEN,
    "EGFR": _EGFR,
    "NRAS": _NRAS,
    "IDH1": _IDH1,
    "IDH2": _IDH2,
    "APC": _APC,
    "RB1": _RB1,
    "BRCA1": _BRCA1,
    "BRCA2": _BRCA2,
    "ATM": _ATM,
    "STK11": _STK11,
    "CDKN2A": _CDKN2A,
    "FGFR3": _FGFR3,
    "MYC": _MYC,
    "CDH1": _CDH1,
    "AR": _AR,
    "NF1": _NF1,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_domains(gene: str) -> List[ProteinDomain]:
    """Return all annotated domains for a gene (empty list if unknown)."""
    return list(DOMAINS.get(gene, ()))


def domain_at_codon(
    gene: str, codon_1based: int
) -> Optional[ProteinDomain]:
    """Return the highest-priority domain covering this codon.

    If two annotated domains overlap (e.g., the BRAF activation loop
    sits inside the kinase domain), the one with the largest
    ``impact_multiplier`` is returned. This biases the multiplier
    toward the most-critical region, which is the patent-evidence
    intent (avoid silently underweighting a mutation that lands in
    both a broad domain and a critical sub-region).
    """
    matching = [
        d for d in DOMAINS.get(gene, ())
        if d.contains_codon(codon_1based)
    ]
    if not matching:
        return None
    return max(matching, key=lambda d: d.impact_multiplier)


def impact_multiplier(gene: Optional[str], codon_1based: int) -> float:
    """Return the multiplier applied to missense impact for this codon.

    Returns 1.0 when no domain matches, or when gene is None / unknown
    (defensive fallback so the classifier never amplifies impact for
    genes the curator has not yet annotated).
    """
    if not gene:
        return 1.0
    d = domain_at_codon(gene, codon_1based)
    return d.impact_multiplier if d else 1.0
