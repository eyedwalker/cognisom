"""
Protein domain annotations for cancer driver genes (Upgrade 3 Stage B).

For each gene in the curated reference set (KRAS, TP53, BRAF), this
module exposes a static list of UniProt-derived domain ranges along
with a "role" classification:

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


DOMAINS: Dict[str, List[ProteinDomain]] = {
    "KRAS": _KRAS,
    "TP53": _TP53,
    "BRAF": _BRAF,
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
