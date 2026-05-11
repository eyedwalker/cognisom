"""
Reference CDS sequences for the gene library.

KRAS uses an authentic 51-codon prefix from NM_004985.5 (covering the
G12/G13 hotspot). TP53 and BRAF use *synthetic test CDSes* constructed
to place the canonical oncogenic hotspots at biologically correct codon
positions:

  - TP53 codon 175 = CGC (Arg) -> CAC (His) is the R175H mutation
  - TP53 codon 248 = CGG (Arg) -> TGG (Trp) is the R248W mutation
  - BRAF codon 600 = GTG (Val) -> GAG (Glu) is the V600E mutation

These synthetic sequences use the real human protein's first 30 codons
(which are well-documented in UniProt: TP53 P04637, BRAF P15056) followed
by GCG (alanine) filler to reach each hotspot. They are NOT the real
NM_000546.6 / NM_004333.6 CDSes. They exist so that the simulator can
exercise mutation-introduction at the canonical positions and the
classifier can validate the resulting amino-acid change.

VERIFY-BEFORE-FILING: Before patent filing, replace TP53_CDS and BRAF_CDS
with the authentic NCBI reference CDSes (NM_000546.6 and NM_004333.6).
The patent claims do not depend on biological authenticity of off-hotspot
codons -- they depend on the position-correct placement of hotspots and
the classifier's ability to evaluate arbitrary substitutions -- but
enablement evidence is strengthened by using authentic sequences.

Logged in DECISIONS.md on 2026-05-11.
"""

from __future__ import annotations

from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# KRAS - real NM_004985.5 CDS, codons 1-51
# ---------------------------------------------------------------------------

KRAS_CDS: str = (
    "ATGACTGAATATAAACTTGTGGTAGTTGGAGCTGGTGGCGTAGGCAAGAGT"
    "GCCTTGACGATACAGCTAATTCAGAATCATTTTGTGGACGAATATGATCCA"
    "ACAATAGAGGATTCCTACAGGAAGCAAGTAGTAGAAGATGCCTTCTACACG"
)

# Hotspots: (0-indexed codon, expected 3-base codon)
KRAS_HOTSPOTS: Dict[int, str] = {
    11: "GGT",  # codon 12 (Gly) - G12D/G12V site
    12: "GGC",  # codon 13 (Gly) - G13D site
}


# ---------------------------------------------------------------------------
# CDS builder for synthetic-with-real-hotspots sequences
# ---------------------------------------------------------------------------

def _build_synthetic_cds(
    real_prefix_codons: list,
    hotspots: Dict[int, str],
    total_codons: int,
    filler_codon: str = "GCG",  # Ala
    stop_codon: str = "TAA",
) -> str:
    """Build a synthetic CDS by placing real_prefix_codons at the start,
    inserting hotspot codons at their canonical 0-indexed positions, and
    filling the rest with filler_codon. Appends stop_codon at position
    total_codons (i.e., codon total_codons+1 in 1-indexed naming).

    Asserts no premature stops in the filler, and that hotspot positions
    don't collide with real_prefix_codons.
    """
    assert filler_codon not in ("TAA", "TAG", "TGA"), "filler must not be a stop"
    assert stop_codon in ("TAA", "TAG", "TGA"), "stop_codon must be a stop"

    codons = []
    for i in range(total_codons):
        if i < len(real_prefix_codons):
            codons.append(real_prefix_codons[i])
        elif i in hotspots:
            codons.append(hotspots[i])
        else:
            codons.append(filler_codon)

    # Sanity check: hotspot positions inside prefix range must match prefix
    for pos, codon in hotspots.items():
        if pos < len(real_prefix_codons):
            assert real_prefix_codons[pos] == codon, (
                f"hotspot at codon {pos+1} requires {codon} but real_prefix "
                f"has {real_prefix_codons[pos]}"
            )

    codons.append(stop_codon)
    return "".join(codons)


# ---------------------------------------------------------------------------
# TP53 - synthetic with real first-30-codon prefix + R175 + R248 hotspots
# Full protein is 393 AA. We use 393 codons + stop.
# Real TP53 protein N-terminal: MEEPQSDPSVEPPLSQETFSDLWKLLPENN
# These first 30 codons use one canonical codon choice each.
# ---------------------------------------------------------------------------

_TP53_PREFIX_30_CODONS = [
    "ATG", "GAG", "GAG", "CCG", "CAG", "TCA", "GAT", "CCG", "TCG", "GTC",  # MEEPQSDPSV
    "GAG", "CCG", "CCG", "CTG", "TCG", "CAG", "GAG", "ACC", "TTC", "TCG",  # EPPLSQETFS
    "GAT", "CTG", "TGG", "AAG", "CTG", "CTG", "CCG", "GAG", "AAC", "AAC",  # DLWKLLPENN
]

TP53_HOTSPOTS: Dict[int, str] = {
    174: "CGC",  # codon 175 (Arg) - R175H site (CGC -> CAC)
    247: "CGG",  # codon 248 (Arg) - R248W site (CGG -> TGG)
}

TP53_CDS: str = _build_synthetic_cds(
    real_prefix_codons=_TP53_PREFIX_30_CODONS,
    hotspots=TP53_HOTSPOTS,
    total_codons=393,
    filler_codon="GCG",
    stop_codon="TAA",
)


# ---------------------------------------------------------------------------
# BRAF - synthetic with real first-30-codon prefix + V600 hotspot
# Full protein is 766 AA. We use 766 codons + stop.
# Real BRAF protein N-terminal: MAALSGGGGGGAEPGQALFNGDMEPEAGAGA
# (this is approximate - VERIFY against UniProt P15056 before filing)
# ---------------------------------------------------------------------------

_BRAF_PREFIX_30_CODONS = [
    "ATG", "GCG", "GCG", "CTG", "TCG", "GGT", "GGC", "GGT", "GGC", "GGC",  # MAALSGGGGG
    "GGC", "GCG", "GAG", "CCG", "GGT", "CAG", "GCG", "CTG", "TTC", "AAC",  # GAEPGQALFN
    "GGC", "GAC", "ATG", "GAG", "CCG", "GAG", "GCG", "GGC", "GCG", "GGC",  # GDMEPEAGAG
]

BRAF_HOTSPOTS: Dict[int, str] = {
    599: "GTG",  # codon 600 (Val) - V600E site (GTG -> GAG)
}

BRAF_CDS: str = _build_synthetic_cds(
    real_prefix_codons=_BRAF_PREFIX_30_CODONS,
    hotspots=BRAF_HOTSPOTS,
    total_codons=766,
    filler_codon="GCG",
    stop_codon="TAA",
)


# ---------------------------------------------------------------------------
# Import-time invariants
# ---------------------------------------------------------------------------

def _assert_codon(seq: str, codon_1indexed: int, expected: str, gene: str) -> None:
    pos = (codon_1indexed - 1) * 3
    actual = seq[pos:pos + 3]
    assert actual == expected, (
        f"{gene} codon {codon_1indexed}: expected {expected}, got {actual}. "
        f"reference_cds.py is corrupted; do not file patent."
    )


def _assert_no_premature_stop(seq: str, gene: str, up_to_codon: int) -> None:
    """Assert no in-frame stop codon before `up_to_codon` (1-indexed)."""
    for i in range(up_to_codon - 1):
        codon = seq[i * 3 : i * 3 + 3]
        assert codon not in ("TAA", "TAG", "TGA"), (
            f"{gene} has premature stop {codon} at codon {i+1}; "
            f"reference is broken."
        )


# KRAS invariants
_assert_codon(KRAS_CDS, 12, "GGT", "KRAS")
_assert_codon(KRAS_CDS, 13, "GGC", "KRAS")
_assert_no_premature_stop(KRAS_CDS, "KRAS", up_to_codon=15)
assert len(KRAS_CDS) == 153, f"KRAS_CDS length {len(KRAS_CDS)}; expected 153"

# TP53 invariants
_assert_codon(TP53_CDS, 1, "ATG", "TP53")     # start
_assert_codon(TP53_CDS, 175, "CGC", "TP53")   # R175 (Arg)
_assert_codon(TP53_CDS, 248, "CGG", "TP53")   # R248 (Arg)
_assert_codon(TP53_CDS, 394, "TAA", "TP53")   # stop
_assert_no_premature_stop(TP53_CDS, "TP53", up_to_codon=393)
assert len(TP53_CDS) == 394 * 3, f"TP53_CDS length {len(TP53_CDS)}; expected {394*3}"

# BRAF invariants
_assert_codon(BRAF_CDS, 1, "ATG", "BRAF")     # start
_assert_codon(BRAF_CDS, 600, "GTG", "BRAF")   # V600 (Val)
_assert_codon(BRAF_CDS, 767, "TAA", "BRAF")   # stop
_assert_no_premature_stop(BRAF_CDS, "BRAF", up_to_codon=766)
assert len(BRAF_CDS) == 767 * 3, f"BRAF_CDS length {len(BRAF_CDS)}; expected {767*3}"


__all__ = [
    "KRAS_CDS", "KRAS_HOTSPOTS",
    "TP53_CDS", "TP53_HOTSPOTS",
    "BRAF_CDS", "BRAF_HOTSPOTS",
]
