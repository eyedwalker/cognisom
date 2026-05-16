"""
Adapter between MAD-pipeline Variant records and patent-pipeline mutations.

The MAD pipeline (cognisom/genomics/) carries variants in HGVS format
on a Variant dataclass:

    Variant(gene="KRAS", protein_change="p.G12D", consequence="missense", ...)

The patent pipeline (engine/py/molecular/) expects bare
``(gene_name, mutation_name)`` tuples where mutation_name is a key in
``Gene.ONCOGENIC_SUBSTITUTIONS`` (e.g., ``"G12D"``):

    molecular.introduce_mutation(cell_id, gene_name="KRAS",
                                 mutation_name="G12D")

This module bridges the two. The audit at
docs/patent/MAD_INTEGRATION_AUDIT.md flagged this as a "no-regrets"
piece of integration work -- it has value beyond the SU2C validation
(e.g., letting the Patent Pipeline dashboard page accept user-uploaded
VCFs).

The adapter is intentionally strict: it only emits tuples for
substitutions that are actually drivable through the patent pipeline.
Frameshifts, fusions, nonsense, synonymous, and substitutions in genes
without curated reference CDSes are all silently dropped (returned as
None) -- this matches how a clinical pipeline filters variants before
handoff.

Patent-evidence claim surface: the adapter is the bridge between the
clinical-decision-support layer (MAD board) and the simulation layer
(patent pipeline). Without it, the patent pipeline cannot consume real
patient data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from engine.py.molecular.nucleic_acids import Gene


# Regex to parse HGVS protein change notation: e.g. "p.G12D", "G12D",
# "p.R248W". Captures (wt_aa, position_1based, mut_aa).
_PROTEIN_CHANGE_PATTERN = re.compile(
    r"^p?\.?([A-Z])(\d+)([A-Z*])$"
)


@dataclass(frozen=True)
class AdapterRejection:
    """Why a Variant could not be adapted to a patent-pipeline mutation.

    Captured per-variant so callers (e.g., the dashboard) can surface
    actionable feedback rather than a silent drop.
    """
    gene: Optional[str]
    protein_change: Optional[str]
    reason: str


def variant_to_patent_mutation(
    variant,
) -> Tuple[Optional[Tuple[str, str]], Optional[AdapterRejection]]:
    """Map a MAD-pipeline Variant to a patent-pipeline (gene, mutation) tuple.

    Returns
    -------
    ((gene_name, mutation_name), None)
        On success. ``gene_name`` matches ``Gene.ONCOGENIC_SUBSTITUTIONS``;
        ``mutation_name`` is the bare label (e.g., ``"G12D"``).
    (None, AdapterRejection)
        On failure. The rejection record explains why -- intended for
        UI feedback or downstream filtering.

    Acceptance criteria (in order of evaluation):
        1. variant.gene must be a curated patent-pipeline gene (key of
           Gene.ONCOGENIC_SUBSTITUTIONS).
        2. variant.protein_change must be parseable as HGVS missense
           (regex ``p?\\.?[A-Z]\\d+[A-Z]``). Frameshift / fusion /
           nonsense are rejected here; INDEL and fusion adapters live
           in their own module (deferred).
        3. The parsed mutation_name must be in
           Gene.ONCOGENIC_SUBSTITUTIONS[variant.gene]. Driver mutations
           outside the curated hotspot list are rejected (the
           reference CDS does not validate them at runtime).
    """
    gene_name = getattr(variant, "gene", None)
    protein_change = getattr(variant, "protein_change", None)

    if not gene_name:
        return None, AdapterRejection(
            gene=None,
            protein_change=protein_change,
            reason="variant has no gene annotation",
        )

    curated_table = Gene.ONCOGENIC_SUBSTITUTIONS.get(gene_name)
    if curated_table is None:
        return None, AdapterRejection(
            gene=gene_name,
            protein_change=protein_change,
            reason=(
                f"gene {gene_name!r} is not in the patent-pipeline "
                f"curated CDS set (currently "
                f"{sorted(Gene.ONCOGENIC_SUBSTITUTIONS)}); adding it "
                "requires inlining its RefSeq CDS in "
                "engine/py/molecular/reference_cds.py"
            ),
        )

    if not protein_change:
        return None, AdapterRejection(
            gene=gene_name,
            protein_change=None,
            reason="variant has no protein_change annotation",
        )

    match = _PROTEIN_CHANGE_PATTERN.match(protein_change.strip())
    if match is None:
        return None, AdapterRejection(
            gene=gene_name,
            protein_change=protein_change,
            reason=(
                f"protein_change {protein_change!r} is not a parseable "
                "missense substitution (frameshifts, fusions, and "
                "complex changes need a different adapter)"
            ),
        )

    wt_aa, pos_str, mut_aa = match.groups()
    if mut_aa == "*":
        return None, AdapterRejection(
            gene=gene_name,
            protein_change=protein_change,
            reason=(
                "nonsense mutations (premature stop) are not in the "
                "patent-pipeline missense path; use a separate adapter"
            ),
        )

    mutation_name = f"{wt_aa}{pos_str}{mut_aa}"

    if mutation_name not in curated_table:
        return None, AdapterRejection(
            gene=gene_name,
            protein_change=protein_change,
            reason=(
                f"{gene_name} {mutation_name} is parseable but not in "
                f"the curated hotspot table "
                f"(known: {sorted(curated_table)}). Adding it requires "
                "a curated (position, new_base) entry in "
                "engine/py/molecular/nucleic_acids.py"
            ),
        )

    return (gene_name, mutation_name), None


def adapt_patient_profile(
    profile,
) -> Tuple[List[Tuple[str, str]], List[AdapterRejection]]:
    """Adapt every driver mutation in a MAD PatientProfile to the
    patent pipeline.

    Returns (drivable_mutations, rejections). Both lists are
    parallel-friendly: drivable_mutations[i] is the tuple, and the
    rejections list explains every variant that was dropped.

    Drivable mutations are deduplicated -- the same (gene, mutation)
    pair appearing twice in the VCF (e.g., subclonal hits) is emitted
    once.
    """
    seen: set = set()
    drivable: List[Tuple[str, str]] = []
    rejections: List[AdapterRejection] = []

    variants = getattr(profile, "cancer_driver_mutations", None)
    if variants is None:
        # Fall back to the broader coding_variants list, then variants.
        variants = (
            getattr(profile, "coding_variants", None)
            or getattr(profile, "variants", [])
        )

    for v in variants:
        result, rejection = variant_to_patent_mutation(v)
        if result is not None:
            if result not in seen:
                seen.add(result)
                drivable.append(result)
        elif rejection is not None:
            rejections.append(rejection)

    return drivable, rejections
