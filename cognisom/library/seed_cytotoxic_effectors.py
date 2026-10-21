"""The molecules a cytotoxic cell kills with.

The library held the apoptotic CASCADE -- caspase-8, caspase-3,
caspase-9, BAX, BID's relatives, cytochrome c, PUMA, MCL-1, XIAP -- and
none of what sets it off. No perforin, no granzyme, no granulysin, no
FasL, no Fas, no TRAIL. The platform's central mechanism is T-cell and NK
killing, the immune module counts kills on every step, and the proteins
that perform the kill were not entities.

So the chain had a hole in the middle: a killer cell, then nothing, then
a caspase. The two halves could not be connected because one of them was
not there.

These eight close it. Every accession was verified against UniProt by
protein name and gene symbol before being written, and the masses come
from the sequences themselves.

TWO PATHWAYS, NOT ONE
---------------------
A cytotoxic lymphocyte kills two ways, and they converge on caspase-3,
which the library already has:

  granule exocytosis   perforin forms the pore; granzyme B enters and
                       cleaves caspase-3 directly and BID to tBID;
                       granzyme A kills caspase-independently; granulysin
                       is antimicrobial and pro-apoptotic; serglycin is
                       the proteoglycan the granzymes are packaged on

  death receptor       FasL on the killer engages Fas on the target,
                       which recruits caspase-8 -- already in the
                       library, already wired to caspase-3. TRAIL does
                       the same through its own receptors.

The second pathway was HALF present: caspase-8 was there, with a note
saying it acts at the DISC, and nothing in the library could form a DISC
because neither the receptor nor its ligand existed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Effector:
    """One cytotoxic effector, with its UniProt identity."""

    name: str
    gene: str
    accession: str
    #: UniProt's own recommended name, as the verification record.
    uniprot_name: str
    description: str
    synonyms: Tuple[str, ...]
    compartments: Tuple[str, ...]
    tags: Tuple[str, ...]
    #: (target entity, relationship, note)
    interacts: Tuple[Tuple[str, str, str], ...] = ()


#: Verified against UniProt on 2026-10-20: each accession's recommended
#: protein name and primary gene symbol were read back before use.
EFFECTORS: Tuple[Effector, ...] = (
    Effector(
        "Perforin (PRF1)", "PRF1", "P14222", "Perforin-1",
        "Pore-forming protein of cytotoxic granules (555 aa, 61.4 kDa). "
        "Polymerises in the target membrane in the presence of calcium "
        "and opens the way for the granzymes. Without it the granzymes "
        "reach the cytosol poorly and the granule pathway stalls, which "
        "is what familial haemophagocytic lymphohistiocytosis type 2 is.",
        ("PRF1", "perforin", "perforin-1", "P1", "pore-forming protein"),
        ("cytotoxic granule", "target plasma membrane"),
        ("cytotoxicity", "granule_exocytosis", "pore_forming", "NK", "CD8"),
        (("Granzyme B (GZMB)", "enables",
          "opens the membrane the granzymes enter through"),),
    ),
    Effector(
        "Granzyme B (GZMB)", "GZMB", "P10144", "Granzyme B",
        "Serine protease of cytotoxic granules (247 aa, 27.7 kDa for the "
        "unglycosylated chain). The fastest route to apoptosis available "
        "to a killer cell: it cleaves caspase-3 directly, so it does not "
        "need the upstream cascade, and cleaves BID to tBID for the "
        "mitochondrial amplification loop. Unique among human granzymes "
        "in cleaving after aspartate, like the caspases it activates.",
        ("GZMB", "granzyme B", "granzyme_B", "CTLA-1", "CSP-B", "cathepsin G-like 1"),
        ("cytotoxic granule", "target cytosol"),
        ("cytotoxicity", "granule_exocytosis", "serine_protease", "apoptosis"),
        (("Caspase-3", "activates", "direct cleavage, bypassing the initiators"),
         ("BAX", "activates", "via tBID, the mitochondrial amplification loop")),
    ),
    Effector(
        "Granzyme A (GZMA)", "GZMA", "P12544", "Granzyme A",
        "Tryptase of cytotoxic granules (262 aa, 29.0 kDa). Kills "
        "caspase-INDEPENDENTLY, through a distinct single-stranded DNA "
        "damage programme, which is why a target that has disabled its "
        "caspases can still be killed. Also drives inflammation by "
        "cleaving gasdermin B.",
        ("GZMA", "granzyme A", "granzyme_A", "CTLA-3", "HFSP"),
        ("cytotoxic granule", "target cytosol"),
        ("cytotoxicity", "granule_exocytosis", "serine_protease",
         "caspase_independent"),
        (),
    ),
    Effector(
        "Granulysin (GNLY)", "GNLY", "P22749", "Granulysin",
        "Saposin-like cytolytic protein of cytotoxic granules (145 aa "
        "precursor, 16.4 kDa; the secreted 9 kDa form is the active one). "
        "Lyses bacteria, fungi and parasites directly, and is the "
        "effector that makes a cytotoxic lymphocyte an antimicrobial cell "
        "and not only an anti-tumour one.",
        ("GNLY", "granulysin", "NKG5", "519", "LAG-2", "TLA519"),
        ("cytotoxic granule",),
        ("cytotoxicity", "granule_exocytosis", "antimicrobial", "saposin_like"),
        (),
    ),
    Effector(
        "Serglycin (SRGN)", "SRGN", "P10124", "Serglycin",
        "Proteoglycan core of the cytotoxic granule (158 aa, 17.7 kDa). "
        "The granzymes and perforin are packaged on its "
        "glycosaminoglycan chains; without it they are not retained in "
        "the granule and the cell cannot store the kill machinery it "
        "needs. The reason a granule is a granule.",
        ("SRGN", "serglycin", "PRG1", "proteoglycan 1", "hematopoietic "
         "proteoglycan core protein"),
        ("cytotoxic granule",),
        ("cytotoxicity", "granule_exocytosis", "proteoglycan", "storage"),
        (("Granzyme B (GZMB)", "binds", "packages the granzymes in the granule"),
         ("Perforin (PRF1)", "binds", "packages perforin in the granule")),
    ),
    Effector(
        "Fas ligand (FASLG/CD178)", "FASLG", "P48023",
        "Tumor necrosis factor ligand superfamily member 6",
        "Death ligand on the activated killer cell (281 aa, 31.5 kDa). "
        "Engages Fas on the target and assembles the DISC. The second of "
        "the two killing routes, and the slower one: it needs the target "
        "to run its own caspase cascade, where granzyme B cleaves "
        "caspase-3 itself.",
        ("FASLG", "FasL", "Fas ligand", "CD178", "CD95L", "CD95-L",
         "APTL", "TNFSF6", "apoptosis antigen ligand"),
        ("killer cell plasma membrane", "secreted"),
        ("cytotoxicity", "death_receptor", "TNF_superfamily", "DISC"),
        (("Fas (FAS/CD95)", "activates", "receptor engagement assembles the DISC"),),
    ),
    Effector(
        "Fas (FAS/CD95)", "FAS", "P25445",
        "Tumor necrosis factor receptor superfamily member 6",
        "Death receptor on the target cell (335 aa, 37.7 kDa). Engaged "
        "by FasL, it recruits FADD and procaspase-8 into the DISC. The "
        "library already held caspase-8 with a note that it acts at the "
        "DISC, and nothing in the library could form one, because neither "
        "this receptor nor its ligand existed.",
        ("FAS", "CD95", "Apo-1", "APO-1", "APT1", "TNFRSF6",
         "Fas antigen", "apoptosis-mediating surface antigen FAS"),
        ("target plasma membrane",),
        ("apoptosis", "death_receptor", "TNF_receptor_superfamily", "DISC"),
        (("Caspase-8", "activates",
          "recruits procaspase-8 to the DISC through FADD"),),
    ),
    Effector(
        "TRAIL (TNFSF10/CD253)", "TNFSF10", "P50591",
        "Tumor necrosis factor ligand superfamily member 10",
        "Death ligand (281 aa, 32.5 kDa) acting through DR4 and DR5. Of "
        "particular interest to this platform because it kills "
        "transformed cells while largely sparing normal ones, which is "
        "the selectivity an anti-tumour agent is chosen for, and because "
        "NK cells use it alongside the granule route.",
        ("TNFSF10", "TRAIL", "Apo-2L", "Apo2L", "CD253", "TL2"),
        ("killer cell plasma membrane", "secreted"),
        ("cytotoxicity", "death_receptor", "TNF_superfamily", "tumour_selective"),
        (("Caspase-8", "activates", "through DR4/DR5 and FADD"),),
    ),
)


def build(effector: Effector):
    """One effector as a library entity, before sourcing."""
    # A Protein and not a bare BioEntity. BioEntity.to_dict() does not
    # serialise an ad-hoc ``.properties`` dict, so the verified UniProt
    # names set below reached the database only when the optional
    # network sourcing ran -- and the automatic seed calls this with
    # source=False. A library built from scratch therefore had every
    # effector's uniprot_recommended_name empty, which is the field that
    # records the accession was checked.
    from .models import EntityType, Protein

    entity = Protein(name=effector.name, entity_type=EntityType.PROTEIN)
    entity.display_name = effector.name
    entity.description = effector.description
    entity.synonyms = list(effector.synonyms)
    entity.tags = list(effector.tags)
    entity.compartments = list(effector.compartments)
    entity.source = "curated"
    entity.source_url = f"https://www.uniprot.org/uniprotkb/{effector.accession}"
    # Typed fields, so they survive to_dict() without the network pass.
    entity.uniprot_id = effector.accession
    entity.gene_symbol = effector.gene
    entity.uniprot_recommended_name = effector.uniprot_name
    entity.interacts_with = [
        {"target": target, "type": kind, "note": note}
        for target, kind, note in effector.interacts
    ]
    # Proteins in this library carry one scale; it is a placeholder there
    # and stays one here rather than being invented per protein.
    entity.scale_um = 0.005
    entity.mesh_type = "mesh"
    entity.usd_prim_type = "BioProtein"
    entity.color_rgb = [0.90, 0.35, 0.30]        # the cytotoxic set
    return entity


def seed(store, *, source: bool = True) -> Tuple[int, List[str]]:
    """Add the effectors, optionally sourcing each from UniProt.

    Returns ``(added, notes)``. An entity that already exists is left
    alone: this is additive, and silently overwriting curation would be
    the worse failure.
    """
    notes: List[str] = []
    added = 0
    for effector in EFFECTORS:
        if store.find_entity_by_name(effector.name) is not None:
            notes.append(f"{effector.name}: already present, left alone")
            continue
        entity = build(effector)
        if source:
            try:
                from .uniprot_sourcing import source_entity
                source_entity(entity, accession=effector.accession)
            except Exception as exc:                # pragma: no cover
                notes.append(f"{effector.name}: UniProt sourcing failed ({exc})")
        store.upsert_entity(entity)
        added += 1
    return added, notes
