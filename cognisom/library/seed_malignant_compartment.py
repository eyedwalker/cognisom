"""Cells this repository names in code and does not hold as entities.

``TissueScaleConfig.DEFAULT_COMPOSITION`` declares ten archetypes and
``ARCHETYPE_ENTITY`` maps eight of them to a library row. The two it does
not map are ``smooth_muscle`` at 10% and ``cancer_epithelial`` at 10%, so
**a fifth of every simulated tissue has no entity behind it** -- and one
of those two is the malignant compartment, which is what this platform
exists to model.

The effect is not an error anywhere. ``resolve_radii_from_library``
records ``"no library row"`` and falls back to ``cell_radius_um``, so
every cancer cell in every run is a 5.0 um sphere by default, with
nothing to curate, cite or version. The library can state the Km of ATM
and could state nothing at all about the cancer cell.

WHY THESE TWO NAMES
-------------------
The composition is prostate: luminal_secretory, basal and neuroendocrine
are the three epithelial lineages of a prostate gland, and the demo
genome is a prostate VCF carrying AR T877A, PTEN loss and a BRCA2
frameshift. So the malignant row is a prostatic adenocarcinoma cell, and
the stromal one is the smooth muscle that makes prostate stroma
*fibromuscular* -- ``Prostatic stroma`` in this library already lists
only a fibroblast and an endothelial cell, which is the same omission
one level up.

WHAT IS SOURCED AND WHAT IS NOT
-------------------------------
The markers are diagnostic criteria, not estimates: a prostatic
adenocarcinoma is identified by AMACR/P504S positivity together with the
LOSS of the basal markers p63 and CK5, and that absence is recorded here
as explicitly as the presence, because "basal markers absent" is the
finding that distinguishes carcinoma from a benign mimic.

TWO MORE THE CODE EMITS
-----------------------
``Monocyte`` was already in this library's vocabulary without being in
the library: ``seed_ultrastructure`` lists it in ``MONOCYTE_NAMES`` and
holds a measured granule volume fraction for it, BNID 115196, with no
entity to attach to. ``Exhausted CD8+ T cell`` is assigned by name in
``spatial_transcriptomics`` and coloured by the spatial dashboard page,
and is the state checkpoint blockade is aimed at -- a twin that predicts
response to a PD-1 inhibitor could not represent the cell the drug acts
on.

Still unresolved on purpose: ``t cell``, ``b cell``, ``dendritic cell``,
``fibroblast``, ``endothelial`` and ``epithelial`` are GENERIC, and each
has two or more specific entities under it. A synonym that two entities
claim cannot resolve, and whichever is indexed first silently becomes
the answer -- so they are left failing visibly rather than guessed.
``normal``, ``cancer`` and ``immune`` are not cell types at all; they
are the three-way classification in ``cellular_module`` and have no
entity to be.

The SIZES are not sourced and are not pretended to be. Nuclear
enlargement is a defining feature of malignancy and this library cannot
currently cite a figure for it: every human nuclear dimension that
resolves in BioNumbers is HeLa, hypertriploid and measured trypsinised
in suspension. So these two entities arrive with the same placeholder
status as the seven other archetypes whose radius is a placeholder, and
the gap is named rather than filled with a number somebody liked.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence, Tuple

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CellRecord:
    """One cell type, with the archetype it answers for."""

    name: str
    archetype: str
    description: str
    tissue_origin: str
    markers: Tuple[str, ...]
    lineage: str
    cell_ontology_id: str
    synonyms: Tuple[str, ...]
    #: Diameter in um. A placeholder, and said to be one.
    scale_um: float
    mesh_type: str
    color_rgb: Tuple[float, float, float]


CELLS: Tuple[CellRecord, ...] = (
    CellRecord(
        name="Prostatic adenocarcinoma cell",
        archetype="cancer_epithelial",
        description=(
            "The malignant luminal-type epithelial cell of prostatic "
            "acinar adenocarcinoma, and the cell this platform's "
            "predictions are about: it is what displays the neoantigen, "
            "what a cytotoxic lymphocyte kills, what downregulates "
            "MHC-I to escape, and what goes hypoxic at the centre of a "
            "growing focus. It arises from the luminal compartment and "
            "keeps its secretory programme -- AR, NKX3-1, KLK3/PSA, "
            "CK8/18 -- which is why serum PSA tracks tumour burden at "
            "all. What identifies it under a microscope is not a marker "
            "it gains but two it LOSES: p63 and CK5, the basal cell "
            "layer, absent because the malignant gland has no basal "
            "layer. AMACR (P504S) is the positive counterpart, "
            "overexpressed in about 80% of cases. In this library it is "
            "the entity behind the cancer_epithelial archetype, which "
            "is 10% of the default tissue composition and previously "
            "had no entity at all."
        ),
        tissue_origin="prostate",
        markers=("AR+", "NKX3-1+", "KLK3+", "AMACR+", "CK8+", "CK18+",
                 "p63-", "CK5-", "PTEN-low", "HLA-ABC-low"),
        lineage="epithelial",
        cell_ontology_id="CL:0002518",
        synonyms=("cancer_epithelial", "cancer epithelial", "tumor_cell",
                  "tumour cell", "tumor cell", "prostate cancer cell",
                  "adenocarcinoma cell", "malignant epithelial cell"),
        scale_um=18.0,
        mesh_type="irregular_ruffled",
        color_rgb=(0.80, 0.26, 0.30),
    ),
    CellRecord(
        name="Prostatic smooth muscle cell",
        archetype="smooth_muscle",
        description=(
            "The contractile stromal cell that makes prostate stroma "
            "FIBROMUSCULAR rather than merely fibrous. It is the "
            "majority stromal cell by volume in a normal gland, "
            "contracts to expel secretions at ejaculation, and carries "
            "AR -- stromal AR signalling, not epithelial, is what drives "
            "normal glandular development through the andromedins "
            "FGF7/10, IGF1 and HGF. It matters to a tumour model "
            "because it is the cell that is LOST: reactive stroma in "
            "prostate cancer is a switch from smooth muscle to "
            "myofibroblast, and the proportion of smooth muscle "
            "remaining is itself prognostic. In this library it is the "
            "entity behind the smooth_muscle archetype, 10% of the "
            "default composition and previously unmapped."
        ),
        tissue_origin="prostate",
        markers=("ACTA2+", "MYH11+", "CNN1+", "DES+", "AR+", "VIM-low"),
        lineage="mesenchymal",
        cell_ontology_id="CL:0000192",
        synonyms=("smooth_muscle", "smooth muscle", "smooth muscle cell",
                  "myocyte", "prostatic smooth muscle"),
        scale_um=25.0,
        mesh_type="elongated_spindle",
        color_rgb=(0.62, 0.48, 0.72),
    ),
    CellRecord(
        name="Monocyte",
        archetype="monocyte",
        description=(
            "The circulating precursor that becomes a tissue "
            "macrophage. It is in this library's vocabulary already "
            "without being in the library: seed_ultrastructure lists "
            "'Monocyte' in MONOCYTE_NAMES and holds a measured granule "
            "volume fraction for it -- 2.6% from Schmid-Schonbein Table "
            "2, BNID 115196 -- which has had no entity to attach to. "
            "cell_state_classifier also maps 'monocyte' into its "
            "macrophage bucket. It is a distinct cell from the "
            "macrophage it becomes: smaller, kidney-nucleated, "
            "CD14-high and still in the blood, and it is the cell CCL2 "
            "recruits into a tumour."
        ),
        tissue_origin="blood",
        markers=("CD14+", "CD16-", "CCR2+", "HLA-DR+", "CD68+", "CD11b+"),
        lineage="myeloid",
        cell_ontology_id="CL:0000576",
        synonyms=("monocyte", "classical monocyte", "CD14+ monocyte"),
        scale_um=15.0,
        mesh_type="ruffled_pseudopod",
        color_rgb=(0.86, 0.66, 0.36),
    ),
    CellRecord(
        name="Exhausted CD8+ T cell",
        archetype="t_cell_exhausted",
        description=(
            "A CD8 T cell that has lost effector function under chronic "
            "antigen: PD-1, TIM-3, LAG-3 and TIGIT co-expressed, TOX "
            "driving the programme, and granzyme and IFN-gamma "
            "production falling away. It is the state checkpoint "
            "blockade is aimed at, so a twin that predicts response to "
            "a PD-1 inhibitor and cannot represent an exhausted T cell "
            "is missing the cell the drug acts on. "
            "spatial_transcriptomics assigns this label directly and "
            "the spatial dashboard page gives it a colour; neither "
            "could reach an entity."
        ),
        tissue_origin="tumor",
        markers=("CD8+", "PD-1+", "TIM-3+", "LAG-3+", "TIGIT+", "TOX+",
                 "granzyme B-low", "IFNg-low", "CD39+"),
        lineage="lymphoid",
        cell_ontology_id="CL:0000625",
        synonyms=("t_cell_exhausted", "exhausted T cell",
                  "exhausted CD8 T cell", "Tex"),
        scale_um=8.0,
        mesh_type="smooth_sphere",
        color_rgb=(0.74, 0.52, 0.28),
    ),
)


def build(record: CellRecord):
    """One record as a library cell_type entity."""
    from .models import CellTypeEntity

    entity = CellTypeEntity(
        name=record.name,
        display_name=record.name,
        description=record.description,
        cell_ontology_id=record.cell_ontology_id,
        tissue_origin=record.tissue_origin,
        markers=list(record.markers),
        lineage=record.lineage,
        source="curated",
        tags=["prostate_cancer", "tissue_archetype", record.archetype],
    )
    entity.synonyms = list(record.synonyms)
    entity.scale_um = record.scale_um
    entity.mesh_type = record.mesh_type
    entity.usd_prim_type = "BioCell"
    entity.color_rgb = list(record.color_rgb)
    entity.compartments = ["cell_membrane"]
    return entity


def seed(store, *, dry_run: bool = False) -> Tuple[int, List[str]]:
    """Add the missing compartments. Additive; existing rows are kept."""
    notes: List[str] = []
    added = 0
    for record in CELLS:
        if store.find_entity_by_name(record.name) is not None:
            notes.append(f"{record.name}: already present, left alone")
            continue
        if not dry_run:
            store.upsert_entity(build(record))
        added += 1
    return added, notes


def main(argv: Sequence[str] | None = None) -> int:
    """Seed an existing database, which the automatic pass will not."""
    import argparse
    import contextlib
    import io

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    from .store import EntityStore
    with contextlib.redirect_stdout(io.StringIO()):
        store = EntityStore()
    added, notes = seed(store, dry_run=args.dry_run)
    print(f"malignant compartment: {added} added, {len(notes)} already present")
    for note in notes:
        print("  ", note)
    if not args.dry_run:
        print(f"aliases attached to existing entities: {seed_aliases(store)}")
    if not args.dry_run and added:
        from .seed_synonyms import seed as seed_synonyms
        touched, syn, dropped = seed_synonyms(store)
        print(f"synonyms: {touched} entities, {syn} forms, {dropped} dropped")
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())


#: Name forms to attach to entities that already exist, where adding a
#: second entity would be a duplicate.
#:
#: ``KLK3`` is the gene for PSA, and this library already holds "PSA" --
#: as a METABOLITE. It is not one: UniProt P07288 is a 261-residue
#: serine protease, kallikrein-3. The row is a category error, probably
#: because PSA is read off a blood test and blood tests mostly report
#: metabolites. It is aliased rather than duplicated here because two
#: entities for one molecule is the worse failure, and the type error is
#: recorded so it can be fixed deliberately rather than discovered
#: again.
#:
#: Worth noting for anyone fixing it: a UniProt lookup of
#: ``gene_exact:KLK3`` returns P03952 FIRST, which is plasma kallikrein,
#: gene KLKB1 -- a different protein with a similar name. P07288 is the
#: one whose primary gene symbol reads back as KLK3. This is the same
#: shape as the FasL/fatty-acid-synthase mix-up this library already
#: shipped once.
ALIASES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("PSA", ("KLK3", "kallikrein-3", "prostate-specific antigen")),
)


def seed_aliases(store) -> int:
    """Attach the alias forms above to entities that already exist."""
    attached = 0
    for name, forms in ALIASES:
        entity = store.find_entity_by_name(name)
        if entity is None:
            continue
        existing = list(getattr(entity, "synonyms", None) or [])
        new = [f for f in forms if f.lower() not in {e.lower() for e in existing}]
        if not new:
            continue
        entity.synonyms = existing + new
        store.update_entity(entity, changed_by="seed_malignant_compartment")
        attached += len(new)
    return attached
