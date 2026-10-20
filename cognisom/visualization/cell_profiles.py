"""Which cell gets which detail, decided from the library's own records.

Eighteen of the fifty-six rendered library entities share one shape,
``smooth_sphere``: every CD4 and CD8 subset, both B cells, all three ILCs,
the gamma-delta T cell. They render identically, which is the same failure
as glucose and oxygen both being a sphere, one level up -- a Th1 and a Treg
are not the same object.

Morphologically they are genuinely close, and pretending otherwise by
inventing shapes would be worse than the sphere. What separates them is
recorded in the library already and read by nothing: every immune cell
carries ``surface_markers`` and ``cytokines_secreting``.

    CD4+ Th1        CD4, CXCR3, CCR5, T-bet
    CD4+ Th2        CD4, CCR4, CCR8, GATA3
    CD4+ Th17       CD4, CCR6, IL23R, RORgt
    CD8+ Effector   CD8, CD45RO, KLRG1, granzyme_B, perforin
    CD8+ Naive      CD8, CD45RA, CCR7, CD62L

THE MARKER LIST MIXES COMPARTMENTS
----------------------------------
The field is called ``surface_markers``, but T-bet, GATA3, RORgt and FOXP3
are transcription factors and live in the NUCLEUS, and granzyme_B,
perforin, MBP, EPO and histamine are the contents of cytoplasmic GRANULES.
Drawing all of them as studs on the membrane would be a visible error, and
it would also throw away the only real signal in the list: a cell that
carries perforin has lytic granules, and one that carries T-bet does not
get granules from it. So :data:`COMPARTMENT` places each molecule where it
actually resides, and the geometry follows from that.

This is what makes the subsets distinguishable without inventing anything:
a CD8 effector gets granules because the library says it holds perforin; a
CD8 naive does not, because it does not.

WHAT IS SOURCED AND WHAT IS NOT
-------------------------------
Each :class:`Profile` field carries its own note. The structural facts --
that lymphocytes bear microvilli, that a lymphocyte is mostly nucleus,
that a cytotoxic cell holds lytic granules -- are textbook. The specific
counts and sizes are, for now, declared here as ESTIMATED with the
reasoning, and :data:`UNSOURCED` lists them so no caller can mistake the
set for curated values. A sourcing pass over them is the obvious next
step, and :func:`profile_for` takes overrides so a sourced number can
replace an estimate without touching this mapping.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from cognisom.visualization import cell_detail as cd

log = logging.getLogger(__name__)

#: Where each molecule in the library's marker lists actually resides.
#: Anything absent is treated as a surface receptor, which is what the
#: field name claims, and logged once so the gap is visible rather than
#: assumed.
COMPARTMENT: Dict[str, str] = {
    # ── plasma membrane: receptors, co-receptors, chemokine receptors ──
    "CD3": "membrane", "CD4": "membrane", "CD8": "membrane",
    "CD19": "membrane", "CD20": "membrane", "CD56": "membrane",
    "CD11c": "membrane", "CD123": "membrane", "CD127": "membrane",
    "CD141": "membrane", "CD1c": "membrane", "CD193": "membrane",
    "CD203c": "membrane", "CD45RA": "membrane", "CD45RO": "membrane",
    "CD62L": "membrane", "CCR4": "membrane", "CCR5": "membrane",
    "CCR6": "membrane", "CCR7": "membrane", "CCR8": "membrane",
    "CXCR3": "membrane", "CXCR5": "membrane", "IL23R": "membrane",
    "KLRG1": "membrane", "NKG2D": "membrane", "TCRgd": "membrane",
    "FceRI": "membrane", "HLA-DR": "membrane", "Siglec-8": "membrane",
    "IgM": "membrane", "IgD": "membrane", "PD-1": "membrane",
    "CTLA-4": "membrane", "CD25": "membrane", "CD69": "membrane",
    # ── nucleus: lineage-defining transcription factors ──
    "T-bet": "nucleus", "GATA3": "nucleus", "RORgt": "nucleus",
    "FOXP3": "nucleus", "BCL6": "nucleus", "EOMES": "nucleus",
    "TCF1": "nucleus", "BLIMP1": "nucleus", "PAX5": "nucleus",
    # ── cytoplasmic granules: the stored cargo ──
    "granzyme_B": "granule", "granzyme_A": "granule", "perforin": "granule",
    "granulysin": "granule", "MBP": "granule", "EPO": "granule",
    "histamine": "granule", "tryptase": "granule", "myeloperoxidase": "granule",
    "elastase": "granule", "lactoferrin": "granule", "defensins": "granule",
}

#: Colours for the markers that appear most, so a CD4 reads as the same
#: blue on every cell that carries it. Anything else gets a stable colour
#: derived from its name: stable so two renders agree, derived so a new
#: marker needs no edit here.
MARKER_COLORS: Dict[str, Tuple[float, float, float]] = {
    "CD3": (0.95, 0.85, 0.35), "CD4": (0.30, 0.55, 0.95),
    "CD8": (0.92, 0.33, 0.33), "CD19": (0.35, 0.80, 0.45),
    "CD20": (0.30, 0.72, 0.52), "CD56": (0.72, 0.45, 0.88),
    "TCRgd": (0.98, 0.60, 0.25), "IgM": (0.40, 0.85, 0.80),
    "HLA-DR": (0.88, 0.70, 0.90), "CD11c": (0.70, 0.60, 0.35),
    "FceRI": (0.95, 0.50, 0.65), "NKG2D": (0.60, 0.80, 0.30),
    "CD45RA": (0.55, 0.75, 0.95), "CD45RO": (0.95, 0.65, 0.45),
}


def marker_color(marker: str) -> Tuple[float, float, float]:
    """A stable colour for a marker, curated where it matters."""
    if marker in MARKER_COLORS:
        return MARKER_COLORS[marker]
    # A hash the Python hash randomisation cannot move between runs.
    import hashlib
    digest = hashlib.sha256(marker.encode()).digest()
    hue = digest[0] / 255.0
    import colorsys
    return tuple(round(c, 4) for c in colorsys.hsv_to_rgb(hue, 0.52, 0.88))


#: The molecules whose presence implies stored lytic granules. A cell
#: holding these is a cell with granules to hold them in, which is how a
#: cytotoxic lymphocyte is told from a naive one without inventing a shape.
LYTIC_CARGO = frozenset({"granzyme_B", "granzyme_A", "perforin", "granulysin"})

#: Counts and sizes still declared here rather than sourced. Named so a
#: report cannot present them as curated. The microvillus and lytic
#: granule fields have left this set -- they are measured now, and the
#: citations sit on the constants they come from.
UNSOURCED: Tuple[str, ...] = (
    "mitochondrial_volume_fraction", "mitochondrion_radius_um",
    "lysosome_count", "lysosome_radius_um",
    "receptor_stud_count", "ruffle_amplitude_um",
)

#: Lytic granules in a primary human NK cell: 63 +/- 23 (PMID 29592963).
#: Explicitly NOT the 194-206 from the same paper's NK-92 and YTS lines,
#: which is what reviews quote as "almost 200 granules" -- immortalised
#: lines carrying three times the load.
LYTIC_GRANULES_PER_CELL = 63

#: Lytic granule radius, um. Diameter 0.25-0.33 across three methods and
#: three papers: 251 +/- 2 nm on primary human NK by 3D-SIM (PMID
#: 21931537), 333 +/- 103 nm on NK-92 by STED (PMID 21931536), and a
#: 3.3 fF capacitance step on primary human CD8 T cells (PMID 26296096),
#: which at 10 fF/um^2 works out at 324 nm.
LYTIC_GRANULE_RADIUS_UM = 0.145


@dataclass
class Profile:
    """The detail one cell gets, and why."""

    #: Projections standing off the membrane. Resting lymphocytes are
    #: densely covered in microvilli and they are functional, not
    #: decorative: TCR concentrates at microvillar tips and the villi
    #: sweep surfaces for antigen. Count and length ESTIMATED.
    microvilli_count: int = 0
    microvillus_length_um: float = 0.5
    microvillus_radius_um: float = 0.05

    #: Membrane ruffles, as a displacement rather than projections: a
    #: macrophage's surface undulates on a scale of a micrometre, where a
    #: lymphocyte's microvilli are a tenth of that.
    ruffle_amplitude_um: float = 0.0

    #: Interior populations, as a fraction of cytoplasmic volume where
    #: morphometry reports one, and a count where it reports a count.
    mitochondrial_volume_fraction: float = 0.0
    mitochondrion_radius_um: float = 0.22
    lysosome_count: int = 0
    lysosome_radius_um: float = 0.25
    lytic_granule_count: int = 0
    lytic_granule_radius_um: float = 0.25

    #: Membrane-embedded receptor studs. A depiction of the marker list,
    #: not a copy number: a T cell carries of order 10^4-10^5 TCRs and
    #: drawing them would be meaningless, so this is a legible density.
    receptor_stud_count: int = 0

    #: Studs per membrane marker, so the receptor complement is what the
    #: eye separates these cells by -- which is what actually
    #: distinguishes them: a CD4 T cell, a CD8 T cell and a CD19 B cell
    #: are told apart in a real laboratory by exactly this, and a single
    #: undifferentiated stud count throws the only signal away.
    membrane_stud_groups: Dict[str, int] = field(default_factory=dict)

    #: Which markers were placed where, for the record.
    membrane_markers: List[str] = field(default_factory=list)
    nuclear_markers: List[str] = field(default_factory=list)
    granule_markers: List[str] = field(default_factory=list)
    unknown_markers: List[str] = field(default_factory=list)

    #: Short account of why this cell got this detail.
    rationale: str = ""

    def to_dict(self) -> Dict[str, object]:
        out = {k: v for k, v in self.__dict__.items()}
        out["unsourced_fields"] = [f for f in UNSOURCED
                                   if getattr(self, f, 0)]
        return out


#: Lymphocyte: microvilli, a nucleus filling most of the cell, few
#: mitochondria and little else. The resting lymphocyte is the cell this
#: whole module exists for, because it is the eighteen-way collision.
#: A lymphocyte's surface grammar is DISCRETE MICROVILLI; a monocyte's
#: and a neutrophil's is RUFFLES (PMID 15130947). That contrast is the
#: one thing in this mapping that lets a viewer tell a lymphocyte from a
#: monocyte at a glance, so the two must not be mixed.
#:
#: The count is not stored here. It comes from the measured density and
#: whatever diameter the entity declares, so it stays right when the cell
#: changes size -- cell_detail.microvillus_count.
_LYMPHOCYTE = dict(
    microvillus_length_um=cd.MICROVILLUS_MEDIAN_LENGTH_UM,
    microvillus_radius_um=cd.MICROVILLUS_DIAMETER_UM / 2.0,
    mitochondrial_volume_fraction=0.038,
    mitochondrion_radius_um=0.15,
    receptor_stud_count=90,
)

#: Phagocyte: no microvilli to speak of, a ruffled surface, many
#: lysosomes, and a small nucleus in a large cytoplasm.
_PHAGOCYTE = dict(
    microvilli_count=0,
    ruffle_amplitude_um=0.35,
    mitochondrial_volume_fraction=0.06,
    mitochondrion_radius_um=0.26,
    lysosome_count=40,
    lysosome_radius_um=0.30,
    receptor_stud_count=120,
)

#: mesh_type -> the class of cell it is, for detail purposes.
_CLASS_BY_MESH: Dict[str, str] = {
    "smooth_sphere": "lymphocyte",
    "granular_sphere": "lymphocyte",
    "ruffled_pseudopod": "phagocyte",
    "irregular_ruffled": "phagocyte",
    "stellate_branching": "dendritic",
    "dense_granular": "granulocyte",
    "bilobed_granular": "granulocyte",
    "multilobed_granular": "granulocyte",
    "eccentric_nucleus_er": "plasma",
    "elongated_spindle": "mesenchymal",
    "flat_squamous": "endothelial",
    "flat_activated_adhesion": "endothelial",
    "cuboidal": "epithelial",
    "cuboidal_tall": "epithelial",
    "polarized_columnar": "epithelial",
}


def classify(mesh_type: str) -> str:
    """Which detail class a ``mesh_type`` belongs to."""
    return _CLASS_BY_MESH.get(mesh_type, "other")


def partition_markers(markers: Sequence[str]) -> Dict[str, List[str]]:
    """Split a marker list by where each molecule actually resides."""
    out: Dict[str, List[str]] = {"membrane": [], "nucleus": [],
                                 "granule": [], "unknown": []}
    for marker in markers or ():
        where = COMPARTMENT.get(marker)
        if where is None:
            out["unknown"].append(marker)
        else:
            out[where].append(marker)
    return out


def profile_for(name: str, mesh_type: str, markers: Sequence[str] = (),
                scale_um: float = 8.0, **overrides) -> Profile:
    """The detail profile for one library entity.

    ``scale_um`` is the cell's diameter, and it is not cosmetic: the
    microvillus count is a measured DENSITY times this cell's own surface
    area, so a profile built without it coats every cell as though it
    were 8 um across.

    ``overrides`` replaces any declared estimate, which is how a sourced
    number arrives without this mapping having to change.
    """
    kind = classify(mesh_type)
    base: Dict[str, object] = {}
    why: List[str] = []

    if kind in ("lymphocyte", "plasma"):
        base.update(_LYMPHOCYTE)
        base["microvilli_count"] = cd.microvillus_count(scale_um)
        why.append(f"lymphocyte: {base['microvilli_count']} microvilli, from "
                   f"a measured {cd.MICROVILLUS_DENSITY_PER_UM2:g}/um^2 over "
                   f"this cell's own {scale_um:g} um")
    elif kind in ("phagocyte", "dendritic"):
        base.update(_PHAGOCYTE)
        why.append("phagocyte: ruffles, not microvilli -- the surface "
                   "grammar that separates a monocyte from a lymphocyte "
                   "at a glance")
    elif kind == "granulocyte":
        base.update(mitochondrial_volume_fraction=0.04,
                    mitochondrion_radius_um=0.2,
                    receptor_stud_count=80,
                    ruffle_amplitude_um=0.2)
        why.append("granulocyte: the granules are built by the mesh_type "
                   "itself, so only the surface and mitochondria are added")
    elif kind in ("endothelial", "epithelial", "mesenchymal"):
        base.update(microvilli_count=0,
                    mitochondrial_volume_fraction=0.05,
                    mitochondrion_radius_um=0.26,
                    lysosome_count=12,
                    receptor_stud_count=60)
        why.append(f"{kind}: a smooth surface, mitochondria and few "
                   "lysosomes")

    placed = partition_markers(markers)
    if placed["granule"] and set(placed["granule"]) & LYTIC_CARGO:
        cargo = sorted(set(placed["granule"]) & LYTIC_CARGO)
        base["lytic_granule_count"] = LYTIC_GRANULES_PER_CELL
        base["lytic_granule_radius_um"] = LYTIC_GRANULE_RADIUS_UM
        why.append("carries " + ", ".join(cargo) +
                   ", so it holds lytic granules")
    if placed["nucleus"]:
        why.append("nuclear markers " + ", ".join(sorted(placed["nucleus"])) +
                   " are transcription factors, drawn in the nucleus and "
                   "not on the membrane")
    if placed["unknown"]:
        log.debug("%s: markers with no recorded compartment: %s",
                  name, placed["unknown"])

    base.update({k: v for k, v in overrides.items() if v is not None})
    profile = Profile(**base)
    # Spread the stud budget across the receptors the cell carries, so the
    # complement is visible rather than a single anonymous density.
    receptors = placed["membrane"] + placed["unknown"]
    total = int(base.get("receptor_stud_count", 0) or 0)
    if receptors and total:
        share, extra = divmod(total, len(receptors))
        profile.membrane_stud_groups = {
            m: share + (1 if i < extra else 0)
            for i, m in enumerate(sorted(receptors))
        }
    profile.membrane_markers = placed["membrane"]
    profile.nuclear_markers = placed["nucleus"]
    profile.granule_markers = placed["granule"]
    profile.unknown_markers = placed["unknown"]
    profile.rationale = "; ".join(why) or "no detail class matched"
    return profile


def markers_of(entity_json: str | dict) -> List[str]:
    """The marker list off a library entity record, under either name."""
    data = json.loads(entity_json) if isinstance(entity_json, str) \
        else (entity_json or {})
    props = data.get("properties") or {}
    markers = props.get("surface_markers") or props.get("markers") or []
    return [str(m) for m in markers]
