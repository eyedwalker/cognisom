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
import re
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

#: The six places a marker can be drawn. ``unknown`` is not a place; it
#: is an admission, and it is kept separate so it can be counted.
BUCKETS: Tuple[str, ...] = ("membrane", "nucleus", "granule", "cytoplasm",
                            "secreted", "unknown")

#: The library's OWN compartment vocabulary, mapped to those buckets.
#:
#: :data:`COMPARTMENT` above is a hand-written table of marker strings, and
#: it does not scale: 115 of the 173 marker strings this library refers to
#: are not in it, and every one of them was drawn as a stud on the plasma
#: membrane. That is not a missing entry here and there -- it is 66% of
#: the vocabulary, and it put nuclear transcription factors, granule
#: cargo, cytoskeleton and secreted chemokines on the cell surface.
#:
#: The entities already record where they live. 188 of 302 carry a
#: ``compartments`` field, in 23 distinct strings, and nothing read it.
#: So the table stays as the hand-checked authority and this map lets the
#: library answer for everything the table never mentioned.
#:
#: Two readings deserve stating. An INTERNAL membrane is not the plasma
#: membrane: a lysosomal or mitochondrial membrane protein is inside the
#: cell, and drawing it as a surface receptor is the same error one
#: organelle further in, so those map to ``cytoplasm``. And
#: ``extracellular`` maps to ``secreted`` only because it is safe to here:
#: it is carried by cytokines, drugs, metabolites, complement, antibodies
#: and soluble proteins, and by no receptor or adhesion molecule in this
#: library. If that ever stops being true this map is wrong.
LIBRARY_COMPARTMENT: Dict[str, str] = {
    # the plasma membrane itself, and structures in it
    "cell_membrane": "membrane", "plasma membrane": "membrane",
    "plasma_membrane": "membrane", "cell membrane": "membrane",
    "target plasma membrane": "membrane",
    "killer cell plasma membrane": "membrane",
    "adherens_junction": "membrane", "focal_adhesions": "membrane",
    "invadopodia": "membrane", "tight_junction": "membrane",
    # the DISC assembles ON the membrane, around a ligated death receptor
    "DISC": "membrane",
    # nucleus
    "nucleus": "nucleus", "nucleolus": "nucleus", "spindle": "nucleus",
    "chromatin": "nucleus",
    # stored cargo -- the bucket that implies granules exist to store it
    "cytotoxic granule": "granule", "granule": "granule",
    "azurophilic granule": "granule", "secretory granule": "granule",
    "specific granule": "granule", "weibel-palade body": "granule",
    # the interior: cytosol, and the organelles and their membranes
    "cytoplasm": "cytoplasm", "cytosol": "cytoplasm",
    "target cytosol": "cytoplasm", "mitochondria": "cytoplasm",
    "mitochondrion": "cytoplasm",
    "mitochondrial_outer_membrane": "cytoplasm",
    "mitochondrial_intermembrane_space": "cytoplasm",
    "mitochondrial_matrix": "cytoplasm",
    "lysosome": "cytoplasm", "lysosome_membrane": "cytoplasm",
    "endosome": "cytoplasm", "centrosome": "cytoplasm",
    "apoptosome": "cytoplasm", "cytoskeleton": "cytoplasm",
    "endoplasmic_reticulum": "cytoplasm", "golgi": "cytoplasm",
    "er membrane": "cytoplasm", "peroxisome": "cytoplasm",
    # outside the cell
    "extracellular": "secreted", "secreted": "secreted",
    "blood": "secreted", "plasma": "secreted", "serum": "secreted",
    "ECM": "secreted", "extracellular_matrix": "secreted",
    "basement_membrane": "secreted",
}

#: Which bucket wins when one entity declares several, most decisive
#: first. A molecule stored in a granule and released onto a target
#: membrane is granule cargo: that is what shapes the cell holding it.
#: Membrane beats cytoplasm so a receptor with a cytoplasmic tail stays a
#: receptor, and secreted is last because almost anything soluble also
#: passes through the cytosol on its way out.
BUCKET_PRECEDENCE: Tuple[str, ...] = ("granule", "nucleus", "membrane",
                                      "cytoplasm", "secreted")


def bucket_for_compartments(compartments: Sequence[str]) -> Optional[str]:
    """The one bucket a list of library compartment strings resolves to."""
    found = set()
    for raw in compartments or ():
        key = str(raw).strip()
        bucket = (LIBRARY_COMPARTMENT.get(key)
                  or LIBRARY_COMPARTMENT.get(key.lower())
                  or LIBRARY_COMPARTMENT.get(key.replace(" ", "_").lower()))
        if bucket:
            found.add(bucket)
    for bucket in BUCKET_PRECEDENCE:
        if bucket in found:
            return bucket
    return None


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


#: MARKER_COLORS keyed for lookup, built once.
_COLOR_BY_KEY = None


def marker_color(marker: str) -> Tuple[float, float, float]:
    """A stable colour for a marker, curated where it matters.

    Keyed through the same normalisation as the compartment table, so
    ``CD8`` and ``CD8+`` are one molecule and get one colour. They did
    not: ``CD8`` took the curated red and ``CD8+`` fell through to a
    hashed green, so the same receptor was two colours on two cells and
    the convention broke on the cell named for it.
    """
    global _COLOR_BY_KEY
    if _COLOR_BY_KEY is None:
        _COLOR_BY_KEY = {_canonical(k): v for k, v in MARKER_COLORS.items()}
    key = _canonical(marker)
    if key in _COLOR_BY_KEY:
        return _COLOR_BY_KEY[key]
    # Hash the CANONICAL key, not the name as written: hashing the name
    # gave granzyme_B and "granzyme B+" two different colours for one
    # molecule, which is the bug the curated table had one level up.
    import hashlib
    digest = hashlib.sha256(key.encode()).digest()
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
    #: Long axis as a multiple of the short. A mitochondrion is not
    #: round, and drawn as a sphere it cannot be told from a granule or a
    #: vesicle -- which is part of why a cytotoxic T cell missing all its
    #: lytic granules did not look obviously wrong.
    mitochondrion_elongation: float = 2.2
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
    #: Cytoskeleton, enzymes and internal-membrane proteins. Drawn inside
    #: the cell, which for a vimentin or a cytokeratin is the whole
    #: point: they are the markers that say mesenchymal or epithelial,
    #: and on the surface they say nothing.
    cytoplasmic_markers: List[str] = field(default_factory=list)
    #: Chemokines, interleukins and matrix proteins. NOT drawn on this
    #: cell at all -- a secreted molecule has left. Recorded so the
    #: secretory repertoire stays legible without being painted onto the
    #: membrane it crossed.
    secreted_markers: List[str] = field(default_factory=list)
    unknown_markers: List[str] = field(default_factory=list)

    #: Short account of why this cell got this detail.
    rationale: str = ""

    #: Where each field's value came from: "library" when it was read off
    #: a provenanced entity parameter, "default" when it fell back to the
    #: constant in this module. Without this the two cannot be told apart
    #: in a report, and a default would pass for evidence.
    sources: Dict[str, str] = field(default_factory=dict)

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


#: Trailing flow-cytometry notation: a level, not part of the name.
_LEVEL = re.compile(r"(hi|lo|low|high|int|dim|bright|\+|-)$", re.I)

#: A trailing word describing the marker's STATE on this cell rather than
#: naming a different molecule. ``VCAM1_inducible`` is VCAM-1, and
#: ``CXCL8_displayed`` is CXCL8 presented on the surface instead of
#: released -- both are the molecule the library already holds.
#:
#: Six marker strings failed to resolve on this alone: ICAM-1_,
#: VCAM-1_, ICAM1_inducible, VCAM1_inducible, E-selectin_inducible and
#: CXCL8_displayed, every one of which has an entity. It is the same
#: failure as CD8 against CD8+, one qualifier along, and it has the same
#: consequence -- the molecule is counted absent and drawn as an
#: anonymous stud.
_STATE = re.compile(
    r"[_\s-]+(inducible|displayed|induced|soluble|bound|surface|"
    r"secreted|cleaved|activated|total|cytoplasmic|nuclear)$", re.I)


def normalise_marker(marker: str) -> str:
    """The molecule underneath a marker string as the library writes it.

    The library writes the same molecule several ways. ``CD8+ T cell``
    carries ``perforin+`` and ``granzyme B+``; ``CD8+ Effector T cell``
    carries ``perforin`` and ``granzyme_B``. Looking these up literally
    made the compartment table miss on the first and hit on the second,
    so a cytotoxic T cell got no lytic granules and had its perforin and
    granzyme B -- soluble cargo stored inside granules -- drawn as studs
    on the OUTSIDE of its plasma membrane. On the one cell whose whole
    identity is that it kills by degranulating.

    The suffix is an expression LEVEL (``CD25hi``, ``CD127lo``, ``AR-``),
    not part of the molecule's name, and the separator between words is
    arbitrary. Both are stripped before any lookup.
    """
    out = str(marker).strip()
    # Strip repeatedly: "CD127lo+" occurs.
    for _ in range(3):
        stripped = _STATE.sub("", _LEVEL.sub("", out)).strip()
        # A trailing separator is left behind by the library's own
        # "ICAM-1_" and would otherwise survive into the lookup key.
        stripped = stripped.rstrip("_- ")
        if stripped == out:
            break
        out = stripped
    return out


def _canonical(marker: str) -> str:
    """A lookup key: normalised, separator- and case-insensitive."""
    return re.sub(r"[\s_\-]+", "", normalise_marker(marker)).lower()


#: The compartment table, keyed for lookup rather than for reading.
_COMPARTMENT_BY_KEY = None


def compartment_of(marker: str,
                   index: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Where this molecule resides, however the marker is written.

    The hand-checked :data:`COMPARTMENT` table answers first, because it
    was written deliberately and a curated placement should not be
    overturned by a bulk import. ``index`` -- from
    :func:`library_compartment_index` -- answers for everything the table
    never mentioned, which is most of the vocabulary.
    """
    global _COMPARTMENT_BY_KEY
    if _COMPARTMENT_BY_KEY is None:
        _COMPARTMENT_BY_KEY = {_canonical(k): v
                               for k, v in COMPARTMENT.items()}
    key = _canonical(marker)
    found = _COMPARTMENT_BY_KEY.get(key)
    if found is not None:
        return found
    return (index or {}).get(key)


def library_compartment_index(store) -> Dict[str, str]:
    """``canonical marker key -> bucket``, from the entities themselves.

    Built from every name form an entity answers to, so a marker list
    saying ``CCL19`` reaches the entity named ``CCL19 (MIP-3beta)`` and
    is placed where that entity says it lives.

    A form two entities claim is DROPPED when they disagree about the
    bucket, and kept when they agree. Guessing between them would put a
    molecule somewhere it is not, which is the failure this whole
    function exists to stop; agreeing is not a guess.
    """
    import json as _json

    from cognisom.library import seed_synonyms as _syn

    rows = [tuple(r) for r in store._conn.execute(
        "SELECT name, data_json FROM entities")]
    claims: Dict[str, set] = {}
    for name, blob in rows:
        data = _json.loads(blob or "{}")
        bucket = bucket_for_compartments(data.get("compartments") or ())
        if bucket is None:
            continue
        forms = {name} | set(data.get("synonyms") or [])
        forms |= set(_syn.derive(name, data.get("properties") or {}))
        for form in forms:
            key = _canonical(form)
            if key:
                claims.setdefault(key, set()).add(bucket)
    return {key: next(iter(buckets)) for key, buckets in claims.items()
            if len(buckets) == 1}


def partition_markers(markers: Sequence[str],
                      index: Optional[Dict[str, str]] = None
                      ) -> Dict[str, List[str]]:
    """Split a marker list by where each molecule actually resides."""
    out: Dict[str, List[str]] = {bucket: [] for bucket in BUCKETS}
    for marker in markers or ():
        where = compartment_of(marker, index)
        out[where if where in out else "unknown"].append(str(marker))
    return out


#: Library parameter -> the Profile field it fills. These are the values
#: that have been sourced and written onto the entities themselves (see
#: ``library/seed_ultrastructure.py``); everything else here is still a
#: declared estimate.
FROM_LIBRARY: Dict[str, str] = {
    "microvillus_length_um": "microvillus_length_um",
    "lytic_granule_count": "lytic_granule_count",
}


def library_detail(entity) -> Dict[str, float]:
    """The provenanced ultrastructure an entity carries, by parameter.

    Reads only MEASURED and CALIBRATED values: a placeholder sitting in
    the right field is not evidence, and silently preferring one over
    this module's declared estimate would hide that.
    """
    from cognisom.library.parameters import Provenance, get_parameter

    out: Dict[str, float] = {}
    if entity is None:
        return out
    for parameter in ("nuclear_volume_fraction", "granule_volume_fraction",
                      "microvillus_density_per_um2", "microvillus_length_um",
                      "microvillus_diameter_um", "lytic_granule_count",
                      "lytic_granule_diameter_um"):
        pv = get_parameter(entity, parameter)
        if pv is None or pv.provenance not in (Provenance.MEASURED,
                                               Provenance.CALIBRATED):
            continue
        try:
            out[parameter] = float(pv.value)
        except (TypeError, ValueError):
            continue
    return out


def profile_for(name: str, mesh_type: str, markers: Sequence[str] = (),
                scale_um: float = 8.0,
                library: Optional[Dict[str, float]] = None,
                compartments: Optional[Dict[str, str]] = None,
                **overrides) -> Profile:
    """The detail profile for one library entity.

    ``scale_um`` is the cell's diameter, and it is not cosmetic: the
    microvillus count is a measured DENSITY times this cell's own surface
    area, so a profile built without it coats every cell as though it
    were 8 um across.

    ``library`` is what the entity itself carries, from
    :func:`library_detail`. Where a parameter is there it WINS over the
    constants in this module, because those are declared estimates and
    these are measured values with citations attached. The profile
    records which of the two each field came from.

    ``overrides`` replaces either, which is how a caller states something
    this mapping cannot know.
    """
    kind = classify(mesh_type)
    library = library or {}
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

    placed = partition_markers(markers, compartments)
    lytic_keys = {_canonical(c) for c in LYTIC_CARGO}
    carried = sorted(m for m in placed["granule"]
                     if _canonical(m) in lytic_keys)
    if carried:
        cargo = carried
        base["lytic_granule_count"] = LYTIC_GRANULES_PER_CELL
        base["lytic_granule_radius_um"] = LYTIC_GRANULE_RADIUS_UM
        why.append("carries " + ", ".join(cargo) +
                   ", so it holds lytic granules")
    if placed["nucleus"]:
        why.append("nuclear markers " + ", ".join(sorted(placed["nucleus"])) +
                   " are transcription factors, drawn in the nucleus and "
                   "not on the membrane")
    if placed["cytoplasm"]:
        why.append("cytoplasmic markers " +
                   ", ".join(sorted(placed["cytoplasm"])) +
                   " are drawn inside the cell, not on it")
    if placed["secreted"]:
        why.append("secreted markers " +
                   ", ".join(sorted(placed["secreted"])) +
                   " are not drawn on this cell: they have left it")
    if placed["unknown"]:
        log.debug("%s: markers with no recorded compartment: %s",
                  name, placed["unknown"])

    # The library's measured values displace this module's estimates.
    sources = {k: "default" for k in base}
    if "microvillus_density_per_um2" in library and kind in ("lymphocyte",
                                                             "plasma"):
        base["microvilli_count"] = cd.microvillus_count(
            scale_um, library["microvillus_density_per_um2"])
        sources["microvilli_count"] = "library"
        why.append(f"microvillus density {library['microvillus_density_per_um2']:g}"
                   "/um^2 read from the entity, not from this module")
    if "microvillus_length_um" in library and base.get("microvilli_count"):
        base["microvillus_length_um"] = library["microvillus_length_um"]
        sources["microvillus_length_um"] = "library"
    if "microvillus_diameter_um" in library and base.get("microvilli_count"):
        base["microvillus_radius_um"] = library["microvillus_diameter_um"] / 2.0
        sources["microvillus_radius_um"] = "library"
    if "lytic_granule_count" in library:
        base["lytic_granule_count"] = int(round(library["lytic_granule_count"]))
        sources["lytic_granule_count"] = "library"
    if "lytic_granule_diameter_um" in library:
        base["lytic_granule_radius_um"] = \
            library["lytic_granule_diameter_um"] / 2.0
        sources["lytic_granule_radius_um"] = "library"

    base.update({k: v for k, v in overrides.items() if v is not None})
    for k in overrides:
        if overrides[k] is not None:
            sources[k] = "override"
    profile = Profile(**base)
    profile.sources = sources
    # Spread the stud budget across the receptors the cell carries, so the
    # complement is visible rather than a single anonymous density.
    # Unknown markers still go on the membrane: the field is called
    # surface_markers and that is the declared default. What changed is
    # that unknown no longer means "not in a hand-written table" -- the
    # library answers first, so what lands here is genuinely unplaced.
    # Cytoplasmic and secreted markers are deliberately NOT here. A
    # vimentin drawn as a surface stud is the error this module exists
    # to prevent, one compartment further out than the perforin it
    # already prevents.
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
    profile.cytoplasmic_markers = placed["cytoplasm"]
    profile.secreted_markers = placed["secreted"]
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
