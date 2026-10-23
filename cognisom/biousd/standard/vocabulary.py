"""The Bio-USD vocabulary, as data.

Everything else in this package reads this module: the writer authors
exactly these primvars, the reader looks for exactly them, the
validator checks exactly them, and SPEC.md is rendered from them. A
vocabulary that lives in prose drifts from the code that implements it
-- this project's own writer and schema file disagreed on one attribute
each by the time anyone compared them -- so here it lives in one place
and the prose is generated.

WHAT BIO-USD IS
---------------
A convention for carrying a population of biological cells in an
OpenUSD stage so that any OpenUSD tool can draw it and any
ontology-aware tool can join it to other data. It adds no prim types.
Cells are instances of a ``UsdGeom.PointInstancer``; their biology is a
set of ``varying`` primvars on that instancer, one value per instance.
That is the whole mechanism, and it is why a Bio-USD file opens in
usdview, Omniverse, Houdini or Blender with nothing installed.

Identity is the part that reaches outside this file. ``bio:cell:type``
is a coarse token for renderers; ``bio:cell:ontologyTerm`` is a Cell
Ontology or NCI Thesaurus CURIE, the identifier CZ CELLxGENE, the Human
Cell Atlas and Tabula Sapiens annotate with, so a scene written here
can be joined to their 150 million cells without a translation table.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

#: Semantic version of the vocabulary. A stage records the version it
#: was written to in its layer data as ``bio:version``.
VERSION = "0.1.2"

#: One stage unit is one micrometre. Cell positions and diameters are
#: in micrometres; this is what ``metersPerUnit`` must say.
METERS_PER_UNIT = 1e-6

#: The ontology prefixes a ``bio:cell:ontologyTerm`` may carry, and
#: where each resolves. Both are OBO-registered. The value is a CURIE
#: such as ``CL:0000625``; an empty string means "identity not
#: asserted", which is honest and a reader can handle, where a wrong
#: term is a citation that will be believed.
CURIE_PREFIXES: Dict[str, str] = {
    "CL": "http://purl.obolibrary.org/obo/CL_",
    "NCIT": "http://purl.obolibrary.org/obo/NCIT_",
}
CURIE_RE = re.compile(r"^(CL|NCIT):[0-9A-Za-z]+$")

#: Coarse lineage tokens, for a renderer that has only the token.
#: ``normal`` is a state, not a lineage, and carries no ontology term;
#: it is kept because every engine has cells it has not typed further.
CELL_TYPES: Tuple[str, ...] = (
    "normal", "cancer", "immune", "stromal", "endothelial",
    "basal", "luminal", "neuroendocrine", "stem", "fibroblast",
)

#: Cell-cycle phases.
CELL_PHASES: Tuple[str, ...] = ("G0", "G1", "S", "G2", "M")


@dataclass(frozen=True)
class Primvar:
    """One per-instance attribute of the convention."""

    name: str
    usd_type: str           # the Sdf value type name, e.g. "int[]"
    required: bool
    description: str
    tokens: Optional[Tuple[str, ...]] = None
    units: str = ""
    minimum: Optional[float] = None

    #: Every Bio-USD primvar is per instance.
    interpolation: str = "varying"


#: The per-cell primvars, in the order a writer authors them. Only the
#: first two are required for conformance; the rest are typed when
#: present.
CELL_PRIMVARS: Tuple[Primvar, ...] = (
    Primvar("bio:cell:id", "int[]", True,
            "Stable identifier, unique within the stage."),
    Primvar("bio:cell:type", "token[]", True,
            "Coarse lineage, for a consumer that has only the token.",
            tokens=CELL_TYPES),
    Primvar("bio:cell:ontologyTerm", "token[]", False,
            "Cell Ontology (CL) or NCI Thesaurus (NCIT) CURIE naming the "
            "cell's identity; empty when not asserted."),
    Primvar("bio:cell:phase", "token[]", False,
            "Cell-cycle phase.", tokens=CELL_PHASES),
    Primvar("bio:cell:age", "float[]", False,
            "Time since the cell's last division.", units="h", minimum=0.0),
    Primvar("bio:cell:alive", "bool[]", False,
            "Viability. A dead cell may be kept in the stage for an "
            "animation; a consumer that counts cells should check it."),
    Primvar("bio:cell:volume", "float[]", False,
            "Volume relative to the type's resting volume (1.0 = resting). "
            "The instance scale carries the absolute diameter in um.",
            minimum=0.0),
)

PRIMVARS_BY_NAME: Dict[str, Primvar] = {p.name: p for p in CELL_PRIMVARS}
REQUIRED_PRIMVARS: Tuple[str, ...] = tuple(
    p.name for p in CELL_PRIMVARS if p.required)

#: Keys a writer may set in the root layer's ``customLayerData``.
LAYER_DATA: Dict[str, str] = {
    "bio:version": "The vocabulary version the stage was written to. Required.",
    "bio:colourBy": "Name of the per-cell field driving displayColor, if one is.",
    "bio:colourMin": "Lower end of the colour ramp, in that field's units.",
    "bio:colourMax": "Upper end of the colour ramp, in that field's units.",
}

#: The geometric carrier.
CARRIER = {
    "prim": "UsdGeom.PointInstancer",
    "positions": "cell centres, um",
    "scales": "cell diameter, um, applied to a prototype authored at "
              "unit diameter (its body spans 1.0 across)",
    "protoIndices": "which prototype each cell draws; prototypes are "
                    "free-form geometry and are not part of this "
                    "vocabulary",
    "invisibleIds": "may be used to hide cells for a cutaway; hidden cells "
                    "are still cells",
}


def is_curie(value: str) -> bool:
    """Whether ``value`` is an acceptable ontology term (empty counts)."""
    return value == "" or bool(CURIE_RE.match(value))


def iri_for(curie: str) -> Optional[str]:
    """Expand a CURIE to the IRI it resolves at, or ``None``."""
    if not curie or ":" not in curie:
        return None
    prefix, local = curie.split(":", 1)
    base = CURIE_PREFIXES.get(prefix)
    return f"{base}{local}" if base else None


def render_spec_markdown() -> str:
    """SPEC.md, rendered from the tables above so it cannot drift."""
    lines = [
        f"# Bio-USD {VERSION} — specification",
        "",
        "Bio-USD is a convention for carrying a population of biological cells "
        "in an OpenUSD stage. It adds no prim types: cells are instances of a "
        "`UsdGeom.PointInstancer`, and their biology is a set of `varying` "
        "primvars on that instancer, one value per instance. Any OpenUSD tool "
        "draws it unmodified; an ontology-aware tool can join it to other data "
        "through the Cell Ontology term each cell may carry.",
        "",
        "This file is generated from `vocabulary.py`. Edit that, not this.",
        "",
        "## Stage",
        "",
        f"- `metersPerUnit` must be `{METERS_PER_UNIT:g}`: one stage unit is one micrometre.",
        "- `upAxis` may be `Y` or `Z` and must be declared.",
        "- The root layer's `customLayerData` carries:",
        "",
        "| key | meaning |",
        "|---|---|",
    ]
    for key, meaning in LAYER_DATA.items():
        lines.append(f"| `{key}` | {meaning} |")
    lines += [
        "",
        "## Carrier",
        "",
        f"The carrier is a `{CARRIER['prim']}`. A stage may hold several; each "
        "that carries `bio:cell:id` is a cell population.",
        "",
        "| attribute | meaning |",
        "|---|---|",
    ]
    for key in ("positions", "scales", "protoIndices", "invisibleIds"):
        lines.append(f"| `{key}` | {CARRIER[key]} |")
    lines += [
        "",
        "## Per-cell primvars",
        "",
        "All are `varying` (one value per instance) and must have exactly as "
        "many values as there are positions. **Required** means required for "
        "conformance; every other primvar is optional but, when present, must "
        "have the type and values given here.",
        "",
        "| name | type | required | units | allowed values | meaning |",
        "|---|---|---|---|---|---|",
    ]
    for p in CELL_PRIMVARS:
        allowed = ", ".join(f"`{t}`" for t in p.tokens) if p.tokens else (
            "CURIE `CL:…` or `NCIT:…`, or empty" if p.name == "bio:cell:ontologyTerm"
            else (f"≥ {p.minimum:g}" if p.minimum is not None else "—"))
        lines.append(
            f"| `{p.name}` | `{p.usd_type}` | {'**yes**' if p.required else 'no'} "
            f"| {p.units or '—'} | {allowed} | {p.description} |")
    lines += [
        "",
        "## Ontology terms",
        "",
        "`bio:cell:ontologyTerm` is a CURIE from one of these prefixes, "
        "resolved by exact label match against the ontology and never by "
        "fuzzy search (a fuzzy match returns a wrong tissue with the same "
        "confidence as a right one). An empty value means the identity is not "
        "asserted.",
        "",
        "| prefix | resolves at |",
        "|---|---|",
    ]
    for prefix, base in CURIE_PREFIXES.items():
        lines.append(f"| `{prefix}` | `{base}` |")
    lines += [
        "",
        "## Conformance",
        "",
        "A stage conforms when it opens, declares the stage metadata above, "
        "holds at least one carrier with the required primvars, every present "
        "primvar has the stated type and `varying` interpolation and as many "
        "values as positions, every token is in its allowed set, every "
        "ontology term is a well-formed CURIE or empty, and `bio:cell:id` is "
        "unique within the carrier. `python -m biousd.validate <file>` checks "
        "exactly this and nothing else.",
        "",
        "## What this is not",
        "",
        "Not a schema of prim types (a codeless schema may follow; nothing here "
        "needs one). Not a simulation format: it says what cells are and where, "
        "not how they got there. Not a rival to MultiCellDS, the community "
        "standard for multicellular data: a MultiCellDS digital snapshot's cell "
        "positions, cycle phase, volume and cell type map onto the primvars "
        "above, and the ontology term is the join between the two.",
        "",
    ]
    return "\n".join(lines)
