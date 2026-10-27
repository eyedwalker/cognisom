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
VERSION = "0.2.0"

#: The codeless applied API schema that declares the primvars below on a
#: PointInstancer (since 0.2.0). Its identifier is what a file carries in
#: ``apiSchemas``; its TfType is LIBRARY_NAME proper-cased plus the
#: identifier, ``BioUsdBioCellAPI``, which is what usdGenSchema derives
#: and what the plugin's ``plugInfo.json`` declares. Schema-family
#: versioning (``BioCellAPI_1``) is reserved for an incompatible change
#: and is independent of VERSION above: this vocabulary version knows
#: version 0 of the family only.
API_SCHEMA = "BioCellAPI"
LIBRARY_NAME = "bioUsd"
CARRIER_PRIM_TYPE = "PointInstancer"

#: The version from which a writer is expected to record UsdProfiles
#: claims (see :mod:`.profiles`); an older file without them is not
#: warned about, because the convention did not exist when it was written.
CLAIMS_SINCE = (0, 2)

#: UsdProfiles capability names the plugin declares (OpenUSD 26.08+
#: reads them; earlier versions ignore the block). Reverse-domain tokens
#: under the one mandatory root ``usd``. :mod:`.profiles` carries their
#: documentation and the graph edges; the names live here so SPEC.md can
#: be rendered from this module alone.
CAPABILITY_ROOT = "usd"
CAPABILITY_BIOUSD = "biousd"
CAPABILITY_CELLS = "biousd.cells"
CAPABILITY_ONTOLOGY = "biousd.ontology"
PROFILE_V1 = "biousd.profile_v1"

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


#: Every finding the validator can report, by code: (level, meaning).
#: ``error`` means the stage does not conform; ``warning`` means it does
#: and something is worth knowing. Two codes carry either level: the
#: meaning says which case is which. The framework validators
#: (:mod:`.validators`) report the same codes as CamelCase error names,
#: so a finding is one thing whichever path found it.
VALIDATOR_CODES: Dict[str, Tuple[str, str]] = {
    "stage.open": ("error", "The file could not be opened as a USD stage (exit status 2)."),
    "stage.metersPerUnit": ("error", f"`metersPerUnit` is not `{METERS_PER_UNIT:g}`."),
    "stage.upAxis": ("error", "`upAxis` is not `Y` or `Z`."),
    "layer.version": ("error / warning",
                      "Error: `customLayerData` has no `bio:version`. Warning: it names "
                      "a different major version than this validator."),
    "layer.colourRange": ("warning", "`bio:colourBy` is set without `bio:colourMin`/`Max`."),
    "carrier.none": ("error", "No PointInstancer is a cell population."),
    "carrier.protoIndices": ("error", "`protoIndices` has a different length than `positions`."),
    "carrier.prototypes": ("error", "A `protoIndex` is outside the prototypes relationship."),
    "carrier.scales": ("error", "A cell has a non-positive diameter."),
    "carrier.invisibleIds": ("error", "An `invisibleIds` entry is not an instance."),
    "primvar.missing": ("error", "A required primvar is not authored."),
    "primvar.type": ("error", "A primvar has a value type other than the one stated."),
    "primvar.interpolation": ("error", "A primvar does not author `varying` interpolation "
                              "(unauthored, it is `constant` to a reader without the plugin; "
                              "the schema fallback is not the file)."),
    "primvar.length": ("error", "A primvar does not have one value per instance."),
    "primvar.token": ("error", "A token value is outside its allowed set."),
    "primvar.range": ("error", "A numeric value is below its minimum."),
    "primvar.curie": ("error / warning",
                      "Error: an ontology term is not a `CL:`/`NCIT:` CURIE. Warning: no "
                      "cell asserts a term."),
    "primvar.id": ("error", "`bio:cell:id` is not unique within the carrier."),
    "primvar.optional": ("warning", "An optional primvar is not authored."),
    "schema.misapplied": ("error", f"`{API_SCHEMA}` is applied to a prim that is not a "
                          f"`{CARRIER_PRIM_TYPE}`."),
    "schema.notApplied": ("warning", f"A cell population does not apply `{API_SCHEMA}` "
                          "(a 0.1 file; still conforming)."),
    "schema.version": ("warning", f"A versioned `{API_SCHEMA}_<n>` is applied; this validator "
                       f"knows `{API_SCHEMA}` only."),
    "profile.absent": ("warning", "No UsdProfiles claims on the default prim, in a stage written "
                       f"to {CLAIMS_SINCE[0]}.{CLAIMS_SINCE[1]} or later."),
    "profile.unknown": ("warning", "A claim names a `biousd.*` capability or profile this "
                        "version does not declare."),
    "profile.usage": ("warning", "A capability usage is not `hard`, `soft` or `enhancement`."),
    "profile.inconsistent": ("warning", "The claims disagree with the stage's content "
                             "(cells or ontology terms present but not claimed, `biousd.cells` "
                             "not `hard`, or the profile not declared)."),
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
        "through the Cell Ontology term each cell may carry. Since 0.2.0 an "
        f"optional codeless API schema, `{API_SCHEMA}`, declares those primvars "
        "to tools that load its plugin; a file is conforming with or without it.",
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
        f"that authors `bio:cell:id` or applies `{API_SCHEMA}` is a cell population.",
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
        "## API schema (optional)",
        "",
        f"`{API_SCHEMA}` is a codeless applied API schema, shipped as the plugin "
        f"`{LIBRARY_NAME}` (`plugin/plugInfo.json` + `plugin/generatedSchema.usda`, "
        "generated from this vocabulary by `python -m biousd.build_plugin`). It "
        "declares every primvar above on the carrier with its type, allowed "
        "tokens, documentation and `varying` interpolation, so a tool that "
        "loads the plugin can show what a cell population is without reading "
        "this document.",
        "",
        f"- Identifier `{API_SCHEMA}`; single-apply; declared for "
        f"`{CARRIER_PRIM_TYPE}` only. USD treats `apiSchemaCanOnlyApplyTo` as "
        "advisory (`CanApplyAPI` says no, `ApplyAPI` applies anyway); this "
        "validator enforces it (`schema.misapplied`).",
        "- **Optional.** A stage without `apiSchemas` is still conforming "
        "(`schema.notApplied` is a warning). A stage with it opens in any "
        "OpenUSD tool whether or not the plugin is installed, because a writer "
        "still authors every primvar with its type and interpolation "
        "explicitly; the schema adds documentation and fallbacks, never data.",
        f"- A writer applies it with `prim.AddAppliedSchema(\"{API_SCHEMA}\")`, "
        "which needs no plugin.",
        f"- Versioning: an incompatible change ships as `{API_SCHEMA}_1` (USD "
        "schema-family versioning), independent of the vocabulary version "
        f"above. This version knows `{API_SCHEMA}` only and warns on any other "
        "(`schema.version`).",
        "- Registration: `import biousd` registers the plugin, or set "
        "`PXR_PLUGINPATH_NAME` to `biousd.plugin_path()` (the schema and the "
        "profiles; the framework validators below are registered separately). "
        "Either must happen before the process first builds its schema "
        "registry, which the first stage opened does; a plugin registered "
        "later is silently unknown.",
        "",
        "## Profiles (OpenUSD 26.08 and later)",
        "",
        "The same `plugInfo.json` declares UsdProfiles capabilities, with "
        f"`{CAPABILITY_ROOT}` as the root:",
        "",
        "| capability | meaning |",
        "|---|---|",
        f"| `{CAPABILITY_BIOUSD}` | A Bio-USD stage. |",
        f"| `{CAPABILITY_CELLS}` | A cell population: the carrier and the required "
        f"primvars. Implied by `{API_SCHEMA}` (`impliesCapabilities`). |",
        f"| `{CAPABILITY_ONTOLOGY}` | `bio:cell:ontologyTerm` resolved against the "
        "Cell Ontology / NCI Thesaurus. |",
        f"| `{PROFILE_V1}` | The profile: `{CAPABILITY_CELLS}` and "
        f"`{CAPABILITY_ONTOLOGY}`. |",
        "",
        "A writer records claims on the default prim through "
        "`UsdProfilesClaimsAPI`: `apiSchemas = [\"ClaimsAPI\"]` and a "
        "`profilesInfo` dictionary in the prim's `customData` with "
        f"`capabilityUsages` (`{CAPABILITY_CELLS}` = `hard`, "
        f"`{CAPABILITY_ONTOLOGY}` = `enhancement` when terms are asserted) and "
        f"`profileCompatibility` (`{PROFILE_V1}` = `[]`, an empty exception "
        "list). A writer on any OpenUSD version authors that by hand "
        "(`biousd.profiles.write_claims`); a reader on 26.08 or later asks "
        f"`ClaimsAPI.IsCompatibleWith(\"{PROFILE_V1}\")`. Claims are optional; "
        "the validator checks them for consistency with the stage (`profile.*`, "
        "warnings only).",
        "",
        "## Conformance",
        "",
        "A stage conforms when it opens, declares the stage metadata above, "
        "holds at least one carrier with the required primvars, every present "
        "primvar has the stated type and `varying` interpolation and as many "
        "values as positions, every token is in its allowed set, every "
        "ontology term is a well-formed CURIE or empty, `bio:cell:id` is "
        f"unique within the carrier, and `{API_SCHEMA}`, where applied, is "
        f"applied to a `{CARRIER_PRIM_TYPE}`. A carrier is a cell population when "
        f"it authors `bio:cell:id` or applies `{API_SCHEMA}`. "
        "`python -m biousd.validate <file>` checks exactly this and nothing "
        "else; exit status 0 conforms, 1 does not, 2 could not be read.",
        "",
        "### Validator findings",
        "",
        "| code | level | meaning |",
        "|---|---|---|",
    ]
    for code, (level, meaning) in VALIDATOR_CODES.items():
        lines.append(f"| `{code}` | {level} | {meaning} |")
    lines += [
        "",
        "`biousd-check` runs the same checks after OpenUSD's own validation "
        "framework, in one report with one exit status. On usd-core 26.05 or "
        "later the checks also register with that framework as "
        "`biousdValidators:StageMetadata` and `biousdValidators:CellPopulation` "
        "(suite `biousdValidators:BioUSDSuite`) -- `import biousd` or "
        "`biousd.register_validators()` does it -- where a framework client "
        "(measured as a `ValidationContext` over every registered validator, "
        "the selection `usdchecker --useNewValidationFramework` makes) runs "
        "them on a Bio-USD stage and gets no finding from them on any other "
        "stage; earlier wheels have no Python registration API, so there the "
        "checks run standalone only and nothing is advertised.",
        "",
        "## What this is not",
        "",
        "Not a schema of prim types: the API schema is applied to an existing "
        "`PointInstancer`, is optional, and nothing here needs it. Not a "
        "simulation format: it says what cells are and where, not how they got "
        "there. Not a rival to MultiCellDS, the community standard for "
        "multicellular data: a MultiCellDS digital snapshot's cell positions, "
        "cycle phase, volume and cell type map onto the primvars above, and the "
        "ontology term is the join between the two.",
        "",
    ]
    return "\n".join(lines)
