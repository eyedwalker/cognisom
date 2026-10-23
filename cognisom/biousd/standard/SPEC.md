# Bio-USD 0.1.2 — specification

Bio-USD is a convention for carrying a population of biological cells in an OpenUSD stage. It adds no prim types: cells are instances of a `UsdGeom.PointInstancer`, and their biology is a set of `varying` primvars on that instancer, one value per instance. Any OpenUSD tool draws it unmodified; an ontology-aware tool can join it to other data through the Cell Ontology term each cell may carry.

This file is generated from `vocabulary.py`. Edit that, not this.

## Stage

- `metersPerUnit` must be `1e-06`: one stage unit is one micrometre.
- `upAxis` may be `Y` or `Z` and must be declared.
- The root layer's `customLayerData` carries:

| key | meaning |
|---|---|
| `bio:version` | The vocabulary version the stage was written to. Required. |
| `bio:colourBy` | Name of the per-cell field driving displayColor, if one is. |
| `bio:colourMin` | Lower end of the colour ramp, in that field's units. |
| `bio:colourMax` | Upper end of the colour ramp, in that field's units. |

## Carrier

The carrier is a `UsdGeom.PointInstancer`. A stage may hold several; each that carries `bio:cell:id` is a cell population.

| attribute | meaning |
|---|---|
| `positions` | cell centres, um |
| `scales` | cell diameter, um, applied to a prototype authored at unit diameter (its body spans 1.0 across) |
| `protoIndices` | which prototype each cell draws; prototypes are free-form geometry and are not part of this vocabulary |
| `invisibleIds` | may be used to hide cells for a cutaway; hidden cells are still cells |

## Per-cell primvars

All are `varying` (one value per instance) and must have exactly as many values as there are positions. **Required** means required for conformance; every other primvar is optional but, when present, must have the type and values given here.

| name | type | required | units | allowed values | meaning |
|---|---|---|---|---|---|
| `bio:cell:id` | `int[]` | **yes** | — | — | Stable identifier, unique within the stage. |
| `bio:cell:type` | `token[]` | **yes** | — | `normal`, `cancer`, `immune`, `stromal`, `endothelial`, `basal`, `luminal`, `neuroendocrine`, `stem`, `fibroblast` | Coarse lineage, for a consumer that has only the token. |
| `bio:cell:ontologyTerm` | `token[]` | no | — | CURIE `CL:…` or `NCIT:…`, or empty | Cell Ontology (CL) or NCI Thesaurus (NCIT) CURIE naming the cell's identity; empty when not asserted. |
| `bio:cell:phase` | `token[]` | no | — | `G0`, `G1`, `S`, `G2`, `M` | Cell-cycle phase. |
| `bio:cell:age` | `float[]` | no | h | ≥ 0 | Time since the cell's last division. |
| `bio:cell:alive` | `bool[]` | no | — | — | Viability. A dead cell may be kept in the stage for an animation; a consumer that counts cells should check it. |
| `bio:cell:volume` | `float[]` | no | — | ≥ 0 | Volume relative to the type's resting volume (1.0 = resting). The instance scale carries the absolute diameter in um. |

## Ontology terms

`bio:cell:ontologyTerm` is a CURIE from one of these prefixes, resolved by exact label match against the ontology and never by fuzzy search (a fuzzy match returns a wrong tissue with the same confidence as a right one). An empty value means the identity is not asserted.

| prefix | resolves at |
|---|---|
| `CL` | `http://purl.obolibrary.org/obo/CL_` |
| `NCIT` | `http://purl.obolibrary.org/obo/NCIT_` |

## Conformance

A stage conforms when it opens, declares the stage metadata above, holds at least one carrier with the required primvars, every present primvar has the stated type and `varying` interpolation and as many values as positions, every token is in its allowed set, every ontology term is a well-formed CURIE or empty, and `bio:cell:id` is unique within the carrier. `python -m biousd.validate <file>` checks exactly this and nothing else.

## What this is not

Not a schema of prim types (a codeless schema may follow; nothing here needs one). Not a simulation format: it says what cells are and where, not how they got there. Not a rival to MultiCellDS, the community standard for multicellular data: a MultiCellDS digital snapshot's cell positions, cycle phase, volume and cell type map onto the primvars above, and the ontology term is the join between the two.
