# Bio-USD 0.2.0 — specification

Bio-USD is a convention for carrying a population of biological cells in an OpenUSD stage. It adds no prim types: cells are instances of a `UsdGeom.PointInstancer`, and their biology is a set of `varying` primvars on that instancer, one value per instance. Any OpenUSD tool draws it unmodified; an ontology-aware tool can join it to other data through the Cell Ontology term each cell may carry. Since 0.2.0 an optional codeless API schema, `BioCellAPI`, declares those primvars to tools that load its plugin; a file is conforming with or without it.

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

The carrier is a `UsdGeom.PointInstancer`. A stage may hold several; each that authors `bio:cell:id` or applies `BioCellAPI` is a cell population.

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

## API schema (optional)

`BioCellAPI` is a codeless applied API schema, shipped as the plugin `bioUsd` (`plugin/plugInfo.json` + `plugin/generatedSchema.usda`, generated from this vocabulary by `python -m biousd.build_plugin`). It declares every primvar above on the carrier with its type, allowed tokens, documentation and `varying` interpolation, so a tool that loads the plugin can show what a cell population is without reading this document.

- Identifier `BioCellAPI`; single-apply; declared for `PointInstancer` only. USD treats `apiSchemaCanOnlyApplyTo` as advisory (`CanApplyAPI` says no, `ApplyAPI` applies anyway); this validator enforces it (`schema.misapplied`).
- **Optional.** A stage without `apiSchemas` is still conforming (`schema.notApplied` is a warning). A stage with it opens in any OpenUSD tool whether or not the plugin is installed, because a writer still authors every primvar with its type and interpolation explicitly; the schema adds documentation and fallbacks, never data.
- A writer applies it with `prim.AddAppliedSchema("BioCellAPI")`, which needs no plugin.
- Versioning: an incompatible change ships as `BioCellAPI_1` (USD schema-family versioning), independent of the vocabulary version above. This version knows `BioCellAPI` only and warns on any other (`schema.version`).
- Registration: `import biousd` registers the plugin, or set `PXR_PLUGINPATH_NAME` to `biousd.plugin_path()` (the schema and the profiles; the framework validators below are registered separately). Either must happen before the process first builds its schema registry, which the first stage opened does; a plugin registered later is silently unknown.

## Profiles (OpenUSD 26.08 and later)

The same `plugInfo.json` declares UsdProfiles capabilities, with `usd` as the root:

| capability | meaning |
|---|---|
| `biousd` | A Bio-USD stage. |
| `biousd.cells` | A cell population: the carrier and the required primvars. Implied by `BioCellAPI` (`impliesCapabilities`). |
| `biousd.ontology` | `bio:cell:ontologyTerm` resolved against the Cell Ontology / NCI Thesaurus. |
| `biousd.profile_v1` | The profile: `biousd.cells` and `biousd.ontology`. |

A writer records claims on the default prim through `UsdProfilesClaimsAPI`: `apiSchemas = ["ClaimsAPI"]` and a `profilesInfo` dictionary in the prim's `customData` with `capabilityUsages` (`biousd.cells` = `hard`, `biousd.ontology` = `enhancement` when terms are asserted) and `profileCompatibility` (`biousd.profile_v1` = `[]`, an empty exception list). A writer on any OpenUSD version authors that by hand (`biousd.profiles.write_claims`); a reader on 26.08 or later asks `ClaimsAPI.IsCompatibleWith("biousd.profile_v1")`. Claims are optional; the validator checks them for consistency with the stage (`profile.*`, warnings only).

## Conformance

A stage conforms when it opens, declares the stage metadata above, holds at least one carrier with the required primvars, every present primvar has the stated type and `varying` interpolation and as many values as positions, every token is in its allowed set, every ontology term is a well-formed CURIE or empty, `bio:cell:id` is unique within the carrier, and `BioCellAPI`, where applied, is applied to a `PointInstancer`. A carrier is a cell population when it authors `bio:cell:id` or applies `BioCellAPI`. `python -m biousd.validate <file>` checks exactly this and nothing else; exit status 0 conforms, 1 does not, 2 could not be read.

### Validator findings

| code | level | meaning |
|---|---|---|
| `stage.open` | error | The file could not be opened as a USD stage (exit status 2). |
| `stage.metersPerUnit` | error | `metersPerUnit` is not `1e-06`. |
| `stage.upAxis` | error | `upAxis` is not `Y` or `Z`. |
| `layer.version` | error / warning | Error: `customLayerData` has no `bio:version`. Warning: it names a different major version than this validator. |
| `layer.colourRange` | warning | `bio:colourBy` is set without `bio:colourMin`/`Max`. |
| `carrier.none` | error | No PointInstancer is a cell population. |
| `carrier.protoIndices` | error | `protoIndices` has a different length than `positions`. |
| `carrier.prototypes` | error | A `protoIndex` is outside the prototypes relationship. |
| `carrier.scales` | error | A cell has a non-positive diameter. |
| `carrier.invisibleIds` | error | An `invisibleIds` entry is not an instance. |
| `primvar.missing` | error | A required primvar is not authored. |
| `primvar.type` | error | A primvar has a value type other than the one stated. |
| `primvar.interpolation` | error | A primvar does not author `varying` interpolation (unauthored, it is `constant` to a reader without the plugin; the schema fallback is not the file). |
| `primvar.length` | error | A primvar does not have one value per instance. |
| `primvar.token` | error | A token value is outside its allowed set. |
| `primvar.range` | error | A numeric value is below its minimum. |
| `primvar.curie` | error / warning | Error: an ontology term is not a `CL:`/`NCIT:` CURIE. Warning: no cell asserts a term. |
| `primvar.id` | error | `bio:cell:id` is not unique within the carrier. |
| `primvar.optional` | warning | An optional primvar is not authored. |
| `schema.misapplied` | error | `BioCellAPI` is applied to a prim that is not a `PointInstancer`. |
| `schema.notApplied` | warning | A cell population does not apply `BioCellAPI` (a 0.1 file; still conforming). |
| `schema.version` | warning | A versioned `BioCellAPI_<n>` is applied; this validator knows `BioCellAPI` only. |
| `profile.absent` | warning | No UsdProfiles claims on the default prim, in a stage written to 0.2 or later. |
| `profile.unknown` | warning | A claim names a `biousd.*` capability or profile this version does not declare. |
| `profile.usage` | warning | A capability usage is not `hard`, `soft` or `enhancement`. |
| `profile.inconsistent` | warning | The claims disagree with the stage's content (cells or ontology terms present but not claimed, `biousd.cells` not `hard`, or the profile not declared). |

`biousd-check` runs the same checks after OpenUSD's own validation framework, in one report with one exit status. On usd-core 26.05 or later the checks also register with that framework as `biousdValidators:StageMetadata` and `biousdValidators:CellPopulation` (suite `biousdValidators:BioUSDSuite`) -- `import biousd` or `biousd.register_validators()` does it -- where a framework client (measured as a `ValidationContext` over every registered validator, the selection `usdchecker --useNewValidationFramework` makes) runs them on a Bio-USD stage and gets no finding from them on any other stage; earlier wheels have no Python registration API, so there the checks run standalone only and nothing is advertised.

## What this is not

Not a schema of prim types: the API schema is applied to an existing `PointInstancer`, is optional, and nothing here needs it. Not a simulation format: it says what cells are and where, not how they got there. Not a rival to MultiCellDS, the community standard for multicellular data: a MultiCellDS digital snapshot's cell positions, cycle phase, volume and cell type map onto the primvars above, and the ontology term is the join between the two.
