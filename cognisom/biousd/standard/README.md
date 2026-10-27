# Bio-USD

**Biological cells on OpenUSD, as a convention anyone can write to.**
Draft v0.2.0 · Apache 2.0 · [SPEC.md](SPEC.md) · DOI [10.5281/zenodo.23603569](https://doi.org/10.5281/zenodo.23603569)

A Bio-USD file is an ordinary OpenUSD stage. Cells are instances of a
`PointInstancer`; what each cell *is* rides along as per-instance primvars.
Because it adds no prim types, every OpenUSD tool already reads it —
usdview, Omniverse, Houdini, Blender, Mol* via a loader — and because each
cell may carry a **Cell Ontology** term, an ontology-aware tool can join a
scene to CZ CELLxGENE, the Human Cell Atlas and Tabula Sapiens, which
annotate their 150 million cells with the same identifiers.

```
#usda 1.0
(
    customLayerData = { string "bio:version" = "0.2.0" }
    defaultPrim = "Cells"
    metersPerUnit = 1e-06
    upAxis = "Y"
)
def PointInstancer "Cells" (
    prepend apiSchemas = ["BioCellAPI"]            # optional, new in 0.2.0
)
{
    def Sphere "Prototype"                         # unit diameter: the instance scale is the cell's
    {
        double radius = 0.5
    }
    rel prototypes = </Cells/Prototype>
    point3f[] positions = [(12.4, 8.1, 50.0), (31.0, 9.7, 50.0)]
    int[] protoIndices = [0, 0]
    float3[] scales = [(12, 12, 12), (8, 8, 8)]
    int[] primvars:bio:cell:id = [0, 1] (interpolation = "varying")
    token[] primvars:bio:cell:type = ["cancer", "immune"] (interpolation = "varying")
    token[] primvars:bio:cell:ontologyTerm = ["NCIT:C2919", "CL:0000625"] (interpolation = "varying")
}
```

That is the whole idea, and that block is a conforming file (a test saves
it and runs the validator on it). Two primvars are required — an id and a
coarse type — and five more are typed when present: ontology term, cycle
phase, age, viability, relative volume. Every primvar authors its
`varying` interpolation: a reader without the schema plugin sees USD's
default, `constant`, otherwise. Positions and diameters are in
micrometres. [SPEC.md](SPEC.md) is the normative list and is generated from
[`vocabulary.py`](vocabulary.py), so the code and the prose cannot drift.

## Which term do I use?

[`terms.py`](terms.py) is a starter map from 43 cell identities to Cell
Ontology / NCI Thesaurus terms — every one resolved by exact label match,
never fuzzy search, with the pitfalls that forced that rule written down.
`curie_for("Macrophage M1")` is `CL:0000863`; `curie_for("Macrophage M2")`
is `CL:0000890`; an identity it does not know returns `None`, not a guess.

## Write one, check one

```python
import biousd                     # or: from cognisom.biousd import standard as biousd

biousd.write_cells("scene.usdc", [
    {"position": (12.4, 8.1, 50.0), "cell_type": "cancer",
     "ontology_term": "NCIT:C2919", "scale_um": 12.0, "phase": "S"},
    {"position": (31.0, 9.7, 50.0), "cell_type": "immune",
     "ontology_term": "CL:0000625", "scale_um": 8.0},
])
table = biousd.read_cells("scene.usdc")
print(table.n, table["bio:cell:ontologyTerm"])
```

```
$ biousd-validate scene.usdc
scene.usdc: CONFORMS to Bio-USD 0.2.0 -- 1 population(s), 2 cells, 0 error(s), 0 warning(s)

$ biousd-check scene.usdc
scene.usdc
  OpenUSD validation framework: 29 validator(s), 0 finding(s), biousdValidators among them
  Bio-USD 0.2.0: 1 population(s), 2 cells, 0 error(s), 0 warning(s) (the same checks as the framework's biousdValidators above)
  => CONFORMS
```

`biousd-validate` checks exactly what the spec says and nothing else;
`biousd-check` runs OpenUSD's own validation framework first and gives one
report and one exit status. Both exit 0, 1 or 2, so they drop into CI.
Every finding has a code; SPEC.md lists them all. On a wheel that cannot
run everything the report says so in two more lines: usd-core 25.11 ships
no shader definitions, so the `usdShadeValidators:*` validators are left
out (they raised on every stage with a material, a traceback instead of an
exit status), and Bio-USD's own validators are reported as not in the
framework, with the reason (no Python registration before OpenUSD 26.05).

## The schema (new in 0.2.0)

`BioCellAPI` is a **codeless applied API schema** — a `plugInfo.json` and a
`generatedSchema.usda` under [`plugin/`](plugin/), nothing to compile —
that declares the seven primvars on a `PointInstancer`: their types,
allowed tokens, documentation and `varying` interpolation. It is generated
from `vocabulary.py` by `python -m biousd.build_plugin` (usdGenSchema in
codeless mode), and a test holds the checked-in copy byte-identical to a
fresh build.

What it gives a tool that loads it: with the schema applied,
`primvars:bio:cell:phase` on the instancer has `allowedTokens`, a doc
string, and `varying` as its fallback interpolation even if a file forgot
to author it; `CanApplyAPI` refuses an `Xform`; UsdProfiles (26.08+) learns
that the prim uses `biousd.cells`. What it does **not** do: change the
file, or the verdict. The writer still authors every primvar with its type
and interpolation explicitly, so a reader without the plugin sees exactly
the 0.1 file plus two metadata entries it ignores, and a file without
`apiSchemas` is still conforming (the validator says so with a warning,
`schema.notApplied`). A file that leans on the fallback — no authored
interpolation, or a value type the schema would correct — does not
conform, with or without the plugin: conformance is a property of the
file, and the validator reads what the file authors, not what the
registry resolves.

**Registering it.** USD builds its schema registry once per process, the
first time anything needs a prim definition — opening a stage does — and a
plugin registered after that is never consulted (measured: `RegisterPlugins`
returns the plugin, the registry still does not know the schema). So:

```python
import biousd            # registers the plugin; do this before opening any stage
biousd.schema_is_known() # True in a process that imported it in time
```

or, for a tool you do not control (usdview, usdcat, a DCC):

```
export PXR_PLUGINPATH_NAME=$(python -c 'import biousd; print(biousd.plugin_path())')
```

That registers the schema and the profiles, and nothing that could fail to
load: the validation-framework validators live in a separate plugin (see
below). The import itself touches only `Plug` and does not build the
schema registry, so a plugin of your own registered after `import biousd`
is still known. A late import logs one warning, when `schema_is_known()`
is first asked, and everything else still works. `BIOUSD_NO_PLUGIN=1`
skips the registration.

**What the plugin cannot do, measured:**

- Apple's bundled USD tools (`/usr/bin/usdchecker`, `usdcat`, `usdtree`,
  version 0.25.2) load no external plugin at all, however
  `PXR_PLUGINPATH_NAME` is set. They remain a "the file still opens
  cleanly" check.
- USD treats `apiSchemaCanOnlyApplyTo` as advisory: `ApplyAPI` succeeds on
  an `Xform`. The validator enforces it (`schema.misapplied`, an error).
- No builtin validator checks `allowedTokens` or a misapplied schema (on
  26.11, with the plugin loaded, a `bio:cell:type` of `"alien"` on an
  applied instancer produces no framework finding; only a wrong value type
  does). That is what `biousd-validate` is for.

**Versioning.** An incompatible change to the schema ships as
`BioCellAPI_1`, USD's schema-family versioning, independent of the 0.x
vocabulary version; this version knows `BioCellAPI` only and warns on any
other (`schema.version`).

## Profiles (OpenUSD 26.08 and later)

The same `plugInfo.json` declares UsdProfiles capabilities — `biousd`,
`biousd.cells`, `biousd.ontology` — and the profile `biousd.profile_v1`.
The writer records claims on the default prim (`ClaimsAPI` with
`profilesInfo` in its `customData`: `biousd.cells` is `hard`,
`biousd.ontology` an `enhancement`, compatible with `biousd.profile_v1`),
which is plain metadata, so a 25.x writer authors it by hand
(`biousd.profiles.write_claims`) and a 26.11 reader with the plugin answers
`ClaimsAPI.IsCompatibleWith("biousd.profile_v1") -> ValidPath` (measured;
without the plugin it answers `NoPath`, honestly). Upstream ships four
capabilities and no profile, and nothing upstream checks a claim; this is a
declaration for consumers that will, and `biousd-validate` checks the
claims against the stage (`profile.*`, warnings only).

## The validation framework

On usd-core **26.05 or later**, `import biousd` hands a second plugin,
`plugin/biousdValidators/`, to Plug, and `biousd.register_validators()`
(which `biousd-check` calls) makes sure `biousdValidators:StageMetadata`,
`biousdValidators:CellPopulation` and the suite `biousdValidators:BioUSDSuite`
are registered with `pxr.UsdValidation` — through the plugin when the
validation registry is built after the import, explicitly otherwise. A
framework client then runs them beside the builtin validators: measured as
a `ValidationContext` over every registered validator, the selection
`usdchecker --useNewValidationFramework` makes (`usdchecker` itself has not
been run with them; no wheel here ships the binary, and Apple's loads no
plugin). They are the same functions as the standalone validator, so the
two cannot disagree; a code such as `primvar.token` appears there as
`biousdValidators:CellPopulation.PrimvarToken`. They leave any stage that
is not Bio-USD alone — no `bio:version`, no cell population, no
`BioCellAPI` — so registering them fails no ordinary scene (it did: three
errors on a plain `Xform` + `Sphere` + `PointInstancer`).

On usd-core **25.x** the framework has no Python registration API (it
arrived in OpenUSD 26.05, PR #4015): nothing is advertised, nothing can
fail to load, and `biousd-check` reports the same findings from the
standalone pass and says why. To load the validators from the environment
instead, name both directories — `PXR_PLUGINPATH_NAME=<plugin>:<plugin>/biousdValidators`
— and put `<plugin>` on `PYTHONPATH` as well, because Plug loads them with
`import biousdValidators`; on 25.x that route advertises names the wheel
cannot register, and the registry skips them silently.

## Status, honestly

- **Draft.** One implementation writes it (Cognisom, a prostate-cancer
  digital twin); nobody else does yet. It becomes a standard when someone
  else writes to it, not before.
- **The ontology join is the point.** `bio:cell:type` exists for renderers
  that only have a token; `bio:cell:ontologyTerm` is what makes a file
  mean something outside the tool that wrote it. Terms are resolved by
  exact label match only — a fuzzy search returned four wrong tissues
  with full confidence.
- **No prim types.** The schema is applied to an existing `PointInstancer`
  and is optional; nothing here needs it, and that is deliberate: it is
  why the files open everywhere.
- **Not a simulation format.** It says what cells are and where, not how
  they got there. It is meant to sit beside MultiCellDS, the community
  standard for multicellular data, not replace it; the ontology term is
  the join between the two.
- **Prototypes are free-form.** The minimal writer draws spheres; a tool
  with real morphology authors its own prototypes at unit diameter and
  sets `protoIndices`. The instance scale is then the cell's diameter.

## From PhysiCell

A PhysiCell / MultiCellDS snapshot converts directly; the label table in
the XML says which matrix row is which, and the cycle, phase and death
codes come from PhysiCell's own constants source. Cell-type names are
not in the output, so you supply them:

```
$ pip install biousd[physicell]
$ biousd-from-physicell output00000001.xml output00000001.usdc --type 0=normal --type 1=cancer
125 cells (118 alive) at t = 60.0 min -> output00000001.usdc; 94 phases mapped to their nearest token, 0 unknown
```

Verified on a PhysiCell 1.14.1 output. `bio:cell:age` is not asserted
(PhysiCell records time in phase, not since division), `volume` is
relative to the median living volume of the type, and composite phases
(G0G1, G2M) and the phase-less `live` model map to their nearest token
and are counted in the report. The microenvironment matrices are not
read.

## Examples

`examples/` holds three files written by the reference writer, with the
script that made them: a 200-cell scene as readable `.usda`, a 10,000-cell
tissue slab, and a 2,000-cell scene coloured by a per-cell field. All three
apply the schema and carry claims.

## Feedback

Open an issue, or write a file and run the validator on it. The questions
worth answering first: is the required set right, is the coarse type list
useful or a nuisance, and what is missing for your tool to read it.

## Citation

See [CITATION.cff](CITATION.cff). Please cite the version you wrote to.

## Changes

**0.2.0** (2026-10-27)

- The codeless `BioCellAPI` applied schema and its plugin, generated from
  the vocabulary (`python -m biousd.build_plugin`); the writer applies it;
  a stage without it is still conforming.
- UsdProfiles capabilities and the `biousd.profile_v1` profile in the same
  plugin; the writer records claims; the validator checks them.
- `biousd-check`: OpenUSD's validation framework plus Bio-USD in one
  report, and framework validators of the same checks on usd-core 26.05+,
  which run only on a Bio-USD stage. On a wheel without shader
  definitions (25.11) the report names the validators it had to skip.
- Reader: primvars are gated on authored data, and the value type is the
  authored one. The 0.1 reader, with the plugin loaded, reported a
  conforming file that omitted an optional primvar as non-conforming
  (`primvar.length`), and would have passed a wrong value type the schema
  resolved away; it does neither, and tests run each case with and
  without the plugin.
- New validator findings: `schema.misapplied` (error), `schema.notApplied`,
  `schema.version`, `profile.*` (warnings). Every existing code is kept;
  `primvar.interpolation` now also names a primvar that authors no
  interpolation, since without the plugin it is `constant`. SPEC.md lists
  them.
- `vendor/openusd/`: two OpenUSD v25.11 files usdGenSchema needs on the
  25.11 wheel (see NOTICE).

**0.1.2** (2026-10-23) — first archived release: the vocabulary, writer,
reader, validator, starter term map, PhysiCell converter and examples.
