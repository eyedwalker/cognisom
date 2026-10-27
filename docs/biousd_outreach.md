# Bio-USD — the traction test

Three parties, one question each, one readable file. The test is not "do
they like it"; it is "would your tool read this?" One yes means the
absence of a life-sciences group at AOUSD is the opening. Silence from all
three is the warning, and it costs a week instead of a standards process.

What to attach, every time: the repository link, and
`examples/minimal_200_cells.usda` — it is 23 KB, opens in a text editor,
and shows the whole convention in the first thirty lines.

Repository: https://github.com/eyedwalker/biousd — public as of 2026-10-23,
Apache 2.0, with the PhysiCell exporter and three example files. Ready-to-
paste copies of the three messages below are kept beside this file as the
artifact page "Bio-USD outreach messages".

---

## 1. PhysiCell / MultiCellDS — Paul Macklin's lab (Indiana University)

**Why them:** the dominant open multicellular simulator and the authors of
MultiCellDS, the community standard Bio-USD is meant to sit beside. If
PhysiCell could export a Bio-USD file, every PhysiCell user gets a
renderer for free and Bio-USD gets its second implementation.

**Channel (verified):** macklinp@iu.edu; the lab's contact page at
MathCancer.org/Contact.php; or a GitHub Discussion on
github.com/MathCancer/PhysiCell. A Discussion is public and lets others
weigh in; email is faster.

**The ask:** the exporter exists (`biousd-from-physicell`, verified on a
PhysiCell 1.14.1 snapshot; the cycle/phase codes are read from
`PhysiCell_constants.cpp`). Would a PR adding it to PhysiCell's tooling
be welcome, and is the per-cell vocabulary (id, type, ontology term,
phase, age, alive, volume) the right first set? One thing only they can
answer: the output carries `cell_type` as an integer and the names live
in the settings file — should the exporter read `PhysiCell_settings.xml`
for them, or should the output carry the names?

> Subject: A USD carrier for MultiCellDS snapshots — would PhysiCell read it?
>
> Dr Macklin — we build a prostate-cancer digital twin and have been
> exporting cell populations to OpenUSD so they render in Omniverse,
> Houdini and the like without a plugin. We wrote the convention down —
> seven per-instance primvars on a PointInstancer, each cell carrying a
> Cell Ontology term — and published it as a draft under Apache 2.0:
> https://github.com/eyedwalker/biousd. The attached 200-cell file is the
> whole idea in thirty lines.
>
> It is deliberately not a rival to MultiCellDS: it says what cells are
> and where, nothing about how they got there, and the ontology term is
> meant to be the join between the two. We have written the converter —
> `biousd-from-physicell output00000001.xml out.usdc --type 1=cancer`
> turns a 1.14.1 snapshot into a stage that opens in usdview — and would
> offer it as a PR to PhysiCell's tooling if that is welcome. Two
> questions we would value your answer to: is this the right first
> vocabulary, and should the exporter read cell-type names from
> `PhysiCell_settings.xml` or would you rather the output carried them?
> A no is as useful as a yes.
>
> — David Walker, eyentelligence inc.

---

## 2. Mol* / Mesoscale Explorer — Sehnal (Mol*), Autin & Goodsell (Scripps)

**Why them:** the one group that has solved large-scale cell
visualisation *in a browser* (Mesoscale Explorer, Protein Science 2024,
built on Mol*). If Mol* could load a Bio-USD population, the standard gets
a reader that needs no GPU workstation — the cheapest route to "a
scientist opens a link."

**Channel (partly verified):** Mesoscale Explorer lives at
mesoscope.scripps.edu/explorer and is a Mol* app; the Mol* code is on
GitHub under github.com/molstar — open an issue there titled as a
feature proposal. Direct contact for Ludovic Autin (Scripps, Olson lab)
and David Sehnal: **verify current addresses before sending**; the
search did not confirm them.

**The ask:** would a Mol* loader for a PointInstancer-with-primvars
population be in scope, and what would the file need to carry that it
does not?

> Title: Loading a cell population from OpenUSD (PointInstancer + primvars)
>
> Mesoscale Explorer showed that a browser can hold a cell. We have been
> writing cell populations — up to 500,000 instances with real
> morphology — to OpenUSD as a PointInstancer whose per-instance primvars
> carry each cell's Cell Ontology term, phase, age and so on, and we have
> published the convention as a draft (Apache 2.0):
> https://github.com/eyedwalker/biousd. Attached is a 200-cell file in
> text form.
>
> The question: would a loader for this shape be in scope for Mol*, and
> what is missing from the file for that to be straightforward? We would
> write the loader if the answer is yes.

---

## 3. NVIDIA — Omniverse / OpenUSD developer programme, and AOUSD

**Why them:** USD's largest promoter, with a stated push into biology, and
the body (AOUSD) with no life-sciences group. This is the direct probe of
"opening or warning."

**Channel (general — no biology-specific contact could be verified):**
the NVIDIA OpenUSD Developer Program (developer.nvidia.com/usd), the
Omniverse developer forum (forums.developer.nvidia.com, OpenUSD
category), and a post on forum.aousd.org asking whether a life-sciences
interest group would have any takers. The AOUSD post is the one that
answers the question; post it even if NVIDIA never replies.

**The ask (AOUSD forum):** is anyone else carrying biological cells on
USD, and would an interest group find a second member?

> Title: Biological cells on OpenUSD — is anyone else doing this?
>
> We carry simulated cell populations on USD: a PointInstancer whose
> per-instance primvars hold each cell's identity as a Cell Ontology term,
> plus cycle phase, age, viability and volume. No new prim types, so it
> opens in any USD tool; the ontology term joins it to the single-cell
> atlases. Draft convention, Apache 2.0: https://github.com/eyedwalker/biousd.
>
> There is no life-sciences group here. Before proposing one: is anyone
> else carrying cells, tissues or molecules on USD, and would an interest
> group have a second member? If not, we would rather know.

---

## Tracking

| party | channel | sent | replied | verdict |
|---|---|---|---|---|
| Macklin / PhysiCell | email or GitHub Discussion | | | |
| Mol* / Scripps | GitHub issue (+ email once verified) | | | |
| NVIDIA / AOUSD | AOUSD forum post (+ developer forum) | posted (DW) | 2026-10-27, spiff (Pixar USD lead) | **opening** — three integration suggestions, see below |

### What came back from the forum (2026-10-27)

spiff (Pixar) replied with three ways to make the convention "more robust,
introspectable and discoverable": (1) an API schema plugin applied to the
PointInstancer (and verifiably only to PointInstancers); (2) registering
the validator with USD's Validation framework so usdchecker and usdview
discover it; (3) the Profiles featureset, so scenes using the extension
are identifiable. Each was proven against real USD builds the same day
(scratch spikes, re-run by a critic; nothing shipped yet):

- A codeless `BioCellAPI` works on our usd-core 25.11: generated from
  `vocabulary.py`, two data files, files stay valid without the plugin.
  Two traps: the PointInstancer restriction is advisory (`ApplyAPI`
  applies anyway), and our reader treats a schema-defined-but-unauthored
  optional primvar as present, so with the plugin loaded an omitted
  optional becomes an error. Fix the reader first; ship as 0.2.0 (the
  spec says "adds no prim types").
- Validators can be registered from Python only from OpenUSD 26.05;
  verified end to end on usd-core 26.11. On 25.x the plugin is listed but
  never runs. Apple's bundled macOS `usdchecker` loads no plugins at all.
- Profiles shipped in 26.08 (`UsdProfiles`), a stub upstream (four
  capabilities, no profiles); half a session on top of the schema.
- Our USD-touching tests pass unchanged under usd-core 26.11, so the pin
  bump that enables all three is cheap (wheels need Python ≥ 3.11).

**Shipped as Bio-USD 0.2.0 (2026-10-27, on the `realism` branch; not yet
pushed to the public mirror, so the forum reply cannot link it until it
is).** All three suggestions are code now, not spikes, each verified on
usd-core 25.11 and 26.11 by tests that run under both
(`scripts/usd_matrix.sh`; a nightly CI job installs 26.11 over the
25.11 pin, which itself is unchanged):

- `BioCellAPI`: a codeless applied schema generated from `vocabulary.py`
  (`schema.usda` → `plugin/plugInfo.json` + `generatedSchema.usda`;
  regeneration is byte-identical under either wheel), registered by
  `import biousd` before the first stage is opened, with the late-import
  trap documented and warned once. The reader regression is fixed
  (gates on authored primvars, not defined ones): the spike's file that
  omitted an optional primvar now conforms with one warning with and
  without the plugin. The PointInstancer restriction is enforced by the
  validator (`schema.misapplied`) since USD treats it as advisory.
- Validation framework: `biousdValidators:StageMetadata`,
  `:CellPopulation` and the suite register with `pxr.UsdValidation` on
  26.05 and later, implemented on the same check functions as the
  standalone validator; `biousd-check` chains the framework and Bio-USD
  into one report with one exit status on any wheel, and on 25.11 says
  why the framework path cannot run instead of running it silently.
- Profiles: `biousd`, `biousd.cells`, `biousd.ontology` and the profile
  `biousd.profile_v1` are declared in the one `plugInfo.json`; the writer
  records `ClaimsAPI` claims by hand on any wheel, and a 26.11 reader
  with the plugin answers `IsCompatibleWith("biousd.profile_v1") →
  ValidPath`. The validator checks the claims against the stage
  (`profile.*`, warnings only).
- Cognisom's own exports — the still (`usd_writer.write_cells`) and the
  scrubbable run timeline (`run_usd.write_run`, both populations) — apply
  the schema and write the claims, and the dashboard registers the
  plugin at startup so the schema is known in the process that writes.

Still to send: the reply itself (the draft stands — schema works, reader
bug found and fixed, Python validator registration confirmed on 26.05+,
one README discrepancy, `UsdValidation.TimeRange` vs
`ValidationTimeRange` — plus two questions: is `apiSchemaCanOnlyApplyTo`
meant to be advisory, and is a geometry/PointInstancer capability
planned in the core profile graph), after the 0.2.0 package is pushed so
the links resolve.

**Decision rule:** one "yes, we'd read it" = the opening; proceed to a
second implementation with them and propose the AOUSD group. Three
silences after two weeks = the warning; keep Bio-USD as Cognisom's
export format, keep the ontology terms, stop investing in the standards
track.

## Before sending

1. ~~Flip the repository to public.~~ Done 2026-10-23.
2. ~~Add the Zenodo DOI to CITATION.cff.~~ Done 2026-10-23: concept DOI
   10.5281/zenodo.23603569 (v0.1.2 is 10.5281/zenodo.23603570); the messages cite it.
3. Verify the two unverified addresses.
