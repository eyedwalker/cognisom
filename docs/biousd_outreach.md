# Bio-USD — the traction test

Three parties, one question each, one readable file. The test is not "do
they like it"; it is "would your tool read this?" One yes means the
absence of a life-sciences group at AOUSD is the opening. Silence from all
three is the warning, and it costs a week instead of a standards process.

What to attach, every time: the repository link, and
`examples/minimal_200_cells.usda` — it is 23 KB, opens in a text editor,
and shows the whole convention in the first thirty lines.

Repository: https://github.com/eyedwalker/biousd (private until you flip
it; flip it before sending — a private link is a dead link to them).

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

**The ask:** would a PhysiCell `.mat`/MultiCellDS snapshot → Bio-USD
exporter be worth a pull request from us, and is the per-cell vocabulary
(id, type, ontology term, phase, age, alive, volume) the right first set?

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
> meant to be the join between the two. The question we would value your
> answer to: if we wrote a PhysiCell → Bio-USD exporter as a PR, would it
> be welcome, and is this the right first vocabulary? A no is as useful as
> a yes.
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
| NVIDIA / AOUSD | AOUSD forum post (+ developer forum) | | | |

**Decision rule:** one "yes, we'd read it" = the opening; proceed to a
second implementation with them and propose the AOUSD group. Three
silences after two weeks = the warning; keep Bio-USD as Cognisom's
export format, keep the ontology terms, stop investing in the standards
track.

## Before sending

1. Flip the repository to public.
2. Add the Zenodo DOI to CITATION.cff (connect the repo on zenodo.org,
   then create the v0.1.0 release — the tag already exists).
3. Verify the two unverified addresses.
