#!/usr/bin/env python3
"""GenScript x NVIDIA Life Sciences PitchFest deck for Cognisom.

Ten slides of HTML printed to PDF with headless Chrome at 16:9, sized for
the form's 30 MB cap. Machinery adapted from the Eyentelligence NVIDIA
Inception deck builder; the content is entirely Cognisom's.

ACCURACY IS THE POINT OF THIS FILE. The judging panel is GenScript,
NVIDIA and Eli Lilly, so every claim here is one that survived an audit
of the repo's import graph rather than a reading of its documentation --
the project's standing defect pattern is a capability fully implemented,
documented and tested that nothing ever calls, which surfaces as an
honest null rather than an error. Rules applied, and worth re-reading
before editing any slide:

  * The TCGA result is a SURVIVING NEGATIVE PREDICTION, never a
    validation or a confirmation. Slide 6 says so in its own words, and
    carries the reported null beside it.
  * No patent claim appears anywhere. Cognisom's provisional is not
    being pursued. (The Cogs deck's "patent pending" is true of Cogs and
    must never be copied here.)
  * The multi-agent tumour board is DETERMINISTIC by its own docstring
    -- "no new ML models, fully deterministic", no LLM call anywhere in
    cognisom/mad/. It is never described as agentic AI; its value is
    auditability.
  * The Omniverse handoff is a FILE, not a live coupling:
    cognisom/omniverse has zero engine callers. The engine writes USD
    through cognisom/visualization and Isaac Sim opens it.
  * cognisom/agents (data sync, knowledge graph, literature monitor,
    simulation validation) has ZERO call sites and does not appear.
  * r = 0.9998 is a round trip and appears nowhere.
  * ESR1/ERBB2 do not survive somatic filtering on HCC1395; only ARID1A
    and NOTCH1 do.
  * The market sized is BIOSIMULATION, the market we sell into -- not
    the oncology therapeutics market, which is the customers'.

Every FILL(...) is a fact only the founder can supply. It renders as a
magenta chip and MUST be replaced before the deck is submitted.

Usage: python3 pitch/build_pitchfest_deck.py [--pdf]
"""
import html
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
OUT_HTML = HERE / "PITCHFEST_DECK.html"
OUT_PDF = HERE / "PITCHFEST_DECK.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def FILL(text):
    return f'<span class="fill">FILL: {html.escape(text)}</span>'


def esc(s):
    return html.escape(s, quote=False)


def li(items):
    return "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


def card(title, body, accent=None):
    bar = f' style="border-top-color:{accent}"' if accent else ""
    return f'<div class="card"{bar}><h3>{title}</h3>{body}</div>'


def headshot():
    """The avatar, as a photo if one has been dropped in.

    Put a photo at pitch/img/headshot.jpg (or .jpeg/.png) and rebuild --
    no code change needed. Until then this renders the grey placeholder
    circle, which is visibly unfinished on purpose: a deck that goes out
    with a grey disc where the founder should be is worse than one that
    makes the gap obvious while drafting.
    """
    for ext in ("jpg", "jpeg", "png"):
        f = HERE / "img" / ("headshot." + ext)
        if f.exists():
            return ('<div class="avatar"><img src="img/%s" alt="David Walker"></div>'
                    % f.name)
    return '<div class="avatar">headshot</div>'


S = []


def slide(section, headline, body, footer=None, cls=""):
    n = len(S) + 1
    foot = f'<div class="foot">{footer}</div>' if footer else ""
    return (f'<section class="slide {cls}"><div class="eyebrow">'
            f'<span>{esc(section)}</span>'
            f'<span>Cognisom · Eyentelligence, Inc. · {n:02d}</span></div>'
            f'<h2>{headline}</h2><div class="body">{body}</div>{foot}</section>')


# 1. Cover
S.append('''<section class="slide cover">
<div class="eyebrow"><span>GenScript &times; NVIDIA Life Sciences PitchFest · January 2027</span><span>Non-confidential</span></div>
<div class="cover-grid">
 <div>
  <div class="brand">Cognisom <span>by Eyentelligence, Inc.</span></div>
  <h1>A patient's own sequencing drives a <em>mechanistic simulation</em> of their tumour and immune system, predicting which of their mutations their own immune system can see.</h1>
  <p class="lede">Variants in, mechanism out: somatic calling on NVIDIA Parabricks, HLA typing, neural peptide&ndash;MHC binding, AlphaFold2-Multimer structure through NVIDIA BioNeMo, then a ten-module agent-based tumour&ndash;immune simulation that renders in NVIDIA Omniverse. Every parameter is traceable to a published measurement or flagged as an estimate.</p>
  <div class="chips">
   <span class="chip">cognisom.com</span>
   <span class="chip">Eyentelligence, Inc. &middot; Delaware C-corporation, incorporated June 2026 &middot; Flower Mound, TX</span>
   <span class="chip">David Walker &middot; david@eyentelligence.com</span>
   <span class="chip">Harvard Innovation Labs venture</span>
   <span class="chip">NVIDIA Inception member</span>
   <span class="chip">Research use only &mdash; not a clinical decision tool</span>
  </div>
 </div>
 <div class="side">
  <div class="card" style="border-top-color:#76b900"><h3>What we are asking for<span>Two things, and the second matters as much</span></h3>
  <p><b>1. Experimental evidence.</b> Our predictions are wet-lab-testable and untested; GenScript credits would synthesise the peptides we predict and measure whether they bind.</p>
  <p style="margin-top:6pt"><b>2. Research-level scientific expertise.</b> Built AI-first by one founder, the engineering is ahead of the domain review. We are actively seeking immunology and tumour-biology experts to validate and prioritise direction.</p></div>
  <div class="card" style="border-top-color:#2e86ab"><h3>Where NVIDIA already sits<span>Five technologies, in the pipeline today</span></h3>
  <p>BioNeMo models &middot; Parabricks &middot; Omniverse &middot; Data Center GPUs (L40S) &middot; NVIDIA Warp. Plus Bio-USD, our draft open OpenUSD schema for biological data, published under Apache 2.0 with a citable DOI.</p></div>
 </div>
</div>
</section>''')

# 2. Problem and customer
S.append(slide("2 · Problem and target customer",
  "Choosing an immunotherapy candidate means predicting whether a patient's immune system will engage their tumour. Today's tools each answer half of that, so a failure teaches you nothing.",
  f'''<div class="cols">
{card("The gap, concretely", li([
  "<b>Binding predictors</b> return a ranked peptide list with no mechanism. Nothing tells you whether a predicted epitope matters in a tumour that is hypoxic, immune-excluded, or already selecting for escape.",
  "<b>Mechanistic systems-biology models</b> carry that context but are population-average, so they cannot be driven by one patient's variants and HLA type.",
  "Neither answers <i>why</i> a candidate should work in a specific tumour. So a negative result is uninformative and iteration is slow and expensive."]))}
{card("Who has the problem", li([
  "Immuno-oncology discovery and translational teams in biotech and pharma, selecting neoantigen and combination candidates",
  "Academic cancer-immunology labs testing mechanistic hypotheses about the microenvironment",
  "End user: the computational biologist or translational scientist deciding which candidates go into assays",
  "Application area: <b>preclinical immuno-oncology R&amp;D</b> &mdash; neoantigen selection and combination hypothesis generation"]))}
</div>''',
  "We are not proposing a therapeutic and not proposing a clinical tool. The product is software and simulation sold into discovery, which is why the market we compete in is biosimulation rather than oncology therapeutics."))

# 3. Solution: the patient-specific loop
S.append(slide("3 · Solution: one patient, end to end",
  "The same pipeline runs from a patient's raw variants to a simulation of their own tumour&ndash;immune system, with the patient's HLA type deciding what is visible.",
  f'''<div class="flow">
 <div style="border-color:#d9731f"><b>Variants</b><span>VCF ingest and annotation; matched tumour&ndash;normal <strong>somatic calling on NVIDIA Parabricks</strong>, so germline variants are not mistaken for drivers</span></div>
 <i style="color:#d9731f">&#9654;</i>
 <div style="border-color:#d9731f"><b>What is presentable</b><span>HLA typing, then neural peptide&ndash;MHC binding (MHCflurry) against <strong>that patient's own alleles</strong> &mdash; not a reference panel</span></div>
 <i style="color:#d9731f">&#9654;</i>
 <div style="border-color:#2e86ab"><b>Structure</b><span><strong>NVIDIA BioNeMo</strong> hosted NIMs: AlphaFold2-Multimer predicts the peptide&ndash;MHC complex, so an epitope becomes a modelled structure rather than a score</span></div>
 <i style="color:#2e86ab">&#9654;</i>
 <div style="border-color:#76b900"><b>Mechanism</b><span>A ten-module agent-based simulation: cancer, normal, T, NK and macrophage cells under literature-sourced rules. Exports to OpenUSD and renders in <strong>Omniverse</strong></span></div>
</div>
<div class="cols three">
{card("Mechanistic, not predictive", "<p>Cells follow rules drawn from published measurements and the tumour&ndash;immune outcome <i>emerges</i>. So a result decomposes into named causes a scientist can inspect and dispute &mdash; the property a drug programme needs and a black-box score cannot give.</p>", "#76b900")}
{card("Provenance by construction", "<p>Library parameters carry their source (PMID) or are explicitly flagged ESTIMATE. We can say which of our numbers are measured and which are guesses, per parameter. Coverage is partial and openly tracked.</p>", "#76b900")}
{card("Deterministic decision layer", "<p>A multi-agent tumour board ranks treatments with traceable evidence, model cards and provenance, under an FDA clinical-decision-support framing. It is <b>deterministic by design</b> &mdash; no ML model, no LLM &mdash; because auditability is the point.</p>", "#2e86ab")}
</div>''',
  "The Omniverse step is a file handoff, stated precisely: the engine writes a USD stage and Isaac Sim opens it. It is not a live bidirectional coupling."))

# 4. Proof on real data
S.append(slide("4 · Proof on real patient data",
  "On a public reference case the pipeline narrows 1,495 variants to two somatic drivers and three predicted binders on that patient's own HLA &mdash; each one a wet-lab-testable claim.",
  f'''<div class="proof">
 <div class="nums">
  <div><b>1,495 &rarr; 2</b><span>SEQC2 HCC1395 breast cancer: 1,495 variants, 29 coding, <strong>2 surviving somatic filtering</strong> &mdash; ARID1A and NOTCH1</span></div>
  <div><b>3</b><span>predicted strong binders on this patient's typed HLA alleles, via neural pMHC prediction</span></div>
  <div><b>3,841</b><span>automated tests passing, 9 skipped, across the engine, library and genomics paths</span></div>
  <div><b>k = 1.05</b><span>whole-engine cost is near-linear in cell count; about 2.1 s per simulated hour at 2,000 cells</span></div>
 </div>
 <div class="side">
  {card("Why the filtering matters", "<p>Two well-known breast-cancer genes, ESR1 and ERBB2, appear in the raw calls and <b>do not survive</b> somatic filtering on this sample. A pipeline that reported them would look more impressive and be wrong. Reporting two genes instead of four is the result.</p>", "#d9731f")}
  {card("Engineering the long runs need", li([
    "Checkpoint and restart, verified bit-identical to an uninterrupted run across 17 observables including the RNG stream position",
    "Scaling measured and re-measured per module, not asserted"]), "#2e86ab")}
 </div>
</div>''',
  "HCC1395 is a public tumour&ndash;normal reference pair, which is the point: anyone can re-run this. No patient data of our own is used, and none appears in this deck."))

# 5. External validation, stated precisely
S.append(slide("5 · The one external test we have run, in two cohorts",
  "A modelled mechanism was tested against human tumour data the model never saw, and the prediction survived &mdash; then survived again, out-of-sample, in the tissue we actually model. It is a surviving negative prediction, not a confirmation &mdash; and we say so.",
  f'''<div class="cols">
{card("The test", li([
  "<b>Prediction.</b> Our MHC-I mechanism encodes hypoxia destroying MHC-I <i>protein</i> through the unfolded protein response while HLA mRNA rises. If that is right, a hypoxia signature must <b>not</b> correlate negatively with HLA transcript abundance across human tumours.",
  "<b>Data.</b> 592 TCGA colorectal tumours, cBioPortal, RNA-Seq V2 &mdash; the same study, assay and statistic the source paper used.",
  "<b>Result.</b> With IFN-&gamma; partialled out, no MHC-I gene is negatively correlated with hypoxia. The prediction survives.",
  "<b>Replicated in prostate.</b> 493 TCGA prostate tumours &mdash; the tissue our engine models. Same genes, same statistic, same refutation rule; only the cohort changed. HLA-A &minus;0.03, HLA-B +0.11, HLA-C +0.28, B2M +0.27. It survives there too.",
  "<b>Two cohorts run, two reported.</b> None was run and discarded. Adding cohorts until one fails would make this a search rather than a test.",
  "<b>What it is not.</b> mRNA is not surface protein; bulk expression mixes tumour, stroma and immune cells; and a null is consistent with the model rather than proof of it."]))}
{card("Three traps found while running it", li([
  "GAPDH was used as a negative control. It cannot be one &mdash; it is a glycolytic enzyme and a canonical HIF target &mdash; and it came out at rho +0.41 with the hypoxia score.",
  "On raw RSEM <i>every</i> gene correlated positively, beta-actin included. Something sample-wide inflates everything, so each value became a within-sample log ratio.",
  "Hypoxia and IFN-&gamma; are themselves correlated (rho +0.33) and every gene in the panel is IFN-&gamma; inducible. The partial correlation is what the result actually rests on.",
  "<b>Each of the three would have produced a confident wrong answer.</b>"]))}
</div>
<div class="ask"><b>We report our nulls on the same footing.</b> A separate pre-specified outcome test on a clinical cohort came back null at p = 0.55. It is on the record and has not been retired or re-specified. A platform whose negative results are visible is the only kind whose positive results mean anything.</div>''',
  "This is the whole of our external validation. One mechanism, two cohorts, one surviving negative prediction &mdash; and the replication is a new tissue, not a new method, so an analysis artefact would survive into it. Everything else in the system is internally consistent and externally untested &mdash; which is precisely the gap this programme would close."))

# 6. NVIDIA
S.append(slide("6 · The NVIDIA stack, and what we contribute back",
  "Five NVIDIA technologies are in the pipeline today, spanning sequence to structure to simulation to render &mdash; and the schema we built to carry biology in OpenUSD is already public.",
  f'''<div class="cols three">
{card("BioNeMo models", "<p>Eleven hosted NIM clients. <b>AlphaFold2-Multimer</b> predicts peptide&ndash;MHC complexes inside the neoantigen path itself; ESM-2, OpenFold3, Boltz-2, DiffDock, MolMIM, GenMol, RFdiffusion, ProteinMPNN, Evo2 and MSA-Search are integrated alongside it.</p>", "#76b900")}
{card("Parabricks", "<p>Matched tumour&ndash;normal somatic variant calling on an L40S &mdash; the front of the pipeline, and what makes the two-gene result on HCC1395 defensible rather than a germline artefact.</p>", "#76b900")}
{card("Omniverse &amp; OpenUSD", "<p>A run exports to USD and streams in Isaac Sim 6.0 at 37&ndash;50 fps, so a simulation is something a scientist watches and interrogates rather than a table they read.</p>", "#76b900")}
{card("Data Center GPUs", "<p>L40S for the genomics and rendering stages.</p>", "#2e86ab")}
{card("NVIDIA Warp", "<p>Warp kernels back the physics paths and initialise on every run. The agent-based engine itself is CPU and measured near-linear; we do not claim it is GPU-resident.</p>", "#2e86ab")}
{card("Bio-USD &mdash; our contribution", "<p>A draft open schema for biological entities in OpenUSD. Published Apache 2.0 with a citable concept DOI (10.5281/zenodo.23603569) and engaged directly by Pixar's team on the OpenUSD forum. Infrastructure the field can adopt, not a feature we keep.</p>", "#d9731f")}
</div>''',
  "We are an Omniverse and BioNeMo user with a published OpenUSD contribution, so the partnership conversation is about an ecosystem contribution rather than a favour."))

# 7. The ask
S.append(slide("7 · The ask: one pre-registered experiment",
  "Synthesise the peptides we predict from a patient's own variants and HLA type, and measure whether they bind. The prediction is registered before the assay runs, so either outcome is informative.",
  f'''<div class="cols">
{card("The experiment", li([
  "<b>Input.</b> The three peptides our pipeline predicts as strong binders on HCC1395's own typed HLA alleles, plus predicted non-binders from the same sample as negative controls.",
  "<b>GenScript capability.</b> Peptide synthesis, then binding measurement against those alleles &mdash; Experimental Validation and Closed-Loop Learning.",
  "<b>Registered in advance.</b> The peptide list and the predicted call for each are fixed before synthesis. No post-hoc selection is possible."]))}
{card("Why either outcome pays", li([
  "<b>If they bind:</b> the patient-specific neoantigen path has experimental evidence for the first time, and the claim changes from in-silico to evidenced.",
  "<b>If they do not:</b> the error localises to presentation scoring rather than to the simulation, which is a far more useful failure than a model that was never tested.",
  "<b>Either way:</b> protein expression and functional validation would then let us replace estimated immune-engagement constants with measured ones &mdash; the closed loop that turns a mechanistic model into a calibrated one."]))}
</div>
<div class="ask"><b>And a second ask, which we rate as highly as the credits.</b> Cognisom is built AI-first, which is why one founder covers genomics, simulation, parameter sourcing and rendering. The consequence is that the engineering is ahead of the domain review: our open questions are now scientific, not technical &mdash; which immune mechanisms deserve priority, which estimated constants most need measuring, which predictions are worth pre-registering next. This programme offers introductions to scientific experts, and that is the input we most need. We are looking to add research-level immunology, tumour-biology and translational expertise to validate and steer direction.</div>'''))

# 8. Market
S.append(slide("8 · Market and traction",
  "We sell software and simulation into drug discovery, so our market is biosimulation. Traction is technical, not commercial, and we would rather say so than inflate it.",
  f'''<div class="cols">
{card("The market we compete in", li([
  "<b>Biosimulation: USD 5.7 bn in 2026</b>, from 4.9 bn in 2025, forecast to 25.1 bn by 2035 at a 17.9% CAGR.",
  "Led by Simulations Plus, Certara, Dassault Syst&egrave;mes, Schr&ouml;dinger and ACD/Labs, together about 35% share &mdash; an established category with public comparables, not a category we have to invent.",
  "<b>The demand driver sits one layer out:</b> personalised neoantigen cancer vaccines, USD 1.4 bn in 2026, forecast to 12.8 bn by 2036 at 24.6%. That is the market our <i>customers</i> chase &mdash; not revenue available to us.",
  "Both are third-party vendor estimates and we treat them as directional."]))}
{card("Traction, stated plainly", li([
  "<b>No pilots, letters of intent, design partners or grants to date.</b> Effort has gone into making the prediction path real and falsifiable before taking on commitments we could not support.",
  "What exists: a closed loop on public reference data, one externally tested mechanism, a published open standard with a DOI, and five NVIDIA technologies in production use.",
  "The next milestone is deliberately experimental rather than commercial &mdash; the pre-registered binding assay on the previous slide.",
  "Institutional: Cognisom is a <b>Harvard Innovation Labs</b> venture, and Eyentelligence is an <b>NVIDIA Inception</b> member.",
  "<b>No formal scientific advisor yet</b> &mdash; early conversations with a Harvard faculty member, and nothing signed. Closing that is what we want from this programme's introductions."]))}
</div>''',
  'Sources: biosimulation figures, Global Market Insights, Biosimulation Market, accessed 2 November 2026. Neoantigen cancer vaccine figures, Meticulous Research, accessed 2 November 2026.'))

# 9. What we do not claim
S.append(slide("9 · What we do not claim",
  "A platform is only as credible as the list of things it refuses to say. This slide is the list, and we would rather hand it to you than have you find it.",
  f'''<div class="cols three">
{card("Not validated", "<p>One mechanism has survived one external falsifiable test. That is not a validated platform, and we never describe it as one. Our clinical-cohort outcome test came back null and stays on the record.</p>", "#c2185b")}
{card("Not clinical", "<p>Research use only. No clinical validation exists, and nothing here predicts a patient's response to therapy. The decision layer is framed as clinical decision support and is not a cleared device.</p>", "#c2185b")}
{card("Not experimentally confirmed", "<p>No prediction this system makes has been tested at a bench. The neoantigen calls, the binding predictions and the simulated immune dynamics are all in-silico today.</p>", "#c2185b")}
{card("Not fully sourced", "<p>Parameter provenance is partial. Unsourced constants are flagged ESTIMATE rather than hidden, and several load-bearing numbers &mdash; immune kill rates among them &mdash; are still estimates.</p>", "#5b6373")}
{card("Not GPU-resident, not agentic", "<p>The agent-based engine runs on CPU; GPU paths are opt-in. The tumour board is deterministic with no ML model and no LLM. Both are accurate descriptions, not modesty.</p>", "#5b6373")}
{card("No patent claim", "<p>We make none. Our contribution to the commons is the published Bio-USD schema under Apache 2.0, which is the part we would rather be judged on.</p>", "#5b6373")}
</div>''',
  "Every item above is recorded in the project's own engineering notes, which is why it can be stated this precisely. We would rather be the team that volunteers its falsification criteria than the one that gets caught lacking them."))

# 10. Team and close
S.append(slide("10 · Team: AI-led, and deliberately adding science",
  "Cognisom is built AI-first. That is how one founder covers genomics to simulation to rendering &mdash; and it is why the expertise we are adding next is scientific, not engineering.",
  f'''<div class="team">
 <div class="person">{headshot()}
  <h3>David Walker<span>Founder and CEO, Eyentelligence, Inc. &mdash; AI-led development &middot; Flower Mound, TX</span></h3>
  {li(["<b>Builds with AI as the primary method.</b> The whole pipeline &mdash; variant ingestion and somatic calling, HLA typing and neoantigen prediction, the ten-module agent-based tumour&ndash;immune engine, the literature-sourced parameter library, the OpenUSD and Omniverse layers &mdash; built by one founder working AI-first",
       "<b>With the discipline to match the speed:</b> 3,841 automated tests; optimisations verified bit-identical before shipping; scaling measured and re-measured rather than asserted; every parameter sourced to a PMID or flagged ESTIMATE",
       "Authored and published Bio-USD, the draft open OpenUSD schema for biological data, under Apache 2.0 with a citable DOI, engaged directly by Pixar's OpenUSD team",
       "<b>38 years in eyecare technology and supply chain</b> before this: Essilor of America 1989&ndash;2017; VP, AcuityLogic at Eyefinity 2019&ndash;2026, leading its first AI capabilities. Harvard Extension School ALM from September 2026, ALB expected 2027",
       "Cognisom is a <b>Harvard Innovation Labs</b> venture; Eyentelligence is an <b>NVIDIA Inception</b> member. <b>No formal scientific advisor is in place</b> &mdash; early conversations with a Harvard faculty member, nothing signed"])}
 </div>
 <div class="side">
  {card("What we are adding, and why", "<p>AI-first development moved the bottleneck. The engineering is ahead of the domain review, so <b>our open questions are now scientific rather than technical.</b> We are actively seeking research-level immunology, tumour-biology and translational collaborators to validate direction and tell us which mechanisms and which unmeasured constants deserve priority.</p>", "#d9731f")}
  {card("If selected", li([
    "Run the pre-registered peptide assay and publish the result either way",
    "Convene scientific reviewers on direction &mdash; the introductions matter as much as the credits",
    "Feed measured binding back into the library, replacing estimates with measurements"]), "#76b900")}
  {card("Contact", f"<p><b>David Walker</b><br>david@eyentelligence.com<br>cognisom.com<br>214-208-5261</p><p style='color:#5b6373;font-size:9.5pt'>Research use only. Not for clinical use. No confidential information and no patient data in this deck.</p>")}
 </div>
</div>'''))

CSS = '''
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Sans:wght@400;600&display=swap');
:root{--navy:#14213d;--ink:#17213a;--muted:#5b6373;--paper:#f7f6f2;--panel:#ffffff;--line:#d9dde3;--orange:#d9731f;--blue:#2e86ab;--green:#76b900;--fill:#c2185b}
*{box-sizing:border-box}
@page{size:13.333in 7.5in;margin:0}
html,body{margin:0;padding:0;background:#777;font-family:'IBM Plex Sans',Arial,sans-serif;color:var(--ink)}
.slide{width:13.333in;height:7.5in;padding:0.55in 0.7in 0.5in;background:var(--paper);position:relative;overflow:hidden;page-break-after:always;break-after:page;display:flex;flex-direction:column;gap:0.26in;margin:0 auto}
@media screen{.slide{margin:12px auto;box-shadow:0 2px 12px rgba(0,0,0,.35)}}
.eyebrow{display:flex;justify-content:space-between;font-size:10.5pt;letter-spacing:.08em;text-transform:uppercase;color:var(--orange);font-weight:600}
h1,h2,h3{font-family:'Source Serif 4',Georgia,serif;margin:0;text-wrap:balance}
h2{font-size:21pt;line-height:1.18;font-weight:600;color:var(--navy);max-width:12in}
h3{font-size:12.5pt;font-weight:600;margin-bottom:4pt}
h3 span{display:block;font-family:'IBM Plex Sans',Arial,sans-serif;font-size:10pt;color:var(--muted);font-weight:400}
.body{flex:1;display:flex;flex-direction:column;gap:0.14in;min-height:0;justify-content:center}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:0.22in}
.cols.three{grid-template-columns:1fr 1fr 1fr}
.card{background:var(--panel);border:1px solid var(--line);border-top:3px solid var(--navy);border-radius:6px;padding:0.16in 0.2in;font-size:10.5pt;line-height:1.38}
.card p{margin:0}
ul{margin:0;padding-left:1.05em}li{margin:0 0 4pt}
.foot{font-size:9pt;color:var(--muted);border-top:1px solid var(--line);padding-top:5pt;line-height:1.35}
.fill{display:inline-block;background:#fde7f0;color:var(--fill);border:1px dashed var(--fill);border-radius:4px;padding:0 5px;font-size:.92em;font-weight:600}
.cover{background:var(--navy);color:#f3f1ea}
.cover .eyebrow{color:#e9893a}
.cover-grid{display:grid;grid-template-columns:1.25fr 1fr;gap:0.4in;align-items:center;flex:1}
.brand{font-family:'Source Serif 4',Georgia,serif;font-size:30pt;font-weight:600;margin-bottom:0.12in}
.brand span{font-family:'IBM Plex Sans',Arial,sans-serif;font-size:12pt;font-weight:400;color:#bfc8d8;margin-left:8pt}
.cover h1{font-size:22pt;line-height:1.2;font-weight:600;color:#f3f1ea}
.cover h1 em{font-style:normal;color:#e9893a}
.lede{font-size:12pt;line-height:1.4;color:#d5dbe6;margin:0.14in 0 0}
.chips{display:flex;flex-wrap:wrap;gap:6pt;margin-top:0.16in}
.chip{background:#243358;border-radius:999px;padding:3pt 10pt;font-size:9.5pt;color:#eef1f7}
.cover .fill{background:#3a1f33;color:#ff8fc0;border-color:#ff8fc0}
.cover .card{background:#1b2a4a;border-color:#2c3e63;color:#e8ecf4}
.cover .card h3{color:#f3f1ea}
.cover .card h3 span{color:#9fb0cc}
figure{margin:0;display:flex;flex-direction:column;gap:5pt;min-width:0}
figure img{width:100%;border-radius:6px;display:block}
figcaption{font-size:9pt;color:var(--muted)}
.cover figcaption{color:#bfc8d8}
.proof{display:grid;grid-template-columns:1.1fr 1fr;gap:0.3in;align-items:start}
.nums{display:grid;grid-template-columns:1fr 1fr;gap:0.12in}
.nums div{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:0.12in 0.16in}
.nums b{display:block;font-family:'Source Serif 4',Georgia,serif;font-size:20pt;color:var(--navy);line-height:1.1}
.nums span{font-size:9.5pt;color:var(--muted);line-height:1.3;display:block;margin-top:3pt}
.matrix{width:100%;border-collapse:collapse;font-size:10pt;background:var(--panel)}
.matrix th,.matrix td{border:1px solid var(--line);padding:6pt 8pt;text-align:left;vertical-align:top}
.matrix th{background:#eef1f6;font-weight:600}
.team{display:grid;grid-template-columns:1.2fr 1fr;gap:0.25in}
.person{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:0.18in 0.22in;font-size:10.5pt;line-height:1.38;display:grid;grid-template-columns:1.1in 1fr;gap:0.18in;align-content:start}
.person h3{grid-column:2}.person ul{grid-column:2}
.avatar{width:1.1in;height:1.1in;border-radius:50%;background:#e7e9ee;display:flex;align-items:center;justify-content:center;text-align:center;font-size:8pt;grid-row:1/3;align-self:start;overflow:hidden}
.avatar img{width:100%;height:100%;object-fit:cover;display:block}
.side{display:flex;flex-direction:column;gap:0.16in}
.flow{display:grid;grid-template-columns:1fr auto 1fr auto 1fr auto 1fr;gap:0.08in;align-items:center}
.flow div{background:var(--panel);border:1.5px solid var(--navy);border-radius:6px;padding:0.1in 0.14in;align-self:stretch}
.flow b{display:block;font-family:'Source Serif 4',Georgia,serif;font-size:12pt;color:var(--navy);margin-bottom:3pt}
.flow span{font-size:9.5pt;line-height:1.32;display:block}
.flow i{font-style:normal;font-size:13pt}
.ask{background:#eef5e3;border-left:4px solid var(--green);padding:0.12in 0.18in;font-size:11pt;line-height:1.4;border-radius:4px}
</style>'''

doc = ("<!doctype html><html><head><meta charset='utf-8'>"
       "<title>Cognisom · GenScript × NVIDIA Life Sciences PitchFest</title>"
       + CSS + "</head><body>" + "\n".join(S) + "</body></html>")
OUT_HTML.write_text(doc, encoding="utf-8")
print("html", OUT_HTML, len(S), "slides")
if "--pdf" in sys.argv:
    subprocess.run([CHROME, "--headless", "--no-sandbox", "--disable-gpu",
                    "--no-pdf-header-footer", "--virtual-time-budget=8000",
                    f"--print-to-pdf={OUT_PDF}", f"file://{OUT_HTML}"],
                   check=True, capture_output=True)
    size = OUT_PDF.stat().st_size
    print("pdf", OUT_PDF, size, "bytes (%.1f MB, cap is 30)" % (size / 1e6))
