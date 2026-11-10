"""Whose genome a run is about, and what geometry it runs on.

The gap this closes
-------------------
``EngineRunner`` grew two parameters that decide what a run *means* --
``patient``, which puts this patient's own missense mutations and HLA
class-I alleles on the cancer cells, and ``tissue``, which chooses the
geometry those cells sit on -- and no page in the dashboard passed
either one. Every simulation the app ran was therefore a generic
``KRAS_G12D`` tumour on the close-packed slab, with nothing on screen
saying so. Personalisation is the platform's central claim, which made
this the most consequential instance of the built-but-never-connected
pattern still open in the UI.

Both halves already existed and already agreed on a key: seven pages
write a built profile to ``st.session_state["patient_profile"]``, and
``engine_runner`` reads a profile through
``patient_engine.engine_inputs``. Only the join was missing.

Why this is a module and not three lines inside a page
-----------------------------------------------------
The defect was never a crash -- it was silence. A run with no patient
produced a perfectly ordinary-looking result that simply was not about
anybody. So the fix is a sentence the page is obliged to print, and a
sentence can be tested where a Streamlit widget cannot: every function
here takes a plain mapping and returns plain data, so a test can ask
"does an unpersonalised run say it is unpersonalised" without a
browser.

The subject is described BEFORE the run, not after, because a 24 h run
costs ~15 s and a 72 h run ~45 s; being told whose genome it was after
waiting is too late to be useful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

#: Where every page that builds a profile puts it. app.py, 26_genomic_twin,
#: 28_digital_twin, 32_parabricks, 35_orchestrator, 37_mad_agent and
#: 38_real_patient all write this key; they agreed on it long before
#: anything read it for a simulation.
SESSION_PATIENT_KEY = "patient_profile"

#: What the engine falls back to with no patient. Not a neutral default:
#: KRAS G12D is a real driver this patient may well not carry, and the
#: generic HLA panel is not their typing, so neoantigen presentation and
#: NK licensing are both about somebody who does not exist.
GENERIC_HEADLINE = "No patient loaded — generic genome"
GENERIC_CAVEAT = (
    "Every cancer cell will carry the placeholder KRAS_G12D driver on a "
    "generic HLA panel. This run is a mechanism demonstration, not a "
    "personalised twin."
)

#: ``EngineRunner(tissue=...)`` accepts these. The first is its default.
TISSUE_CHOICES = ("section", "slab", "ring")

TISSUE_LABELS = {
    "slab": "Close-packed slab",
    "section": "Measured prostate section",
    "ring": "Legacy calibration ring",
}

#: Each caption says what the geometry IS and what it costs. The slab
#: fills every site of its lattice; a real open prostate section fills
#: about 0.60 of a close packing, so the slab hands each cell ~7.8
#: neighbours where the section gives 5.0 -- it over-connects every cell
#: by roughly 1.6x. That is why the section became the default on
#: 2026-11-07, and why the slab's caption still has to say so.
#: Measured from a 10x Xenium prostate section; see project memory
#: "Real section geometry 2026-11-03".
TISSUE_CAPTIONS = {
    "slab": (
        "Every lattice site filled. Over-connects each cell by ~1.6x "
        "against real prostate (7.8 neighbours vs 5.0), so contact-"
        "mediated effects are stronger than in tissue. Kept because the "
        "longest-running pinned outcomes were measured on it."
    ),
    "section": (
        "Occupancy taken from a measured human prostate section (10x "
        "Xenium): glands pack, lumens stay empty. ~5.0 neighbours per "
        "cell, which is the honest figure. The default."
    ),
    "ring": (
        "The 90-cell necklace the module dynamics were first calibrated "
        "on. Kept so those calibrations can still be reproduced; it is "
        "not a tissue geometry."
    ),
}


@dataclass
class RunSubject:
    """Who and what a run is about, in a form a page can print."""

    patient: Optional[Any]
    patient_id: str = ""
    personalised: bool = False
    headline: str = GENERIC_HEADLINE
    #: Everything true of this subject that weakens the result. Empty is
    #: a claim, so it is only ever empty when nothing applies.
    caveats: List[str] = field(default_factory=list)
    mutation_labels: List[str] = field(default_factory=list)
    hla_alleles: List[str] = field(default_factory=list)
    hla_measured: bool = False
    #: (gene, change, reason) for each coding variant that cannot cross.
    skipped: List[tuple] = field(default_factory=list)


def patient_from_session(session: Optional[Mapping[str, Any]]) -> Optional[Any]:
    """The profile a page left in session state, or None.

    Duck-typed on purpose. A ``PatientProfile`` is what the pages store,
    but ``engine_inputs`` reads it entirely through ``getattr`` with
    defaults, so anything carrying the same attributes works and an
    isinstance check would reject a legitimate stand-in. What is NOT
    accepted is an object missing both of the fields that make a profile
    mean anything -- a bare dict or a sentinel would otherwise sail
    through and produce an empty, silent personalisation.
    """
    if not session:
        return None
    try:
        profile = session.get(SESSION_PATIENT_KEY)
    except Exception:                                        # noqa: BLE001
        return None
    if profile is None:
        return None
    if not hasattr(profile, "coding_variants") and not hasattr(profile, "hla_alleles"):
        return None
    return profile


def describe_subject(patient: Optional[Any]) -> RunSubject:
    """What this run is about, and every reason to discount it.

    Calls ``engine_inputs`` -- the same mapping the engine will do -- so
    the page reports what will actually cross over rather than what the
    profile happens to contain. That is a second call to a cheap pure
    function, deliberately: describing the run from a different code
    path than the one that performs it is how the two drift apart.
    """
    if patient is None:
        return RunSubject(patient=None, caveats=[GENERIC_CAVEAT])

    from cognisom.dashboard.patient_engine import engine_inputs

    inputs = engine_inputs(patient)
    caveats: List[str] = []

    n_mut = len(inputs.mutations)
    alleles = list(dict.fromkeys(inputs.hla_alleles))
    headline = (
        f"Patient {inputs.patient_id} — {n_mut} missense "
        f"mutation{'s' if n_mut != 1 else ''} on "
        f"{len(alleles)} HLA class-I allele{'s' if len(alleles) != 1 else ''}"
    )

    if n_mut == 0:
        caveats.append(
            "None of this patient's coding variants makes a canonical "
            "missense neoantigen, so the cancer cells present nothing "
            "patient-specific. The run is still on their HLA."
        )
    if not alleles:
        caveats.append(
            "No HLA class-I alleles on this profile, so presentation falls "
            "back to the generic panel and NK licensing has no typing."
        )
    elif not inputs.hla_measured:
        caveats.append(
            f"HLA typing method is {inputs.hla_method!r}, which is not "
            "derived from this patient's reads. The NK licensing model "
            "refuses an unmeasured typing on purpose."
        )
    if getattr(patient, "has_matched_normal", False) is False:
        caveats.append(
            "The callset had no matched normal, so these variants are not "
            "established as somatic — a rare inherited variant is "
            "indistinguishable from a tumour mutation here."
        )
    if inputs.skipped:
        caveats.append(
            f"{len(inputs.skipped)} coding variant"
            f"{'s' if len(inputs.skipped) != 1 else ''} cannot cross over "
            "(nonsense, frameshift, synonymous, or a wild-type residue that "
            "does not match the reference). Each is listed with its reason."
        )

    # Disclosures the profile carries about its own provenance -- biopsy
    # site, prior treatment, genome build. These cannot be derived from
    # the callset, so they are declared where the cohort is declared
    # (``genomics/demo_patients.py``) and surfaced here beside the ones
    # computed above. They go LAST because the computed caveats describe
    # what this run did, and these describe what it is a run OF.
    caveats.extend(
        str(c) for c in (getattr(patient, "provenance_caveats", None) or [])
    )

    return RunSubject(
        patient=patient,
        patient_id=inputs.patient_id,
        personalised=True,
        headline=headline,
        caveats=caveats,
        mutation_labels=[m.label for m in inputs.mutations],
        hla_alleles=alleles,
        hla_measured=bool(inputs.hla_measured),
        skipped=list(inputs.skipped),
    )


def presentation_lines(report: Optional[Dict[str, Any]]) -> List[str]:
    """``EngineRunner.presentation_report`` as sentences.

    A mutation the immune system cannot see is a result, not an error,
    so an unpresented mutation gets a line of its own rather than being
    omitted for looking like a failure.
    """
    if not report:
        return []
    lines: List[str] = []
    presented = list(report.get("presented") or [])
    unpresented = list(report.get("unpresented") or [])
    displaying = int(report.get("cells_displaying") or 0)
    cancer_cells = int(report.get("cancer_cells") or 0)

    if reason := report.get("reason"):
        lines.append(f"Nothing presented: {reason}.")
        return lines

    if presented:
        lines.append(
            f"Presented on MHC-I: {', '.join(presented)} — displayed by "
            f"{displaying} of {cancer_cells} founding cancer cells."
        )
    else:
        lines.append(
            f"None of this patient's mutations bound any of their own HLA "
            f"alleles, so all {cancer_cells} founding cancer cells are "
            "invisible to a T cell. That is a result, not a failure."
        )
    for item in unpresented:
        gene = item.get("gene") or item.get("label") or "?"
        why = item.get("reason") or "no peptide bound"
        lines.append(f"Not presented — {gene}: {why}.")
    return lines


def tissue_caption(tissue: str) -> str:
    """The one-line honest description of a geometry choice."""
    return TISSUE_CAPTIONS.get(tissue, "")
