"""
Provenanced simulation parameters for entity-library entities.

Why this exists
---------------

Two defects in the library had the same root cause: a number that nobody
could tell apart from a measurement.

Every seeded cytokine carried a serum half-life of 0.0, and every seeded
immune cell type carried a per-contact kill probability of 0.8. Neither
had been curated. Both were class defaults that no seeding code
overrode, and because a bare ``float`` records nothing about where it
came from, a reader (or a simulator, or a reviewer) had no way to
distinguish them from values someone had looked up.

Filling those fields in with plausible numbers would have made the
problem worse rather than better. An unsourced value that *looks*
authoritative is more dangerous than an obvious placeholder, because it
survives review. The failure mode is not hypothetical in this codebase:
a variant annotator once generated protein changes from a four-entry
lookup table, and its output was indistinguishable downstream from real
transcript-aware annotation.

So the fix is not "add the numbers". It is to make a parameter carry its
own evidence, and to make code able to ask how good that evidence is.

What a parameter must declare
-----------------------------

A :class:`ParameterValue` binds four things that a bare float separates:

    value       the number
    unit        what the number is in
    provenance  how it was arrived at
    evidence    a citation, or a rationale, depending on provenance

The provenance levels form a quality ordering, and each carries an
obligation that :meth:`ParameterValue.validate` enforces:

    MEASURED     from a publication or curated database. Needs a citation.
    DERIVED      computed from other measured values. Needs the derivation.
    ESTIMATED    expert judgement or analogy. Needs a written rationale.
    PLACEHOLDER  arbitrary, to make something run. Needs nothing, and is
                 excluded from simulation-grade by construction.

The point of PLACEHOLDER is that it is *sayable*. Smoke tests and demos
legitimately need a number. What they must not do is pretend it is one.

Usage
-----

    >>> from cognisom.library.models import Cytokine
    >>> from cognisom.library.parameters import (
    ...     ParameterValue, Provenance, set_parameter, get_parameter)
    >>> il2 = Cytokine(name="IL-2")
    >>> set_parameter(il2, "half_life_hours", ParameterValue(
    ...     value=0.28, unit="h", provenance=Provenance.MEASURED,
    ...     source="Konrad et al.", citation="PMID:2985038",
    ...     conditions="human, intravenous bolus, serum"))
    >>> get_parameter(il2, "half_life_hours").value
    0.28
    >>> il2.physics_params["half_life_hours"]     # flat read path preserved
    0.28

The flat ``physics_params`` scalar is written alongside the record, so
every existing reader keeps working unchanged. The provenance lives in a
sidecar keyed by the same name, and :func:`set_parameter` is the only
supported way to write either, so the two cannot drift apart.

What a single scalar per entity cannot say
------------------------------------------

``inapplicable_when`` removes the parameters a lineage does not have, but
several rows that keep a parameter still cannot be described honestly by
one number, and that is a property of the schema rather than of the
curation. Recording it here because the alternative is a curated value
that looks settled:

* **Activation state is unrepresentable.** Nearly every immune effector
  quantity is conditional -- a resting eosinophil releases nothing, a
  pDC kills only via TLR7/9-induced TRAIL, a memory T cell's progeny kill
  while the resting cell does not. Entities carry ``activated``,
  ``activation_state`` and ``phase``, but no solver reads them; they are
  only serialized. So "only when activated" cannot be stored at all.
* **One row per lineage lumps distinct subsets.** ``CD8+ Memory T cell``
  spans Tcm, Tem and Trm, whose cytotoxicity differs; its recorded
  markers (CCR7+) are the least cytotoxic of the three while its
  description claims rapid killing. ``Gamma-delta T cell`` spans
  cytotoxic Vg9Vd2 and pro-tumour IL-17+ subsets, and its own
  ``cytokines_secreting`` lists both IFN-g and IL-17A. ``NK cell``
  lumps CD56bright and CD56dim.
* **Context can invert the sign.** Tumour-associated neutrophils and M2
  macrophages are net pro-tumour; ILC3's IL-22 is tumour-promoting in
  colitis-associated cancer; Treg granzyme-B killing targets effector
  lymphocytes and so suppresses clearance. A kill probability has no
  sign, so these cannot be expressed as a smaller kill probability.
* **Restriction elements are missing.** CD4 killing is MHC class II
  restricted and NKT recognition is CD1d restricted. The model has only
  ``mhc1_expression``, so a kill licence on those rows is enforced
  against the wrong molecule.
* **No lifespan parameter exists for a cell.** Marking
  ``division_time_hours`` inapplicable for the six post-mitotic rows
  leaves them with no temporal parameter at all: a neutrophil that
  neither divides nor clears. ``half_life_hours`` applies only to
  CYTOKINE, LIGAND and PROTEIN, and ``apoptosis_rate_per_hour`` is unset
  on all 28. The plasma-cell span -- days for a plasmablast against
  years for a marrow resident -- is a lifespan bimodality with nowhere
  to live.
* **Secreted-mediator fields are not a cytotoxicity signal.**
  ``cytokines_secreting`` is used as "secreted mediators" and carries
  perforin, granzyme_B, NO, histamine and MBP, none of which are
  cytokines. That is why the exclusion tables key on lineage and not on
  this field.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from .models import BioEntity, EntityType

# Key under which the provenance sidecar is stored on an entity.
PROVENANCE_KEY = "parameter_provenance"


class Provenance(str, Enum):
    """How a parameter value was arrived at, worst to best."""

    PLACEHOLDER = "placeholder"
    ESTIMATED = "estimated"
    #: Fitted so that the model reproduces a stated observable. Not a
    #: measurement of the quantity itself -- nobody measures "the
    #: per-contact kill probability of a CD8 T cell", because it is an
    #: abstraction over a process -- but not a guess either, because it is
    #: constrained by something real. Without this grade such values sit
    #: as placeholders forever, counted as missing citations that no
    #: database will ever supply.
    CALIBRATED = "calibrated"
    DERIVED = "derived"
    MEASURED = "measured"


# Ascending quality. Used for comparisons and for picking the better of
# two competing values for the same parameter.
_PROVENANCE_RANK: Dict[Provenance, int] = {
    Provenance.PLACEHOLDER: 0,
    Provenance.ESTIMATED: 1,
    Provenance.CALIBRATED: 2,
    Provenance.DERIVED: 3,
    Provenance.MEASURED: 4,
}

#: Provenance levels a simulation may legitimately consume.
SIMULATION_GRADE = frozenset(
    {Provenance.ESTIMATED, Provenance.CALIBRATED, Provenance.DERIVED,
     Provenance.MEASURED}
)

#: The stricter bar: values defensible in a paper or a patent filing.
#: Defensible in a paper. A calibrated value qualifies only because it is
#: required to name the observable it was fitted to, which is the thing a
#: reviewer would ask for.
PUBLICATION_GRADE = frozenset(
    {Provenance.CALIBRATED, Provenance.DERIVED, Provenance.MEASURED}
)


@dataclass
class ParameterValue:
    """A simulation parameter together with the evidence behind it."""

    value: float
    unit: str
    provenance: Provenance
    #: Stable identifier for the evidence: "PMID:2985038", a DOI, or a
    #: database accession. Required for MEASURED.
    citation: str = ""
    #: Human-readable origin: "SABIO-RK", "BRENDA", "Alberts 6e".
    source: str = ""
    #: Experimental conditions the value was measured under. Two Km
    #: values at different pH are not interchangeable.
    conditions: str = ""
    #: Why this number, for values that are not straight measurements.
    #: Required for ESTIMATED and DERIVED.
    rationale: str = ""
    #: What a CALIBRATED value was fitted to reproduce: the observable,
    #: its source, and the tolerance. Required for that provenance, because
    #: a calibration nobody can re-run is not evidence.
    calibrated_against: str = ""
    #: Optional subjective confidence in [0, 1].
    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        if isinstance(self.provenance, str):
            self.provenance = Provenance(self.provenance)

    # ── Quality ──────────────────────────────────────────────────────

    @property
    def is_simulation_grade(self) -> bool:
        """True if a simulation may consume this value."""
        return self.provenance in SIMULATION_GRADE

    @property
    def is_publication_grade(self) -> bool:
        """True if this value is defensible in a paper or a filing."""
        return self.provenance in PUBLICATION_GRADE

    @property
    def rank(self) -> int:
        return _PROVENANCE_RANK[self.provenance]

    # ── Validation ───────────────────────────────────────────────────

    def validate(self, spec: Optional["ParameterSpec"] = None) -> List[str]:
        """Return a list of problems; empty means the value is sound.

        Checks the evidence obligation that the declared provenance
        carries, and, when a spec is supplied, the unit and plausible
        range.
        """
        problems: List[str] = []

        if not isinstance(self.value, (int, float)) or isinstance(self.value, bool):
            problems.append(f"value must be a number, got {type(self.value).__name__}")
        elif not math.isfinite(float(self.value)):
            problems.append(f"value must be finite, got {self.value}")

        if not self.unit:
            problems.append("unit is required; a bare number is not a quantity")

        # The obligation attached to each provenance level. This is what
        # stops a guess being recorded as a measurement.
        if self.provenance is Provenance.MEASURED and not self.citation:
            problems.append(
                "MEASURED requires a citation (PMID, DOI, or database "
                "accession); use ESTIMATED with a rationale if there is none"
            )
        if self.provenance is Provenance.DERIVED and not self.rationale:
            problems.append(
                "DERIVED requires a rationale giving the derivation"
            )
        if self.provenance is Provenance.ESTIMATED and not self.rationale:
            problems.append(
                "ESTIMATED requires a rationale saying what the estimate "
                "is based on"
            )
        if self.provenance is Provenance.CALIBRATED and not self.calibrated_against:
            problems.append(
                "CALIBRATED requires calibrated_against: the observable the "
                "value was fitted to reproduce. A fit with no stated target "
                "is a guess wearing a better word"
            )

        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            problems.append(f"confidence must be in [0, 1], got {self.confidence}")

        if spec is not None:
            problems.extend(spec.check(self))

        return problems

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "value": float(self.value),
            "unit": self.unit,
            "provenance": self.provenance.value,
        }
        for key in ("citation", "source", "conditions", "rationale",
                    "calibrated_against"):
            val = getattr(self, key)
            if val:
                d[key] = val
        if self.confidence is not None:
            d["confidence"] = self.confidence
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ParameterValue":
        return cls(
            value=d["value"],
            unit=d.get("unit", ""),
            provenance=Provenance(d.get("provenance", "placeholder")),
            citation=d.get("citation", ""),
            source=d.get("source", ""),
            conditions=d.get("conditions", ""),
            rationale=d.get("rationale", ""),
            calibrated_against=d.get("calibrated_against", ""),
            confidence=d.get("confidence"),
        )

    def evidence_lines(self) -> List[Tuple[str, str]]:
        """Every evidence field that is set, in the order a reader needs.

        The fields are not alternatives and must not be rendered with
        ``a or b or c``: a citation names the paper, ``conditions`` says
        what was measured and in what buffer, and ``rationale`` says how
        the stored number was got from the paper. Short-circuiting on the
        citation is how a DERIVED median over four orders of magnitude
        came to render as a plain PMID -- and every DERIVED value is
        required to carry a rationale precisely because for a derived
        number the rationale IS the evidence.
        """
        pairs = (("citation", self.citation),
                 ("calibrated against", self.calibrated_against),
                 ("source", self.source),
                 ("conditions", self.conditions),
                 ("rationale", self.rationale))
        return [(label, str(value)) for label, value in pairs if value]

    def describe(self) -> str:
        """One line, which says what it is leaving out.

        A single line cannot carry a 400-character rationale, so it names
        the first evidence field and then NAMES THE REST rather than
        dropping them silently. A reader who sees "(+conditions,
        +rationale)" knows to look; a reader who saw only the PMID did
        not.
        """
        lines = self.evidence_lines()
        if not lines:
            evidence = "no evidence"
        else:
            rest = [label for label, _ in lines[1:]]
            evidence = lines[0][1]
            if rest:
                evidence += " (+" + ", +".join(rest) + ")"
        return f"{self.value} {self.unit} [{self.provenance.value}: {evidence}]"


@dataclass(frozen=True)
class ParameterSpec:
    """The contract for one named parameter.

    Declaring the unit and a plausible range is what lets a wrong-unit
    or nonsensical value be rejected at write time rather than producing
    quietly wrong dynamics.
    """

    name: str
    unit: str
    description: str
    #: Entity types this parameter applies to.
    applies_to: Tuple[EntityType, ...]
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    #: True if a simulation cannot meaningfully use the entity without it.
    required_for_simulation: bool = False
    #: True when the quantity is an abstraction over a process rather than
    #: something anyone measures. Nobody reports "the per-contact kill
    #: probability of a CD8 T cell": it stands in for synapse formation,
    #: granule delivery and target resistance together, at whatever
    #: timestep the model runs. Such a parameter can be calibrated against
    #: an observable but never sourced, so a curation report that lists it
    #: beside a protein mass is sending someone to look for a paper that
    #: does not exist.
    model_construct: bool = False
    #: Returns a reason this parameter does not apply to a particular entity,
    #: or None when it does. A reason rather than a bool, because an
    #: exclusion that cannot say why is the same defect as an uncited number
    #: and harder to notice: it leaves nothing behind to review.
    #:
    #: ``applies_to`` is a tuple of EntityType, which is too coarse to carry
    #: this. A plasma cell and a CD8 effector T cell are both
    #: ``EntityType.IMMUNE_CELL``, but only one of them kills, so asking both
    #: for a kill radius put sixteen rows on a curation worklist where the
    #: correct answer was that the question does not arise.
    inapplicable_when: Optional[Callable[[Any], Optional[str]]] = None

    def inapplicable_to(self, entity) -> Optional[str]:
        """Why this parameter does not apply to ``entity``, or None.

        An empty string from the predicate reads as "applies", never as an
        unexplained exclusion.
        """
        if self.inapplicable_when is None:
            return None
        return self.inapplicable_when(entity) or None

    def applies(self, entity) -> bool:
        """True if the parameter is both declared for the type and applicable."""
        return (entity.entity_type in self.applies_to
                and self.inapplicable_to(entity) is None)

    def check(self, pv: ParameterValue) -> List[str]:
        problems: List[str] = []
        if pv.unit and pv.unit != self.unit:
            problems.append(
                f"{self.name} is declared in {self.unit!r} but the value is "
                f"in {pv.unit!r}"
            )
        try:
            val = float(pv.value)
        except (TypeError, ValueError):
            return problems
        if self.minimum is not None and val < self.minimum:
            problems.append(
                f"{self.name}={val} is below the plausible minimum "
                f"{self.minimum} {self.unit}"
            )
        if self.maximum is not None and val > self.maximum:
            problems.append(
                f"{self.name}={val} is above the plausible maximum "
                f"{self.maximum} {self.unit}"
            )
        return problems


def _spec(*args, **kwargs) -> Tuple[str, ParameterSpec]:
    s = ParameterSpec(*args, **kwargs)
    return s.name, s


#: The declared parameter contract.
#:
#: Ranges are deliberately generous. Their job is to catch unit errors
#: and typos, not to encode biology: a kill probability above 1 or a
#: half-life of a century is a mistake regardless of cell type.
# ── Which immune lineages the killing parameters actually describe ────
#
# The kill parameters and the division time are declared for
# EntityType.IMMUNE_CELL, which lumps a CD8 effector T cell with a plasma
# cell, a Treg and three rows of vascular endothelium. Requiring all five
# parameters of all twenty-eight put thirty-eight rows on a curation
# worklist where the honest answer was that the question does not arise.
#
# Both tables are deliberately ASYMMETRIC, and the asymmetry is the whole
# design: only defensible exclusions are listed, and anything absent stays
# applicable. A false inclusion leaves a row on a worklist where somebody
# eventually reads it. A false exclusion deletes the question, and nobody
# looks again. So where the biology was contested, the row is NOT excluded.
#
# The keys are (immune_type, immune_subtype) and not surface_markers. The
# NKT and gamma-delta rows record no perforin, granzyme or FasL yet both
# are cytotoxic, so a marker-derived predicate would silently exclude them.
# ``cytokines_secreting`` is no better: it is being used as "secreted
# mediators" and carries perforin, granzyme_B, NO, histamine and MBP, none
# of which are cytokines, so it is not trustworthy as a lytic signal.

#: Lineages for which a contact kill radius and a per-contact kill
#: probability do not describe anything the cell does.
_NO_CONTACT_CYTOTOXICITY: Dict[Tuple[str, str], str] = {
    ("B_cell", "plasma_cell"):
        "a plasma cell is a dedicated antibody factory with no lytic "
        "granules; its effect on a target is mediated by secreted "
        "immunoglobulin, not by contact",
    ("B_cell", "naive"):
        "a naive B cell presents and secretes; it has no cytolytic "
        "machinery",
    ("B_cell", "memory"):
        "a memory B cell's effector route is differentiation into an "
        "antibody-secreting plasma cell, not cytolysis",
    ("dendritic", "cDC"):
        "a conventional DC is an antigen-presenting cell; its contribution "
        "to killing is priming a cytotoxic lymphocyte elsewhere",
    ("dendritic", "pDC"):
        "a plasmacytoid DC acts through type I interferon. It can acquire "
        "a TRAIL-mediated kill pathway on TLR7/9 stimulation (Imiquimod is "
        "already a seeded entity), so this is a statement about the "
        "baseline row and not a claim that pDCs never kill",
    ("T_cell", "Th2"):
        "a Th2 cell acts through IL-4/IL-5/IL-13; it carries no lytic "
        "granule programme",
    ("T_cell", "Th17"):
        "a Th17 cell acts through IL-17; it carries no lytic granule "
        "programme",
    ("T_cell", "Tfh"):
        "a Tfh cell licenses B cells in the germinal centre; it does not "
        "lyse targets",
    ("T_cell", "naive"):
        "a naive CD8 T cell has no preformed perforin or granzyme and "
        "cannot kill until it is primed and differentiates; the kill "
        "belongs to its effector progeny",
    ("mast_cell", "mast_cell"):
        "a mast cell degranulates histamine and proteases into the "
        "surrounding tissue; it does not form a cytotoxic synapse with a "
        "target cell",
    ("basophil", "basophil"):
        "a basophil degranulates into the surrounding tissue; it forms no "
        "lytic synapse",
    ("ILC", "ILC2"):
        "ILC2s act through IL-5/IL-13; they are non-cytotoxic",
    ("ILC", "ILC3"):
        "ILC3s act through IL-22/IL-17 and are non-cytotoxic. NKp44 "
        "appears in this row's surface_markers and the library elsewhere "
        "calls NKp44 a natural cytotoxicity receptor, but on ILC3 it marks "
        "the IL-22-producing subset and confers no lytic function",
}

#: Lineages that are terminally differentiated: a division time is not a
#: quantity they have.
#:
#: Excluding these six leaves them with no temporal parameter at all -- a
#: gap this table exposes rather than creates, and the defaults are exactly
#: inverted for it. ``_PHYSICS_DEFAULTS["immune_cell"]`` supplies a division
#: time and no clearance term while ``["cell_type"]`` supplies both:
#: measured, 28 immune rows carry a division time and 0 carry an apoptosis
#: rate, against 12 and 12 for cell types. So the cells whose defining
#: temporal property is their lifespan are the ones with no way to say it.
#:
#: The fix is not a uniform apoptosis rate across 28 rows. A circulating
#: neutrophil lives hours and a memory T cell years, so one default would
#: be meaningless in the same way the 48 h division time already was. It
#: needs per-lineage values, and ``apoptosis_rate_per_hour`` already
#: applies to IMMUNE_CELL, so the parameter exists and only the numbers are
#: missing.
_NOT_CYCLING: Dict[Tuple[str, str], str] = {
    ("neutrophil", "neutrophil"):
        "a circulating neutrophil is post-mitotic; it leaves the marrow "
        "fully differentiated and dies without dividing. The governing "
        "quantity is its lifespan, and the seeded 48 h division time is "
        "shorter than the cell's own recorded 5-90 h life",
    ("eosinophil", "eosinophil"):
        "a tissue eosinophil is post-mitotic; it is replaced by marrow "
        "output rather than by local division",
    ("basophil", "basophil"):
        "a circulating basophil is post-mitotic",
    ("B_cell", "plasma_cell"):
        "a plasma cell is terminally differentiated and does not divide. "
        "What varies is lifespan -- days for a plasmablast against years "
        "for a bone-marrow resident -- and the library has no lifespan "
        "parameter for a cell at all",
    ("dendritic", "cDC"):
        "a differentiated cDC does not divide; the population is "
        "maintained by influx of marrow-derived precursors. Short-lived, "
        "not long-lived: there is no recruitment or residence-time "
        "parameter to carry that influx",
    ("dendritic", "pDC"):
        "a differentiated pDC does not divide; its population is "
        "maintained by precursor influx",
}

# ("endothelial", "") was here until the three vessel-wall rows were retyped
# to CELL_TYPE, which is the real fix: they are no longer immune cells, so
# the kill parameters are not declared for them at all and nothing needs to
# exclude them. The key is gone rather than kept, because a table entry that
# can never match is config that looks like a safeguard and is not one.
#
# Deliberately NOT excluded from the kill parameters, each for a stated
# reason, because excluding them would delete a live question:
#   Treg            its own description lists granzyme-B/perforin killing of
#                   effector cells. That is real, but the target is an
#                   effector lymphocyte and the effect SUPPRESSES tumour
#                   clearance, so it needs its own parameter with its own
#                   target class rather than a kill probability that would
#                   invert the sign.
#   ILC1            the library states ILC1s lack perforin, but the ILC1/NK
#                   boundary is contested (intraepithelial CD103+ ILC1
#                   granzyme B, liver ILC1 TRAIL).
#   CD4+ Th1        granzyme-B+ CD4 CTLs arise from Th1-like cells under
#                   chronic antigen. All CD4 killing is MHC class II
#                   restricted and the model has no MHC-II field at all.
#   Macrophage M1   antibody-dependent cellular phagocytosis is contact
#                   killing, though it is near zero without an opsonising
#                   IgG and relief of CD47-SIRPalpha.
#   Macrophage M2   retains ADCP under therapeutic antibody or CD47/PD-1
#                   blockade, so "non-cytotoxic" here would mean "no
#                   contact cytolysis", not "no effect".
#   Neutrophil      trogocytic killing is CD11b/CD18-dependent sustained
#                   contact, not purely extracellular. Baseline killing
#                   without a therapeutic antibody is near zero, and
#                   tumour-associated neutrophils are net pro-tumour, so
#                   the SIGN here is context-dependent.
#   Eosinophil      kills by discharging MBP/EPO/ECP and EETs. Real, but
#                   spatially non-targeted, so a contact radius is the
#                   wrong abstraction rather than an inapplicable one. Do
#                   not read its inclusion as licence to fill in a radius.
#   CD8+ Memory     its progeny kill; a resting CCR7+ central-memory cell
#                   does not. One row cannot hold Tcm, Tem and Trm.
#   NKT, gamma-delta, NK, CD8+ Effector  cytotoxic, uncontested.


#: Entities typed RECEPTOR whose own description says they catalyse a
#: reaction. A turnover number and a Michaelis constant describe what these
#: do; a ligand dissociation constant does not, and both already carry
#: kcat_per_s and km_um. Keyed on name because a receptor row has no
#: lineage field to key on, and listed explicitly so the claim is auditable
#: rather than inferred from a description by pattern match -- a regex over
#: descriptions also catches CD47, CTLA-4, TIGIT and TIM-3, which recruit
#: phosphatases but are not themselves enzymes.
_CATALYTIC_NOT_A_RECEPTOR: Dict[str, str] = {
    "IDO1":
        "indoleamine 2,3-dioxygenase 1 is an enzyme, not a receptor -- its "
        "own description calls it 'a 403-amino acid heme-containing enzyme "
        "that catalyzes the rate-limiting step of tryptophan catabolism'. "
        "It is characterised by kcat_per_s and km_um, both of which it "
        "carries. This row is typed RECEPTOR and the library has no enzyme "
        "type; a Ki against an inhibitor such as epacadostat would be a "
        "different quantity from this spec's ligand Kd",
    "CD73 (NT5E)":
        "ecto-5'-nucleotidase is an enzyme, not a receptor -- its own "
        "description calls it an 'Ecto-5'-nucleotidase ... that converts "
        "extracellular AMP to adenosine'. It is characterised by kcat_per_s "
        "and km_um for AMP, both of which it carries. This row is typed "
        "RECEPTOR and the library has no enzyme type",
}


def _catalytic_not_a_receptor(entity) -> Optional[str]:
    return _CATALYTIC_NOT_A_RECEPTOR.get(getattr(entity, "name", ""))


def alias_sources(entity, canonical: str) -> List[Tuple[str, float, str]]:
    """Every legacy key on ``entity`` that can supply ``canonical``.

    Returns ``(legacy_name, converted_value, partner)`` per source, so a
    caller can see when more than one exists. That is not hypothetical:
    TP53 carries both ``dna_binding_kd_nm`` (0.5) and
    ``tetramerization_kd_nm`` (5.0), and a plain read of
    ``binding_kd_nm`` returns whichever alias is declared first, which is
    an arbitrary answer to a question that has two.
    """
    params = getattr(entity, "physics_params", None) or {}
    out: List[Tuple[str, float, str]] = []
    for alias in PARAMETER_ALIASES:
        if alias.canonical != canonical or alias.attribute:
            continue
        raw = params.get(alias.legacy)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw:
            out.append((alias.legacy, float(raw) * alias.factor, alias.substrate))
    return out


def parameter_partner(entity, canonical: str) -> str:
    """The partner or substrate ``canonical`` was measured against.

    Empty when nothing recorded one, and empty when the entity carries
    more than one source, because in that case there is no single answer
    and returning one of them would hide the ambiguity.
    """
    sources = [p for _, _, p in alias_sources(entity, canonical) if p]
    return sources[0] if len(sources) == 1 else ""


def km_substrate(entity) -> str:
    """The substrate this entity's ``km_um`` was measured against."""
    return parameter_partner(entity, "km_um")


#: What the library's own descriptions say about how long the post-mitotic
#: lineages live. Quoted rather than paraphrased, because these are the only
#: lifespan statements the repository holds and whoever fills lifespan_hours
#: should start from them rather than from recall. Two of the six are
#: covered; the other four say nothing about lifespan at all.
#: lineage -> (hours or None, note). A value is present only where the
#: repository's own record supports one number; where the record describes
#: two populations it is None, because choosing between them is a modelling
#: decision and not a lookup -- the same bar the UniProt sourcing applies to
#: a protein that declares several chains.
_LIFESPAN_EVIDENCE: Dict[Tuple[str, str], Tuple[Optional[float], str]] = {
    ("neutrophil", "neutrophil"): (
        21.2,
        "its own description: 'Neutrophils are short-lived (~5-90 hours in "
        "circulation) and undergo constitutive apoptosis'. 21.2 h is the "
        "geometric centre of that stated 5-90 h range, which is the "
        "defensible single value for a span covering eighteenfold; the "
        "range travels with it in the conditions",
    ),
    ("B_cell", "plasma_cell"): (
        None,
        "its own description gives two populations, not one: short-lived "
        "plasmablasts '(~3-5 days)' and long-lived bone-marrow residents "
        "that 'persist for decades'. Roughly three orders of magnitude "
        "apart, so no single scalar represents this row and none is "
        "written until the entity is split",
    ),
}


def apply_recorded_lifespans(entities) -> Dict[str, str]:
    """Write the lifespans the repository's own descriptions support.

    Returns a name -> outcome map. Entities whose lineage the evidence
    table does not cover are untouched and left on the curation worklist,
    which is where a missing number belongs.
    """
    written: Dict[str, str] = {}
    for entity in entities:
        record = _LIFESPAN_EVIDENCE.get(_lineage(entity))
        if record is None:
            continue
        hours, note = record
        if hours is None:
            written[entity.name] = f"left unwritten: {note}"
            continue
        problems = set_parameter(entity, "lifespan_hours", ParameterValue(
            value=float(hours), unit="h", provenance=Provenance.ESTIMATED,
            source="the entity's own curated description in this library",
            conditions="circulating lifespan; the description states a 5-90 h range",
            rationale=note,
        ), strict=False)
        written[entity.name] = "; ".join(problems) if problems else "written"
    return written


def _cycles(entity) -> Optional[str]:
    """Why a lifespan is not the temporal parameter for this lineage.

    The inverse of :data:`_NOT_CYCLING`. A cell either divides or it does
    not, and whichever is true decides which temporal parameter it owes:
    a cycling lineage is described by its division time, a post-mitotic
    one only by how long it lives. Declaring it this way means the six
    post-mitotic rows are asked for a lifespan and nothing else is.
    """
    if _lineage(entity) in _NOT_CYCLING:
        return None
    return (
        "this lineage divides, so its temporal parameter is "
        "division_time_hours; a lifespan is the parameter for a cell that "
        "cannot divide"
    )


def _lineage(entity) -> Tuple[str, str]:
    return (getattr(entity, "immune_type", "") or "",
            getattr(entity, "immune_subtype", "") or "")


def _not_contact_cytotoxic(entity) -> Optional[str]:
    return _NO_CONTACT_CYTOTOXICITY.get(_lineage(entity))


def _not_cycling(entity) -> Optional[str]:
    return _NOT_CYCLING.get(_lineage(entity))


PARAMETER_SPECS: Dict[str, ParameterSpec] = dict([
    # ── Cytokines: what a diffusible-factor field needs ──
    # ── Half-lives: four quantities, not one ──────────────────────
    #
    # ``half_life_hours`` was declared once, for cytokine, ligand,
    # protein and drug, and described as "Serum or interstitial
    # half-life" -- which names two different things in six words. The
    # stored data named four:
    #
    #   drug     23 values, 22 of them MEASURED against FDA labels. The
    #            pharmacokinetic elimination half-life, read by
    #            treatment_simulator to decay drug exposure.
    #   gene     18 values, all placeholder, and gene is not even in the
    #            applies_to above. reaction_builder reads AR's as the
    #            mRNA degradation rate k_ar_m_deg, so these are mRNA
    #            half-lives and they are load-bearing.
    #   cytokine 30 values, all placeholder, every one of them 4.0 --
    #            one default written thirty times. Circulating serum
    #            half-life of a secreted signal.
    #   protein   4 values, all placeholder. Intracellular turnover,
    #            measured by pulse-chase, nothing to do with serum.
    #
    # These differ by orders of magnitude and come from unrelated
    # experiments. Filling 126 entities under one key would make every
    # value look alike while meaning different things -- which is
    # exactly the Km-alias failure this library already paid to find.
    _spec(
        "elimination_half_life_hours", "h",
        "Pharmacokinetic elimination half-life: how long the body takes "
        "to clear half a dose. A whole-organism property measured in "
        "plasma, and the only one of these four that this library "
        "currently has evidence for -- 22 drugs carry it with an FDA "
        "label citation.",
        (EntityType.DRUG,),
        minimum=0.0, maximum=24.0 * 365,
        required_for_simulation=True,
    ),
    _spec(
        "circulating_half_life_hours", "h",
        "Serum half-life of a secreted signalling molecule: how long it "
        "survives in circulation before clearance. Minutes to hours for "
        "most cytokines, and NOT the same as how long the protein "
        "survives inside the cell that made it.",
        (EntityType.CYTOKINE, EntityType.LIGAND),
        minimum=0.0, maximum=24.0 * 365,
        required_for_simulation=True,
    ),
    _spec(
        "protein_turnover_half_life_hours", "h",
        "Intracellular degradation half-life: how long a protein "
        "survives inside the cell before proteasomal or lysosomal "
        "turnover. Measured by pulse-chase or SILAC, hours to days, and "
        "unrelated to anything measured in plasma.",
        (EntityType.PROTEIN, EntityType.RECEPTOR,
         EntityType.ADHESION_MOLECULE, EntityType.ANTIBODY,
         EntityType.COMPLEMENT_COMPONENT,
         EntityType.PATTERN_RECOGNITION_RECEPTOR,
         EntityType.MHC_MOLECULE),
        minimum=0.0, maximum=24.0 * 365,
    ),
    _spec(
        "mrna_half_life_hours", "h",
        "Half-life of the TRANSCRIPT, not the protein it encodes. This "
        "is what a gene row's half-life has always meant here: "
        "reaction_builder reads AR's and uses it as the mRNA "
        "degradation rate k_ar_m_deg. A transcript and its protein "
        "routinely differ by a factor of ten, so the two cannot share a "
        "name.",
        (EntityType.GENE,),
        minimum=0.0, maximum=24.0 * 365,
    ),
    _spec(
        "diffusion_coeff_um2_s", "um^2/s",
        "Free diffusion coefficient in tissue.",
        (EntityType.CYTOKINE, EntityType.LIGAND, EntityType.METABOLITE),
        minimum=0.0, maximum=1e4,
        required_for_simulation=True,
    ),
    _spec(
        "secretion_rate_molecules_per_cell_per_hour", "molecules/cell/h",
        "Production rate by a secreting cell.",
        (EntityType.CYTOKINE,),
        minimum=0.0,
    ),
    _spec(
        "receptor_kd_nm", "nM",
        "Dissociation constant for the primary receptor.",
        (EntityType.CYTOKINE, EntityType.LIGAND),
        minimum=0.0, maximum=1e9,
    ),

    # ── Receptors: what the membrane model needs ──
    _spec(
        "kd_nm", "nM",
        "Ligand dissociation constant.",
        (EntityType.RECEPTOR, EntityType.ADHESION_MOLECULE),
        minimum=0.0, maximum=1e9,
        required_for_simulation=True,
        inapplicable_when=_catalytic_not_a_receptor,
    ),
    _spec(
        "kon_per_nm_per_s", "1/(nM*s)",
        "Association rate constant.",
        (EntityType.RECEPTOR, EntityType.ADHESION_MOLECULE),
        minimum=0.0,
    ),
    _spec(
        "koff_per_s", "1/s",
        "Dissociation rate constant.",
        (EntityType.RECEPTOR, EntityType.ADHESION_MOLECULE),
        minimum=0.0,
    ),
    _spec(
        "surface_density_per_cell", "molecules/cell",
        "Receptor copies on the cell surface.",
        (EntityType.RECEPTOR,),
        minimum=0.0, maximum=1e9,
    ),

    # ── Adhesion molecules: what the rolling/arrest model needs ──
    #
    # kd_nm, kon_per_nm_per_s and koff_per_s above already apply to this
    # type. These two are the parameters the library actually carried on
    # all fifteen adhesion molecules, as an unchecked type-level default
    # that no spec described -- so no unit check, no range check and no
    # applies_to check ran on either of them.
    _spec(
        "bond_lifetime_seconds", "s",
        "Mean lifetime of a single adhesive bond. For a bond breaking "
        "as a first-order process this is 1/koff, which is the model "
        "the diapedesis solver uses, so the two must agree.",
        (EntityType.ADHESION_MOLECULE,),
        minimum=0.0, maximum=1e4,
    ),
    _spec(
        "binding_force_pN", "pN",
        "Rupture force of a single bond. Defined only together with a "
        "loading rate, and selectin bonds are catch bonds, so a bare "
        "scalar here is not a measurable quantity.",
        (EntityType.ADHESION_MOLECULE,),
        minimum=0.0, maximum=1e4,
    ),
    _spec(
        "internalization_rate_per_min", "1/min",
        "Internalization rate of the bound receptor.",
        (EntityType.RECEPTOR,),
        minimum=0.0,
    ),

    # ── Cell types: what an agent needs to behave ──
    _spec(
        "division_time_hours", "h",
        "Mean time between divisions for a cycling cell.",
        (EntityType.CELL_TYPE, EntityType.IMMUNE_CELL, EntityType.PHYSICAL_CELL),
        minimum=0.0, maximum=24.0 * 365,
        required_for_simulation=True,
        inapplicable_when=_not_cycling,
    ),
    _spec(
        "apoptosis_rate_per_hour", "1/h",
        "Baseline probability of apoptosis per hour.",
        (EntityType.CELL_TYPE, EntityType.IMMUNE_CELL),
        minimum=0.0, maximum=1.0,
    ),
    _spec(
        "radius_um", "um",
        "Mean cell radius, used for volume exclusion and contact tests.",
        (EntityType.CELL_TYPE, EntityType.IMMUNE_CELL, EntityType.PHYSICAL_CELL),
        minimum=0.0, maximum=1e3,
        required_for_simulation=True,
    ),

    # ── Cell ultrastructure, measured ──
    # These carry the morphometry the visualisation needs. They live on
    # the cell type rather than on an organelle, because that is what is
    # measured: a nuclear volume fraction is a fact about a lymphocyte,
    # not about a free-floating nucleus.
    _spec(
        "nuclear_volume_fraction", "dimensionless",
        "Nucleus as a fraction of cell volume. A FRACTION, never a radius "
        "ratio: the radius is its cube root, and setting a nuclear radius "
        "to the fraction gives a nucleus at the cube of it -- 0.444 "
        "becomes 8.8% of the cell, which is how this was wrong before.",
        (EntityType.CELL_TYPE, EntityType.IMMUNE_CELL, EntityType.PHYSICAL_CELL),
        minimum=0.01, maximum=0.90,
    ),
    _spec(
        "granule_volume_fraction", "dimensionless",
        "Cytoplasmic granules as a fraction of cell volume. The strongest "
        "single discriminator between these cells -- a 77-fold measured "
        "spread from eosinophil to lymphocyte in one table "
        "(PMID 6775712) -- so a value here that is uniform across cell "
        "types is a placeholder however plausible each one looks.",
        (EntityType.CELL_TYPE, EntityType.IMMUNE_CELL),
        minimum=0.0, maximum=0.5,
    ),
    _spec(
        "microvillus_density_per_um2", "1/um^2",
        "Microvilli per unit surface area. A DENSITY and not a count, so "
        "it stays right when the cell's diameter does not: the count is "
        "this times the cell's own area, which is 462 on a 7 um "
        "lymphocyte and 603 on an 8 um one.",
        (EntityType.CELL_TYPE, EntityType.IMMUNE_CELL),
        minimum=0.0, maximum=60.0,
    ),
    _spec(
        "microvillus_length_um", "um",
        "MEDIAN microvillar length. The distribution is strongly right-"
        "skewed -- lengths run from 0.1 um to several micrometres -- so "
        "this is a median and a renderer that treats it as a constant has "
        "the wrong shape, not merely the wrong number.",
        (EntityType.CELL_TYPE, EntityType.IMMUNE_CELL),
        minimum=0.02, maximum=10.0,
    ),
    _spec(
        "microvillus_diameter_um", "um",
        "Microvillar shaft diameter. Set by the actin bundle inside, "
        "which does not taper, so the shaft is a cylinder and not a cone.",
        (EntityType.CELL_TYPE, EntityType.IMMUNE_CELL),
        minimum=0.02, maximum=0.5,
    ),
    _spec(
        "lytic_granule_count", "count",
        "Lytic granules per cytotoxic cell. Primary-cell counts only: "
        "the immortalised NK lines carry around three times as many, and "
        "that larger figure is what reviews quote.",
        (EntityType.CELL_TYPE, EntityType.IMMUNE_CELL),
        minimum=0.0, maximum=1000.0,
    ),
    _spec(
        "lytic_granule_diameter_um", "um",
        "Lytic granule diameter.",
        (EntityType.CELL_TYPE, EntityType.IMMUNE_CELL),
        minimum=0.05, maximum=2.0,
    ),

    # ── Subcellular structure ──
    # Organelles are not entities in this library, so their dimensions are
    # held as properties of the cell type that has them -- which is also
    # the more honest shape, since a nuclear volume is a fact about a cell
    # type and not about a free-floating nucleus. These specs exist mainly
    # so that a wrong-unit write is refused: a lipid-bilayer thickness and
    # an organelle diameter are both "a length" to a schema that declares
    # no unit, and they differ by four orders of magnitude.
    _spec(
        "nucleus_volume_um3", "um^3",
        "Interphase nuclear volume. BioNumbers holds no citable NORMAL "
        "human somatic value: every human record that resolves is HeLa, "
        "which is hypertriploid and was measured trypsinized and rounded "
        "up in suspension (BNID 106665, 220; BNID 101402, 374). So this "
        "can carry a cancer nucleus with provenance but cannot support a "
        "normal-versus-malignant comparison on its own, and nuclear "
        "enlargement must not be asserted from it.",
        (EntityType.CELL_TYPE, EntityType.IMMUNE_CELL, EntityType.PHYSICAL_CELL),
        minimum=5.0, maximum=4000.0,
    ),
    _spec(
        "mitochondrion_volume_um3", "um^3",
        "Volume of ONE mitochondrion, not the per-cell mitochondrial "
        "volume -- the two differ by a factor of hundreds and BioNumbers "
        "records do not always say which they mean (BNID 109399, 0.29, "
        "has an empty comments field). In many cell types mitochondria "
        "form a connected reticulum rather than discrete bodies, in which "
        "case a per-organelle volume is not well defined at all.",
        (EntityType.CELL_TYPE, EntityType.IMMUNE_CELL, EntityType.PHYSICAL_CELL),
        minimum=0.001, maximum=50.0,
    ),
    _spec(
        "lysosome_diameter_um", "um",
        "Median lysosomal diameter. A median, not a size: the population "
        "spans 0.1-1.2 um (BNID 106072), a factor of 12 in diameter and "
        "~1700 in volume, so instancing one value uniformly asserts a "
        "monodispersity the measurement contradicts. Note also that the "
        "pH in BNID 106078's title is the culture medium's, not the "
        "lumen's, which is nearer 4.5-5.5.",
        (EntityType.CELL_TYPE, EntityType.IMMUNE_CELL, EntityType.PHYSICAL_CELL),
        minimum=0.02, maximum=3.0,
    ),
    _spec(
        "ribosome_diameter_um", "um",
        "Diameter of a cytoplasmic ribosome, 0.025-0.030 um in human "
        "cells. A 70S record is BACTERIAL and does not apply: human "
        "cytoplasmic ribosomes are 80S and larger, and a search for "
        "'ribosome diameter' returns E. coli 70S first. Beware also of "
        "volume records computed from partial specific volumes (BNID "
        "102474, 4000 nm^3): those are anhydrous compact volumes that "
        "exclude the solvent-filled channels, so converting one to an "
        "equivalent sphere understates the envelope by around a quarter.",
        (EntityType.CELL_TYPE, EntityType.IMMUNE_CELL, EntityType.PHYSICAL_CELL),
        minimum=0.015, maximum=0.05,
    ),
    _spec(
        "er_surface_area_um2", "um^2",
        "Endoplasmic-reticulum membrane area per cell. The ER is a "
        "continuous network, so an area and a tubule diameter are what "
        "get measured and a 'length' is not meaningful. The only citable "
        "human record (BNID 105910) is an aged Leydig cell, a "
        "steroidogenic cell whose smooth ER is hypertrophied to make "
        "testosterone and is its dominant organelle -- near the ceiling "
        "of the human range and not a lymphocyte or a carcinoma cell.",
        (EntityType.CELL_TYPE, EntityType.IMMUNE_CELL, EntityType.PHYSICAL_CELL),
        minimum=50.0, maximum=2e5,
    ),
    _spec(
        "golgi_extent_um", "um",
        "Longest extent of the Golgi apparatus, a stack of 4-8 cisternae. "
        "Do not fill this from a membrane-thickness record: a search for "
        "Golgi dimensions returns BNID 105820, 39.5 angstroms, which is "
        "the phosphate-to-phosphate distance across ONE bilayer and is "
        "correct for what it measures and 760x too small for the "
        "organelle. It is the highest-risk record in this area precisely "
        "because it is citable, human-adjacent and sitting in the field a "
        "consumer would read.",
        (EntityType.CELL_TYPE, EntityType.IMMUNE_CELL, EntityType.PHYSICAL_CELL),
        minimum=0.3, maximum=15.0,
    ),
    _spec(
        "migration_speed_um_per_min", "um/min",
        "Mean migration speed.",
        (EntityType.CELL_TYPE, EntityType.IMMUNE_CELL),
        minimum=0.0, maximum=1e3,
    ),
    _spec(
        "oxygen_consumption_fmol_per_cell_per_hour", "fmol/cell/h",
        "Oxygen uptake, the sink term for the oxygen field.",
        (EntityType.CELL_TYPE, EntityType.IMMUNE_CELL),
        minimum=0.0,
    ),
    _spec(
        "glucose_consumption_fmol_per_cell_per_hour", "fmol/cell/h",
        "Glucose uptake, the sink term for the glucose field.",
        (EntityType.CELL_TYPE, EntityType.IMMUNE_CELL),
        minimum=0.0,
    ),

    # ── Immune cells: what the killing loop needs ──
    _spec(
        "detection_radius_um", "um",
        "Radius within which a target can be detected.",
        (EntityType.IMMUNE_CELL,),
        minimum=0.0, maximum=1e3,
        required_for_simulation=True,
        model_construct=True,
    ),
    _spec(
        "kill_radius_um", "um",
        "Contact distance at which killing can occur.",
        (EntityType.IMMUNE_CELL,),
        minimum=0.0, maximum=1e3,
        required_for_simulation=True,
        inapplicable_when=_not_contact_cytotoxic,
        model_construct=True,
    ),
    _spec(
        "kill_probability", "dimensionless",
        "Per-contact probability of killing a recognized target. For a "
        "lineage that forms no lytic synapse the parameter is reported "
        "not-applicable with a reason rather than curated as zero: a "
        "sentence cannot be mistaken for a measurement, and it lets "
        "set_parameter refuse a later non-zero write.",
        (EntityType.IMMUNE_CELL,),
        minimum=0.0, maximum=1.0,
        required_for_simulation=True,
        inapplicable_when=_not_contact_cytotoxic,
        model_construct=True,
    ),
    # The time a contact takes, which kill_probability does not say: the
    # probability is per contact, and these say how long a contact lasts.
    # Unlike the probability they are measured directly, by time-lapse
    # imaging of effector-target pairs, so they are sourced, not
    # calibrated. The immune module draws each contact's duration from an
    # exponential with this mean.
    _spec(
        "lethal_contact_minutes", "min",
        "Time from an effector's contact with a target to the target's "
        "death, for a contact that kills.",
        (EntityType.IMMUNE_CELL, EntityType.CELL_TYPE),
        minimum=0.0, maximum=24.0 * 60.0,
        inapplicable_when=_not_contact_cytotoxic,
    ),
    _spec(
        "nonlethal_contact_minutes", "min",
        "Duration of an effector-target contact that does not kill, before "
        "the effector detaches and is free to engage again.",
        (EntityType.IMMUNE_CELL, EntityType.CELL_TYPE),
        minimum=0.0, maximum=24.0 * 60.0,
        inapplicable_when=_not_contact_cytotoxic,
    ),
    # ── Binding constants beyond the receptor path ──
    # Eighteen key spellings carry a dissociation constant, most of them
    # naming their partner (vegfr2_kd_pm, sirpa_kd_nm, dna_binding_kd_nm).
    # kd_nm is deliberately not widened to cover them: it is
    # required_for_simulation, so extending its applies_to would put a
    # hundred new rows on the curation worklist for a parameter those types
    # are not simulated through. This is the same quantity under a spec that
    # describes rather than demands.
    _spec(
        "binding_kd_nm", "nM",
        "Dissociation constant for a named binding partner. The partner "
        "belongs in the ParameterValue conditions, not in the parameter "
        "name; the alias table records which partner each legacy spelling "
        "referred to.",
        (EntityType.GENE, EntityType.PROTEIN, EntityType.CYTOKINE,
         EntityType.LIGAND, EntityType.RECEPTOR, EntityType.METABOLITE),
        minimum=0.0, maximum=1e9,
    ),

    # ── Drug response ──
    # Nineteen drug rows carry these five. They are bounded fractions and
    # multipliers whose range is the only check available, since a
    # dimensionless effect has no unit to validate against.
    _spec(
        "exhaustion_reversal", "dimensionless",
        "Fraction of T-cell exhaustion this agent reverses, 0 to 1.",
        (EntityType.DRUG,), minimum=0.0, maximum=1.0,
        model_construct=True,
    ),
    _spec(
        "treg_effect", "dimensionless",
        "Signed effect on regulatory T-cell activity: negative depletes, "
        "positive expands.",
        (EntityType.DRUG,), minimum=-1.0, maximum=1.0,
        model_construct=True,
    ),
    _spec(
        "irae_base_risk", "dimensionless",
        "Baseline probability of an immune-related adverse event, 0 to 1.",
        (EntityType.DRUG,), minimum=0.0, maximum=1.0,
    ),
    _spec(
        "effect_onset_days", "d",
        "Days from first dose until an effect is measurable.",
        (EntityType.DRUG,), minimum=0.0, maximum=3650.0,
    ),
    _spec(
        "effect_ramp_days", "d",
        "Days from onset to full effect.",
        (EntityType.DRUG,), minimum=0.0, maximum=3650.0,
    ),
    _spec(
        "bioavailability", "dimensionless",
        "Fraction of an administered dose reaching systemic circulation.",
        (EntityType.DRUG,), minimum=0.0, maximum=1.0,
    ),
    _spec(
        "volume_of_distribution_L", "L",
        "Apparent volume of distribution for a standard adult. The cap is "
        "generous on purpose: abiraterone's label states 19,669 L, because "
        "an apparent volume counts tissue binding and is not a physical "
        "volume at all.",
        (EntityType.DRUG,), minimum=0.0, maximum=1e5,
    ),

    # ── Mutation and gene-level effects ──
    _spec(
        "mhc1_downregulation_effect", "dimensionless",
        "Fraction by which this lesion lowers MHC class I surface "
        "presentation, 0 to 1.",
        (EntityType.GENE, EntityType.MUTATION), minimum=0.0, maximum=1.0,
        model_construct=True,
    ),
    _spec(
        "tumor_growth_multiplier", "dimensionless",
        "Multiplier on baseline tumour growth rate; 1.0 is no effect.",
        (EntityType.GENE, EntityType.MUTATION), minimum=0.0, maximum=100.0,
        model_construct=True,
    ),

    # ── Molecular properties ──
    _spec(
        "molecular_extent_nm", "nm",
        "Longest axis of the folded molecule. The size a packing or "
        "crowding calculation needs, and the one this library did not "
        "have: scale_um records 0.005 um -- 50 angstrom -- for every "
        "protein in the catalogue, from cytochrome c to a 1972-residue "
        "myosin heavy chain. Measured from a structure, so it is DERIVED "
        "rather than MEASURED, and the confident-atom fraction it was "
        "computed over belongs in the conditions: a disordered tail "
        "stretches a bounding box without making the molecule bigger.",
        (EntityType.PROTEIN, EntityType.RECEPTOR, EntityType.CYTOKINE,
         EntityType.ADHESION_MOLECULE, EntityType.ANTIBODY,
         EntityType.COMPLEMENT_COMPONENT,
         EntityType.PATTERN_RECOGNITION_RECEPTOR,
         EntityType.MHC_MOLECULE),
        minimum=0.5, maximum=500.0,
    ),
    _spec(
        "radius_of_gyration_nm", "nm",
        "Root-mean-square distance of the atoms from the centre of "
        "mass. Preferred over the longest axis wherever a single number "
        "has to stand for a molecule's size, because it is robust to "
        "the disordered tails that dominate a bounding box: perforin "
        "measures 12.8 nm along its longest axis over all atoms and 9.8 "
        "nm over the confident ones, while its radius of gyration is "
        "3.5 nm either way.",
        (EntityType.PROTEIN, EntityType.RECEPTOR, EntityType.CYTOKINE,
         EntityType.ADHESION_MOLECULE, EntityType.ANTIBODY,
         EntityType.COMPLEMENT_COMPONENT,
         EntityType.PATTERN_RECOGNITION_RECEPTOR,
         EntityType.MHC_MOLECULE),
        minimum=0.2, maximum=200.0,
    ),
    _spec(
        "molecular_weight_kda", "kDa",
        "Molecular mass. Declared in kilodaltons because that is the unit "
        "the library already writes; mass_daltons is absent on every entity. "
        "For a secreted protein this is the mature chain, not the translated "
        "precursor -- the two differ twenty-two-fold for EGF -- and the "
        "ParameterValue conditions say which was used.",
        (EntityType.PROTEIN, EntityType.CYTOKINE, EntityType.LIGAND,
         EntityType.RECEPTOR, EntityType.ANTIBODY, EntityType.GENE,
         EntityType.ADHESION_MOLECULE, EntityType.COMPLEMENT_COMPONENT,
         EntityType.PATTERN_RECOGNITION_RECEPTOR),
        minimum=0.0, maximum=1e5,
    ),
    _spec(
        "copies_per_cell", "molecules/cell",
        "TOTAL protein molecules of this species in one cell, counting "
        "every compartment. Not the same quantity as "
        "surface_density_per_cell, which is the surface pool alone, and "
        "the difference is not small: a shotgun proteome includes the ER, "
        "Golgi, endosomal and secretory pools of a receptor along with "
        "whatever is displayed. Writing one into the other put 24 million "
        "CD14 on a macrophage surface -- 43% of the entire membrane "
        "covered by a single protein -- which is how the distinction was "
        "noticed. A receptor's surface fraction is its own measurement "
        "and this library does not have it for anything.",
        (EntityType.RECEPTOR, EntityType.PROTEIN, EntityType.CYTOKINE,
         EntityType.ADHESION_MOLECULE, EntityType.MHC_MOLECULE,
         EntityType.PATTERN_RECOGNITION_RECEPTOR),
        minimum=1.0, maximum=1e9,
    ),
    _spec(
        "constant_region_mass_kda", "kDa",
        "Mass of ONE immunoglobulin heavy-chain constant region. It is not "
        "the mass of an antibody and must never be read as one: an IGH* "
        "accession translates to a single CH region, with no VH, no light "
        "chain, no multimer, no J chain and no secretory component. All "
        "eight isotypes here carried that number as molecular_weight_kda, "
        "graded MEASURED, so IgM (pentameric) read 51.9 kDa against a real "
        "~970 and IgG1 read 43.9 against ~146 -- wrong by 3x to 19x, and "
        "worst for the two entities whose names promise a multimer. "
        "molecular_weight_kda has a floor of zero because a cytokine may "
        "be 8 kDa, so the guard that existed could never refuse a value "
        "for being too SMALL. Naming the quantity is the guard.",
        (EntityType.ANTIBODY,),
        minimum=5.0, maximum=200.0,
    ),
    _spec(
        "sequence_length_aa", "aa",
        "Residues in the mature chain. Sourced from the same UniProt record "
        "as the mass, so the two always describe the same molecule, and "
        "cross-checked against the cached structure's residue count where "
        "one exists.",
        (EntityType.PROTEIN, EntityType.CYTOKINE, EntityType.LIGAND,
         EntityType.RECEPTOR, EntityType.ANTIBODY, EntityType.GENE,
         EntityType.ADHESION_MOLECULE, EntityType.COMPLEMENT_COMPONENT,
         EntityType.PATTERN_RECOGNITION_RECEPTOR),
        minimum=1.0, maximum=4e4,
    ),
    _spec(
        "normal_concentration_mM", "mM",
        "Physiological concentration in the compartment this metabolite "
        "is modelled in.",
        (EntityType.METABOLITE,), minimum=0.0, maximum=1e4,
    ),
    _spec(
        "production_rate_pg_per_cell_per_hour", "pg/cell/h",
        "Mass secreted per cell per hour. Distinct from "
        "secretion_rate_molecules_per_cell_per_hour: converting between "
        "them needs the molecular weight, so they are separate parameters "
        "rather than aliases.",
        (EntityType.CYTOKINE, EntityType.LIGAND),
        minimum=0.0, maximum=1e6,
    ),
    _spec(
        "transcription_rate_mrna_per_hour", "mRNA/h",
        "Transcripts produced per hour per active gene copy.",
        (EntityType.GENE,), minimum=0.0, maximum=1e5,
    ),
    _spec(
        "endocytosis_rate_per_min", "1/min",
        "Rate constant for receptor internalisation.",
        (EntityType.RECEPTOR,), minimum=0.0, maximum=10.0,
    ),

    # ── Viral replication ──
    _spec(
        "replication_cycle_hours", "h",
        "Time from cell entry to release of progeny virions.",
        (EntityType.VIRUS,), minimum=0.0, maximum=1e4,
    ),
    _spec(
        "burst_size", "virions/cell",
        "Progeny virions released per infected cell.",
        (EntityType.VIRUS,), minimum=0.0, maximum=1e6,
    ),

    # ── Enzyme kinetics ──
    # 26 entities carry a turnover number or a Michaelis constant and no
    # spec described either, so both were written with no unit check. The
    # bounds below are physical rather than biological: the diffusion limit
    # for an enzyme-substrate encounter caps kcat near 1e7 /s, and a Km of
    # 1 M is past any cellular concentration. They reject a misplaced
    # decimal or a per-minute value read as per-second without pretending
    # to know the right value for any particular enzyme.
    _spec(
        "kcat_per_s", "1/s",
        "Catalytic turnover number: substrate molecules converted per "
        "active site per second.",
        (EntityType.GENE, EntityType.PROTEIN, EntityType.RECEPTOR),
        minimum=0.0, maximum=1e7,
    ),
    _spec(
        "km_um", "uM",
        "Michaelis constant: substrate concentration at half-maximal "
        "rate. The substrate it was measured against belongs in the "
        "ParameterValue conditions, not in the parameter name.",
        (EntityType.GENE, EntityType.PROTEIN, EntityType.RECEPTOR),
        minimum=0.0, maximum=1e6,
    ),
    _spec(
        "lifespan_hours", "h",
        "How long a post-mitotic cell survives before clearance. Required "
        "of exactly the lineages division_time_hours does not apply to, so "
        "every cell owes one temporal parameter and no cell owes both. "
        "Where a lineage's span is a range or is bimodal -- a neutrophil's "
        "5-90 h, a plasma cell's days against decades -- record the range "
        "in the ParameterValue conditions; see _LIFESPAN_EVIDENCE for what "
        "the library's own descriptions already state.",
        (EntityType.IMMUNE_CELL, EntityType.CELL_TYPE),
        minimum=0.0, maximum=1e6,
        required_for_simulation=True,
        inapplicable_when=_cycles,
    ),
    _spec(
        "exhaustion_rate", "1/h",
        "Rate at which an engaged effector accumulates exhaustion. Declared "
        "so the value carries a unit and a range: it was the only parameter "
        "the immune rows carried that no spec described, and it is one of "
        "the three fields parameter_resolver tie-breaks the cell_type / "
        "immune_cell name collision on.",
        (EntityType.IMMUNE_CELL,),
        minimum=0.0, maximum=1.0,
        model_construct=True,
    ),
    _spec(
        "mhc1_expression", "dimensionless",
        "Relative MHC class I surface expression, 1.0 being typical.",
        (EntityType.IMMUNE_CELL, EntityType.CELL_TYPE),
        minimum=0.0, maximum=1.0,
    ),
])


def specs_for(entity_type: EntityType) -> Dict[str, ParameterSpec]:
    """Every parameter declared for an entity type."""
    return {
        name: spec for name, spec in PARAMETER_SPECS.items()
        if entity_type in spec.applies_to
    }


def required_for_entity(entity) -> Dict[str, ParameterSpec]:
    """Required parameters that actually apply to this particular entity.

    :func:`required_for` answers at the level of the type, which is what
    the curation report needs as a skip guard. This answers at the level
    of the entity, which is what deciding whether one row is ready needs:
    a plasma cell and a CD8 effector T cell share a type but not a kill
    radius.
    """
    return {
        name: spec for name, spec in required_for(entity.entity_type).items()
        if spec.inapplicable_to(entity) is None
    }


def required_for(entity_type: EntityType) -> Dict[str, ParameterSpec]:
    """The parameters an entity type needs before it can drive a run."""
    return {
        name: spec for name, spec in specs_for(entity_type).items()
        if spec.required_for_simulation
    }


# ── Reading and writing on entities ──────────────────────────────────

class ParameterError(ValueError):
    """Raised when a parameter write would record an unsound value."""


def _sidecar(entity: BioEntity) -> Dict[str, Any]:
    store = entity.physics_params.setdefault(PROVENANCE_KEY, {})
    if not isinstance(store, dict):  # defensive: legacy junk under the key
        store = {}
        entity.physics_params[PROVENANCE_KEY] = store
    return store


@dataclass(frozen=True)
class ParameterAlias:
    """A legacy parameter name, and how to read it as a canonical one.

    The spec table and the seeded data disagree on both names and
    units. ``PARAMETER_SPECS`` declares ``diffusion_coeff_um2_s`` in
    um^2/s, while the seeded cytokines carry
    ``diffusion_coefficient_m2_per_s`` in m^2/s. So
    ``get_parameter(cytokine, "diffusion_coeff_um2_s")`` returned None
    while the value sat right there under another name, and
    :func:`assess` reported it *missing* -- which made the curation
    report ask for a number the library already had.

    An alias is declared only where the conversion is exact and the
    quantity is genuinely the same. ``oxygen_consumption_rate`` is
    deliberately *not* aliased to
    ``oxygen_consumption_fmol_per_cell_per_hour``: nothing in the
    codebase states what unit the former is in, so no factor can be
    asserted, and guessing one would silently rescale an oxygen sink.

    Measured since, which settles it: that field holds 0.05 on all twelve
    entities carrying it -- one distinct value, written by
    ``enrichment._PHYSICS_DEFAULTS["cell_type"]`` onto a fibroblast, an
    endothelial cell and a cancer stem cell alike. So it is not a
    measurement whose unit was lost; it is a default that never had one,
    and there is nothing to convert. An alias would manufacture a unit for
    a number that has none, and the twelve values would then read as
    twelve measurements. ``glucose_consumption_rate`` is referenced by
    ``parameter_resolver`` and carried by no entity at all.
    """

    legacy: str
    canonical: str
    #: Multiply the legacy value by this to get the canonical unit.
    factor: float
    #: Why the factor is what it is.
    note: str
    #: For a catalytic constant, the substrate it was measured against.
    #: The legacy Km names bake the substrate into the key
    #: (``km_amp_um``, ``cdk1_km_um``), and aliasing them all onto one
    #: ``km_um`` would throw that away -- after which a Km for AMP could be
    #: applied to a reaction with an entirely different substrate. It is
    #: recorded here so a caller can check before using the value.
    substrate: str = ""
    #: Read ``legacy`` from a top-level attribute of the entity rather
    #: than from ``physics_params``. The seeded cell types carry their
    #: size as the dataclass field ``scale_um``, not as a parameter, so
    #: an aliasing reader that only consults ``physics_params`` reports
    #: ``radius_um`` absent on forty entities that state their size.
    attribute: bool = False


#: Legacy name -> canonical name, with the exact conversion.
PARAMETER_ALIASES: Tuple[ParameterAlias, ...] = (
    ParameterAlias(
        "diffusion_coefficient_m2_per_s", "diffusion_coeff_um2_s", 1e12,
        "1 m^2 = (1e6 um)^2 = 1e12 um^2, so m^2/s * 1e12 = um^2/s",
    ),
    ParameterAlias(
        "ligand_kd_nm", "kd_nm", 1.0,
        "the same quantity in the same unit under a different name",
    ),
    # Dissociation constants, written under sixteen further spellings that
    # each name their binding partner. They alias onto one binding_kd_nm and
    # record the partner, because a Kd is only a Kd for the pair it was
    # measured on -- the same rule the Km aliases follow. Only TP53 carries
    # two of these (its DNA-binding and tetramerization constants), which is
    # exactly why the partner has to travel with the number.
    ParameterAlias(
        "dna_binding_kd_nm", "binding_kd_nm", 1.0,
        "the same quantity in nanomolar, qualified by partner",
        substrate="its DNA response element",
    ),
    ParameterAlias(
        "p53_binding_kd_um", "binding_kd_nm", 1000.0,
        "micromolar to nanomolar: 1 uM = 1000 nM",
        substrate="p53",
    ),
    ParameterAlias(
        "tetramerization_kd_nm", "binding_kd_nm", 1.0,
        "the same quantity in nanomolar, qualified by partner",
        substrate="itself (tetramer assembly)",
    ),
    ParameterAlias(
        "e2f_binding_kd_nm", "binding_kd_nm", 1.0,
        "the same quantity in nanomolar, qualified by partner",
        substrate="E2F",
    ),
    ParameterAlias(
        "substrate_kd_nm", "binding_kd_nm", 1.0,
        "the same quantity in nanomolar, qualified by partner",
        substrate="its substrate",
    ),
    ParameterAlias(
        "bh3_groove_kd_nm", "binding_kd_nm", 1.0,
        "the same quantity in nanomolar, qualified by partner",
        substrate="a BH3 domain",
    ),
    ParameterAlias(
        "vegfr2_kd_pm", "binding_kd_nm", 0.001,
        "picomolar to nanomolar: 1 pM = 0.001 nM",
        substrate="VEGFR2",
    ),
    ParameterAlias(
        "met_kd_nm", "binding_kd_nm", 1.0,
        "the same quantity in nanomolar, qualified by partner",
        substrate="MET",
    ),
    ParameterAlias(
        "egfr_kd_nm", "binding_kd_nm", 1.0,
        "the same quantity in nanomolar, qualified by partner",
        substrate="EGFR",
    ),
    ParameterAlias(
        "bmpr1a_kd_nm", "binding_kd_nm", 1.0,
        "the same quantity in nanomolar, qualified by partner",
        substrate="BMPR1A",
    ),
    ParameterAlias(
        "sirpa_kd_nm", "binding_kd_nm", 1.0,
        "the same quantity in nanomolar, qualified by partner",
        substrate="SIRP-alpha",
    ),
    ParameterAlias(
        "kit_kd_nm", "binding_kd_nm", 1.0,
        "the same quantity in nanomolar, qualified by partner",
        substrate="KIT",
    ),
    ParameterAlias(
        "igf1r_kd_nm", "binding_kd_nm", 1.0,
        "the same quantity in nanomolar, qualified by partner",
        substrate="IGF1R",
    ),
    ParameterAlias(
        "fgfr1_kd_nm", "binding_kd_nm", 1.0,
        "the same quantity in nanomolar, qualified by partner",
        substrate="FGFR1",
    ),
    ParameterAlias(
        "il2rb_kd_nm", "binding_kd_nm", 1.0,
        "the same quantity in nanomolar, qualified by partner",
        substrate="IL2RB",
    ),
    ParameterAlias(
        "pdgfrb_kd_nm", "binding_kd_nm", 1.0,
        "the same quantity in nanomolar, qualified by partner",
        substrate="PDGFRB",
    ),
    # Two more exact conversions, each the same quantity under a spelling
    # that states a different unit.
    ParameterAlias(
        "molecular_weight_da", "molecular_weight_kda", 1e-3,
        "daltons to kilodaltons: 1 Da = 0.001 kDa",
    ),
    ParameterAlias(
        "half_life_min", "half_life_hours", 1.0 / 60.0,
        "minutes to hours: 1 min = 1/60 h",
    ),
    # kcat, written under six names and in two different time units. The
    # per-minute pair is the hazard: `methyltransferase_kcat_per_min` 1.5
    # beside `kinase_kcat_per_s` 15 means a reader assuming seconds is 60x
    # out, in whichever direction it guesses.
    ParameterAlias(
        "kinase_kcat_per_s", "kcat_per_s", 1.0,
        "the same quantity in the same unit, qualified by enzyme class",
    ),
    ParameterAlias(
        "catalytic_kcat_per_s", "kcat_per_s", 1.0,
        "the same quantity in the same unit under a different name",
    ),
    ParameterAlias(
        "protease_kcat_per_s", "kcat_per_s", 1.0,
        "the same quantity in the same unit, qualified by enzyme class",
    ),
    ParameterAlias(
        "ubiquitin_ligase_kcat_per_min", "kcat_per_s", 1.0 / 60.0,
        "per minute to per second: 1/min = (1/60)/s",
    ),
    ParameterAlias(
        "methyltransferase_kcat_per_min", "kcat_per_s", 1.0 / 60.0,
        "per minute to per second: 1/min = (1/60)/s",
    ),
    # Km, written under eleven names that each bake in the substrate. All
    # are micromolar, so the conversion is exact and only the substrate is
    # lost -- which is why the spec says to record it in `conditions`.
    # Two Km values on one entity cannot both be read through this alias;
    # no entity in the library carries two, and a write through
    # set_parameter is the supported way to add one with its substrate.
    ParameterAlias(
        "km_amp_um", "km_um", 1.0, "Km for AMP, in micromolar",
        substrate="AMP",
    ),
    ParameterAlias(
        "km_tryptophan_um", "km_um", 1.0, "Km for tryptophan, in micromolar",
        substrate="L-tryptophan",
    ),
    ParameterAlias(
        "km_devd_um", "km_um", 1.0,
        "Km for the DEVD peptide substrate, in micromolar",
        substrate="DEVD peptide",
    ),
    ParameterAlias(
        # Named so that nobody can reach for it as a protein Km. A
        # fluorogenic tetrapeptide reports on the active site; it is not
        # procaspase-3 or BID, and the two need not agree.
        "km_ietd_um", "km_um", 1.0,
        "Km for the Ac-IETD-AFC peptide substrate, in micromolar",
        substrate="Ac-IETD-AFC peptide",
    ),
    # Each selectin binds several ligands with constants that differ by
    # orders of magnitude: P-selectin binds PSGL-1 at 0.32 uM while
    # E-selectin binds ESL-1 at 62 uM -- 200x weaker -- and E-selectin
    # binds PSGL-1 as well. A bare kd_nm on "E-selectin" does not say
    # which bond it is, and the adhesion cascade is where that matters.
    ParameterAlias(
        "kd_psgl1_nm", "kd_nm", 1.0,
        "Kd for PSGL-1, in nanomolar", substrate="PSGL-1",
    ),
    ParameterAlias(
        "kd_esl1_nm", "kd_nm", 1.0,
        "Kd for ESL-1, in nanomolar", substrate="ESL-1",
    ),
    ParameterAlias(
        "kd_glycam1_nm", "kd_nm", 1.0,
        "Kd for GlyCAM-1, in nanomolar", substrate="GlyCAM-1",
    ),
    ParameterAlias(
        "kd_icam1_nm", "kd_nm", 1.0,
        "Kd for ICAM-1, in nanomolar", substrate="ICAM-1",
    ),
    ParameterAlias(
        "kd_lfa1_nm", "kd_nm", 1.0,
        "Kd for LFA-1, in nanomolar", substrate="LFA-1",
    ),
    ParameterAlias(
        "km_collagen_iv_um", "km_um", 1.0,
        "Km for collagen IV, in micromolar",
        substrate="collagen IV",
    ),
    ParameterAlias(
        "km_gelatin_um", "km_um", 1.0, "Km for gelatin, in micromolar",
        substrate="gelatin",
    ),
    ParameterAlias(
        "km_pip2_um", "km_um", 1.0, "Km for PIP2, in micromolar",
        substrate="PIP2",
    ),
    ParameterAlias(
        "km_pip3_um", "km_um", 1.0, "Km for PIP3, in micromolar",
        substrate="PIP3",
    ),
    ParameterAlias(
        "ctd_km_um", "km_um", 1.0,
        "Km for the RNA polymerase II C-terminal domain, in micromolar",
        substrate="RNA Pol II CTD",
    ),
    ParameterAlias(
        "rb_km_um", "km_um", 1.0, "Km for RB1, in micromolar",
        substrate="RB1",
    ),
    ParameterAlias(
        "cdk1_km_um", "km_um", 1.0, "Km for CDK1, in micromolar",
        substrate="CDK1",
    ),
    ParameterAlias(
        "cdk4_km_um", "km_um", 1.0,
        "Km of CDK4 for this substrate, in micromolar",
        substrate="CDK4",
    ),
    ParameterAlias(
        "scale_um", "radius_um", 0.5,
        "scale_um is the cell's full extent, not its radius: "
        "morphology.build() lays a scale_um-wide box over the grid "
        "(resolution = scale_um / grid) and test_morphology asserts a "
        "scale_um=20 cell measures ~20 um across. The seeded values read "
        "as diameters against textbook sizes -- erythrocyte 7.5, naive "
        "CD8 T cell 7, naive B cell 8, NK cell 12, neutrophil 13, "
        "eosinophil 13, plasma cell 14, M1 macrophage 20 -- so radius is "
        "half of it. Taking scale_um as a radius would double every cell "
        "and eightfold every cell volume",
        attribute=True,
    ),
)

#: canonical name -> the aliases that can supply it.
_ALIASES_BY_CANONICAL: Dict[str, Tuple[ParameterAlias, ...]] = {}
for _alias in PARAMETER_ALIASES:
    _ALIASES_BY_CANONICAL[_alias.canonical] = (
        _ALIASES_BY_CANONICAL.get(_alias.canonical, ()) + (_alias,)
    )


def unspecced_names(entities, *, include_aliased: bool = False) -> Dict[str, int]:
    """Parameter names carried by data but declared by no spec.

    A name with no spec is written with no unit check, no range check
    and no applies_to check, so this is the list of parameters that can
    hold anything at all. It is a worklist, not a failure.

    A legacy name that :data:`PARAMETER_ALIASES` maps onto a specced
    parameter is excluded by default. It is still unvalidated on write,
    but it is readable in a known unit by an exact declared factor, which
    is a materially weaker problem than a name nothing can interpret --
    and counting the two together made declaring an alias look like it
    had achieved nothing. Pass ``include_aliased=True`` for the full set.
    """
    aliased = set() if include_aliased else {
        alias.legacy for alias in PARAMETER_ALIASES
        if alias.canonical in PARAMETER_SPECS
    }
    counts: Dict[str, int] = {}
    for entity in entities:
        for name, _ in iter_parameters(entity):
            if name in PARAMETER_SPECS or name in aliased:
                continue
            counts[name] = counts.get(name, 0) + 1
    return counts


def set_parameter(
    entity: BioEntity,
    name: str,
    pv: ParameterValue,
    *,
    strict: bool = True,
) -> List[str]:
    """Attach a provenanced parameter to an entity.

    Writes the scalar into ``physics_params[name]`` so that every
    existing flat reader keeps working, and the evidence into the
    sidecar under the same name. This is the only supported way to write
    either, so they cannot drift.

    Returns the list of validation problems. With ``strict`` (the
    default) any problem raises instead, because silently storing an
    unsound value is the failure this module exists to prevent.
    """
    spec = PARAMETER_SPECS.get(name)
    if spec is not None and entity.entity_type not in spec.applies_to:
        problems = [
            f"{name} is not declared for {entity.entity_type.value}; it "
            f"applies to {[t.value for t in spec.applies_to]}"
        ]
    elif spec is not None and spec.inapplicable_to(entity) is not None:
        # Without this, set_parameter(plasma_cell, "kill_probability", ...)
        # succeeded and the curation report went on calling the parameter
        # not-applicable -- the stored number and the report disagreeing is
        # exactly the drift the single-writer rule exists to prevent.
        problems = [
            f"{name} does not apply to {entity.name!r}: "
            f"{spec.inapplicable_to(entity)}. If that is wrong, widen the "
            "spec rather than storing a value the curation report will go "
            "on calling not-applicable"
        ]
    else:
        problems = pv.validate(spec)

    if problems and strict:
        raise ParameterError(
            f"cannot set {name!r} on {entity.name!r}: " + "; ".join(problems)
        )

    if not problems:
        entity.physics_params[name] = float(pv.value)
        _sidecar(entity)[name] = pv.to_dict()

    return problems


def get_parameter(entity: BioEntity, name: str) -> Optional[ParameterValue]:
    """The provenanced value for ``name``, or None if not curated.

    A scalar present in ``physics_params`` without a sidecar entry is
    reported as PLACEHOLDER rather than invented provenance: it is a
    number of unknown origin, which is exactly what PLACEHOLDER means.
    """
    record = _sidecar(entity).get(name)
    if record is not None:
        return ParameterValue.from_dict(record)

    raw = entity.physics_params.get(name)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        spec = PARAMETER_SPECS.get(name)
        return ParameterValue(
            value=float(raw),
            unit=spec.unit if spec else "",
            provenance=Provenance.PLACEHOLDER,
            rationale=(
                "present in physics_params with no recorded provenance; "
                "origin unknown"
            ),
        )

    # An alias that carries its OWN evidence. A substrate-named
    # parameter -- km_ietd_um, kd_lfa1_nm -- is not a legacy spelling
    # lying around; it is where the curated record deliberately lives,
    # because the bare name cannot say which substrate or which ligand
    # the constant belongs to. Falling through to the PLACEHOLDER branch
    # below would report a fully cited, graded value as a number of
    # unknown origin, which is how a correction turns into a regression.
    for alias in _ALIASES_BY_CANONICAL.get(name, ()):
        record = _sidecar(entity).get(alias.legacy)
        if record is None:
            continue
        pv = ParameterValue.from_dict(record)
        if alias.factor == 1.0:
            return pv
        spec = PARAMETER_SPECS.get(name)
        return ParameterValue(
            value=pv.value * alias.factor,
            unit=spec.unit if spec else pv.unit,
            provenance=pv.provenance, citation=pv.citation,
            source=pv.source, conditions=pv.conditions,
            rationale=(pv.rationale + f"\nRead through {alias.legacy!r} "
                       f"(x{alias.factor:g}): {alias.note}"),
            calibrated_against=pv.calibrated_against,
            confidence=pv.confidence)

    # Last resort: the value may be present under a legacy name. This
    # stays PLACEHOLDER -- a unit conversion does not create evidence,
    # it only stops the library reporting a number it holds as absent.
    for alias in _ALIASES_BY_CANONICAL.get(name, ()):
        if alias.attribute:
            legacy = getattr(entity, alias.legacy, None)
            # A dataclass size field defaults to 0.0, so an unset one is
            # indistinguishable from a measurement of zero by type alone.
            # Reporting radius_um=0 would hand a solver a dimensionless
            # cell; reporting it absent sends a curator to fill it in.
            if legacy == 0:
                continue
        else:
            legacy = entity.physics_params.get(alias.legacy)
        if isinstance(legacy, (int, float)) and not isinstance(legacy, bool):
            spec = PARAMETER_SPECS.get(name)
            return ParameterValue(
                value=float(legacy) * alias.factor,
                unit=spec.unit if spec else "",
                provenance=Provenance.PLACEHOLDER,
                rationale=(
                    f"read from the legacy parameter {alias.legacy!r} "
                    f"(x{alias.factor:g}: {alias.note}); the underlying "
                    "number still has no recorded provenance"
                ),
            )
    return None


def iter_parameters(entity: BioEntity) -> Iterator[Tuple[str, ParameterValue]]:
    """Every parameter on the entity, provenanced ones first."""
    seen = set()
    for name in _sidecar(entity):
        seen.add(name)
        pv = get_parameter(entity, name)
        if pv is not None:
            yield name, pv
    for name, raw in entity.physics_params.items():
        if name in seen or name == PROVENANCE_KEY or name.startswith("_"):
            continue
        pv = get_parameter(entity, name)
        if pv is not None:
            yield name, pv


# ── Readiness and coverage ───────────────────────────────────────────

@dataclass
class Readiness:
    """Whether one entity is fit to parameterize a simulation."""

    entity_name: str
    entity_type: EntityType
    missing: List[str] = field(default_factory=list)
    placeholder: List[str] = field(default_factory=list)
    curated: List[str] = field(default_factory=list)
    #: Required parameters that do not apply to this entity, mapped to the
    #: reason. A dict rather than a list so that "not applicable" can never
    #: reach a report without the reason that justifies it.
    not_applicable: Dict[str, str] = field(default_factory=dict)

    #: False when the entity's type declares no required parameter at all,
    #: so there is nothing to be ready or unready about. Twelve of the
    #: library's types are like this -- gene, drug, pathway, mutation,
    #: complement, mhc, prr, virus, bacterium, antibody, tissue_type,
    #: organ -- covering 148 of 294 entities. They cannot fail the check,
    #: and counting them as ready reported the library as 157/294
    #: simulation-ready when nine entities had actually been assessed.
    assessable: bool = True

    @property
    def is_ready(self) -> bool:
        """No *applicable* required parameter is missing or a placeholder.

        An entity nothing is required of is not "ready"; it is outside the
        question, which :attr:`assessable` records separately.

        Where a type does require parameters and every one of them is
        inapplicable to this entity, "ready" would be vacuous, and the
        likely cause is that the entity is typed wrongly. That is not a
        live case in the library today -- no entity has an entirely
        inapplicable required set -- so this is a guard, not a fix.
        """
        if not self.assessable:
            return False
        if not (self.curated or self.missing or self.placeholder):
            return False
        return not self.missing and not self.placeholder

    def describe(self) -> str:
        if self.is_ready:
            return f"{self.entity_name}: ready ({len(self.curated)} curated)"
        bits = []
        if self.missing:
            bits.append(f"missing {sorted(self.missing)}")
        if self.placeholder:
            bits.append(f"placeholder-only {sorted(self.placeholder)}")
        if self.not_applicable:
            bits.append(
                f"not applicable {sorted(self.not_applicable)}")
        if not self.assessable:
            return (
                f"{self.entity_name}: no simulation parameter is required "
                f"of a {self.entity_type.value}, so readiness does not apply")
        if not bits:
            return (
                f"{self.entity_name}: every required parameter of "
                f"{self.entity_type.value} is inapplicable to it -- the "
                "entity is probably typed wrongly")
        return f"{self.entity_name}: " + ", ".join(bits)


def assess(entity: BioEntity) -> Readiness:
    """Check one entity against the required parameters for its type."""
    required = required_for(entity.entity_type)
    result = Readiness(entity_name=entity.name, entity_type=entity.entity_type,
                       assessable=bool(required))
    for name, spec in required.items():
        why = spec.inapplicable_to(entity)
        if why is not None:
            result.not_applicable[name] = why
            continue
        pv = get_parameter(entity, name)
        if pv is None:
            result.missing.append(name)
        elif not pv.is_simulation_grade:
            result.placeholder.append(name)
        else:
            result.curated.append(name)
    return result


@dataclass
class CoverageReport:
    """How much of a collection is fit to drive a simulation."""

    total: int = 0
    ready: int = 0
    #: Entities whose type requires no simulation parameter, so they are
    #: neither ready nor unready. Reported separately because folding them
    #: into ``ready`` is what made the library look half-curated.
    not_assessable: int = 0
    #: Entities where the parameter is absent altogether.
    by_parameter_missing: Dict[str, int] = field(default_factory=dict)
    #: Entities where a number is present but carries no evidence. A
    #: report that calls these "missing" sends a curator looking for a
    #: value that is already there; what they are missing is the
    #: citation, and the two need different work.
    by_parameter_placeholder: Dict[str, int] = field(default_factory=dict)
    #: Entities the parameter does not apply to. Deliberately NOT part of
    #: :attr:`by_parameter_uncurated`: a not-applicable parameter is not a
    #: gap, and folding the two together is what produced reports reading
    #: "28 entities need kill_radius_um" when sixteen of them do not kill.
    by_parameter_not_applicable: Dict[str, int] = field(default_factory=dict)
    not_ready: List[Readiness] = field(default_factory=list)

    @property
    def assessable(self) -> int:
        """Entities the readiness question actually applies to."""
        return self.total - self.not_assessable

    @property
    def fraction_ready(self) -> float:
        """Ready as a fraction of the entities that can be assessed.

        Over ``total`` this read 0.53 for a library in which nine entities
        were curated, because 148 entities of types with no required
        parameters cannot fail.
        """
        return (self.ready / self.assessable) if self.assessable else 0.0

    @property
    def by_parameter_uncurated(self) -> Dict[str, int]:
        """Absent plus unevidenced, per parameter: the worklist size."""
        out = dict(self.by_parameter_missing)
        for name, count in self.by_parameter_placeholder.items():
            out[name] = out.get(name, 0) + count
        return out

    def describe(self) -> str:
        lines = [
            f"{self.ready}/{self.total} entities ready "
            f"({self.fraction_ready:.0%})"
        ]
        uncurated = self.by_parameter_uncurated
        if uncurated:
            lines.append("uncurated parameters, most common first:")
            for name, count in sorted(
                uncurated.items(), key=lambda kv: (-kv[1], kv[0]),
            ):
                absent = self.by_parameter_missing.get(name, 0)
                unevidenced = self.by_parameter_placeholder.get(name, 0)
                lines.append(
                    f"  {name}: {count} "
                    f"({absent} absent, {unevidenced} unevidenced)"
                )
        return "\n".join(lines)


def coverage(entities) -> CoverageReport:
    """Summarize curation coverage over a collection of entities.

    This is what turns "the numbers are missing" from an anecdote into a
    tracked quantity: it names which parameters are missing and on how
    many entities, so curation can be prioritized and progress measured.
    """
    report = CoverageReport()
    for entity in entities:
        report.total += 1
        r = assess(entity)
        # Tallied for every entity, not only the not-ready ones, or the
        # per-parameter counts stop summing to the entity total.
        for name in r.not_applicable:
            report.by_parameter_not_applicable[name] = (
                report.by_parameter_not_applicable.get(name, 0) + 1
            )
        if not r.assessable:
            report.not_assessable += 1
        elif r.is_ready:
            report.ready += 1
        else:
            report.not_ready.append(r)
            for name in r.missing:
                report.by_parameter_missing[name] = (
                    report.by_parameter_missing.get(name, 0) + 1
                )
            for name in r.placeholder:
                report.by_parameter_placeholder[name] = (
                    report.by_parameter_placeholder.get(name, 0) + 1
                )
    return report


# ── Ingest from measured-data sources ────────────────────────────────

def from_kinetic_parameter(kp, *, source: str = "") -> ParameterValue:
    """Adapt a :mod:`kinetics_client` measurement into a ParameterValue.

    That client already returns everything MEASURED requires: a value, a
    unit, a database, a PMID, and the experimental conditions. It was
    only ever wired to a dashboard button, so those measurements never
    reached the library in a form anything could trust. This is the
    adapter that lets a batch ingest write them with their evidence
    intact.
    """
    conditions = getattr(kp, "conditions", "") or ""
    if not conditions:
        bits = []
        for attr, label in (
            ("organism", ""), ("tissue", ""),
            ("ph", "pH "), ("temperature_c", ""),
        ):
            val = getattr(kp, attr, None)
            if val:
                suffix = "C" if attr == "temperature_c" else ""
                bits.append(f"{label}{val}{suffix}")
        conditions = ", ".join(bits)

    pmid = getattr(kp, "pubmed_id", "") or getattr(kp, "pmid", "") or ""
    citation = f"PMID:{pmid}" if pmid and not str(pmid).startswith("PMID") else str(pmid)

    return ParameterValue(
        value=float(kp.value),
        unit=getattr(kp, "unit", "") or "",
        provenance=(
            Provenance.MEASURED if citation else Provenance.ESTIMATED
        ),
        citation=citation,
        source=source or getattr(kp, "source_db", "") or "",
        conditions=conditions,
        rationale=(
            "" if citation else
            "database record carried no publication identifier"
        ),
    )


def absorb_legacy_attribution(entity: BioEntity) -> int:
    """Migrate the old whole-entity ``_source`` / ``_pmid`` convention.

    ``kinetics_client`` stamped attribution as underscore-prefixed keys
    in the same flat dict, which describes the entity rather than any
    particular parameter. An entity whose Km came from one paper and
    whose half-life came from another could only record one of them.

    Every bare scalar without its own record is upgraded to MEASURED
    under that shared citation. Returns the number of parameters
    migrated.
    """
    legacy_source = entity.physics_params.get("_source", "")
    legacy_pmid = entity.physics_params.get("_pmid", "")
    legacy_conditions = entity.physics_params.get("_conditions", "")
    if not (legacy_source or legacy_pmid):
        return 0

    citation = ""
    if legacy_pmid:
        citation = (
            str(legacy_pmid) if str(legacy_pmid).startswith("PMID")
            else f"PMID:{legacy_pmid}"
        )

    sidecar = _sidecar(entity)
    migrated = 0
    for name, raw in list(entity.physics_params.items()):
        if name == PROVENANCE_KEY or name.startswith("_"):
            continue
        if name in sidecar:
            continue
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            continue
        spec = PARAMETER_SPECS.get(name)
        sidecar[name] = ParameterValue(
            value=float(raw),
            unit=spec.unit if spec else "",
            provenance=(
                Provenance.MEASURED if citation else Provenance.ESTIMATED
            ),
            citation=citation,
            source=str(legacy_source),
            conditions=str(legacy_conditions),
            rationale=(
                "" if citation else
                "migrated from whole-entity attribution with no PMID"
            ),
        ).to_dict()
        migrated += 1
    return migrated
