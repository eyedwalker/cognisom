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
    DERIVED = "derived"
    MEASURED = "measured"


# Ascending quality. Used for comparisons and for picking the better of
# two competing values for the same parameter.
_PROVENANCE_RANK: Dict[Provenance, int] = {
    Provenance.PLACEHOLDER: 0,
    Provenance.ESTIMATED: 1,
    Provenance.DERIVED: 2,
    Provenance.MEASURED: 3,
}

#: Provenance levels a simulation may legitimately consume.
SIMULATION_GRADE = frozenset(
    {Provenance.ESTIMATED, Provenance.DERIVED, Provenance.MEASURED}
)

#: The stricter bar: values defensible in a paper or a patent filing.
PUBLICATION_GRADE = frozenset({Provenance.DERIVED, Provenance.MEASURED})


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
        for key in ("citation", "source", "conditions", "rationale"):
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
            confidence=d.get("confidence"),
        )

    def describe(self) -> str:
        """One-line human summary, for reports and dashboards."""
        evidence = self.citation or self.source or self.rationale or "no evidence"
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
    ("endothelial", ""):
        "vascular endothelium does not kill tumour cells. This row is "
        "typed IMMUNE_CELL but its own immune_type says endothelial; it "
        "belongs to the vessel wall, and its role here is leukocyte "
        "adhesion and transmigration",
}

#: Lineages that are terminally differentiated: a division time is not a
#: quantity they have.
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


def _lineage(entity) -> Tuple[str, str]:
    return (getattr(entity, "immune_type", "") or "",
            getattr(entity, "immune_subtype", "") or "")


def _not_contact_cytotoxic(entity) -> Optional[str]:
    return _NO_CONTACT_CYTOTOXICITY.get(_lineage(entity))


def _not_cycling(entity) -> Optional[str]:
    return _NOT_CYCLING.get(_lineage(entity))


PARAMETER_SPECS: Dict[str, ParameterSpec] = dict([
    # ── Cytokines: what a diffusible-factor field needs ──
    _spec(
        "half_life_hours", "h",
        "Serum or interstitial half-life.",
        (EntityType.CYTOKINE, EntityType.LIGAND, EntityType.PROTEIN),
        minimum=0.0, maximum=24.0 * 365,
        required_for_simulation=True,
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
    ),
    _spec(
        "kill_radius_um", "um",
        "Contact distance at which killing can occur.",
        (EntityType.IMMUNE_CELL,),
        minimum=0.0, maximum=1e3,
        required_for_simulation=True,
        inapplicable_when=_not_contact_cytotoxic,
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
        "exhaustion_rate", "1/h",
        "Rate at which an engaged effector accumulates exhaustion. Declared "
        "so the value carries a unit and a range: it was the only parameter "
        "the immune rows carried that no spec described, and it is one of "
        "the three fields parameter_resolver tie-breaks the cell_type / "
        "immune_cell name collision on.",
        (EntityType.IMMUNE_CELL,),
        minimum=0.0, maximum=1.0,
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
    """

    legacy: str
    canonical: str
    #: Multiply the legacy value by this to get the canonical unit.
    factor: float
    #: Why the factor is what it is.
    note: str
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
    ),
    ParameterAlias(
        "km_tryptophan_um", "km_um", 1.0, "Km for tryptophan, in micromolar",
    ),
    ParameterAlias(
        "km_devd_um", "km_um", 1.0,
        "Km for the DEVD peptide substrate, in micromolar",
    ),
    ParameterAlias(
        "km_collagen_iv_um", "km_um", 1.0,
        "Km for collagen IV, in micromolar",
    ),
    ParameterAlias(
        "km_gelatin_um", "km_um", 1.0, "Km for gelatin, in micromolar",
    ),
    ParameterAlias(
        "km_pip2_um", "km_um", 1.0, "Km for PIP2, in micromolar",
    ),
    ParameterAlias(
        "km_pip3_um", "km_um", 1.0, "Km for PIP3, in micromolar",
    ),
    ParameterAlias(
        "ctd_km_um", "km_um", 1.0,
        "Km for the RNA polymerase II C-terminal domain, in micromolar",
    ),
    ParameterAlias(
        "rb_km_um", "km_um", 1.0, "Km for RB1, in micromolar",
    ),
    ParameterAlias(
        "cdk1_km_um", "km_um", 1.0, "Km for CDK1, in micromolar",
    ),
    ParameterAlias(
        "cdk4_km_um", "km_um", 1.0,
        "Km of CDK4 for this substrate, in micromolar",
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
