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
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple

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
    ),
    _spec(
        "kill_probability", "dimensionless",
        "Per-contact probability of killing a recognized target. Zero is "
        "meaningful and correct for non-cytotoxic types.",
        (EntityType.IMMUNE_CELL,),
        minimum=0.0, maximum=1.0,
        required_for_simulation=True,
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

    @property
    def is_ready(self) -> bool:
        """No required parameter is missing or merely a placeholder."""
        return not self.missing and not self.placeholder

    def describe(self) -> str:
        if self.is_ready:
            return f"{self.entity_name}: ready ({len(self.curated)} curated)"
        bits = []
        if self.missing:
            bits.append(f"missing {sorted(self.missing)}")
        if self.placeholder:
            bits.append(f"placeholder-only {sorted(self.placeholder)}")
        return f"{self.entity_name}: " + ", ".join(bits)


def assess(entity: BioEntity) -> Readiness:
    """Check one entity against the required parameters for its type."""
    result = Readiness(entity_name=entity.name, entity_type=entity.entity_type)
    for name in required_for(entity.entity_type):
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
    by_parameter_missing: Dict[str, int] = field(default_factory=dict)
    not_ready: List[Readiness] = field(default_factory=list)

    @property
    def fraction_ready(self) -> float:
        return (self.ready / self.total) if self.total else 0.0

    def describe(self) -> str:
        lines = [
            f"{self.ready}/{self.total} entities ready "
            f"({self.fraction_ready:.0%})"
        ]
        if self.by_parameter_missing:
            lines.append("uncurated parameters, most common first:")
            for name, count in sorted(
                self.by_parameter_missing.items(),
                key=lambda kv: (-kv[1], kv[0]),
            ):
                lines.append(f"  {name}: {count}")
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
        if r.is_ready:
            report.ready += 1
        else:
            report.not_ready.append(r)
            for name in r.missing + r.placeholder:
                report.by_parameter_missing[name] = (
                    report.by_parameter_missing.get(name, 0) + 1
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
