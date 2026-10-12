"""
Measured binding kinetics for the leukocyte-adhesion molecules.

Why this file exists
--------------------

The library shipped 15 adhesion molecules carrying a Kd, a kon and a
koff each. The numbers were family-level round figures: all five
selectin-pathway entities carried Kd = 100 uM, LFA-1 and ICAM-1 both
carried 1.0 uM, MAC-1 / ICAM-2 / CD99 all carried 5.0 uM. Seven
distinct Kd values covered fifteen molecules, and none of them cited
anything.

Some of those round numbers happened to be close to the literature.
Others were wrong by two orders of magnitude: P-selectin binds PSGL-1
with a Kd of 320 nM (Mehta, Cummings & McEver 1998, PMID 9829984), not
100 uM. Nothing in the library could tell those two cases apart,
because a bare float records no evidence. That is the defect
:mod:`cognisom.library.parameters` exists to fix, and this module is
the data that makes the fix mean something for this family.

What is in here
---------------

One :class:`BondKinetics` record per literature measurement, holding
the constants *in the units the paper reports them in*, together with
the PMID, the method and the experimental conditions. The conversion to
the library's own units happens in exactly one place
(:meth:`BondKinetics.parameters`), so a unit error has one place to
hide and :func:`check_internal_consistency` checks for it.

Every citation in this file was verified against NCBI E-utilities
(``esummary`` for the journal/volume/page, ``efetch`` for the abstract
the constants were transcribed from) rather than recalled. A molecule
with no sourceable measurement is listed in :data:`UNSOURCED` with the
reason, and its parameters stay PLACEHOLDER. That is the correct
outcome for a value nobody has measured, and an honest gap list is
worth more than a full table with invented entries.

Provenance assignment
---------------------

Kd and koff are MEASURED: the cited paper reports them as measured
quantities. Every kon is DERIVED, including the ones the papers print,
because each of those papers describes its kon as *calculated* from the
measured Kd and koff rather than fitted directly -- "the calculated kon
was 4.4 x 10(6) M-1 s-1" (PMID 9829984). DERIVED is publication-grade
and carries the derivation in its rationale, which is exactly what that
situation is.

``bond_lifetime_seconds`` is DERIVED as 1/koff. For a bond that breaks
as a first-order process -- which is the model the diapedesis solver
itself uses, ``p_unbind = koff * dt`` -- the lifetime is exponentially
distributed with mean 1/koff. The shipped value of 2.0 s for all
fifteen molecules is not consistent with any of the measured off-rates:
the real figures run from 0.10 s (E-selectin) to 10 s (LFA-1), a
hundredfold spread that is the whole difference between rolling and
firm arrest.

What this does not fix
----------------------

An integrin has no single Kd. LFA-1's affinity for ICAM-1 spans about
four orders of magnitude between the bent and the extended-open
conformation: locking the alphaL I domain open raises it 9,000-fold
(PMID 11353828). The value recorded here is for soluble monomeric
LFA-1, which is the resting ectodomain, and ``conditions`` says so. A
simulation that models integrin activation needs a conformer-resolved
parameter, which this schema has no field for; using one scalar for
both states would be wrong by that same 9,000-fold.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .models import BioEntity, EntityType
from .parameters import ParameterValue, Provenance, set_parameter

log = logging.getLogger(__name__)

#: Molar -> nanomolar.
_M_TO_NM = 1e9
#: 1/(M*s) -> 1/(nM*s). An association constant has inverse
#: concentration units, so it scales by the reciprocal factor.
_PER_M_TO_PER_NM = 1e-9


def _clean(x: float) -> float:
    """Drop binary-representation noise from a unit conversion.

    ``39.60e-9 * 1e9`` is ``39.599999999999994`` in binary floating
    point. Keeping twelve significant figures removes the artefact
    without altering any value: no constant in this file is quoted to
    more than three.
    """
    return float(f"{x:.12g}")


@dataclass(frozen=True)
class BondKinetics:
    """One published kinetic characterization of one adhesive bond.

    The constants are stored as the paper prints them -- Kd in molar,
    kon in 1/(M*s) -- so that a reader can check this file against the
    abstract without doing arithmetic. Conversion to the library's
    units is :meth:`parameters`' job alone.
    """

    #: Entity name in the library, exactly as seeded.
    molecule: str
    #: The binding partner these constants describe. A Kd is a property
    #: of a pair, not of a molecule, so recording only the molecule
    #: would lose half the measurement.
    partner: str
    citation: str
    source: str
    method: str
    organism: str
    #: Construct and buffer detail that makes the value reproducible.
    detail: str = ""
    kd_molar: Optional[float] = None
    koff_per_s: Optional[float] = None
    kon_per_molar_per_s: Optional[float] = None
    #: True when the paper gives the rate constants as lower bounds
    #: (">= 10 s-1") because dissociation was too fast to resolve.
    rates_are_lower_bounds: bool = False
    #: True when the paper qualifies the Kd ("about 0.8 x 10-4 M").
    kd_is_approximate: bool = False
    #: Extra caveat appended to every condition string for this bond.
    caveat: str = ""

    # ── Conditions ───────────────────────────────────────────────────

    def conditions(self) -> str:
        bits = [f"{self.molecule} / {self.partner}", self.method, self.organism]
        if self.detail:
            bits.append(self.detail)
        if self.caveat:
            bits.append(self.caveat)
        return "; ".join(b for b in bits if b)

    # ── The library's units ──────────────────────────────────────────

    @property
    def kd_nm(self) -> Optional[float]:
        if self.kd_molar is None:
            return None
        return _clean(self.kd_molar * _M_TO_NM)

    @property
    def kon_published(self) -> bool:
        """True when the cited paper prints a kon of its own."""
        return self.kon_per_molar_per_s is not None

    @property
    def kon_per_nm_per_s(self) -> Optional[float]:
        """kon in the library's units.

        When the paper prints one, that figure converted. When it
        prints only Kd and koff -- as PMID:9780171 does -- kon follows
        from the definition Kd = koff/kon, which is a derivation from
        two measured values rather than a new number.
        """
        if self.kon_per_molar_per_s is not None:
            return _clean(self.kon_per_molar_per_s * _PER_M_TO_PER_NM)
        if self.koff_per_s is None or not self.kd_nm:
            return None
        return _clean(self.koff_per_s / self.kd_nm)

    @property
    def bond_lifetime_seconds(self) -> Optional[float]:
        if not self.koff_per_s:
            return None
        return _clean(1.0 / self.koff_per_s)

    # ── Expansion into provenanced parameters ────────────────────────

    def parameters(self) -> Dict[str, ParameterValue]:
        """The provenanced parameters this measurement supports."""
        out: Dict[str, ParameterValue] = {}
        conditions = self.conditions()

        if self.kd_nm is not None:
            out["kd_nm"] = ParameterValue(
                value=self.kd_nm, unit="nM",
                provenance=Provenance.MEASURED,
                citation=self.citation, source=self.source,
                conditions=conditions,
                rationale=(
                    "reported as approximate in the cited paper"
                    if self.kd_is_approximate else ""
                ),
            )

        if self.koff_per_s is not None:
            # A published *bound* is not a measurement of the bound.
            # Nicholson et al. report koff >= 10 1/s because dissociation
            # was faster than SPR could resolve; recording 10.0 as
            # MEASURED would claim the paper measured 10.0, which it did
            # not. Provenance.ESTIMATED exists for exactly this -- a
            # defensible number that is not a direct readout -- and it
            # obliges a written rationale, which the bound supplies.
            # Both levels stay out of PUBLICATION_GRADE honestly: an
            # ESTIMATED value is simulation-grade but not citable as a
            # measured constant.
            bounded = self.rates_are_lower_bounds
            out["koff_per_s"] = ParameterValue(
                value=self.koff_per_s, unit="1/s",
                provenance=(
                    Provenance.ESTIMATED if bounded else Provenance.MEASURED
                ),
                citation=self.citation, source=self.source,
                conditions=conditions,
                rationale=(
                    "the cited paper gives this only as a lower bound "
                    f"(koff >= {self.koff_per_s:g} 1/s); dissociation was "
                    "faster than the method could resolve, so the stored "
                    "number is the bound, not a fitted value"
                    if bounded else ""
                ),
            )

        if self.kon_per_nm_per_s is not None:
            published = self.kon_published
            out["kon_per_nm_per_s"] = ParameterValue(
                value=self.kon_per_nm_per_s, unit="1/(nM*s)",
                provenance=(
                    Provenance.ESTIMATED if self.rates_are_lower_bounds
                    else Provenance.DERIVED
                ),
                citation=self.citation, source=self.source,
                conditions=conditions,
                rationale=(
                    f"kon = koff/Kd = {self.koff_per_s}/{self.kd_nm:.4g} nM. "
                    "The cited paper reports this as a calculated rather "
                    "than a directly fitted constant"
                    if published else
                    f"kon = koff/Kd = {self.koff_per_s}/{self.kd_nm:.4g} nM; "
                    "the cited paper reports only Kd and koff"
                ) + (
                    "; a lower bound, following koff"
                    if self.rates_are_lower_bounds else ""
                ),
            )

        return out


# ── The curated table ────────────────────────────────────────────────
#
# Each record was transcribed from the abstract fetched from NCBI
# E-utilities, and each PMID was confirmed to resolve to the journal,
# volume and pages given here. Nothing in this table was recalled.

_SELECTIN_NOTE = (
    "monomeric selectin ectodomain; the physiological bond is "
    "multivalent and force-dependent, so this affinity is not a bond "
    "strength under shear"
)

MEASUREMENTS: Tuple[BondKinetics, ...] = (
    # Mehta P, Cummings RD, McEver RP. J Biol Chem 1998 Dec 4;273(49):
    # 32506-13. "Soluble P-selectin bound saturably to a single class of
    # sites on PSGL-1 with a dissociation constant (Kd) of 320 +/- 20
    # nM. The measured koff was 1.4 +/- 0.1 s-1, and the calculated kon
    # was 4.4 x 10(6) M-1 s-1."
    BondKinetics(
        molecule="P-selectin (CD62P)", partner="PSGL-1",
        citation="PMID:9829984",
        source="Mehta, Cummings & McEver, J Biol Chem 1998;273:32506-13",
        method="surface plasmon resonance", organism="human",
        detail=(
            "soluble monomeric P-selectin over PSGL-1 purified from "
            "human neutrophils; Ca2+-dependent, abolished by sialidase"
        ),
        kd_molar=320e-9, koff_per_s=1.4, kon_per_molar_per_s=4.4e6,
        caveat=_SELECTIN_NOTE,
    ),
    # Same bond, recorded on the ligand so that PSGL-1 carries its own
    # evidence rather than pointing at a sibling row.
    BondKinetics(
        molecule="PSGL-1 (CD162)", partner="P-selectin",
        citation="PMID:9829984",
        source="Mehta, Cummings & McEver, J Biol Chem 1998;273:32506-13",
        method="surface plasmon resonance", organism="human",
        detail=(
            "soluble monomeric P-selectin over PSGL-1 purified from "
            "human neutrophils; PSGL-1 also binds E- and L-selectin, "
            "whose constants differ"
        ),
        kd_molar=320e-9, koff_per_s=1.4, kon_per_molar_per_s=4.4e6,
        caveat=_SELECTIN_NOTE,
    ),
    # Wild MK, Huang MC, Schulze-Horsel U, van der Merwe PA, Vestweber
    # D. J Biol Chem 2001 Aug 24;276(34):31602-12. "E-selectin bound to
    # ESL-1 with a fast dissociation rate constant of 4.6 s(-1) and a
    # calculated association rate constant of 7.4 x 10(4) m(-1) s(-1).
    # We determined a dissociation constant (K(d)) of 62 microm."
    BondKinetics(
        molecule="E-selectin (CD62E)", partner="ESL-1",
        citation="PMID:11404363",
        source="Wild et al., J Biol Chem 2001;276:31602-12",
        method="surface plasmon resonance", organism="mouse ligand",
        detail=(
            "monomeric E-selectin over ESL-1 from mouse bone marrow "
            "cells; affinity flat from 5 to 37 C; recombinant ESL-1 "
            "gave a similar Kd of 66 uM"
        ),
        kd_molar=62e-6, koff_per_s=4.6, kon_per_molar_per_s=7.4e4,
        caveat=_SELECTIN_NOTE,
    ),
    # Nicholson MW, Barclay AN, Singer MS, Rosen SD, van der Merwe PA.
    # J Biol Chem 1998 Jan 9;273(2):763-70. "a soluble monomeric form of
    # CD62L binds to purified immobilized GlyCAM-1 with a dissociation
    # constant (Kd) of 108 microM. CD62L dissociates from GlyCAM-1 with
    # a very fast dissociation rate constant (>/=10 s-1) ... The
    # calculated association rate constant is >/=10(5) M-1 s-1."
    BondKinetics(
        molecule="L-selectin (CD62L)", partner="GlyCAM-1",
        citation="PMID:9422729",
        source="Nicholson et al., J Biol Chem 1998;273:763-70",
        method="surface plasmon resonance", organism="human CD62L",
        detail=(
            "soluble monomeric CD62L over immobilized GlyCAM-1; the "
            "off-rate agrees with CD62L-mediated leukocyte tethers"
        ),
        kd_molar=108e-6, koff_per_s=10.0, kon_per_molar_per_s=1e5,
        rates_are_lower_bounds=True,
        caveat=_SELECTIN_NOTE,
    ),
    # Tominaga Y et al. J Immunol 1998 Oct 15;161(8):4016-22. "sLFA-1
    # was bound to the D1D2-IgG chimera with a Kd of 500 nM and
    # dissociated with a k(diss) of 0.1 s(-1)." The paper reports no
    # kon, so kon here is this module's own derivation from those two.
    BondKinetics(
        molecule="LFA-1 (CD11a/CD18)", partner="ICAM-1",
        citation="PMID:9780171",
        source="Tominaga et al., J Immunol 1998;161:4016-22",
        method="surface plasmon resonance", organism="human",
        detail=(
            "soluble monomeric LFA-1 over an ICAM-1 D1D2-IgG chimera; "
            "membrane-purified monomeric LFA-1 gave similar kinetics. "
            "This is the resting ectodomain: locking the alphaL I "
            "domain open raises ICAM-1 affinity 9,000-fold "
            "(PMID:11353828), so one scalar cannot describe both "
            "conformational states"
        ),
        kd_molar=500e-9, koff_per_s=0.1,
        kon_per_molar_per_s=None,  # derived below from Kd and koff
    ),
    BondKinetics(
        molecule="ICAM-1 (CD54)", partner="LFA-1",
        citation="PMID:9780171",
        source="Tominaga et al., J Immunol 1998;161:4016-22",
        method="surface plasmon resonance", organism="human",
        detail=(
            "soluble monomeric LFA-1 over an ICAM-1 D1D2-IgG chimera. "
            "ICAM-1 also binds MAC-1, at a separate site in Ig domain "
            "3, for which no affinity could be sourced"
        ),
        kd_molar=500e-9, koff_per_s=0.1,
        kon_per_molar_per_s=None,
    ),
    # Chakraborty S et al. PLoS One 2015;10(3):e0121399. FRET gave
    # Kd = 41.82 +/- 2.36 nM and SPR gave Kd = 39.60 +/- 1.78 nM. The
    # SPR figure is recorded, for method consistency with every other
    # row in this table; the FRET figure is in `detail`.
    BondKinetics(
        molecule="VLA-4 (α4β1)", partner="VCAM-1",
        citation="PMID:25793408",
        source="Chakraborty et al., PLoS One 2015;10:e0121399 "
               "(doi:10.1371/journal.pone.0121399)",
        method="surface plasmon resonance", organism="human",
        detail=(
            "in-solution quantitative FRET on the same pair gave "
            "41.82 +/- 2.36 nM; no rate constants were reported. "
            "alpha4beta1 affinity is activation-state dependent "
            "(Mn2+, chemokine-triggered inside-out signalling)"
        ),
        kd_molar=39.60e-9,
    ),
    BondKinetics(
        molecule="VCAM-1 (CD106)", partner="VLA-4 (α4β1)",
        citation="PMID:25793408",
        source="Chakraborty et al., PLoS One 2015;10:e0121399 "
               "(doi:10.1371/journal.pone.0121399)",
        method="surface plasmon resonance", organism="human",
        detail=(
            "in-solution quantitative FRET on the same pair gave "
            "41.82 +/- 2.36 nM; no rate constants were reported. "
            "VCAM-1 also binds alpha4beta7, for which no affinity "
            "could be sourced"
        ),
        kd_molar=39.60e-9,
    ),
    # Baumgartner W, Drenckhahn D. Eur Biophys J 2002 Dec;31(7):532-8.
    # "we determined ... the dissociation constant K(D) to be about
    # 0.8 x 10(-4) M."
    BondKinetics(
        molecule="VE-cadherin (CD144)", partner="VE-cadherin (homophilic)",
        citation="PMID:12451422",
        source="Baumgartner & Drenckhahn, Eur Biophys J 2002;31:532-8 "
               "(doi:10.1007/s00249-002-0248-9)",
        method="affinity-column mobility shift", organism="mouse",
        detail=(
            "trans interaction of soluble dimeric VE-cadherin "
            "ectodomains; measured alongside a surface density of "
            "5e3 dimers/um2, the pair of numbers the authors' "
            "cooperative-linkage argument rests on"
        ),
        kd_molar=0.8e-4, kd_is_approximate=True,
    ),
)


#: Molecules whose kinetics could not be sourced, and why. These keep
#: whatever number the seed table carried, labelled PLACEHOLDER, so
#: that the gap is visible rather than disguised as a measurement.
#:
#: Each entry is the result of a literature search that came back
#: without a usable primary measurement. "Usable" means a peer-reviewed
#: determination of the named bond's affinity -- vendor datasheets,
#: patent examples and antibody-binding constants are not that, and are
#: recorded here as absences rather than promoted into the table.
UNSOURCED: Dict[str, str] = {
    "MAC-1 (CD11b/CD18)": (
        "no Kd found for alphaMbeta2 binding ICAM-1. The I domain is "
        "mapped as the recognition site for ICAM-1, iC3b, fibrinogen "
        "and factor X (Diamond et al. 1993) and ICAM-1's MAC-1 site is "
        "localized to Ig domain 3, but neither reports an affinity"
    ),
    "ICAM-2 (CD102)": (
        "no Kd found for ICAM-2 binding LFA-1. The affinity order "
        "ICAM-1 > ICAM-2 > ICAM-3 is published (PMID:11353828) but "
        "without a constant for ICAM-2"
    ),
    "CD31/PECAM-1": (
        "no Kd found for PECAM-1 homophilic binding. The interface is "
        "mapped to Ig domains 1-2 and the required contact residues "
        "are known, but the binding studies are qualitative"
    ),
    "JAM-A": (
        "no Kd found for JAM-A homophilic binding. The U-shaped dimer "
        "and its R(V,I,L)E dimerization motif are established by "
        "crystallography without an accompanying affinity measurement"
    ),
    "CD99": (
        "no Kd found for CD99 homophilic binding. Self-association is "
        "shown only qualitatively, by ELISA on recombinant CD99-Fc"
    ),
    "MAdCAM-1": (
        "no peer-reviewed Kd found for MAdCAM-1 binding alpha4beta7. "
        "The single-cell force-spectroscopy work is qualitative, and "
        "the only number located was a reagent-vendor QC figure, "
        "which is not a citable measurement"
    ),
}


#: The rupture force of an adhesive bond is not a property of the bond
#: alone: it depends on the loading rate, and selectin bonds are catch
#: bonds whose lifetime first *rises* with force. A single
#: ``binding_force_pN`` is therefore not a quantity that can be
#: measured, which is why the shipped 50.0 pN stays a placeholder on
#: every molecule rather than being curated.
BINDING_FORCE_RATIONALE = (
    "a bond rupture force is defined only together with a loading "
    "rate, and selectin bonds are catch bonds whose lifetime rises "
    "with force before falling; a single scalar is not a measurable "
    "quantity and this one has no source"
)

#: ``EntityEnricher._PHYSICS_DEFAULTS["adhesion_molecule"]``, duplicated
#: here on purpose and pinned by a test.
#:
#: The enricher fills ``physics_params`` only when it is empty. Once
#: this module writes the kinetics at seed time that dict is no longer
#: empty, so the enricher skips the type entirely -- and a fresh library
#: would quietly come up without ``binding_force_pN`` at all. A
#: curation pass that silences a default it did not replace is the same
#: class of defect as an uncited number: something disappears and
#: nothing says so. So the pass carries both defaults itself.
#:
#: ``bond_lifetime_seconds`` is always recomputed as 1/koff and never
#: read from here; it is listed so the test can check the whole table.
TYPE_DEFAULTS: Dict[str, Tuple[float, str]] = {
    "bond_lifetime_seconds": (2.0, "s"),
    "binding_force_pN": (50.0, "pN"),
}

#: What an unsourced Kd / kon / koff is: a family-level round number.
_UNSOURCED_RATIONALE = (
    "carried from the original seed table, where seven distinct Kd "
    "values covered fifteen molecules; no primary measurement of this "
    "bond could be sourced"
)

#: Appended when the entity's uncited rate constants contradict its own
#: measured Kd. Kd = koff/kon is a definition, so a measured Kd and a
#: pair of invented rates cannot both stand; saying so is better than
#: picking one of the two rates to keep and deriving the other, which
#: would manufacture agreement out of nothing.
_INCONSISTENT_RATES = (
    "WARNING: with the entity's uncited on-rate these rates imply a Kd "
    "of {implied:.4g} nM, against the measured {measured:.4g} nM "
    "({citation}) -- a factor of {ratio:.3g}. The three constants are "
    "not a self-consistent kinetic description and must not be used as "
    "one"
)


def _by_molecule() -> Dict[str, BondKinetics]:
    """First measurement per molecule, for the one-value-per-entity lookup.

    A molecule can characterise more than one bond -- ICAM-1 binds both LFA-1
    and MAC-1, with different kinetics -- so a second record for the same
    molecule with a DIFFERENT partner is a second bond, not a duplicate. Only
    a repeat of the same (molecule, partner) pair is an error. Raising on the
    second partner would have made adding a real measurement break the import.
    """
    out: Dict[str, BondKinetics] = {}
    for m in MEASUREMENTS:
        prev = out.get(m.molecule)
        if prev is not None:
            if prev.partner == m.partner:
                raise ValueError(
                    f"two records for {m.molecule!r} / {m.partner!r}")
            continue  # a different partner is a different bond
        out[m.molecule] = m
    return out


#: Measurement by library entity name. One per molecule -- see BY_PAIR when a
#: molecule characterises more than one bond.
BY_MOLECULE: Dict[str, BondKinetics] = _by_molecule()

#: Every record, keyed by the pair it characterises, so a second bond on the
#: same molecule stays reachable and reportable rather than being dropped.
BY_PAIR: Dict[Tuple[str, str], BondKinetics] = {
    (m.molecule, m.partner): m for m in MEASUREMENTS
}


def kinetics_for(name: str) -> Optional[BondKinetics]:
    """The curated measurement for one library entity name, if any."""
    return BY_MOLECULE.get(name)


# ── Internal consistency ─────────────────────────────────────────────

def check_internal_consistency(tolerance: float = 0.02) -> List[str]:
    """Check Kd = koff/kon on every record that publishes all three.

    A unit slip in this file would show up here and nowhere else: the
    conversions run in opposite directions (1e9 for a concentration,
    1e-9 for an inverse concentration), so getting one wrong breaks the
    identity by eighteen orders of magnitude. Two per cent absorbs the
    rounding in figures quoted to two significant figures -- the two
    checkable records come in at 0.6% (P-selectin) and 0.3%
    (E-selectin).

    Two kinds of record are excluded, each for a reason rather than for
    convenience:

    * a kon this module derived from Kd and koff satisfies the identity
      by construction, so checking it would prove nothing;
    * a record whose rates are published as lower bounds has no point
      value to check. L-selectin is the case: koff >= 10 1/s and
      kon >= 1e5 1/(M*s) imply 100 uM against a fitted Kd of 108 uM,
      and that 7% gap is the bound, not an error.
    """
    problems: List[str] = []
    for m in MEASUREMENTS:
        kd, koff, kon = m.kd_nm, m.koff_per_s, m.kon_per_nm_per_s
        if kd is None or not koff or not kon:
            continue
        if not m.kon_published or m.rates_are_lower_bounds:
            continue
        implied = koff / kon
        if abs(implied - kd) > tolerance * kd:
            problems.append(
                f"{m.molecule}: koff/kon = {implied:.4g} nM but Kd is "
                f"{kd:.4g} nM ({m.citation})"
            )
    return problems


# ── Writing the curation onto entities ───────────────────────────────

@dataclass
class EntityCuration:
    """What :func:`apply_curation` did to one entity."""

    entity_name: str
    measured: List[str] = field(default_factory=list)
    derived: List[str] = field(default_factory=list)
    #: Simulation-grade but not publication-grade: a value published
    #: only as a bound, which is not a measurement of the bound.
    #: Folding these into `placeholder` would understate them and
    #: folding them into `measured` would overstate them.
    estimated: List[str] = field(default_factory=list)
    placeholder: List[str] = field(default_factory=list)
    refused: List[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.measured or self.derived or self.estimated
                    or self.placeholder)

    def describe(self) -> str:
        bits = []
        if self.measured:
            bits.append(f"measured {sorted(self.measured)}")
        if self.derived:
            bits.append(f"derived {sorted(self.derived)}")
        if self.estimated:
            bits.append(f"estimated {sorted(self.estimated)}")
        if self.placeholder:
            bits.append(f"placeholder {sorted(self.placeholder)}")
        if self.refused:
            bits.append(f"REFUSED {self.refused}")
        return f"{self.entity_name}: " + (", ".join(bits) or "nothing to do")


def _placeholder(value: float, unit: str, *notes: str) -> ParameterValue:
    bits = [_UNSOURCED_RATIONALE] + [n for n in notes if n]
    return ParameterValue(
        value=float(value), unit=unit, provenance=Provenance.PLACEHOLDER,
        rationale=". ".join(bits),
    )


def unsourced_parameters(entity: BioEntity) -> Dict[str, ParameterValue]:
    """The entity's own uncited kinetics, labelled as placeholders.

    The seed table's Kd, kon and koff live on the dataclass as
    ``binding_affinity_kd`` (uM), ``on_rate`` (1/(uM*s)) and
    ``off_rate`` (1/s), outside ``physics_params`` and therefore outside
    everything that reads provenance: ``get_parameter``, ``assess`` and
    the entity-library page never saw them. Copying them in as explicit
    placeholders does not make them better numbers; it makes them
    visibly unevidenced instead of invisibly so.
    """
    why = UNSOURCED.get(entity.name, "")
    out: Dict[str, ParameterValue] = {}

    kd_um = getattr(entity, "binding_affinity_kd", 0.0) or 0.0
    koff = getattr(entity, "off_rate", 0.0) or 0.0
    kon_um = getattr(entity, "on_rate", 0.0) or 0.0

    # When the Kd is measured but the rates are not, the rates are very
    # likely to disagree with it, because they were made up separately.
    clash = ""
    measurement = BY_MOLECULE.get(entity.name)
    if (measurement is not None and measurement.kd_nm
            and measurement.koff_per_s is None and koff > 0 and kon_um > 0):
        implied = (koff / kon_um) * 1e3  # (1/s) / (1/(uM*s)) -> uM -> nM
        clash = _INCONSISTENT_RATES.format(
            implied=implied, measured=measurement.kd_nm,
            citation=measurement.citation,
            ratio=measurement.kd_nm / implied,
        )

    if kd_um > 0:
        out["kd_nm"] = _placeholder(kd_um * 1e3, "nM", why)
    if koff > 0:
        out["koff_per_s"] = _placeholder(koff, "1/s", why, clash)
    if kon_um > 0:
        out["kon_per_nm_per_s"] = _placeholder(
            kon_um * 1e-3, "1/(nM*s)", why, clash)
    return out


def apply_curation(entity: BioEntity) -> EntityCuration:
    """Write this entity's provenanced kinetics onto it.

    Curated molecules get MEASURED Kd and koff, a DERIVED kon and a
    DERIVED bond lifetime. Everything else gets its existing seed-table
    numbers recorded as PLACEHOLDER, which is what they are.
    ``binding_force_pN`` is always a placeholder -- see
    :data:`BINDING_FORCE_RATIONALE`.

    Writes go through :func:`~cognisom.library.parameters.set_parameter`
    in non-strict mode, so a value that fails validation is reported in
    ``refused`` and not stored. Nothing here is important enough to
    justify writing an unsound number.
    """
    result = EntityCuration(entity_name=entity.name)
    if entity.entity_type is not EntityType.ADHESION_MOLECULE:
        return result

    measurement = BY_MOLECULE.get(entity.name)
    curated = measurement.parameters() if measurement else {}

    # Any kinetic parameter the measurement does not cover falls back to
    # the entity's own uncited number, labelled for what it is.
    to_write: Dict[str, ParameterValue] = {}
    for name, pv in unsourced_parameters(entity).items():
        if name not in curated:
            to_write[name] = pv
    to_write.update(curated)

    # tau = 1/koff is the definition of the mean lifetime of a bond
    # that breaks as a first-order process -- the same model the
    # diapedesis solver uses in `p_unbind = koff * dt`. Applying it
    # uniformly is what stops the library asserting a 2.0 s lifetime
    # alongside an off-rate that implies 0.22 s. A derivation is only
    # as good as its input, so the provenance follows koff's: DERIVED
    # from a measurement, PLACEHOLDER from a placeholder.
    koff_pv = to_write.get("koff_per_s")
    if koff_pv is not None and koff_pv.value > 0:
        # tau is a definition applied to koff, so it inherits koff's
        # grade rather than being judged on its own: DERIVED from a
        # measurement, ESTIMATED from a published bound, PLACEHOLDER
        # from a placeholder. Collapsing the middle case to PLACEHOLDER
        # would discard a cited number; promoting it to DERIVED would
        # make a bound look like a measurement.
        measured = koff_pv.provenance is Provenance.MEASURED
        bounded = koff_pv.provenance is Provenance.ESTIMATED
        to_write["bond_lifetime_seconds"] = ParameterValue(
            value=_clean(1.0 / koff_pv.value), unit="s",
            provenance=(
                Provenance.DERIVED if measured
                else Provenance.ESTIMATED if bounded
                else Provenance.PLACEHOLDER
            ),
            citation=koff_pv.citation, source=koff_pv.source,
            conditions=koff_pv.conditions,
            rationale=(
                f"mean lifetime of a bond breaking as a first-order "
                f"process is 1/koff = 1/{koff_pv.value} 1/s"
                + (
                    "; an upper bound, because the cited koff is a "
                    "lower bound"
                    if "lower bound" in koff_pv.rationale else ""
                )
                + (
                    "" if measured else
                    "; an upper bound, inherited from a cited off-rate "
                    "that is itself only a lower bound" if bounded else
                    "; no better than the uncited off-rate it comes from"
                )
            ),
        )

    default_force, force_unit = TYPE_DEFAULTS["binding_force_pN"]
    force = entity.physics_params.get("binding_force_pN", default_force)
    if not isinstance(force, (int, float)) or isinstance(force, bool):
        force = default_force
    to_write["binding_force_pN"] = ParameterValue(
        value=float(force), unit=force_unit,
        provenance=Provenance.PLACEHOLDER,
        rationale=BINDING_FORCE_RATIONALE,
    )

    for name, pv in sorted(to_write.items()):
        problems = set_parameter(entity, name, pv, strict=False)
        if problems:
            result.refused.append(f"{name}: {'; '.join(problems)}")
            continue
        if pv.provenance is Provenance.MEASURED:
            result.measured.append(name)
        elif pv.provenance is Provenance.DERIVED:
            result.derived.append(name)
        elif pv.provenance is Provenance.ESTIMATED:
            result.estimated.append(name)
        else:
            result.placeholder.append(name)

    return result


def reconcile_legacy_fields(entity: BioEntity) -> List[str]:
    """Bring the dataclass kinetics into line with the curated values.

    ``AdhesionMolecule`` keeps its own ``binding_affinity_kd`` (uM),
    ``on_rate`` (1/(uM*s)) and ``off_rate`` (1/s). Leaving those at the
    seed table's figures while ``physics_params`` carried the measured
    ones would give the library two contradicting answers for the same
    constant -- a worse state than having only the wrong one. Returns
    the names of the fields changed.
    """
    measurement = BY_MOLECULE.get(entity.name)
    if measurement is None:
        return []

    changed: List[str] = []
    if measurement.kd_nm is not None:
        kd_um = measurement.kd_nm / 1e3
        if getattr(entity, "binding_affinity_kd", None) != kd_um:
            entity.binding_affinity_kd = kd_um
            changed.append("binding_affinity_kd")
    if measurement.koff_per_s is not None:
        if getattr(entity, "off_rate", None) != measurement.koff_per_s:
            entity.off_rate = measurement.koff_per_s
            changed.append("off_rate")
    if measurement.kon_per_nm_per_s is not None:
        kon_um = measurement.kon_per_nm_per_s * 1e3
        if getattr(entity, "on_rate", None) != kon_um:
            entity.on_rate = kon_um
            changed.append("on_rate")
    return changed


# ── Store-level pass ─────────────────────────────────────────────────

@dataclass
class CurationRun:
    """What a pass over the store's adhesion molecules did."""

    entities: List[EntityCuration] = field(default_factory=list)
    legacy_fields_changed: Dict[str, List[str]] = field(default_factory=dict)
    persisted: int = 0
    persist_errors: List[str] = field(default_factory=list)

    @property
    def measured_parameters(self) -> int:
        return sum(len(e.measured) for e in self.entities)

    @property
    def derived_parameters(self) -> int:
        return sum(len(e.derived) for e in self.entities)

    @property
    def estimated_parameters(self) -> int:
        return sum(len(e.estimated) for e in self.entities)

    @property
    def placeholder_parameters(self) -> int:
        return sum(len(e.placeholder) for e in self.entities)

    @property
    def refusals(self) -> List[str]:
        return [
            f"{e.entity_name}: {r}" for e in self.entities for r in e.refused
        ]

    def describe(self) -> str:
        lines = [
            f"adhesion kinetics: {len(self.entities)} entities, "
            f"{self.measured_parameters} measured, "
            f"{self.derived_parameters} derived, "
            f"{self.estimated_parameters} estimated, "
            f"{self.placeholder_parameters} placeholder, "
            f"{self.persisted} persisted"
        ]
        for e in self.entities:
            lines.append("  " + e.describe())
        if self.legacy_fields_changed:
            lines.append("  corrected dataclass fields:")
            for name, fields_ in sorted(self.legacy_fields_changed.items()):
                lines.append(f"    {name}: {sorted(fields_)}")
        for r in self.refusals:
            lines.append(f"  REFUSED {r}")
        for e in self.persist_errors:
            lines.append(f"  PERSIST FAILED {e}")
        return "\n".join(lines)


def curate_store(store: Any, *, persist: bool = True) -> CurationRun:
    """Apply the curated kinetics to every adhesion molecule in a store.

    Idempotent: a second pass rewrites the same values and the same
    sidecars. Safe to call on an already-curated store, which is what
    makes it usable both from the auto-seed path and as a migration for
    a database that was seeded before this table existed.
    """
    run = CurationRun()
    entities, _ = store.search(
        entity_type=EntityType.ADHESION_MOLECULE.value, limit=5000,
    )
    for entity in sorted(entities, key=lambda e: e.name):
        fields_ = reconcile_legacy_fields(entity)
        if fields_:
            run.legacy_fields_changed[entity.name] = fields_
        result = apply_curation(entity)
        run.entities.append(result)
        if not persist or not (result.changed or fields_):
            continue
        try:
            store.update_entity(entity, changed_by="adhesion_kinetics")
            run.persisted += 1
        except Exception as exc:  # pragma: no cover - store-specific
            run.persist_errors.append(f"{entity.name}: {exc}")
    return run


def main(argv: Optional[List[str]] = None) -> int:
    """``python -m cognisom.library.adhesion_kinetics``.

    Migrates a library that was seeded before this table existed. A
    fresh database gets the same values from the auto-seed path, so this
    is only needed for an existing one.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply curated adhesion-molecule kinetics to the library.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would change without writing.",
    )
    args = parser.parse_args(argv)

    problems = check_internal_consistency()
    if problems:
        print("curated table is internally inconsistent:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 2

    from .store import EntityStore

    run = curate_store(EntityStore(), persist=not args.dry_run)
    print(run.describe())
    if UNSOURCED:
        print("\nunsourced, left as placeholders:")
        for name, why in sorted(UNSOURCED.items()):
            print(f"  {name}: {why}")
    return 1 if (run.refusals or run.persist_errors) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
