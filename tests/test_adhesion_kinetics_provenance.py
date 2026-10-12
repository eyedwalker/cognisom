"""
Tests for the curated adhesion-molecule kinetics.

Two things are being defended here, and they are different.

The first is arithmetic. The curated table holds constants in the units
the papers print -- Kd in molar, kon in 1/(M*s) -- and converts them to
the library's nanomolar units in one place. The conversions run in
opposite directions, so a slip gives an answer wrong by eighteen orders
of magnitude while still looking like a number. The identity
Kd = koff/kon catches exactly that, and so do the transcription
regression tests, which pin the published figures against the abstracts
they came from.

The second is honesty. Six of the fifteen molecules have no sourceable
measurement. The risk is not that someone writes a wrong number; it is
that someone quietly moves a molecule out of the gap list without
finding a citation, or records a family-level round figure as MEASURED.
Several tests below exist only to make that impossible to do silently:
the table and the gap list must partition the fifteen molecules
exactly, every MEASURED value must carry a PMID or a DOI, and no
entity may end up asserting a measured Kd alongside rate constants
that contradict it.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.library.adhesion_kinetics import (
    BINDING_FORCE_RATIONALE,
    TYPE_DEFAULTS,
    BY_MOLECULE,
    MEASUREMENTS,
    UNSOURCED,
    BondKinetics,
    apply_curation,
    check_internal_consistency,
    curate_store,
    kinetics_for,
    reconcile_legacy_fields,
)
from cognisom.library.models import AdhesionMolecule, Cytokine, EntityType
from cognisom.library.parameters import (
    PARAMETER_SPECS,
    Provenance,
    assess,
    coverage,
    get_parameter,
    iter_parameters,
)

#: The three kinetic parameters, plus the two the type default carried.
_KINETIC = ("kd_nm", "koff_per_s", "kon_per_nm_per_s")

#: A PMID or a DOI. "curated" is not an identifier, and neither is a
#: journal name: the point of a citation is that a reader can resolve it.
_CITATION = re.compile(r"^(PMID:\d{4,9}|(doi:)?10\.\d{4,9}/\S+)$")


# ── The curated table: arithmetic ────────────────────────────────────

def test_the_published_constants_satisfy_kd_equals_koff_over_kon():
    """The unit-error detector. A concentration scales by 1e9 and an
    inverse concentration by 1e-9, so one wrong conversion breaks this
    identity by eighteen orders of magnitude."""
    assert check_internal_consistency() == []


def test_a_wrong_unit_conversion_would_be_caught():
    """The previous test only means something if it can fail."""
    broken = BondKinetics(
        molecule="x", partner="y", citation="PMID:1", source="s",
        method="m", organism="o",
        # kon converted the wrong way: 1e9 instead of 1e-9.
        kd_molar=320e-9, koff_per_s=1.4, kon_per_molar_per_s=4.4e6 * 1e18,
    )
    implied = broken.koff_per_s / broken.kon_per_nm_per_s
    assert abs(implied - broken.kd_nm) > 0.02 * broken.kd_nm


@pytest.mark.parametrize("molecule,kd_nm,koff,kon", [
    # Mehta, Cummings & McEver, J Biol Chem 1998;273:32506-13, PMID
    # 9829984: "Kd of 320 +/- 20 nM. The measured koff was 1.4 +/- 0.1
    # s-1, and the calculated kon was 4.4 x 10(6) M-1 s-1."
    ("P-selectin (CD62P)", 320.0, 1.4, 4.4e-3),
    ("PSGL-1 (CD162)", 320.0, 1.4, 4.4e-3),
    # Wild et al., J Biol Chem 2001;276:31602-12, PMID 11404363:
    # "dissociation rate constant of 4.6 s(-1) and a calculated
    # association rate constant of 7.4 x 10(4) m(-1) s(-1) ... K(d) of
    # 62 microm."
    ("E-selectin (CD62E)", 62_000.0, 4.6, 7.4e-5),
    # Nicholson et al., J Biol Chem 1998;273:763-70, PMID 9422729:
    # "Kd of 108 microM ... dissociation rate constant (>/=10 s-1) ...
    # association rate constant is >/=10(5) M-1 s-1."
    ("L-selectin (CD62L)", 108_000.0, 10.0, 1.0e-4),
    # Tominaga et al., J Immunol 1998;161:4016-22, PMID 9780171: "a Kd
    # of 500 nM and dissociated with a k(diss) of 0.1 s(-1)." kon is
    # this module's derivation: 0.1 / 500 nM = 2e-4 1/(nM*s).
    ("LFA-1 (CD11a/CD18)", 500.0, 0.1, 2.0e-4),
    ("ICAM-1 (CD54)", 500.0, 0.1, 2.0e-4),
    # Chakraborty et al., PLoS One 2015;10:e0121399, PMID 25793408:
    # SPR Kd = 39.60 +/- 1.78 nM. No rate constants reported.
    ("VLA-4 (α4β1)", 39.6, None, None),
    ("VCAM-1 (CD106)", 39.6, None, None),
    # Baumgartner & Drenckhahn, Eur Biophys J 2002;31:532-8, PMID
    # 12451422: "K(D) to be about 0.8 x 10(-4) M."
    ("VE-cadherin (CD144)", 80_000.0, None, None),
])
def test_each_constant_matches_the_abstract_it_was_transcribed_from(
        molecule, kd_nm, koff, kon):
    m = kinetics_for(molecule)
    assert m is not None, f"{molecule} dropped out of the curated table"
    assert m.kd_nm == pytest.approx(kd_nm, rel=1e-9)
    assert m.koff_per_s == (None if koff is None else pytest.approx(koff))
    assert m.kon_per_nm_per_s == (
        None if kon is None else pytest.approx(kon, rel=1e-9))


def test_the_p_selectin_correction_is_not_quietly_reverted():
    """The library shipped Kd = 100 uM for P-selectin. The measured
    value is 320 nM: a factor of 312. If someone restores the round
    number, this fails rather than the simulation silently getting a
    three-hundred-fold weaker bond."""
    kd = kinetics_for("P-selectin (CD62P)").kd_nm
    assert kd < 1_000.0, "100 uM is the uncited seed value, not a measurement"
    assert 100_000.0 / kd > 100


# ── The curated table: honesty ───────────────────────────────────────

def test_every_citation_is_a_resolvable_identifier():
    """"curated" is not a citation. Each of these PMIDs was confirmed
    against NCBI E-utilities to resolve to the journal, volume and
    pages recorded in the table's comments."""
    for m in MEASUREMENTS:
        assert _CITATION.match(m.citation), (
            f"{m.molecule}: {m.citation!r} is not a PMID or a DOI")


def test_every_measured_value_carries_a_citation_and_conditions():
    for m in MEASUREMENTS:
        for name, pv in m.parameters().items():
            if pv.provenance is Provenance.MEASURED:
                assert pv.citation, f"{m.molecule}.{name} cites nothing"
            assert pv.conditions, f"{m.molecule}.{name} records no conditions"
            assert pv.validate(PARAMETER_SPECS.get(name)) == [], (
                f"{m.molecule}.{name} is not a sound value")


def test_a_bond_records_the_partner_it_was_measured_against():
    """A Kd belongs to a pair. Recording it against the molecule alone
    loses half the measurement, and these constants differ by three
    orders of magnitude between partners of the same selectin."""
    for m in MEASUREMENTS:
        assert m.partner, f"{m.molecule} names no binding partner"
        assert m.partner in m.conditions()


def test_the_table_and_the_gap_list_partition_the_fifteen_molecules():
    """The honesty invariant. A molecule is either curated or listed as
    a gap with a reason -- never silently neither, which is how an
    uncited round number becomes indistinguishable from a measurement
    again."""
    from cognisom.library.seed_immunology import _seed_adhesion_molecules

    class _Collect:
        def __init__(self):
            self.names = []

        def add_entity(self, entity):
            self.names.append(entity.name)
            return True

    sink = _Collect()
    _seed_adhesion_molecules(sink)
    seeded = set(sink.names)
    assert len(seeded) == 15

    curated = set(BY_MOLECULE)
    gaps = set(UNSOURCED)
    assert not (curated & gaps), f"both curated and a gap: {curated & gaps}"
    assert curated | gaps == seeded, (
        f"neither curated nor listed as a gap: {seeded - curated - gaps}; "
        f"named but not seeded: {curated | gaps - seeded}"
    )


def test_every_gap_says_why_rather_than_just_being_blank():
    for name, why in UNSOURCED.items():
        assert len(why) > 40, f"{name}: {why!r} is not a reason"


def test_the_rupture_force_stays_a_placeholder_on_every_molecule():
    """A rupture force without a loading rate is not a quantity, and
    selectin bonds are catch bonds. There is nothing here to curate."""
    assert "loading rate" in BINDING_FORCE_RATIONALE
    entity = AdhesionMolecule(name="P-selectin (CD62P)")
    entity.physics_params["binding_force_pN"] = 50.0
    apply_curation(entity)
    pv = get_parameter(entity, "binding_force_pN")
    assert pv.provenance is Provenance.PLACEHOLDER
    assert not pv.is_simulation_grade


# ── Derivation ───────────────────────────────────────────────────────

def test_the_bond_lifetime_is_one_over_koff_on_every_molecule():
    """tau = 1/koff is the mean lifetime of a bond that breaks as a
    first-order process, which is the model the diapedesis solver uses.
    The shipped 2.0 s on all fifteen molecules agreed with none of the
    off-rates."""
    for name in list(BY_MOLECULE) + list(UNSOURCED):
        entity = AdhesionMolecule(name=name)
        seeded = _seeded(name)
        entity.off_rate = seeded.off_rate
        entity.on_rate = seeded.on_rate
        entity.binding_affinity_kd = seeded.binding_affinity_kd
        reconcile_legacy_fields(entity)
        apply_curation(entity)
        koff = get_parameter(entity, "koff_per_s")
        tau = get_parameter(entity, "bond_lifetime_seconds")
        assert koff is not None and tau is not None, name
        assert tau.value == pytest.approx(1.0 / koff.value, rel=1e-9), name
        assert tau.value != 2.0 or koff.value == 0.5


def test_a_derivation_is_no_better_than_its_input():
    """A lifetime derived from an uncited off-rate is a placeholder, not
    a DERIVED value. Promoting it would launder an uncited number into
    publication grade."""
    gap = AdhesionMolecule(name="CD99")
    gap.off_rate, gap.on_rate, gap.binding_affinity_kd = 0.1, 0.3, 5.0
    apply_curation(gap)
    assert get_parameter(
        gap, "bond_lifetime_seconds").provenance is Provenance.PLACEHOLDER

    cited = AdhesionMolecule(name="P-selectin (CD62P)")
    apply_curation(cited)
    tau = get_parameter(cited, "bond_lifetime_seconds")
    assert tau.provenance is Provenance.DERIVED
    assert tau.is_publication_grade
    assert "1/koff" in tau.rationale


def test_a_calculated_kon_is_never_recorded_as_measured():
    """Every one of these papers describes its kon as calculated from
    the measured Kd and koff. DERIVED with the derivation written down
    is what that is; MEASURED would overstate it. Where the underlying
    rates are published only as bounds the kon drops a further level to
    ESTIMATED, because a bound is not a measurement of the bound."""
    for m in MEASUREMENTS:
        pv = m.parameters().get("kon_per_nm_per_s")
        if pv is None:
            continue
        assert pv.provenance is not Provenance.MEASURED
        expected = (
            Provenance.ESTIMATED if m.rates_are_lower_bounds
            else Provenance.DERIVED
        )
        assert pv.provenance is expected, m.molecule
        assert "koff/Kd" in pv.rationale


# ── A published bound is not a measurement of the bound ──────────────

def test_a_rate_published_only_as_a_bound_is_not_called_measured():
    """Nicholson et al. report L-selectin's off-rate as ">= 10 1/s"
    because dissociation outran the instrument. Storing 10.0 as
    MEASURED would assert the paper measured 10.0.

    The number is still worth keeping -- it is the best available and a
    simulation may consume it -- so ESTIMATED is the right level: in
    SIMULATION_GRADE, out of PUBLICATION_GRADE, and obliged to carry a
    rationale that says it is a bound. Without this, the hundred-fold
    spread of off-rates across this family would be reported at a
    confidence none of it has."""
    bounded = [m for m in MEASUREMENTS if m.rates_are_lower_bounds]
    assert bounded, "the bound case must stay represented in the table"

    for m in bounded:
        params = m.parameters()

        koff = params["koff_per_s"]
        assert koff.provenance is Provenance.ESTIMATED, m.molecule
        assert koff.is_simulation_grade
        assert not koff.is_publication_grade
        # The rationale must name the bound, not merely hint at it.
        assert ">=" in koff.rationale
        assert "bound" in koff.rationale.lower()
        # The citation is retained: the bound came from somewhere.
        assert koff.citation.startswith("PMID:")
        assert koff.validate() == []

        # A fitted Kd in the same paper keeps its own, higher grade.
        kd = params.get("kd_nm")
        if kd is not None:
            assert kd.provenance is Provenance.MEASURED, m.molecule


def test_an_unbounded_rate_keeps_its_measured_grade():
    """The downgrade must be specific to the bound, not a blanket
    pessimism that would hide the four genuinely fitted off-rates."""
    fitted = [m for m in MEASUREMENTS
              if not m.rates_are_lower_bounds and m.koff_per_s is not None]
    assert len(fitted) >= 4

    for m in fitted:
        koff = m.parameters()["koff_per_s"]
        assert koff.provenance is Provenance.MEASURED, m.molecule
        assert koff.is_publication_grade
        assert koff.validate() == []


# ── No two answers for the same constant ─────────────────────────────

def _seeded(name):
    """The entity exactly as the seeder builds it."""
    from cognisom.library.seed_immunology import _seed_adhesion_molecules

    captured = {}

    class _Collect:
        def add_entity(self, entity):
            captured[entity.name] = entity
            return True

    _seed_adhesion_molecules(_Collect())
    return captured[name]


@pytest.mark.parametrize("molecule", sorted(BY_MOLECULE))
def test_the_dataclass_field_and_the_provenanced_parameter_agree(molecule):
    """AdhesionMolecule keeps its own Kd in uM. Letting that drift from
    the curated nM value would give the library two contradicting
    answers, which is worse than having only the wrong one."""
    entity = _seeded(molecule)
    m = BY_MOLECULE[molecule]
    assert entity.binding_affinity_kd == pytest.approx(m.kd_nm / 1e3)
    kd = get_parameter(entity, "kd_nm")
    assert kd.value == pytest.approx(entity.binding_affinity_kd * 1e3)
    if m.koff_per_s is not None:
        assert entity.off_rate == pytest.approx(m.koff_per_s)
        assert entity.on_rate == pytest.approx(m.kon_per_nm_per_s * 1e3)


def test_no_entity_asserts_a_measured_kd_with_rates_that_contradict_it():
    """Kd = koff/kon is a definition. Three constants that cannot all
    be true must not all be simulation-grade; where the measured Kd and
    the uncited rates disagree, the rates stay placeholders and say so."""
    for molecule in sorted(BY_MOLECULE):
        entity = _seeded(molecule)
        kd = get_parameter(entity, "kd_nm")
        koff = get_parameter(entity, "koff_per_s")
        kon = get_parameter(entity, "kon_per_nm_per_s")
        if not (kd and koff and kon):
            continue
        grade = [p.is_simulation_grade for p in (kd, koff, kon)]
        if all(grade):
            implied = koff.value / kon.value
            assert implied == pytest.approx(kd.value, rel=0.08), (
                f"{molecule}: koff/kon = {implied} but Kd = {kd.value}")
        else:
            assert not koff.is_simulation_grade
            assert "WARNING" in koff.rationale


def test_the_warning_names_both_numbers_so_a_reader_can_check_it():
    entity = _seeded("VLA-4 (α4β1)")
    koff = get_parameter(entity, "koff_per_s")
    assert "39.6" in koff.rationale
    assert "PMID:25793408" in koff.rationale


# ── Writing onto entities ────────────────────────────────────────────

def test_curation_refuses_rather_than_stores_an_unsound_value():
    """set_parameter runs in non-strict mode here, so a bad value has
    to be reported and dropped rather than raise. Nothing in this table
    is worth writing an unsound number for."""
    from cognisom.library.parameters import ParameterValue, set_parameter

    entity = AdhesionMolecule(name="E-selectin (CD62E)")
    problems = set_parameter(entity, "bond_lifetime_seconds", ParameterValue(
        value=0.22, unit="furlongs_per_fortnight",
        provenance=Provenance.MEASURED, citation="PMID:11404363",
    ), strict=False)
    assert problems
    assert "bond_lifetime_seconds" not in entity.physics_params


def test_curation_leaves_a_non_adhesion_entity_alone():
    cytokine = Cytokine(name="IL-2")
    result = apply_curation(cytokine)
    assert not result.changed
    assert cytokine.physics_params == {} or "kd_nm" not in cytokine.physics_params


def test_curation_is_idempotent():
    first = _seeded("P-selectin (CD62P)")
    before = {n: pv.to_dict() for n, pv in iter_parameters(first)}
    apply_curation(first)
    apply_curation(first)
    after = {n: pv.to_dict() for n, pv in iter_parameters(first)}
    assert before == after


def test_an_unsourced_molecule_becomes_visibly_rather_than_invisibly_unevidenced():
    """Before curation the seed table's Kd lived on the dataclass,
    outside physics_params, where get_parameter and assess never saw
    it. It is still an uncited round number; it is now an uncited round
    number that says so."""
    entity = _seeded("MAdCAM-1")
    pv = get_parameter(entity, "kd_nm")
    assert pv is not None, "the number is now visible to the provenance system"
    assert pv.value == pytest.approx(2_000.0)  # 2.0 uM, as seeded
    assert pv.provenance is Provenance.PLACEHOLDER
    assert "no primary measurement" in pv.rationale
    assert not assess(entity).is_ready


# ── Through the store ────────────────────────────────────────────────

@pytest.fixture(scope="module")
def seeded_store(tmp_path_factory):
    from cognisom.library.store import EntityStore

    path = tmp_path_factory.mktemp("adhesion") / "entities.db"
    return EntityStore(db_path=str(path))


def test_a_freshly_seeded_library_carries_the_citations(seeded_store):
    entities, _ = seeded_store.search(
        entity_type=EntityType.ADHESION_MOLECULE.value, limit=100)
    assert len(entities) == 15
    cited = {
        pv.citation
        for e in entities for _, pv in iter_parameters(e) if pv.citation
    }
    assert cited == {
        "PMID:11404363", "PMID:12451422", "PMID:25793408",
        "PMID:9422729", "PMID:9780171", "PMID:9829984",
    }


def test_the_curated_molecules_are_simulation_ready_and_the_gaps_are_not(
        seeded_store):
    entities, _ = seeded_store.search(
        entity_type=EntityType.ADHESION_MOLECULE.value, limit=100)
    report = coverage(entities)
    assert report.total == 15
    assert report.ready == len(BY_MOLECULE) == 9
    assert {r.entity_name for r in report.not_ready} == set(UNSOURCED)


def test_no_adhesion_molecule_is_missing_its_kd_any_more(seeded_store):
    """Before this, assess() reported kd_nm absent on all fifteen while
    a number sat in the dataclass. Absent and unevidenced need
    different work, so the report must not conflate them."""
    entities, _ = seeded_store.search(
        entity_type=EntityType.ADHESION_MOLECULE.value, limit=100)
    report = coverage(entities)
    assert report.by_parameter_missing.get("kd_nm", 0) == 0
    assert report.by_parameter_placeholder.get("kd_nm", 0) == len(UNSOURCED)
    assert report.by_parameter_uncurated["kd_nm"] == len(UNSOURCED)


def test_provenance_survives_the_store_round_trip(seeded_store):
    entity = seeded_store.find_entity_by_name(
        "P-selectin (CD62P)", EntityType.ADHESION_MOLECULE.value)
    assert entity is not None
    pv = get_parameter(entity, "kd_nm")
    assert pv.value == pytest.approx(320.0)
    assert pv.citation == "PMID:9829984"
    assert pv.provenance is Provenance.MEASURED
    assert "surface plasmon resonance" in pv.conditions


def test_the_migration_pass_is_idempotent_on_a_seeded_store(seeded_store):
    """An existing database runs curate_store as a migration; a fresh
    one already has the values. Running it twice must not differ from
    running it once."""
    first = curate_store(seeded_store)
    assert first.refusals == []
    assert first.persist_errors == []
    second = curate_store(seeded_store)
    assert second.refusals == []
    assert second.measured_parameters == first.measured_parameters
    assert second.derived_parameters == first.derived_parameters
    assert second.legacy_fields_changed == {}


def test_the_curation_report_separates_absent_from_unevidenced(seeded_store):
    from cognisom.library.curation_report import build_report

    text = build_report(seeded_store, (EntityType.ADHESION_MOLECULE,))
    assert "present but unevidenced  kd_nm" in text
    assert "absent  kd_nm" not in text


# ── Nothing the enricher used to supply goes missing ─────────────────

def test_the_type_defaults_match_the_enrichers():
    """They are written down in two places, so a test has to hold them
    together. If the enricher's table changes, this fails rather than
    the curation pass silently dropping or resurrecting a key."""
    from cognisom.library.enrichment import EntityEnricher

    enricher = EntityEnricher._PHYSICS_DEFAULTS["adhesion_molecule"]
    assert set(TYPE_DEFAULTS) == set(enricher)
    for name, (value, _unit) in TYPE_DEFAULTS.items():
        assert enricher[name] == value


def test_a_freshly_seeded_molecule_keeps_every_key_the_enricher_declares(
        seeded_store):
    """The enricher fills physics_params only when it is empty, so
    writing the kinetics at seed time makes it skip this type. Both of
    its defaults must therefore come from the curation pass."""
    from cognisom.library.enrichment import EntityEnricher

    declared = set(EntityEnricher._PHYSICS_DEFAULTS["adhesion_molecule"])
    entities, _ = seeded_store.search(
        entity_type=EntityType.ADHESION_MOLECULE.value, limit=100)
    assert entities
    for entity in entities:
        present = {name for name, _ in iter_parameters(entity)}
        assert declared <= present, (
            f"{entity.name} lost {sorted(declared - present)}")


# ── The new specs ────────────────────────────────────────────────────

@pytest.mark.parametrize("name,unit", [
    ("bond_lifetime_seconds", "s"),
    ("binding_force_pN", "pN"),
])
def test_the_type_default_parameters_now_have_a_declared_unit(name, unit):
    """Both were carried on all fifteen adhesion molecules with no spec
    at all, so no unit, range or applies_to check ran on either."""
    spec = PARAMETER_SPECS[name]
    assert spec.unit == unit
    assert spec.applies_to == (EntityType.ADHESION_MOLECULE,)


# ── A dead source must not look like an empty one ────────────────────
#
# SABIO-RK's REST paths answer HTTP 200 with a 1182-byte HTML shell for
# every query (verified live: both /kineticLaws and
# /searchKineticLaws/parameters). The old client read a non-JSON 200 as
# `entries = []`, so enrich_all_enzymes reported
# {"enriched": 0, "total_params": 0, "errors": []} across the whole
# library -- a retired endpoint presented as "no enzyme here has
# published kinetics". These tests use a stub response, so they need no
# network.

class _FakeResponse:
    def __init__(self, status, ctype, body=b""):
        self.status_code = status
        self.headers = {"content-type": ctype}
        self.content = body
        self.text = body.decode("utf-8", "replace")

    def json(self):
        import json
        return json.loads(self.text)


def _html_shell():
    return _FakeResponse(200, "text/html; charset=utf-8", b"<html>" + b"x" * 1170)


def test_an_html_shell_is_refused_not_read_as_an_empty_result():
    from cognisom.library.kinetics_client import (
        SABIORKClient, KineticsSourceUnavailable)

    client = SABIORKClient()
    client.session.get = lambda *a, **k: _html_shell()

    with pytest.raises(KineticsSourceUnavailable) as exc:
        client.search_by_enzyme("MMP9")

    msg = str(exc.value)
    # The message has to be diagnosable in one read: what was asked,
    # what came back, and how big it was.
    assert "text/html" in msg
    assert "1176 bytes" in msg or "bytes" in msg
    assert "retired" in msg


def test_a_reaction_search_refuses_the_same_way():
    from cognisom.library.kinetics_client import (
        SABIORKClient, KineticsSourceUnavailable)

    client = SABIORKClient()
    client.session.get = lambda *a, **k: _html_shell()

    with pytest.raises(KineticsSourceUnavailable):
        client.search_by_reaction("glucose", "glucose-6-phosphate")


def test_a_real_json_payload_still_parses():
    """The guard must not refuse a working endpoint. If SABIO ever
    returns, the client has to keep working."""
    from cognisom.library.kinetics_client import SABIORKClient

    client = SABIORKClient()
    client.session.get = lambda *a, **k: _FakeResponse(
        200, "application/json", b"[]")

    result = client.search_by_enzyme("hexokinase")
    assert result.parameters == []


def test_a_non_200_is_refused_rather_than_emptied():
    from cognisom.library.kinetics_client import (
        SABIORKClient, KineticsSourceUnavailable)

    client = SABIORKClient()
    client.session.get = lambda *a, **k: _FakeResponse(503, "text/html")

    with pytest.raises(KineticsSourceUnavailable) as exc:
        client.search_by_enzyme("MMP9")
    assert "503" in str(exc.value)


def test_the_batch_reports_the_outage_instead_of_a_clean_zero():
    """This is the defect that mattered: the batch answered
    `{"enriched": 0, "errors": []}` over every gene in the library,
    which reads as a finding about the library rather than about the
    endpoint."""
    from cognisom.library import kinetics_client
    from cognisom.library.kinetics_client import (
        enrich_all_enzymes, KineticsSourceUnavailable)

    class _Store:
        def search(self, entity_type=None, limit=100):
            class _E:
                name = "TP53"
            return [_E(), _E(), _E()], 3

    def _boom(store, name, etype=None):
        raise KineticsSourceUnavailable("endpoint retired")

    original = kinetics_client.enrich_entity_kinetics
    kinetics_client.enrich_entity_kinetics = _boom
    try:
        results = enrich_all_enzymes(_Store())
    finally:
        kinetics_client.enrich_entity_kinetics = original

    assert results["source_available"] is False
    assert results["errors"], "an outage must leave a trace in errors"
    assert "unavailable" in results["errors"][0]
    # It stops rather than grinding through the library to the same end.
    assert len(results["errors"]) == 1
    assert results["enriched"] == 0


# ── A value held under a legacy name is not an absent value ──────────

def test_a_legacy_diffusion_coefficient_resolves_to_the_specced_name():
    """PARAMETER_SPECS declares diffusion_coeff_um2_s in um^2/s; the
    seeded cytokines carry diffusion_coefficient_m2_per_s in m^2/s. So
    get_parameter returned None for a number sitting right there, and
    assess() called it *missing* -- the curation report asked for a
    value the library already held."""
    from cognisom.library.models import Cytokine
    from cognisom.library.parameters import get_parameter, assess

    c = Cytokine(name="IL-test")
    c.physics_params = {"diffusion_coefficient_m2_per_s": 1e-10}

    pv = get_parameter(c, "diffusion_coeff_um2_s")
    assert pv is not None
    # 1e-10 m^2/s * 1e12 um^2/m^2 = 100 um^2/s, the right order for a
    # ~17 kDa protein in free solution.
    assert pv.value == pytest.approx(100.0)
    assert pv.unit == "um^2/s"
    # A unit conversion creates no evidence.
    assert pv.provenance is Provenance.PLACEHOLDER
    assert "diffusion_coefficient_m2_per_s" in pv.rationale

    # And the readiness report now says "present but unevidenced"
    # rather than "absent", which are different jobs.
    readiness = assess(c)
    assert "diffusion_coeff_um2_s" not in readiness.missing
    assert "diffusion_coeff_um2_s" in readiness.placeholder


def test_the_alias_conversion_factor_is_exact():
    from cognisom.library.parameters import PARAMETER_ALIASES

    by_legacy = {a.legacy: a for a in PARAMETER_ALIASES}
    # 1 m = 1e6 um, so 1 m^2 = 1e12 um^2. Not 1e6.
    assert by_legacy["diffusion_coefficient_m2_per_s"].factor == 1e12
    # ligand_kd_nm is a pure rename: both sides are nM.
    assert by_legacy["ligand_kd_nm"].factor == 1.0
    for alias in PARAMETER_ALIASES:
        assert alias.note, "a conversion factor must say why it is what it is"


def test_an_ambiguous_unit_is_deliberately_not_aliased():
    """oxygen_consumption_rate is NOT aliased to
    oxygen_consumption_fmol_per_cell_per_hour. Nothing in the codebase
    states what unit the former is in, so no factor can be asserted,
    and guessing one would silently rescale an oxygen sink term. This
    pins the omission as a decision rather than an oversight."""
    from cognisom.library.parameters import PARAMETER_ALIASES

    legacy = {a.legacy for a in PARAMETER_ALIASES}
    assert "oxygen_consumption_rate" not in legacy


def test_a_real_sidecar_still_wins_over_an_alias():
    """An alias is a last resort. A curated value must never be
    shadowed by a legacy scalar."""
    from cognisom.library.models import Cytokine
    from cognisom.library.parameters import (
        set_parameter, get_parameter, ParameterValue)

    c = Cytokine(name="IL-test2")
    c.physics_params = {"diffusion_coefficient_m2_per_s": 1e-10}
    set_parameter(c, "diffusion_coeff_um2_s", ParameterValue(
        value=42.0, unit="um^2/s", provenance=Provenance.MEASURED,
        citation="PMID:9829984", source="test"))

    pv = get_parameter(c, "diffusion_coeff_um2_s")
    assert pv.value == 42.0
    assert pv.provenance is Provenance.MEASURED


def test_unspecced_names_reports_the_uncontracted_remainder():
    """118 of 125 parameter names in use are declared by no spec, so
    they are written with no unit, range or applies_to check. This is
    the worklist; reporting it is not the same as fixing it."""
    from cognisom.library.parameters import unspecced_names, PARAMETER_SPECS
    from cognisom.library.models import Cytokine

    c = Cytokine(name="IL-test3")
    c.physics_params = {"half_life_hours": 4.0, "not_a_declared_name": 1.0}

    counts = unspecced_names([c])
    assert counts.get("not_a_declared_name") == 1
    assert "half_life_hours" not in counts
    assert "half_life_hours" in PARAMETER_SPECS


# ── The run summary must not launder a level ─────────────────────────

def _curation_of_all_fifteen():
    """apply_curation's verdict on every seeded adhesion molecule."""
    from cognisom.library.adhesion_kinetics import apply_curation
    from cognisom.library.seed_immunology import _seed_adhesion_molecules

    captured = []

    class _Collect:
        def add_entity(self, entity):
            captured.append(entity)
            return True

    _seed_adhesion_molecules(_Collect())
    return {e.name: apply_curation(e) for e in captured}


def test_an_estimated_value_is_counted_as_estimated_not_as_placeholder():
    """EntityCuration reports per-level buckets. Folding ESTIMATED into
    `placeholder` would understate a cited bound; folding it into
    `measured` would overstate it. Either way the summary line a
    migration prints would be a different claim from what was written.

    This is the same conflation the curation report had elsewhere
    (reporting a present-but-unevidenced value as "missing"), so it is
    worth pinning rather than trusting."""
    results = _curation_of_all_fifteen()
    assert len(results) == 15

    lsel = results["L-selectin (CD62L)"]
    assert "koff_per_s" in lsel.estimated
    assert "koff_per_s" not in lsel.placeholder
    assert "koff_per_s" not in lsel.measured
    assert "koff_per_s" not in lsel.derived
    # Its fitted Kd is still reported as measured.
    assert "kd_nm" in lsel.measured
    # And the per-entity description names the level, so a reader of
    # the migration output cannot miss it.
    assert "estimated" in lsel.describe()

    # No name may appear in two buckets at once.
    for name, r in results.items():
        buckets = [r.measured, r.derived, r.estimated, r.placeholder]
        flat = [p for b in buckets for p in b]
        assert len(flat) == len(set(flat)), name


def test_the_run_summary_reports_the_estimated_count_separately():
    from cognisom.library.adhesion_kinetics import CurationRun

    run = CurationRun(entities=list(_curation_of_all_fifteen().values()))
    assert run.estimated_parameters == 3
    # The four buckets partition every written parameter.
    assert (run.measured_parameters + run.derived_parameters
            + run.estimated_parameters + run.placeholder_parameters) == 75
    assert "estimated" in run.describe()


def test_a_bond_lifetime_inherits_the_grade_of_the_off_rate_it_came_from():
    """tau = 1/koff is a definition, so the lifetime is exactly as good
    as its input and no better. A lifetime derived from a bound must not
    be presented as publication-grade, and must not be thrown away as a
    placeholder either -- it came from a cited number."""
    results = _curation_of_all_fifteen()

    # A published bound in, a bound out.
    assert "bond_lifetime_seconds" in results["L-selectin (CD62L)"].estimated
    # A fitted off-rate in, a derivation out.
    assert "bond_lifetime_seconds" in results["P-selectin (CD62P)"].derived
    # No citation in, a placeholder out.
    assert "bond_lifetime_seconds" in results["MAC-1 (CD11b/CD18)"].placeholder
