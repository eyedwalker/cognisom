"""Sourcing a number from a database must not source the wrong molecule.

This module converts placeholders into evidence, which makes its failure mode
worse than the problem it solves: a value with a real citation attached to the
wrong protein survives review in a way an obvious placeholder never does.

Two checks carry that weight. The accession is confirmed against an
independent source -- the cached AlphaFold model's residue count must match
UniProt's sequence length, and a mismatch refuses the write. And the mature
chain is distinguished from the translated precursor, because for a secreted
protein those are different quantities by a wide margin: EGF's precursor is
1207 residues and 134 kDa against a mature chain of 53 residues and 6 kDa.

Everything here runs offline against fixture records shaped like UniProt's.
"""
from __future__ import annotations

import pytest

from cognisom.library.models import Cytokine, EntityType, Protein
from cognisom.library.parameters import (
    Provenance, ParameterValue, get_parameter, set_parameter,
)
from cognisom.library.uniprot_sourcing import (
    DIFFUSION_VALIDATION, SourcingRefused, derive_diffusion,
    diffusion_in_water_m2_s, hydrodynamic_radius_nm, mature_chain,
    sequence_mass_da,
)

#: Insulin A chain, whose average mass is a published, checkable number.
INSULIN_A = "GIVEQCCTSICSLYQLENYCN"


def _record(sequence: str, features=()):
    return {
        "sequence": {"value": sequence, "length": len(sequence)},
        "features": [
            {"type": t, "description": d,
             "location": {"start": {"value": s}, "end": {"value": e}}}
            for t, s, e, d in features
        ],
    }


# ── the mass arithmetic ──────────────────────────────────────────────────

def test_the_mass_of_a_known_peptide_is_right():
    """Insulin A chain is 2383.7 Da by average residue mass."""
    assert sequence_mass_da(INSULIN_A) == pytest.approx(2383.7, abs=1.0)


def test_mass_scales_with_the_sequence():
    assert sequence_mass_da("AA") > sequence_mass_da("A")
    assert sequence_mass_da("") == pytest.approx(18.015, abs=0.01)


# ── precursor versus mature chain ────────────────────────────────────────

def test_a_record_with_no_chain_feature_yields_the_whole_sequence():
    """Correct for an intracellular protein that is never processed."""
    start, end, description, chain = mature_chain(_record("MKVLA"))

    assert (start, end) == (1, 5)
    assert chain == "MKVLA"
    assert "no chain feature" in description


def test_a_single_chain_is_taken_however_it_is_described():
    """The fragment filter must only break ties, never veto a lone chain.

    Several real rows -- VISTA, TIGIT, MDA5 -- have one chain whose
    description contains the word "domain". Filtering those out left no
    candidate and wrongly refused the whole entity.
    """
    rec = _record("MKVLAWTY", [("Chain", 2, 7, "V-type immunoglobulin domain-containing protein")])

    start, end, _, chain = mature_chain(rec)

    assert (start, end) == (2, 7)
    assert chain == "KVLAWT"


def test_the_mature_chain_is_preferred_over_the_precursor():
    """The EGF shape: a pro-form chain and the product nested inside it."""
    rec = _record("A" * 60, [
        ("Signal", 1, 5, ""),
        ("Chain", 6, 60, "Pro-epidermal growth factor"),
        ("Chain", 40, 52, "Epidermal growth factor"),
    ])

    start, end, description, chain = mature_chain(rec)

    assert (start, end) == (40, 52), (
        "the precursor was returned; for a secreted protein that is the "
        "wrong quantity, and for EGF it is wrong by twenty-two fold")
    assert description == "Epidermal growth factor"
    assert len(chain) == 13


def test_a_genuinely_multi_chain_record_is_refused():
    """Caspase p17/p12, complement C3's twelve chains, TNF's two forms.

    Choosing among them is a modelling decision. Refusing is the honest
    outcome and leaves the row on the worklist.
    """
    rec = _record("A" * 40, [
        ("Chain", 1, 20, "Hepatocyte growth factor alpha chain"),
        ("Chain", 21, 40, "Hepatocyte growth factor beta chain"),
    ])

    with pytest.raises(SourcingRefused) as exc:
        mature_chain(rec)

    assert "modelling decision" in str(exc.value)


# ── the derived diffusion coefficient ────────────────────────────────────

def test_the_hydrodynamic_radius_grows_with_mass():
    assert hydrodynamic_radius_nm(100) > hydrodynamic_radius_nm(10)
    # A 15 kDa globular protein sits near 2 nm.
    assert 1.5 < hydrodynamic_radius_nm(15) < 2.5


@pytest.mark.parametrize("name,kda,published", [
    ("lysozyme", 14.3, 1.06e-10),
    ("ribonuclease A", 13.7, 1.19e-10),
    ("ovalbumin", 45.0, 7.76e-11),
    ("serum albumin", 66.5, 6.1e-11),
    ("catalase", 250.0, 4.1e-11),
])
def test_the_relation_reproduces_published_diffusion(name, kda, published):
    """Validated at the temperature those values were measured at, 20 C.

    The relation runs consistently low, which is stated in the rationale
    rather than corrected away on five points.
    """
    computed = diffusion_in_water_m2_s(kda, temperature_k=293.15,
                                       viscosity_pa_s=1.002e-3)

    assert 0.7 < computed / published < 1.0, (
        f"{name}: computed {computed:.3g} against published {published:.3g}")


def test_the_validation_is_recorded_for_whoever_reads_the_value():
    assert "0.78" in DIFFUSION_VALIDATION and "0.83" in DIFFUSION_VALIDATION


def test_diffusion_is_not_derived_from_a_placeholder_mass():
    """The trap this module was written to avoid.

    Deriving from an unevidenced number launders a default into a
    derivation, which is harder to spot than the default was.
    """
    cytokine = Cytokine(name="test cytokine")
    cytokine.physics_params["molecular_weight_kda"] = 20.0   # no provenance

    assert derive_diffusion(cytokine, write=False) is None


def test_diffusion_is_derived_from_an_evidenced_mass():
    cytokine = Cytokine(name="test cytokine")
    set_parameter(cytokine, "molecular_weight_kda", ParameterValue(
        value=20.0, unit="kDa", provenance=Provenance.MEASURED,
        citation="UniProt:P00000", source="fixture"))

    value = derive_diffusion(cytokine, write=True)

    assert value is not None
    stored = get_parameter(cytokine, "diffusion_coeff_um2_s")
    assert stored.provenance is Provenance.DERIVED
    assert stored.citation == "UniProt:P00000", (
        "the derivation must carry the citation of the measurement it rests on")
    assert "tissue is lower" in stored.conditions, (
        "a dilute-aqueous value used as a tissue diffusivity is wrong by "
        "two- to threefold; the caveat has to travel with the number")


# ── what the library actually holds now ──────────────────────────────────

@pytest.fixture(scope="module")
def entities():
    import contextlib
    import io
    from cognisom.library.store import EntityStore
    with contextlib.redirect_stdout(io.StringIO()):
        store = EntityStore()
    ids = [r[0] for r in
           store._conn.execute("SELECT entity_id FROM entities").fetchall()]
    return [e for e in (store.get_entity(i) for i in ids) if e]


def test_sourced_masses_carry_a_uniprot_accession(entities):
    sourced = [(e, get_parameter(e, "molecular_weight_kda")) for e in entities]
    sourced = [(e, pv) for e, pv in sourced
               if pv is not None and pv.provenance is Provenance.MEASURED]

    assert len(sourced) >= 100, f"only {len(sourced)} masses are evidenced"
    for entity, pv in sourced:
        assert pv.citation.startswith("UniProt:"), (
            f"{entity.name} has a measured mass with citation {pv.citation!r}")
        assert pv.conditions, f"{entity.name} records no conditions"


def test_the_mass_and_the_length_describe_the_same_molecule(entities):
    """Both come from one record, so they must agree on the chain."""
    for entity in entities:
        mass = get_parameter(entity, "molecular_weight_kda")
        length = get_parameter(entity, "sequence_length_aa")
        if not (mass and length and mass.provenance is Provenance.MEASURED):
            continue
        assert mass.citation == length.citation
        # Average residue mass is 110 Da, so the ratio is bounded.
        implied = mass.value * 1000.0 / length.value
        assert 90 < implied < 135, (
            f"{entity.name}: {mass.value} kDa over {length.value} aa implies "
            f"{implied:.0f} Da per residue, which is not a protein")
