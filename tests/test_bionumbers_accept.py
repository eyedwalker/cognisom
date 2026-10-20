"""What ``accept`` lets into the library, and what it must refuse.

``accept`` is the only way a BioNumbers record becomes a stored
MEASURED value, so it is the gate in front of the library's evidence.
Two things were wrong with it, and the second is the dangerous one.

**There were no specs for subcellular dimensions**, so
``unit=spec.unit if spec else record.unit`` fell to the record's own
unit and ``ParameterValue.validate(None)`` checked only that the unit
string was non-empty -- no unit consistency, no range. A surface area in
um^2 and a thickness in angstroms would both store cleanly against a
field meant to hold a micrometre length.

**And with a spec it was worse, not better**: the spec's unit was passed
straight through, which *relabelled* the record instead of checking it.
These are the records a sourcing pass over nucleus, mitochondrion,
ribosome, lysosome, Golgi, ER and proteasome actually surfaced:

    BNID 105820   39.5 angstrom   "Average thickness of hepatocyte
                                   Golgi membrane" -- a lipid bilayer,
                                   correct for what it measures and 760x
                                   too small for the organelle
    BNID 105910   30000 um^2      a surface AREA, where what was wanted
                                   was a tubule radius
    BNID 103061   1.26e6 A^3      a yeast 20S proteasome shell volume
                                   from a PhD thesis, uncitable

Relabelled rather than refused, the first becomes "39.5 um" -- a Golgi
apparatus ten thousand times too large, with nothing in the stored value
to show it. That is the failure these tests exist to prevent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.library import bionumbers as bn
from cognisom.library.models import BioEntity, EntityType
from cognisom.library.parameters import (PARAMETER_SPECS, Provenance,
                                         get_parameter)


def _record(bnid, value, unit, quantity, *, pmid="12345678", **kw):
    return bn.BioNumberRecord(
        bnid=bnid, quantity=quantity, value=value, unit=unit,
        organism=kw.get("organism", "Human Homo sapiens"),
        reference=kw.get("reference", "Someone A, Other B. A paper. J. 2000"),
        method=kw.get("method", ""), comments=kw.get("comments", ""),
        range_text=kw.get("range_text", ""),
        pmid=pmid, pmid_title=kw.get("title", "A paper"),
    )


def _cell():
    return BioEntity(name="HeLa cell", entity_type=EntityType.CELL_TYPE)


# The real records, as BioNumbers returns them.
GOLGI_BILAYER = _record("105820", 39.5, "Å",
                        "Average thickness of hepatocyte Golgi membrane")
ER_AREA = _record("105910", 30000.0, "µm^2",
                  "Surface area of ER of Leydig (secretory) cells")
NUCLEUS_HELA = _record("106665", 220.0, "µm^3", "Nuclear volume of HeLa cell")
#: Greek small mu, not the micro sign -- BioNumbers uses both, sometimes
#: in neighbouring records.
MITO_VOLUME = _record("109399", 0.29, "μm^3", "Mitochondria volume in HeLa cell")
LYSOSOME = _record("106078", 0.5, "µm",
                   "Lysosome diameter in human mammary epithelial cell at pH 7.4",
                   range_text="0.1 - 1.2 um")


# ── the specs now exist ──────────────────────────────────────────────

@pytest.mark.parametrize("name,unit", [
    ("nucleus_volume_um3", "um^3"),
    ("mitochondrion_volume_um3", "um^3"),
    ("lysosome_diameter_um", "um"),
    ("ribosome_diameter_um", "um"),
    ("er_surface_area_um2", "um^2"),
    ("golgi_extent_um", "um"),
])
def test_a_subcellular_dimension_declares_its_unit_and_range(name, unit):
    spec = PARAMETER_SPECS.get(name)
    assert spec is not None, f"{name} has no spec, so nothing checks its unit"
    assert spec.unit == unit
    assert spec.minimum is not None and spec.maximum is not None
    assert spec.minimum < spec.maximum
    assert EntityType.CELL_TYPE in spec.applies_to


def test_the_specs_record_the_trap_each_one_hides():
    """A range stops a wrong number; the description is what stops someone
    reaching for the wrong record in the first place."""
    assert "70S" in PARAMETER_SPECS["ribosome_diameter_um"].description
    assert "bilayer" in PARAMETER_SPECS["golgi_extent_um"].description
    assert "median" in PARAMETER_SPECS["lysosome_diameter_um"].description.lower()
    assert "HeLa" in PARAMETER_SPECS["nucleus_volume_um3"].description
    assert "Leydig" in PARAMETER_SPECS["er_surface_area_um2"].description
    assert "reticulum" in PARAMETER_SPECS["mitochondrion_volume_um3"].description


# ── units are compared, not overwritten ──────────────────────────────

def test_a_bilayer_thickness_cannot_become_an_organelle_size():
    """39.5 angstroms into a micrometre field. Relabelled, this is a Golgi
    apparatus 10,000x too large."""
    problems = bn.accept(_cell(), "golgi_extent_um", GOLGI_BILAYER)
    assert problems
    joined = " ".join(problems)
    assert "angstrom" in joined.lower() or "Å" in joined
    assert "um" in joined
    assert "105820" in joined


def test_a_surface_area_cannot_become_a_length():
    problems = bn.accept(_cell(), "golgi_extent_um", ER_AREA)
    assert problems and "um^2" in " ".join(problems).replace("µ", "u")


def test_the_refusal_says_how_to_do_it_properly():
    """A refusal that does not say what to do instead gets worked around."""
    joined = " ".join(bn.accept(_cell(), "golgi_extent_um", GOLGI_BILAYER))
    assert "value" in joined and "note" in joined
    assert "Average thickness" in joined, "the refusal should quote the quantity"


def test_a_deliberate_conversion_is_allowed_through():
    """Passing `value` is the caller saying it converted on purpose. That
    has to stay possible, or every record in a different unit is unusable."""
    cell = _cell()
    problems = bn.accept(cell, "golgi_extent_um", GOLGI_BILAYER,
                         value=0.00395, note="39.5 A converted to um")
    # The unit no longer blocks it; the RANGE does, and should: 0.00395 um
    # is still not a Golgi apparatus.
    assert problems and "below the plausible minimum" in " ".join(problems)

    # A converted value inside the range is accepted.
    ok = bn.accept(cell, "golgi_extent_um",
                   _record("999", 30000.0, "nm", "Golgi stack long axis"),
                   value=3.0, note="30000 nm = 3 um")
    assert ok == [], ok
    assert get_parameter(cell, "golgi_extent_um").value == pytest.approx(3.0)


def test_the_right_unit_in_either_spelling_is_accepted():
    """BioNumbers writes micrometres with the micro sign in one record and
    Greek small mu in the next, and exponents as carets or superscripts."""
    for record, parameter, want in (
            (NUCLEUS_HELA, "nucleus_volume_um3", 220.0),
            (MITO_VOLUME, "mitochondrion_volume_um3", 0.29),
            (LYSOSOME, "lysosome_diameter_um", 0.5),
            (ER_AREA, "er_surface_area_um2", 30000.0)):
        cell = _cell()
        assert bn.accept(cell, parameter, record) == [], (parameter, record.unit)
        stored = get_parameter(cell, parameter)
        assert stored.value == pytest.approx(want)
        assert stored.unit == PARAMETER_SPECS[parameter].unit


def test_same_unit_normalises_the_spellings_that_actually_occur():
    assert bn.same_unit("µm^3", "um^3")        # micro sign
    assert bn.same_unit("μm^3", "um^3")        # Greek small mu
    assert bn.same_unit("µm³", "um^3")    # superscript three
    assert bn.same_unit("Å", "angstrom")
    assert bn.same_unit(" UM ", "um")
    assert not bn.same_unit("nm", "um")
    assert not bn.same_unit("um^2", "um^3")
    assert not bn.same_unit("um/min", "um")


# ── what was already right, and must stay right ──────────────────────

def test_an_uncitable_record_is_refused_whatever_its_unit():
    """The yeast proteasome volume: a 2006 PhD thesis with no PubMed
    record. A sourcing pass still recommended a number from it."""
    thesis = bn.BioNumberRecord(
        bnid="103061", quantity="Shell volume of 20S proteasome",
        value=1.26e6, unit="Å^3", organism="Budding yeast",
        reference="Neil R. Voss. 2006 Geometric Studies of RNA and "
                  "Ribosomes. Table 3.4 pp. 97",
        pmid="", pmid_title="")
    assert not thesis.citable
    problems = bn.accept(_cell(), "nucleus_volume_um3", thesis)
    assert problems and "not citable" in problems[0]


def test_an_accepted_value_carries_its_paper_and_its_conditions():
    cell = _cell()
    assert bn.accept(cell, "lysosome_diameter_um", LYSOSOME,
                     note="median of the population") == []
    stored = get_parameter(cell, "lysosome_diameter_um")
    assert stored.provenance is Provenance.MEASURED
    assert stored.citation == "PMID:12345678"
    assert "106078" in stored.source
    # The spread is in the record and must travel with the median, or the
    # value reads as a size rather than a central tendency.
    assert "0.1 - 1.2" in stored.conditions
    assert "Human" in stored.conditions
    assert "median of the population" in stored.conditions


def test_a_value_outside_the_plausible_range_is_still_refused():
    """The range check was the only thing standing between a relabelled
    unit and the library. It has to keep working."""
    absurd = _record("777", 9e9, "µm^3", "Nuclear volume of something")
    problems = bn.accept(_cell(), "nucleus_volume_um3", absurd)
    assert problems and "above the plausible maximum" in " ".join(problems)


def test_a_record_with_no_unit_at_all_is_not_blocked_by_the_unit_check():
    """Some records parse a value and no unit. That is a different defect
    and validate() already reports it; the unit comparison must not turn
    it into a confusing message about a mismatch."""
    bare = _record("888", 220.0, "", "Nuclear volume")
    problems = bn.accept(_cell(), "nucleus_volume_um3", bare)
    assert not any("declared in" in p for p in problems)
