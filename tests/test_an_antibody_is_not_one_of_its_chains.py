"""Eight MEASURED masses that were each one piece of the molecule.

Every antibody isotype in this library took its ``molecular_weight_kda``
from an IGH* accession, which translates to ONE heavy-chain constant
region: no VH, no light chain, no multimer, no J chain, no secretory
component. Graded MEASURED, so each asserted that somebody had checked.

    IgM (pentameric)   51.9 kDa stored   ~970 real   19x
    IgA1 (secretory)   42.8               ~385        9x
    IgG1               43.9               ~146        3x

The two worst are the two whose entity NAMES promise the assembly.

``molecular_weight_kda`` has a floor of zero because a cytokine may
weigh 8 kDa, so the range guard could never refuse a value for being too
small. The guard is the name: the chain mass is real and is kept as
``constant_region_mass_kda``, and the assembled mass is left absent
because this library has no source for it and will not invent one.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ISOTYPES = ("IgA1 (secretory)", "IgD", "IgE", "IgG1", "IgG2", "IgG3",
            "IgG4", "IgM (pentameric)")


@pytest.fixture(scope="module")
def store():
    with contextlib.redirect_stdout(io.StringIO()):
        from cognisom.library.store import EntityStore
        return EntityStore()


@pytest.mark.parametrize("name", ISOTYPES)
def test_no_isotype_claims_a_chain_mass_as_its_own(store, name):
    from cognisom.library.parameters import get_parameter

    entity = store.find_entity_by_name(name)
    assert entity is not None, name
    assert get_parameter(entity, "molecular_weight_kda") is None, (
        f"{name} is back to asserting it weighs one constant region")


@pytest.mark.parametrize("name", ISOTYPES)
def test_the_chain_mass_is_kept_under_a_name_that_says_so(store, name):
    """The number is a real UniProt measurement. Deleting it would lose a
    fact; the defect was only ever what it was called."""
    from cognisom.library.parameters import Provenance, get_parameter

    pv = get_parameter(store.find_entity_by_name(name),
                       "constant_region_mass_kda")
    assert pv is not None and 30.0 < pv.value < 70.0
    assert pv.provenance is Provenance.MEASURED
    assert pv.citation.startswith("UniProt:")


def test_the_absent_assembled_mass_says_why_it_is_absent(store):
    """A missing value that nobody explained reads as "not got to yet".
    This one is a decision."""
    entity = store.find_entity_by_name("IgM (pentameric)")
    note = (getattr(entity, "sourced", None) or {}).get("assembled_mass", "")
    assert "970" in note and "not invent" in note


def test_the_source_cannot_write_it_again(store):
    """Fixing the data without fixing the sourcing means the next pass
    puts it back."""
    text = (REPO_ROOT / "cognisom" / "library"
            / "uniprot_sourcing.py").read_text()
    assert "constant_region_mass_kda" in text
    assert "EntityType.ANTIBODY" in text


def test_the_spec_can_refuse_an_implausible_chain_mass():
    """molecular_weight_kda spans 0 to 1e5 because it covers cytokines
    and whole proteins. A spec for one CH region does not have to."""
    from cognisom.library.parameters import PARAMETER_SPECS

    spec = PARAMETER_SPECS["constant_region_mass_kda"]
    assert spec.minimum == 5.0 and spec.maximum == 200.0


# ── retraction: the correction has to reach a deployed library ───────

def test_a_wrong_value_is_removed_from_a_library_that_already_has_it():
    """apply_curated is additive by design: a value absent from the file
    usually means the exporting machine had not sourced it, not that it
    is wrong. So dropping a bad record from the export does nothing for
    production, which is never rebuilt. RETRACTED is the way back."""
    from cognisom.library import curated
    from cognisom.library.parameters import (ParameterValue, Provenance,
                                             get_parameter, set_parameter)
    from cognisom.library.store import EntityStore

    with tempfile.TemporaryDirectory() as tmp:
        with contextlib.redirect_stdout(io.StringIO()):
            store = EntityStore(db_path=str(Path(tmp) / "deployed.db"))

        entity = store.find_entity_by_name("IgM (pentameric)")
        set_parameter(entity, "molecular_weight_kda", ParameterValue(
            value=51.92, unit="kDa", provenance=Provenance.MEASURED,
            citation="UniProt:P01871"))
        store.update_entity(entity, changed_by="test")
        assert get_parameter(store.find_entity_by_name("IgM (pentameric)"),
                             "molecular_weight_kda") is not None

        report = curated.apply_curated(store, force=True)
        assert report.retracted >= 1
        assert get_parameter(store.find_entity_by_name("IgM (pentameric)"),
                             "molecular_weight_kda") is None


def test_the_retraction_travels_in_the_file():
    """A deployed library reads the file, not this module."""
    from cognisom.library.curated import CURATED_PATH

    document = json.loads(CURATED_PATH.read_text())
    retracted = document.get("retracted") or []
    names = {r["name"] for r in retracted
             if r["parameter"] == "molecular_weight_kda"}
    assert set(ISOTYPES) <= names
    for row in retracted:
        assert row.get("why"), f"{row['name']} is retracted without a reason"


def test_retracting_is_idempotent():
    from cognisom.library import curated
    from cognisom.library.store import EntityStore

    with tempfile.TemporaryDirectory() as tmp:
        with contextlib.redirect_stdout(io.StringIO()):
            store = EntityStore(db_path=str(Path(tmp) / "twice.db"))
        curated.apply_curated(store, force=True)
        again = curated.apply_curated(store, force=True)
        assert again.retracted == 0, (
            "a second pass retracted something that was already gone")
