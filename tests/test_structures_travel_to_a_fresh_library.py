"""The chemical structures reach a library that has never seen the network.

`structures.backfill_structures` resolves a SMILES for every small molecule
from PubChem and verifies it against the entity's own formula and weight. It
made that pass RE-RUNNABLE, which its docstring argues for at length -- "a
sourcing pass that cannot be re-run is not a source" -- and that was half the
problem. The other half is that the pass needs the network, the database is
local state, and nothing carried the result between machines.

So the twenty structures lived on one laptop. Everywhere else the library
built itself from the seeds, which hold exactly ONE SMILES literal, and
`tests/test_molecular_usd.py` -- which asserts twenty -- failed. It had been
failing on `main` for two nights and in every CI run of the merge that found
it, with the message "only 1 entities carry a SMILES", which names the symptom
and not the cause.

`structures.json` beside the module is the transport, the same shape as
`curated_parameters.json`: the pass runs once, its result is committed, and
`EntityStore` applies it on open. The formula and weight in that file are
RECOMPUTED from each SMILES rather than copied from the record, so they are
something `apply_structures` can check the string against -- a SMILES that no
longer produces its recorded formula is a different molecule, and nobody reads
a SMILES and notices. Enzalutamide is the cautionary case the module was
written around: the shipped library once held a structure under that name
whose formula was C28H24F6N4O2 against the real C21H16F4N4O2S.
"""

from __future__ import annotations

import contextlib
import io
import json

import pytest

from cognisom.library import structures as st


@pytest.fixture(scope="module")
def committed():
    assert st.STRUCTURES_PATH.exists(), (
        f"{st.STRUCTURES_PATH} is missing; regenerate it with "
        "`python -m cognisom.library.structures export`")
    return json.loads(st.STRUCTURES_PATH.read_text())["entities"]


def test_the_committed_file_carries_the_structures_the_export_needs(committed):
    # test_molecular_usd asserts >= 20; this is where they have to come from.
    assert len(committed) >= 20, (
        f"only {len(committed)} structures are committed; the molecular USD "
        "export needs 20 and a fresh library has no other source for them")


def test_every_committed_smiles_still_makes_the_molecule_it_claims(committed):
    """The check that makes the file safe to apply unread."""
    wrong = []
    for name, structure in sorted(committed.items()):
        check = st.validate_structure(
            structure["smiles"],
            expected_formula=structure.get("molecular_formula", ""),
            expected_mw=float(structure.get("molecular_weight") or 0) or None)
        if not check.ok:
            wrong.append(f"{name}: {'; '.join(check.problems)}")
    assert not wrong, wrong


def test_enzalutamide_is_the_molecule_and_not_the_one_that_was_there_before(
        committed):
    """The specific record this module exists because of."""
    enza = committed["Enzalutamide"]
    assert enza["molecular_formula"] == "C21H16F4N4O2S"
    assert enza["inchikey"].startswith("WXCXUHSOUPDCQV")


def test_a_library_that_never_saw_the_network_has_them(tmp_path, monkeypatch):
    """The whole point: a fresh database, no PubChem, twenty structures.

    Builds a real library in a temporary data directory -- seeds and all --
    and counts what carries a structure. Before `structures.json` this was 1.
    """
    monkeypatch.setenv("COGNISOM_DATA_DIR", str(tmp_path))

    def refuse(*_args, **_kwargs):      # any network call is a failure here
        raise AssertionError("a fresh library must not need PubChem")

    monkeypatch.setattr(st, "resolve_small_molecule", refuse)

    from cognisom.library.store import EntityStore
    with contextlib.redirect_stdout(io.StringIO()):
        store = EntityStore()
    entities, _ = store.search(limit=10000)
    with_smiles = [e.name for e in entities if (getattr(e, "smiles", "") or "")]
    assert len(with_smiles) >= 20, (
        f"a fresh library carries {len(with_smiles)} structures: "
        f"{sorted(with_smiles)}")
    assert (tmp_path / "library" / "entities.db").exists(), (
        "the fixture read an existing library instead of building one")


def test_applying_refuses_a_structure_that_does_not_match_its_formula(tmp_path):
    """A damaged or edited file cannot quietly change which molecule an
    entity means -- the same guard the PubChem path applies to its answers."""
    import copy

    payload = {"entities": copy.deepcopy(
        json.loads(st.STRUCTURES_PATH.read_text())["entities"])}
    # Aspirin's SMILES under enzalutamide's recorded formula.
    payload["entities"]["Enzalutamide"]["smiles"] = "CC(=O)OC1=CC=CC=C1C(=O)O"
    path = tmp_path / "structures.json"
    path.write_text(json.dumps(payload))

    class _Store:
        def __init__(self):
            self.written = []

        def find_entity_by_name(self, name):
            return type("E", (), {"name": name, "smiles": "",
                                  "properties": {}})()

        def update_entity(self, entity, changed_by=""):
            self.written.append(entity.name)

    store = _Store()
    report = st.apply_structures(store, path)
    assert "Enzalutamide" not in store.written
    assert any(r.startswith("Enzalutamide:") for r in report["refused"]), report


def test_an_existing_structure_is_not_overwritten(tmp_path):
    """A library that resolved its own is left alone."""
    class _Store:
        def __init__(self):
            self.written = []

        def find_entity_by_name(self, name):
            return type("E", (), {"name": name, "smiles": "C", "properties": {}})()

        def update_entity(self, entity, changed_by=""):
            self.written.append(entity.name)

    store = _Store()
    report = st.apply_structures(store, st.STRUCTURES_PATH)
    assert not store.written
    assert len(report["already"]) >= 20


def test_the_export_is_current(tmp_path):
    """The committed file is what this library would write today."""
    import contextlib as _c
    import io as _i

    from cognisom.library.store import EntityStore
    with _c.redirect_stdout(_i.StringIO()):
        store = EntityStore()
    out = tmp_path / "structures.json"
    st.export_structures(store, out)
    fresh = json.loads(out.read_text())["entities"]
    committed = json.loads(st.STRUCTURES_PATH.read_text())["entities"]
    assert fresh == committed, (
        "structures.json is stale; regenerate it with "
        "`python -m cognisom.library.structures export`")
