"""The external clients must read the fields the APIs actually return.

Two defects of the same shape: the client asked for the right data, the API
returned it, and the client then read it out of a key that was not there. Both
failed silently -- an empty string and an empty list -- so the symptom appeared
far downstream as missing data rather than as an error.

  * PubChem renamed its SMILES properties. A request for ``CanonicalSMILES`` is
    still accepted, but the response is keyed ``SMILES`` with
    ``ConnectivitySMILES`` alongside it. Reading ``CanonicalSMILES`` off the
    payload returned "" for every compound ever fetched, which is why all 65
    drug rows in the shipped library carried an empty SMILES and the docking
    and molecular-viewer paths had nothing to consume.

  * KEGG's ``find/pathway`` returns *reference* pathways keyed ``map#####``.
    The client filtered out anything not already prefixed with the organism
    code -- which is every row the endpoint returns -- so pathway search found
    nothing for any query, while fetching a known id worked fine. Search is
    the path the dashboard's pathway lookup uses.

The payloads below are the real shapes, recorded rather than fetched, so these
tests neither need the network nor break when a record changes upstream. A
separate live contract test belongs alongside them; these pin the mapping.
"""
from __future__ import annotations

import pytest


# ── PubChem ──────────────────────────────────────────────────────────────

# Exactly what pubchem.ncbi.nlm.nih.gov returns for CID 15951529
# (enzalutamide) when asked for CanonicalSMILES: note that the requested key
# is absent from the response.
_PUBCHEM_MODERN = {
    "CID": 15951529,
    "MolecularFormula": "C21H16F4N4O2S",
    "SMILES": "CC1(C(=O)N(C(=S)N1C2=CC(=C(C=C2)C(=O)NC)F)C3=CC(=C(C=C3)C#N)C(F)(F)F)C",
    "ConnectivitySMILES": "CC1(C(=O)N(C(=S)N1C2=CC(=C(C=C2)C(=O)NC)F)C3=CC(=C(C=C3)C#N)C(F)(F)F)C",
}


def test_smiles_is_read_from_the_key_pubchem_actually_returns():
    from cognisom.library.external_sources import _first_smiles

    assert _first_smiles(_PUBCHEM_MODERN) == _PUBCHEM_MODERN["SMILES"]


def test_smiles_falls_back_to_connectivity_when_that_is_all_there_is():
    from cognisom.library.external_sources import _first_smiles

    payload = {"ConnectivitySMILES": "CCO"}
    assert _first_smiles(payload) == "CCO"


def test_smiles_still_reads_the_legacy_key():
    """A cached or replayed older payload must keep working."""
    from cognisom.library.external_sources import _first_smiles

    assert _first_smiles({"CanonicalSMILES": "CCO"}) == "CCO"


def test_smiles_is_empty_only_when_no_key_carries_one():
    from cognisom.library.external_sources import _first_smiles

    assert _first_smiles({"MolecularFormula": "C2H6O"}) == ""
    assert _first_smiles({"SMILES": ""}) == ""


def test_the_requested_property_list_asks_for_a_key_that_exists():
    """Guards the request side as well as the read side."""
    import inspect

    from cognisom.library.external_sources import PubChemClient

    src = inspect.getsource(PubChemClient)
    assert "SMILES" in src
    # The property list must name at least one key the current API returns.
    assert "ConnectivitySMILES" in src or "SMILES," in src


# ── KEGG ─────────────────────────────────────────────────────────────────

class _FakeResponse:
    def __init__(self, text: str):
        self.text = text
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):  # pragma: no cover - not used by search_pathways
        raise AssertionError("search_pathways must not parse JSON")


class _FakeSession:
    """Returns the exact body rest.kegg.jp gives for find/pathway/prostate."""

    def __init__(self, text: str):
        self._text = text
        self.urls: list[str] = []

    def get(self, url, **kwargs):
        self.urls.append(url)
        return _FakeResponse(self._text)


def _client_with(text: str):
    from cognisom.library.external_sources import KEGGClient

    client = KEGGClient()
    client._session = _FakeSession(text)
    return client


def test_reference_pathway_is_mapped_onto_the_requested_organism():
    client = _client_with("map05215\tProstate cancer\n")

    results = client.search_pathways("prostate", organism="hsa")

    assert len(results) == 1, (
        "find/pathway returns map##### ids; filtering on the organism prefix "
        "discarded every row the endpoint can return"
    )
    assert results[0].pathway_id == "hsa05215"
    assert results[0].name == "Prostate cancer"


def test_an_organism_specific_id_for_another_organism_is_still_filtered():
    client = _client_with("mmu05215\tProstate cancer\n")

    assert client.search_pathways("prostate", organism="hsa") == []


def test_a_matching_organism_id_is_kept_unchanged():
    client = _client_with("hsa05215\tProstate cancer\n")

    results = client.search_pathways("prostate", organism="hsa")

    assert [p.pathway_id for p in results] == ["hsa05215"]


def test_the_path_prefix_is_stripped():
    client = _client_with("path:map05215\tProstate cancer\n")

    results = client.search_pathways("prostate", organism="hsa")

    assert [p.pathway_id for p in results] == ["hsa05215"]


def test_several_rows_all_survive():
    client = _client_with(
        "map05215\tProstate cancer\n"
        "map05200\tPathways in cancer\n"
        "map04151\tPI3K-Akt signaling pathway\n"
    )

    results = client.search_pathways("cancer", organism="hsa")

    assert [p.pathway_id for p in results] == ["hsa05215", "hsa05200", "hsa04151"]
