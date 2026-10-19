"""A measured constant still has to be a constant of the right thing.

ChEMBL's citation chain is shorter than BioNumbers' -- each measurement names
its document and the document carries a PubMed id outright -- which makes it
easy to attach a real reference to a number quickly, and therefore easy to
attach a real reference to the wrong number.

Two ways that happens, both pinned below. A target carries measurements of
many ligands: the androgen receptor has 38 Kd values, and they are affinities
of different compounds, mostly synthetic analogues from screening series, so a
value written without naming its ligand describes nothing. And a measurement
qualified by ``>`` or ``<`` is a bound rather than a value; recording one as
if it were measured is the quiet error this module is otherwise well placed
to make.
"""
from __future__ import annotations

import pytest

from cognisom.library.chembl_sourcing import Activity, accept
from cognisom.library.models import Receptor
from cognisom.library.parameters import Provenance, get_parameter


def _activity(**kw):
    base = dict(
        target_chembl_id="CHEMBL1871", molecule_chembl_id="CHEMBL13024",
        standard_type="Kd", value=0.61, units="nM", relation="=",
        assay_description="Dissociation constant for [3H]R-1881 binding to "
                          "human androgen receptor",
        document_chembl_id="CHEMBL1140243", organism="Homo sapiens",
        pmid="15771445", doi="10.1/x", title="Androgen receptor binding",
        journal="J Med Chem", year=2005, molecule_name="METRIBOLONE",
    )
    base.update(kw)
    return Activity(**base)


# ── a bound is not a value ───────────────────────────────────────────────

@pytest.mark.parametrize("relation,exact", [
    ("=", True), ("~", True), (">", False), ("<", False), (">=", False),
])
def test_only_an_equality_counts_as_a_measurement(relation, exact):
    assert _activity(relation=relation).is_exact is exact


def test_a_bound_is_refused_even_with_a_pmid():
    """">10000 nM" means the assay could not measure it, not that it is 10000."""
    receptor = Receptor(name="AR")

    problems = accept(receptor, "kd_nm", _activity(relation=">", value=10000.0))

    assert problems and "bound" in problems[0]
    assert get_parameter(receptor, "kd_nm") is None


def test_a_measurement_with_no_pubmed_id_is_refused():
    receptor = Receptor(name="AR")

    problems = accept(receptor, "kd_nm", _activity(pmid=""))

    assert problems and "PubMed" in problems[0]


def test_an_activity_with_no_value_is_refused():
    receptor = Receptor(name="AR")

    problems = accept(receptor, "kd_nm", _activity(value=None))

    assert problems


# ── the ligand must travel with the constant ─────────────────────────────

def test_an_accepted_constant_names_the_ligand_it_was_measured_against():
    """A dissociation constant without its ligand describes nothing."""
    receptor = Receptor(name="AR")

    problems = accept(receptor, "kd_nm", _activity())

    assert problems == []
    stored = get_parameter(receptor, "kd_nm")
    assert stored.value == pytest.approx(0.61)
    assert stored.provenance is Provenance.MEASURED
    assert stored.citation == "PMID:15771445"
    assert "METRIBOLONE" in stored.conditions
    assert "CHEMBL1871" in stored.source


def test_an_unnamed_compound_still_records_its_identifier():
    receptor = Receptor(name="AR")

    accept(receptor, "kd_nm", _activity(molecule_name="",
                                        molecule_chembl_id="CHEMBL89136"))

    assert "CHEMBL89136" in get_parameter(receptor, "kd_nm").conditions


def test_a_note_is_carried_into_the_conditions():
    """Where the chosen ligand is a proxy, the reasoning has to be visible."""
    receptor = Receptor(name="AR")

    accept(receptor, "kd_nm", _activity(),
           note="R1881 is the standard tracer; DHT is comparable")

    assert "standard tracer" in get_parameter(receptor, "kd_nm").conditions


def test_the_spec_still_validates_an_accepted_value():
    receptor = Receptor(name="AR")

    problems = accept(receptor, "kd_nm", _activity(value=-5.0))

    assert problems, "a negative dissociation constant was accepted"


def test_citable_requires_a_value_an_equality_and_a_paper():
    assert _activity().citable
    assert not _activity(relation=">").citable
    assert not _activity(pmid="").citable
    assert not _activity(value=None).citable
