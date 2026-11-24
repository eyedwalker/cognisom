"""The AR pathway's rates are declared, and the declaration is checked
against the pathway that actually runs.

``reaction_builder.build_ar_signaling`` is the one AR pathway the engine
has: ARSignalingModule builds it on every run, so these rates produce
the signalling fraction that gates division, the androgen-withdrawal
hazard, and the PSA the dashboard presents. Nine of them were inline
literals with no provenance, in a pathway with two externally tested
couplings hanging off it -- and not in the entity library, so the
library's own curation report could not see them either.

The risk with a file of provenance labels is that it becomes
decoration: the labels drift from the numbers and nobody notices. So
the test that matters here is the one comparing every declared value
against the value the built system carries. If they disagree, the
labels describe a pathway that is not running.

Nothing is marked 'sourced', deliberately. The 2026-10-17 audit had 73%
of agent-supplied citations fail independent re-retrieval, so a
citation added without someone reading the paper would be worse than
the honest ESTIMATE it replaced. A test below enforces that anything
claiming 'sourced' carries a real reference.
"""

from __future__ import annotations

import math

import pytest

from cognisom.genomics import ar_pathway_parameters as arp


def test_every_parameter_declares_a_status_and_a_source():
    assert arp.ALL, "no parameters declared"
    for p in arp.ALL:
        assert p.status in arp.STATUSES, (p.name, p.status)
        assert p.source.strip(), p.name
        assert p.unit.strip(), p.name


def test_a_bad_status_or_empty_source_is_refused():
    """NON-VACUITY for the test above: the validation has to bite."""
    with pytest.raises(ValueError):
        arp.PathwayParameter("x", 1.0, "/h", "guess", "something")
    with pytest.raises(ValueError):
        arp.PathwayParameter("x", 1.0, "/h", arp.ESTIMATE, "")


def test_the_half_life_translation_is_the_right_arithmetic():
    """The whole point of `implies` is that a reader can check the
    number without reading the ODE, so the arithmetic must be right."""
    p = arp.PathwayParameter("t", 0.2, "/h", arp.ESTIMATE, "test")
    assert p.half_life_hours == pytest.approx(math.log(2) / 0.2)
    assert p.half_life_hours == pytest.approx(3.4657, abs=1e-3)
    # not a first-order rate -> no half-life claimed
    assert arp.K_PSA_TRANSLATION.half_life_hours is None
    assert "implies" not in arp.K_PSA_TRANSLATION.as_dict()
    assert "half-life" in str(arp.K_PSA_DEGRADATION.as_dict()["implies"])


def test_nothing_claims_to_be_sourced_without_a_reference():
    """Guard for the future. An ESTIMATE promoted to 'sourced' has to
    carry something retrievable -- a PMID, or an author and a year --
    because this project has already been burned by citations that
    could not be re-retrieved."""
    for p in arp.ALL:
        if p.status == arp.SOURCED:
            has_pmid = "PMID" in p.source
            has_year = any(str(y) in p.source for y in range(1950, 2040))
            assert has_pmid or has_year, (
                f"{p.name} claims to be sourced but its source text "
                f"names no PMID and no year: {p.source!r}")


def test_the_psa_rate_keeps_the_serum_distinction():
    """The most inviting mistake in this file. Clinical serum PSA
    half-life (~2-3 days) is CLEARANCE from blood; k_psa_deg is
    degradation of an intracellular pool. Substituting one for the
    other is the category error this project keeps cataloguing, and the
    warning has to stay attached to the number."""
    note = arp.K_PSA_DEGRADATION.note
    assert "serum" in note.lower()
    assert "clearance" in note.lower()
    assert "undefined" in note.lower() or "not" in note.lower()


# ── the test that stops this becoming decoration ───────────────────


@pytest.mark.slow
def test_every_declared_value_is_the_value_the_pathway_runs_on():
    """Compare the declarations against the built ODE system.

    This is what makes the labels mean something. A declared 0.2 beside
    a running 0.05 would be worse than no label at all, because a
    reader would trust it.
    """
    pytest.importorskip("scipy")
    from cognisom.genomics.reaction_builder import library_ar_system

    system = library_ar_system()
    running = dict(system.parameters)

    # the nine literals plus DHT_input, by their key in the system
    declared = {p.name: p.value for p in arp.ALL
                if "(fallback)" not in p.name}
    assert declared, "nothing to check"

    missing = [k for k in declared if k not in running]
    assert not missing, (
        f"declared rates that the pathway does not have: {missing}. "
        f"the system's keys are {sorted(running)}")

    for key, value in sorted(declared.items()):
        assert running[key] == pytest.approx(value, rel=1e-12), (
            f"{key}: declared {value}, the pathway runs {running[key]}. "
            "The provenance file and the pathway have drifted; fix the "
            "file to match the code, or the label is a false claim.")


@pytest.mark.slow
def test_the_pathway_has_no_undeclared_rate_left():
    """Any rate in the system that nothing here describes is a number
    with no provenance again -- which is the state this file exists to
    end. Rates fed from the entity library are exempt: they carry their
    provenance there."""
    pytest.importorskip("scipy")
    from cognisom.genomics.reaction_builder import library_ar_system

    system = library_ar_system()
    declared = {p.name for p in arp.ALL}
    # Fed from the entity library, or sourced under their own name.
    from_library = {"k_ar_tx", "k_ar_m_deg", "k_bind", "k_unbind", "K_psa"}

    undeclared = sorted(set(system.parameters) - declared - from_library)
    assert not undeclared, (
        f"AR-pathway rates with no provenance anywhere: {undeclared}. "
        "Declare each in ar_pathway_parameters, with what its value "
        "implies biologically, or source it in the entity library.")
