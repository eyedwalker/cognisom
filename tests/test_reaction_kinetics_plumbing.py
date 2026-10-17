"""The library's kinetics must be the kinetics the solver runs.

``build_from_entity_names`` chose a turnover number with
``pp.get("kinase_kcat_per_s", 2.0)``. That is a literal-key read, and the
library writes kcat under six names across two time units, so every entity
storing it as ``kcat_per_s``, ``catalytic_kcat_per_s``,
``protease_kcat_per_s`` or either ``_per_min`` variant fell silently to 2.0
while the library went on displaying something else. Nine of the thirteen
entities with an ``activates`` edge diverged that way. EZH2 was the worst:
a stored 0.025 /s against a used 2.0, which makes a slow methyltransferase
turn over as fast as everything else in the network.

Which of the two numbers is right is a separate question, and mostly the
answer is "neither is evidenced". The defect here is the divergence: the
library shows a number a reader will quote, the solver integrates a
different one, and nothing says so. The Michaelis constant was worse -- it
was never read at all, and a dissociation constant was substituted for it.
"""
from __future__ import annotations

import contextlib
import io
import logging

import pytest

from cognisom.genomics.reaction_builder import (
    DEFAULT_KM_UM, DEFAULT_VMAX_PER_S, _km_um, _vmax_per_s,
)
from cognisom.library.parameters import get_parameter


@pytest.fixture(scope="module")
def entities():
    logging.disable(logging.WARNING)
    from cognisom.library.store import EntityStore
    with contextlib.redirect_stdout(io.StringIO()):
        store = EntityStore()
    ids = [r[0] for r in
           store._conn.execute("SELECT entity_id FROM entities").fetchall()]
    yield [e for e in (store.get_entity(i) for i in ids) if e]
    logging.disable(logging.NOTSET)


def _activators(entities):
    return [e for e in entities
            if any(i.get("type") == "activates"
                   for i in (e.interacts_with or []))]


def test_the_solver_uses_the_turnover_the_library_records(entities):
    """The regression: nine entities silently ran at 2.0 instead."""
    diverged = []
    for entity in _activators(entities):
        pv = get_parameter(entity, "kcat_per_s")
        if pv is None or pv.value <= 0:
            continue
        used = _vmax_per_s(entity, "any_target")
        if abs(used - pv.value) > 1e-9:
            diverged.append((entity.name, pv.value, used))

    assert not diverged, (
        "the library and the solver disagree about these turnover numbers, "
        f"so a reader quoting the library is quoting a number that is not "
        f"being integrated: {diverged}")


def test_a_per_minute_turnover_reaches_the_solver_converted(entities):
    """EZH2 stores 1.5/min. Read literally that is 90x its real rate."""
    ezh2 = next(e for e in entities if e.name == "EZH2")
    raw = ezh2.physics_params["methyltransferase_kcat_per_min"]

    assert _vmax_per_s(ezh2, "x") == pytest.approx(raw / 60.0)
    assert _vmax_per_s(ezh2, "x") != pytest.approx(DEFAULT_VMAX_PER_S)


def test_the_michaelis_constant_is_read_at_all(entities):
    """Eleven were stored and none was consulted.

    The target passed here is each entity's own recorded substrate, which
    is the only reaction the stored constant describes. Passing an
    arbitrary target instead would exercise the substrate guard rather
    than the read, and would pass for the wrong reason.
    """
    from cognisom.library.parameters import km_substrate

    checked = 0
    for entity in entities:
        pv = get_parameter(entity, "km_um")
        if pv is None or pv.value <= 0:
            continue
        substrate = km_substrate(entity)
        assert substrate, (
            f"{entity.name} holds a km_um whose substrate is unrecorded, so "
            "nothing can tell which reaction it belongs to")
        assert _km_um(entity, substrate, kd_nm=999.0) == pytest.approx(pv.value), (
            f"{entity.name} records a Michaelis constant of {pv.value} uM "
            f"against {substrate} and it was not used for that substrate")
        checked += 1

    assert checked >= 11, f"only {checked} Michaelis constants were found"


def test_a_dissociation_constant_is_only_a_fallback_and_is_converted():
    """Kd is in nM, Km in uM. Substituting one unconverted is a 1000x error."""
    class Bare:
        name = "bare"
        entity_type = None
        physics_params: dict = {}

    bare = Bare()
    assert _km_um(bare, "x", kd_nm=500.0) == pytest.approx(0.5)
    assert _km_um(bare, "x", kd_nm=None) == pytest.approx(DEFAULT_KM_UM)


def test_reaching_a_fallback_is_logged(entities, caplog):
    """A uniform fallback cannot be right for more than one enzyme."""
    class Bare:
        name = "no_kinetics_recorded"
        entity_type = None
        physics_params: dict = {}

    logging.disable(logging.NOTSET)
    try:
        with caplog.at_level(logging.WARNING):
            _vmax_per_s(Bare(), "some_target")
            _km_um(Bare(), "some_target", kd_nm=None)
    finally:
        logging.disable(logging.WARNING)

    text = caplog.text
    assert "no_kinetics_recorded" in text
    assert "not a measurement" in text or "fallback" in text


def test_the_fallback_is_the_value_atm_stores(entities):
    """ATM's stored kcat is exactly the hardcoded fallback.

    That is the clearest sign in the library of a default being copied back
    into the data and then read as if it were measured. Pinned so that if
    ATM is ever given a real value, this test fails and says why.
    """
    atm = next(e for e in entities if e.name == "ATM")

    assert atm.physics_params.get("kinase_kcat_per_s") == DEFAULT_VMAX_PER_S, (
        "ATM no longer stores exactly the fallback value -- if it has been "
        "given an evidenced number, delete this test")


# ── a Km is only a Km for the substrate it was measured against ──────────

def test_a_km_is_not_used_for_a_different_substrate(entities):
    """The hazard created by aliasing eleven substrate-specific key names.

    ``km_amp_um``, ``cdk1_km_um`` and the rest all alias onto one
    ``km_um``, which loses the substrate. Reading that value for any
    reaction turns a lost label into a wrong number: measured on this
    library, none of the nine entities holding both a Michaelis constant
    and an ``activates`` edge has a matching substrate. CD73's Km is for
    AMP while it activates A2AR; IDO1's is for tryptophan while it
    activates AhR through kynurenine.
    """
    from cognisom.genomics.reaction_builder import resolve_kd_nm
    from cognisom.library.parameters import km_substrate

    misapplied = []
    for entity in entities:
        substrate = km_substrate(entity)
        if not substrate:
            continue
        stored = get_parameter(entity, "km_um")
        if stored is None:
            continue
        for inter in (entity.interacts_with or []):
            if inter.get("type") != "activates":
                continue
            target = inter.get("target", "")
            if substrate.lower() in target.lower():
                continue
            used = _km_um(entity, target, resolve_kd_nm(inter))
            if abs(used - stored.value) < 1e-9:
                misapplied.append((entity.name, substrate, target, used))

    assert not misapplied, (
        "a catalytic Michaelis constant was applied to a reaction with a "
        f"different substrate: {misapplied}")


def test_a_matching_substrate_is_still_used():
    """The check must not reject the case it exists to allow."""
    from cognisom.genomics.reaction_builder import _same_substrate

    assert _same_substrate("CDK1", "CDK1")
    assert _same_substrate("RB1", "RB1")
    assert _same_substrate("PIP2", "PIP2 membrane pool")
    assert not _same_substrate("AMP", "A2AR")
    assert not _same_substrate("L-tryptophan", "AhR")
    assert not _same_substrate("", "anything"), (
        "an unrecorded substrate must not match, or the check is vacuous")


def test_every_km_alias_records_its_substrate():
    """An alias with no substrate silently re-opens the misapplication."""
    from cognisom.library.parameters import PARAMETER_ALIASES

    for alias in PARAMETER_ALIASES:
        if alias.canonical != "km_um":
            continue
        assert alias.substrate, (
            f"{alias.legacy} aliases onto km_um without recording which "
            "substrate it was measured against")
