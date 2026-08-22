"""
Unit and thermodynamic consistency at the entity -> simulation boundary.

The entity library stores biophysical constants with their units encoded
in the key name (``kd_nm`` vs ``kd_um``, ``half_life_hours`` vs
``half_life_min``) and no parser. Every read site therefore had to do its
own conversion, and three of them got it wrong in ways that changed
simulated biology by orders of magnitude:

  * ``inter.get("kd_nm", inter.get("kd_um", 0))`` returned a micromolar
    number unconverted -- 1000x too small a Kd. Both key variants are in
    use: seed_data writes ``kd_nm``, seed_checkpoints writes ``kd_um``.

  * ``if "min" in str(pp.get("half_life_min", ""))`` asks whether the
    substring "min" occurs inside the *number*. ``str(30.0)`` is
    ``"30.0"``, so the branch never fired and every minutes-valued
    half-life was integrated as hours -- degradation 60x too slow.

  * ``k_on = 10/Kd`` with ``k_off = Kd`` gives an effective
    ``k_off/k_on = Kd^2/10``, so the network equilibrates to a
    dissociation constant that is not the one the entity declares, with
    error growing quadratically in Kd.

These tests pin the conversions and the Kd round-trip.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.genomics.reaction_builder import (
    DEFAULT_K_ON_PER_NM_PER_HOUR,
    _get_binding_rate,
    _get_unbinding_rate,
    degradation_rate_per_hour,
    rates_from_kd,
    resolve_half_life_hours,
    resolve_kd_nm,
)


class _Entity:
    """Minimal stand-in for a BioEntity's interaction surface."""

    def __init__(self, interacts_with):
        self.interacts_with = interacts_with


# ── Kd units ────────────────────────────────────────────────────────

@pytest.mark.parametrize("interaction,expected_nm", [
    ({"kd_nm": 0.1}, 0.1),
    ({"kd_um": 0.4}, 400.0),          # seed_checkpoints uses micromolar
    ({"kd_pm": 250.0}, 0.25),
    ({"kd_mm": 1.0}, 1e6),
    ({"kd_m": 1e-9}, 1.0),
])
def test_kd_is_normalised_to_nanomolar(interaction, expected_nm):
    assert resolve_kd_nm(interaction) == pytest.approx(expected_nm)


def test_micromolar_kd_is_not_read_as_nanomolar():
    """The specific 1000x regression."""
    assert resolve_kd_nm({"kd_um": 0.4}) == pytest.approx(400.0)
    assert resolve_kd_nm({"kd_um": 0.4}) != pytest.approx(0.4)


def test_nanomolar_wins_when_both_are_present():
    assert resolve_kd_nm({"kd_nm": 5.0, "kd_um": 99.0}) == pytest.approx(5.0)


@pytest.mark.parametrize("interaction", [
    {}, None, {"kd_nm": 0}, {"kd_nm": None}, {"kd_nm": "not-a-number"},
])
def test_missing_or_unusable_kd_returns_none(interaction):
    assert resolve_kd_nm(interaction) is None


# ── Half-life units ─────────────────────────────────────────────────

@pytest.mark.parametrize("params,expected_hours", [
    ({"half_life_hours": 3.0}, 3.0),
    ({"half_life_min": 30.0}, 0.5),      # TP53 basal turnover
    ({"half_life_min": 20.0}, 1.0 / 3.0),
    ({"half_life_sec": 3600.0}, 1.0),
    ({"half_life_days": 2.0}, 48.0),
])
def test_half_life_is_normalised_to_hours(params, expected_hours):
    assert resolve_half_life_hours(params) == pytest.approx(expected_hours)


def test_minutes_are_not_read_as_hours():
    """The specific 60x regression.

    A 20-minute half-life is one third of an hour. Read as 20 hours it
    yields a degradation rate 60x too slow, so a protein the model should
    clear within the first timestep instead persists for days.
    """
    minutes = resolve_half_life_hours({"half_life_min": 20.0})
    assert minutes == pytest.approx(1.0 / 3.0)

    k_fast = degradation_rate_per_hour({"half_life_min": 20.0})
    k_slow = degradation_rate_per_hour({"half_life_hours": 20.0})
    assert k_fast / k_slow == pytest.approx(60.0)


def test_degradation_rate_is_ln2_over_half_life():
    import math
    assert degradation_rate_per_hour({"half_life_hours": 1.0}) == pytest.approx(
        math.log(2)
    )
    assert degradation_rate_per_hour({}) is None


# ── Detailed balance ────────────────────────────────────────────────

@pytest.mark.parametrize("kd_nm", [0.1, 0.3, 2.0, 5.0, 10.0, 50.0, 400.0])
def test_rates_from_kd_reproduce_the_declared_kd(kd_nm):
    """Kd = k_off / k_on. This must hold exactly, for every Kd."""
    k_on, k_off = rates_from_kd(kd_nm)
    assert k_off / k_on == pytest.approx(kd_nm)
    assert k_on == pytest.approx(DEFAULT_K_ON_PER_NM_PER_HOUR)


@pytest.mark.parametrize("kd_nm", [0.1, 50.0])
def test_old_conversion_would_have_broken_detailed_balance(kd_nm):
    """Documents the magnitude of what was fixed.

    The previous pair was k_on = 10/Kd, k_off = Kd, giving an effective
    Kd of Kd^2/10 -- 1000x too tight at the AR-DHT Kd of 0.1 nM, and 5x
    too loose at the AR-FOXA1 Kd of 50 nM.
    """
    legacy_effective_kd = (kd_nm * kd_nm) / 10.0
    assert legacy_effective_kd != pytest.approx(kd_nm)

    k_on, k_off = rates_from_kd(kd_nm)
    assert k_off / k_on == pytest.approx(kd_nm)


def test_binding_and_unbinding_rates_agree_on_one_kd():
    """The two accessors must not be derivable to different Kds."""
    entity = _Entity([{"target": "DHT", "type": "binds_to", "kd_nm": 0.1}])

    k_on = _get_binding_rate(entity, "DHT")
    k_off = _get_unbinding_rate(entity, "DHT")
    assert k_off / k_on == pytest.approx(0.1)


def test_binding_rate_honours_micromolar_keys():
    entity = _Entity([{"target": "SIRPA", "type": "binds_to", "kd_um": 0.4}])
    k_on = _get_binding_rate(entity, "SIRPA")
    k_off = _get_unbinding_rate(entity, "SIRPA")
    assert k_off / k_on == pytest.approx(400.0)


def test_defaults_are_used_when_no_kd_is_declared():
    entity = _Entity([{"target": "X", "type": "binds_to"}])
    assert _get_binding_rate(entity, "X", default_on=100.0) == 100.0
    assert _get_unbinding_rate(entity, "X", default_off=10.0) == 10.0
    # Unknown target falls back too.
    assert _get_binding_rate(entity, "MISSING", default_on=7.0) == 7.0


def test_non_binding_interaction_is_not_treated_as_binding():
    entity = _Entity([{"target": "PSA", "type": "activates", "kd_nm": 5.0}])
    assert _get_binding_rate(entity, "PSA", default_on=100.0) == 100.0
