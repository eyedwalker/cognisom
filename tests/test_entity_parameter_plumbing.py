"""
Regression tests for four silent parameter-plumbing defects.

Each of these had the same shape: a value was accepted at one end of a
pipe and quietly discarded before the other end, so the system reported
success while doing nothing. None of them raised, and none of them
showed up as a failing test, which is why all four survived.

  1. Two of eight drug-class lookup keys were written with hyphens while
     the lookup normalized hyphens to underscores, so they could never
     match. One was the anti-androgen class.
  2. The solver modules were never exported or registered, so every
     config the parameter bridge produced for them was dropped.
  3. ODEModule accepted rate-constant overrides and never applied them
     to the system it built, so the anti-androgen binding change was
     inert even once it arrived.
  4. Cytokine and immune-cell behavioural parameters were never passed
     to their constructors, so every entity carried a class default that
     read as curated data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ── 1. Drug-class lookup keys ────────────────────────────────────────

def test_every_drug_class_key_is_reachable():
    """A key that does not survive its own lookup's normalization can
    never be matched by anything the caller passes."""
    from cognisom.workflow.parameter_bridge import (
        DRUG_CLASS_PARAM_MAP,
        normalize_lookup_key,
    )
    unreachable = [
        k for k in DRUG_CLASS_PARAM_MAP if k != normalize_lookup_key(k)
    ]
    assert not unreachable, (
        f"these drug-class keys can never match a query: {unreachable}"
    )


@pytest.mark.parametrize("raw", [
    "anti-androgen", "Anti-Androgen", "anti androgen", "ANTI-ANDROGEN",
])
def test_anti_androgen_resolves_however_it_is_written(raw):
    """The flagship prostate drug class must resolve regardless of how a
    curator typed it. It previously resolved for no spelling at all."""
    from cognisom.workflow.parameter_bridge import (
        DRUG_CLASS_PARAM_MAP,
        normalize_lookup_key,
    )
    assert normalize_lookup_key(raw) in DRUG_CLASS_PARAM_MAP


def test_gene_tables_keep_uppercase_symbols():
    """Gene and target tables are matched with .upper(), a different
    convention from drug classes. Guard it so a well-meaning
    lowercasing does not silently kill those lookups instead."""
    from cognisom.workflow.parameter_bridge import (
        DRUG_TARGET_PARAM_MAP,
        GENE_SPECIFIC_PARAM_MAP,
    )
    for table in (DRUG_TARGET_PARAM_MAP, GENE_SPECIFIC_PARAM_MAP):
        assert all(k == k.upper() for k in table)


def test_import_guard_rejects_an_unreachable_key():
    """The import-time guard is what stops this class of bug returning,
    so prove it actually fires."""
    from cognisom.workflow import parameter_bridge as pb

    pb.DRUG_CLASS_PARAM_MAP["not-normalized"] = {}
    try:
        with pytest.raises(ValueError, match="no lookup can match"):
            pb._assert_keys_reachable()
    finally:
        del pb.DRUG_CLASS_PARAM_MAP["not-normalized"]


# ── 2. Solver module registration ────────────────────────────────────

@pytest.mark.parametrize("name", ["ode", "bngl", "hybrid", "smoldyn"])
def test_solver_modules_are_exported(name):
    """These were importable by path but absent from the package's
    exports, which is why EngineRunner could not register them."""
    import cognisom.modules as m

    cls_name = {
        "ode": "ODEModule", "bngl": "BNGLModule",
        "hybrid": "HybridModule", "smoldyn": "SmoldynModule",
    }[name]
    assert hasattr(m, cls_name), f"cognisom.modules must export {cls_name}"


@pytest.mark.parametrize("name", ["ode", "bngl", "hybrid", "smoldyn"])
def test_engine_runner_can_register_every_module_the_bridge_enables(name):
    """ParameterBridge.infer_enabled_modules turns these on. If the
    runner cannot register them their configs are discarded in silence,
    which is exactly what happened."""
    import inspect

    from cognisom.dashboard import engine_runner

    src = inspect.getsource(engine_runner.EngineRunner.build)
    assert f'"{name}"' in src, (
        f"EngineRunner.build does not register '{name}', so anything "
        f"ParameterBridge emits for it is silently dropped"
    )


# ── 3. ODE rate-constant overrides ───────────────────────────────────

def test_ode_module_applies_rate_override_to_the_system():
    """The anti-androgen mapping encodes a ten-fold reduction in
    androgen-receptor binding. It reached module config and stopped
    there, because the rate lives in the factory's parameters dict."""
    from cognisom.gpu.ode_solver import ODESystem
    from cognisom.modules import ODEModule

    default = ODESystem.ar_signaling_pathway().parameters["k_bind"]
    assert default != 10.0, "test needs an override distinct from default"

    mod = ODEModule({
        "system": "ar_signaling", "n_cells": 4, "k_bind": 10.0,
    })
    mod.initialize()

    assert mod.system.parameters["k_bind"] == 10.0, (
        "k_bind override did not reach the ODE system's parameters"
    )


def test_ode_module_leaves_unrelated_parameters_alone():
    from cognisom.gpu.ode_solver import ODESystem
    from cognisom.modules import ODEModule

    reference = ODESystem.ar_signaling_pathway().parameters
    mod = ODEModule({
        "system": "ar_signaling", "n_cells": 4, "k_bind": 10.0,
    })
    mod.initialize()

    for key, value in reference.items():
        if key != "k_bind":
            assert mod.system.parameters[key] == value


def test_ode_module_warns_on_a_parameter_that_does_not_exist(caplog):
    """A typo in a scenario should not quietly produce baseline
    dynamics."""
    from cognisom.modules import ODEModule

    mod = ODEModule({
        "system": "ar_signaling", "n_cells": 4, "k_bnid": 10.0,
    })
    with caplog.at_level("WARNING"):
        mod.initialize()

    assert "k_bnid" in caplog.text


def test_reserved_config_keys_are_not_treated_as_rate_constants(caplog):
    """method / rtol / atol configure the integrator, not the model, and
    must not be reported as unknown parameters."""
    from cognisom.modules import ODEModule

    mod = ODEModule({
        "system": "ar_signaling", "n_cells": 4,
        "method": "bdf", "rtol": 1e-3, "atol": 1e-6, "heterogeneity": 0.1,
    })
    with caplog.at_level("WARNING"):
        mod.initialize()

    for reserved in ("method", "rtol", "atol", "heterogeneity", "n_cells"):
        assert reserved not in caplog.text


# ── 4. Entity behavioural parameters ─────────────────────────────────

def test_uncurated_cytokine_half_life_is_unknown_not_zero():
    """0.0 is not a missing value, it is a claim that the cytokine
    decays instantly. Every seeded cytokine made that claim."""
    from cognisom.library.models import Cytokine

    assert Cytokine(name="IL-2").half_life_hours is None


def test_cytokine_half_life_survives_a_round_trip():
    from cognisom.library.models import BioEntity, Cytokine

    original = Cytokine(name="IL-2", half_life_hours=0.28)
    restored = BioEntity.from_dict(original.to_dict())

    assert restored.half_life_hours == 0.28


def test_uncurated_immune_parameters_are_unknown_not_cytotoxic():
    """The class defaults described a killer cell, and no seeding code
    overrode them, so Tregs and naive CD8s carried a per-contact kill
    chance of 0.8."""
    from cognisom.library.models import ImmuneCellEntity

    cell = ImmuneCellEntity(name="Regulatory T cell")
    assert cell.kill_probability is None
    assert cell.detection_radius is None
    assert cell.kill_radius is None
    assert cell.mhc1_expression is None


def test_immune_cell_state_fields_still_start_at_zero():
    """State is legitimately zero at construction; only uncurated
    parameters became None. Keep the two apart."""
    from cognisom.library.models import ImmuneCellEntity

    cell = ImmuneCellEntity(name="Naive CD8 T cell")
    assert cell.activation_state == 0.0
    assert cell.exhaustion_level == 0.0
    assert cell.activated is False
    assert cell.alive is True


def test_immune_parameters_survive_a_round_trip():
    from cognisom.library.models import BioEntity, ImmuneCellEntity

    original = ImmuneCellEntity(
        name="NK cell", kill_probability=0.9, detection_radius=12.0,
    )
    restored = BioEntity.from_dict(original.to_dict())

    assert restored.kill_probability == 0.9
    assert restored.detection_radius == 12.0


def test_seeded_entities_no_longer_assert_uncurated_numbers():
    """End to end through the real seeding path: nothing may claim
    instant cytokine decay, and immune cell types may not all share one
    identical behavioural profile."""
    store = pytest.importorskip("cognisom.library.store")
    from cognisom.library.models import EntityType

    s = store.EntityStore()
    cytokines, _ = s.search(
        entity_type=EntityType.CYTOKINE.value, limit=500,
    )
    immune, _ = s.search(
        entity_type=EntityType.IMMUNE_CELL.value, limit=500,
    )

    assert cytokines, "expected seeded cytokines"
    assert immune, "expected seeded immune cell types"

    instant_decay = [c.name for c in cytokines if c.half_life_hours == 0.0]
    assert not instant_decay, (
        f"these cytokines assert instant decay: {instant_decay}"
    )

    bogus = [c.name for c in immune if c.kill_probability == 0.8]
    assert not bogus, (
        f"these immune cell types carry an uncurated kill probability "
        f"of 0.8: {bogus}"
    )
