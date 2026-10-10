"""Two subsystems that produce confident wrong answers now refuse instead.

Neither of these is a fix for the underlying defect. Both are the cheap half:
they convert a silent wrong answer into a loud refusal, so the real work can
happen without a demo quoting a number in the meantime.

**GPU orchestrator.** ``_patch_immune`` replaces ``ImmuneModule.update``
wholesale, and the replacement predates the hardening of the recognition path.
``ImmuneModule`` carries 49 references to TCR match state, per-target kill
probability, engagement registration and the peptidome; the replacement
carries none. Its T-cell rule is ``mhc > mhc1_threshold_t`` (default 0.2), so
the neoantigen -> peptidome -> MHC loading -> TCR repertoire loop is bypassed,
along with the ECM barrier, exhaustion and its events, and the sympathetic
multiplier. ``use_gpu=True`` is reachable from the application entry point, and
the 76 passing GPU tests only exercise construction in forced-CPU mode, so
nothing would have noticed.

**BNGL expansion.** The expander drops bond assignments from products, so a
rule re-matches its own product forever. The bundled ``simple_receptor`` model
is one 1:1 binding rule whose correct network is 3 species and 2 reactions; it
expands to ``max_species`` of them with tens of thousands of bogus polymer
reactions, and ``BNGLModule`` then integrated that network without complaint
and lost mass. Terminating on the species cap means the network is truncated,
which is never a network worth integrating.
"""
from __future__ import annotations

import logging

import pytest


# ── GPU orchestrator ─────────────────────────────────────────────────────

class _FakeImmuneModule:
    """Stands in for ImmuneModule by class name, which is what dispatch uses."""

    def update(self, dt):  # pragma: no cover - must never be replaced
        return None


_FakeImmuneModule.__name__ = "ImmuneModule"


class _FakeEngine:
    def __init__(self, modules):
        self.modules = modules


def _orchestrator(**kwargs):
    from cognisom.gpu.orchestrator import GPUOrchestrator

    return GPUOrchestrator(force_cpu=True, **kwargs)


def test_immune_module_is_not_accelerated_by_default():
    module = _FakeImmuneModule()
    original = module.update

    report = _orchestrator().accelerate(_FakeEngine({"immune": module}))

    assert module.update == original, (
        "ImmuneModule.update was replaced; the GPU path reverts T-cell "
        "recognition to an MHC threshold and drops exhaustion"
    )
    assert not any("immune" in m for m in report.modules_accelerated)
    assert any("immune" in m for m in report.modules_skipped)


def test_the_skip_says_what_would_have_been_lost():
    report = _orchestrator().accelerate(_FakeEngine({"immune": _FakeImmuneModule()}))

    skipped = " ".join(report.modules_skipped).lower()
    assert "tcr" in skipped or "recognition" in skipped
    assert "accelerate_immune" in skipped, (
        "the message should name the opt-in, so a caller who needs it can "
        "find it without reading the source"
    )


def test_the_skip_is_logged_at_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="cognisom.gpu.orchestrator"):
        _orchestrator().accelerate(_FakeEngine({"immune": _FakeImmuneModule()}))

    assert any(r.levelno >= logging.WARNING for r in caplog.records), (
        "a silently skipped module is how this went unnoticed in the first place"
    )


def test_it_can_still_be_opted_into_explicitly():
    """The guard is a default, not a removal -- the code path still exists."""
    module = _FakeImmuneModule()
    original = module.update

    report = _orchestrator(accelerate_immune=True).accelerate(
        _FakeEngine({"immune": module})
    )

    # _patch_immune ran, so update is no longer the bound original.
    assert module.update != original
    assert any("immune" in m for m in report.modules_accelerated)
    assert not any("immune" in m for m in report.modules_skipped)


def test_opting_in_still_warns_that_the_path_is_degraded(caplog):
    with caplog.at_level(logging.WARNING, logger="cognisom.gpu.orchestrator"):
        _orchestrator(accelerate_immune=True).accelerate(
            _FakeEngine({"immune": _FakeImmuneModule()})
        )

    assert any("recognition" in r.getMessage().lower()
               or "tcr" in r.getMessage().lower()
               for r in caplog.records), (
        "opting in should still record which capabilities the run lost"
    )


def test_other_modules_are_unaffected_by_the_guard():
    class _Other:
        def update(self, dt):
            return None

    _Other.__name__ = "SomethingElse"
    other = _Other()

    report = _orchestrator().accelerate(_FakeEngine({"other": other}))

    assert any("other" in m for m in report.modules_skipped)


# ── BNGL expansion ───────────────────────────────────────────────────────

def test_expander_records_that_it_hit_the_species_limit():
    from cognisom.bngl.parser import BNGLModel
    from cognisom.bngl.rules import RuleExpander

    model = BNGLModel.simple_receptor()
    expander = RuleExpander(model.molecule_types, model.rules, model.parameters)

    expander.expand(
        seed_species=[s.species for s in model.seed_species],
        max_species=50,
    )

    assert expander.hit_species_limit is True


def test_the_module_refuses_to_initialize_a_truncated_network():
    from cognisom.modules.bngl_module import BNGLModule

    mod = BNGLModule({'model': 'simple_receptor', 'max_species': 50})

    with pytest.raises(RuntimeError) as excinfo:
        mod.initialize()

    message = str(excinfo.value)
    assert "truncated" in message
    assert "50" in message, "the message should name the limit it hit"
    assert "max_species" in message, "and how to change it"


def test_the_refusal_is_specific_about_the_cause():
    """A rule matching its own product is the likely cause, so say so."""
    from cognisom.modules.bngl_module import BNGLModule

    mod = BNGLModule({'model': 'simple_receptor', 'max_species': 50})

    with pytest.raises(RuntimeError, match="own product"):
        mod.initialize()


def test_a_converging_expansion_does_not_set_the_flag():
    """The guard must not fire on a network that finished properly."""
    from cognisom.bngl.rules import RuleExpander

    # No rules at all converges immediately on the seed species.
    expander = RuleExpander(molecule_types={}, rules=[], parameters={})
    reactions, species = expander.expand(seed_species=[], max_species=10)

    assert expander.hit_species_limit is False
    assert reactions == []
