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

**BNGL expansion.** Terminating on the species cap means the network is
truncated -- a prefix of something larger, across which mass is not
conserved -- which is never a network worth integrating. This was reachable
for the wrong reason until the expander was fixed: it dropped bond
assignments from products, so a rule re-matched its own product forever and
the bundled ``simple_receptor`` model, one 1:1 binding rule whose correct
network is 3 species and 2 reactions, expanded to ``max_species`` of them.
The guard stays as a backstop, because a rule set can be genuinely
unbounded; the cases below use a polymer model that really is.
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

#: A two-site monomer really does polymerize without end, so expansion of
#: it can only ever terminate on the species cap. The bundled
#: simple_receptor model used to do the same for the wrong reason.
POLYMER_BNGL = """begin parameters
    kp 1.0
    km 0.1
end parameters

begin molecule types
    A(l,r)
end molecule types

begin seed species
    A(l,r) 1000
end seed species

begin reaction rules
    A(r) + A(l) <-> A(r!1).A(l!1) kp, km
end reaction rules
"""


@pytest.fixture
def polymer_model(tmp_path):
    path = tmp_path / "polymer.bngl"
    path.write_text(POLYMER_BNGL)
    return path


def test_expander_records_that_it_hit_the_species_limit(polymer_model):
    from cognisom.bngl.parser import load_bngl_model
    from cognisom.bngl.rules import RuleExpander

    model = load_bngl_model(polymer_model)
    expander = RuleExpander(model.molecule_types, model.rules, model.parameters)

    expander.expand(
        seed_species=model.get_seed_species_list(),
        max_species=12,
    )

    assert expander.hit_species_limit is True


def test_the_module_refuses_to_initialize_a_truncated_network(polymer_model):
    from cognisom.modules.bngl_module import BNGLModule

    mod = BNGLModule({'model': str(polymer_model), 'max_species': 12})

    with pytest.raises(RuntimeError) as excinfo:
        mod.initialize()

    message = str(excinfo.value)
    assert "truncated" in message
    assert "12" in message, "the message should name the limit it hit"
    assert "max_species" in message, "and how to change it"


def test_the_refusal_names_a_cause_worth_checking(polymer_model):
    """Hitting the cap now means the network may genuinely be unbounded."""
    from cognisom.modules.bngl_module import BNGLModule

    mod = BNGLModule({'model': str(polymer_model), 'max_species': 12})

    with pytest.raises(RuntimeError, match="unbounded"):
        mod.initialize()


def test_a_converging_expansion_does_not_set_the_flag():
    """The guard must not fire on a network that finished properly."""
    from cognisom.bngl.rules import RuleExpander

    # No rules at all converges immediately on the seed species.
    expander = RuleExpander(molecule_types={}, rules=[], parameters={})
    reactions, species = expander.expand(seed_species=[], max_species=10)

    assert expander.hit_species_limit is False
    assert reactions == []


@pytest.mark.parametrize(
    "name,n_species", [("simple_receptor", 3), ("egfr_signaling", 15)]
)
def test_the_bundled_models_do_not_trip_the_guard(name, n_species):
    """The guard is a backstop, not the normal path.

    Both bundled models converge well inside a modest cap. Before the
    expander was fixed both ran to the cap instead, so this is the case
    that only passes with the fix in place.
    """
    from cognisom.modules.bngl_module import BNGLModule

    mod = BNGLModule({'model': name, 'max_species': 50})
    mod.initialize()

    assert len(mod.species_list) == n_species
