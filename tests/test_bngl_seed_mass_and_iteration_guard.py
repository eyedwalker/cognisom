"""Two defects that only appear once species identity becomes canonical.

Making the expander correct meant making species identity invariant to bond
ids, molecule order and within-molecule component order. That is right, and it
is what takes ``simple_receptor`` from 69,938 reactions down to 2. But it
changes what counts as "the same species", and two things were keyed on the
old, finer notion:

**Seed mass.** ``BNGLModule.initialize`` placed seed amounts by ``str(sp)``.
BNGL permits the same complex to be spelled more than one way, and the new
identity collapses those spellings into one species -- so only one spelling
matched and the rest of the declared mass silently vanished before the first
step. Measured on the model below: 300 of a declared 1000, a 70% loss, and a
regression against the pre-canonicalisation code rather than a pre-existing
bug. Seeds are now keyed on the same identity the expander interns on and
summed, and a seed that matches nothing is a refusal.

**The iteration limit.** The species-cap guard only fires if the network
reaches ``max_species`` within ``max_iterations``. A rule set that adds few
species per iteration -- monomer addition onto a growing filament, where one
reactant pattern names every site free so only a monomer matches -- is
unbounded yet never reaches the cap, so expansion stopped by running out of
iterations and nothing noticed. ``max_iterations`` was also hardcoded in
``RuleExpander.expand`` and not reachable through ``BNGLModule`` at all.
"""
from __future__ import annotations

import re
import textwrap

import pytest

# The same complex, spelled two ways: component order swapped inside R.
# BNGL permits both, and canonical identity makes them one species.
COMPONENT_ORDER_MODEL = textwrap.dedent("""
    begin parameters
      kon 1e-3
      koff 0.1
    end parameters

    begin molecule types
      L(r)
      R(l,Y~U~P)
    end molecule types

    begin seed species
      L(r!1).R(l!1,Y~U) 300
      L(r!1).R(Y~U,l!1) 700
    end seed species

    begin observables
      Molecules Ltot L()
    end observables

    begin reaction rules
      L(r) + R(l) <-> L(r!1).R(l!1) kon, koff
    end reaction rules
""").strip()


@pytest.fixture
def component_order_model(tmp_path):
    path = tmp_path / "comporder.bngl"
    path.write_text(COMPONENT_ORDER_MODEL + "\n")
    return str(path)


def _module(**config):
    from cognisom.modules.bngl_module import BNGLModule

    return BNGLModule(config)


# ── seed mass ────────────────────────────────────────────────────────────

def test_seeds_spelled_two_ways_sum_to_the_declared_mass(component_order_model):
    mod = _module(model=component_order_model, max_species=500)
    mod.initialize()

    assert float(mod.state.sum()) == pytest.approx(1000.0), (
        f"declared 1000 across two spellings, got {float(mod.state.sum())}"
    )


def test_the_observable_sees_the_full_declared_mass(component_order_model):
    mod = _module(model=component_order_model, max_species=500)
    mod.initialize()

    assert float(mod.get_observables()["Ltot"]) == pytest.approx(1000.0)


def test_the_merged_species_carries_it_all_on_one_row(component_order_model):
    """Both spellings are one species, so one row should hold 1000."""
    mod = _module(model=component_order_model, max_species=500)
    mod.initialize()

    nonzero = [float(v) for v in mod.state if v]
    assert nonzero == [pytest.approx(1000.0)], (
        f"expected a single seeded row of 1000, got {nonzero}"
    )


def test_bundled_models_keep_their_declared_seed_amounts():
    """The fix must not disturb the ordinary single-spelling case."""
    for name in ("simple_receptor", "egfr_signaling"):
        mod = _module(model=name, max_species=10000)
        mod.initialize()

        declared = sum(ss.count for ss in mod.model.seed_species)
        assert float(mod.state.sum()) == pytest.approx(declared), (
            f"{name}: declared {declared}, seeded {float(mod.state.sum())}"
        )


def test_molecule_counts_are_conserved_through_the_ode_path():
    """Binding reduces the species count, so totals per type are the invariant.

    The pre-fix expander drifted 2000.0 -> 2001.16 over 200 steps, i.e. it
    created mass. Summing state is not conserved under binding (two molecules
    become one complex); the per-molecule-type totals are.
    """
    for name in ("simple_receptor", "egfr_signaling"):
        mod = _module(model=name, max_species=10000, method="ode")
        mod.initialize()
        types = sorted(mod.model.molecule_types)

        def totals():
            out = dict.fromkeys(types, 0.0)
            for sp, amount in zip(mod.species_list, mod.state):
                text = str(sp)
                for t in types:
                    hits = len(re.findall(
                        r"(?<![A-Za-z0-9_])" + re.escape(t) + r"\(", text))
                    out[t] += hits * float(amount)
            return out

        before = totals()
        for _ in range(200):
            mod.update(1e-6)
        after = totals()

        for t in types:
            assert after[t] == pytest.approx(before[t], rel=1e-9), (
                f"{name}: {t} went {before[t]} -> {after[t]}"
            )


# ── the iteration limit ──────────────────────────────────────────────────

def test_the_iteration_limit_is_configurable():
    mod = _module(model="simple_receptor", max_iterations=7)

    assert mod.max_iterations == 7


def test_the_iteration_limit_defaults_without_a_config():
    """It was hardcoded in expand() and unreachable through the module."""
    mod = _module(model="simple_receptor")

    assert mod.max_iterations == 100


def test_a_network_truncated_by_the_iteration_limit_is_refused():
    """The species cap cannot catch this: the cap is never reached."""
    mod = _module(model="egfr_signaling", max_species=10000, max_iterations=2)

    with pytest.raises(RuntimeError) as excinfo:
        mod.initialize()

    message = str(excinfo.value)
    assert "iteration limit" in message
    assert "truncated" in message
    assert "2" in message, "the refusal should name the limit it hit"
    assert "max_iterations" in message, "and how to change it"


def test_the_refusal_distinguishes_itself_from_the_species_cap():
    """Two different truncation modes, two different messages."""
    iteration_mod = _module(
        model="egfr_signaling", max_species=10000, max_iterations=2)
    with pytest.raises(RuntimeError) as iter_err:
        iteration_mod.initialize()

    assert "species limit" not in str(iter_err.value)


def test_the_bundled_models_converge_and_are_not_refused():
    """Neither guard may fire on a network that finished properly."""
    for name in ("simple_receptor", "egfr_signaling"):
        mod = _module(model=name, max_species=10000)
        mod.initialize()  # must not raise

        assert mod.species_list
        assert mod.reactions
