"""The BNGL rule expander: what a rule DOES, not what its shape suggests.

``cognisom/bngl`` had no test file, which is how three independent
semantic defects survived in a module the dashboard exposes:

* product bond assignments were dropped, so ``L(r) + R(l) <-> L(r!1).R(l!1)``
  produced ``L(r).R(l)`` with both sites still reading free, the rule
  re-matched its own product, and the 3-species network became
  ``max_species`` of them with tens of thousands of bogus polymer reactions;
* ``ReactionRule.state_changes`` was declared and never populated, so
  ``R(Y~U) -> R(Y~P)`` expanded to an identity reaction;
* ``parse_rule`` split pattern strings on a bare ``'+'``, so the ``'+'`` in
  ``'!+'`` was read as a separator and every ``!+`` rule crashed the parser.

Two more were found while fixing those: the wildcard tests in
``ComponentInstance.matches`` were applied to the species side rather than
the pattern side, so no ``!+`` pattern could match anything; and an
undeclared state name parsed as ``state=None``, which means "any state" to
the matcher, quietly turning a specific pattern into a universal one.
"""
from __future__ import annotations

import math

import pytest

from cognisom.bngl.molecules import (
    BondState,
    MoleculeType,
    parse_pattern,
    parse_species,
)
from cognisom.bngl.observables import Observable
from cognisom.bngl.parser import BNGLModel, BNGLParser
from cognisom.bngl.rules import RuleExpander, derive_operations, parse_rule
from cognisom.modules.bngl_module import BNGLModule


# ── fixtures ─────────────────────────────────────────────────────────────

LR_TYPES = {
    "L": MoleculeType.from_string("L(r)"),
    "R": MoleculeType.from_string("R(l,Y~U~P)"),
}

#: A(l,r) with a head-to-tail rule really does polymerize without end, so
#: expansion of it can only ever terminate on the species cap. Used for the
#: cases that must still hit the guard after the fix.
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


def _bind_rule(kon=1.0, koff=0.1):
    return parse_rule(
        f"L(r) + R(l) <-> L(r!1).R(l!1) {kon}, {koff}", LR_TYPES, {}, "bind"
    )


def _expand(rules, seeds, types=LR_TYPES, **kw):
    expander = RuleExpander(types, rules, {}, **kw)
    reactions, species = expander.expand(seed_species=seeds)
    return expander, reactions, species


@pytest.fixture
def polymer_model(tmp_path):
    path = tmp_path / "polymer.bngl"
    path.write_text(POLYMER_BNGL)
    return path


# ── (a) bond assignments are carried into product species ────────────────

def test_simple_receptor_is_three_species_and_two_reactions():
    """One 1:1 reversible binding rule. L, R, L.R and one reaction each way.

    This expanded to max_species (200 species / 69,938 reactions at
    max_species=200) before the fix.
    """
    model = BNGLModel.simple_receptor()
    expander = RuleExpander(model.molecule_types, model.rules, model.parameters)
    reactions, species = expander.expand(
        seed_species=model.get_seed_species_list(), max_species=10000
    )

    assert len(species) == 3
    assert len(reactions) == 2
    assert expander.hit_species_limit is False
    assert expander.hit_iteration_limit is False
    assert {sp.canonical_string() for sp in species} == {
        "L(r)", "R(l)", "L(r!1).R(l!1)"
    }


def test_the_product_of_a_binding_rule_carries_the_bond():
    _, _, species = _expand(
        [_bind_rule()],
        [parse_species("L(r)", LR_TYPES), parse_species("R(l,Y~U)", LR_TYPES)],
    )
    complexes = [sp for sp in species if sp.n_molecules == 2]
    assert len(complexes) == 1
    complex_ = complexes[0]

    assert len(complex_.bonds) == 1, "Species.bonds must be derived, not empty"
    # Both endpoints must agree on one bond id, and the id must have
    # exactly two endpoints -- a dangling half-bond reads as bound while
    # connecting nothing.
    endpoints = complex_.bond_endpoints()
    assert list(endpoints.values()) and all(
        len(eps) == 2 for eps in endpoints.values()
    )
    assert "!" in str(complex_)


def test_a_bound_site_does_not_read_as_free():
    """The root cause of the explosion: a bound site that still matched the
    rule's own reactant pattern, so every iteration appended a monomer."""
    _, _, species = _expand(
        [_bind_rule()],
        [parse_species("L(r)", LR_TYPES), parse_species("R(l,Y~U)", LR_TYPES)],
    )
    complex_ = next(sp for sp in species if sp.n_molecules == 2)

    for mol in complex_.molecules:
        for comp in mol.components:
            if comp.name in ("r", "l"):
                assert comp.bond_state == BondState.BOUND
                assert comp.bond_id is not None

    assert not complex_.matches(parse_pattern("L(r)", LR_TYPES))
    assert not complex_.matches(parse_pattern("R(l)", LR_TYPES))


def test_egfr_signaling_matches_the_independently_enumerated_network():
    """15 species and 60 reactions, 10 per rule direction.

    Derived without the expander: an EGFR half is (L bound?, Y phosphorylated?),
    so there are 4 halves; 4 monomers + C(4+1,2)=10 unordered dimers + free
    EGF = 15 species. Each of the six rule directions acts on exactly 10 of
    them (see the enumeration in the docstring of the audit).
    """
    model = BNGLModel.egfr_signaling()
    expander = RuleExpander(model.molecule_types, model.rules, model.parameters)
    reactions, species = expander.expand(
        seed_species=model.get_seed_species_list(), max_species=10000
    )

    assert len(species) == 15
    assert len(reactions) == 60
    assert expander.hit_species_limit is False

    per_direction = {}
    for rxn in reactions:
        key = rxn.name.rsplit("_", 1)[0]
        per_direction[key] = per_direction.get(key, 0) + 1
    assert per_direction == {
        "EGF_binding_fwd": 10,
        "EGF_binding_rev": 10,
        "dimerization_fwd": 10,
        "dimerization_rev": 10,
        "phosphorylation_fwd": 10,
        "dephosphorylation_fwd": 10,
    }
    # Every dimer has exactly one CR1-CR1 bond; nothing polymerized.
    assert max(sp.n_molecules for sp in species) == 4
    for sp in species:
        assert all(len(eps) == 2 for eps in sp.bond_endpoints().values())


def test_no_species_is_double_counted_under_a_different_bond_numbering():
    model = BNGLModel.egfr_signaling()
    _, _, species = _expand(
        model.rules, model.get_seed_species_list(), model.molecule_types
    )
    keys = [sp.canonical_string() for sp in species]
    assert len(set(keys)) == len(keys)


def test_canonical_string_ignores_molecule_order_and_bond_numbers():
    a = parse_species("L(r!1).R(l!1,Y~U)", LR_TYPES)
    b = parse_species("R(l!7,Y~U).L(r!7)", LR_TYPES)

    assert a.canonical_string() == b.canonical_string()
    assert a == b
    assert hash(a) == hash(b)
    # ...and str() is left alone, because BNGLModule and the observables
    # key their tables on it.
    assert str(a) != str(b)


def test_canonical_string_is_injective_over_a_real_network():
    model = BNGLModel.egfr_signaling()
    _, _, species = _expand(
        model.rules, model.get_seed_species_list(), model.molecule_types
    )
    assert len({sp.canonical_string() for sp in species}) == len(species)
    assert all(sp.canonical_is_exact for sp in species)


# ── (b) state-change rules actually change state ─────────────────────────

def test_state_changes_are_derived_at_parse_time():
    rule = parse_rule("R(Y~U) -> R(Y~P) 0.3", LR_TYPES, {}, "phos")

    assert rule.operations_derived is True
    assert len(rule.state_changes) == 1
    change = rule.state_changes[0]
    assert (change.component_name, change.old_state, change.new_state) == (
        "Y", "U", "P",
    )
    assert rule.bond_changes == []


def test_a_state_change_rule_creates_the_phosphorylated_species():
    rule = parse_rule("R(Y~U) -> R(Y~P) 0.3", LR_TYPES, {}, "phos")
    _, reactions, species = _expand(
        [rule], [parse_species("R(l,Y~U)", LR_TYPES)]
    )

    assert {sp.canonical_string() for sp in species} == {
        "R(Y~U,l)", "R(Y~P,l)",
    }
    assert len(reactions) == 1
    rxn = reactions[0]
    assert str(rxn.reactants[0]) != str(rxn.products[0]), (
        "an identity reaction is the symptom: reactant and product were "
        "the same species"
    )


def test_a_state_change_inside_a_complex_keeps_the_complex_intact():
    rule = parse_rule("R(Y~U) -> R(Y~P) 0.3", LR_TYPES, {}, "phos")
    _, reactions, species = _expand(
        [_bind_rule(), rule],
        [parse_species("L(r)", LR_TYPES), parse_species("R(l,Y~U)", LR_TYPES)],
    )

    assert len(species) == 5
    assert len(reactions) == 6
    phos = [r for r in reactions if r.name.startswith("phos")]
    in_complex = [r for r in phos if r.reactants[0].n_molecules == 2]
    assert len(in_complex) == 1
    rxn = in_complex[0]
    assert len(rxn.products) == 1, "a state change must not split the complex"
    assert rxn.products[0].canonical_string() == "L(r!1).R(Y~P,l!1)"


def test_a_state_change_rule_does_not_invent_a_bond():
    rule = parse_rule("R(Y~U) -> R(Y~P) 0.3", LR_TYPES, {}, "phos")
    _, _, species = _expand([rule], [parse_species("R(l,Y~U)", LR_TYPES)])
    for sp in species:
        assert sp.bonds == []
        assert all(
            c.bond_state == BondState.UNBOUND
            for m in sp.molecules for c in m.components
        )


def test_an_undeclared_state_is_rejected_rather_than_ignored():
    """state=None means "any state" to the matcher, so a typo would turn a
    phosphorylation rule into an identity reaction."""
    with pytest.raises(ValueError, match="no state 'Q'"):
        parse_rule("R(Y~Q) -> R(Y~P) 1.0", LR_TYPES, {}, "typo")
    with pytest.raises(ValueError, match="no state 'Q'"):
        parse_species("R(l,Y~Q)", LR_TYPES)


# ── (c) the !+ wildcard ──────────────────────────────────────────────────

def test_a_bound_wildcard_in_a_rule_no_longer_crashes_the_parser():
    rule = parse_rule("R(l!+,Y~U) -> R(l!+,Y~P) 2.0", LR_TYPES, {}, "ctx")

    assert len(rule.reactant_patterns) == 1
    assert len(rule.product_patterns) == 1
    assert str(rule.reactant_patterns[0]) == "R(l!+,Y~U)"
    assert rule.operations_derived is True


def test_a_bound_wildcard_survives_a_pattern_separator():
    types = {
        "A": MoleculeType.from_string("A(b,s~0~1)"),
        "B": MoleculeType.from_string("B(a)"),
    }
    rule = parse_rule(
        "A(b!+,s~0) + B(a) -> A(b!+,s~1) + B(a) 1.0", types, {}, "two"
    )
    assert [str(p) for p in rule.reactant_patterns] == ["A(b!+,s~0)", "B(a)"]
    assert [str(p) for p in rule.product_patterns] == ["A(b!+,s~1)", "B(a)"]


def test_a_bound_wildcard_rule_fires_only_in_the_bound_context():
    ctx = parse_rule("R(l!+,Y~U) -> R(l!+,Y~P) 2.0", LR_TYPES, {}, "ctx")
    _, reactions, _ = _expand(
        [_bind_rule(), ctx],
        [parse_species("L(r)", LR_TYPES), parse_species("R(l,Y~U)", LR_TYPES)],
    )
    fired = [r for r in reactions if r.name.startswith("ctx")]

    assert len(fired) == 1, "exactly one R is in a bound context"
    assert fired[0].reactants[0].n_molecules == 2
    assert fired[0].rate == 2.0


def test_the_bound_wildcard_observable_counts_bound_molecules():
    """``Molecules L_bound L(r!+)`` read 0 at HEAD, because the wildcard
    test was applied to the species side."""
    bound = parse_species("L(r!1).R(l!1,Y~U)", LR_TYPES)
    free = parse_species("L(r)", LR_TYPES)
    species = [free, bound]
    counts = {str(free): 600.0, str(bound): 400.0}

    obs = Observable.from_string("Molecules L_bound L(r!+)", LR_TYPES)
    assert obs.count(counts, species) == 400.0
    free_obs = Observable.from_string("Molecules L_free L(r)", LR_TYPES)
    assert free_obs.count(counts, species) == 600.0


def test_the_bundled_egfr_bound_observable_is_not_stuck_at_zero():
    model = BNGLModel.egfr_signaling()
    mod = BNGLModule({"model": "egfr_signaling", "max_species": 10000})
    mod.initialize()
    for _ in range(50):
        mod.update(1e-3)
    assert mod.get_observables()["EGFR_bound"] > 0.0


# ── pattern embeddings honour the pattern's own bonds ────────────────────

def test_a_dissociation_pattern_requires_an_actual_bond():
    bound = parse_species("L(r!1).R(l!1,Y~U)", LR_TYPES)
    pattern = parse_pattern("L(r!1).R(l!1)", LR_TYPES)
    assert bound.matches(pattern)

    # An L sitting next to an R in the same complex but bonded through
    # something else must not match.
    types = {
        "L": MoleculeType.from_string("L(r,x)"),
        "R": MoleculeType.from_string("R(l,x)"),
    }
    sideways = parse_species("L(r,x!1).R(l,x!1)", types)
    assert not sideways.matches(parse_pattern("L(r!1).R(l!1)", types))


def test_every_embedding_is_enumerated_not_just_the_first():
    """A pattern can embed into one species two ways and give two different
    products -- phosphorylating the EGF-loaded half of a dimer is not the
    same reaction as phosphorylating the free half."""
    model = BNGLModel.egfr_signaling()
    _, reactions, species = _expand(
        model.rules, model.get_seed_species_list(), model.molecule_types
    )
    target = next(
        sp for sp in species
        if sp.canonical_string()
        == "EGF(R!1).EGFR(CR1!2,L!1,Y~U).EGFR(CR1!2,L,Y~U)"
    )
    phos = [
        r for r in reactions
        if r.name.startswith("phosphorylation") and r.reactants[0] is target
    ]
    assert len(phos) == 2
    assert len({p.canonical_string() for r in phos for p in r.products}) == 2


# ── reaction deduplication ───────────────────────────────────────────────

def test_reactions_are_not_duplicated_across_iterations():
    model = BNGLModel.egfr_signaling()
    _, reactions, _ = _expand(
        model.rules, model.get_seed_species_list(), model.molecule_types
    )
    keys = [
        (
            r.rule.name,
            tuple(sorted(s.canonical_string() for s in r.reactants)),
            tuple(sorted(s.canonical_string() for s in r.products)),
        )
        for r in reactions
    ]
    assert len(set(keys)) == len(keys)


def test_a_symmetric_binding_rule_yields_one_reaction_per_reactant_pair():
    types = {"A": MoleculeType.from_string("A(x)")}
    rule = parse_rule("A(x) + A(x) <-> A(x!1).A(x!1) 1.0, 0.5", types, {}, "dim")
    _, reactions, species = _expand([rule], [parse_species("A(x)", types)], types)

    assert len(species) == 2
    assert len(reactions) == 2, (
        "A+A -> A2 enumerated in both slot orders must still be one reaction"
    )


# ── statistical factors (BioNetGen's convention) ─────────────────────────

def test_a_homodimerization_rule_gets_the_one_half_symmetry_factor():
    """A rule whose two reactant patterns are interchangeable counts each
    pair twice, so BioNetGen divides by the rule's automorphism count."""
    types = {"A": MoleculeType.from_string("A(x)")}
    rule = parse_rule("A(x) + A(x) <-> A(x!1).A(x!1) 1.0, 0.5", types, {}, "dim")
    _, reactions, _ = _expand([rule], [parse_species("A(x)", types)], types)

    fwd = next(r for r in reactions if r.name.startswith("dim_fwd"))
    rev = next(r for r in reactions if r.name.startswith("dim_rev"))
    assert fwd.rate == pytest.approx(0.5)   # 1.0 x 1 embedding / aut 2
    assert rev.rate == pytest.approx(0.5)   # 0.5 x 2 embeddings / aut 2


def test_a_pattern_that_embeds_twice_gets_twice_the_rate():
    model = BNGLModel.egfr_signaling()
    _, reactions, species = _expand(
        model.rules, model.get_seed_species_list(), model.molecule_types
    )
    uu_dimer = next(
        sp for sp in species
        if sp.canonical_string() == "EGFR(CR1!1,L,Y~U).EGFR(CR1!1,L,Y~U)"
    )
    phos = [
        r for r in reactions
        if r.name.startswith("phosphorylation") and r.reactants[0] is uu_dimer
    ]
    assert len(phos) == 1
    # kphos = 1.0, two unphosphorylated Y sites, one product either way.
    assert phos[0].rate == pytest.approx(2.0)


def test_statistical_factors_can_be_turned_off():
    types = {"A": MoleculeType.from_string("A(x)")}
    rule = parse_rule("A(x) + A(x) <-> A(x!1).A(x!1) 1.0, 0.5", types, {}, "dim")
    _, reactions, _ = _expand(
        [rule], [parse_species("A(x)", types)], types,
        apply_statistical_factors=False,
    )
    assert next(r for r in reactions if "fwd" in r.name).rate == pytest.approx(1.0)


# ── mass conservation through BNGLModule's ODE path ──────────────────────

@pytest.mark.parametrize("model_name", ["simple_receptor", "egfr_signaling"])
def test_every_reaction_endpoint_is_in_the_stoichiometry_index(model_name):
    """An endpoint missing from _species_idx is silently skipped by
    _build_stoichiometry, so the reaction consumes or produces nothing and
    the integrator sees a non-conservative row. 390 of 900 endpoints were
    orphaned at HEAD with max_species=20."""
    mod = BNGLModule({"model": model_name, "max_species": 10000})
    mod.initialize()

    assert len(mod._species_idx) == len(mod.species_list), (
        "two species rendered to the same str(), so one overwrote the other"
    )
    for rxn in mod.reactions:
        for endpoint in list(rxn.reactants) + list(rxn.products):
            assert str(endpoint) in mod._species_idx, (
                f"{rxn.name}: {endpoint} is not in the species index"
            )


@pytest.mark.parametrize("model_name", ["simple_receptor", "egfr_signaling"])
def test_each_molecule_type_is_conserved_by_the_ode_path(model_name):
    mod = BNGLModule({"model": model_name, "max_species": 10000})
    mod.initialize()

    def totals():
        out = {}
        for i, sp in enumerate(mod.species_list):
            for mol in sp.molecules:
                out[mol.name] = out.get(mol.name, 0.0) + mod.state[i]
        return out

    before = totals()
    for _ in range(200):
        mod.update(1e-6)
    after = totals()

    assert set(before) == set(after)
    for name, start in before.items():
        assert after[name] == pytest.approx(start, rel=1e-9), (
            f"{name} drifted {after[name] - start:+.6g}"
        )


def test_simple_receptor_integrates_to_the_analytic_equilibrium():
    """Pins the network, the rates and the integrator together.

    At equilibrium kon*[L][R] = koff*[LR] with L+LR = R+LR = 1000, so with
    Kd = koff/kon = 100, [LR] solves x^2 - (L0+R0+Kd)x + L0*R0 = 0.
    """
    mod = BNGLModule({"model": "simple_receptor", "max_species": 10000})
    mod.initialize()
    for _ in range(100000):
        mod.update(1e-3)

    counts = {str(sp): mod.state[i] for i, sp in enumerate(mod.species_list)}
    L0 = R0 = 1000.0
    kd = 0.1 / 1e-3
    b = L0 + R0 + kd
    expected = (b - math.sqrt(b * b - 4 * L0 * R0)) / 2
    assert expected == pytest.approx(729.843788, abs=1e-5)

    lr = counts["L(r!1).R(l!1)"]
    assert lr == pytest.approx(expected, rel=1e-6)
    assert counts["L(r)"] + lr == pytest.approx(L0, rel=1e-9)
    assert counts["R(l)"] + lr == pytest.approx(R0, rel=1e-9)


# ── rules whose transformation cannot be derived ─────────────────────────

def test_a_rule_that_changes_a_molecule_is_not_derivable():
    state_changes, bond_changes, derived = derive_operations(
        [parse_pattern("L(r)", LR_TYPES)], [parse_pattern("R(l)", LR_TYPES)]
    )
    assert derived is False
    assert state_changes == [] and bond_changes == []


def test_a_degradation_rule_is_refused_rather_than_guessed_at():
    rule = parse_rule("L(r) -> R(l) 1.0", LR_TYPES, {}, "transmute")
    assert rule.operations_derived is False

    with pytest.raises(ValueError, match="cannot derive the transformation"):
        _expand([rule], [parse_species("L(r)", LR_TYPES)])


def test_an_empty_rule_set_converges_immediately():
    expander = RuleExpander(molecule_types={}, rules=[], parameters={})
    reactions, species = expander.expand(seed_species=[], max_species=10)

    assert reactions == []
    assert species == []
    assert expander.hit_species_limit is False
    assert expander.hit_iteration_limit is False


# ── the reverse direction is matched in its own right ────────────────────

def test_a_complex_seeded_alone_still_dissociates():
    """The reverse direction is expanded by matching the product pattern,
    not by flipping a forward reaction that may never have been generated."""
    _, reactions, species = _expand(
        [_bind_rule()], [parse_species("L(r!1).R(l!1,Y~U)", LR_TYPES)]
    )

    assert len(species) == 3
    rev = [r for r in reactions if r.name.startswith("bind_rev")]
    assert len(rev) == 1
    assert len(rev[0].products) == 2
    assert rev[0].rate == pytest.approx(0.1)


# ── the truncation guard stays reachable as a backstop ───────────────────

def test_a_genuinely_unbounded_rule_still_hits_the_species_limit(polymer_model):
    from cognisom.bngl.parser import load_bngl_model

    model = load_bngl_model(polymer_model)
    expander = RuleExpander(model.molecule_types, model.rules, model.parameters)
    expander.expand(
        seed_species=model.get_seed_species_list(), max_species=12
    )
    assert expander.hit_species_limit is True


def test_the_expansion_of_a_polymer_is_a_chain_not_a_double_booked_site(
    polymer_model,
):
    from cognisom.bngl.parser import load_bngl_model

    model = load_bngl_model(polymer_model)
    expander = RuleExpander(model.molecule_types, model.rules, model.parameters)
    _, species = expander.expand(
        seed_species=model.get_seed_species_list(), max_species=12
    )

    # One species per chain length, each a simple path: every bond has
    # exactly two endpoints and no site is double booked.
    assert sorted(sp.n_molecules for sp in species) == list(range(1, 13))
    for sp in species:
        assert all(len(eps) == 2 for eps in sp.bond_endpoints().values())
        assert len(sp.bond_endpoints()) == sp.n_molecules - 1
        assert len(sp.connected_components()) == 1


def test_canonicalization_stays_exact_for_a_long_polymer(polymer_model):
    from cognisom.bngl.parser import load_bngl_model

    model = load_bngl_model(polymer_model)
    expander = RuleExpander(model.molecule_types, model.rules, model.parameters)
    _, species = expander.expand(
        seed_species=model.get_seed_species_list(), max_species=25
    )
    assert expander.canonicalization_degraded is False
    assert len({sp.canonical_string() for sp in species}) == len(species)


# ── canonicalization is exact, or it refuses ─────────────────────────────

HUB_BNGL = """begin parameters
    kp 1.0
end parameters

begin molecule types
    H(s,s,s,s,s,s,s,s)
    S(x)
end molecule types

begin seed species
    H(s,s,s,s,s,s,s,s) 100
    S(x) 100
end seed species

begin reaction rules
    H(s) + S(x) -> H(s!1).S(x!1) kp
end reaction rules
"""


@pytest.fixture
def hub_model(tmp_path):
    """A hub with eight interchangeable sites. The fully loaded complex has
    one Weisfeiler-Leman tie class of eight satellites, i.e. 8! = 40320
    orderings, which is past the exhaustive tie-break budget."""
    path = tmp_path / "hub.bngl"
    path.write_text(HUB_BNGL)
    return path


def test_repeated_component_names_are_matched_injectively(hub_model):
    """Eight components named 's' on one molecule: a pattern naming one of
    them must land on a free one, and the eight loading steps must give
    nine distinct hub species, not 2**8."""
    from cognisom.bngl.parser import load_bngl_model

    model = load_bngl_model(hub_model)
    expander = RuleExpander(model.molecule_types, model.rules, model.parameters)
    _, species = expander.expand(
        seed_species=model.get_seed_species_list(), max_species=1000
    )
    hubs = [sp for sp in species if any(m.name == "H" for m in sp.molecules)]
    assert len(hubs) == 9
    assert sorted(sp.n_molecules for sp in hubs) == list(range(1, 10))


def test_an_inexact_canonicalization_is_flagged_not_hidden(hub_model):
    from cognisom.bngl.parser import load_bngl_model

    model = load_bngl_model(hub_model)
    expander = RuleExpander(model.molecule_types, model.rules, model.parameters)
    _, species = expander.expand(
        seed_species=model.get_seed_species_list(), max_species=1000
    )
    assert expander.canonicalization_degraded is True
    inexact = [sp for sp in species if not sp.canonical_is_exact]
    assert [sp.n_molecules for sp in inexact] == [9]


def test_the_module_refuses_a_network_it_cannot_canonicalize(hub_model):
    """Exact or refuse -- never silently approximate, because a species
    counted twice inflates every number derived from the network."""
    mod = BNGLModule({"model": str(hub_model), "max_species": 1000})

    with pytest.raises(RuntimeError, match="too\\s+symmetric"):
        mod.initialize()


# ── grafted: the statistical factor, pinned against two closed forms ─────
#
# The 1:1 equilibrium above cannot catch a symmetry-factor error, because
# simple_receptor has multiplicity 1 and |Aut| 1 everywhere. These two do.

LUMPED_BNGL = """begin parameters
    kp 0.9
end parameters

begin molecule types
    R(Y~U~P,Y~U~P)
end molecule types

begin seed species
    R(Y~U,Y~U) 1000
end seed species

begin reaction rules
    R(Y~U) -> R(Y~P) kp
end reaction rules
"""

DISTINGUISHABLE_BNGL = """begin parameters
    kp 0.9
end parameters

begin molecule types
    R(Y1~U~P,Y2~U~P)
end molecule types

begin seed species
    R(Y1~U,Y2~U) 1000
end seed species

begin reaction rules
    R(Y1~U) -> R(Y1~P) kp
    R(Y2~U) -> R(Y2~P) kp
end reaction rules
"""

HOMODIMER_BNGL = """begin parameters
    kon 0.001
    koff 0.1
end parameters

begin molecule types
    A(x)
end molecule types

begin seed species
    A(x) 1000
end seed species

begin reaction rules
    A(x) + A(x) <-> A(x!1).A(x!1) kon, koff
end reaction rules
"""


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def _integrate(path, steps, dt, max_species=100):
    mod = BNGLModule(
        {"model": str(path), "max_species": max_species, "method": "ode"}
    )
    mod.initialize()
    for _ in range(steps):
        mod.update(dt)
    return mod


def test_lumping_identical_sites_reproduces_the_distinguishable_model(tmp_path):
    """The strongest available statement about the statistical factor.

    A receptor with two interchangeable Y sites is the exact lumping of a
    receptor with two labelled ones. Lumping is exact if and only if the
    first phosphorylation carries 2k and the second k, so matching the
    labelled model's trajectory pins the multiplicity rule rather than
    restating the convention.
    """
    from cognisom.bngl.parser import load_bngl_model

    lumped_path = _write(tmp_path, "lumped.bngl", LUMPED_BNGL)
    model = load_bngl_model(lumped_path)
    expander = RuleExpander(model.molecule_types, model.rules, model.parameters)
    reactions, species = expander.expand(
        seed_species=model.get_seed_species_list(), max_species=100
    )

    # UU, UP and PP -- UP and PU are one species, not two.
    assert len(species) == 3
    rates = {str(r.reactants[0]): r.rate for r in reactions}
    assert rates["R(Y~U,Y~U)"] == pytest.approx(1.8), "either site can take it"
    assert rates["R(Y~P,Y~U)"] == pytest.approx(0.9), "only one site is left"

    labelled_path = _write(tmp_path, "labelled.bngl", DISTINGUISHABLE_BNGL)
    lumped = _integrate(lumped_path, steps=4000, dt=1e-3)
    labelled = _integrate(labelled_path, steps=4000, dt=1e-3)

    def by_phosphate_count(module):
        out = {0: 0.0, 1: 0.0, 2: 0.0}
        for key, value in module.get_all_species_counts().items():
            out[key.count("~P")] += float(value)
        return out

    lumped_profile = by_phosphate_count(lumped)
    labelled_profile = by_phosphate_count(labelled)
    assert labelled_profile[2] > 900.0, "the labelled model must reach PP"
    for n in (0, 1, 2):
        assert lumped_profile[n] == pytest.approx(
            labelled_profile[n], rel=1e-12
        )


def test_a_homodimer_integrates_to_its_analytic_equilibrium(tmp_path):
    """A second closed form, chosen because its answer depends on the
    symmetry factor: the forward channel must carry kon/2, so equilibrium
    solves kon*A^2 + koff*A - koff*A_total = 0.
    """
    path = _write(tmp_path, "homodimer.bngl", HOMODIMER_BNGL)

    from cognisom.bngl.parser import load_bngl_model

    model = load_bngl_model(path)
    expander = RuleExpander(model.molecule_types, model.rules, model.parameters)
    reactions, species = expander.expand(
        seed_species=model.get_seed_species_list(), max_species=50
    )
    assert len(species) == 2 and len(reactions) == 2
    forward = [r for r in reactions if len(r.reactants) == 2][0]
    assert forward.rate == pytest.approx(0.0005), "kon/2, not kon"

    mod = _integrate(path, steps=50000, dt=1e-3, max_species=50)
    counts = mod.get_all_species_counts()
    free_a = [v for k, v in counts.items() if "!" not in k][0]

    kon, koff, total = 0.001, 0.1, 1000.0
    analytic = (-koff + math.sqrt(koff ** 2 + 4 * kon * total * koff)) / (2 * kon)
    assert analytic == pytest.approx(270.1562118716424, rel=1e-12)
    assert free_a == pytest.approx(analytic, rel=1e-9)
