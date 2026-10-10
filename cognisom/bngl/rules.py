"""
BNGL Reaction Rules and Rule Expansion
======================================

Defines reaction rules and the rule expander for generating explicit
reaction networks from rule-based specifications.

Reaction rules in BNGL specify transformations using patterns rather
than explicit species. The rule expander generates all possible
concrete reactions by matching patterns against the current species.

Example rule::

    # Ligand binding: L binds to R at l component
    L(r) + R(l) <-> L(r!1).R(l!1) kon, koff

This rule generates reactions for all species containing free L and R.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from .observables import _split_patterns
from .molecules import (
    Bond,
    BondState,
    Embedding,
    Component,
    ComponentInstance,
    ComponentState,
    MoleculeInstance,
    MoleculeType,
    Pattern,
    Species,
    parse_pattern,
    parse_species,
)

log = logging.getLogger(__name__)


# ── Rate Expressions ─────────────────────────────────────────────────────

class RateType(Enum):
    """Type of rate expression."""
    CONSTANT = auto()       # Simple rate constant
    EXPRESSION = auto()     # Mathematical expression with parameters


@dataclass
class RateExpression:
    """
    A rate constant or expression.

    Attributes
    ----------
    value : float
        Numeric value (for constant rates)
    expression : str
        Mathematical expression (for expression rates)
    rate_type : RateType
        Type of rate
    """
    value: float = 0.0
    expression: str = ""
    rate_type: RateType = RateType.CONSTANT

    def evaluate(self, parameters: Dict[str, float] = None) -> float:
        """Evaluate the rate expression."""
        if self.rate_type == RateType.CONSTANT:
            return self.value
        # Simple expression evaluation
        if parameters is None:
            parameters = {}
        try:
            return float(eval(self.expression, {"__builtins__": {}}, parameters))
        except Exception:
            return self.value

    @classmethod
    def from_string(cls, s: str, parameters: Dict[str, float] = None) -> "RateExpression":
        """Parse rate expression from string."""
        s = s.strip()
        if parameters is None:
            parameters = {}

        # Try to parse as number
        try:
            return cls(value=float(s), rate_type=RateType.CONSTANT)
        except ValueError:
            pass

        # Try to evaluate as expression
        try:
            value = float(eval(s, {"__builtins__": {}}, parameters))
            return cls(value=value, expression=s, rate_type=RateType.EXPRESSION)
        except Exception:
            # Store as expression for later evaluation
            return cls(expression=s, rate_type=RateType.EXPRESSION)


# ── Reaction Rules ───────────────────────────────────────────────────────

class RuleType(Enum):
    """Type of reaction rule."""
    UNIDIRECTIONAL = auto()    # Forward only (->)
    BIDIRECTIONAL = auto()     # Reversible (<->)


@dataclass
class StateChange:
    """
    A state change operation in a rule.

    Attributes
    ----------
    molecule_idx : int
        Index of molecule in pattern
    component_name : str
        Name of component
    old_state : Optional[str]
        Required old state (None = any)
    new_state : str
        New state after reaction
    """
    molecule_idx: int
    component_name: str
    old_state: Optional[str]
    new_state: str
    #: Index of the component within the pattern molecule. The name alone
    #: is ambiguous when a molecule declares two components of the same
    #: name; the index is what the embedding resolves.
    comp_idx: int = -1


@dataclass
class BondChange:
    """
    A bond creation/deletion operation in a rule.

    Attributes
    ----------
    operation : str
        'create' or 'delete'
    mol1_idx : int
        Index of first molecule
    comp1_name : str
        Component name on first molecule
    mol2_idx : int
        Index of second molecule (for create)
    comp2_name : str
        Component name on second molecule (for create)
    """
    operation: str  # 'create' or 'delete'
    mol1_idx: int
    comp1_name: str
    mol2_idx: int = -1
    comp2_name: str = ""
    #: Component indices within the pattern molecules; see StateChange.
    comp1_idx: int = -1
    comp2_idx: int = -1


@dataclass
class ReactionRule:
    """
    A BNGL reaction rule.

    Reaction rules define transformations using patterns. When applied
    to concrete species, they generate explicit reactions.

    Attributes
    ----------
    name : str
        Rule name/identifier
    reactant_patterns : List[Pattern]
        Reactant patterns (left side of rule)
    product_patterns : List[Pattern]
        Product patterns (right side of rule)
    forward_rate : RateExpression
        Forward rate constant
    reverse_rate : Optional[RateExpression]
        Reverse rate constant (for bidirectional rules)
    rule_type : RuleType
        Unidirectional or bidirectional

    Example
    -------
    >>> # L + R <-> L.R binding rule
    >>> rule = ReactionRule(
    ...     name="bind_L_R",
    ...     reactant_patterns=[Pattern([l_pattern]), Pattern([r_pattern])],
    ...     product_patterns=[Pattern([lr_complex])],
    ...     forward_rate=RateExpression(value=1e6),
    ...     reverse_rate=RateExpression(value=0.1),
    ...     rule_type=RuleType.BIDIRECTIONAL,
    ... )
    """
    name: str
    reactant_patterns: List[Pattern] = field(default_factory=list)
    product_patterns: List[Pattern] = field(default_factory=list)
    forward_rate: RateExpression = field(default_factory=RateExpression)
    reverse_rate: Optional[RateExpression] = None
    rule_type: RuleType = RuleType.UNIDIRECTIONAL

    # Derived operations, populated by parse_rule via derive_operations.
    state_changes: List[StateChange] = field(default_factory=list)
    bond_changes: List[BondChange] = field(default_factory=list)
    #: False when the two pattern sides could not be mapped onto each other
    #: (a molecule created or destroyed, or molecules named in a different
    #: order). A rule whose operations were never derived cannot be
    #: expanded correctly, so the expander refuses rather than guessing
    #: from the pattern count.
    operations_derived: bool = False

    @property
    def is_bidirectional(self) -> bool:
        return self.rule_type == RuleType.BIDIRECTIONAL

    @property
    def n_reactants(self) -> int:
        return sum(len(p.molecules) for p in self.reactant_patterns)

    def __str__(self) -> str:
        reactants = " + ".join(str(p) for p in self.reactant_patterns)
        products = " + ".join(str(p) for p in self.product_patterns)
        arrow = "<->" if self.is_bidirectional else "->"
        return f"{self.name}: {reactants} {arrow} {products}"


# ── Explicit Reactions ───────────────────────────────────────────────────

@dataclass
class Reaction:
    """
    An explicit (concrete) reaction.

    Generated by applying rules to specific species.

    Attributes
    ----------
    name : str
        Reaction name
    reactants : List[Species]
        Reactant species
    products : List[Species]
        Product species
    rate : float
        Rate constant
    rule : Optional[ReactionRule]
        Source rule (if generated from rule expansion)
    """
    name: str
    reactants: List[Species] = field(default_factory=list)
    products: List[Species] = field(default_factory=list)
    rate: float = 0.0
    rule: Optional[ReactionRule] = None

    def __str__(self) -> str:
        reactants = " + ".join(str(r) for r in self.reactants)
        products = " + ".join(str(p) for p in self.products)
        return f"{self.name}: {reactants} -> {products} [{self.rate}]"

    @property
    def reactant_strings(self) -> List[str]:
        return [str(r) for r in self.reactants]

    @property
    def product_strings(self) -> List[str]:
        return [str(p) for p in self.products]


# ── Directed rules and derived operations ────────────────────────────────

@dataclass
class _DirectedRule:
    """One direction of a ReactionRule, with its transformation derived.

    The expander never asks "is this a binding rule?". It only applies the
    operations in state_changes/bond_changes and then splits the result by
    connectivity, so binding (2->1), dissociation (1->2) and state change
    (1->1) all fall out of the same code path.
    """
    name: str
    rule: ReactionRule
    rule_index: int
    direction: str                      # 'fwd' or 'rev'
    reactant_patterns: List[Pattern]
    product_patterns: List[Pattern]
    rate: float
    state_changes: List[StateChange]
    bond_changes: List[BondChange]
    aut: int = 1

    @property
    def key(self) -> Tuple[int, str]:
        return (self.rule_index, self.direction)


def _flatten_patterns(patterns: List[Pattern]) -> List[Tuple[int, int, MoleculeInstance]]:
    """Flatten a pattern list to (pattern index, molecule index, molecule)."""
    out = []
    for pi, pat in enumerate(patterns):
        for mi, mol in enumerate(pat.molecules):
            out.append((pi, mi, mol))
    return out


def _pattern_molecule_shape(mol: MoleculeInstance) -> str:
    """Bond-id-independent, component-order-independent pattern label."""
    parts = []
    for c in mol.components:
        s = c.name
        if c.state is not None:
            s += "~" + c.state.name
        if c.bond_state == BondState.BOUND:
            s += "!"
        elif c.bond_state == BondState.WILDCARD:
            s += "!?"
        elif c.bond_state == BondState.BOUND_WILDCARD:
            s += "!+"
        parts.append(s)
    return f"{mol.name}({','.join(sorted(parts))})"


def derive_operations(
    reactant_patterns: List[Pattern],
    product_patterns: List[Pattern],
) -> Tuple[List[StateChange], List[BondChange], bool]:
    """
    Derive what a rule DOES by comparing its two pattern sides.

    The Nth reactant molecule is mapped onto the Nth product molecule --
    BNGL's own default molecule mapping -- and the two are checked to have
    the same molecule name. Per component this emits a StateChange when the
    product names a different state, a create half-edge on UNBOUND->BOUND
    and a delete half-edge on BOUND->UNBOUND; half-edges sharing a pattern
    bond label pair into one BondChange.

    Returns
    -------
    (state_changes, bond_changes, derived)
        ``derived`` is False when the two sides cannot be mapped onto each
        other at all -- a molecule created or destroyed, molecules named in
        a different order, or a half-written bond. The caller must refuse
        such a rule rather than guess at it.
    """
    r_flat = _flatten_patterns(reactant_patterns)
    p_flat = _flatten_patterns(product_patterns)
    if not r_flat or len(r_flat) != len(p_flat):
        return [], [], False

    state_changes: List[StateChange] = []
    creates: Dict[int, List[Tuple[int, int, str]]] = {}
    deletes: Dict[int, List[Tuple[int, int, str]]] = {}
    half_deletes: List[Tuple[int, int, str]] = []

    for gi, ((_, _, r_mol), (_, _, p_mol)) in enumerate(zip(r_flat, p_flat)):
        if r_mol.name != p_mol.name:
            return [], [], False

        def by_name(mol):
            d: Dict[str, List[Tuple[int, ComponentInstance]]] = {}
            for ci, c in enumerate(mol.components):
                d.setdefault(c.name, []).append((ci, c))
            return d

        r_by, p_by = by_name(r_mol), by_name(p_mol)
        for name, p_list in p_by.items():
            r_list = r_by.get(name)
            if r_list is None or len(r_list) != len(p_list):
                # The product names a component the reactant pattern does
                # not, so there is no precondition to key the operation on.
                return [], [], False
            for (r_ci, r_c), (_, p_c) in zip(r_list, p_list):
                if p_c.state is not None and (
                    r_c.state is None or r_c.state.name != p_c.state.name
                ):
                    state_changes.append(StateChange(
                        molecule_idx=gi,
                        component_name=name,
                        old_state=r_c.state.name if r_c.state is not None else None,
                        new_state=p_c.state.name,
                        comp_idx=r_ci,
                    ))
                rb, pb = r_c.bond_state, p_c.bond_state
                if rb == BondState.UNBOUND and pb == BondState.BOUND:
                    creates.setdefault(p_c.bond_id, []).append((gi, r_ci, name))
                elif rb == BondState.BOUND and pb == BondState.UNBOUND:
                    deletes.setdefault(r_c.bond_id, []).append((gi, r_ci, name))
                elif rb == BondState.BOUND_WILDCARD and pb == BondState.UNBOUND:
                    # "!+ -> free": break whatever it was bound to.
                    half_deletes.append((gi, r_ci, name))
                elif rb == BondState.UNBOUND and pb in (
                    BondState.BOUND_WILDCARD, BondState.WILDCARD
                ):
                    # Cannot bind to an unnamed partner.
                    return [], [], False
                # BOUND->BOUND, wildcard->wildcard and UNBOUND->UNBOUND are
                # context, not operations.

    bond_changes: List[BondChange] = []
    for bid in sorted(creates, key=lambda b: (b is None, b)):
        eps = creates[bid]
        if len(eps) != 2:
            return [], [], False
        (m1, c1, n1), (m2, c2, n2) = eps
        bond_changes.append(BondChange('create', m1, n1, m2, n2, c1, c2))
    for bid in sorted(deletes, key=lambda b: (b is None, b)):
        eps = deletes[bid]
        if len(eps) == 2:
            (m1, c1, n1), (m2, c2, n2) = eps
            bond_changes.append(BondChange('delete', m1, n1, m2, n2, c1, c2))
        elif len(eps) == 1:
            (m1, c1, n1), = eps
            bond_changes.append(BondChange('delete', m1, n1, -1, "", c1, -1))
        else:
            return [], [], False
    for (m1, c1, n1) in half_deletes:
        bond_changes.append(BondChange('delete', m1, n1, -1, "", c1, -1))

    return state_changes, bond_changes, True


#: Above this many molecules in one rule's reactant patterns the
#: automorphism count is not computed exhaustively. Real rules have one or
#: two; 7! = 5040 permutations is already generous.
AUT_MAX_PATTERN_MOLECULES = 7

#: Ditto for interchangeable components within one pattern molecule.
AUT_MAX_COMPONENT_ORDERINGS = 5040


def pattern_automorphism_count(
    reactant_patterns: List[Pattern],
    state_changes: List[StateChange],
    bond_changes: List[BondChange],
) -> int:
    """
    How many ways the rule's own reactant patterns map onto themselves
    while preserving the operations.

    This is the divisor in the statistical factor. For the symmetric
    homodimerization rule ``EGFR(CR1) + EGFR(CR1) -> EGFR(CR1!1).EGFR(CR1!1)``
    swapping the two reactant slots leaves both the patterns and the bond
    it creates unchanged, so the count is 2 and the rate-derived reaction
    gets k/2 -- BioNetGen's convention. For its reverse, the two molecules
    of ``EGFR(CR1!1).EGFR(CR1!1)`` are also interchangeable, giving 2,
    which cancels the 2 embeddings of that pattern into a symmetric dimer.
    """
    flat = _flatten_patterns(reactant_patterns)
    n = len(flat)
    if n == 0:
        return 1
    if n > AUT_MAX_PATTERN_MOLECULES:
        log.warning(
            "rule has %d pattern molecules; skipping the automorphism "
            "count (statistical factors may be too large by an integer)", n,
        )
        return 1

    block = [pi for (pi, _, _) in flat]
    shapes = [_pattern_molecule_shape(m) for (_, _, m) in flat]
    mols = [m for (_, _, m) in flat]

    def comp_label(c):
        return (
            c.name,
            c.state.name if c.state is not None else None,
            c.bond_state.name,
        )
    labels = [[comp_label(c) for c in m.components] for m in mols]

    # Component bijections from molecule gi onto molecule gj, label-preserving.
    cmaps: Dict[Tuple[int, int], List[Tuple[int, ...]]] = {}
    budget = 1
    for gi in range(n):
        for gj in range(n):
            if shapes[gi] != shapes[gj]:
                continue
            opts = [
                perm for perm in itertools.permutations(range(len(labels[gj])))
                if all(labels[gi][k] == labels[gj][perm[k]]
                       for k in range(len(labels[gi])))
            ]
            if opts:
                cmaps[(gi, gj)] = opts
        budget *= max(1, len(cmaps.get((gi, gi), [()])))
        if budget > AUT_MAX_COMPONENT_ORDERINGS:
            log.warning(
                "rule pattern has too many interchangeable components to "
                "count its automorphisms; statistical factors may be too "
                "large by an integer"
            )
            return 1

    # Bonds, state changes and bond changes keyed on (molecule, COMPONENT
    # INDEX). The component NAME is not enough: a molecule may declare two
    # components of the same name, and permuting those two is exactly the
    # automorphism this count exists to find.
    eps: Dict[int, List[Tuple[int, int]]] = {}
    for gi, mol in enumerate(mols):
        for ci, c in enumerate(mol.components):
            if c.bond_state == BondState.BOUND and c.bond_id is not None:
                eps.setdefault(c.bond_id, []).append((gi, ci))
    bonds = frozenset(
        frozenset(v) for v in (tuple(x) for x in eps.values()) if len(v) == 2
    )

    sc_set = frozenset(
        (s.molecule_idx, s.comp_idx, s.old_state, s.new_state)
        for s in state_changes
    )
    bc_set = set()
    for b in bond_changes:
        if b.mol2_idx >= 0:
            bc_set.add((b.operation, frozenset({
                (b.mol1_idx, b.comp1_idx), (b.mol2_idx, b.comp2_idx)
            })))
        else:
            bc_set.add((b.operation + "_half",
                        frozenset({(b.mol1_idx, b.comp1_idx)})))
    bc_set = frozenset(bc_set)

    count = 0
    for perm in itertools.permutations(range(n)):
        if any(shapes[i] != shapes[perm[i]] for i in range(n)):
            continue
        bmap: Dict[int, int] = {}
        ok = True
        for i in range(n):
            tgt = block[perm[i]]
            if bmap.setdefault(block[i], tgt) != tgt:
                ok = False
                break
        if not ok or len(set(bmap.values())) != len(bmap):
            continue
        choices = [cmaps.get((i, perm[i])) for i in range(n)]
        if any(c is None for c in choices):
            continue
        for cmb in itertools.product(*choices):
            def img(g, c, perm=perm, cmb=cmb):
                return (perm[g], cmb[g][c])
            if frozenset(
                frozenset(img(g, c) for (g, c) in e) for e in bonds
            ) != bonds:
                continue
            if frozenset(
                (perm[m], cmb[m][c], o, nn) for (m, c, o, nn) in sc_set
            ) != sc_set:
                continue
            if frozenset(
                (op, frozenset(img(g, c) for (g, c) in e))
                for op, e in bc_set
            ) != bc_set:
                continue
            count += 1
    return max(count, 1)


# ── Rule Expander ────────────────────────────────────────────────────────

class RuleExpander:
    """
    Expands BNGL rules into explicit reactions.

    Uses network generation algorithm:
    1. Start with seed species
    2. Apply each rule to each species combination
    3. If new species are created, add them and repeat
    4. Stop when no new species or max_species reached

    Attributes
    ----------
    molecule_types : Dict[str, MoleculeType]
        Molecule type definitions
    rules : List[ReactionRule]
        Reaction rules
    parameters : Dict[str, float]
        Parameter values

    Example
    -------
    >>> expander = RuleExpander(model.molecule_types, model.rules)
    >>> reactions, species = expander.expand(
    ...     seed_species=model.seed_species,
    ...     max_species=10000,
    ... )
    """

    def __init__(
        self,
        molecule_types: Dict[str, MoleculeType],
        rules: List[ReactionRule],
        parameters: Dict[str, float] = None,
        apply_statistical_factors: bool = True,
    ):
        self.molecule_types = molecule_types
        self.rules = rules
        self.parameters = parameters or {}
        #: When True (BioNetGen's convention) a rule-derived reaction's rate
        #: is the rule's rate times the number of distinct embeddings that
        #: produce it, divided by the rule's own pattern automorphism count.
        #: Setting it False emits the bare rule rate, which is wrong by an
        #: integer factor for symmetric rules and for a pattern that embeds
        #: into one species more than once.
        self.apply_statistical_factors = apply_statistical_factors
        #: True when the last expand() stopped because it ran into
        #: max_species rather than because no new species were produced. A
        #: network that was truncated is not the network the rules describe,
        #: so anything downstream must refuse to integrate it.
        self.hit_species_limit = False
        #: True when the last expand() stopped because it ran into
        #: max_iterations without converging.
        self.hit_iteration_limit = False
        #: True when some species in the last expand() was too symmetric
        #: for the canonicalization budget, so two isomorphic complexes
        #: could in principle have been counted as two species.
        self.canonicalization_degraded = False

    def expand(
        self,
        seed_species: List[Species] = None,
        max_species: int = 10000,
        max_iterations: int = 100,
    ) -> Tuple[List[Reaction], List[Species]]:
        """
        Expand rules to generate reaction network.

        Parameters
        ----------
        seed_species : List[Species]
            Initial species
        max_species : int
            Maximum number of species to generate
        max_iterations : int
            Maximum expansion iterations

        Returns
        -------
        Tuple[List[Reaction], List[Species]]
            Generated reactions and final species list
        """
        if seed_species is None:
            seed_species = []

        # Species set (keyed on the canonical string, which is invariant to
        # molecule order and bond numbering).
        species_dict: Dict[str, Species] = {}
        new_species: List[Species] = []
        for sp in seed_species:
            key = sp.canonical_string()
            if key not in species_dict:
                species_dict[key] = sp
                new_species.append(sp)

        reactions: List[Reaction] = []
        self.hit_species_limit = False
        self.hit_iteration_limit = True  # cleared below if we converge
        self.canonicalization_degraded = False
        #: Applications already counted, so a reaction is neither emitted
        #: twice nor counted twice toward its own multiplicity.
        self._seen_applications: Set[Tuple] = set()
        self._reaction_by_key: Dict[Tuple, Reaction] = {}
        self._multiplicity: Dict[Tuple, int] = {}

        directed = self._directed_rules()

        for iteration in range(max_iterations):
            if len(species_dict) >= max_species:
                # Terminating here means the network was truncated, not
                # completed. Expansion that runs away usually indicates a
                # defect in the rules or in the expander rather than a
                # genuinely large network -- a single 1:1 binding rule should
                # give 3 species, not max_species of them.
                log.error(
                    "rule expansion hit the species limit (%d) after %d "
                    "iterations; the network is truncated and does not "
                    "represent the rules as written",
                    max_species, iteration,
                )
                self.hit_species_limit = True
                self.hit_iteration_limit = False
                break

            if not new_species and iteration > 0:
                log.debug(f"No new species after {iteration} iterations")
                self.hit_iteration_limit = False
                break

            species_to_process = new_species
            new_species = []

            # Apply each directed rule. The reverse direction is matched
            # against its own reactant pattern rather than synthesized by
            # flipping a forward reaction, so a complex seeded on its own
            # still dissociates.
            for drule in directed:
                self._apply_rule(
                    drule,
                    list(species_dict.values()),
                    species_to_process,
                    species_dict,
                    new_species,
                    reactions,
                    max_species,
                )
                if self.hit_species_limit:
                    break

            if self.hit_species_limit:
                self.hit_iteration_limit = False
                break

        degraded = [
            sp for sp in species_dict.values() if not sp.canonical_is_exact
        ]
        if degraded:
            self.canonicalization_degraded = True
            log.error(
                "%d species were too symmetric to canonicalize exactly; "
                "two isomorphic complexes may have been counted twice",
                len(degraded),
            )

        log.info(f"Rule expansion: {len(species_dict)} species, "
                 f"{len(reactions)} reactions")

        return reactions, list(species_dict.values())

    def _directed_rules(self) -> List["_DirectedRule"]:
        """Flatten self.rules into forward/reverse views with operations.

        Doing this before the expansion loop means a rule whose
        transformation cannot be derived fails immediately, instead of
        after a partial network has already been produced.
        """
        out: List[_DirectedRule] = []
        undecidable: List[str] = []

        for ri, rule in enumerate(self.rules):
            specs = [(
                'fwd', rule.reactant_patterns, rule.product_patterns,
                rule.forward_rate,
            )]
            if rule.is_bidirectional and rule.reverse_rate is not None:
                specs.append((
                    'rev', rule.product_patterns, rule.reactant_patterns,
                    rule.reverse_rate,
                ))
            for direction, r_pats, p_pats, rate in specs:
                sc, bc, derived = derive_operations(r_pats, p_pats)
                if not derived:
                    undecidable.append(
                        f"{rule.name} ({direction}): "
                        f"{' + '.join(str(p) for p in r_pats)} -> "
                        f"{' + '.join(str(p) for p in p_pats)}"
                    )
                    continue
                drule = _DirectedRule(
                    name=f"{rule.name}_{direction}",
                    rule=rule,
                    rule_index=ri,
                    direction=direction,
                    reactant_patterns=r_pats,
                    product_patterns=p_pats,
                    rate=rate.evaluate(self.parameters) if rate is not None else 0.0,
                    state_changes=sc,
                    bond_changes=bc,
                )
                drule.aut = pattern_automorphism_count(r_pats, sc, bc)
                out.append(drule)

        if undecidable:
            raise ValueError(
                "cannot derive the transformation for "
                f"{len(undecidable)} rule direction(s), so expanding them "
                "would mean guessing at what they do: "
                + "; ".join(undecidable)
                + ". Each side of a rule must name the same molecules in the "
                "same order; synthesis and degradation are not supported."
            )
        return out

    def _apply_rule(
        self,
        drule: "_DirectedRule",
        all_species: List[Species],
        focus_species: List[Species],
        species_dict: Dict[str, Species],
        new_species: List[Species],
        reactions: List[Reaction],
        max_species: int,
    ) -> None:
        """Apply one directed rule, interning products and deduplicating."""
        for combo in self._find_matches(drule, all_species, focus_species):
            app_sig = (drule.key,) + tuple(
                (sp.canonical_string(), emb.signature()) for sp, emb in combo
            )
            if app_sig in self._seen_applications:
                continue
            self._seen_applications.add(app_sig)

            reactants = [sp for sp, _ in combo]
            embeddings = [emb for _, emb in combo]
            products = self._apply_transformations(drule, reactants, embeddings)
            if not products:
                continue

            # Intern every product, so a reaction always references the
            # identical object the network stores. Without this a product
            # built in a different molecule order is a different object
            # with a different str(), and anything keyed on str() -- such
            # as BNGLModule._build_stoichiometry -- silently drops that
            # endpoint and integrates a non-conservative row.
            interned: List[Species] = []
            for prod in products:
                key = prod.canonical_string()
                stored = species_dict.get(key)
                if stored is None:
                    species_dict[key] = prod
                    new_species.append(prod)
                    stored = prod
                    if len(species_dict) >= max_species:
                        self.hit_species_limit = True
                interned.append(stored)

            rxn_key = (
                drule.key,
                tuple(sorted(sp.canonical_string() for sp in reactants)),
                tuple(sorted(sp.canonical_string() for sp in interned)),
            )
            rxn = self._reaction_by_key.get(rxn_key)
            if rxn is None:
                rxn = Reaction(
                    name=f"{drule.name}_{len(reactions)}",
                    reactants=reactants,
                    products=interned,
                    rate=0.0,
                    rule=drule.rule,
                )
                self._reaction_by_key[rxn_key] = rxn
                self._multiplicity[rxn_key] = 0
                reactions.append(rxn)

            self._multiplicity[rxn_key] += 1
            if self.apply_statistical_factors:
                rxn.rate = drule.rate * self._multiplicity[rxn_key] / drule.aut
            else:
                rxn.rate = drule.rate

            if self.hit_species_limit:
                return

    def _find_matches(
        self,
        drule: "_DirectedRule",
        all_species: List[Species],
        focus_species: List[Species],
    ) -> List[Tuple[Tuple[Species, Embedding], ...]]:
        """
        Every (species, embedding) combination matching the reactant patterns.

        EVERY embedding, not the first: a pattern can embed into one species
        two ways and give two different products -- phosphorylating the
        EGF-loaded half of a dimer is not the same reaction as
        phosphorylating the free half.

        At least one slot must come from focus_species, so an old pair is
        not re-derived every iteration. Membership is by object identity,
        which is exact because every species is interned.
        """
        patterns = drule.reactant_patterns
        if not patterns:
            return []
        focus_ids = {id(sp) for sp in focus_species}

        per_pattern: List[List[Tuple[Species, Embedding]]] = []
        for pat in patterns:
            opts = [
                (sp, emb)
                for sp in all_species
                for emb in sp.embeddings(pat)
            ]
            if not opts:
                return []
            per_pattern.append(opts)

        return [
            combo
            for combo in itertools.product(*per_pattern)
            if any(id(sp) in focus_ids for sp, _ in combo)
        ]

    def _apply_transformations(
        self,
        drule: "_DirectedRule",
        reactants: List[Species],
        embeddings: List[Embedding],
    ) -> List[Species]:
        """
        Execute the rule's operations on the matched reactants.

        Shape-free: merge every reactant into one working graph, apply
        state changes, then bond deletions, then bond creations, then split
        the graph by connectivity. Binding (2->1), dissociation (1->2) and
        state change (1->1) are all the same code path, so nothing has to
        assume what kind of rule this is.
        """
        merged, mol_offsets = self._merge_all(reactants)

        # Flattened pattern molecule index -> merged molecule / component.
        gmol: Dict[int, int] = {}
        gcomp: Dict[Tuple[int, int], int] = {}
        gi = 0
        for pi, pat in enumerate(drule.reactant_patterns):
            emb = embeddings[pi]
            off = mol_offsets[pi]
            for pk in range(len(pat.molecules)):
                gmol[gi] = off + emb.mol_map[pk]
                for (k, pc), sc in emb.comp_map.items():
                    if k == pk:
                        gcomp[(gi, pc)] = sc
                gi += 1

        # 1. State changes.
        for change in drule.state_changes:
            mi = gmol[change.molecule_idx]
            ci = gcomp[(change.molecule_idx, change.comp_idx)]
            comp = merged.molecules[mi].components[ci]
            new_state = next(
                (s for s in comp.component.states if s.name == change.new_state),
                None,
            )
            if new_state is None:
                log.debug("state %r is not declared on %r", change.new_state, comp.name)
                return []
            comp.state = new_state

        # 2. Bond deletions, resolved against the connectivity as it was
        #    before any of them were applied.
        endpoints = merged.bond_endpoints()
        for change in drule.bond_changes:
            if change.operation != 'delete':
                continue
            mi = gmol[change.mol1_idx]
            ci = gcomp[(change.mol1_idx, change.comp1_idx)]
            c1 = merged.molecules[mi].components[ci]
            if c1.bond_state != BondState.BOUND or c1.bond_id is None:
                return []
            bid = c1.bond_id
            if change.mol2_idx >= 0:
                mj = gmol[change.mol2_idx]
                cj = gcomp[(change.mol2_idx, change.comp2_idx)]
                c2 = merged.molecules[mj].components[cj]
                if c2.bond_state != BondState.BOUND or c2.bond_id != bid:
                    # The pattern says these two sites are bonded to each
                    # other; in this species they are not.
                    return []
            for (mm, cc) in endpoints.get(bid, []):
                comp = merged.molecules[mm].components[cc]
                comp.bond_state = BondState.UNBOUND
                comp.bond_id = None

        # 3. Bond creations.
        next_bond = merged.max_bond_id + 1
        for change in drule.bond_changes:
            if change.operation != 'create':
                continue
            mi = gmol[change.mol1_idx]
            ci = gcomp[(change.mol1_idx, change.comp1_idx)]
            mj = gmol[change.mol2_idx]
            cj = gcomp[(change.mol2_idx, change.comp2_idx)]
            if (mi, ci) == (mj, cj):
                return []                      # a site cannot bond to itself
            c1 = merged.molecules[mi].components[ci]
            c2 = merged.molecules[mj].components[cj]
            if (c1.bond_state == BondState.BOUND
                    or c2.bond_state == BondState.BOUND):
                # Refusing an occupied site is what makes a 1:1 binding
                # rule unable to polymerize: the product's bound sites no
                # longer satisfy the rule's own reactant patterns, and even
                # if a wildcard let them through, this stops it.
                return []
            c1.bond_state = c2.bond_state = BondState.BOUND
            c1.bond_id = c2.bond_id = next_bond
            next_bond += 1

        merged.rebuild_bonds_from_components()
        return self._fragments(merged)

    def _merge_all(
        self,
        reactants: List[Species],
    ) -> Tuple[Species, List[int]]:
        """
        Concatenate the reactants into one working graph.

        Each reactant's bond ids are shifted by a running offset, so two
        reactants that both happen to use ``!1`` stay distinct bonds.
        """
        molecules: List[MoleculeInstance] = []
        offsets: List[int] = []
        shift = 0
        for sp in reactants:
            offsets.append(len(molecules))
            copy = sp.copy()
            for mol in copy.molecules:
                for comp in mol.components:
                    if comp.bond_state == BondState.BOUND and comp.bond_id is not None:
                        comp.bond_id += shift
            shift += sp.max_bond_id
            molecules.extend(copy.molecules)
        merged = Species(molecules=molecules)
        merged.rebuild_bonds_from_components()
        return merged, offsets

    def _fragments(self, sp: Species) -> List[Species]:
        """Split a species into one Species per connected component."""
        out: List[Species] = []
        for group in sp.connected_components():
            remap: Dict[int, int] = {}
            next_id = 1
            molecules: List[MoleculeInstance] = []
            for mi in group:
                mol = sp.molecules[mi]
                comps: List[ComponentInstance] = []
                for c in mol.components:
                    bond_id = c.bond_id
                    if c.bond_state == BondState.BOUND and bond_id is not None:
                        if bond_id not in remap:
                            remap[bond_id] = next_id
                            next_id += 1
                        bond_id = remap[bond_id]
                    comps.append(ComponentInstance(
                        component=c.component,
                        state=c.state,
                        bond_state=c.bond_state,
                        bond_id=bond_id,
                    ))
                molecules.append(MoleculeInstance(
                    molecule_type=mol.molecule_type,
                    components=comps,
                ))
            frag = Species(molecules=molecules)
            frag.rebuild_bonds_from_components()
            out.append(frag)
        return out


# ── Utility Functions ────────────────────────────────────────────────────

def parse_rule(
    rule_str: str,
    molecule_types: Dict[str, MoleculeType],
    parameters: Dict[str, float] = None,
    name: str = "rule",
) -> ReactionRule:
    """
    Parse a reaction rule from BNGL string.

    Format: reactants -> products rate
            reactants <-> products kon, koff

    Example: "L(r) + R(l) <-> L(r!1).R(l!1) kon, koff"
    """
    if parameters is None:
        parameters = {}

    # Determine direction
    if '<->' in rule_str:
        rule_type = RuleType.BIDIRECTIONAL
        parts = rule_str.split('<->')
    else:
        rule_type = RuleType.UNIDIRECTIONAL
        parts = rule_str.split('->')

    if len(parts) != 2:
        raise ValueError(f"Invalid rule format: {rule_str}")

    reactant_str = parts[0].strip()
    product_rate_str = parts[1].strip()

    # Split products and rate(s)
    # Simple heuristic: rate comes after last species
    # This is simplified - real parser would be more robust
    tokens = product_rate_str.split()
    product_tokens = []
    rate_tokens = []

    for i, token in enumerate(tokens):
        # If token looks like rate (number or parameter name)
        if _looks_like_rate(token, parameters):
            rate_tokens = tokens[i:]
            break
        product_tokens.append(token)

    product_str = " ".join(product_tokens)

    # Parse reactants
    reactant_patterns = []
    for r_str in _split_patterns(reactant_str):
        r_str = r_str.strip()
        if r_str:
            reactant_patterns.append(parse_pattern(r_str, molecule_types))

    # Parse products
    product_patterns = []
    for p_str in _split_patterns(product_str):
        p_str = p_str.strip()
        if p_str:
            product_patterns.append(parse_pattern(p_str, molecule_types))

    # Parse rates
    forward_rate = RateExpression(value=1.0)
    reverse_rate = None

    if rate_tokens:
        rate_str = " ".join(rate_tokens).strip().rstrip(',')
        if ',' in rate_str:
            rate_parts = rate_str.split(',')
            forward_rate = RateExpression.from_string(rate_parts[0].strip(), parameters)
            reverse_rate = RateExpression.from_string(rate_parts[1].strip(), parameters)
        else:
            forward_rate = RateExpression.from_string(rate_str, parameters)

    state_changes, bond_changes, derived = derive_operations(
        reactant_patterns, product_patterns
    )

    return ReactionRule(
        name=name,
        reactant_patterns=reactant_patterns,
        product_patterns=product_patterns,
        forward_rate=forward_rate,
        reverse_rate=reverse_rate,
        rule_type=rule_type,
        state_changes=state_changes,
        bond_changes=bond_changes,
        operations_derived=derived,
    )


def _looks_like_rate(token: str, parameters: Dict[str, float]) -> bool:
    """Check if token looks like a rate constant."""
    token = token.strip().rstrip(',')
    try:
        float(token)
        return True
    except ValueError:
        pass
    return token in parameters
