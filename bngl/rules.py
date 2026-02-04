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

from .molecules import (
    Bond,
    BondState,
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

    # Derived operations (computed from patterns)
    state_changes: List[StateChange] = field(default_factory=list)
    bond_changes: List[BondChange] = field(default_factory=list)

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
    ):
        self.molecule_types = molecule_types
        self.rules = rules
        self.parameters = parameters or {}

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

        # Species set (using canonical strings for deduplication)
        species_dict: Dict[str, Species] = {}
        for sp in seed_species:
            species_dict[sp.canonical_string()] = sp

        reactions: List[Reaction] = []
        new_species = list(seed_species)

        for iteration in range(max_iterations):
            if len(species_dict) >= max_species:
                log.warning(f"Max species limit ({max_species}) reached")
                break

            if not new_species and iteration > 0:
                log.debug(f"No new species after {iteration} iterations")
                break

            species_to_process = new_species
            new_species = []

            # Apply each rule
            for rule in self.rules:
                new_rxns, generated = self._apply_rule(
                    rule,
                    list(species_dict.values()),
                    species_to_process,
                )

                reactions.extend(new_rxns)

                # Check for new species
                for sp in generated:
                    key = sp.canonical_string()
                    if key not in species_dict:
                        species_dict[key] = sp
                        new_species.append(sp)

                        if len(species_dict) >= max_species:
                            break

        log.info(f"Rule expansion: {len(species_dict)} species, "
                 f"{len(reactions)} reactions")

        return reactions, list(species_dict.values())

    def _apply_rule(
        self,
        rule: ReactionRule,
        all_species: List[Species],
        focus_species: List[Species],
    ) -> Tuple[List[Reaction], List[Species]]:
        """
        Apply a rule to species, returning new reactions and species.

        Parameters
        ----------
        rule : ReactionRule
            Rule to apply
        all_species : List[Species]
            All known species
        focus_species : List[Species]
            New species to specifically check

        Returns
        -------
        Tuple[List[Reaction], List[Species]]
            New reactions and product species
        """
        reactions = []
        new_species = []

        # Find all matching reactant combinations
        matches = self._find_matches(rule, all_species, focus_species)

        for reactants in matches:
            # Generate products by applying rule transformations
            products = self._apply_transformations(rule, reactants)

            if products:
                # Create forward reaction
                rate = rule.forward_rate.evaluate(self.parameters)
                rxn = Reaction(
                    name=f"{rule.name}_fwd_{len(reactions)}",
                    reactants=list(reactants),
                    products=products,
                    rate=rate,
                    rule=rule,
                )
                reactions.append(rxn)
                new_species.extend(products)

                # Create reverse reaction if bidirectional
                if rule.is_bidirectional and rule.reverse_rate:
                    rev_rate = rule.reverse_rate.evaluate(self.parameters)
                    rev_rxn = Reaction(
                        name=f"{rule.name}_rev_{len(reactions)}",
                        reactants=products,
                        products=list(reactants),
                        rate=rev_rate,
                        rule=rule,
                    )
                    reactions.append(rev_rxn)

        return reactions, new_species

    def _find_matches(
        self,
        rule: ReactionRule,
        all_species: List[Species],
        focus_species: List[Species],
    ) -> List[Tuple[Species, ...]]:
        """
        Find all species combinations matching rule reactants.

        At least one species must be from focus_species to avoid
        regenerating existing reactions.
        """
        matches = []

        n_patterns = len(rule.reactant_patterns)
        if n_patterns == 0:
            return matches

        # Generate all combinations
        if n_patterns == 1:
            # Single reactant
            for sp in focus_species:
                if sp.matches(rule.reactant_patterns[0]):
                    matches.append((sp,))
        elif n_patterns == 2:
            # Two reactants - need at least one from focus
            for sp1 in all_species:
                if not sp1.matches(rule.reactant_patterns[0]):
                    continue
                for sp2 in all_species:
                    if not sp2.matches(rule.reactant_patterns[1]):
                        continue
                    # At least one must be new
                    if sp1 in focus_species or sp2 in focus_species:
                        matches.append((sp1, sp2))
        else:
            # General case - N reactants
            for combo in itertools.product(*([all_species] * n_patterns)):
                if any(sp in focus_species for sp in combo):
                    if all(
                        sp.matches(pattern)
                        for sp, pattern in zip(combo, rule.reactant_patterns)
                    ):
                        matches.append(combo)

        return matches

    def _apply_transformations(
        self,
        rule: ReactionRule,
        reactants: Tuple[Species, ...],
    ) -> List[Species]:
        """
        Apply rule transformations to reactants, producing products.

        This is simplified - a full implementation would:
        1. Identify mapping between pattern and species
        2. Apply state changes
        3. Create/delete bonds
        4. Merge/split complexes as needed
        """
        # Simple binding rule: two separate species -> one complex
        if len(reactants) == 2 and len(rule.product_patterns) == 1:
            # Assuming this is a binding reaction
            sp1, sp2 = reactants
            product = self._merge_species(sp1, sp2)
            return [product] if product else []

        # Simple unbinding: one complex -> two species
        if len(reactants) == 1 and len(rule.product_patterns) == 2:
            # Assuming this is an unbinding reaction
            sp = reactants[0]
            products = self._split_species(sp)
            return products if products else []

        # State change in single species
        if len(reactants) == 1 and len(rule.product_patterns) == 1:
            sp = reactants[0]
            # Apply state changes from rule
            if rule.state_changes:
                new_sp = self._apply_state_changes(sp, rule.state_changes)
                return [new_sp] if new_sp else []
            # Return copy
            return [sp.copy()]

        # Default: return copies of reactants (no transformation)
        return [r.copy() for r in reactants]

    def _merge_species(self, sp1: Species, sp2: Species) -> Optional[Species]:
        """Merge two species into one complex."""
        # Simple merge: concatenate molecules
        new_molecules = []
        for m in sp1.molecules:
            new_molecules.append(MoleculeInstance(
                molecule_type=m.molecule_type,
                components=[ComponentInstance(
                    component=c.component,
                    state=c.state,
                    bond_state=c.bond_state,
                    bond_id=c.bond_id,
                ) for c in m.components]
            ))

        for m in sp2.molecules:
            new_molecules.append(MoleculeInstance(
                molecule_type=m.molecule_type,
                components=[ComponentInstance(
                    component=c.component,
                    state=c.state,
                    bond_state=c.bond_state,
                    bond_id=c.bond_id,
                ) for c in m.components]
            ))

        return Species(molecules=new_molecules, bonds=list(sp1.bonds) + list(sp2.bonds))

    def _split_species(self, sp: Species) -> List[Species]:
        """Split a species into constituent molecules."""
        return [
            Species(molecules=[MoleculeInstance(
                molecule_type=m.molecule_type,
                components=[ComponentInstance(
                    component=c.component,
                    state=c.state,
                    bond_state=BondState.UNBOUND,
                    bond_id=None,
                ) for c in m.components]
            )])
            for m in sp.molecules
        ]

    def _apply_state_changes(
        self,
        sp: Species,
        changes: List[StateChange],
    ) -> Optional[Species]:
        """Apply state changes to a species."""
        new_sp = sp.copy()

        for change in changes:
            if change.molecule_idx < len(new_sp.molecules):
                mol = new_sp.molecules[change.molecule_idx]
                comp = mol.get_component(change.component_name)
                if comp:
                    # Find new state
                    for s in comp.component.states:
                        if s.name == change.new_state:
                            comp.state = s
                            break

        return new_sp


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
    for r_str in reactant_str.split('+'):
        r_str = r_str.strip()
        if r_str:
            reactant_patterns.append(parse_pattern(r_str, molecule_types))

    # Parse products
    product_patterns = []
    for p_str in product_str.split('+'):
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

    return ReactionRule(
        name=name,
        reactant_patterns=reactant_patterns,
        product_patterns=product_patterns,
        forward_rate=forward_rate,
        reverse_rate=reverse_rate,
        rule_type=rule_type,
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
