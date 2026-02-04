"""
BioNetGen Language (BNGL) Package
=================================

Rule-based modeling for biological networks with combinatorial complexity.
Parses BNGL model files and expands reaction rules into explicit reactions.

This package provides:
- BNGL parser for model specification
- Molecule type definitions with components and states
- Pattern matching for species
- Rule expansion to generate reaction networks
- Observable computation

VCell Parity Phase 4 - Rule-based modeling.

Usage::

    from cognisom.bngl import BNGLParser, BNGLModel

    # Parse a BNGL model file
    parser = BNGLParser()
    model = parser.parse_file("model.bngl")

    # Or parse from string
    model = parser.parse('''
        begin model
            begin molecule types
                R(l,Y~U~P)
            end molecule types
            ...
        end model
    ''')

    # Expand rules to get explicit reactions
    from cognisom.bngl import RuleExpander
    expander = RuleExpander(model)
    reactions = expander.expand(max_species=10000)

References:
    Faeder, J. R., Blinov, M. L., & Bhavsar, W. S. (2009).
    Rule-based modeling of biochemical systems with BioNetGen.
    Methods Mol. Biol., 500, 113-167.
"""

from .molecules import (
    Component,
    ComponentState,
    MoleculeType,
    MoleculeInstance,
    Bond,
    Species,
    Pattern,
)
from .rules import (
    RateExpression,
    ReactionRule,
    RuleExpander,
)
from .parser import (
    BNGLParser,
    BNGLModel,
)
from .observables import (
    Observable,
    ObservableType,
)

__all__ = [
    # Molecules
    "Component",
    "ComponentState",
    "MoleculeType",
    "MoleculeInstance",
    "Bond",
    "Species",
    "Pattern",
    # Rules
    "RateExpression",
    "ReactionRule",
    "RuleExpander",
    # Parser
    "BNGLParser",
    "BNGLModel",
    # Observables
    "Observable",
    "ObservableType",
]
