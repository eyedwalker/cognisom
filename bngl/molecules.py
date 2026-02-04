"""
BNGL Molecule Types and Species
===============================

Defines molecule types, components, states, and species for rule-based modeling.

In BNGL, molecules have:
- Components (binding sites, phosphorylation sites, etc.)
- States for each component (e.g., phosphorylated/unphosphorylated)
- Bonds between components of different molecules

Example BNGL molecule type::

    R(l,Y~U~P)

This defines a receptor R with:
- Component 'l' (ligand binding site, no explicit states)
- Component 'Y' with states 'U' (unphosphorylated) and 'P' (phosphorylated)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union


# ── Component and State Definitions ──────────────────────────────────────

@dataclass(frozen=True)
class ComponentState:
    """A possible state of a component (e.g., U for unphosphorylated)."""
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass
class Component:
    """
    A component of a molecule (binding site, modification site, etc.).

    Components can have:
    - Multiple possible states (e.g., U, P for phosphorylation)
    - Binding capability (can form bonds with other components)

    Attributes
    ----------
    name : str
        Component name (e.g., 'l', 'Y', 'SH2')
    states : List[ComponentState]
        Possible states (empty if no states defined)
    """
    name: str
    states: List[ComponentState] = field(default_factory=list)

    @property
    def has_states(self) -> bool:
        return len(self.states) > 0

    @property
    def state_names(self) -> List[str]:
        return [s.name for s in self.states]

    def __str__(self) -> str:
        if self.states:
            states_str = "~".join(s.name for s in self.states)
            return f"{self.name}~{states_str}"
        return self.name


@dataclass
class MoleculeType:
    """
    Definition of a molecule type.

    A molecule type defines the structure: what components it has
    and what states those components can have.

    Attributes
    ----------
    name : str
        Molecule name (e.g., 'R', 'L', 'Sos')
    components : List[Component]
        Component definitions

    Example
    -------
    >>> # Receptor with ligand site and phosphorylation site
    >>> r = MoleculeType("R", [
    ...     Component("l"),           # binding site, no states
    ...     Component("Y", [ComponentState("U"), ComponentState("P")]),
    ... ])
    """
    name: str
    components: List[Component] = field(default_factory=list)

    @property
    def component_names(self) -> List[str]:
        return [c.name for c in self.components]

    def get_component(self, name: str) -> Optional[Component]:
        for c in self.components:
            if c.name == name:
                return c
        return None

    def __str__(self) -> str:
        if not self.components:
            return self.name
        comps = ",".join(str(c) for c in self.components)
        return f"{self.name}({comps})"

    @classmethod
    def from_string(cls, s: str) -> "MoleculeType":
        """
        Parse molecule type from BNGL string.

        Example: "R(l,Y~U~P)" -> MoleculeType with components l and Y
        """
        # Match: Name(component1,component2,...)
        match = re.match(r'(\w+)\(([^)]*)\)', s.strip())
        if not match:
            # Simple molecule with no components
            return cls(name=s.strip())

        name = match.group(1)
        comp_str = match.group(2)

        components = []
        if comp_str:
            for comp in comp_str.split(','):
                comp = comp.strip()
                if '~' in comp:
                    # Component with states: Y~U~P
                    parts = comp.split('~')
                    comp_name = parts[0]
                    states = [ComponentState(p) for p in parts[1:] if p]
                    components.append(Component(comp_name, states))
                else:
                    # Component without states
                    components.append(Component(comp))

        return cls(name=name, components=components)


# ── Molecule Instances and Bonds ─────────────────────────────────────────

class BondState(Enum):
    """State of a component's bond."""
    UNBOUND = auto()        # Component is free (.)
    BOUND = auto()          # Component is bound (!N where N is bond number)
    WILDCARD = auto()       # Any bond state (!)
    BOUND_WILDCARD = auto() # Bound to something (!+)


@dataclass
class ComponentInstance:
    """
    Instance of a component with specific state and bond.

    Attributes
    ----------
    component : Component
        Reference to component definition
    state : Optional[ComponentState]
        Current state (None if any state matches)
    bond_state : BondState
        Bond state
    bond_id : Optional[int]
        Bond identifier (for BOUND state)
    """
    component: Component
    state: Optional[ComponentState] = None
    bond_state: BondState = BondState.UNBOUND
    bond_id: Optional[int] = None

    @property
    def name(self) -> str:
        return self.component.name

    def __str__(self) -> str:
        s = self.name
        if self.state:
            s += f"~{self.state.name}"
        if self.bond_state == BondState.UNBOUND:
            pass  # No bond notation
        elif self.bond_state == BondState.BOUND:
            s += f"!{self.bond_id}"
        elif self.bond_state == BondState.WILDCARD:
            s += "!"
        elif self.bond_state == BondState.BOUND_WILDCARD:
            s += "!+"
        return s

    def matches(self, other: "ComponentInstance") -> bool:
        """Check if this component instance matches another (for pattern matching)."""
        if self.name != other.name:
            return False
        # State matching
        if self.state is not None and other.state is not None:
            if self.state.name != other.state.name:
                return False
        # Bond matching
        if self.bond_state == BondState.WILDCARD:
            return True  # Matches any bond state
        if self.bond_state == BondState.BOUND_WILDCARD:
            return other.bond_state == BondState.BOUND
        return self.bond_state == other.bond_state


@dataclass
class MoleculeInstance:
    """
    Instance of a molecule with specific component states and bonds.

    Attributes
    ----------
    molecule_type : MoleculeType
        Reference to molecule type definition
    components : List[ComponentInstance]
        Component instances with states and bonds
    """
    molecule_type: MoleculeType
    components: List[ComponentInstance] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.molecule_type.name

    def get_component(self, name: str) -> Optional[ComponentInstance]:
        for c in self.components:
            if c.name == name:
                return c
        return None

    def __str__(self) -> str:
        if not self.components:
            return self.name
        comps = ",".join(str(c) for c in self.components)
        return f"{self.name}({comps})"

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other) -> bool:
        if not isinstance(other, MoleculeInstance):
            return False
        return str(self) == str(other)

    def matches(self, pattern: "MoleculeInstance") -> bool:
        """Check if this molecule matches a pattern."""
        if self.name != pattern.name:
            return False
        # Check each pattern component
        for p_comp in pattern.components:
            found = False
            for s_comp in self.components:
                if s_comp.matches(p_comp):
                    found = True
                    break
            if not found:
                return False
        return True

    @classmethod
    def from_type(cls, mol_type: MoleculeType) -> "MoleculeInstance":
        """Create instance from type with default (first) states."""
        components = []
        for comp in mol_type.components:
            state = comp.states[0] if comp.states else None
            components.append(ComponentInstance(
                component=comp,
                state=state,
                bond_state=BondState.UNBOUND,
            ))
        return cls(molecule_type=mol_type, components=components)


@dataclass(frozen=True)
class Bond:
    """
    A bond between two molecule components.

    Bonds are identified by an integer ID and connect specific
    components of two molecules.

    Attributes
    ----------
    id : int
        Bond identifier
    mol1_idx : int
        Index of first molecule in species
    comp1_name : str
        Name of component in first molecule
    mol2_idx : int
        Index of second molecule in species
    comp2_name : str
        Name of component in second molecule
    """
    id: int
    mol1_idx: int
    comp1_name: str
    mol2_idx: int
    comp2_name: str


# ── Species and Patterns ─────────────────────────────────────────────────

@dataclass
class Species:
    """
    A molecular species (complex of one or more molecules).

    A species is a fully specified molecular entity with all states
    and bonds defined.

    Attributes
    ----------
    molecules : List[MoleculeInstance]
        List of molecule instances in the complex
    bonds : List[Bond]
        Bonds between molecules

    Example
    -------
    >>> # L.R complex: L(r!1).R(l!1,Y~U)
    >>> species = Species(
    ...     molecules=[l_instance, r_instance],
    ...     bonds=[Bond(1, 0, 'r', 1, 'l')],
    ... )
    """
    molecules: List[MoleculeInstance] = field(default_factory=list)
    bonds: List[Bond] = field(default_factory=list)

    @property
    def n_molecules(self) -> int:
        return len(self.molecules)

    def __str__(self) -> str:
        return ".".join(str(m) for m in self.molecules)

    def __hash__(self) -> int:
        return hash(self.canonical_string())

    def __eq__(self, other) -> bool:
        if not isinstance(other, Species):
            return False
        return self.canonical_string() == other.canonical_string()

    def canonical_string(self) -> str:
        """Return canonical string representation (sorted)."""
        # Sort molecules alphabetically for canonical form
        mol_strs = sorted(str(m) for m in self.molecules)
        return ".".join(mol_strs)

    def matches(self, pattern: "Pattern") -> bool:
        """Check if this species matches a pattern."""
        # Each molecule in pattern must match some molecule in species
        used = set()
        for p_mol in pattern.molecules:
            matched = False
            for i, s_mol in enumerate(self.molecules):
                if i not in used and s_mol.matches(p_mol):
                    used.add(i)
                    matched = True
                    break
            if not matched:
                return False
        return True

    def copy(self) -> "Species":
        """Create a deep copy."""
        return Species(
            molecules=[MoleculeInstance(
                molecule_type=m.molecule_type,
                components=[ComponentInstance(
                    component=c.component,
                    state=c.state,
                    bond_state=c.bond_state,
                    bond_id=c.bond_id,
                ) for c in m.components]
            ) for m in self.molecules],
            bonds=list(self.bonds),
        )


@dataclass
class Pattern:
    """
    A pattern for matching species.

    Patterns are like species but can have wildcards and unspecified
    states/bonds. Used in reaction rules to match reactants.

    Attributes
    ----------
    molecules : List[MoleculeInstance]
        Pattern molecules (may have wildcards)
    """
    molecules: List[MoleculeInstance] = field(default_factory=list)

    def __str__(self) -> str:
        return ".".join(str(m) for m in self.molecules)

    @property
    def is_single_molecule(self) -> bool:
        return len(self.molecules) == 1


# ── Pattern Parsing ──────────────────────────────────────────────────────

def parse_component_instance(
    comp_str: str,
    comp_def: Component,
) -> ComponentInstance:
    """
    Parse a component instance from string.

    Examples:
        "l" -> unbound, no state
        "l!1" -> bound with bond 1
        "Y~P" -> state P, unbound
        "Y~P!1" -> state P, bound with bond 1
        "Y!" -> any bond state
        "Y!+" -> must be bound
    """
    # Parse state
    state = None
    if '~' in comp_str:
        parts = comp_str.split('~')
        comp_name = parts[0]
        state_str = parts[1].split('!')[0] if '!' in parts[1] else parts[1]
        for s in comp_def.states:
            if s.name == state_str:
                state = s
                break
    else:
        comp_name = comp_str.split('!')[0]

    # Parse bond
    bond_state = BondState.UNBOUND
    bond_id = None
    if '!' in comp_str:
        bond_part = comp_str.split('!')[-1]
        if bond_part == '':
            bond_state = BondState.WILDCARD
        elif bond_part == '+':
            bond_state = BondState.BOUND_WILDCARD
        else:
            try:
                bond_id = int(bond_part)
                bond_state = BondState.BOUND
            except ValueError:
                bond_state = BondState.WILDCARD

    return ComponentInstance(
        component=comp_def,
        state=state,
        bond_state=bond_state,
        bond_id=bond_id,
    )


def parse_molecule_instance(
    mol_str: str,
    mol_types: Dict[str, MoleculeType],
) -> MoleculeInstance:
    """
    Parse a molecule instance from BNGL string.

    Example: "R(l!1,Y~P)" -> MoleculeInstance with bound l and phosphorylated Y
    """
    match = re.match(r'(\w+)\(([^)]*)\)', mol_str.strip())
    if not match:
        # Simple molecule
        name = mol_str.strip()
        if name not in mol_types:
            raise ValueError(f"Unknown molecule type: {name}")
        return MoleculeInstance.from_type(mol_types[name])

    name = match.group(1)
    if name not in mol_types:
        raise ValueError(f"Unknown molecule type: {name}")

    mol_type = mol_types[name]
    comp_str = match.group(2)

    components = []
    if comp_str:
        for comp in comp_str.split(','):
            comp = comp.strip()
            # Find component name
            comp_name = comp.split('~')[0].split('!')[0]
            comp_def = mol_type.get_component(comp_name)
            if comp_def is None:
                raise ValueError(f"Unknown component {comp_name} in {name}")
            components.append(parse_component_instance(comp, comp_def))

    return MoleculeInstance(molecule_type=mol_type, components=components)


def parse_species(
    species_str: str,
    mol_types: Dict[str, MoleculeType],
) -> Species:
    """
    Parse a species from BNGL string.

    Example: "L(r!1).R(l!1,Y~U)" -> Species with L-R complex
    """
    molecules = []
    for mol_str in species_str.split('.'):
        mol_str = mol_str.strip()
        if mol_str:
            molecules.append(parse_molecule_instance(mol_str, mol_types))

    # Extract bonds from component instances
    bonds = []
    bond_endpoints = {}  # bond_id -> [(mol_idx, comp_name), ...]

    for mol_idx, mol in enumerate(molecules):
        for comp in mol.components:
            if comp.bond_state == BondState.BOUND and comp.bond_id is not None:
                if comp.bond_id not in bond_endpoints:
                    bond_endpoints[comp.bond_id] = []
                bond_endpoints[comp.bond_id].append((mol_idx, comp.name))

    for bond_id, endpoints in bond_endpoints.items():
        if len(endpoints) == 2:
            bonds.append(Bond(
                id=bond_id,
                mol1_idx=endpoints[0][0],
                comp1_name=endpoints[0][1],
                mol2_idx=endpoints[1][0],
                comp2_name=endpoints[1][1],
            ))

    return Species(molecules=molecules, bonds=bonds)


def parse_pattern(
    pattern_str: str,
    mol_types: Dict[str, MoleculeType],
) -> Pattern:
    """
    Parse a pattern from BNGL string.

    Same syntax as species but can have wildcards.
    """
    molecules = []
    for mol_str in pattern_str.split('.'):
        mol_str = mol_str.strip()
        if mol_str:
            molecules.append(parse_molecule_instance(mol_str, mol_types))

    return Pattern(molecules=molecules)
