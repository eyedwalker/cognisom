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

import itertools
import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

log = logging.getLogger(__name__)


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

    def matches(self, pattern: "ComponentInstance") -> bool:
        """
        Does this component (a concrete species component) satisfy ``pattern``?

        The wildcards live on the PATTERN side, never on the species side:
        callers pass ``s_comp.matches(p_comp)``. Testing them on ``self``
        makes every ``!+`` pattern unmatchable, because the species side
        says BOUND and the comparison is BOUND == BOUND_WILDCARD.
        """
        if self.name != pattern.name:
            return False
        # State: a pattern that does not name a state matches any state.
        if pattern.state is not None:
            if self.state is None or self.state.name != pattern.state.name:
                return False
        # Bond
        if pattern.bond_state == BondState.WILDCARD:
            return True                                    # '!'  any bond state
        if pattern.bond_state == BondState.BOUND_WILDCARD:
            return self.bond_state == BondState.BOUND       # '!+' bound to anything
        if pattern.bond_state == BondState.BOUND:
            # A number in a pattern is a local label, not an identity: which
            # bond it is gets resolved by Species.embeddings, not here.
            return self.bond_state == BondState.BOUND
        return self.bond_state == BondState.UNBOUND


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
        """
        Check if this molecule matches a pattern.

        Pattern components are matched injectively: two pattern components
        with the same name must land on two different components of this
        molecule, not both on the same one.
        """
        if self.name != pattern.name:
            return False
        return bool(_component_maps(self, pattern, limit=1))

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


# ── Pattern embeddings ───────────────────────────────────────────────────

@dataclass(frozen=True)
class Embedding:
    """
    One injective embedding of a pattern into a species.

    Attributes
    ----------
    mol_map : Tuple[int, ...]
        ``mol_map[k]`` is the species molecule index that pattern molecule
        ``k`` was mapped onto.
    comp_map : Dict[Tuple[int, int], int]
        ``comp_map[(k, pc)]`` is the component index, within that species
        molecule, that pattern component ``pc`` of pattern molecule ``k``
        was mapped onto.
    """
    mol_map: Tuple[int, ...]
    comp_map: Dict[Tuple[int, int], int]

    def __hash__(self) -> int:
        return hash((self.mol_map, tuple(sorted(self.comp_map.items()))))

    def signature(self) -> Tuple:
        """Order-independent, hashable identity for this embedding."""
        return (self.mol_map, tuple(sorted(self.comp_map.items())))


def _component_maps(
    s_mol: MoleculeInstance,
    p_mol: MoleculeInstance,
    limit: int = 0,
) -> List[Dict[int, int]]:
    """
    Every injective map {pattern component index -> species component index}
    under which each species component satisfies its pattern component.
    """
    n_p = len(p_mol.components)
    if n_p > len(s_mol.components):
        return []
    out: List[Dict[int, int]] = []
    cur: Dict[int, int] = {}
    used: Set[int] = set()

    def recurse(pc: int) -> None:
        if limit and len(out) >= limit:
            return
        if pc == n_p:
            out.append(dict(cur))
            return
        p_comp = p_mol.components[pc]
        for sc, s_comp in enumerate(s_mol.components):
            if sc in used:
                continue
            if not s_comp.matches(p_comp):
                continue
            used.add(sc)
            cur[pc] = sc
            recurse(pc + 1)
            del cur[pc]
            used.discard(sc)
            if limit and len(out) >= limit:
                return

    recurse(0)
    return out


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
    #: Memoized canonical string. Any in-place mutation of molecules,
    #: states or bonds must clear it via invalidate_canonical().
    _canon: Optional[str] = field(default=None, compare=False, repr=False)
    _canon_exact: bool = field(default=True, compare=False, repr=False)

    def invalidate_canonical(self) -> None:
        self._canon = None
        self._canon_exact = True

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

    # ── Bond bookkeeping ─────────────────────────────────────────────────
    #
    # The component bond ids (ComponentInstance.bond_state + .bond_id) are
    # the single source of truth for connectivity, because __str__ renders
    # from them. Species.bonds is strictly derived: anything that mutates
    # components must call rebuild_bonds_from_components().

    def bond_endpoints(self) -> Dict[int, List[Tuple[int, int]]]:
        """Map bond id -> list of (molecule index, component index).

        Addressing endpoints by component INDEX rather than name is what
        makes a molecule with two components of the same name unambiguous.
        """
        out: Dict[int, List[Tuple[int, int]]] = {}
        for mi, mol in enumerate(self.molecules):
            for ci, comp in enumerate(mol.components):
                if comp.bond_state == BondState.BOUND and comp.bond_id is not None:
                    out.setdefault(comp.bond_id, []).append((mi, ci))
        return out

    @property
    def max_bond_id(self) -> int:
        """Largest bond id in use, or 0 when the species has no bonds."""
        ids = [
            c.bond_id
            for m in self.molecules
            for c in m.components
            if c.bond_state == BondState.BOUND and c.bond_id is not None
        ]
        return max(ids) if ids else 0

    def bond_partner(self, mol_idx: int, comp_idx: int) -> Optional[Tuple[int, int]]:
        """Return the (molecule index, component index) bonded to this site."""
        comp = self.molecules[mol_idx].components[comp_idx]
        if comp.bond_state != BondState.BOUND or comp.bond_id is None:
            return None
        for ep in self.bond_endpoints().get(comp.bond_id, []):
            if ep != (mol_idx, comp_idx):
                return ep
        return None

    def rebuild_bonds_from_components(self) -> None:
        """Derive Species.bonds from the component bond ids.

        A bond id that does not have exactly two endpoints is not a bond:
        every endpoint carrying it is demoted to UNBOUND rather than left
        dangling, because a dangling half-bond reads as "bound" to the
        matcher while connecting nothing.
        """
        bonds: List[Bond] = []
        for bid, eps in sorted(self.bond_endpoints().items()):
            if len(eps) == 2:
                (m1, c1), (m2, c2) = eps
                bonds.append(Bond(
                    id=bid,
                    mol1_idx=m1,
                    comp1_name=self.molecules[m1].components[c1].name,
                    mol2_idx=m2,
                    comp2_name=self.molecules[m2].components[c2].name,
                ))
            else:
                for (mi, ci) in eps:
                    comp = self.molecules[mi].components[ci]
                    comp.bond_state = BondState.UNBOUND
                    comp.bond_id = None
        self.bonds = bonds
        self.invalidate_canonical()

    def connected_components(self) -> List[List[int]]:
        """Molecule indices grouped by bond connectivity, each group sorted."""
        n = len(self.molecules)
        adj: List[Set[int]] = [set() for _ in range(n)]
        for eps in self.bond_endpoints().values():
            if len(eps) != 2:
                continue
            (m1, _), (m2, _) = eps
            if m1 != m2:
                adj[m1].add(m2)
                adj[m2].add(m1)
        seen: Set[int] = set()
        groups: List[List[int]] = []
        for start in range(n):
            if start in seen:
                continue
            stack = [start]
            group: List[int] = []
            seen.add(start)
            while stack:
                cur = stack.pop()
                group.append(cur)
                for nxt in adj[cur]:
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            groups.append(sorted(group))
        return groups

    # ── Canonical form ───────────────────────────────────────────────────
    #
    # canonical_string() is the species IDENTITY: __hash__ and __eq__ are
    # built on it, and the rule expander keys its species table on it. It
    # must therefore be invariant under (i) the order molecules happen to
    # sit in the list, (ii) the order components sit in a molecule, and
    # (iii) the integers used as bond ids -- otherwise L(r!1).R(l!1) and
    # L(r!7).R(l!7) are two species and a 1:1 rule still "polymerises",
    # just in a new form.

    #: Budget on the number of molecule orderings the exhaustive tie-break
    #: may try, per connected fragment. Weisfeiler-Leman refinement reduces
    #: most complexes to a single ordering; a highly symmetric complex can
    #: exceed this, in which case the result is still invariant but may not
    #: be complete, and canonical_is_exact goes False.
    CANON_ORDERING_BUDGET = 5040

    def _molecule_shape(self, mol_idx: int) -> str:
        """Bond-id-independent, component-order-independent molecule label."""
        mol = self.molecules[mol_idx]
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

    def _refined_labels(self) -> List[int]:
        """Weisfeiler-Leman refinement of the molecule labels.

        Labels are interned to small integers after every round. Carrying
        the previous round's label forward as a string instead doubles the
        label length each round, which is exponential in complex size.
        """
        n = len(self.molecules)
        shapes = [self._molecule_shape(i) for i in range(n)]
        intern = {s: k for k, s in enumerate(sorted(set(shapes)))}
        labels = [intern[s] for s in shapes]

        nbr: List[List[Tuple[str, str, int]]] = [[] for _ in range(n)]
        for eps in self.bond_endpoints().values():
            if len(eps) != 2:
                continue
            (m1, c1), (m2, c2) = eps
            n1 = self.molecules[m1].components[c1].name
            n2 = self.molecules[m2].components[c2].name
            nbr[m1].append((n1, n2, m2))
            nbr[m2].append((n2, n1, m1))

        for _ in range(n):
            sigs = [
                (labels[i], tuple(sorted((a, b, labels[j]) for a, b, j in nbr[i])))
                for i in range(n)
            ]
            intern2 = {sig: k for k, sig in enumerate(sorted(set(sigs)))}
            new = [intern2[s] for s in sigs]
            if new == labels:
                break
            labels = new
        return labels

    def _render_with_order(
        self,
        order: List[int],
        start_bond: int = 1,
        comp_orders: Optional[Dict[int, List[int]]] = None,
    ) -> str:
        """Render the given molecules in the given order.

        Bond ids are renumbered from ``start_bond`` in first-encounter
        order, and the components of each molecule are emitted in a
        deterministic order that does not depend on how they were written.
        """
        pos = {mi: k for k, mi in enumerate(order)}
        eps = self.bond_endpoints()

        def comp_key(mi: int, ci: int) -> Tuple[str, str, str, int, str]:
            c = self.molecules[mi].components[ci]
            partner_pos, partner_name = -1, ""
            if c.bond_state == BondState.BOUND and c.bond_id is not None:
                for (pm, pc) in eps.get(c.bond_id, []):
                    if (pm, pc) != (mi, ci):
                        partner_pos = pos.get(pm, -1)
                        partner_name = self.molecules[pm].components[pc].name
                        break
            return (
                c.name,
                c.state.name if c.state is not None else "",
                c.bond_state.name,
                partner_pos,
                partner_name,
            )

        self._last_comp_key = comp_key
        remap: Dict[int, int] = {}
        next_id = start_bond
        rendered: List[str] = []
        for mi in order:
            mol = self.molecules[mi]
            if comp_orders is not None and mi in comp_orders:
                idxs = comp_orders[mi]
            else:
                idxs = sorted(
                    range(len(mol.components)), key=lambda ci: comp_key(mi, ci)
                )
            parts = []
            for ci in idxs:
                c = mol.components[ci]
                s = c.name
                if c.state is not None:
                    s += "~" + c.state.name
                if c.bond_state == BondState.BOUND and c.bond_id is not None:
                    if c.bond_id not in remap:
                        remap[c.bond_id] = next_id
                        next_id += 1
                    s += f"!{remap[c.bond_id]}"
                elif c.bond_state == BondState.WILDCARD:
                    s += "!"
                elif c.bond_state == BondState.BOUND_WILDCARD:
                    s += "!+"
                parts.append(s)
            rendered.append(f"{mol.name}({','.join(parts)})" if parts else mol.name)
        return ".".join(rendered)

    def _canonical_fragment(self, idxs: List[int]) -> Tuple[str, List[int]]:
        """Lexicographically smallest rendering of one connected fragment."""
        self._best_comp_orders = None
        labels = self._refined_labels()
        classes: Dict[int, List[int]] = {}
        for mi in idxs:
            classes.setdefault(labels[mi], []).append(mi)
        keys = sorted(classes)

        budget = 1
        for k in keys:
            budget *= math.factorial(len(classes[k]))
            if budget > self.CANON_ORDERING_BUDGET:
                break

        if budget > self.CANON_ORDERING_BUDGET:
            self._canon_exact = False
            order = [mi for k in keys for mi in sorted(classes[k])]
            log.warning(
                "canonicalization of a %d-molecule complex exceeded the "
                "ordering budget (%d); the key is still invariant but may "
                "not distinguish two isomorphic complexes",
                len(idxs), self.CANON_ORDERING_BUDGET,
            )
            return self._render_with_order(order), order

        best: Optional[str] = None
        best_order: Optional[List[int]] = None
        best_comps: Optional[Dict[int, List[int]]] = None
        per_class = [list(itertools.permutations(sorted(classes[k]))) for k in keys]
        for combo in itertools.product(*per_class):
            order = [mi for block in combo for mi in block]
            # Rendering once primes comp_key for this molecule order; the
            # components of a molecule that tie under it have to be
            # permuted too, or the canonical string keeps whatever order
            # the species happened to be WRITTEN in.
            self._render_with_order(order)
            ck = self._last_comp_key
            groups: List[Tuple[int, List[List[int]]]] = []
            extra = 1
            for mi in order:
                idxs = sorted(
                    range(len(self.molecules[mi].components)),
                    key=lambda ci: ck(mi, ci),
                )
                runs: List[List[int]] = []
                for ci in idxs:
                    if runs and ck(mi, runs[-1][0]) == ck(mi, ci):
                        runs[-1].append(ci)
                    else:
                        runs.append([ci])
                # Only a run of BOUND components needs permuting: two tied
                # UNBOUND components render identically, so their order
                # cannot change the string.
                perms_per_run = [
                    list(itertools.permutations(r))
                    if len(r) > 1 and self.molecules[mi].components[
                        r[0]].bond_state == BondState.BOUND
                    else [tuple(r)]
                    for r in runs
                ]
                opts = [
                    [c for run in perm_runs for c in run]
                    for perm_runs in itertools.product(*perms_per_run)
                ]
                extra *= len(opts)
                groups.append((mi, opts))
            if extra > self.CANON_ORDERING_BUDGET:
                self._canon_exact = False
                opts_sel = {mi: o[0] for mi, o in groups}
                s = self._render_with_order(order, comp_orders=opts_sel)
                if best is None or s < best:
                    best, best_order, best_comps = s, order, opts_sel
                continue
            for pick in itertools.product(*[o for _, o in groups]):
                comp_orders = {mi: list(pick[k]) for k, (mi, _) in enumerate(groups)}
                s = self._render_with_order(order, comp_orders=comp_orders)
                if best is None or s < best:
                    best, best_order, best_comps = s, order, comp_orders
        self._best_comp_orders = best_comps
        return best or "", best_order or []

    def canonical_string(self) -> str:
        """Canonical, relabeling-invariant string for this species.

        This is the identity key. Connected fragments are canonicalized
        independently, sorted, then re-rendered with a running bond offset
        so the result stays a parseable BNGL species string.
        """
        if self._canon is not None:
            return self._canon
        self._canon_exact = True
        frags = []
        for g in self.connected_components():
            key, order = self._canonical_fragment(g)
            frags.append((key, order, self._best_comp_orders))
        frags.sort(key=lambda fo: fo[0])
        out: List[str] = []
        next_bond = 1
        for _, order, comps in frags:
            out.append(self._render_with_order(
                order, start_bond=next_bond, comp_orders=comps))
            n_bonds = sum(
                1 for bid, eps in self.bond_endpoints().items()
                if len(eps) == 2 and eps[0][0] in order
            )
            next_bond += n_bonds
        self._canon = ".".join(out)
        return self._canon

    @property
    def canonical_is_exact(self) -> bool:
        """False when canonicalization fell back past the ordering budget."""
        self.canonical_string()
        return self._canon_exact

    # ── Pattern matching ─────────────────────────────────────────────────

    def embeddings(
        self,
        pattern: "Pattern",
        limit: int = 0,
    ) -> List["Embedding"]:
        """Every injective embedding of ``pattern`` into this species.

        An embedding maps each pattern molecule to a distinct molecule of
        this species and each pattern component to a distinct component of
        that molecule, honouring states, bond states, AND the pattern's own
        internal bonds -- without which a dissociation pattern
        ``L(r!1).R(l!1)`` would match any L sitting beside any R.

        A rule needs every embedding, not the first: a pattern can embed
        into one species two ways and give two different products.
        """
        p_mols = pattern.molecules
        n_p = len(p_mols)
        if n_p == 0 or n_p > len(self.molecules):
            return []

        # Per (pattern molecule, species molecule): all component maps.
        cand: Dict[Tuple[int, int], List[Dict[int, int]]] = {}
        for pk, p_mol in enumerate(p_mols):
            for si, s_mol in enumerate(self.molecules):
                if s_mol.name != p_mol.name:
                    continue
                maps = _component_maps(s_mol, p_mol)
                if maps:
                    cand[(pk, si)] = maps

        # The pattern's own bonds, as (pm, pc) endpoint pairs.
        p_bond_eps: Dict[int, List[Tuple[int, int]]] = {}
        for pk, p_mol in enumerate(p_mols):
            for pc, comp in enumerate(p_mol.components):
                if comp.bond_state == BondState.BOUND and comp.bond_id is not None:
                    p_bond_eps.setdefault(comp.bond_id, []).append((pk, pc))
        p_bonds = [tuple(v) for v in p_bond_eps.values() if len(v) == 2]

        results: List[Embedding] = []
        mol_map: List[int] = []
        comp_map: Dict[Tuple[int, int], int] = {}

        def recurse(pk: int) -> None:
            if limit and len(results) >= limit:
                return
            if pk == n_p:
                for (pm1, pc1), (pm2, pc2) in p_bonds:
                    a = (mol_map[pm1], comp_map[(pm1, pc1)])
                    b = (mol_map[pm2], comp_map[(pm2, pc2)])
                    if self.bond_partner(*a) != b:
                        return
                results.append(Embedding(tuple(mol_map), dict(comp_map)))
                return
            for si in range(len(self.molecules)):
                if si in mol_map:
                    continue
                for cmap in cand.get((pk, si), ()):
                    mol_map.append(si)
                    for pc, sc in cmap.items():
                        comp_map[(pk, pc)] = sc
                    recurse(pk + 1)
                    for pc in cmap:
                        del comp_map[(pk, pc)]
                    mol_map.pop()
                    if limit and len(results) >= limit:
                        return

        recurse(0)
        return results

    def matches(self, pattern: "Pattern") -> bool:
        """Check if this species matches a pattern."""
        return bool(self.embeddings(pattern, limit=1))

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
        if state is None:
            # Falling through with state=None means "any state" to the
            # matcher, so "R(Y~Q)" would silently become "R(Y)" and a
            # phosphorylation rule would become an identity reaction.
            declared = "~".join(st.name for st in comp_def.states) or "<none>"
            raise ValueError(
                f"component {comp_def.name!r} has no state {state_str!r} "
                f"(declared states: {declared})"
            )
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

    species = Species(molecules=molecules)
    species.rebuild_bonds_from_components()
    return species


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
