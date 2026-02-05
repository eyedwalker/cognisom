"""
BNGL Parser
===========

Parses BioNetGen Language (BNGL) model files.

BNGL models have this structure::

    begin model
        begin parameters
            kon  1e6
            koff 0.1
        end parameters

        begin molecule types
            L(r)
            R(l,Y~U~P)
        end molecule types

        begin seed species
            L(r) 1000
            R(l,Y~U) 1000
        end seed species

        begin observables
            Molecules L_free L(r)
            Molecules L_bound L(r!+)
        end observables

        begin reaction rules
            L(r) + R(l) <-> L(r!1).R(l!1) kon, koff
            R(Y~U) -> R(Y~P) 0.1
        end reaction rules
    end model
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .molecules import (
    Component,
    ComponentState,
    MoleculeInstance,
    MoleculeType,
    Pattern,
    Species,
    parse_pattern,
    parse_species,
)
from .observables import Observable, ObservableSet
from .rules import (
    RateExpression,
    ReactionRule,
    RuleExpander,
    parse_rule,
)

log = logging.getLogger(__name__)


@dataclass
class SeedSpecies:
    """Initial species with count."""
    species: Species
    count: float

    def __str__(self) -> str:
        return f"{self.species} {self.count}"


@dataclass
class BNGLModel:
    """
    A complete BNGL model.

    Contains all components needed to run a simulation:
    - Parameters
    - Molecule types
    - Seed species
    - Observables
    - Reaction rules

    Attributes
    ----------
    name : str
        Model name
    parameters : Dict[str, float]
        Parameter values
    molecule_types : Dict[str, MoleculeType]
        Molecule type definitions
    seed_species : List[SeedSpecies]
        Initial species and counts
    observables : ObservableSet
        Observable definitions
    rules : List[ReactionRule]
        Reaction rules
    """
    name: str = "model"
    parameters: Dict[str, float] = field(default_factory=dict)
    molecule_types: Dict[str, MoleculeType] = field(default_factory=dict)
    seed_species: List[SeedSpecies] = field(default_factory=list)
    observables: ObservableSet = field(default_factory=ObservableSet)
    rules: List[ReactionRule] = field(default_factory=list)

    @property
    def n_parameters(self) -> int:
        return len(self.parameters)

    @property
    def n_molecule_types(self) -> int:
        return len(self.molecule_types)

    @property
    def n_seed_species(self) -> int:
        return len(self.seed_species)

    @property
    def n_observables(self) -> int:
        return len(self.observables.observables)

    @property
    def n_rules(self) -> int:
        return len(self.rules)

    def get_seed_species_list(self) -> List[Species]:
        """Get list of seed species (without counts)."""
        return [ss.species for ss in self.seed_species]

    def get_initial_counts(self) -> Dict[str, float]:
        """Get initial species counts."""
        return {str(ss.species): ss.count for ss in self.seed_species}

    def summary(self) -> str:
        """Get model summary string."""
        return (
            f"BNGL Model: {self.name}\n"
            f"  Parameters: {self.n_parameters}\n"
            f"  Molecule types: {self.n_molecule_types}\n"
            f"  Seed species: {self.n_seed_species}\n"
            f"  Observables: {self.n_observables}\n"
            f"  Rules: {self.n_rules}"
        )

    @staticmethod
    def egfr_signaling() -> "BNGLModel":
        """
        Create EGFR signaling model.

        EGF receptor dimerization and phosphorylation.
        Classic model for testing rule-based approaches.
        """
        model = BNGLModel(name="egfr_signaling")

        # Parameters
        model.parameters = {
            "EGF_tot": 1000,
            "EGFR_tot": 1000,
            "kp1": 1e-4,      # EGF-EGFR binding
            "km1": 0.1,       # EGF-EGFR unbinding
            "kp2": 1e-3,      # EGFR dimerization
            "km2": 0.1,       # EGFR dimer dissociation
            "kphos": 1.0,     # Phosphorylation
            "kdephos": 0.1,   # Dephosphorylation
        }

        # Molecule types
        model.molecule_types = {
            "EGF": MoleculeType.from_string("EGF(R)"),
            "EGFR": MoleculeType.from_string("EGFR(L,CR1,Y~U~P)"),
        }

        # Seed species
        egf_species = parse_species("EGF(R)", model.molecule_types)
        egfr_species = parse_species("EGFR(L,CR1,Y~U)", model.molecule_types)

        model.seed_species = [
            SeedSpecies(egf_species, model.parameters["EGF_tot"]),
            SeedSpecies(egfr_species, model.parameters["EGFR_tot"]),
        ]

        # Observables
        model.observables.add(Observable.from_string(
            "Molecules EGF_free EGF(R)", model.molecule_types
        ))
        model.observables.add(Observable.from_string(
            "Molecules EGFR_bound EGFR(L!+)", model.molecule_types
        ))
        model.observables.add(Observable.from_string(
            "Molecules EGFR_phos EGFR(Y~P)", model.molecule_types
        ))

        # Rules
        model.rules.append(parse_rule(
            "EGF(R) + EGFR(L) <-> EGF(R!1).EGFR(L!1) kp1, km1",
            model.molecule_types, model.parameters, "EGF_binding"
        ))
        model.rules.append(parse_rule(
            "EGFR(CR1) + EGFR(CR1) <-> EGFR(CR1!1).EGFR(CR1!1) kp2, km2",
            model.molecule_types, model.parameters, "dimerization"
        ))
        model.rules.append(parse_rule(
            "EGFR(Y~U) -> EGFR(Y~P) kphos",
            model.molecule_types, model.parameters, "phosphorylation"
        ))
        model.rules.append(parse_rule(
            "EGFR(Y~P) -> EGFR(Y~U) kdephos",
            model.molecule_types, model.parameters, "dephosphorylation"
        ))

        return model

    @staticmethod
    def simple_receptor() -> "BNGLModel":
        """
        Create simple receptor binding model.

        L + R <-> L.R

        Minimal model for testing.
        """
        model = BNGLModel(name="simple_receptor")

        model.parameters = {
            "L_tot": 1000,
            "R_tot": 1000,
            "kon": 1e-3,
            "koff": 0.1,
        }

        model.molecule_types = {
            "L": MoleculeType.from_string("L(r)"),
            "R": MoleculeType.from_string("R(l)"),
        }

        model.seed_species = [
            SeedSpecies(parse_species("L(r)", model.molecule_types), 1000),
            SeedSpecies(parse_species("R(l)", model.molecule_types), 1000),
        ]

        model.observables.add(Observable.from_string(
            "Molecules L_free L(r)", model.molecule_types
        ))
        model.observables.add(Observable.from_string(
            "Species LR_complex L(r!1).R(l!1)", model.molecule_types
        ))

        model.rules.append(parse_rule(
            "L(r) + R(l) <-> L(r!1).R(l!1) kon, koff",
            model.molecule_types, model.parameters, "binding"
        ))

        return model


class BNGLParser:
    """
    Parser for BNGL model files.

    Parses BNGL syntax into BNGLModel objects.

    Example
    -------
    >>> parser = BNGLParser()
    >>> model = parser.parse_file("model.bngl")
    >>> model = parser.parse(bngl_string)
    """

    def __init__(self):
        self._current_block = None
        self._model = None

    def parse_file(self, filepath: Union[str, Path]) -> BNGLModel:
        """
        Parse BNGL model from file.

        Parameters
        ----------
        filepath : str or Path
            Path to .bngl file

        Returns
        -------
        BNGLModel
            Parsed model
        """
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            content = f.read()
        return self.parse(content, name=filepath.stem)

    def parse(self, source: str, name: str = "model") -> BNGLModel:
        """
        Parse BNGL model from string.

        Parameters
        ----------
        source : str
            BNGL model string
        name : str
            Model name

        Returns
        -------
        BNGLModel
            Parsed model
        """
        self._model = BNGLModel(name=name)
        self._current_block = None

        # Remove comments
        lines = []
        for line in source.split('\n'):
            # Remove inline comments
            if '#' in line:
                line = line[:line.index('#')]
            line = line.strip()
            if line:
                lines.append(line)

        # Parse line by line
        for line in lines:
            self._parse_line(line)

        log.info(self._model.summary())
        return self._model

    def _parse_line(self, line: str):
        """Parse a single line."""
        # Check for block start/end
        if line.lower().startswith('begin '):
            block_name = line[6:].strip().lower()
            self._current_block = block_name
            return

        if line.lower().startswith('end '):
            self._current_block = None
            return

        # Parse based on current block
        if self._current_block == 'parameters':
            self._parse_parameter(line)
        elif self._current_block == 'molecule types':
            self._parse_molecule_type(line)
        elif self._current_block == 'seed species':
            self._parse_seed_species(line)
        elif self._current_block == 'species':
            self._parse_seed_species(line)
        elif self._current_block == 'observables':
            self._parse_observable(line)
        elif self._current_block == 'reaction rules':
            self._parse_rule(line)

    def _parse_parameter(self, line: str):
        """Parse parameter definition."""
        # Format: name value
        parts = line.split()
        if len(parts) >= 2:
            name = parts[0]
            try:
                # Try to evaluate as expression
                value = eval(parts[1], {"__builtins__": {}}, self._model.parameters)
                self._model.parameters[name] = float(value)
            except Exception:
                # Store as string for later
                pass

    def _parse_molecule_type(self, line: str):
        """Parse molecule type definition."""
        mol_type = MoleculeType.from_string(line)
        self._model.molecule_types[mol_type.name] = mol_type

    def _parse_seed_species(self, line: str):
        """Parse seed species definition."""
        # Format: species count
        # Need to find where species ends and count begins
        # Species can contain (), so find last number

        parts = line.rsplit(None, 1)
        if len(parts) == 2:
            species_str = parts[0]
            count_str = parts[1]

            try:
                count = float(eval(count_str, {"__builtins__": {}}, self._model.parameters))
            except Exception:
                count = 0.0

            species = parse_species(species_str, self._model.molecule_types)
            self._model.seed_species.append(SeedSpecies(species, count))

    def _parse_observable(self, line: str):
        """Parse observable definition."""
        obs = Observable.from_string(line, self._model.molecule_types)
        self._model.observables.add(obs)

    def _parse_rule(self, line: str):
        """Parse reaction rule."""
        # Skip lines that look like rule numbering (e.g., "1:")
        if re.match(r'^\d+:', line):
            line = line.split(':', 1)[1].strip()

        rule = parse_rule(
            line,
            self._model.molecule_types,
            self._model.parameters,
            name=f"rule_{len(self._model.rules) + 1}",
        )
        self._model.rules.append(rule)


# ── Convenience Functions ────────────────────────────────────────────────

def load_bngl_model(filepath: Union[str, Path]) -> BNGLModel:
    """
    Load BNGL model from file.

    Parameters
    ----------
    filepath : str or Path
        Path to .bngl file

    Returns
    -------
    BNGLModel
        Parsed model
    """
    parser = BNGLParser()
    return parser.parse_file(filepath)


def expand_model(
    model: BNGLModel,
    max_species: int = 10000,
) -> Tuple[List, List[Species]]:
    """
    Expand model rules to generate reaction network.

    Parameters
    ----------
    model : BNGLModel
        Parsed model
    max_species : int
        Maximum species to generate

    Returns
    -------
    Tuple[List, List[Species]]
        (reactions, species)
    """
    expander = RuleExpander(
        molecule_types=model.molecule_types,
        rules=model.rules,
        parameters=model.parameters,
    )

    return expander.expand(
        seed_species=model.get_seed_species_list(),
        max_species=max_species,
    )
