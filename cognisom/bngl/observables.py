"""
BNGL Observables
================

Observables define quantities to track during simulation.
They count species or molecules matching specific patterns.

Example observables::

    begin observables
        Molecules L_free L(r)           # Count free L molecules
        Molecules L_bound L(r!+)        # Count bound L molecules
        Species   LR_complex L(r!1).R(l!1)  # Count L-R complexes
    end observables
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

import numpy as np

from .molecules import MoleculeType, Pattern, Species, parse_pattern


def _split_patterns(pattern_str: str) -> List[str]:
    """
    Split observable pattern string on '+' but not inside parentheses.

    In BNGL, '+' has two meanings:
    - Between patterns: "L(r) + R(l)" means sum of matches
    - Inside component: "L(r!+)" means "r is bound to any"

    This function correctly handles both cases.

    Examples:
        "L(r!+)" -> ["L(r!+)"]
        "L(r) + R(l)" -> ["L(r)", "R(l)"]
        "A(b!+) + B(a!+)" -> ["A(b!+)", "B(a!+)"]
    """
    parts = []
    current = []
    depth = 0

    for char in pattern_str:
        if char == '(':
            depth += 1
            current.append(char)
        elif char == ')':
            depth -= 1
            current.append(char)
        elif char == '+' and depth == 0:
            # This is a pattern separator
            parts.append(''.join(current))
            current = []
        else:
            current.append(char)

    if current:
        parts.append(''.join(current))

    return parts


class ObservableType(Enum):
    """Type of observable counting."""
    MOLECULES = auto()  # Count matching molecules (can count same molecule twice in complex)
    SPECIES = auto()    # Count matching species (each species counted once)


@dataclass
class Observable:
    """
    An observable quantity to track.

    Attributes
    ----------
    name : str
        Observable name (for output)
    obs_type : ObservableType
        How to count: molecules or species
    patterns : List[Pattern]
        Patterns to match (count sum of all matches)
    weight : float
        Weight factor for counting (default 1.0)
    """
    name: str
    obs_type: ObservableType
    patterns: List[Pattern] = field(default_factory=list)
    weight: float = 1.0

    def count(
        self,
        species_counts: Dict[str, int],
        species_list: List[Species],
    ) -> float:
        """
        Compute observable value given species counts.

        Parameters
        ----------
        species_counts : Dict[str, int]
            Mapping from species string to count
        species_list : List[Species]
            List of all species

        Returns
        -------
        float
            Observable value
        """
        total = 0.0

        for sp in species_list:
            sp_key = str(sp)
            if sp_key not in species_counts:
                continue

            count = species_counts[sp_key]

            for pattern in self.patterns:
                if self.obs_type == ObservableType.MOLECULES:
                    # Count each matching molecule
                    n_matches = self._count_molecule_matches(sp, pattern)
                    total += count * n_matches * self.weight
                else:
                    # Count species (once per match)
                    if sp.matches(pattern):
                        total += count * self.weight

        return total

    def _count_molecule_matches(self, sp: Species, pattern: Pattern) -> int:
        """Count how many molecules in species match pattern."""
        if not pattern.is_single_molecule:
            return 1 if sp.matches(pattern) else 0

        count = 0
        pattern_mol = pattern.molecules[0]
        for mol in sp.molecules:
            if mol.matches(pattern_mol):
                count += 1
        return count

    @classmethod
    def from_string(
        cls,
        obs_str: str,
        molecule_types: Dict[str, MoleculeType],
    ) -> "Observable":
        """
        Parse observable from BNGL string.

        Format: Type Name Pattern [+ Pattern ...]

        Example: "Molecules L_total L(r) + L(r!+)"
        """
        parts = obs_str.strip().split(None, 2)
        if len(parts) < 3:
            raise ValueError(f"Invalid observable format: {obs_str}")

        type_str = parts[0].lower()
        name = parts[1]
        pattern_str = parts[2]

        if type_str == "molecules":
            obs_type = ObservableType.MOLECULES
        elif type_str == "species":
            obs_type = ObservableType.SPECIES
        else:
            raise ValueError(f"Unknown observable type: {type_str}")

        # Parse patterns - split on '+' but not inside parentheses
        # (e.g., "L!+" in BNGL means "bound to any", should not split there)
        patterns = []
        for p_str in _split_patterns(pattern_str):
            p_str = p_str.strip()
            if p_str:
                patterns.append(parse_pattern(p_str, molecule_types))

        return cls(name=name, obs_type=obs_type, patterns=patterns)


@dataclass
class ObservableSet:
    """
    Collection of observables for tracking simulation output.

    Attributes
    ----------
    observables : List[Observable]
        List of observables
    """
    observables: List[Observable] = field(default_factory=list)

    def add(self, obs: Observable):
        """Add an observable."""
        self.observables.append(obs)

    def get(self, name: str) -> Optional[Observable]:
        """Get observable by name."""
        for obs in self.observables:
            if obs.name == name:
                return obs
        return None

    @property
    def names(self) -> List[str]:
        """Get list of observable names."""
        return [obs.name for obs in self.observables]

    def compute_all(
        self,
        species_counts: Dict[str, int],
        species_list: List[Species],
    ) -> Dict[str, float]:
        """
        Compute all observable values.

        Parameters
        ----------
        species_counts : Dict[str, int]
            Mapping from species string to count
        species_list : List[Species]
            List of all species

        Returns
        -------
        Dict[str, float]
            Observable name -> value
        """
        return {
            obs.name: obs.count(species_counts, species_list)
            for obs in self.observables
        }

    def compute_trajectory(
        self,
        times: np.ndarray,
        species_trajectories: Dict[str, np.ndarray],
        species_list: List[Species],
    ) -> Dict[str, np.ndarray]:
        """
        Compute observables over time.

        Parameters
        ----------
        times : np.ndarray
            Time points
        species_trajectories : Dict[str, np.ndarray]
            Species counts over time
        species_list : List[Species]
            List of all species

        Returns
        -------
        Dict[str, np.ndarray]
            Observable trajectories
        """
        n_times = len(times)
        results = {obs.name: np.zeros(n_times) for obs in self.observables}

        for t_idx in range(n_times):
            # Build counts dict for this time point
            counts = {
                sp_key: int(traj[t_idx])
                for sp_key, traj in species_trajectories.items()
            }

            # Compute observables
            for obs in self.observables:
                results[obs.name][t_idx] = obs.count(counts, species_list)

        return results
