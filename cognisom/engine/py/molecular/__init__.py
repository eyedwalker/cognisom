"""
Molecular Module
================

Molecular-level simulation with actual sequences and structures.

Features:
- DNA/RNA sequences (actual bases)
- Protein sequences (actual amino acids)
- Mutations and modifications
- Chemical properties
- Binding interactions
- Exosome-mediated transfer
"""

from .nucleic_acids import DNA, RNA, Gene, Mutation
from .exosomes import Exosome, ExosomeSystem

# NOTE: `Mutation` is re-exported from nucleic_acids, where it is a real
# dataclass. It was previously imported from a placeholder `mutations.py`
# containing `class Mutation: pass`, which silently shadowed the real type for
# anyone doing `from ...molecular import Mutation`. That placeholder — and the
# matching `proteins.py` placeholder exporting empty `Protein`/`MutantProtein`
# classes — have been removed. Protein handling lives in `protein_domains.py`,
# `peptidome.py`, and `esm_stability.py`; import from those directly.

__all__ = [
    'DNA',
    'RNA',
    'Gene',
    'Mutation',
    'Exosome',
    'ExosomeSystem',
]
