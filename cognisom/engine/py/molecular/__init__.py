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

# proteins.py and mutations.py held four classes with `pass` bodies --
# Protein, MutantProtein, Mutation, OncogenicMutation -- re-exported from
# here as though they were implementations. The Mutation stub was worse
# than useless: it SHADOWED the real @dataclass Mutation defined in
# nucleic_acids.py, so `from ...molecular import Mutation` returned the
# empty placeholder. Both files are deleted and the real dataclass is
# exported instead. Nothing outside this package imported any of the
# four, so this removes a trap rather than a capability.
from .nucleic_acids import DNA, RNA, Gene, Mutation
from .exosomes import Exosome, ExosomeSystem

__all__ = [
    'DNA',
    'RNA',
    'Gene',
    'Mutation',
    'Exosome',
    'ExosomeSystem',
]
