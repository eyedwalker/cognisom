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

from .nucleic_acids import DNA, RNA, Gene
from .proteins import Protein, MutantProtein
from .exosomes import Exosome, ExosomeSystem
from .mutations import Mutation, OncogenicMutation

__all__ = [
    'DNA',
    'RNA',
    'Gene',
    'Protein',
    'MutantProtein',
    'Exosome',
    'ExosomeSystem',
    'Mutation',
    'OncogenicMutation'
]
