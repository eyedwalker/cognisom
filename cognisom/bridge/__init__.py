"""
Bridge Module
=============

Connects NVIDIA NIM outputs to the cognisom simulation engine.
Translates between NIM data formats and simulation module formats.
"""

from .drug_bridge import DrugBridge
from .pipeline import DiscoveryPipeline
from .structure_bridge import StructureBridge

__all__ = ['DrugBridge', 'DiscoveryPipeline', 'StructureBridge']
