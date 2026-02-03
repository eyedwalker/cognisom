"""
NVIDIA NIM Client Wrappers
==========================

Unified Python clients for BioNeMo NIM microservices.
All clients use the NVIDIA_API_KEY from environment/.env.

11 NIMs:
- MolMIM / GenMol: Small molecule generation
- RFdiffusion: Protein binder design
- ProteinMPNN: Sequence design for 3D backbones
- DiffDock: Molecular docking
- ESM2: Protein sequence embeddings
- OpenFold3: All-atom biomolecular complex prediction
- Boltz-2: Protein-ligand/DNA complex prediction
- Evo2: Genomic foundation model (DNA generation/analysis)
- AlphaFold2-Multimer: Multi-chain protein complex prediction
- MSA-Search: Multiple sequence alignment
"""

from .client import NIMClient
from .molmim import MolMIMClient
from .genmol import GenMolClient
from .rfdiffusion import RFdiffusionClient
from .proteinmpnn import ProteinMPNNClient
from .diffdock import DiffDockClient
from .esm2 import ESM2Client
from .openfold3 import OpenFold3Client
from .boltz2 import Boltz2Client
from .evo2 import Evo2Client
from .alphafold2_multimer import AlphaFold2MultimerClient
from .msa_search import MSASearchClient

__all__ = [
    'NIMClient',
    # Molecule generation
    'MolMIMClient',
    'GenMolClient',
    # Protein design
    'RFdiffusionClient',
    'ProteinMPNNClient',
    # Docking
    'DiffDockClient',
    # Embeddings
    'ESM2Client',
    # Structure prediction
    'OpenFold3Client',
    'Boltz2Client',
    'AlphaFold2MultimerClient',
    # Genomics
    'Evo2Client',
    # Alignment
    'MSASearchClient',
]
