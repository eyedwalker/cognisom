"""
Genomics Pipeline
=================

Personal genomic data ingestion and analysis for the Molecular Digital Twin.

Modules:
- vcf_parser: Parse VCF files into structured variant records
- variant_annotator: Classify variants and map to genes/proteins
- gene_protein_mapper: Gene symbols → UniProt IDs → protein sequences
- patient_profile: Aggregate patient data into a unified profile
- cell2sentence: Cell2Sentence-Scale 27B integration
- expression_ranker: scRNA-seq → ranked gene sentences
- cell_state_classifier: Predict cell states (exhaustion, polarization)
- twin_config: Merge all data into personalized digital twin config
- treatment_simulator: Predict immunotherapy response
- spatial_transcriptomics: Spatial gene expression mapping
"""

from .vcf_parser import VCFParser, Variant
from .variant_annotator import VariantAnnotator
from .gene_protein_mapper import GeneProteinMapper, ProteinInfo
from .patient_profile import PatientProfile, PatientProfileBuilder
from .expression_ranker import ExpressionRanker
from .cell2sentence import Cell2SentenceModel, CellStatePrediction
from .cell_state_classifier import CellStateClassifier, ImmuneClassification
from .twin_config import DigitalTwinConfig
from .treatment_simulator import TreatmentSimulator, TreatmentResult
from .spatial_transcriptomics import SpatialData, SpatialSpot, SpatialStats

__all__ = [
    # Phase 1: Genomics
    "VCFParser",
    "Variant",
    "VariantAnnotator",
    "GeneProteinMapper",
    "ProteinInfo",
    "PatientProfile",
    "PatientProfileBuilder",
    # Phase 2: Cell States
    "ExpressionRanker",
    "Cell2SentenceModel",
    "CellStatePrediction",
    "CellStateClassifier",
    "ImmuneClassification",
    # Phase 3: Digital Twin
    "DigitalTwinConfig",
    "TreatmentSimulator",
    "TreatmentResult",
    # Phase 5: Spatial Transcriptomics
    "SpatialData",
    "SpatialSpot",
    "SpatialStats",
]
