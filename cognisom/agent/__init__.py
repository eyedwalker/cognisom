"""
Research Agent
==============

Interactive tool-based agent for querying genomic databases,
running NIM-powered analyses, and exploring cancer biology.
"""

from .tools import Tool, ToolResult, ToolRegistry
from .db_tools import (
    NCBIGeneTool,
    UniProtTool,
    PDBSearchTool,
    CBioPortalTool,
    PubMedSearchTool,
)
from .nim_tools import (
    StructurePredictionTool,
    MoleculeGenerationTool,
    ProteinEmbeddingTool,
    MutationImpactTool,
    DockingTool,
)
from .agent import ResearchAgent

__all__ = [
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "NCBIGeneTool",
    "UniProtTool",
    "PDBSearchTool",
    "CBioPortalTool",
    "PubMedSearchTool",
    "StructurePredictionTool",
    "MoleculeGenerationTool",
    "ProteinEmbeddingTool",
    "MutationImpactTool",
    "DockingTool",
    "ResearchAgent",
]
