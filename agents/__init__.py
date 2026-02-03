"""
AI Maintenance Agents (Phase 7)
================================

Autonomous agents that keep the Cognisom biological knowledge base current.

Agents:
    1. DataSyncAgent       — Sync with GEO/TCGA/CellxGene for new datasets
    2. LiteratureMonitorAgent — Monitor PubMed/bioRxiv for relevant publications
    3. SimulationValidationAgent — Validate simulation accuracy vs benchmarks
    4. KnowledgeGraphAgent — Maintain entity relationships and detect inconsistencies

Usage::

    from cognisom.agents import AgentOrchestrator

    orchestrator = AgentOrchestrator(store=entity_store)
    orchestrator.run_all()  # Run all maintenance tasks

    # Or run individual agents
    from cognisom.agents import DataSyncAgent
    agent = DataSyncAgent()
    report = agent.sync_geo_datasets()
"""

from .data_sync_agent import DataSyncAgent, DataSyncReport
from .literature_monitor_agent import LiteratureMonitorAgent, LiteratureReport
from .simulation_validation_agent import SimulationValidationAgent, ValidationReport
from .knowledge_graph_agent import KnowledgeGraphAgent, GraphMaintenanceReport
from .orchestrator import AgentOrchestrator

__all__ = [
    "DataSyncAgent",
    "DataSyncReport",
    "LiteratureMonitorAgent",
    "LiteratureReport",
    "SimulationValidationAgent",
    "ValidationReport",
    "KnowledgeGraphAgent",
    "GraphMaintenanceReport",
    "AgentOrchestrator",
]
