"""
Agent Orchestrator (Phase 7)
=============================

Coordinates all AI maintenance agents and provides a unified interface
for running scheduled maintenance tasks.

The orchestrator:
- Runs agents on schedule or on-demand
- Aggregates reports from all agents
- Triggers alerts when issues are detected
- Logs maintenance activity

Usage::

    from cognisom.agents import AgentOrchestrator

    orchestrator = AgentOrchestrator(store=entity_store)

    # Run all maintenance
    summary = orchestrator.run_all()
    print(summary.overall_status)

    # Run specific agents
    summary = orchestrator.run_agents(["literature", "validation"])

    # Schedule regular maintenance
    orchestrator.schedule(interval_hours=24)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable

log = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result from a single agent run."""
    agent_name: str = ""
    success: bool = True
    elapsed_sec: float = 0.0
    summary: str = ""
    issues_count: int = 0
    actions_taken: int = 0
    error: Optional[str] = None


@dataclass
class OrchestrationSummary:
    """Summary of orchestrated agent runs."""
    timestamp: float = 0.0
    agents_run: int = 0
    agents_succeeded: int = 0
    agents_failed: int = 0
    total_issues: int = 0
    total_actions: int = 0
    results: List[AgentResult] = field(default_factory=list)
    elapsed_sec: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    @property
    def overall_status(self) -> str:
        if self.agents_failed > 0:
            return "DEGRADED"
        if self.total_issues > 10:
            return "NEEDS_ATTENTION"
        return "HEALTHY"

    def summary(self) -> str:
        lines = [
            f"AI Maintenance Orchestrator [{self.overall_status}]",
            f"  Agents: {self.agents_succeeded}/{self.agents_run} succeeded",
            f"  Issues found: {self.total_issues}",
            f"  Actions taken: {self.total_actions}",
            f"  Total time: {self.elapsed_sec:.1f}s",
            "",
        ]
        for r in self.results:
            status = "✓" if r.success else "✗"
            lines.append(f"  {status} {r.agent_name}: {r.summary}")
        return "\n".join(lines)


class AgentOrchestrator:
    """Coordinates all AI maintenance agents.

    Provides unified interface for running scheduled and on-demand maintenance.
    """

    # Available agents
    AGENT_REGISTRY = {
        "data_sync": "DataSyncAgent",
        "literature": "LiteratureMonitorAgent",
        "validation": "SimulationValidationAgent",
        "knowledge_graph": "KnowledgeGraphAgent",
        "ontology": "OntologyAgent",
    }

    def __init__(
        self,
        store=None,
        data_dir: str = "data",
        auto_fix: bool = False
    ) -> None:
        """Initialize the orchestrator.

        Args:
            store: EntityStore for agents that need it
            data_dir: Base data directory
            auto_fix: Enable automatic issue fixing
        """
        self._store = store
        self._data_dir = Path(data_dir)
        self._auto_fix = auto_fix
        self._agents: Dict[str, object] = {}
        self._last_run: Dict[str, float] = {}
        self._history: List[OrchestrationSummary] = []

        # Initialize agents lazily on first use

    def run_all(self) -> OrchestrationSummary:
        """Run all maintenance agents."""
        return self.run_agents(list(self.AGENT_REGISTRY.keys()))

    def run_agents(self, agent_names: List[str]) -> OrchestrationSummary:
        """Run specific agents by name."""
        t0 = time.time()
        summary = OrchestrationSummary()

        for name in agent_names:
            if name not in self.AGENT_REGISTRY:
                log.warning("Unknown agent: %s", name)
                continue

            result = self._run_agent(name)
            summary.results.append(result)
            summary.agents_run += 1

            if result.success:
                summary.agents_succeeded += 1
            else:
                summary.agents_failed += 1

            summary.total_issues += result.issues_count
            summary.total_actions += result.actions_taken

        summary.elapsed_sec = time.time() - t0
        self._history.append(summary)
        self._log_summary(summary)

        return summary

    def run_quick_check(self) -> OrchestrationSummary:
        """Run lightweight checks (validation only)."""
        return self.run_agents(["validation"])

    def run_data_update(self) -> OrchestrationSummary:
        """Run data-focused agents (sync and literature)."""
        return self.run_agents(["data_sync", "literature"])

    def run_graph_maintenance(self) -> OrchestrationSummary:
        """Run knowledge graph maintenance."""
        return self.run_agents(["knowledge_graph", "ontology"])

    # ── Agent Execution ──────────────────────────────────────────────

    def _run_agent(self, name: str) -> AgentResult:
        """Run a single agent and capture results."""
        result = AgentResult(agent_name=name)
        t0 = time.time()

        try:
            agent = self._get_agent(name)
            if agent is None:
                result.success = False
                result.error = f"Failed to initialize agent: {name}"
                return result

            # Run agent-specific logic
            if name == "data_sync":
                reports = agent.sync_all()
                result.issues_count = sum(len(r.errors) for r in reports)
                result.actions_taken = sum(r.datasets_downloaded for r in reports)
                result.summary = f"Synced {sum(r.datasets_new for r in reports)} new datasets"

            elif name == "literature":
                report = agent.scan_recent(days=7)
                result.issues_count = len(report.errors)
                result.actions_taken = report.relevant_papers
                result.summary = f"Found {report.relevant_papers} relevant papers"

            elif name == "validation":
                report = agent.validate_all()
                result.issues_count = report.benchmarks_failed
                result.actions_taken = report.benchmarks_passed
                result.summary = f"{report.benchmarks_passed}/{report.benchmarks_run} benchmarks passed"

            elif name == "knowledge_graph":
                report = agent.maintain(auto_fix=self._auto_fix)
                result.issues_count = len([i for i in report.issues if i.severity == "error"])
                result.actions_taken = report.duplicates_merged + report.orphans_linked
                result.summary = f"Graph health: {report.health_score:.2f}"

            elif name == "ontology":
                report = agent.full_audit()
                result.issues_count = len(report.errors)
                result.actions_taken = 0  # Ontology agent is read-only
                result.summary = report.summary().split("\n")[0]

            self._last_run[name] = time.time()

        except Exception as e:
            result.success = False
            result.error = str(e)
            result.summary = f"Error: {str(e)[:50]}"
            log.error("Agent %s failed: %s", name, e)

        result.elapsed_sec = time.time() - t0
        return result

    def _get_agent(self, name: str):
        """Get or create an agent instance."""
        if name in self._agents:
            return self._agents[name]

        try:
            if name == "data_sync":
                from .data_sync_agent import DataSyncAgent
                self._agents[name] = DataSyncAgent(
                    data_dir=str(self._data_dir / "scrna")
                )

            elif name == "literature":
                from .literature_monitor_agent import LiteratureMonitorAgent
                self._agents[name] = LiteratureMonitorAgent(
                    cache_dir=str(self._data_dir / "literature_cache")
                )

            elif name == "validation":
                from .simulation_validation_agent import SimulationValidationAgent
                self._agents[name] = SimulationValidationAgent(
                    history_dir=str(self._data_dir / "validation_history")
                )

            elif name == "knowledge_graph":
                from .knowledge_graph_agent import KnowledgeGraphAgent
                self._agents[name] = KnowledgeGraphAgent(store=self._store)

            elif name == "ontology":
                from cognisom.agent.ontology_sync import OntologyAgent
                self._agents[name] = OntologyAgent(self._store)

            return self._agents.get(name)

        except Exception as e:
            log.error("Failed to create agent %s: %s", name, e)
            return None

    # ── Scheduling ───────────────────────────────────────────────────

    def schedule(self, interval_hours: float = 24.0, callback: Optional[Callable] = None):
        """Schedule periodic maintenance runs.

        Note: This is a simple blocking scheduler. For production,
        use a proper scheduler like APScheduler or Celery.

        Args:
            interval_hours: Hours between runs
            callback: Optional callback after each run
        """
        import threading

        def run_loop():
            while True:
                summary = self.run_all()
                if callback:
                    callback(summary)
                time.sleep(interval_hours * 3600)

        thread = threading.Thread(target=run_loop, daemon=True)
        thread.start()
        log.info("Scheduled maintenance every %.1f hours", interval_hours)

    # ── Status and History ───────────────────────────────────────────

    def get_status(self) -> Dict:
        """Get current agent status."""
        status = {
            "agents": {},
            "last_orchestration": None,
            "overall_health": "UNKNOWN",
        }

        for name in self.AGENT_REGISTRY:
            last = self._last_run.get(name)
            status["agents"][name] = {
                "last_run": datetime.fromtimestamp(last).isoformat() if last else None,
                "hours_since_run": (time.time() - last) / 3600 if last else None,
            }

        if self._history:
            last_summary = self._history[-1]
            status["last_orchestration"] = {
                "timestamp": datetime.fromtimestamp(last_summary.timestamp).isoformat(),
                "status": last_summary.overall_status,
                "issues": last_summary.total_issues,
            }
            status["overall_health"] = last_summary.overall_status

        return status

    def get_history(self, limit: int = 10) -> List[OrchestrationSummary]:
        """Get recent orchestration history."""
        return self._history[-limit:]

    # ── Logging ──────────────────────────────────────────────────────

    def _log_summary(self, summary: OrchestrationSummary) -> None:
        """Log orchestration summary to file."""
        import json

        log_dir = self._data_dir / "agent_logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"orchestration_{int(summary.timestamp)}.json"
        data = {
            "timestamp": summary.timestamp,
            "status": summary.overall_status,
            "agents_run": summary.agents_run,
            "agents_succeeded": summary.agents_succeeded,
            "total_issues": summary.total_issues,
            "total_actions": summary.total_actions,
            "elapsed_sec": summary.elapsed_sec,
            "results": [
                {
                    "agent": r.agent_name,
                    "success": r.success,
                    "summary": r.summary,
                    "issues": r.issues_count,
                    "actions": r.actions_taken,
                    "error": r.error,
                }
                for r in summary.results
            ],
        }
        log_file.write_text(json.dumps(data, indent=2))
        log.info("Orchestration logged to %s", log_file)
