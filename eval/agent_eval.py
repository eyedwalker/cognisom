"""
Agent Evaluator (Phase 5)
=========================

Evaluate AI agent performance for the data flywheel.
Captures agent interactions and measures quality for model distillation.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class AgentInteraction:
    """A single agent interaction for evaluation."""
    interaction_id: str = ""
    agent_type: str = ""              # "researcher", "designer", "simulator"
    query: str = ""
    response: str = ""
    context: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    timestamp: float = 0.0

    # Quality scores (0-1, set by evaluator)
    relevance_score: float = 0.0
    accuracy_score: float = 0.0
    helpfulness_score: float = 0.0
    factuality_score: float = 0.0

    # Human feedback (if available)
    human_rating: Optional[int] = None  # 1-5
    human_feedback: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


@dataclass
class AgentEvalReport:
    """Agent evaluation report."""
    timestamp: float = 0.0
    agent_type: str = ""
    interactions_evaluated: int = 0
    mean_relevance: float = 0.0
    mean_accuracy: float = 0.0
    mean_helpfulness: float = 0.0
    mean_factuality: float = 0.0
    mean_latency_ms: float = 0.0
    total_tokens: int = 0
    quality_distribution: Dict[str, int] = field(default_factory=dict)
    low_quality_samples: List[str] = field(default_factory=list)
    distillation_candidates: int = 0
    recommendations: List[str] = field(default_factory=list)
    elapsed_sec: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    @property
    def overall_quality(self) -> float:
        """Overall quality score (0-100)."""
        scores = [
            self.mean_relevance,
            self.mean_accuracy,
            self.mean_helpfulness,
            self.mean_factuality,
        ]
        return sum(scores) / len(scores) * 100 if scores else 0.0

    def summary(self) -> str:
        lines = [
            f"Agent Evaluation Report ({self.agent_type})",
            f"=" * 50,
            f"Interactions evaluated: {self.interactions_evaluated}",
            f"Overall quality: {self.overall_quality:.1f}/100",
            "",
            "Quality Metrics:",
            f"  Relevance: {self.mean_relevance:.3f}",
            f"  Accuracy: {self.mean_accuracy:.3f}",
            f"  Helpfulness: {self.mean_helpfulness:.3f}",
            f"  Factuality: {self.mean_factuality:.3f}",
            "",
            "Performance:",
            f"  Mean latency: {self.mean_latency_ms:.0f}ms",
            f"  Total tokens: {self.total_tokens:,}",
            "",
            f"Distillation candidates: {self.distillation_candidates}",
        ]

        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in self.recommendations[:3]:
                lines.append(f"  - {rec}")

        return "\n".join(lines)


class AgentEvaluator:
    """Evaluates AI agent quality for the data flywheel.

    Captures and scores agent interactions to:
    1. Identify high-quality examples for distillation
    2. Track quality trends over time
    3. Generate training data for smaller models
    """

    # Quality thresholds
    HIGH_QUALITY_THRESHOLD = 0.8
    LOW_QUALITY_THRESHOLD = 0.4
    DISTILLATION_THRESHOLD = 0.85

    def __init__(
        self,
        storage_dir: str = "data/agent_interactions",
        model_name: str = "default"
    ) -> None:
        """Initialize the evaluator.

        Args:
            storage_dir: Directory to store interactions
            model_name: Name of the model being evaluated
        """
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._model_name = model_name

        self._interactions: List[AgentInteraction] = []
        self._load_interactions()

    def capture(
        self,
        query: str,
        response: str,
        agent_type: str = "researcher",
        context: Optional[List[str]] = None,
        tools_used: Optional[List[str]] = None,
        latency_ms: float = 0.0,
        tokens_in: int = 0,
        tokens_out: int = 0
    ) -> AgentInteraction:
        """Capture an agent interaction.

        Args:
            query: User query/prompt
            response: Agent response
            agent_type: Type of agent
            context: Retrieved context documents
            tools_used: Tools called during interaction
            latency_ms: Response latency
            tokens_in: Input tokens
            tokens_out: Output tokens

        Returns:
            Captured interaction
        """
        interaction = AgentInteraction(
            interaction_id=f"{agent_type}_{int(time.time() * 1000)}",
            agent_type=agent_type,
            query=query,
            response=response,
            context=context or [],
            tools_used=tools_used or [],
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

        # Auto-score
        self._auto_score(interaction)

        self._interactions.append(interaction)
        self._save_interaction(interaction)

        return interaction

    def evaluate_batch(
        self,
        interactions: Optional[List[AgentInteraction]] = None,
        agent_type: Optional[str] = None
    ) -> AgentEvalReport:
        """Evaluate a batch of interactions.

        Args:
            interactions: Interactions to evaluate (uses stored if None)
            agent_type: Filter by agent type

        Returns:
            Evaluation report
        """
        t0 = time.time()

        # Get interactions
        if interactions is None:
            interactions = self._interactions

        if agent_type:
            interactions = [i for i in interactions if i.agent_type == agent_type]

        report = AgentEvalReport(
            agent_type=agent_type or "all",
            interactions_evaluated=len(interactions),
        )

        if not interactions:
            return report

        # Aggregate scores
        relevances = []
        accuracies = []
        helpfulness_scores = []
        factualities = []
        latencies = []
        total_tokens = 0
        quality_counts = {"high": 0, "medium": 0, "low": 0}
        distillation_candidates = []
        low_quality_ids = []

        for inter in interactions:
            relevances.append(inter.relevance_score)
            accuracies.append(inter.accuracy_score)
            helpfulness_scores.append(inter.helpfulness_score)
            factualities.append(inter.factuality_score)
            latencies.append(inter.latency_ms)
            total_tokens += inter.tokens_in + inter.tokens_out

            # Classify quality
            avg_score = np.mean([
                inter.relevance_score,
                inter.accuracy_score,
                inter.helpfulness_score,
                inter.factuality_score,
            ])

            if avg_score >= self.HIGH_QUALITY_THRESHOLD:
                quality_counts["high"] += 1
                if avg_score >= self.DISTILLATION_THRESHOLD:
                    distillation_candidates.append(inter)
            elif avg_score >= self.LOW_QUALITY_THRESHOLD:
                quality_counts["medium"] += 1
            else:
                quality_counts["low"] += 1
                low_quality_ids.append(inter.interaction_id)

        report.mean_relevance = float(np.mean(relevances))
        report.mean_accuracy = float(np.mean(accuracies))
        report.mean_helpfulness = float(np.mean(helpfulness_scores))
        report.mean_factuality = float(np.mean(factualities))
        report.mean_latency_ms = float(np.mean(latencies))
        report.total_tokens = total_tokens
        report.quality_distribution = quality_counts
        report.low_quality_samples = low_quality_ids[:10]
        report.distillation_candidates = len(distillation_candidates)

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        report.elapsed_sec = time.time() - t0

        # Export distillation candidates
        if distillation_candidates:
            self._export_distillation_data(distillation_candidates)

        return report

    def add_human_feedback(
        self,
        interaction_id: str,
        rating: int,
        feedback: str = ""
    ) -> bool:
        """Add human feedback to an interaction.

        Args:
            interaction_id: ID of interaction
            rating: Human rating (1-5)
            feedback: Optional text feedback

        Returns:
            True if feedback was added
        """
        for inter in self._interactions:
            if inter.interaction_id == interaction_id:
                inter.human_rating = rating
                inter.human_feedback = feedback
                self._save_interaction(inter)
                return True
        return False

    def get_distillation_data(
        self,
        min_quality: float = 0.85,
        limit: int = 1000
    ) -> List[Dict]:
        """Get high-quality interactions for model distillation.

        Args:
            min_quality: Minimum average quality score
            limit: Maximum number of samples

        Returns:
            List of training examples
        """
        examples = []

        for inter in self._interactions:
            avg_score = np.mean([
                inter.relevance_score,
                inter.accuracy_score,
                inter.helpfulness_score,
                inter.factuality_score,
            ])

            if avg_score >= min_quality:
                examples.append({
                    "query": inter.query,
                    "response": inter.response,
                    "context": inter.context,
                    "agent_type": inter.agent_type,
                    "quality_score": avg_score,
                })

                if len(examples) >= limit:
                    break

        return examples

    # ── Auto-Scoring ────────────────────────────────────────────────────

    def _auto_score(self, interaction: AgentInteraction) -> None:
        """Auto-score an interaction using heuristics."""
        query = interaction.query.lower()
        response = interaction.response.lower()

        # Relevance: Does response address the query?
        query_terms = set(query.split())
        response_terms = set(response.split())
        term_overlap = len(query_terms & response_terms) / max(len(query_terms), 1)
        interaction.relevance_score = min(1.0, term_overlap * 2 + 0.3)

        # Accuracy: Check for hedging/uncertainty markers
        uncertainty_markers = ["i'm not sure", "i think", "possibly", "maybe", "might be"]
        uncertainty_count = sum(1 for m in uncertainty_markers if m in response)
        interaction.accuracy_score = max(0.3, 1.0 - uncertainty_count * 0.15)

        # Helpfulness: Response length and structure
        response_len = len(response)
        has_structure = any(c in response for c in ["1.", "- ", "* ", "\n\n"])
        helpfulness = min(1.0, response_len / 500) * 0.6
        if has_structure:
            helpfulness += 0.3
        if len(interaction.context) > 0:
            helpfulness += 0.1
        interaction.helpfulness_score = min(1.0, helpfulness)

        # Factuality: Presence of citations/references
        has_citation = any(m in response for m in ["pmid", "doi:", "et al", "[", "]"])
        has_numbers = any(c.isdigit() for c in response)
        interaction.factuality_score = 0.5
        if has_citation:
            interaction.factuality_score += 0.3
        if has_numbers:
            interaction.factuality_score += 0.2

    # ── Storage ─────────────────────────────────────────────────────────

    def _load_interactions(self) -> None:
        """Load stored interactions."""
        interactions_file = self._storage_dir / f"{self._model_name}_interactions.jsonl"
        if interactions_file.exists():
            try:
                with open(interactions_file) as f:
                    for line in f:
                        data = json.loads(line)
                        inter = AgentInteraction(**data)
                        self._interactions.append(inter)
                log.info("Loaded %d interactions", len(self._interactions))
            except Exception as e:
                log.warning("Failed to load interactions: %s", e)

    def _save_interaction(self, interaction: AgentInteraction) -> None:
        """Save interaction to storage."""
        interactions_file = self._storage_dir / f"{self._model_name}_interactions.jsonl"
        with open(interactions_file, "a") as f:
            data = {
                "interaction_id": interaction.interaction_id,
                "agent_type": interaction.agent_type,
                "query": interaction.query,
                "response": interaction.response,
                "context": interaction.context,
                "tools_used": interaction.tools_used,
                "latency_ms": interaction.latency_ms,
                "tokens_in": interaction.tokens_in,
                "tokens_out": interaction.tokens_out,
                "timestamp": interaction.timestamp,
                "relevance_score": interaction.relevance_score,
                "accuracy_score": interaction.accuracy_score,
                "helpfulness_score": interaction.helpfulness_score,
                "factuality_score": interaction.factuality_score,
                "human_rating": interaction.human_rating,
                "human_feedback": interaction.human_feedback,
            }
            f.write(json.dumps(data) + "\n")

    def _export_distillation_data(
        self,
        candidates: List[AgentInteraction]
    ) -> None:
        """Export distillation training data."""
        export_file = self._storage_dir / f"{self._model_name}_distillation.jsonl"

        with open(export_file, "w") as f:
            for inter in candidates:
                example = {
                    "messages": [
                        {"role": "user", "content": inter.query},
                        {"role": "assistant", "content": inter.response},
                    ],
                    "context": inter.context,
                    "agent_type": inter.agent_type,
                }
                f.write(json.dumps(example) + "\n")

        log.info("Exported %d distillation examples to %s", len(candidates), export_file)

    def _generate_recommendations(self, report: AgentEvalReport) -> List[str]:
        """Generate recommendations."""
        recommendations = []

        if report.mean_relevance < 0.7:
            recommendations.append(
                "Improve query understanding - consider fine-tuning on domain queries"
            )

        if report.mean_factuality < 0.7:
            recommendations.append(
                "Increase citation usage - integrate RAG more tightly"
            )

        if report.mean_latency_ms > 5000:
            recommendations.append(
                f"High latency ({report.mean_latency_ms:.0f}ms) - "
                "consider smaller model or caching"
            )

        if report.quality_distribution.get("low", 0) > report.interactions_evaluated * 0.2:
            recommendations.append(
                "High proportion of low-quality responses - review prompts"
            )

        if report.distillation_candidates > 100:
            recommendations.append(
                f"{report.distillation_candidates} high-quality examples ready for distillation"
            )

        return recommendations
