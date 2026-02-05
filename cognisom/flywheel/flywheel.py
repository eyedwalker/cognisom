"""
Data Flywheel Orchestration (Phase 5)
=====================================

Main orchestration for the continuous improvement loop.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Thread, Event
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)


class FlywheelStatus(str, Enum):
    """Flywheel operational status."""
    STOPPED = "stopped"
    RUNNING = "running"
    COLLECTING = "collecting"
    EVALUATING = "evaluating"
    DISTILLING = "distilling"
    DEPLOYING = "deploying"
    ERROR = "error"


@dataclass
class FlywheelConfig:
    """Data flywheel configuration."""
    data_dir: str = "data/flywheel"
    min_samples_for_distillation: int = 500
    quality_threshold: float = 0.85
    evaluation_interval_hours: float = 24.0
    distillation_interval_hours: float = 168.0  # Weekly
    auto_deploy: bool = False
    teacher_model: str = "nemotron-49b"
    student_model: str = "nemotron-8b"
    lora_rank: int = 16
    max_training_samples: int = 10000


@dataclass
class FlywheelMetrics:
    """Flywheel performance metrics."""
    total_interactions: int = 0
    high_quality_samples: int = 0
    distillations_run: int = 0
    models_deployed: int = 0
    cost_savings_pct: float = 0.0
    quality_improvement_pct: float = 0.0
    last_evaluation: Optional[float] = None
    last_distillation: Optional[float] = None


class DataFlywheel:
    """Orchestrates the data flywheel for continuous improvement.

    The flywheel:
    1. Continuously collects agent interactions
    2. Evaluates quality and identifies training candidates
    3. Periodically runs model distillation
    4. Deploys improved models with A/B testing
    """

    def __init__(
        self,
        config: Optional[FlywheelConfig] = None,
        evaluator=None,
        registry=None
    ) -> None:
        """Initialize the data flywheel.

        Args:
            config: Flywheel configuration
            evaluator: AgentEvaluator instance
            registry: ModelRegistry instance
        """
        self._config = config or FlywheelConfig()
        self._status = FlywheelStatus.STOPPED
        self._metrics = FlywheelMetrics()

        self._data_dir = Path(self._config.data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Components (lazy initialization)
        self._evaluator = evaluator
        self._registry = registry
        self._distillation_pipeline = None
        self._feedback_collector = None

        # Threading
        self._stop_event = Event()
        self._thread: Optional[Thread] = None

        # Callbacks
        self._on_evaluation: List[Callable] = []
        self._on_distillation: List[Callable] = []
        self._on_deployment: List[Callable] = []

    @property
    def status(self) -> FlywheelStatus:
        """Current flywheel status."""
        return self._status

    @property
    def metrics(self) -> FlywheelMetrics:
        """Current metrics."""
        return self._metrics

    # ── Lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the data flywheel."""
        if self._status == FlywheelStatus.RUNNING:
            log.warning("Flywheel already running")
            return

        self._initialize_components()
        self._status = FlywheelStatus.RUNNING
        self._stop_event.clear()

        self._thread = Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        log.info("Data flywheel started")

    def stop(self) -> None:
        """Stop the data flywheel."""
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

        self._status = FlywheelStatus.STOPPED
        log.info("Data flywheel stopped")

    def _initialize_components(self) -> None:
        """Initialize flywheel components."""
        from cognisom.eval import AgentEvaluator

        if self._evaluator is None:
            self._evaluator = AgentEvaluator(
                storage_dir=str(self._data_dir / "interactions")
            )

        if self._registry is None:
            from .model_registry import ModelRegistry
            self._registry = ModelRegistry(
                registry_dir=str(self._data_dir / "models")
            )

        if self._distillation_pipeline is None:
            from .distillation import DistillationPipeline
            self._distillation_pipeline = DistillationPipeline(
                output_dir=str(self._data_dir / "distilled"),
                teacher_model=self._config.teacher_model,
                student_model=self._config.student_model,
            )

        if self._feedback_collector is None:
            from .feedback import FeedbackCollector
            self._feedback_collector = FeedbackCollector(
                storage_dir=str(self._data_dir / "feedback")
            )

    # ── Main Loop ───────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        """Main flywheel loop."""
        eval_interval = self._config.evaluation_interval_hours * 3600
        distill_interval = self._config.distillation_interval_hours * 3600

        while not self._stop_event.is_set():
            try:
                current_time = time.time()

                # Check if evaluation needed
                if (self._metrics.last_evaluation is None or
                    current_time - self._metrics.last_evaluation >= eval_interval):
                    self._run_evaluation()

                # Check if distillation needed
                if (self._metrics.high_quality_samples >= self._config.min_samples_for_distillation and
                    (self._metrics.last_distillation is None or
                     current_time - self._metrics.last_distillation >= distill_interval)):
                    self._run_distillation_cycle()

                # Sleep for a bit
                self._stop_event.wait(timeout=60.0)

            except Exception as e:
                log.error("Flywheel error: %s", e)
                self._status = FlywheelStatus.ERROR
                time.sleep(60.0)

    # ── Evaluation ──────────────────────────────────────────────────────

    def _run_evaluation(self) -> None:
        """Run quality evaluation on collected interactions."""
        self._status = FlywheelStatus.EVALUATING
        log.info("Running flywheel evaluation...")

        try:
            report = self._evaluator.evaluate_batch()

            self._metrics.total_interactions = report.interactions_evaluated
            self._metrics.high_quality_samples = report.distillation_candidates
            self._metrics.last_evaluation = time.time()

            log.info(
                "Evaluation complete: %d interactions, %d high-quality",
                report.interactions_evaluated,
                report.distillation_candidates
            )

            # Notify callbacks
            for callback in self._on_evaluation:
                try:
                    callback(report)
                except Exception as e:
                    log.warning("Evaluation callback error: %s", e)

        except Exception as e:
            log.error("Evaluation failed: %s", e)

        self._status = FlywheelStatus.RUNNING

    def run_evaluation(self):
        """Manually trigger evaluation."""
        self._run_evaluation()
        return self._evaluator.evaluate_batch()

    # ── Distillation ────────────────────────────────────────────────────

    def _run_distillation_cycle(self) -> None:
        """Run model distillation cycle."""
        self._status = FlywheelStatus.DISTILLING
        log.info("Running distillation cycle...")

        try:
            # Get high-quality training data
            training_data = self._evaluator.get_distillation_data(
                min_quality=self._config.quality_threshold,
                limit=self._config.max_training_samples
            )

            if len(training_data) < self._config.min_samples_for_distillation:
                log.info("Not enough samples for distillation (%d)", len(training_data))
                return

            # Run distillation
            result = self._distillation_pipeline.distill(
                training_data=training_data,
                lora_rank=self._config.lora_rank
            )

            self._metrics.distillations_run += 1
            self._metrics.last_distillation = time.time()

            log.info("Distillation complete: %s", result.model_path)

            # Register new model
            version = self._registry.register(
                model_path=result.model_path,
                metrics=result.metrics,
                training_samples=len(training_data),
            )

            # Notify callbacks
            for callback in self._on_distillation:
                try:
                    callback(result, version)
                except Exception as e:
                    log.warning("Distillation callback error: %s", e)

            # Auto-deploy if configured
            if self._config.auto_deploy and result.quality_improved:
                self._deploy_model(version)

        except Exception as e:
            log.error("Distillation failed: %s", e)

        self._status = FlywheelStatus.RUNNING

    def run_distillation(self, force: bool = False):
        """Manually trigger distillation.

        Args:
            force: Run even if below sample threshold
        """
        if force:
            self._config.min_samples_for_distillation = 0

        self._run_distillation_cycle()

    # ── Deployment ──────────────────────────────────────────────────────

    def _deploy_model(self, version) -> None:
        """Deploy a new model version."""
        self._status = FlywheelStatus.DEPLOYING
        log.info("Deploying model version %s...", version.version_id)

        try:
            # Deploy with A/B testing (10% traffic initially)
            self._registry.deploy(
                version_id=version.version_id,
                traffic_percent=10,
                ab_test=True
            )

            self._metrics.models_deployed += 1

            # Notify callbacks
            for callback in self._on_deployment:
                try:
                    callback(version)
                except Exception as e:
                    log.warning("Deployment callback error: %s", e)

            log.info("Model deployed with 10%% traffic")

        except Exception as e:
            log.error("Deployment failed: %s", e)

        self._status = FlywheelStatus.RUNNING

    def deploy_model(self, version_id: str, traffic_percent: int = 100) -> bool:
        """Manually deploy a model version.

        Args:
            version_id: Version to deploy
            traffic_percent: Percentage of traffic to route

        Returns:
            True if deployment successful
        """
        try:
            self._registry.deploy(version_id, traffic_percent)
            return True
        except Exception as e:
            log.error("Manual deployment failed: %s", e)
            return False

    # ── Callbacks ───────────────────────────────────────────────────────

    def on_evaluation(self, callback: Callable) -> None:
        """Register evaluation callback."""
        self._on_evaluation.append(callback)

    def on_distillation(self, callback: Callable) -> None:
        """Register distillation callback."""
        self._on_distillation.append(callback)

    def on_deployment(self, callback: Callable) -> None:
        """Register deployment callback."""
        self._on_deployment.append(callback)

    # ── Status ──────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Get flywheel status."""
        return {
            "status": self._status.value,
            "metrics": {
                "total_interactions": self._metrics.total_interactions,
                "high_quality_samples": self._metrics.high_quality_samples,
                "distillations_run": self._metrics.distillations_run,
                "models_deployed": self._metrics.models_deployed,
                "cost_savings_pct": self._metrics.cost_savings_pct,
            },
            "last_evaluation": (
                datetime.fromtimestamp(self._metrics.last_evaluation).isoformat()
                if self._metrics.last_evaluation else None
            ),
            "last_distillation": (
                datetime.fromtimestamp(self._metrics.last_distillation).isoformat()
                if self._metrics.last_distillation else None
            ),
            "config": {
                "teacher_model": self._config.teacher_model,
                "student_model": self._config.student_model,
                "quality_threshold": self._config.quality_threshold,
                "auto_deploy": self._config.auto_deploy,
            },
        }
