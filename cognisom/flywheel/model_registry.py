"""
Model Registry (Phase 5)
========================

Version control and A/B testing for models.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


class ModelStatus(str, Enum):
    """Model deployment status."""
    REGISTERED = "registered"
    TESTING = "testing"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    ROLLED_BACK = "rolled_back"


@dataclass
class ModelVersion:
    """A registered model version."""
    version_id: str = ""
    model_path: str = ""
    base_model: str = ""
    created_at: float = 0.0
    status: ModelStatus = ModelStatus.REGISTERED
    training_samples: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    traffic_percent: int = 0
    ab_test_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "model_path": self.model_path,
            "base_model": self.base_model,
            "created_at": self.created_at,
            "status": self.status.value,
            "training_samples": self.training_samples,
            "metrics": self.metrics,
            "traffic_percent": self.traffic_percent,
            "ab_test_id": self.ab_test_id,
            "tags": self.tags,
            "metadata": self.metadata,
        }


@dataclass
class ABTest:
    """An A/B test configuration."""
    test_id: str = ""
    control_version: str = ""         # Current production model
    treatment_version: str = ""       # New model being tested
    control_traffic: int = 90         # Percent traffic to control
    treatment_traffic: int = 10       # Percent traffic to treatment
    start_time: float = 0.0
    end_time: Optional[float] = None
    control_metrics: Dict[str, float] = field(default_factory=dict)
    treatment_metrics: Dict[str, float] = field(default_factory=dict)
    winner: Optional[str] = None
    status: str = "running"           # running, completed, stopped


class ModelRegistry:
    """Registry for model versioning and deployment.

    Provides:
    - Model version registration
    - A/B testing for deployments
    - Rollback capability
    - Metrics tracking
    """

    def __init__(self, registry_dir: str = "data/model_registry") -> None:
        """Initialize the model registry.

        Args:
            registry_dir: Directory for registry data
        """
        self._registry_dir = Path(registry_dir)
        self._registry_dir.mkdir(parents=True, exist_ok=True)

        self._versions: Dict[str, ModelVersion] = {}
        self._ab_tests: Dict[str, ABTest] = {}
        self._current_production: Optional[str] = None

        self._load_registry()

    def register(
        self,
        model_path: str,
        base_model: str = "",
        metrics: Optional[Dict[str, float]] = None,
        training_samples: int = 0,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> ModelVersion:
        """Register a new model version.

        Args:
            model_path: Path to model artifacts
            base_model: Base model name
            metrics: Model metrics
            training_samples: Number of training samples
            tags: Version tags
            metadata: Additional metadata

        Returns:
            Registered model version
        """
        version_id = f"v{int(time.time())}"

        version = ModelVersion(
            version_id=version_id,
            model_path=model_path,
            base_model=base_model,
            training_samples=training_samples,
            metrics=metrics or {},
            tags=tags or [],
            metadata=metadata or {},
        )

        self._versions[version_id] = version
        self._save_registry()

        log.info("Registered model version %s", version_id)
        return version

    def get(self, version_id: str) -> Optional[ModelVersion]:
        """Get a model version by ID."""
        return self._versions.get(version_id)

    def list_versions(
        self,
        status: Optional[ModelStatus] = None,
        tags: Optional[List[str]] = None
    ) -> List[ModelVersion]:
        """List model versions with optional filtering.

        Args:
            status: Filter by status
            tags: Filter by tags (any match)

        Returns:
            List of matching versions
        """
        versions = list(self._versions.values())

        if status:
            versions = [v for v in versions if v.status == status]

        if tags:
            tag_set = set(tags)
            versions = [v for v in versions if tag_set & set(v.tags)]

        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def deploy(
        self,
        version_id: str,
        traffic_percent: int = 100,
        ab_test: bool = False
    ) -> bool:
        """Deploy a model version.

        Args:
            version_id: Version to deploy
            traffic_percent: Percentage of traffic (1-100)
            ab_test: Enable A/B testing with current production

        Returns:
            True if deployment successful
        """
        version = self._versions.get(version_id)
        if not version:
            log.error("Version %s not found", version_id)
            return False

        if ab_test and self._current_production:
            # Set up A/B test
            test_id = f"ab_{int(time.time())}"
            ab = ABTest(
                test_id=test_id,
                control_version=self._current_production,
                treatment_version=version_id,
                control_traffic=100 - traffic_percent,
                treatment_traffic=traffic_percent,
                start_time=time.time(),
            )
            self._ab_tests[test_id] = ab
            version.ab_test_id = test_id
            version.status = ModelStatus.TESTING
            log.info("Started A/B test %s: %s vs %s", test_id,
                    self._current_production, version_id)
        else:
            # Direct deployment
            if self._current_production:
                old = self._versions.get(self._current_production)
                if old:
                    old.status = ModelStatus.DEPRECATED
                    old.traffic_percent = 0

            version.status = ModelStatus.DEPLOYED
            version.traffic_percent = traffic_percent
            self._current_production = version_id
            log.info("Deployed version %s with %d%% traffic",
                    version_id, traffic_percent)

        self._save_registry()
        return True

    def rollback(self, to_version: Optional[str] = None) -> bool:
        """Rollback to a previous version.

        Args:
            to_version: Target version (uses previous if None)

        Returns:
            True if rollback successful
        """
        if to_version is None:
            # Find previous deployed version
            deployed = [v for v in self._versions.values()
                       if v.status in (ModelStatus.DEPRECATED, ModelStatus.DEPLOYED)]
            deployed.sort(key=lambda v: v.created_at, reverse=True)

            if len(deployed) < 2:
                log.error("No previous version to rollback to")
                return False

            to_version = deployed[1].version_id

        target = self._versions.get(to_version)
        if not target:
            log.error("Rollback target %s not found", to_version)
            return False

        # Update statuses
        if self._current_production:
            current = self._versions.get(self._current_production)
            if current:
                current.status = ModelStatus.ROLLED_BACK
                current.traffic_percent = 0

        target.status = ModelStatus.DEPLOYED
        target.traffic_percent = 100
        self._current_production = to_version

        self._save_registry()
        log.info("Rolled back to version %s", to_version)
        return True

    def record_ab_metrics(
        self,
        test_id: str,
        control_metrics: Dict[str, float],
        treatment_metrics: Dict[str, float]
    ) -> None:
        """Record metrics for an A/B test.

        Args:
            test_id: A/B test ID
            control_metrics: Metrics from control group
            treatment_metrics: Metrics from treatment group
        """
        test = self._ab_tests.get(test_id)
        if not test:
            return

        test.control_metrics = control_metrics
        test.treatment_metrics = treatment_metrics
        self._save_registry()

    def conclude_ab_test(
        self,
        test_id: str,
        promote_treatment: bool
    ) -> Optional[str]:
        """Conclude an A/B test.

        Args:
            test_id: Test to conclude
            promote_treatment: If True, promote treatment to production

        Returns:
            Winner version ID
        """
        test = self._ab_tests.get(test_id)
        if not test:
            log.error("A/B test %s not found", test_id)
            return None

        test.end_time = time.time()
        test.status = "completed"

        if promote_treatment:
            test.winner = test.treatment_version
            self.deploy(test.treatment_version, traffic_percent=100)
        else:
            test.winner = test.control_version
            # Deprecate treatment
            treatment = self._versions.get(test.treatment_version)
            if treatment:
                treatment.status = ModelStatus.DEPRECATED
                treatment.traffic_percent = 0

        self._save_registry()
        log.info("A/B test %s concluded, winner: %s", test_id, test.winner)
        return test.winner

    def get_current_production(self) -> Optional[ModelVersion]:
        """Get currently deployed production model."""
        if self._current_production:
            return self._versions.get(self._current_production)
        return None

    def get_active_ab_tests(self) -> List[ABTest]:
        """Get active A/B tests."""
        return [t for t in self._ab_tests.values() if t.status == "running"]

    # ── Persistence ─────────────────────────────────────────────────────

    def _load_registry(self) -> None:
        """Load registry from disk."""
        registry_file = self._registry_dir / "registry.json"
        if registry_file.exists():
            try:
                data = json.loads(registry_file.read_text())

                for v_data in data.get("versions", []):
                    v = ModelVersion(
                        version_id=v_data["version_id"],
                        model_path=v_data["model_path"],
                        base_model=v_data.get("base_model", ""),
                        created_at=v_data["created_at"],
                        status=ModelStatus(v_data["status"]),
                        training_samples=v_data.get("training_samples", 0),
                        metrics=v_data.get("metrics", {}),
                        traffic_percent=v_data.get("traffic_percent", 0),
                        tags=v_data.get("tags", []),
                    )
                    self._versions[v.version_id] = v

                self._current_production = data.get("current_production")

                log.info("Loaded %d model versions", len(self._versions))

            except Exception as e:
                log.warning("Failed to load registry: %s", e)

    def _save_registry(self) -> None:
        """Save registry to disk."""
        registry_file = self._registry_dir / "registry.json"

        data = {
            "versions": [v.to_dict() for v in self._versions.values()],
            "current_production": self._current_production,
            "ab_tests": [
                {
                    "test_id": t.test_id,
                    "control_version": t.control_version,
                    "treatment_version": t.treatment_version,
                    "status": t.status,
                    "winner": t.winner,
                }
                for t in self._ab_tests.values()
            ],
        }

        registry_file.write_text(json.dumps(data, indent=2))
