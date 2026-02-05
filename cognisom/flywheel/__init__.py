"""
Data Flywheel (Phase 5)
=======================

Continuous improvement loop for Cognisom AI components.

This package provides:
- ``DataFlywheel``: Main orchestration for the improvement loop
- ``DistillationPipeline``: Model distillation from large to small
- ``ModelRegistry``: Version control and A/B testing for models
- ``FeedbackCollector``: Human feedback integration

The data flywheel:
1. Captures production agent interactions
2. Evaluates quality automatically
3. Identifies high-quality examples for training
4. Distills large models into smaller, cheaper ones
5. Deploys improved models with A/B testing

Usage::

    from cognisom.flywheel import DataFlywheel

    flywheel = DataFlywheel()
    flywheel.start()  # Begin continuous improvement

    # Manual distillation
    flywheel.run_distillation()
"""

from .flywheel import DataFlywheel, FlywheelConfig, FlywheelStatus
from .distillation import DistillationPipeline, DistillationConfig
from .model_registry import ModelRegistry, ModelVersion
from .feedback import FeedbackCollector, FeedbackEntry

__all__ = [
    "DataFlywheel",
    "FlywheelConfig",
    "FlywheelStatus",
    "DistillationPipeline",
    "DistillationConfig",
    "ModelRegistry",
    "ModelVersion",
    "FeedbackCollector",
    "FeedbackEntry",
]
