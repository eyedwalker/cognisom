"""
Evaluation Framework (Phase 5)
==============================

Comprehensive evaluation and benchmarking for Cognisom simulations.

This package provides:
- ``simulation_accuracy``: Compare predictions to experimental outcomes
- ``drug_response_eval``: Validate drug candidate rankings
- ``tissue_fidelity``: Compare digital twin to histology data
- ``agent_eval``: Evaluate AI agent performance
- ``metrics``: Standard metrics and scoring functions

Usage::

    from cognisom.eval import SimulationEvaluator, DrugResponseEvaluator

    # Evaluate simulation accuracy
    evaluator = SimulationEvaluator()
    report = evaluator.evaluate_against_benchmarks()
    print(report.summary())

    # Evaluate drug predictions
    drug_eval = DrugResponseEvaluator()
    rankings = drug_eval.evaluate_candidates(candidates, ground_truth)
"""

from .simulation_accuracy import SimulationEvaluator, AccuracyReport
from .drug_response_eval import DrugResponseEvaluator, DrugEvalReport
from .tissue_fidelity import TissueFidelityEvaluator, FidelityReport
from .agent_eval import AgentEvaluator, AgentEvalReport
from .metrics import (
    mean_absolute_error,
    mean_squared_error,
    correlation_coefficient,
    concordance_index,
    ndcg_score,
    precision_at_k,
)

__all__ = [
    "SimulationEvaluator",
    "AccuracyReport",
    "DrugResponseEvaluator",
    "DrugEvalReport",
    "TissueFidelityEvaluator",
    "FidelityReport",
    "AgentEvaluator",
    "AgentEvalReport",
    "mean_absolute_error",
    "mean_squared_error",
    "correlation_coefficient",
    "concordance_index",
    "ndcg_score",
    "precision_at_k",
]
