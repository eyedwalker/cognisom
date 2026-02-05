"""
Integration Tests: Evaluation Framework & Data Flywheel (Phase 5)
==================================================================

Tests that the Phase 5 components work correctly:
1. Evaluation metrics computation
2. Simulation accuracy evaluation
3. Drug response evaluation
4. Tissue fidelity evaluation
5. Agent evaluation and interaction capture
6. Data flywheel orchestration
7. Model distillation pipeline
8. Model registry and A/B testing
9. Feedback collection

Run: pytest cognisom/tests/test_eval_flywheel.py -v
"""

import sys
from pathlib import Path

# Ensure project root on path
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import pytest
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Evaluation Metrics Tests
# ═══════════════════════════════════════════════════════════════════════

class TestEvalMetrics:
    """Test evaluation metrics module."""

    def test_mae(self):
        from cognisom.eval.metrics import mean_absolute_error
        actual = [1.0, 2.0, 3.0, 4.0]
        predicted = [1.1, 1.9, 3.2, 3.8]
        mae = mean_absolute_error(actual, predicted)
        assert 0.1 < mae < 0.2

    def test_mse(self):
        from cognisom.eval.metrics import mean_squared_error
        actual = [1.0, 2.0, 3.0, 4.0]
        predicted = [1.0, 2.0, 3.0, 4.0]
        mse = mean_squared_error(actual, predicted)
        assert mse == 0.0

    def test_correlation(self):
        from cognisom.eval.metrics import correlation_coefficient
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]  # Perfect correlation
        corr = correlation_coefficient(x, y)
        assert corr == pytest.approx(1.0, abs=0.001)

    def test_spearman(self):
        """Test spearman correlation - requires scipy."""
        try:
            from scipy import stats
        except ImportError:
            pytest.skip("scipy not available")
        from cognisom.eval.metrics import spearman_correlation
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]  # Perfect negative rank correlation
        corr = spearman_correlation(x, y)
        assert corr == pytest.approx(-1.0, abs=0.001)

    def test_ndcg(self):
        from cognisom.eval.metrics import ndcg_score
        # Test with perfect ranking (relevance already sorted)
        relevance = [3, 2, 1, 0, 0]  # Already ideal order
        predicted = [1.0, 0.8, 0.6, 0.4, 0.2]  # Matches the order
        score = ndcg_score(relevance, predicted, k=5)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_precision_at_k(self):
        from cognisom.eval.metrics import precision_at_k
        # Binary relevance labels
        y_true = [1, 0, 1, 0, 1]  # Items 0, 2, 4 are relevant
        y_pred = [0.9, 0.8, 0.7, 0.3, 0.1]  # Ranking by score
        # Top 3: indices 0, 1, 2 -> relevance 1, 0, 1 -> 2 relevant in top 3
        p3 = precision_at_k(y_true, y_pred, k=3)
        assert p3 == pytest.approx(2/3, abs=0.01)

    def test_r_squared(self):
        from cognisom.eval.metrics import r_squared
        actual = [1.0, 2.0, 3.0, 4.0, 5.0]
        predicted = [1.0, 2.0, 3.0, 4.0, 5.0]
        r2 = r_squared(actual, predicted)
        assert r2 == pytest.approx(1.0, abs=0.001)

    def test_auc_roc(self):
        """Test AUC-ROC - requires sklearn."""
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            pytest.skip("sklearn not available")
        from cognisom.eval.metrics import auc_roc
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.4, 0.35, 0.8]
        auc = auc_roc(y_true, y_score)
        assert 0.5 < auc <= 1.0


# ═══════════════════════════════════════════════════════════════════════
# Simulation Accuracy Tests
# ═══════════════════════════════════════════════════════════════════════

class TestSimulationAccuracy:
    """Test simulation accuracy evaluator."""

    def test_evaluator_creation(self):
        from cognisom.eval.simulation_accuracy import SimulationEvaluator
        evaluator = SimulationEvaluator()
        assert evaluator is not None

    def test_builtin_benchmarks(self):
        from cognisom.eval.simulation_accuracy import SimulationEvaluator
        evaluator = SimulationEvaluator()
        assert len(evaluator._benchmarks) > 0

    def test_report_has_grade(self):
        from cognisom.eval.simulation_accuracy import SimulationEvaluator
        evaluator = SimulationEvaluator()

        # Empty results - still generates a report with grade
        sim_results = {}
        report = evaluator.evaluate_against_benchmarks(sim_results)
        assert hasattr(report, "overall_grade")
        assert report.overall_grade in ["A", "B", "C", "D", "F"]


# ═══════════════════════════════════════════════════════════════════════
# Drug Response Evaluation Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDrugResponseEval:
    """Test drug response evaluator."""

    def test_evaluator_creation(self):
        from cognisom.eval.drug_response_eval import DrugResponseEvaluator
        evaluator = DrugResponseEvaluator()
        assert evaluator is not None

    def test_builtin_ground_truth(self):
        from cognisom.eval.drug_response_eval import DrugResponseEvaluator
        evaluator = DrugResponseEvaluator()
        assert len(evaluator._ground_truth) > 0

    def test_evaluate_with_dicts(self):
        """Test drug candidate evaluation - requires scipy."""
        try:
            from scipy import stats
        except ImportError:
            pytest.skip("scipy not available (required for drug response eval)")

        from cognisom.eval.drug_response_eval import DrugResponseEvaluator
        evaluator = DrugResponseEvaluator()

        # Use dicts instead of DrugCandidate objects
        candidates = [
            {
                "drug_name": "enzalutamide",
                "predicted_ic50": 0.05,
                "predicted_efficacy": 0.85,
            },
            {
                "drug_name": "docetaxel",
                "predicted_ic50": 0.01,
                "predicted_efficacy": 0.90,
            },
        ]

        report = evaluator.evaluate_candidates(candidates)
        assert hasattr(report, "ranking_metrics")
        assert hasattr(report, "ic50_metrics")


# ═══════════════════════════════════════════════════════════════════════
# Tissue Fidelity Tests
# ═══════════════════════════════════════════════════════════════════════

class TestTissueFidelity:
    """Test tissue fidelity evaluator."""

    def test_evaluator_creation(self):
        from cognisom.eval.tissue_fidelity import TissueFidelityEvaluator
        evaluator = TissueFidelityEvaluator()
        assert evaluator is not None

    def test_evaluate_returns_report(self):
        from cognisom.eval.tissue_fidelity import TissueFidelityEvaluator
        evaluator = TissueFidelityEvaluator()

        # Create synthetic tissue data
        tissue_data = {
            "cell_positions": np.random.randn(100, 3) * 50,
            "cell_types": ["epithelial"] * 60 + ["stromal"] * 30 + ["immune"] * 10,
            "cell_volumes": np.random.uniform(500, 2000, 100),
        }

        report = evaluator.evaluate(tissue_data)
        assert hasattr(report, "overall_fidelity")
        assert hasattr(report, "grade")
        # Fidelity can be 0-100 as a percentage
        assert 0 <= report.overall_fidelity <= 100


# ═══════════════════════════════════════════════════════════════════════
# Agent Evaluator Tests
# ═══════════════════════════════════════════════════════════════════════

class TestAgentEvaluator:
    """Test agent evaluation module."""

    def test_evaluator_creation(self, tmp_path):
        from cognisom.eval.agent_eval import AgentEvaluator
        evaluator = AgentEvaluator(storage_dir=str(tmp_path))
        assert evaluator is not None

    def test_capture_interaction(self, tmp_path):
        from cognisom.eval.agent_eval import AgentEvaluator
        evaluator = AgentEvaluator(storage_dir=str(tmp_path))

        interaction = evaluator.capture(
            query="What is the role of TP53 in cancer?",
            response="TP53 is a tumor suppressor gene...",
            agent_type="researcher",
            latency_ms=250.0,
        )

        assert interaction.query == "What is the role of TP53 in cancer?"
        assert interaction.agent_type == "researcher"
        assert interaction.relevance_score > 0  # Auto-scored

    def test_evaluate_batch(self, tmp_path):
        from cognisom.eval.agent_eval import AgentEvaluator
        evaluator = AgentEvaluator(storage_dir=str(tmp_path))

        # Capture multiple interactions
        for i in range(5):
            evaluator.capture(
                query=f"Question {i}",
                response=f"Answer {i} with some detail.",
                agent_type="researcher",
            )

        report = evaluator.evaluate_batch()
        assert report.interactions_evaluated == 5
        assert report.overall_quality >= 0

    def test_distillation_data_export(self, tmp_path):
        from cognisom.eval.agent_eval import AgentEvaluator
        evaluator = AgentEvaluator(storage_dir=str(tmp_path))

        # High quality interaction
        evaluator.capture(
            query="Explain metabolic pathways",
            response="Metabolic pathways are series of chemical reactions [1]. "
                     "The citric acid cycle et al. involves ATP production.",
            agent_type="researcher",
        )

        data = evaluator.get_distillation_data(min_quality=0.0)
        assert len(data) > 0


# ═══════════════════════════════════════════════════════════════════════
# Data Flywheel Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDataFlywheel:
    """Test data flywheel orchestration."""

    def test_flywheel_creation(self, tmp_path):
        from cognisom.flywheel.flywheel import DataFlywheel, FlywheelConfig

        config = FlywheelConfig(
            data_dir=str(tmp_path),
            evaluation_interval_hours=1,
        )
        flywheel = DataFlywheel(config)
        assert flywheel is not None

    def test_flywheel_status(self, tmp_path):
        from cognisom.flywheel.flywheel import (
            DataFlywheel,
            FlywheelConfig,
            FlywheelStatus,
        )

        config = FlywheelConfig(data_dir=str(tmp_path))
        flywheel = DataFlywheel(config)

        assert flywheel.status == FlywheelStatus.STOPPED

    def test_flywheel_callbacks(self, tmp_path):
        from cognisom.flywheel.flywheel import DataFlywheel, FlywheelConfig

        config = FlywheelConfig(data_dir=str(tmp_path))
        flywheel = DataFlywheel(config)

        callback_called = []

        def on_eval(report):
            callback_called.append("eval")

        flywheel.on_evaluation_complete = on_eval

        # Verify callback is set
        assert flywheel.on_evaluation_complete is not None


# ═══════════════════════════════════════════════════════════════════════
# Distillation Pipeline Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDistillationPipeline:
    """Test model distillation pipeline."""

    def test_pipeline_creation(self, tmp_path):
        from cognisom.flywheel.distillation import DistillationPipeline

        pipeline = DistillationPipeline(
            output_dir=str(tmp_path),
            teacher_model="nemotron-49b",
            student_model="nemotron-8b",
        )
        assert pipeline is not None

    def test_config_creation(self):
        from cognisom.flywheel.distillation import DistillationConfig

        config = DistillationConfig(
            lora_rank=8,
            lora_alpha=16,
        )

        assert config.lora_rank == 8
        assert config.lora_alpha == 16

    def test_distill_returns_result(self, tmp_path):
        from cognisom.flywheel.distillation import DistillationPipeline

        pipeline = DistillationPipeline(output_dir=str(tmp_path))

        # Minimal training data
        training_data = [
            {"query": "Q1", "response": "R1"},
            {"query": "Q2", "response": "R2"},
        ]

        result = pipeline.distill(training_data)
        assert hasattr(result, "cost_reduction_pct")
        assert hasattr(result, "latency_reduction_pct")


# ═══════════════════════════════════════════════════════════════════════
# Model Registry Tests
# ═══════════════════════════════════════════════════════════════════════

class TestModelRegistry:
    """Test model registry and versioning."""

    def test_registry_creation(self, tmp_path):
        from cognisom.flywheel.model_registry import ModelRegistry
        registry = ModelRegistry(registry_dir=str(tmp_path))
        assert registry is not None

    def test_register_model(self, tmp_path):
        from cognisom.flywheel.model_registry import ModelRegistry
        registry = ModelRegistry(registry_dir=str(tmp_path))

        version = registry.register(
            model_path="/fake/path",
            base_model="test-model",
            metrics={"accuracy": 0.95},
        )

        assert version.model_path == "/fake/path"
        assert version.base_model == "test-model"
        assert version.status.value == "registered"

    def test_list_versions(self, tmp_path):
        from cognisom.flywheel.model_registry import ModelRegistry
        registry = ModelRegistry(registry_dir=str(tmp_path))

        # Register a model and verify list_versions works
        v1 = registry.register("/path/a", base_model="model-a", metrics={"acc": 0.9})

        versions = registry.list_versions()
        assert len(versions) >= 1
        assert any(v.model_path == "/path/a" for v in versions)

        # Verify the version was registered with correct data
        retrieved = registry.get(v1.version_id)
        assert retrieved is not None
        assert retrieved.base_model == "model-a"

    def test_ab_test_creation(self, tmp_path):
        from cognisom.flywheel.model_registry import ModelRegistry
        registry = ModelRegistry(registry_dir=str(tmp_path))

        v1 = registry.register("/v1", base_model="model-x", metrics={"acc": 0.9})
        v2 = registry.register("/v2", base_model="model-x", metrics={"acc": 0.92})

        # Deploy first version
        registry.deploy(v1.version_id, traffic_percent=100)

        # Deploy second version with AB test
        success = registry.deploy(v2.version_id, traffic_percent=10, ab_test=True)
        assert success


# ═══════════════════════════════════════════════════════════════════════
# Feedback Collector Tests
# ═══════════════════════════════════════════════════════════════════════

class TestFeedbackCollector:
    """Test feedback collection system."""

    def test_collector_creation(self, tmp_path):
        from cognisom.flywheel.feedback import FeedbackCollector
        collector = FeedbackCollector(storage_dir=str(tmp_path))
        assert collector is not None

    def test_submit_rating(self, tmp_path):
        from cognisom.flywheel.feedback import FeedbackCollector, FeedbackType
        collector = FeedbackCollector(storage_dir=str(tmp_path))

        entry = collector.submit_rating(
            interaction_id="int_123",
            rating=4,
            comment="Helpful response",
        )

        assert entry.rating == 4
        assert entry.feedback_type == FeedbackType.RATING

    def test_submit_thumbs(self, tmp_path):
        from cognisom.flywheel.feedback import FeedbackCollector, FeedbackType
        collector = FeedbackCollector(storage_dir=str(tmp_path))

        entry = collector.submit_thumbs(
            interaction_id="int_456",
            thumbs_up=True,
        )

        assert entry.rating == 1
        assert entry.feedback_type == FeedbackType.THUMBS

    def test_submit_correction(self, tmp_path):
        from cognisom.flywheel.feedback import FeedbackCollector, FeedbackType
        collector = FeedbackCollector(storage_dir=str(tmp_path))

        entry = collector.submit_correction(
            interaction_id="int_789",
            original_response="The mitochondria is the powerhouse",
            corrected_response="The mitochondria are the powerhouses of the cell",
        )

        assert entry.feedback_type == FeedbackType.CORRECTION
        assert "powerhouses" in entry.corrected_response

    def test_get_summary(self, tmp_path):
        from cognisom.flywheel.feedback import FeedbackCollector
        collector = FeedbackCollector(storage_dir=str(tmp_path))

        # Add some feedback
        collector.submit_rating("int_1", 5)
        collector.submit_rating("int_2", 4)
        collector.submit_rating("int_3", 3)
        collector.submit_thumbs("int_4", True)
        collector.submit_thumbs("int_5", False)

        summary = collector.get_summary()
        assert summary.total_feedback == 5
        assert summary.average_rating == pytest.approx(4.0, abs=0.1)

    def test_corrections_for_training(self, tmp_path):
        from cognisom.flywheel.feedback import FeedbackCollector
        collector = FeedbackCollector(storage_dir=str(tmp_path))

        collector.submit_correction(
            "int_1",
            "Original A",
            "Corrected A",
        )
        collector.submit_correction(
            "int_2",
            "Original B",
            "Corrected B",
        )

        training_data = collector.get_corrections_for_training()
        assert len(training_data) == 2
        assert "corrected" in training_data[0]

    def test_preferences_for_rlhf(self, tmp_path):
        from cognisom.flywheel.feedback import FeedbackCollector
        collector = FeedbackCollector(storage_dir=str(tmp_path))

        collector.submit_preference(
            interaction_id="int_1",
            response_a="Response A is concise",
            response_b="Response B is detailed and thorough",
            preferred="b",
        )

        rlhf_data = collector.get_preferences_for_rlhf()
        assert len(rlhf_data) == 1
        assert rlhf_data[0]["chosen"] == "Response B is detailed and thorough"


# ═══════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════

class TestFlywheelIntegration:
    """Integration tests for the complete flywheel flow."""

    def test_end_to_end_evaluation_flow(self, tmp_path):
        """Test complete flow: capture -> evaluate -> feedback -> training data."""
        from cognisom.eval.agent_eval import AgentEvaluator
        from cognisom.flywheel.feedback import FeedbackCollector

        # 1. Capture interactions
        agent_eval = AgentEvaluator(storage_dir=str(tmp_path / "agent"))

        interaction = agent_eval.capture(
            query="What causes prostate cancer?",
            response="Prostate cancer is caused by genetic mutations [PMID:12345]. "
                     "Risk factors include age, family history, and diet.",
            agent_type="researcher",
        )

        # 2. Evaluate
        report = agent_eval.evaluate_batch()
        assert report.interactions_evaluated >= 1

        # 3. Collect feedback
        feedback = FeedbackCollector(storage_dir=str(tmp_path / "feedback"))
        feedback.submit_rating(
            interaction_id=interaction.interaction_id,
            rating=4,
            comment="Good answer",
        )

        # 4. Get training data
        corrections = feedback.get_corrections_for_training()
        distillation = agent_eval.get_distillation_data(min_quality=0.0)

        # Verify data flows correctly
        summary = feedback.get_summary()
        assert summary.total_feedback == 1
        assert len(distillation) >= 1
