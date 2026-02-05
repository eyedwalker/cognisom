#!/usr/bin/env python3
"""
Cognisom Data Flywheel Demo
===========================

Demonstrates the Phase 5 Data Flywheel & Optimization features:
1. Agent interaction capture and evaluation
2. Simulation accuracy benchmarking
3. Feedback collection for RLHF
4. Model distillation pipeline
5. Model registry with A/B testing

Run: python examples/demo_flywheel.py
"""

import sys
import time
from pathlib import Path

# Ensure project root on path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)


def section(title: str):
    """Print a section header."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print()


def demo_agent_evaluation():
    """Demo: Agent interaction capture and evaluation."""
    section("1. Agent Evaluation")

    from cognisom.eval.agent_eval import AgentEvaluator

    # Create evaluator with temp storage
    evaluator = AgentEvaluator(storage_dir="data/demo_agent")

    # Simulate some research agent interactions
    queries_responses = [
        (
            "What is the role of AR in prostate cancer?",
            "The androgen receptor (AR) is a nuclear receptor that plays a "
            "central role in prostate cancer [PMID:12345]. In normal prostate, "
            "AR regulates growth and differentiation. In cancer, AR signaling "
            "becomes dysregulated, driving proliferation even at low androgen levels."
        ),
        (
            "How does enzalutamide work?",
            "Enzalutamide is a potent androgen receptor inhibitor that works via "
            "three mechanisms: 1) Competitive binding to AR ligand-binding domain, "
            "2) Preventing AR nuclear translocation, 3) Impairing DNA binding. "
            "It is effective in CRPC [et al., 2012]."
        ),
        (
            "Explain the Warburg effect",
            "The Warburg effect describes cancer cells' preference for glycolysis "
            "over oxidative phosphorylation, even with oxygen available. This "
            "metabolic shift supports rapid proliferation by providing biosynthetic "
            "precursors. ATP production is less efficient but faster."
        ),
    ]

    print("Capturing agent interactions...")
    for query, response in queries_responses:
        interaction = evaluator.capture(
            query=query,
            response=response,
            agent_type="researcher",
            latency_ms=250.0,
        )
        print(f"  - {query[:40]}...")
        print(f"    Relevance: {interaction.relevance_score:.2f}, "
              f"Helpfulness: {interaction.helpfulness_score:.2f}, "
              f"Factuality: {interaction.factuality_score:.2f}")

    # Evaluate batch
    print()
    print("Evaluating batch...")
    report = evaluator.evaluate_batch()
    print(f"  Interactions evaluated: {report.interactions_evaluated}")
    print(f"  Overall quality: {report.overall_quality:.1f}/100")
    print(f"  Mean relevance: {report.mean_relevance:.2f}")
    print(f"  Mean helpfulness: {report.mean_helpfulness:.2f}")
    print(f"  Distillation candidates: {report.distillation_candidates}")

    # Get distillation data
    distill_data = evaluator.get_distillation_data(min_quality=0.5)
    print(f"  High-quality examples for training: {len(distill_data)}")


def demo_simulation_accuracy():
    """Demo: Simulation accuracy evaluation."""
    section("2. Simulation Accuracy Evaluation")

    from cognisom.eval.simulation_accuracy import SimulationEvaluator

    evaluator = SimulationEvaluator()

    # Simulated results from a hypothetical simulation run
    sim_results = {
        "doubling_time_hours": 30.0,  # Close to benchmark ~28h
        "t_cell_density": 120.0,      # Cells/mm²
        "cell_counts_over_time": [
            (0, 100), (24, 195), (48, 380), (72, 750)
        ],
    }

    print("Running simulation accuracy evaluation...")
    print(f"  Doubling time: {sim_results['doubling_time_hours']}h")
    print(f"  T-cell density: {sim_results['t_cell_density']} cells/mm²")

    report = evaluator.evaluate_against_benchmarks(sim_results)

    print()
    print(f"Results:")
    print(f"  Overall Grade: {report.overall_grade}")
    print(f"  Benchmarks evaluated: {report.benchmarks_evaluated}")
    print(f"  Benchmarks passed: {report.benchmarks_passed}")
    print(f"  Pass rate: {report.pass_rate * 100:.0f}%")

    if report.category_scores:
        print(f"  Category scores:")
        for cat, score in report.category_scores.items():
            print(f"    {cat}: {score:.1f}%")


def demo_feedback_collection():
    """Demo: Feedback collection for RLHF."""
    section("3. Feedback Collection")

    from cognisom.flywheel.feedback import FeedbackCollector

    collector = FeedbackCollector(storage_dir="data/demo_feedback")

    # Submit various types of feedback
    print("Collecting feedback...")

    # Ratings
    collector.submit_rating("int_001", rating=5, comment="Excellent explanation")
    collector.submit_rating("int_002", rating=3, comment="Could be more specific")
    collector.submit_rating("int_003", rating=4)
    print("  - Submitted 3 ratings")

    # Thumbs
    collector.submit_thumbs("int_004", thumbs_up=True)
    collector.submit_thumbs("int_005", thumbs_up=False, comment="Inaccurate")
    print("  - Submitted 2 thumbs up/down")

    # Corrections
    collector.submit_correction(
        "int_006",
        original_response="The mitochondria is the powerhouse of the cell.",
        corrected_response="Mitochondria (plural) are organelles responsible for "
                          "cellular energy production through oxidative phosphorylation.",
        comment="Grammar and more precise"
    )
    print("  - Submitted 1 correction")

    # Preferences (for RLHF)
    collector.submit_preference(
        "int_007",
        response_a="Cancer grows fast.",
        response_b="Cancer is characterized by uncontrolled cell proliferation, "
                   "driven by accumulated genetic mutations that dysregulate "
                   "normal growth control mechanisms.",
        preferred="b",
        comment="More informative"
    )
    print("  - Submitted 1 preference")

    # Get summary
    summary = collector.get_summary()
    print()
    print(f"Feedback Summary:")
    print(f"  Total feedback: {summary.total_feedback}")
    print(f"  Average rating: {summary.average_rating:.1f}/5")
    print(f"  Thumbs up %: {summary.thumbs_up_pct:.0f}%")
    print(f"  Corrections: {summary.corrections_count}")

    # Get training data
    corrections = collector.get_corrections_for_training()
    preferences = collector.get_preferences_for_rlhf()
    print()
    print(f"Training data ready:")
    print(f"  Corrections for SFT: {len(corrections)}")
    print(f"  Preferences for RLHF: {len(preferences)}")


def demo_distillation():
    """Demo: Model distillation pipeline."""
    section("4. Model Distillation")

    from cognisom.flywheel.distillation import DistillationPipeline, DistillationConfig

    # Create config
    config = DistillationConfig(
        teacher_model="nemotron-49b",
        student_model="nemotron-8b",
        lora_rank=16,
        num_epochs=3,
    )

    print(f"Distillation Config:")
    print(f"  Teacher: {config.teacher_model}")
    print(f"  Student: {config.student_model}")
    print(f"  LoRA rank: {config.lora_rank}")
    print(f"  Temperature: {config.temperature}")

    pipeline = DistillationPipeline(
        output_dir="data/demo_distilled",
        config=config
    )

    # Prepare training data
    training_data = [
        {
            "query": "What causes prostate cancer?",
            "response": "Prostate cancer arises from accumulated genetic mutations...",
            "context": ["AR signaling", "PTEN loss", "TP53 mutations"],
        },
        {
            "query": "How does immunotherapy work?",
            "response": "Immunotherapy harnesses the immune system to fight cancer...",
            "context": ["checkpoint inhibitors", "T-cell activation"],
        },
        # ... more examples would come from agent evaluator
    ]

    print()
    print(f"Training samples: {len(training_data)}")
    print(f"Running distillation (simulated)...")

    result = pipeline.distill(training_data)

    print()
    print(f"Distillation Result:")
    print(f"  Epochs completed: {result.epochs_completed}")
    print(f"  Final loss: {result.final_loss:.4f}")
    print(f"  Cost reduction: {result.cost_reduction_pct:.1f}%")
    print(f"  Latency reduction: {result.latency_reduction_pct:.1f}%")


def demo_model_registry():
    """Demo: Model registry with A/B testing."""
    section("5. Model Registry & A/B Testing")

    from cognisom.flywheel.model_registry import ModelRegistry

    registry = ModelRegistry(registry_dir="data/demo_registry")

    # Register initial model
    print("Registering models...")
    v1 = registry.register(
        model_path="/models/nemotron-8b-base",
        base_model="nemotron-8b",
        metrics={"accuracy": 0.82, "latency_ms": 150},
        tags=["baseline"],
    )
    print(f"  v1: {v1.version_id} (baseline)")

    time.sleep(1.1)  # Ensure different timestamp

    # Register distilled model
    v2 = registry.register(
        model_path="/models/nemotron-8b-distilled",
        base_model="nemotron-8b",
        metrics={"accuracy": 0.88, "latency_ms": 145},
        training_samples=5000,
        tags=["distilled", "lora"],
    )
    print(f"  v2: {v2.version_id} (distilled)")

    # Deploy v1 to production
    print()
    print("Deploying models...")
    registry.deploy(v1.version_id, traffic_percent=100)
    print(f"  Deployed {v1.version_id} to production (100% traffic)")

    # Start A/B test with v2
    registry.deploy(v2.version_id, traffic_percent=10, ab_test=True)
    print(f"  Started A/B test: {v2.version_id} gets 10% traffic")

    # List versions
    print()
    print("Model Versions:")
    for v in registry.list_versions():
        print(f"  - {v.version_id}: {v.status.value}, "
              f"traffic={v.traffic_percent}%, "
              f"accuracy={v.metrics.get('accuracy', 'N/A')}")


def demo_complete_flow():
    """Demo: Complete flywheel flow."""
    section("6. Complete Flywheel Flow")

    from cognisom.flywheel.flywheel import DataFlywheel, FlywheelConfig, FlywheelStatus

    config = FlywheelConfig(
        data_dir="data/demo_flywheel",
        min_samples_for_distillation=10,
        quality_threshold=0.7,
        evaluation_interval_hours=1,
    )

    flywheel = DataFlywheel(config)

    print(f"Flywheel Configuration:")
    print(f"  Min samples for distillation: {config.min_samples_for_distillation}")
    print(f"  Quality threshold: {config.quality_threshold}")
    print(f"  Teacher model: {config.teacher_model}")
    print(f"  Student model: {config.student_model}")

    print()
    print(f"Flywheel Status: {flywheel.status.value}")
    print()
    print("The data flywheel continuously:")
    print("  1. Collects agent interactions")
    print("  2. Evaluates quality and identifies training candidates")
    print("  3. Periodically runs model distillation")
    print("  4. Deploys improved models with A/B testing")
    print("  5. Collects human feedback to improve scoring")


def main():
    """Run all demos."""
    print()
    print("*" * 60)
    print("  COGNISOM DATA FLYWHEEL DEMO")
    print("  Phase 5: Data Flywheel & Optimization")
    print("*" * 60)

    try:
        demo_agent_evaluation()
        demo_simulation_accuracy()
        demo_feedback_collection()
        demo_distillation()
        demo_model_registry()
        demo_complete_flow()

        section("Demo Complete!")
        print("All Phase 5 components demonstrated successfully.")
        print()
        print("Key capabilities:")
        print("  - Agent quality evaluation with auto-scoring")
        print("  - Simulation accuracy benchmarking")
        print("  - Multi-type feedback collection (rating, thumbs, corrections, preferences)")
        print("  - LoRA-based model distillation")
        print("  - Model registry with A/B testing support")
        print("  - Complete flywheel orchestration")
        print()

    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
