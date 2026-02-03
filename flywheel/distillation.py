"""
Model Distillation Pipeline (Phase 5)
=====================================

Distill knowledge from large teacher models into smaller student models.
Uses LoRA fine-tuning for efficient adaptation.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class DistillationConfig:
    """Configuration for model distillation."""
    teacher_model: str = "nemotron-49b"
    student_model: str = "nemotron-8b"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    max_seq_length: int = 2048
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    temperature: float = 2.0          # Distillation temperature
    alpha_ce: float = 0.5             # Cross-entropy weight
    alpha_kl: float = 0.5             # KL divergence weight


@dataclass
class DistillationResult:
    """Result of a distillation run."""
    model_path: str = ""
    training_samples: int = 0
    epochs_completed: int = 0
    final_loss: float = 0.0
    eval_loss: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    quality_improved: bool = False
    cost_reduction_pct: float = 0.0
    latency_reduction_pct: float = 0.0
    elapsed_sec: float = 0.0
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    def summary(self) -> str:
        return (
            f"Distillation Result\n"
            f"==================\n"
            f"Model: {self.model_path}\n"
            f"Training samples: {self.training_samples}\n"
            f"Epochs: {self.epochs_completed}\n"
            f"Final loss: {self.final_loss:.4f}\n"
            f"Eval loss: {self.eval_loss:.4f}\n"
            f"Quality improved: {self.quality_improved}\n"
            f"Cost reduction: {self.cost_reduction_pct:.1f}%\n"
            f"Latency reduction: {self.latency_reduction_pct:.1f}%\n"
            f"Elapsed: {self.elapsed_sec:.1f}s"
        )


class DistillationPipeline:
    """Pipeline for distilling large models into smaller ones.

    Uses LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
    and knowledge distillation from teacher model outputs.
    """

    # Model size estimates (billions of parameters)
    MODEL_SIZES = {
        "nemotron-49b": 49.0,
        "nemotron-8b": 8.0,
        "llama-3-70b": 70.0,
        "llama-3-8b": 8.0,
        "mistral-7b": 7.0,
    }

    # Inference cost estimates (relative)
    MODEL_COSTS = {
        "nemotron-49b": 1.0,
        "nemotron-8b": 0.16,
        "llama-3-70b": 1.2,
        "llama-3-8b": 0.16,
        "mistral-7b": 0.14,
    }

    def __init__(
        self,
        output_dir: str = "data/distilled_models",
        teacher_model: str = "nemotron-49b",
        student_model: str = "nemotron-8b",
        config: Optional[DistillationConfig] = None
    ) -> None:
        """Initialize the distillation pipeline.

        Args:
            output_dir: Directory for output models
            teacher_model: Teacher model name
            student_model: Student model name
            config: Distillation configuration
        """
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._teacher_model = teacher_model
        self._student_model = student_model
        self._config = config or DistillationConfig(
            teacher_model=teacher_model,
            student_model=student_model
        )

        self._training_history: List[DistillationResult] = []

    def distill(
        self,
        training_data: List[Dict],
        validation_data: Optional[List[Dict]] = None,
        lora_rank: Optional[int] = None
    ) -> DistillationResult:
        """Run knowledge distillation.

        Args:
            training_data: Training examples
                [{"query": ..., "response": ..., "context": ...}, ...]
            validation_data: Optional validation set
            lora_rank: Override LoRA rank

        Returns:
            Distillation result
        """
        t0 = time.time()
        result = DistillationResult(
            training_samples=len(training_data)
        )

        if lora_rank:
            self._config.lora_rank = lora_rank

        log.info(
            "Starting distillation: %s -> %s (%d samples)",
            self._teacher_model,
            self._student_model,
            len(training_data)
        )

        try:
            # Prepare training data
            prepared_data = self._prepare_data(training_data)

            # Split validation if not provided
            if validation_data is None:
                split_idx = int(len(prepared_data) * 0.9)
                train_data = prepared_data[:split_idx]
                val_data = prepared_data[split_idx:]
            else:
                train_data = prepared_data
                val_data = self._prepare_data(validation_data)

            # Run training
            train_result = self._train_lora(train_data, val_data)

            result.epochs_completed = self._config.num_epochs
            result.final_loss = train_result.get("train_loss", 0.0)
            result.eval_loss = train_result.get("eval_loss", 0.0)
            result.model_path = train_result.get("model_path", "")
            result.metrics = train_result.get("metrics", {})

            # Evaluate improvement
            result.quality_improved = self._evaluate_improvement(result)

            # Calculate cost reduction
            teacher_cost = self.MODEL_COSTS.get(self._teacher_model, 1.0)
            student_cost = self.MODEL_COSTS.get(self._student_model, 0.5)
            result.cost_reduction_pct = (1 - student_cost / teacher_cost) * 100

            # Estimate latency reduction (proportional to model size)
            teacher_size = self.MODEL_SIZES.get(self._teacher_model, 50)
            student_size = self.MODEL_SIZES.get(self._student_model, 8)
            result.latency_reduction_pct = (1 - student_size / teacher_size) * 100

        except Exception as e:
            log.error("Distillation failed: %s", e)
            result.metrics["error"] = str(e)

        result.elapsed_sec = time.time() - t0

        self._training_history.append(result)
        self._save_result(result)

        return result

    def _prepare_data(self, data: List[Dict]) -> List[Dict]:
        """Prepare data for training."""
        prepared = []

        for item in data:
            query = item.get("query", "")
            response = item.get("response", "")
            context = item.get("context", [])

            # Format as conversation
            if context:
                context_text = "\n\nContext:\n" + "\n".join(context[:3])
            else:
                context_text = ""

            prepared.append({
                "instruction": query + context_text,
                "output": response,
                "input": "",  # For compatibility with instruction tuning format
            })

        return prepared

    def _train_lora(
        self,
        train_data: List[Dict],
        val_data: List[Dict]
    ) -> Dict[str, Any]:
        """Train LoRA adapter.

        In production, this would use:
        - NeMo Customizer for NVIDIA models
        - PEFT/transformers for open models
        - Axolotl for advanced configurations

        This is a simulation for the framework structure.
        """
        log.info("Training LoRA adapter (r=%d)...", self._config.lora_rank)

        # Create output directory
        timestamp = int(time.time())
        model_dir = self._output_dir / f"{self._student_model}_lora_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save training config
        config_path = model_dir / "config.json"
        config_data = {
            "base_model": self._student_model,
            "lora_rank": self._config.lora_rank,
            "lora_alpha": self._config.lora_alpha,
            "learning_rate": self._config.learning_rate,
            "epochs": self._config.num_epochs,
            "training_samples": len(train_data),
            "validation_samples": len(val_data),
        }
        config_path.write_text(json.dumps(config_data, indent=2))

        # Save training data for reproducibility
        train_path = model_dir / "train_data.jsonl"
        with open(train_path, "w") as f:
            for item in train_data:
                f.write(json.dumps(item) + "\n")

        # Simulate training (in production, call actual training API)
        # This would be replaced with NeMo Customizer or PEFT
        simulated_metrics = self._simulate_training(len(train_data))

        # Save adapter config (mock)
        adapter_config = {
            "peft_type": "LORA",
            "r": self._config.lora_rank,
            "lora_alpha": self._config.lora_alpha,
            "lora_dropout": self._config.lora_dropout,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "bias": "none",
        }
        (model_dir / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2))

        return {
            "model_path": str(model_dir),
            "train_loss": simulated_metrics["train_loss"],
            "eval_loss": simulated_metrics["eval_loss"],
            "metrics": simulated_metrics,
        }

    def _simulate_training(self, n_samples: int) -> Dict[str, float]:
        """Simulate training metrics (placeholder for actual training)."""
        import random

        # Simulate loss decreasing with more samples
        base_loss = 2.0
        sample_factor = min(1.0, n_samples / 1000)
        noise = random.uniform(-0.1, 0.1)

        train_loss = base_loss * (1 - sample_factor * 0.6) + noise
        eval_loss = train_loss * (1 + random.uniform(0.05, 0.15))

        return {
            "train_loss": max(0.1, train_loss),
            "eval_loss": max(0.15, eval_loss),
            "perplexity": 2 ** eval_loss,
            "accuracy": 0.7 + sample_factor * 0.2 + random.uniform(-0.05, 0.05),
        }

    def _evaluate_improvement(self, result: DistillationResult) -> bool:
        """Evaluate if the distilled model improved quality."""
        # Check if loss is reasonable
        if result.eval_loss > 3.0:
            return False

        # Check if accuracy is acceptable
        accuracy = result.metrics.get("accuracy", 0)
        if accuracy < 0.7:
            return False

        # Compare to previous best
        if self._training_history:
            prev_best_loss = min(r.eval_loss for r in self._training_history if r.eval_loss > 0)
            if result.eval_loss > prev_best_loss * 1.1:
                return False

        return True

    def _save_result(self, result: DistillationResult) -> None:
        """Save distillation result."""
        results_file = self._output_dir / "distillation_history.jsonl"
        with open(results_file, "a") as f:
            data = {
                "timestamp": result.timestamp,
                "model_path": result.model_path,
                "training_samples": result.training_samples,
                "epochs": result.epochs_completed,
                "train_loss": result.final_loss,
                "eval_loss": result.eval_loss,
                "quality_improved": result.quality_improved,
                "cost_reduction_pct": result.cost_reduction_pct,
                "metrics": result.metrics,
            }
            f.write(json.dumps(data) + "\n")

    def get_history(self) -> List[DistillationResult]:
        """Get distillation history."""
        return self._training_history

    def estimate_cost_savings(
        self,
        monthly_queries: int,
        teacher_cost_per_1k: float = 0.03,
        student_cost_per_1k: float = 0.005
    ) -> Dict[str, float]:
        """Estimate cost savings from distillation.

        Args:
            monthly_queries: Expected monthly query volume
            teacher_cost_per_1k: Cost per 1000 queries for teacher
            student_cost_per_1k: Cost per 1000 queries for student

        Returns:
            Cost analysis dictionary
        """
        queries_in_thousands = monthly_queries / 1000

        teacher_monthly = queries_in_thousands * teacher_cost_per_1k
        student_monthly = queries_in_thousands * student_cost_per_1k

        return {
            "teacher_monthly_cost": teacher_monthly,
            "student_monthly_cost": student_monthly,
            "monthly_savings": teacher_monthly - student_monthly,
            "annual_savings": (teacher_monthly - student_monthly) * 12,
            "savings_percentage": (1 - student_monthly / teacher_monthly) * 100,
        }
