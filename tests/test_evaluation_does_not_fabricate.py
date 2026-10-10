"""Nothing in the evaluation path may substitute values for a result it did not obtain.

Three separate code paths used to answer "the simulation did not run" by
returning the benchmark's own observed values perturbed by a few percent. The
comparison then agreed with its own ground truth, so the benchmark passed, the
report graded the run "A", and 222 such reports accumulated in
``data/eval_results`` -- one minted on every ``pytest`` invocation.

The sites were:

  * ``eval/simulation_accuracy._run_simulation``   -- observed +/-10% when engine is None
  * ``eval/simulation_accuracy._evaluate_benchmark`` -- observed +/-15% on any exception
  * ``agents/simulation_validation_agent._run_simulation`` -- observed +/-10% on any exception

``validation/validator.py`` already handled this correctly, returning
``passed=False`` with a zero score, and is the pattern these now follow.

These tests pin the behaviour by source inspection as well as by call, because
the defect is specifically the *presence* of a plausible-value fallback: a test
that only checked the return value would pass again if someone reintroduced one
that happened to be within tolerance.
"""
from __future__ import annotations

import inspect
import re

import pytest


# The signature of the defect: a random perturbation applied to a value that
# came from the benchmark rather than from a simulation.
_PERTURBATION = re.compile(r"random\.uniform|np\.random\.uniform|rng\.uniform")


def _source_of(obj) -> str:
    return inspect.getsource(obj)


class TestSimulationAccuracyEvaluator:
    def test_no_engine_raises_rather_than_guessing(self):
        from cognisom.eval.simulation_accuracy import SimulationEvaluator

        ev = SimulationEvaluator()
        benchmark = {"name": "b", "data_points": [(0.0, 1.0), (1.0, 2.0)]}

        with pytest.raises(ValueError, match="no simulation engine"):
            ev._run_simulation(benchmark, engine=None)

    def test_a_failed_benchmark_reports_no_values_and_does_not_pass(self):
        from cognisom.eval.simulation_accuracy import SimulationEvaluator

        ev = SimulationEvaluator()
        comp = ev._evaluate_benchmark(
            {"name": "b", "category": "c", "data_points": [(0.0, 10.0), (1.0, 20.0)]},
            engine=None,
            tolerance=20.0,
        )

        assert comp.passed is False
        assert comp.simulated_values == []
        assert comp.error
        # The observed values are still reported -- they are real data.
        assert comp.observed_values == [10.0, 20.0]

    def test_a_benchmark_with_no_data_is_skipped_not_passed(self):
        from cognisom.eval.simulation_accuracy import SimulationEvaluator

        ev = SimulationEvaluator()
        comp = ev._evaluate_benchmark(
            {"name": "empty", "category": "c", "data_points": []},
            engine=None,
            tolerance=20.0,
        )

        assert comp.skipped is True
        assert comp.passed is False

    def test_skipped_benchmarks_move_neither_side_of_the_pass_rate(self):
        from cognisom.eval.simulation_accuracy import SimulationEvaluator

        ev = SimulationEvaluator()
        ev._benchmarks = [
            {"name": "empty", "category": "c", "data_points": []},
            {"name": "real", "category": "c", "data_points": [(0.0, 1.0)]},
        ]

        report = ev.evaluate_against_benchmarks(engine=None)

        assert report.benchmarks_skipped == 1
        assert report.benchmarks_evaluated == 1
        assert report.benchmarks_passed == 0
        assert report.pass_rate == 0.0

    def test_an_unsimulated_run_grades_f(self):
        from cognisom.eval.simulation_accuracy import SimulationEvaluator

        report = SimulationEvaluator().evaluate_against_benchmarks(engine=None)

        assert report.overall_grade == "F"
        assert report.benchmarks_passed == 0

    def test_no_perturbation_fallback_in_source(self):
        from cognisom.eval.simulation_accuracy import SimulationEvaluator

        for fn in (
            SimulationEvaluator._run_simulation,
            SimulationEvaluator._evaluate_benchmark,
        ):
            assert not _PERTURBATION.search(_source_of(fn)), (
                f"{fn.__qualname__} perturbs a value to stand in for a "
                f"simulation result; it must report failure instead"
            )


class TestSimulationValidationAgent:
    def test_a_failed_simulation_raises_rather_than_approximating(self):
        sva = pytest.importorskip("cognisom.agents.simulation_validation_agent")

        agent = sva.SimulationValidationAgent()

        class Benchmark:
            name = "b"
            category = "c"
            source = ""
            tolerance_pct = 20.0
            data_points = [(0.0, 10.0), (1.0, 20.0)]
            # No sim_config/module, so _run_simulation cannot build an engine.

        with pytest.raises(Exception):
            agent._run_simulation(Benchmark())

    def test_the_caller_turns_that_into_a_failed_result(self):
        sva = pytest.importorskip("cognisom.agents.simulation_validation_agent")

        agent = sva.SimulationValidationAgent()

        class Benchmark:
            name = "b"
            category = "c"
            source = ""
            tolerance_pct = 20.0
            data_points = [(0.0, 10.0), (1.0, 20.0)]

        result = agent._run_benchmark(Benchmark())

        assert result.passed is False
        assert "failed" in result.details.lower()

    def test_an_empty_benchmark_is_skipped_not_passed(self):
        sva = pytest.importorskip("cognisom.agents.simulation_validation_agent")

        agent = sva.SimulationValidationAgent()

        class Benchmark:
            name = "empty"
            category = "c"
            source = ""
            tolerance_pct = 20.0
            data_points = []

        result = agent._run_benchmark(Benchmark())

        assert result.skipped is True
        assert result.passed is False

    def test_no_perturbation_fallback_in_source(self):
        sva = pytest.importorskip("cognisom.agents.simulation_validation_agent")

        src = _source_of(sva.SimulationValidationAgent._run_simulation)
        assert not _PERTURBATION.search(src), (
            "_run_simulation perturbs the expected values to stand in for a "
            "simulation result; it must report failure instead"
        )
