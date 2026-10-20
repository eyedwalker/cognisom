"""A peptide-allele pair is scored by the real predictor once.

MHCflurry is a neural network whose per-invocation overhead dwarfs the
marginal cost of another peptide, so how often it is called decides whether
a run takes seconds or minutes. Two holes made the simulation call it once
per pair per cell per step:

- ``NeoantigenPredictor._predict_binding`` read ``_binding_cache`` and never
  filled it on the MHCflurry path, so every repeat of a pair paid another
  invocation -- and in a simulation the same mutation's peptides are
  re-scored on every cell on every step;
- ``MHCLoader.score_all``, which is what the cellular module runs on every
  mutation, asked pair by pair. The neoantigen path had already learned to
  batch (``_warm_binding_cache``); this one had not.

Together they blew the 120 s timeout on the nine-module fixture in CI,
where the real predictor runs. These tests run against whichever scorer is
installed -- they count calls rather than asserting affinities, so they are
meaningful under the PWM fallback too.
"""
from __future__ import annotations

import pytest

from cognisom.engine.py.immune.mhc_loading import MHCLoader
from cognisom.engine.py.molecular.peptidome import Peptide
from cognisom.genomics.neoantigen_predictor import NeoantigenPredictor

ALLELES = ["HLA-A*02:01", "HLA-A*24:02", "HLA-B*07:02"]
SEQUENCES = ["LVVVGADGVG", "GLATEKSRWS", "KLVVVGADGV"]


def _peptides():
    return [Peptide(sequence=s, source_gene="KRAS", length=len(s), is_mutant=True,
                    wild_type_sequence=s[:-1] + "A", mutation_label="G12D",
                    anchor_position_in_peptide=4, parent_position_1based=10,
                    cleavage_score=0.8)
            for s in SEQUENCES]


def test_a_pair_is_predicted_once_however_often_it_is_asked_for(monkeypatch):
    predictor = NeoantigenPredictor()
    calls = []

    def counted(peptide, allele):
        calls.append((peptide, allele))
        raise RuntimeError("no scorer here; the count is the subject")

    import cognisom.genomics.mhcflurry_binding as mb
    monkeypatch.setattr(mb, "is_mhcflurry_available", lambda: True)
    monkeypatch.setattr(mb, "predict_binding", counted)
    # Every call fails and falls through to the PWM estimate, so this test
    # measures how often the real predictor is reached, not what it says.
    first = predictor._predict_binding(SEQUENCES[0], ALLELES[0])
    assert len(calls) == 1

    # A failure is not cached: the next ask tries the real scorer again.
    predictor._predict_binding(SEQUENCES[0], ALLELES[0])
    assert len(calls) == 2

    # A success is cached, and the cached value is what comes back.
    class _Result:
        affinity_nm, method, percentile_rank, presentation_score = 123.0, "mhcflurry", 0.4, 0.9

    monkeypatch.setattr(mb, "predict_binding", lambda p, a: _Result())
    assert predictor._predict_binding(SEQUENCES[0], ALLELES[0]) == 123.0
    monkeypatch.setattr(mb, "predict_binding", counted)
    for _ in range(5):
        assert predictor._predict_binding(SEQUENCES[0], ALLELES[0]) == 123.0
    assert len(calls) == 2, "a cached pair went back to the predictor"
    assert isinstance(first, float)


def test_the_cache_keeps_the_trained_scorers_extra_outputs(monkeypatch):
    """Percentile and presentation score are kept only when they are real."""
    import cognisom.genomics.mhcflurry_binding as mb
    monkeypatch.setattr(mb, "is_mhcflurry_available", lambda: True)

    class _Result:
        affinity_nm, percentile_rank, presentation_score = 50.0, 0.3, 0.8
        method = "mhcflurry"

    class _Approx(_Result):
        method = "pwm"

    for result, keeps in ((_Result(), True), (_Approx(), False)):
        predictor = NeoantigenPredictor()
        monkeypatch.setattr(mb, "predict_binding", lambda p, a, r=result: r)
        predictor._predict_binding(SEQUENCES[0], ALLELES[0])
        cached = predictor._binding_cache[(SEQUENCES[0], ALLELES[0])]
        assert cached.affinity_nm == 50.0
        assert (cached.percentile_rank is not None) is keeps
        assert (cached.presentation_score is not None) is keeps


def test_scoring_a_batch_of_peptides_warms_every_pair_in_one_call(monkeypatch):
    """What the cellular module runs on every mutation: one batched call for
    all 9 pairs, not 9 separate ones."""
    loader = MHCLoader()
    batches, singles = [], []

    class _Result:
        affinity_nm, method, percentile_rank, presentation_score = 200.0, "mhcflurry", 0.5, 0.5

    import cognisom.genomics.mhcflurry_binding as mb
    monkeypatch.setattr(mb, "is_mhcflurry_available", lambda: True)

    def batch(peptides, alleles):
        batches.append(len(peptides))
        return [_Result() for _ in peptides]

    monkeypatch.setattr(mb, "predict_binding_batch", batch)
    monkeypatch.setattr(mb, "predict_binding",
                        lambda p, a: singles.append((p, a)) or _Result())

    presentations = loader.score_all(_peptides(), ALLELES)

    assert batches == [len(SEQUENCES) * len(ALLELES)], batches
    assert singles == [], "a pair was scored singly after the batch warmed it"
    assert len(presentations) == len(SEQUENCES) * len(ALLELES)
    assert {p.ic50_nm for p in presentations} == {200.0}


def test_a_second_mutation_only_scores_what_is_new(monkeypatch):
    loader = MHCLoader()
    asked = []

    class _Result:
        affinity_nm, method, percentile_rank, presentation_score = 200.0, "mhcflurry", 0.5, 0.5

    import cognisom.genomics.mhcflurry_binding as mb
    monkeypatch.setattr(mb, "is_mhcflurry_available", lambda: True)
    monkeypatch.setattr(mb, "predict_binding_batch",
                        lambda peps, als: asked.extend(zip(peps, als))
                        or [_Result() for _ in peps])
    monkeypatch.setattr(mb, "predict_binding", lambda p, a: _Result())

    loader.score_all(_peptides(), ALLELES)
    assert len(asked) == 9
    loader.score_all(_peptides(), ALLELES)          # the same mutation again
    assert len(asked) == 9, "re-scored peptides it had already seen"


def test_a_mutation_that_presents_nothing_says_so_once(caplog, monkeypatch):
    """An HLA-mismatched mutation presenting nothing is a real outcome, but
    silence makes it indistinguishable from a broken scorer. CI reads this
    line: under the real predictor the closed-loop tests present nothing,
    and the number and the scorer's name are what says why."""
    import contextlib
    import io
    import logging

    import numpy as np

    from cognisom.modules import cellular_module as cm
    from cognisom.modules.cellular_module import CellularModule
    from cognisom.modules.molecular_module import MolecularModule
    from cognisom.core import SimulationConfig, SimulationEngine

    monkeypatch.setenv("COGNISOM_ALLOW_PWM_FALLBACK", "1")
    cm._NO_BINDER_REPORTED.clear()
    np.random.seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        engine = SimulationEngine(SimulationConfig(dt=0.01, duration=0.03, use_gpu=False))
        engine.register_module("molecular", MolecularModule, {
            "transcription_rate": 0.0, "exosome_release_rate": 0.0, "mutation_rate": 0.0})
        engine.register_module("cellular", CellularModule, {
            "n_normal_cells": 0, "n_cancer_cells": 0,
            "hla_alleles": ["HLA-B*58:01"]})       # no KRAS G12D peptide binds this
        engine.initialize()
        mol, cel = engine.modules["molecular"], engine.modules["cellular"]
        cel.set_molecular_module(mol)
        cell_id = cel.add_cell(position=[100, 100, 50], cell_type="cancer")
        mol.add_cell(cell_id)
        with caplog.at_level(logging.INFO, logger="cognisom.modules.cellular_module"):
            mol.introduce_mutation(cell_id, "KRAS", "G12D")
            engine.run(duration=0.03)

    assert cel.total_peptides_presented == 0
    lines = [r.getMessage() for r in caplog.records if "none binding" in r.getMessage()]
    assert len(lines) == 1, lines          # once per (gene, mutation), not per step
    assert "KRAS G12D" in lines[0] and "HLA-B*58:01" in lines[0]
    assert "500 nM" in lines[0] and "Scorer:" in lines[0]
    cm._NO_BINDER_REPORTED.clear()
