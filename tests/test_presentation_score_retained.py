"""The scorer's calibrated percentile and presentation score must survive.

MHCflurry returns three numbers per peptide-allele pair: an IC50, a calibrated
percentile rank, and a presentation score that folds in antigen processing.
The cache used to keep only the IC50, so ranking ran on raw affinity and the
reported percentile was a four-valued step function (0.5 / 2.0 / 10.0 / 50.0)
derived from that affinity. These tests pin that the real values are kept when
the trained scorer produces them, and that the approximate path does not
pretend to produce them.
"""
import pytest

from cognisom.genomics.neoantigen_predictor import (
    NeoantigenPredictor,
    _CachedBinding,
)
from cognisom.genomics import mhcflurry_binding as mb


STEP_VALUES = {0.5, 2.0, 10.0, 50.0}


def _result(peptide, allele, affinity, percentile, presentation, method):
    r = mb.BindingResult(
        peptide=peptide,
        hla_allele=allele,
        affinity_nm=affinity,
        percentile_rank=percentile,
        presentation_score=presentation,
        is_strong_binder=affinity < 50,
        is_weak_binder=affinity < 500,
    )
    # `method` is set by the scorer; mirror that here.
    object.__setattr__(r, "method", method) if hasattr(r, "__dataclass_fields__") \
        else setattr(r, "method", method)
    return r


class TestTrainedScorerValuesAreKept:
    def test_calibrated_percentile_is_used_not_the_step_function(self, monkeypatch):
        # 0.37 is deliberately not one of the step function's four outputs.
        monkeypatch.setattr(mb, "is_mhcflurry_available", lambda: True)
        monkeypatch.setattr(
            mb, "predict_binding_batch",
            lambda peps, alls: [
                _result(p, a, 123.0, 0.37, 0.88, "mhcflurry")
                for p, a in zip(peps, alls)
            ],
        )
        pred = NeoantigenPredictor()
        pred._warm_binding_cache(["SIINFEKL"], ["HLA-A*02:01"])

        cached = pred._binding_cache[("SIINFEKL", "HLA-A*02:01")]
        assert cached.percentile_rank == pytest.approx(0.37)
        assert cached.presentation_score == pytest.approx(0.88)
        assert cached.percentile_rank not in STEP_VALUES
        # The affinity still resolves through the usual accessor.
        assert pred._predict_binding("SIINFEKL", "HLA-A*02:01") == pytest.approx(123.0)

    def test_presentation_score_is_not_derivable_from_affinity(self, monkeypatch):
        # Two pairs with the SAME affinity but different presentation scores
        # must stay distinguishable — impossible if only IC50 were kept.
        monkeypatch.setattr(mb, "is_mhcflurry_available", lambda: True)
        scores = iter([0.20, 0.95])
        monkeypatch.setattr(
            mb, "predict_binding_batch",
            lambda peps, alls: [
                _result(p, a, 300.0, 1.5, next(scores), "mhcflurry")
                for p, a in zip(peps, alls)
            ],
        )
        pred = NeoantigenPredictor()
        pred._warm_binding_cache(["SIINFEKLA"], ["HLA-A*02:01", "HLA-B*07:02"])
        got = {k[1]: v.presentation_score for k, v in pred._binding_cache.items()}
        assert set(got.values()) == {0.20, 0.95}


class TestApproximateScorerDoesNotClaimCalibration:
    def test_pwm_results_carry_no_percentile_or_presentation(self, monkeypatch):
        monkeypatch.setattr(mb, "is_mhcflurry_available", lambda: True)
        monkeypatch.setattr(
            mb, "predict_binding_batch",
            lambda peps, alls: [
                # A PWM result still populates the fields structurally, but
                # the method says what produced them.
                _result(p, a, 52.8, 2.0, 0.5, "pwm")
                for p, a in zip(peps, alls)
            ],
        )
        pred = NeoantigenPredictor()
        pred._warm_binding_cache(["SIINFEKL"], ["HLA-A*02:01"])
        cached = pred._binding_cache[("SIINFEKL", "HLA-A*02:01")]
        assert cached.method == "pwm"
        assert cached.percentile_rank is None, (
            "an approximate score must not be reported as a calibrated percentile"
        )
        assert cached.presentation_score is None
        # The affinity is still usable.
        assert cached.affinity_nm == pytest.approx(52.8)


class TestCacheRecord:
    def test_defaults_are_none_not_placeholders(self):
        c = _CachedBinding(affinity_nm=100.0)
        assert c.percentile_rank is None
        assert c.presentation_score is None
