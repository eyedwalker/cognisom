"""
Upgrade 3 Stage C unit tests: ESM-2 zero-shot stability composition.

Covers engine/py/molecular/esm_stability.py + the Stage C path in
MutationEffectClassifier. Real ESM-2 inference is intentionally NOT
exercised in the default test run -- it requires a ~600MB model
download and takes seconds per inference, which would balloon CI time.
The mock-based tests verify the integration logic; an opt-in smoke
test verifies that RealESMStabilityScorer works end-to-end against
the published model when ENABLE_ESM_SMOKE=1 is set.

What the tests cover:
  * Score mapping: delta_log_likelihood -> stability_modifier
    (sigmoid, monotone, neutral at 0, [0,1] range)
  * StubESMStabilityScorer behavior (constant score, input validation)
  * apply_stability_to_impact composition (monotone, ceiling clamp,
    neutral fixed-point)
  * Classifier integration: Stage B path is preserved when esm_scorer
    is None
  * Classifier integration: destabilizing ESM score pulls impact UP
  * Classifier integration: well-tolerated ESM score pulls impact DOWN
  * Classifier integration: neutral ESM score (dLL=0) doesn't change
    Stage B impact
  * Classifier integration: scorer exception falls back to Stage B
    impact (graceful, no crash)
  * MutationEffect carries esm_delta_log_likelihood / modifier /
    model_name for downstream audit
  * RealESMStabilityScorer raises ImportError gracefully when
    transformers/torch are unavailable (skipped if they are)
  * Opt-in: RealESMStabilityScorer end-to-end smoke
    (ENABLE_ESM_SMOKE=1)
"""
from __future__ import annotations

import math
import os

import pytest

from engine.py.molecular.esm_stability import (
    DEFAULT_ESM_MODEL,
    ESMStabilityResult,
    RealESMStabilityScorer,
    StubESMStabilityScorer,
    apply_stability_to_impact,
    delta_ll_to_stability_modifier,
)
from engine.py.molecular.mutation_effect import MutationEffectClassifier
from engine.py.molecular.reference_cds import KRAS_CDS


# Curated KRAS WT protein for the classifier integration tests
# (translated from NM_004985.5 codons 1-188). Hard-coded so the
# tests do not depend on the molecular_module translation path.
KRAS_PROTEIN: str = (
    "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMR"
    "DQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPSRTVDTKQAQDLARS"
    "YGIPFIETSAKTRQGVDDAFYTLVREIRKHKEKMSKDGKKKKKKSKTKCVIM"
)
KRAS_G12_BASE_POS = 34  # codon 12 middle base for G12D


# ---------------------------------------------------------------------------
# Score mapping
# ---------------------------------------------------------------------------

def test_modifier_neutral_at_zero_delta_ll():
    assert delta_ll_to_stability_modifier(0.0) == pytest.approx(0.5, abs=1e-9)


def test_modifier_destabilizing_at_negative_delta_ll():
    assert delta_ll_to_stability_modifier(-3.0) > 0.9


def test_modifier_well_tolerated_at_positive_delta_ll():
    assert delta_ll_to_stability_modifier(+3.0) < 0.1


def test_modifier_monotone():
    prev = delta_ll_to_stability_modifier(-10.0)
    for d in (-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0):
        cur = delta_ll_to_stability_modifier(d)
        assert cur < prev, f"non-monotone at dLL={d}: {cur} vs prev {prev}"
        prev = cur


def test_modifier_bounded():
    for d in (-1e6, -1e3, 0.0, 1e3, 1e6):
        m = delta_ll_to_stability_modifier(d)
        assert 0.0 <= m <= 1.0


# ---------------------------------------------------------------------------
# StubESMStabilityScorer
# ---------------------------------------------------------------------------

def test_stub_returns_constant_score():
    scorer = StubESMStabilityScorer(delta_log_likelihood=-2.5)
    r = scorer.score_substitution("MAGT", 2, "A", "V")
    assert r.delta_log_likelihood == -2.5
    assert r.stability_modifier == pytest.approx(
        delta_ll_to_stability_modifier(-2.5)
    )
    assert r.model_name == "stub:constant"


def test_stub_rejects_empty_sequence():
    with pytest.raises(ValueError):
        StubESMStabilityScorer().score_substitution("", 1, "A", "V")


def test_stub_rejects_out_of_range_position():
    with pytest.raises(IndexError):
        StubESMStabilityScorer().score_substitution("MAGT", 99, "A", "V")


# ---------------------------------------------------------------------------
# Composition (Stage B + C)
# ---------------------------------------------------------------------------

def test_apply_stability_neutral_is_identity():
    """At modifier = 0.5, impact should not change."""
    for base in (0.0, 0.2, 0.5, 0.85):
        assert apply_stability_to_impact(base, 0.5) == pytest.approx(base)


def test_apply_stability_destabilizing_pushes_up():
    for base in (0.0, 0.3, 0.6):
        for mod in (0.55, 0.7, 0.9):
            out = apply_stability_to_impact(base, mod)
            assert out >= base, (
                f"destabilizing modifier {mod} did not increase impact "
                f"from base {base}: got {out}"
            )


def test_apply_stability_well_tolerated_pulls_down():
    for base in (0.2, 0.5, 0.85):
        for mod in (0.45, 0.2, 0.05):
            out = apply_stability_to_impact(base, mod)
            assert out <= base, (
                f"well-tolerated modifier {mod} did not decrease impact "
                f"from base {base}: got {out}"
            )


def test_apply_stability_clamps_to_ceiling():
    out = apply_stability_to_impact(0.85, 1.0)
    assert out == pytest.approx(0.85, abs=1e-9)


def test_apply_stability_clamps_to_zero():
    out = apply_stability_to_impact(0.85, 0.0)
    assert out == pytest.approx(0.0, abs=1e-9)


def test_apply_stability_rejects_out_of_range_modifier():
    with pytest.raises(ValueError):
        apply_stability_to_impact(0.5, 1.5)
    with pytest.raises(ValueError):
        apply_stability_to_impact(0.5, -0.1)


# ---------------------------------------------------------------------------
# Classifier integration
# ---------------------------------------------------------------------------

@pytest.fixture
def clf():
    return MutationEffectClassifier()


def test_stage_b_path_unchanged_when_esm_scorer_none(clf):
    """Passing no esm_scorer must produce the same impact as Stage B
    alone -- this is the backwards-compat contract."""
    a = clf.classify_substitution(KRAS_CDS, KRAS_G12_BASE_POS, "A", gene_name="KRAS")
    b = clf.classify_substitution(
        KRAS_CDS, KRAS_G12_BASE_POS, "A",
        gene_name="KRAS",
        protein_sequence=KRAS_PROTEIN,
        esm_scorer=None,
    )
    assert a.impact_score == b.impact_score
    assert b.esm_delta_log_likelihood is None
    assert b.esm_stability_modifier is None
    assert b.esm_model_name is None


def test_neutral_esm_score_does_not_change_impact(clf):
    """A dLL = 0 ESM score collapses the Stage C contribution; the
    final impact must match Stage B."""
    stage_b = clf.classify_substitution(
        KRAS_CDS, KRAS_G12_BASE_POS, "A", gene_name="KRAS",
    )
    stage_c = clf.classify_substitution(
        KRAS_CDS, KRAS_G12_BASE_POS, "A",
        gene_name="KRAS",
        protein_sequence=KRAS_PROTEIN,
        esm_scorer=StubESMStabilityScorer(0.0),
    )
    assert stage_c.impact_score == pytest.approx(stage_b.impact_score)
    assert stage_c.esm_delta_log_likelihood == 0.0
    assert stage_c.esm_stability_modifier == pytest.approx(0.5)
    assert stage_c.esm_model_name == "stub:constant"


def test_destabilizing_esm_score_does_not_reduce_critical_impact(clf):
    """For a critical-region missense already at the missense ceiling
    (KRAS G12D in the P-loop, Stage B = 0.85), a destabilizing ESM
    score must keep the impact at the ceiling -- ESM agrees with
    BLOSUM+domain."""
    e = clf.classify_substitution(
        KRAS_CDS, KRAS_G12_BASE_POS, "A",
        gene_name="KRAS",
        protein_sequence=KRAS_PROTEIN,
        esm_scorer=StubESMStabilityScorer(-3.0),
    )
    assert e.impact_score == pytest.approx(0.85, abs=1e-9)
    assert e.esm_delta_log_likelihood == -3.0
    assert e.esm_stability_modifier > 0.9


def test_well_tolerated_esm_score_pulls_critical_impact_down(clf):
    """A strongly-positive ESM score should override Stage B's
    P-loop classification of KRAS G12D and reduce the impact -- this
    is exactly the patent-evidence point of Stage C, that biophysics
    can correct rule-based-only overconfidence."""
    e = clf.classify_substitution(
        KRAS_CDS, KRAS_G12_BASE_POS, "A",
        gene_name="KRAS",
        protein_sequence=KRAS_PROTEIN,
        esm_scorer=StubESMStabilityScorer(+3.0),
    )
    assert e.impact_score < 0.5, (
        f"well-tolerated ESM score should reduce impact below 0.5; "
        f"got {e.impact_score}"
    )
    assert e.esm_delta_log_likelihood == +3.0
    assert e.esm_stability_modifier < 0.1


def test_destabilizing_esm_lifts_linker_position(clf):
    """At a position that has no domain annotation, Stage B leaves
    the base BLOSUM impact alone. Stage C with a destabilizing dLL
    should still pull the impact toward the ceiling -- this is the
    patent-evidence point that ESM rescues mutations the rule-based
    classifier underestimates."""
    # KRAS codon 20 is in a linker (no domain). Find a missense base.
    codon_start = (20 - 1) * 3
    # Try mutations until we find a missense.
    for offset in range(3):
        for new_base in "ACGT":
            pos = codon_start + offset
            if new_base == KRAS_CDS[pos]:
                continue
            base = clf.classify_substitution(KRAS_CDS, pos, new_base)
            if base.category != "missense":
                continue
            stage_b_only = clf.classify_substitution(
                KRAS_CDS, pos, new_base, gene_name="KRAS",
            )
            # Linker -> Stage B = Stage A
            assert stage_b_only.impact_score == base.impact_score
            stage_c = clf.classify_substitution(
                KRAS_CDS, pos, new_base,
                gene_name="KRAS",
                protein_sequence=KRAS_PROTEIN,
                esm_scorer=StubESMStabilityScorer(-4.0),
            )
            # Stage C lifts the impact above Stage B's pass-through
            assert stage_c.impact_score > stage_b_only.impact_score
            return
    pytest.skip("could not engineer a linker missense for the test")


def test_classifier_recovers_from_scorer_exception(clf):
    """If the scorer throws, the classifier must fall back to Stage B
    impact and emit no ESM fields -- never crash."""

    class _BoomScorer:
        def score_substitution(self, **kwargs):
            raise RuntimeError("simulated ESM failure")

    e = clf.classify_substitution(
        KRAS_CDS, KRAS_G12_BASE_POS, "A",
        gene_name="KRAS",
        protein_sequence=KRAS_PROTEIN,
        esm_scorer=_BoomScorer(),
    )
    stage_b = clf.classify_substitution(
        KRAS_CDS, KRAS_G12_BASE_POS, "A", gene_name="KRAS",
    )
    assert e.impact_score == pytest.approx(stage_b.impact_score)
    assert e.esm_delta_log_likelihood is None
    assert e.esm_stability_modifier is None
    assert e.esm_model_name is None


def test_synonymous_path_does_not_call_esm(clf):
    """ESM is only consulted on the missense path. A synonymous
    substitution should leave esm_* unset even if a scorer is
    provided."""

    class _SpyScorer:
        def __init__(self):
            self.calls = 0

        def score_substitution(self, **kwargs):
            self.calls += 1
            return ESMStabilityResult(
                delta_log_likelihood=-1.0,
                stability_modifier=0.7,
                wild_type_logprob=0.0,
                mutant_logprob=-1.0,
                model_name="spy",
            )

    spy = _SpyScorer()
    # Find a synonymous wobble in KRAS
    from engine.py.molecular.mutation_effect import _CODON_TABLE
    found = False
    for codon_1based in range(1, 50):
        cs = (codon_1based - 1) * 3
        if cs + 3 > len(KRAS_CDS):
            break
        wt_codon = KRAS_CDS[cs:cs + 3]
        for new_third in "ACGT":
            if new_third == wt_codon[2]:
                continue
            mut_codon = wt_codon[:2] + new_third
            if _CODON_TABLE.get(wt_codon) == _CODON_TABLE.get(mut_codon):
                e = clf.classify_substitution(
                    KRAS_CDS, cs + 2, new_third,
                    gene_name="KRAS",
                    protein_sequence=KRAS_PROTEIN,
                    esm_scorer=spy,
                )
                assert e.category == "synonymous"
                assert spy.calls == 0
                assert e.esm_delta_log_likelihood is None
                found = True
                break
        if found:
            break
    assert found


# ---------------------------------------------------------------------------
# RealESMStabilityScorer behavior
# ---------------------------------------------------------------------------

def test_real_esm_scorer_constructs_or_raises_clear_error():
    """Either transformers + torch are installed (and construction
    succeeds), or they are not (and the error message is helpful)."""
    try:
        scorer = RealESMStabilityScorer()
        assert scorer.model_name == DEFAULT_ESM_MODEL
        # Don't actually load the weights here -- the test just
        # verifies construction is well-behaved.
    except ImportError as e:
        assert "transformers" in str(e) or "torch" in str(e), (
            f"ImportError should name the missing package: {e}"
        )


@pytest.mark.skipif(
    os.environ.get("ENABLE_ESM_SMOKE") != "1",
    reason="opt-in real-ESM smoke test (set ENABLE_ESM_SMOKE=1 to run; "
           "downloads ~600MB ESM-2 weights and takes ~30s on CPU)",
)
def test_real_esm_scorer_smoke_kras_g12d():
    """End-to-end smoke test against the published ESM-2 150M model.

    Opt-in via ENABLE_ESM_SMOKE=1. Verifies:
      * model loads
      * a substitution at the KRAS G12 position returns a valid
        ESMStabilityResult
      * the result fields are well-formed (numerical, in expected ranges)
    """
    scorer = RealESMStabilityScorer()
    r = scorer.score_substitution(
        protein_sequence=KRAS_PROTEIN,
        position_1based=12,
        wild_type_aa="G",
        mutant_aa="D",
    )
    assert isinstance(r, ESMStabilityResult)
    assert isinstance(r.delta_log_likelihood, float)
    assert 0.0 <= r.stability_modifier <= 1.0
    assert r.model_name == DEFAULT_ESM_MODEL
