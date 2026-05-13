"""
ESM-2 zero-shot stability prediction (Upgrade 3 Stage C).

Adds a biophysics-grounded stability modifier to the rule-based
mutation-effect classifier. For a missense substitution, the ESM-2
protein language model assigns log-likelihoods to the wild-type and
mutant residue at the mutation site (conditioning on the rest of the
protein sequence). The difference

    delta_log_likelihood = log P(mutant | context) - log P(WT | context)

is a zero-shot proxy for the change in free energy: deeply negative
values mean the model is "surprised" by the mutant and the
substitution is likely destabilizing; positive values mean
well-tolerated.

This stage composes with Stage A (BLOSUM-derived impact) and Stage B
(domain-aware multiplier). The patent claim surface is the
composition: a classifier that uses

    - sequence-level conservation     (BLOSUM62, Stage A)
    - functional-region proximity     (UniProt domains, Stage B)
    - biophysics-grounded stability   (ESM-2 pLM, Stage C)

is meaningfully different from prior-art rule-based-only classifiers
(e.g., COSMIC lookup, the original PhysiCell mutation framework)
which use at most one or two of these axes. ESM-2 specifically does
not require either a structure (unlike DDG-style methods such as
FoldX) or a labeled training set (unlike supervised stability
predictors), so the same scorer works on every cancer driver in the
expanded domain panel without per-gene fine-tuning.

Implementation notes
--------------------
* The real ESM-2 model is loaded lazily via HuggingFace transformers
  when the module is first asked to score something. The 150M-parameter
  variant (esm2_t30_150M_UR50D) is the default -- it runs in ~1-2s per
  inference on CPU and is the smallest variant Meta released that still
  produces calibrated zero-shot stability scores.
* If transformers / torch are not installed, ``RealESMStabilityScorer``
  raises a clear error at construction time. Callers can plug in any
  scorer that implements ``score_substitution`` -- the classifier only
  knows about the interface, not the implementation.
* ``StubESMStabilityScorer`` is provided for tests and for "ESM-2
  unavailable" graceful-degradation paths. It returns a constant
  neutral score (delta_log_likelihood = 0.0), which composes with
  Stage A + B without changing their behavior.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Protocol


# ESM-2 model variants. The 150M model is the default for CPU
# inference; bumping to 650M / 3B requires a GPU to be practical.
DEFAULT_ESM_MODEL: str = "facebook/esm2_t30_150M_UR50D"


@dataclass(frozen=True)
class ESMStabilityResult:
    """Outcome of an ESM-2 zero-shot stability query.

    delta_log_likelihood
        log P(mutant | context) - log P(WT | context).
        Negative = destabilizing. Positive = well-tolerated.
        Magnitude is calibrated by training set; ~|2| is a typical
        threshold for "substantially different from WT".

    stability_modifier
        delta_log_likelihood mapped onto [0, 1] via a sigmoid centered
        at 0. Used as a multiplicative impact modifier downstream.
        Destabilizing substitutions push the modifier toward 1.0;
        well-tolerated substitutions push toward 0.0.

    wild_type_logprob, mutant_logprob
        The raw per-residue log-probabilities from ESM-2 at the
        mutation position. Carried for auditability; the composite
        impact uses only delta_log_likelihood.

    model_name
        The specific ESM-2 variant used (e.g., "facebook/esm2_t30_150M
        _UR50D"). Recorded so the patent disclosure can name the
        biophysics oracle exactly.
    """

    delta_log_likelihood: float
    stability_modifier: float
    wild_type_logprob: float
    mutant_logprob: float
    model_name: str


class ESMStabilityScorer(Protocol):
    """Interface the classifier consumes.

    Any object with ``score_substitution(protein_seq, position_1based,
    wt_aa, mut_aa) -> ESMStabilityResult`` qualifies. This keeps the
    classifier decoupled from the real ESM-2 weights so tests can
    inject a stub and production code can swap in a GPU-accelerated
    variant later.
    """

    def score_substitution(
        self,
        protein_sequence: str,
        position_1based: int,
        wild_type_aa: str,
        mutant_aa: str,
    ) -> ESMStabilityResult:
        ...


# ---------------------------------------------------------------------------
# Score mapping
# ---------------------------------------------------------------------------

def delta_ll_to_stability_modifier(delta_ll: float) -> float:
    """Map ESM-2 delta-log-likelihood onto a [0, 1] stability modifier.

    Uses a sigmoid centered at 0 (so a neutral substitution maps to
    0.5). |delta_ll| ~ 2 lands at modifier ~ 0.88 / 0.12, which
    matches the calibration ESM authors observe for the threshold
    between "tolerated" and "destabilizing".

    Negative delta_ll  -> modifier > 0.5 (destabilizing)
    Positive delta_ll  -> modifier < 0.5 (well-tolerated)

    Clamps the input to +/-50 to avoid math.exp overflow on
    pathological inputs. Past +/-50 the sigmoid is already
    numerically indistinguishable from 0 or 1.
    """
    x = max(-50.0, min(50.0, float(delta_ll)))
    return 1.0 / (1.0 + math.exp(x))


# ---------------------------------------------------------------------------
# Stub scorer (always available, used in tests + graceful degradation)
# ---------------------------------------------------------------------------

class StubESMStabilityScorer:
    """Trivial scorer that returns a constant delta_log_likelihood.

    Use cases:
      1. Test fixtures -- inject a known score to verify the
         classifier integrates it correctly.
      2. Graceful degradation -- when ESM-2 is unavailable
         (transformers / torch not installed), the classifier can
         fall back to a stub returning 0.0 so its behavior reduces
         exactly to Stage A + B.
    """

    def __init__(self, delta_log_likelihood: float = 0.0):
        self._delta_ll = float(delta_log_likelihood)

    def score_substitution(
        self,
        protein_sequence: str,
        position_1based: int,
        wild_type_aa: str,
        mutant_aa: str,
    ) -> ESMStabilityResult:
        if not protein_sequence:
            raise ValueError("StubESMStabilityScorer: protein_sequence empty")
        if not (1 <= position_1based <= len(protein_sequence)):
            raise IndexError(
                f"position {position_1based} out of range for protein "
                f"of length {len(protein_sequence)}"
            )
        return ESMStabilityResult(
            delta_log_likelihood=self._delta_ll,
            stability_modifier=delta_ll_to_stability_modifier(self._delta_ll),
            wild_type_logprob=0.0,
            mutant_logprob=self._delta_ll,
            model_name="stub:constant",
        )


# ---------------------------------------------------------------------------
# Real ESM-2 scorer (lazy-loaded; HuggingFace transformers)
# ---------------------------------------------------------------------------

class RealESMStabilityScorer:
    """Wraps HuggingFace ESM-2 for zero-shot stability scoring.

    Lazy-loads the model on first use. The model object is shared
    across all calls within the same scorer instance, so the
    one-time download / load cost (~600MB for the 150M model) is
    amortized.

    Raises ImportError at construction if transformers + torch are
    not installed. This is deliberate: the classifier should not
    silently fall back to no-ESM behavior when the caller explicitly
    asked for real ESM scoring -- the caller should know.
    """

    def __init__(self, model_name: str = DEFAULT_ESM_MODEL):
        try:
            import torch  # noqa: F401
            from transformers import AutoTokenizer, AutoModelForMaskedLM  # noqa: F401
        except Exception as exc:
            raise ImportError(
                "RealESMStabilityScorer requires `transformers` and "
                "`torch`. Install with `pip install transformers torch` "
                "or use StubESMStabilityScorer for tests."
            ) from exc
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._torch = None  # cached reference

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self._model.eval()

    def score_substitution(
        self,
        protein_sequence: str,
        position_1based: int,
        wild_type_aa: str,
        mutant_aa: str,
    ) -> ESMStabilityResult:
        if not protein_sequence:
            raise ValueError("RealESMStabilityScorer: protein_sequence empty")
        if not (1 <= position_1based <= len(protein_sequence)):
            raise IndexError(
                f"position {position_1based} out of range for protein "
                f"of length {len(protein_sequence)}"
            )
        if protein_sequence[position_1based - 1] != wild_type_aa:
            raise ValueError(
                f"wild_type_aa={wild_type_aa!r} disagrees with the "
                f"protein at position {position_1based} "
                f"({protein_sequence[position_1based - 1]!r}); refusing "
                "to score a stability against a mis-specified reference"
            )
        self._ensure_loaded()
        torch = self._torch
        # Mask the residue at the target position and ask ESM-2 for
        # the per-residue log-probability distribution. Compare
        # log P(WT) vs log P(mutant) at the masked slot.
        idx0 = position_1based - 1
        masked = (
            protein_sequence[:idx0] + "<mask>" + protein_sequence[idx0 + 1 :]
        )
        inputs = self._tokenizer(masked, return_tensors="pt")
        with torch.no_grad():
            logits = self._model(**inputs).logits
        # Tokenizer adds <cls> at index 0; the masked residue is at
        # input position idx0 + 1.
        mask_token_id = self._tokenizer.mask_token_id
        # Find the mask position in the tokenized input
        token_ids = inputs["input_ids"][0]
        mask_positions = (token_ids == mask_token_id).nonzero(as_tuple=True)[0]
        if len(mask_positions) != 1:
            raise RuntimeError(
                "expected exactly one <mask> token in tokenized input; "
                f"got {len(mask_positions)}"
            )
        mask_pos = int(mask_positions[0])
        log_probs = torch.log_softmax(logits[0, mask_pos], dim=-1)
        wt_id = self._tokenizer.convert_tokens_to_ids(wild_type_aa)
        mut_id = self._tokenizer.convert_tokens_to_ids(mutant_aa)
        wt_lp = float(log_probs[wt_id].item())
        mut_lp = float(log_probs[mut_id].item())
        delta = mut_lp - wt_lp
        return ESMStabilityResult(
            delta_log_likelihood=delta,
            stability_modifier=delta_ll_to_stability_modifier(delta),
            wild_type_logprob=wt_lp,
            mutant_logprob=mut_lp,
            model_name=self.model_name,
        )


# ---------------------------------------------------------------------------
# Composite impact: Stage A + B + C
# ---------------------------------------------------------------------------

def apply_stability_to_impact(
    stage_b_impact: float,
    stability_modifier: float,
    ceiling: float = 0.85,
) -> float:
    """Combine the Stage A+B impact with an ESM-2 stability modifier.

    The composition rule is "interpolate toward the modifier":

        new_impact = stage_b_impact + (modifier - 0.5) * (1 - stage_b_impact)

    so that:
      modifier = 0.5 (neutral)    -> impact unchanged
      modifier ~ 1 (destabilize)  -> impact moves toward 1, capped at ceiling
      modifier ~ 0 (well-tolerate)-> impact moves toward 0
    The result is clamped to [0, ceiling] so it stays in the missense
    band even when ESM-2 strongly disagrees with BLOSUM + domain.
    """
    if not (0.0 <= stability_modifier <= 1.0):
        raise ValueError(
            f"stability_modifier must be in [0, 1], got {stability_modifier}"
        )
    if stability_modifier >= 0.5:
        # Push toward ceiling for destabilizing substitutions
        # delta in [0, 0.5] -> multiplicative gap-close
        gap = ceiling - stage_b_impact
        gain = 2.0 * (stability_modifier - 0.5)  # 0..1
        out = stage_b_impact + gap * gain
    else:
        # Pull toward 0 for well-tolerated substitutions
        deficit = stage_b_impact
        loss = 2.0 * (0.5 - stability_modifier)  # 0..1
        out = stage_b_impact - deficit * loss
    return max(0.0, min(ceiling, out))
