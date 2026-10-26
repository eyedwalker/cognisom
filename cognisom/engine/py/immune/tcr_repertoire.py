"""
TCR repertoire: stochastic TCR-pMHC affinity
============================================

Stage three of the closed-loop neoantigen presentation pipeline
(Upgrade 2). Models a patient's T-cell receptor repertoire as a
collection of TCRs each represented by a deterministic 16-dim feature
vector derived from a synthetic CDR3 sequence. A pMHC presentation is
embedded into the same 16-dim space, and the affinity score is the
sigmoid of the dot product.

This is intentionally a simulation primitive, not a clinical predictor.
TCRdist3 is the right production-grade replacement -- it computes
biophysically-grounded distances between CDR3 sequences with full
genetic-distance weighting -- and the interface here is shaped to swap
in TCRdist3 without touching downstream consumers (see UPGRADES_SPEC.md
Section 2 step 3: "TCRdist3 deferred").

Patent claim surface: the *closed loop* is the patent surface, not the
exact affinity values. Specifically, the chain
    MUTATION_OCCURRED  ->  PEPTIDE_GENERATED  ->  PEPTIDE_PRESENTED
        ->  TCR_RECOGNIZED  ->  CELL_KILLED_BY_TCELL
is what differentiates this from prior-art rule-based immune modules
(e.g., the threshold heuristic at gpu/spatial_ops.py:200-240).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from cognisom.engine.py.immune.mhc_loading import MHCPresentation


FEATURE_DIM: int = 16
_SIGMOID_SCALE: float = 4.0  # see ``affinity()`` for rationale


# Exhaustion machinery (Dolina et al. 2021; lecture slides 51-52).
# Threshold is the number of TCR-pMHC engagements before a clone
# transitions from PD-1-lo precursor to PD-1-hi exhausted. Empirical
# chronic-antigen models put this in the 3-10 range; 5 is the default.
EXHAUSTION_ENCOUNTER_THRESHOLD: int = 5

# The same transition on a clock, which is the unit the biology is
# measured in. In mouse tumours antigen-specific CD8 T cells were PD1-hi
# and lost IFN-gamma/TNF-alpha by day 5 but could still be rescued; their
# chromatin then fixed "between days 7 and 14", after which IL-15 no
# longer restored them (Philip et al. 2017, Nature 545:452, PMID 28514453;
# Schietinger et al. 2016, PMID 27521269). EXHAUSTED here is that fixed,
# non-rescuable state, so it begins 10 days after a clone first meets its
# antigen. ESTIMATE, range 7-14 days.
#
# The encounter count above is NOT that clock: every T cell that
# recognises a peptide takes the same best-matching clone, so the count is
# shared by the whole compartment and five contacts -- minutes into a run
# -- exhausted every T cell against that peptide. Measured on the
# Immunotherapy slab: 728 T-cell contacts in 6 h, every one from an
# exhausted clone.
EXHAUSTION_ONSET_HOURS: float = 10 * 24.0
EXHAUSTION_ONSET_RANGE_HOURS: Tuple[float, float] = (7 * 24.0, 14 * 24.0)

# Multiplier applied to an exhausted clone's contribution to kill
# probability. Clinical observation (Dolina): exhausted TILs retain
# some cytotoxicity but at a fraction of the precursor pool. 0.1
# matches the order-of-magnitude reduction observed in chronic LCMV
# and tumor models.
EXHAUSTED_KILL_MULTIPLIER: float = 0.1


class ExhaustionState(str, Enum):
    """T-cell exhaustion state per Dolina et al. 2021.

    PRECURSOR: PD-1-lo, fully functional. Rescuable by checkpoint
    blockade -- this is the pool that ICB expands.
    EXHAUSTED: PD-1-hi, dysfunctional. NOT rescued by checkpoint
    blockade alone; this is the patent-evidence-relevant
    distinction the lecture (slide 52) flagged as missing in
    rule-based prior art.
    """
    PRECURSOR = "precursor"
    EXHAUSTED = "exhausted"
# Synthetic CDR3 length range -- the real distribution is V-J recombination
# biased toward 12-18 amino acids. We use a similar bias so the repertoire
# has realistic-looking diversity even though sequences are random.
_CDR3_MIN_LEN: int = 11
_CDR3_MAX_LEN: int = 18

# Standard 20-letter amino acid alphabet (no stops, no ambiguity)
_AA_ALPHABET: str = "ACDEFGHIKLMNPQRSTVWY"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TCR:
    """A single T-cell receptor.

    tcr_id
        Stable identifier (e.g., "TCR-0042").
    cdr3
        Synthetic CDR3-beta sequence. Real repertoires use TRA + TRB
        chains plus V/J gene segments; we collapse to a single CDR3
        string for the feature embedding.
    features
        16-dim float feature vector. Deterministically derived from
        cdr3 via a SHA-256-seeded RNG, so two TCRs with the same cdr3
        always produce the same features.
    clone_frequency
        Estimated frequency of this clone in the patient's repertoire.
        Used by downstream kill probability so common clones generate
        more cytotoxic pressure than rare ones.
    """
    tcr_id: str
    cdr3: str
    features: np.ndarray
    clone_frequency: float


@dataclass(frozen=True)
class TCRMatch:
    """A TCR-pMHC affinity assessment.

    exhaustion_state and encounter_count are populated by
    TCRRepertoire.best_match from the repertoire's per-clone exhaustion
    bookkeeping. Downstream consumers (kill probability, ICB rescue
    gating) read these to decide whether the match can be rescued by
    checkpoint blockade.
    """
    tcr: TCR
    presentation: MHCPresentation
    affinity: float           # in [0, 1]
    is_recognized: bool       # affinity >= recognition_threshold
    exhaustion_state: ExhaustionState = ExhaustionState.PRECURSOR
    encounter_count: int = 0

    @property
    def is_exhausted(self) -> bool:
        return self.exhaustion_state is ExhaustionState.EXHAUSTED


# ---------------------------------------------------------------------------
# Feature embedding
# ---------------------------------------------------------------------------

def cdr3_to_features(cdr3: str) -> np.ndarray:
    """Deterministic 16-dim embedding of a CDR3 string.

    Uses SHA-256 of the CDR3 to seed a fixed-distribution Gaussian RNG.
    Same input always yields the same vector; this is the swap-in point
    for TCRdist3 or a learned encoder.
    """
    digest = hashlib.sha256(cdr3.encode("utf-8")).digest()
    # Use first 4 bytes as a 32-bit seed for reproducibility
    seed = int.from_bytes(digest[:4], "big")
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(FEATURE_DIM).astype(np.float32)
    # L2-normalize so the dot product is a cosine similarity in [-1, 1]
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec


def presentation_to_features(presentation: MHCPresentation) -> np.ndarray:
    """Deterministic 16-dim embedding of a pMHC presentation.

    Same approach as cdr3_to_features but keyed on
    ``sequence|hla_allele|mutation_label`` so peptides displayed on
    different alleles project to different points in feature space.
    """
    key = (
        f"{presentation.peptide.sequence}|{presentation.hla_allele}|"
        f"{presentation.peptide.mutation_label or ''}"
    )
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:4], "big")
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(FEATURE_DIM).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec


# ---------------------------------------------------------------------------
# Affinity scoring
# ---------------------------------------------------------------------------

def affinity(tcr: TCR, presentation: MHCPresentation) -> float:
    """Sigmoid of the dot product between TCR features and pMHC features.

    Returns a value in [0, 1]. A randomly-paired TCR and pMHC sit near
    0.5 (cosine ~0); cognate pairs occupy the high tail.
    """
    tcr_vec = tcr.features
    pmhc_vec = presentation_to_features(presentation)
    # Dot product of unit vectors is cosine in [-1, 1]; the sharpening
    # factor (_SIGMOID_SCALE) controls how aggressively the sigmoid
    # discriminates. Factor 4 keeps the recognition_threshold knob
    # meaningful: cosine 0.5 -> ~0.88, cosine 0.9 -> ~0.97. Larger
    # factors saturate near 1.0 too quickly and make threshold tuning
    # useless.
    cos = float(np.dot(tcr_vec, pmhc_vec))
    return 1.0 / (1.0 + math.exp(-_SIGMOID_SCALE * cos))


# ---------------------------------------------------------------------------
# Repertoire
# ---------------------------------------------------------------------------

class TCRRepertoire:
    """A patient's T-cell receptor repertoire.

    Parameters
    ----------
    size : int
        Number of TCR clones to instantiate (typical patient diversity
        is 10^6-10^7 clones; tests use 100-1000).
    recognition_threshold : float
        Minimum affinity (in [0, 1]) for a TCR to "recognize" a pMHC.
        0.7 yields ~5% recognition rate at random pairings, which is in
        the ballpark for self-reactivity removed thymic-selected
        repertoires. Tune as needed for sensitivity studies.
    seed : int
        RNG seed for the synthetic CDR3 sequences. Same seed = same
        repertoire, which the closed-loop test relies on.
    """

    def __init__(
        self,
        size: int = 1000,
        recognition_threshold: float = 0.7,
        seed: int = 0,
        exhaustion_threshold: int = EXHAUSTION_ENCOUNTER_THRESHOLD,
        exhaustion_onset_hours: Optional[float] = None,
    ):
        self.size = int(size)
        self.recognition_threshold = float(recognition_threshold)
        self.exhaustion_threshold = int(exhaustion_threshold)
        #: When set, a clone exhausts this many hours after its first
        #: engagement instead of at exhaustion_threshold encounters. See
        #: EXHAUSTION_ONSET_HOURS.
        self.exhaustion_onset_hours = (
            None if exhaustion_onset_hours is None
            else float(exhaustion_onset_hours))
        self._first_engaged_h: Dict[str, float] = {}
        self._rng = np.random.default_rng(seed)
        self.tcrs: List[TCR] = self._generate(self.size)

        # Per-clone exhaustion bookkeeping. Every TCR starts as a
        # PRECURSOR with zero encounters; each successful TCR-pMHC
        # engagement (recorded via register_engagement) increments
        # the counter, and crossing exhaustion_threshold transitions
        # the clone to EXHAUSTED.
        self._encounters: Dict[str, int] = {t.tcr_id: 0 for t in self.tcrs}
        self._states: Dict[str, ExhaustionState] = {
            t.tcr_id: ExhaustionState.PRECURSOR for t in self.tcrs
        }

    def _generate(self, size: int) -> List[TCR]:
        tcrs: List[TCR] = []
        # Clonal frequencies follow a power law; normalize so they sum
        # to 1 across the repertoire.
        raw_freqs = self._rng.pareto(2.0, size) + 1.0
        freqs = raw_freqs / raw_freqs.sum()
        for i in range(size):
            length = int(self._rng.integers(_CDR3_MIN_LEN, _CDR3_MAX_LEN + 1))
            cdr3 = "".join(self._rng.choice(list(_AA_ALPHABET), size=length))
            tcrs.append(TCR(
                tcr_id=f"TCR-{i:06d}",
                cdr3=cdr3,
                features=cdr3_to_features(cdr3),
                clone_frequency=float(freqs[i]),
            ))
        return tcrs

    # ----- public API --------------------------------------------------

    def best_match(self, presentation: MHCPresentation) -> Optional[TCRMatch]:
        """Find the highest-affinity TCR for this pMHC.

        Returns None if the repertoire is empty. The returned TCRMatch
        has ``is_recognized`` set when affinity >= recognition_threshold
        and carries the matched clone's current ``exhaustion_state``
        and ``encounter_count`` for downstream kill-probability and
        ICB-rescue gating.
        """
        if not self.tcrs:
            return None
        # Embed the pMHC once, then dot against all TCR features in one
        # vectorized batch (much faster than the Python loop for large
        # repertoires).
        pmhc_vec = presentation_to_features(presentation)
        feats = np.stack([t.features for t in self.tcrs])
        cosines = feats @ pmhc_vec
        affinities = 1.0 / (1.0 + np.exp(-_SIGMOID_SCALE * cosines))
        idx = int(np.argmax(affinities))
        aff = float(affinities[idx])
        chosen = self.tcrs[idx]
        return TCRMatch(
            tcr=chosen,
            presentation=presentation,
            affinity=aff,
            is_recognized=aff >= self.recognition_threshold,
            exhaustion_state=self._states[chosen.tcr_id],
            encounter_count=self._encounters[chosen.tcr_id],
        )

    def recognized_matches(
        self,
        presentations: Iterable[MHCPresentation],
    ) -> List[TCRMatch]:
        """Filter to only recognized TCR-pMHC matches."""
        out: List[TCRMatch] = []
        for pres in presentations:
            match = self.best_match(pres)
            if match is not None and match.is_recognized:
                out.append(match)
        return out

    # ----- exhaustion bookkeeping -------------------------------------

    def register_engagement(self, tcr_id: str,
                            now_h: Optional[float] = None) -> Tuple[int, bool]:
        """Record one successful TCR-pMHC engagement.

        Increments the clone's encounter counter and, when the
        threshold is crossed, transitions the clone from PRECURSOR
        to EXHAUSTED. The transition is one-way -- exhaustion is
        epigenetically enforced in the real biology (Bengsch /
        Wherry), so we model it as terminal here.

        Returns
        -------
        (new_count, did_exhaust)
            new_count: the encounter counter after the increment.
            did_exhaust: True only on the engagement that causes the
            PRECURSOR -> EXHAUSTED transition. False on every
            subsequent engagement of an already-exhausted clone.
        """
        if tcr_id not in self._encounters:
            raise KeyError(f"unknown TCR id {tcr_id!r}")
        new_count = self._encounters[tcr_id] + 1
        self._encounters[tcr_id] = new_count
        if now_h is not None:
            self._first_engaged_h.setdefault(tcr_id, float(now_h))
        if self.exhaustion_onset_hours is not None:
            due = (now_h is not None and now_h - self._first_engaged_h[tcr_id]
                   >= self.exhaustion_onset_hours)
        else:
            due = new_count >= self.exhaustion_threshold
        did_exhaust = False
        if self._states[tcr_id] is ExhaustionState.PRECURSOR and due:
            self._states[tcr_id] = ExhaustionState.EXHAUSTED
            did_exhaust = True
        return new_count, did_exhaust

    def exhaustion_state(self, tcr_id: str) -> ExhaustionState:
        return self._states[tcr_id]

    def encounters(self, tcr_id: str) -> int:
        return self._encounters[tcr_id]

    def precursor_count(self) -> int:
        return sum(
            1 for s in self._states.values()
            if s is ExhaustionState.PRECURSOR
        )

    def exhausted_count(self) -> int:
        return sum(
            1 for s in self._states.values()
            if s is ExhaustionState.EXHAUSTED
        )
