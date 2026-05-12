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
from typing import Iterable, List, Optional, Sequence

import numpy as np

from engine.py.immune.mhc_loading import MHCPresentation


FEATURE_DIM: int = 16
_SIGMOID_SCALE: float = 4.0  # see ``affinity()`` for rationale
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
    """A TCR-pMHC affinity assessment."""
    tcr: TCR
    presentation: MHCPresentation
    affinity: float           # in [0, 1]
    is_recognized: bool       # affinity >= recognition_threshold


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
    ):
        self.size = int(size)
        self.recognition_threshold = float(recognition_threshold)
        self._rng = np.random.default_rng(seed)
        self.tcrs: List[TCR] = self._generate(self.size)

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
        has ``is_recognized`` set when affinity >= recognition_threshold.
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
        return TCRMatch(
            tcr=self.tcrs[idx],
            presentation=presentation,
            affinity=aff,
            is_recognized=aff >= self.recognition_threshold,
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
