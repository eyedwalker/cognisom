"""An immune lineage the kernel cannot model must not inherit a killer's rule.

``_patch_immune`` mapped each cell to a recognition code with
``type_map.get(ic.cell_type, 2)``. Code 2 is the macrophage, and
:func:`immune_detection_batch` treats the macrophage as recognizing every
target unconditionally. So every lineage the kernel has no rule for -- a Treg,
a B cell, a plasma cell, an endothelial cell -- was handed a macrophage's kill
licence against an MHC-I-intact tumour cell, and the kill roll that follows
reads a lineage-blind probability.

A Treg suppresses tumour clearance rather than causing it, so this does not
merely add killing, it inverts the sign of that cell's effect. Exposure was
bounded -- the accelerated path is opt-in and nothing shipped seeds those
lineage strings -- but it is the path any library-to-engine wiring would hit,
which is what makes it worth a guard rather than a comment.

The fix is a sentinel code matching no branch in the recognition chain, so the
default is "patrols, never engages" rather than "kills anything it meets".
"""
from __future__ import annotations

import numpy as np
import pytest

from cognisom.gpu.orchestrator import UNMODELLED_LINEAGE
from cognisom.gpu.spatial_ops import immune_detection_batch

#: The codes the recognition kernel actually implements.
MODELLED = {"T_cell": 0, "NK_cell": 1, "macrophage": 2}

#: Lineages present in the entity library that the kernel has no rule for.
#: 'endothelial' is in this list because three library rows carry it.
UNMODELLED = ("Treg", "B_cell", "plasma_cell", "endothelial", "dendritic",
              "mast_cell", "eosinophil", "basophil", "ILC")


def _detect(code: int, mhc1: float = 1.0):
    """One immune cell, one cancer cell 1 um away, well inside the radius."""
    return immune_detection_batch(
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        detection_radius=10.0,
        cancer_mhc1=np.array([mhc1], dtype=np.float32),
        immune_types=np.array([code], dtype=np.int8),
    )


def test_the_sentinel_matches_no_recognition_rule():
    """The whole fix rests on code 3 falling through the if/elif chain."""
    assert UNMODELLED_LINEAGE not in MODELLED.values()

    assert _detect(UNMODELLED_LINEAGE) == [], (
        "the sentinel recognized a target, so it has acquired a rule and no "
        "longer keeps unmodelled lineages out of the kill loop")


def test_a_macrophage_still_engages():
    """The guard must not have disarmed the lineages that do kill."""
    assert len(_detect(MODELLED["macrophage"])) == 1
    assert len(_detect(MODELLED["T_cell"], mhc1=1.0)) == 1, "T cell needs MHC-I"
    assert len(_detect(MODELLED["NK_cell"], mhc1=0.0)) == 1, "NK needs it absent"


@pytest.mark.parametrize("lineage", UNMODELLED)
def test_an_unmodelled_lineage_does_not_get_the_macrophage_rule(lineage):
    """This is the regression: .get(lineage, 2) is what made a Treg kill."""
    code = MODELLED.get(lineage, UNMODELLED_LINEAGE)

    assert _detect(code) == [], (
        f"{lineage} recognized an MHC-I-intact tumour cell; it has been "
        "mapped onto a modelled killer's recognition rule")


def test_the_default_is_not_a_killing_code():
    """Pins the defaulting behaviour itself, not just its consequence."""
    for lineage in UNMODELLED:
        code = MODELLED.get(lineage, UNMODELLED_LINEAGE)
        assert code != MODELLED["macrophage"], (
            f"{lineage} defaulted to the macrophage code")
