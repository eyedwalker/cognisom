"""The MHC-I baselines are ordered by the CELLxGENE census, not by a demo.

The per-archetype surface levels in antigen_presentation were ten copies
of cell_renderer's demo numbers: every somatic type 0.9, cancer 0.3,
immune 1.0. The census cannot give a surface level (RNA is not protein)
but it can say which cell type sits above which, within one dataset,
and it can bracket tumour against normal. This pins that the stored
evidence says what the table claims, and that the table follows it.
"""

from __future__ import annotations

import json
import os

import pytest

from cognisom.core import antigen_presentation as ap
from cognisom.library import census_mhc1

EVIDENCE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "cognisom", "library", "evidence", "census_prostate_mhc1.json")


@pytest.fixture(scope="module")
def evidence():
    assert os.path.exists(EVIDENCE), "run python -m cognisom.library.census_mhc1"
    with open(EVIDENCE) as fh:
        return json.load(fh)


def _rows(evidence, key):
    table = evidence["tables"][key]
    return {r["cell_type"]: r for r in table["rows"]}


def test_the_evidence_is_pinned_to_one_census_build_and_is_within_dataset(evidence):
    assert evidence["census_version"] == census_mhc1.CENSUS_VERSION == "2025-11-08"
    assert "within one dataset" in evidence["unit"]
    assert evidence["n_cells"] > 200_000 and evidence["n_datasets"] == 7
    for table in evidence["tables"].values():
        assert all(r["n"] >= census_mhc1.MIN_CELLS for r in table["rows"])
        assert all(r["term"].startswith("CL:") for r in table["rows"])


def test_normal_prostate_ranks_basal_below_smooth_muscle_below_luminal(evidence):
    n = _rows(evidence, f"{census_mhc1.NORMAL_DATASET[:8]}:normal")
    lum = n["luminal cell of prostate epithelium"]["mhc1"]
    assert n["basal cell of prostate epithelium"]["mhc1"] < n["smooth muscle cell of prostate"]["mhc1"] < lum
    assert n["prostate gland microvascular endothelial cell"]["mhc1"] > lum
    assert n["basal cell of prostate epithelium"]["mhc1"] / lum == pytest.approx(0.43, abs=0.05)


def test_immune_cells_carry_several_times_the_epithelial_level(evidence):
    b = _rows(evidence, f"{census_mhc1.NORMAL_DATASET[:8]}:benign prostatic hyperplasia")
    lum = b["luminal cell of prostate epithelium"]["mhc1"]
    for immune in ("effector CD8-positive, alpha-beta T cell", "natural killer cell",
                   "classical monocyte", "dendritic cell"):
        assert b[immune]["mhc1"] > 1.5 * lum, immune


def test_tumour_luminal_is_a_third_to_a_half_of_normal_luminal(evidence):
    r = evidence["tumour_vs_normal_luminal"]
    assert 0.25 <= r["direct_cross_dataset"] <= 0.45
    assert 0.40 <= r["t_cell_referenced"] <= 0.65
    assert r["direct_cross_dataset"] < r["t_cell_referenced"]


def test_the_baseline_table_follows_the_census_order():
    b = ap.BASELINE
    luminal, basal, fibro, smooth, endo, cancer, t, mac, nk = (b[i] for i in range(9))
    assert basal < smooth < luminal == fibro <= endo
    assert cancer < basal
    assert t >= endo and nk >= endo and mac >= luminal
    # the tumour bracket the census gives, 0.34-0.51 of normal luminal
    assert 0.3 <= cancer / luminal <= 0.52


def test_the_table_says_the_values_are_still_estimates():
    text = " ".join(ap.CAVEATS)
    assert "census" in text.lower() and "estimate" in text.lower()
    assert "surface protein" in text
    assert ap.CENSUS_EVIDENCE.endswith("census_prostate_mhc1.json")
