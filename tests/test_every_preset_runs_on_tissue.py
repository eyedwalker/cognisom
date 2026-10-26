"""Every preset runs on tissue, not on a bead necklace.

Until now every preset but TissueSlab seeded a RING: about 90 cells on a
circle 1 um thick with a 100 um hole in the middle. It looked like a
necklace in the renderer, and it was wrong in a way that decided the
biology: immune cells were 12.6-36.7% of ALL cells in those presets,
against about 3% in real stroma, so a crowd of killers met a handful of
tumour cells in a patch the width of a hair and every run ended with the
tumour cleared -- including the one labelled "fast-dividing tumor with
reduced immune response".

EngineRunner now defaults to tissue="slab": each preset is re-expressed
on TissueSlab's close-packed volume by one rule anchored on Baseline, so
each keeps its character RELATIVE to Baseline rather than its toy-sized
absolute counts. tissue="ring" keeps the legacy layout for the tests
that pin what was calibrated on it; they now say so explicitly.

Measured on the 2 h slab runs (seed 1):

    preset            cancer   T/NK/Mac   immune share   cancer after 2 h
    Baseline              80   28/18/14        2.9%           80 -> 34
    Aggressive Tumor     240    14/9/7         1.4%          240 -> 224
    Immunotherapy        120   69/35/23        5.9%          120 -> 40

The aggressive tumour now persists. Baseline is still cleared fast: the
innate arm kills on 80% of contacts with no source for that number,
which is item 2 of the realism plan, not this one.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import numpy as np
import pytest

from cognisom.dashboard.engine_runner import (
    SCENARIOS, TISSUE_LAYOUTS, EngineRunner, slab_variant,
)

RING_PRESETS = [name for name, cfg in SCENARIOS.items()
                if cfg.get("cellular", {}).get("seeding") != "slab"]
IMMUNE = ("n_t_cells", "n_nk_cells", "n_macrophages")


def _immune_total(cfg):
    return sum(cfg.get("immune", {}).get(k, 0) for k in IMMUNE)


def _immune_share(cfg):
    cell = cfg["cellular"]
    return _immune_total(cfg) / (
        cell["n_normal_cells"] + cell["n_cancer_cells"] + _immune_total(cfg))


# ── the default ───────────────────────────────────────────────────


def test_the_default_tissue_is_the_slab():
    runner = EngineRunner()
    assert runner.tissue == "slab"
    assert runner.module_configs["cellular"]["seeding"] == "slab"
    assert runner.module_configs["immune"]["seeding"] == "slab"
    assert runner.module_configs["vascular"]["layout"] == "slab"


def test_the_ring_is_still_there_when_asked_for():
    runner = EngineRunner(tissue="ring")
    assert runner.tissue == "ring"
    assert runner.module_configs["cellular"].get("seeding", "ring") == "ring"
    assert runner.module_configs["cellular"]["n_normal_cells"] == 80


def test_an_unknown_tissue_is_refused():
    with pytest.raises(ValueError, match="tissue must be one of"):
        EngineRunner(tissue="sheet")
    assert TISSUE_LAYOUTS == ("slab", "ring")


def test_tissueslab_passes_through_untouched():
    assert slab_variant(SCENARIOS["TissueSlab"]) == SCENARIOS["TissueSlab"]


def test_the_conversion_does_not_edit_the_presets_in_place():
    before = repr(SCENARIOS["Aggressive Tumor"])
    slab_variant(SCENARIOS["Aggressive Tumor"])
    assert repr(SCENARIOS["Aggressive Tumor"]) == before


# ── the artefact the ring carried ─────────────────────────────────


@pytest.mark.parametrize("name", RING_PRESETS)
def test_immune_cells_are_a_tissue_fraction_not_a_crowd(name):
    """In the ring presets immune cells were 12.6-36.7% of all cells.
    In stroma they are a few percent."""
    ring = SCENARIOS[name]
    ring_share = _immune_total(ring) / (
        ring["cellular"]["n_normal_cells"] + ring["cellular"]["n_cancer_cells"]
        + _immune_total(ring))
    slab_share = _immune_share(slab_variant(ring))
    assert ring_share > 0.12, "the ring presets really were this dense"
    assert 0.01 <= slab_share <= 0.07, f"{name}: {slab_share:.1%}"


@pytest.mark.parametrize("name", RING_PRESETS)
def test_every_preset_fills_the_same_tissue_volume(name):
    cell = slab_variant(SCENARIOS[name])["cellular"]
    assert cell["n_normal_cells"] + cell["n_cancer_cells"] == 2040


# ── each preset keeps its character ───────────────────────────────


def test_baseline_maps_onto_the_slab_spec():
    """The anchor: Baseline's 10 ring cancer cells become the slab
    spec's 80, and its 26-cell infiltrate becomes TissueSlab's 60."""
    base = slab_variant(SCENARIOS["Baseline"])
    assert base["cellular"]["n_cancer_cells"] == 80
    assert _immune_total(base) == _immune_total(SCENARIOS["TissueSlab"]) == 60


def test_the_aggressive_tumour_is_bigger_and_less_watched():
    base = slab_variant(SCENARIOS["Baseline"])
    aggr = slab_variant(SCENARIOS["Aggressive Tumor"])
    assert aggr["cellular"]["n_cancer_cells"] > base["cellular"]["n_cancer_cells"]
    assert _immune_total(aggr) < _immune_total(base)
    assert aggr["cellular"]["division_time_cancer"] == 8.0, "passes through"


def test_immunotherapy_keeps_its_infiltrate_and_its_drug():
    base = slab_variant(SCENARIOS["Baseline"])
    immuno = slab_variant(SCENARIOS["Immunotherapy"])
    assert _immune_total(immuno) > 2 * _immune_total(base)
    assert immuno["immune"]["checkpoint_block"] == 1.0


@pytest.mark.parametrize("name", RING_PRESETS)
def test_each_preset_keeps_its_own_lineage_mix(name):
    ring, slab = SCENARIOS[name]["immune"], slab_variant(SCENARIOS[name])["immune"]
    ring_total = sum(ring.get(k, 0) for k in IMMUNE)
    slab_total = sum(slab.get(k, 0) for k in IMMUNE)
    for key in IMMUNE:
        assert slab[key] / slab_total == pytest.approx(
            ring[key] / ring_total, abs=0.03), key


def test_hypoxia_gets_sparser_vessels_agreed_by_all_three_modules():
    """The slab ignores n_capillaries, so the ring's 3-of-8 capillaries
    becomes a wider lattice pitch. The three modules that read the slab
    spec must agree on it, or the tissue and its vessels disagree about
    where the tissue is."""
    hyp = slab_variant(SCENARIOS["Hypoxia"])
    pitches = {hyp[m]["slab"]["vessel_pitch"] for m in ("cellular", "vascular", "immune")}
    assert pitches == {35.9}, pitches
    assert hyp["vascular"]["exchange_rate"] == 0.4, "passes through"
    assert "slab" not in slab_variant(SCENARIOS["Baseline"])["vascular"]


def test_metastasis_keeps_its_lymphatics_on_the_tumour_margin():
    """The ring preset drew its lymphatic ring through the margin of a
    ~17 um tumour at (120, 120, 50). On the slab the nodule is elsewhere
    and bigger, so the ring follows it and keeps its proportions."""
    from cognisom.modules.tissue_slab import seed_slab, slab_spec, rng_from

    meta = slab_variant(SCENARIOS["Metastasis Risk"])
    lym = meta["lymphatic"]
    spec = slab_spec(**meta["cellular"].get("slab", {}))
    layout = seed_slab(meta["cellular"]["n_normal_cells"],
                       meta["cellular"]["n_cancer_cells"],
                       spacing=spec["spacing"], layers=spec["layers"],
                       center=spec["center"], nodule_offset=spec["nodule_offset"],
                       jitter=spec["jitter"], rng=rng_from(spec["seed"]))
    assert lym["ring_centre_um"] == pytest.approx(tuple(layout["nodule_center"]))
    assert lym["ring_inner_um"] < layout["nodule_radius"] < lym["ring_outer_um"]
    assert lym["metastasis_probability"] == 0.01, "passes through"


def test_androgen_deprivation_keeps_its_castrate_serum():
    adt = slab_variant(SCENARIOS["Androgen Deprivation"])
    assert adt["vascular"]["serum_total_testosterone_nM"] == pytest.approx(17.0 * 0.06)


# ── it is tissue ──────────────────────────────────────────────────


def test_the_seeded_cells_touch_their_neighbours():
    """The ring's normal cells sat on one circle with a 100 um hole. In
    the slab every cell is in contact: median nearest neighbour inside
    the 10-12 um contact-to-adhesion band, measured 9.2 um."""
    from cognisom.modules.tissue_slab import nearest_neighbour_distances

    runner = EngineRunner(scenario="Baseline")
    with redirect_stdout(io.StringIO()):
        runner.build()
    cells = runner.engine.modules["cellular"].cells
    pts = np.array([c.position for c in cells.values()], dtype=float)
    median = float(np.median(nearest_neighbour_distances(pts)))
    assert 7.0 < median <= 12.0, median
    extent = pts.max(axis=0) - pts.min(axis=0)
    assert extent[2] > 40.0, "a volume, not a sheet one cell thick"


@pytest.mark.slow
@pytest.mark.timeout(400)
@pytest.mark.parametrize("name", list(SCENARIOS))
def test_every_preset_runs_on_the_slab(name):
    np.random.seed(1)
    runner = EngineRunner(dt=0.05, duration=0.5, scenario=name,
                          snapshot_interval_hours=0.25)
    with redirect_stdout(io.StringIO()):
        runner.build()
        runner.run()
    assert len(runner.cell_snapshots[-1].cell_ids) > 1000


@pytest.mark.slow
@pytest.mark.timeout(400)
def test_the_aggressive_tumour_is_no_longer_wiped_out():
    """On the ring, 13 immune cells cleared all 30 cancer cells within 12 h.
    On the slab, with a tissue-fraction infiltrate, most of the tumour is
    still there after 2 h (measured 240 -> 224)."""
    np.random.seed(1)
    runner = EngineRunner(dt=0.05, duration=2.0, scenario="Aggressive Tumor",
                          snapshot_interval_hours=0.5)
    with redirect_stdout(io.StringIO()):
        runner.build()
        runner.run()
    start = runner.cell_snapshots[0].cell_types.count("cancer")
    end = runner.cell_snapshots[-1].cell_types.count("cancer")
    assert start == 240
    assert end > 0.5 * start, f"{start} -> {end}"
