"""The section layout seeds tissue at a MEASURED density, not a uniform one.

This file is the specification of :mod:`cognisom.modules.tissue_section`.
The thing being pinned is a property, not a picture: real prostate fills
about 60% of the sites a close packing offers, so a layout built from a
measured section must be less connected than the fully-packed slab at
the same cell count, and the slab must therefore be over-connecting
every cell it simulates.

Numbers measured here are written beside the claim they support rather
than on their own, and the guards say so: if one moves, re-measure it
against the fixture and work out which claim changed. Re-pinning a
number to make a test pass discards the only evidence the test carried.

The population is deliberately identical to the slab's (1,960 + 80 in
both specs) so that every difference between the two layouts is a
difference in ARRANGEMENT. A test here that changed the cell count
would be comparing two things at once.
"""

import contextlib
import io

import numpy as np
import pytest
from scipy.spatial import cKDTree

from cognisom.modules import tissue_section as ts
from cognisom.modules.tissue_slab import seed_slab
from cognisom.modules.tissue_slab import rng_from as slab_rng

#: Centre-to-centre distance out to which cell-cell adhesion still acts
#: (tissue_slab.ADHESION_CUTOFF_UM). Neighbour counts below are taken at
#: this radius because it is the distance the engine's contact biology
#: actually uses, not a round number.
ADHESION_UM = 12.0

#: The shipped population, from SECTION_SPEC and SLAB_SPEC alike.
N_NORMAL, N_CANCER = 1960, 80


def _neighbour_counts(positions, radius=ADHESION_UM):
    tree = cKDTree(positions)
    return np.array([len(tree.query_ball_point(p, radius)) - 1
                     for p in positions])


def _occupied(layout):
    if len(layout["cancer"]):
        return np.vstack([layout["normal"], layout["cancer"]])
    return layout["normal"]


# ── The fixture ──────────────────────────────────────────────────────

def test_the_fixture_loads_with_the_columns_the_seeder_needs():
    data = ts.load_section()
    assert set(data) == {"x", "y", "d_eq", "seg"}
    n = len(data["x"])
    assert n > 1500, "fixture shrank to %d cells" % n
    assert all(len(data[k]) == n for k in data)
    # Re-originned to the window corner, so no absolute slide coords.
    assert data["x"].min() >= 0.0 and data["y"].min() >= 0.0
    assert data["x"].max() <= ts.SECTION_WINDOW_UM + 1.0
    assert data["y"].max() <= ts.SECTION_WINDOW_UM + 1.0


def test_the_fixture_carries_its_licence_and_its_limits():
    """Provenance travels with the data or it does not travel.

    The positions are CC BY 4.0, so the attribution has to be reachable
    from the file itself rather than from a commit message. The caveats
    are in there for the same reason: a reader who finds the file
    without this module should still learn that the segmentation method
    changes the diameter and that cell types are absent on purpose.
    """
    with open(ts.section_path(), "r") as handle:
        header = "".join(line for line in handle if line.startswith("#"))
    assert "CC BY 4.0" in header
    assert "10x Genomics" in header
    assert "sha256" in header, "no checksum for the source file"
    assert "NO z" in header, "header must say z is unmeasured"
    assert "CELL TYPES ARE NOT IN THIS FILE" in header
    assert "10x Genomics" in ts.SECTION_ATTRIBUTION


def test_the_segmentation_method_changes_the_diameter_so_it_is_stratified():
    """Pooling segmentation methods biases the cell size low.

    The interior (18S) stain bounds RNA, which stops short of the
    membrane, and it is the MAJORITY of the fixture -- so a reader who
    takes ``d_eq`` without stratifying gets a number pulled down by an
    artefact. ``membrane_diameters`` exists to make the right subset the
    easy one to reach for.

    Measured: membrane 9.74 um median, interior 7.48, pooled 8.37.
    If these move, re-measure against the fixture; do not re-pin.
    """
    data = ts.load_section()
    seg = data["seg"].astype(str)
    membrane = np.median(data["d_eq"][seg == "boundary"])
    interior = np.median(data["d_eq"][seg == "interior"])
    pooled = np.median(data["d_eq"])

    assert interior < pooled < membrane, (
        "the interior stain no longer reads smaller than the membrane; "
        "the stratification in membrane_diameters may be unnecessary or "
        "may be selecting the wrong subset")
    assert membrane / interior > 1.15, (
        "membrane/interior ratio fell to %.3f from a measured 1.30 -- "
        "re-measure, do not re-pin" % (membrane / interior))
    assert np.median(ts.membrane_diameters()) == pytest.approx(
        membrane, abs=1e-9)
    assert membrane == pytest.approx(9.74, abs=0.3)


# ── The measured fill fraction, which is the whole point ─────────────

def test_real_prostate_fills_about_sixty_percent_of_a_packed_layer():
    """The measurement the layout is built on.

    Stable across window size is what makes it trustworthy: a number
    that drifted with the window would be reporting the window, not the
    tissue. Measured 0.588 / 0.595 / 0.617 at 240 / 300 / 600 um.

    It matters beyond this module because it is also what the
    cell-diameter and lattice-pitch consistency check needs
    independently -- a packed volume fraction near 0.6 is what puts the
    11 um pitch in the right band.
    """
    fills = [ts.measured_layer_fill(w) for w in (240.0, 300.0, 600.0)]
    for window, fill in zip((240.0, 300.0, 600.0), fills):
        assert 0.4 < fill < 0.8, (
            "measured fill at %.0f um is %.3f, outside the range a packed "
            "tissue can have -- re-measure, do not re-pin" % (window, fill))
    assert max(fills) - min(fills) < 0.1, (
        "fill fraction now varies by %.3f across window size, so it is "
        "reporting the window rather than the tissue" % (max(fills) - min(fills)))
    assert fills == pytest.approx([0.588, 0.595, 0.617], abs=0.02)
    assert ts.MEASURED_LAYER_FILL == pytest.approx(np.mean(fills), abs=0.05)


def test_the_slab_fills_every_site_and_so_over_connects_every_cell():
    """THE CLAIM: full close packing is not what tissue does.

    The slab offers every lattice site and takes it, giving a fill of
    1.0 against the measured ~0.60. That is the mechanism behind the
    connectivity gap the next test measures, and it means
    contact-dependent biology on the slab runs against more neighbours
    than prostate has.
    """
    layout = seed_slab(N_NORMAL, N_CANCER, rng=slab_rng(1))
    occupied = _occupied(layout)
    extents = layout["extents"]
    # The slab's own sites, in plane: its population divided by the
    # layers it stacks them in, over one layer's capacity.
    per_layer = len(occupied) / layout["layers"]
    capacity = ts.layer_capacity(
        float(np.sqrt(extents[0] * extents[1])), layout["spacing"])
    slab_fill = per_layer / capacity
    assert slab_fill > 0.9, (
        "the slab's layer fill is %.3f, no longer ~1.0; if the slab has "
        "stopped filling every site then this module's premise changed"
        % slab_fill)
    assert slab_fill / ts.MEASURED_LAYER_FILL > 1.3, (
        "the slab/measured fill ratio fell to %.2f from 1.6, so the "
        "over-connection claim needs re-deriving"
        % (slab_fill / ts.MEASURED_LAYER_FILL))


def test_the_section_is_less_connected_than_the_slab_at_the_same_population():
    """THE CLAIM, at the level the engine feels it.

    Same cell count, different arrangement: the section's cells have
    about two-thirds of the slab's neighbours inside the adhesion
    cutoff, because 40% of the sites are lumen. Measured over seed 1:
    slab 7.80 neighbours and 1 isolated cell, section 5.07 and 33.

    The populations are asserted equal first, because a connectivity
    difference between different cell counts would mean nothing.
    """
    section = ts.seed_section(N_NORMAL, N_CANCER, rng=ts.rng_from(1))
    slab = seed_slab(N_NORMAL, N_CANCER, rng=slab_rng(1))
    sec_pos, slab_pos = _occupied(section), _occupied(slab)
    assert len(sec_pos) == len(slab_pos) == N_NORMAL + N_CANCER

    sec_n = _neighbour_counts(sec_pos).mean()
    slab_n = _neighbour_counts(slab_pos).mean()
    assert sec_n < slab_n, (
        "the section is no longer less connected than the slab "
        "(%.2f vs %.2f); the measured occupancy is not reaching the "
        "layout" % (sec_n, slab_n))
    ratio = sec_n / slab_n
    assert 0.5 < ratio < 0.85, (
        "section/slab neighbour ratio is %.3f, measured 0.65 -- "
        "re-measure, do not re-pin" % ratio)
    assert sec_n == pytest.approx(5.07, abs=0.6)
    assert slab_n == pytest.approx(7.80, abs=0.6)


def test_the_section_keeps_the_lattice_pitch_and_only_changes_occupancy():
    """What the section does NOT change.

    The pitch is independently right -- the measured packed cell
    diameter puts it at 9.8-11.5 um and the engine runs at 11.0 -- so
    the section must leave it alone. Its nearest-neighbour distance
    therefore stays at the lattice's, within the jitter, rather than
    stretching: thinning a packing removes neighbours, it does not move
    the ones that remain.
    """
    section = ts.seed_section(N_NORMAL, N_CANCER, rng=ts.rng_from(1))
    slab = seed_slab(N_NORMAL, N_CANCER, rng=slab_rng(1))
    sec_nn = ts.median_nn_distance(_occupied(section))
    slab_nn = ts.median_nn_distance(_occupied(slab))
    assert section["spacing"] == pytest.approx(slab["spacing"])
    assert abs(sec_nn - slab_nn) < 1.5, (
        "section NN %.2f vs slab %.2f: thinning a packing should not "
        "move the surviving neighbours" % (sec_nn, slab_nn))
    assert sec_nn == pytest.approx(9.52, abs=0.4)


def test_the_measured_voids_exist_at_the_scale_lumens_do():
    """Void fraction is scale-dependent, and 50 um is the wrong scale.

    At 50 um every subcell of the window holds a cell, so a 50 um void
    fraction reads 0.0 and would wrongly suggest the layout has no gaps.
    The gaps are a ~10-20 um feature. This is pinned because an earlier
    measurement of this very quantity was wrong twice over -- an
    off-by-one in the bin count and the wrong scale -- and reported
    15-18% where the truth at 50 um is zero.
    """
    data = ts.load_section()
    positions = np.column_stack(
        [data["x"], data["y"], np.zeros(len(data["x"]))])
    coarse = ts.void_fraction(positions, 50.0)
    fine = ts.void_fraction(positions, 20.0)
    finer = ts.void_fraction(positions, 10.0)
    assert coarse == pytest.approx(0.0, abs=1e-9), (
        "50 um void fraction is %.3f, not 0 -- if the fixture changed, "
        "re-measure the scale at which its voids appear" % coarse)
    assert finer > fine > coarse, "void fraction must fall as bins coarsen"
    assert fine == pytest.approx(0.130, abs=0.04)
    assert finer == pytest.approx(0.521, abs=0.08)


# ── The domain follows the population ────────────────────────────────

def test_the_shipped_domain_constant_matches_what_the_seeder_derives():
    """DOMAIN_UM is stated, not computed, so it is pinned equal.

    Computing it at import would read the fixture at import time. The
    cost of stating it is that it can drift from
    ``section_domain``, and this is the guard that stops it.
    """
    assert tuple(ts.DOMAIN_UM) == ts.section_domain(N_NORMAL + N_CANCER)
    assert ts.DOMAIN_CENTER_UM == tuple(0.5 * d for d in ts.DOMAIN_UM)


@pytest.mark.parametrize("n_normal,n_cancer", [
    (50, 5), (400, 20), (900, 60), (N_NORMAL, N_CANCER), (2900, 100),
])
def test_every_cell_stays_inside_the_domain_and_the_nominal_box(
        n_normal, n_cancer):
    """Containment across a 50x population range.

    Two separate things, because they failed separately while this was
    being built. The nominal box must be a SUPERSET of the cells, which
    it was not until the jitter margin was added -- the jitter is
    applied after centring, so a cell can sit beyond the window edge.
    And the domain must contain the cells at a non-negative coordinate,
    which it did not for large populations until the depth was derived
    from the count instead of being a constant.
    """
    layout = ts.seed_section(n_normal, n_cancer, rng=ts.rng_from(3))
    occupied = _occupied(layout)
    lo, hi = ts.section_bounds(n_normal, n_cancer)
    assert (occupied >= lo - 1e-9).all() and (occupied <= hi + 1e-9).all(), (
        "cells outside the nominal box, which vascular and immune size "
        "themselves from -- the outermost cells would be unperfused")
    domain = np.asarray(layout["domain"])
    assert (occupied >= 0.0).all(), "negative coordinate: outside the grid"
    assert (occupied <= domain + 1e-9).all(), (
        "cells beyond the domain the layout reports, so the spatial grid "
        "built from it would exclude them")


def test_more_cells_makes_the_tissue_deeper_not_denser():
    """Density is measured, so only depth is free.

    This is what stops the layout quietly becoming a fill-fraction knob:
    doubling the population must add layers at the same occupancy rather
    than packing the existing ones tighter.
    """
    small = ts.seed_section(400, 20, rng=ts.rng_from(5))
    large = ts.seed_section(1600, 80, rng=ts.rng_from(5))
    assert large["layers"] > small["layers"]
    assert large["layer_fill"] == pytest.approx(small["layer_fill"], abs=1e-9)
    assert large["domain"][2] > small["domain"][2]
    assert large["domain"][0] == small["domain"][0], (
        "the in-plane window must not grow with the population; the "
        "measured density field is only defined over the window")


def test_the_spatial_grid_covers_the_domain_in_whole_voxels():
    """A grid that did not cover the cells would fail silently.

    The fields would simply not reach the cells beyond their extent, and
    nothing raises. So the grid is derived from the domain and checked
    to cover it, rather than being a constant a caller remembers.
    """
    for n in (420, N_NORMAL + N_CANCER, 3000):
        domain = ts.section_domain(n)
        grid = ts.section_grid(domain)
        covered = [c * grid["resolution"] for c in grid["grid_size"]]
        for extent, span in zip(domain, covered):
            assert span >= extent - 1e-9, (
                "grid covers %.1f um of a %.1f um domain" % (span, extent))
            assert span - extent < grid["resolution"], (
                "grid overshoots the domain by more than a voxel")
    assert ts.section_grid()["resolution"] == 10.0, (
        "the section must not re-grade the solver's voxel size, only its "
        "extent")


def test_asking_for_more_cells_than_were_measured_is_refused():
    """The monolayer cannot invent cells, and says so.

    Padding a shortfall with lattice sites would turn a measured
    arrangement back into a generated one while still reporting itself
    as measured, which is the failure mode with no symptom.
    """
    available = len(ts.load_section()["x"])
    with pytest.raises(ValueError, match="cannot be invented"):
        ts.seed_monolayer(available + 1, 0)


# ── The monolayer, and why it is not the default ─────────────────────

def test_the_monolayer_is_measurably_worse_connected_than_the_volume():
    """Why :func:`seed_section` and not :func:`seed_monolayer`.

    Sectioning truncates each cell's neighbourhood -- a 5 um slice shows
    only the neighbours that also cross it -- so the literal section is
    under-connected relative to both the volume layout and the slab, and
    leaves cells adhering to nothing. Measured: 2.8 neighbours and 121
    isolated cells of 2,040, against the volume's 5.1 and 26.

    The monolayer is kept because it is the right thing for comparing
    against imaging. This test records why it must not drive the engine.
    """
    mono = ts.seed_monolayer(N_NORMAL, N_CANCER, rng=ts.rng_from(1))
    vol = ts.seed_section(N_NORMAL, N_CANCER, rng=ts.rng_from(1))
    mono_pos, vol_pos = _occupied(mono), _occupied(vol)
    assert len(mono_pos) == len(vol_pos)

    mono_deg = _neighbour_counts(mono_pos)
    vol_deg = _neighbour_counts(vol_pos)
    assert mono_deg.mean() < vol_deg.mean(), (
        "the monolayer is no longer the less connected of the two, so the "
        "reason for the default has changed")
    assert (mono_deg == 0).sum() > 4 * (vol_deg == 0).sum(), (
        "monolayer isolated %d vs volume %d: the truncation this test "
        "exists to record has gone"
        % (int((mono_deg == 0).sum()), int((vol_deg == 0).sum())))
    assert mono["layers"] == 1
    assert vol["layers"] > 1


def test_the_monolayer_reports_its_own_caveat():
    """A report that omitted this would let a reader acquire a claim.

    ``section_report`` is the dict a run records, so the distinction
    between measured positions and an unmeasured z -- and the warning
    that a section is not comparable to a volume -- has to be in it.
    """
    mono = ts.section_report(ts.seed_monolayer(400, 20, rng=ts.rng_from(1)))
    assert "truncates" in mono["caveat"]
    assert "unmeasured" in mono["z"]
    assert mono["positions"].startswith("measured")

    vol = ts.section_report(ts.seed_section(400, 20, rng=ts.rng_from(1)))
    assert "caveat" not in vol
    assert vol["positions"] == "lattice, occupancy from the measured density"
    assert vol["density"] == "measured"


def test_every_report_says_cell_types_are_not_measured():
    """The claim this layout must never acquire.

    The fixture has no cell-type column on purpose: typing cells from
    the panel's own markers and then consuming the result would be a
    round trip. So both layouts have to state that their cell types come
    from the population prior, on every run.
    """
    for layout in (ts.seed_section(400, 20, rng=ts.rng_from(1)),
                   ts.seed_monolayer(400, 20, rng=ts.rng_from(1))):
        report = ts.section_report(layout)
        assert "NOT measured" in report["cell_types"]
        assert "10x Genomics" in report["attribution"]


def test_a_seeded_layout_is_reproducible_from_its_seed():
    a = ts.seed_section(500, 20, rng=ts.rng_from(7))
    b = ts.seed_section(500, 20, rng=ts.rng_from(7))
    assert np.array_equal(a["normal"], b["normal"])
    assert np.array_equal(a["cancer"], b["cancer"])
    c = ts.seed_section(500, 20, rng=ts.rng_from(8))
    assert not np.array_equal(a["normal"], c["normal"])


def test_an_empty_population_is_not_an_error():
    for layout in (ts.seed_section(0, 0), ts.seed_monolayer(0, 0)):
        assert layout["n_sites"] == 0
        assert len(layout["normal"]) == 0 and len(layout["cancer"]) == 0


def test_an_unknown_spec_key_is_rejected_rather_than_ignored():
    """A typo in a preset's ``section`` sub-dict must not silently seed
    the default geometry -- that is a run reporting a layout it did not
    use."""
    with pytest.raises(ValueError, match="unknown section spec keys"):
        ts.section_spec(windwo=240.0)


# ── Wiring: the layout has to reach the engine ───────────────────────

def test_the_section_preset_is_population_matched_to_the_slab():
    """section_variant is derived FROM slab_variant for this reason.

    Both specs hold 1,960 + 80, so a difference between a slab run and a
    section run is a difference in arrangement. If these diverge, every
    slab/section comparison in this file is comparing two things.
    """
    from cognisom.dashboard.engine_runner import (
        SCENARIOS, section_variant, slab_variant,
    )
    from cognisom.modules.tissue_slab import SLAB_SPEC

    assert (ts.SECTION_SPEC["n_normal"], ts.SECTION_SPEC["n_cancer"]) == (
        SLAB_SPEC["n_normal"], SLAB_SPEC["n_cancer"])

    for name in ("Baseline", "Aggressive Tumor", "Immunotherapy"):
        slab = slab_variant(SCENARIOS[name])
        section = section_variant(SCENARIOS[name])
        for key in ("n_normal_cells", "n_cancer_cells"):
            assert section["cellular"][key] == slab["cellular"][key], (
                "%s: section %s disagrees with slab" % (name, key))
        assert section["cellular"]["seeding"] == "section"
        assert section["vascular"]["layout"] == "section"
        assert section["immune"]["seeding"] == "section"
        # The slab's override sub-dict must not survive into a section
        # run: the key names are shared but the spec is not.
        for module in ("cellular", "vascular", "immune"):
            assert "slab" not in section.get(module, {})


def test_the_section_preset_sizes_the_spatial_grid():
    """Left unset, the fields would cover 200 um of a 260 um window.

    Nothing raises in that case -- the cells beyond the grid simply stop
    interacting with every field -- so the grid being set is asserted
    rather than assumed.
    """
    from cognisom.dashboard.engine_runner import SCENARIOS, section_variant

    cfg = section_variant(SCENARIOS["Baseline"])
    grid = cfg["spatial"]
    assert grid["grid_size"] == ts.section_grid()["grid_size"]
    covered = grid["grid_size"][0] * grid["resolution"]
    assert covered >= ts.DOMAIN_UM[0] - 1e-9


def test_the_lymphatic_ring_follows_the_sections_own_nodule():
    """The two layouts put their nodules in different places.

    Carrying the slab's ring centre into a section run would ring a
    tumour that is not there, which no assertion elsewhere would catch.
    """
    from cognisom.dashboard.engine_runner import (
        SCENARIOS, section_variant, slab_variant,
    )
    name = "Metastasis Risk"
    if "ring_centre_um" not in SCENARIOS[name].get("lymphatic", {}):
        pytest.skip("%s no longer hand-places a lymphatic ring" % name)
    slab = slab_variant(SCENARIOS[name])["lymphatic"]
    section = section_variant(SCENARIOS[name])["lymphatic"]
    assert section["ring_centre_um"] != slab["ring_centre_um"]
    layout = ts.seed_section(
        N_NORMAL, N_CANCER, nodule_offset=ts.DEFAULT_NODULE_OFFSET,
        rng=ts.rng_from(None))
    assert section["ring_centre_um"] == pytest.approx(
        tuple(float(v) for v in layout["nodule_center"]))


@pytest.mark.slow
def test_a_section_run_builds_and_perfuses_without_starving():
    """The risk the vessel lattice had to clear.

    The section's in-plane window is larger than the slab's (240 vs
    ~165 um) at the same vessel pitch, so the question was whether the
    pitch still reaches every cell. It does: measured over three seeds,
    the section's MINIMUM oxygen is slightly higher than the slab's
    (0.1273 vs 0.1249) and neither produces a hypoxic region.
    """
    from cognisom.dashboard.engine_runner import EngineRunner

    np.random.seed(1)
    runner = EngineRunner(dt=0.05, duration=0.5, scenario="Baseline",
                          tissue="section", record_interval=100000,
                          snapshot_interval_hours=None)
    with contextlib.redirect_stdout(io.StringIO()):
        runner.build()
        runner.run()

    cellular = runner.engine.modules["cellular"]
    vascular = runner.engine.modules["vascular"]
    spatial = runner.engine.modules["spatial"]
    alive = [c for c in cellular.cells.values() if c.alive]

    assert len(alive) >= N_NORMAL, "population collapsed on build"
    assert cellular.section_layout is not None
    assert cellular.section_layout["layer_fill"] == pytest.approx(0.588,
                                                                  abs=0.03)
    assert vascular.hypoxic_regions == 0, (
        "the section starved where the slab does not; the vessel pitch no "
        "longer reaches its larger in-plane window")
    assert min(c.oxygen for c in alive) > 0.05, "cells near anoxia"

    positions = np.array([c.position for c in alive])
    covered = np.array([n * spatial.resolution for n in spatial.grid_size])
    assert (positions >= 0).all() and (positions <= covered).all(), (
        "cells outside the diffusion grid, which is a silent failure")


@pytest.mark.slow
def test_a_section_run_is_less_connected_than_a_slab_run():
    """The claim, measured on real runs rather than on layouts alone.

    Both layouts are seeded, perfused and stepped by the same engine, so
    this is the form of the comparison that a result would actually be
    quoted from. Measured over seeds 1-3: slab 7.77 neighbours against
    the section's 5.04, at an identical 2,044 cells.
    """
    from cognisom.dashboard.engine_runner import EngineRunner

    means, counts = {}, {}
    for tissue in ("slab", "section"):
        np.random.seed(1)
        runner = EngineRunner(dt=0.05, duration=0.5, scenario="Baseline",
                              tissue=tissue, record_interval=100000,
                              snapshot_interval_hours=None)
        with contextlib.redirect_stdout(io.StringIO()):
            runner.build()
            runner.run()
        alive = [c for c in runner.engine.modules["cellular"].cells.values()
                 if c.alive]
        positions = np.array([c.position for c in alive])
        means[tissue] = _neighbour_counts(positions).mean()
        counts[tissue] = len(alive)

    assert counts["slab"] == counts["section"], (
        "the two runs seeded %d and %d cells, so their connectivity is not "
        "comparable" % (counts["slab"], counts["section"]))
    assert means["section"] < means["slab"]
    ratio = means["section"] / means["slab"]
    assert 0.5 < ratio < 0.85, (
        "section/slab neighbour ratio in a real run is %.3f, measured 0.65 "
        "-- re-measure, do not re-pin" % ratio)
