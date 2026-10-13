"""The TissueSlab scenario must seed tissue, and must not disturb the ring.

Every preset before this one seeds a RING: ``CellularModule._seed_ring``
puts each normal cell on a circle of radius 50 um about (100, 100, 50).
Measured on the Baseline preset, the normal cells span radius
50.00-50.97 um -- a "distribution" of two values -- with a 100 um hole in
the middle and a separate cancer cloud that touches nothing. That is why
the engine's renders have always looked like a bead necklace.

These tests pin the replacement and its blast radius:

* the slab is a VOLUME, with a radius distribution and a
  nearest-neighbour distribution rather than a pair of values;
* its cells are in contact, by the mechanics module's own contact and
  adhesion distances;
* the tumour nodule is solid, roughly spherical, and has stroma on every
  side of it;
* the capillary lattice actually perfuses it, judged against the
  vascular module's own Krogh calibration rather than by eye;
* and the ring, the radial vessel network and the window infiltrate that
  every other preset depends on are byte-for-byte unchanged.
"""

import io
from contextlib import redirect_stdout

import numpy as np
import pytest

from cognisom.core import SimulationConfig, SimulationEngine
from cognisom.dashboard.engine_runner import SCENARIOS
from cognisom.modules.cellular_module import CellularModule
from cognisom.modules.immune_module import ImmuneModule
from cognisom.modules.vascular_module import VascularModule
from cognisom.modules import tissue_slab as ts

# vascular_module: the zonation _exchange_with_tissue documents.
NORMOXIC_LIMIT_UM = 25.0
LETHAL_CORNER_UM = 50.0
O2_HYPOXIC = 0.05
O2_DEATH = 0.02
# The 200 x 200 x 100 um domain, from MorphogenModule.tissue_size and the
# default SpatialModule grid.
DOMAIN = np.array([200.0, 200.0, 100.0])


def _quiet(fn, *args, **kwargs):
    with redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _module(cls, config, seed=12345):
    np.random.seed(seed)
    mod = cls(config)
    _quiet(mod.initialize)
    return mod


def _slab_cells(seed=12345):
    cel = _module(CellularModule, {"seeding": "slab"}, seed=seed)
    normal = np.array([c.position for c in cel.cells.values()
                       if c.cell_type == "normal"], dtype=float)
    cancer = np.array([c.position for c in cel.cells.values()
                       if c.cell_type == "cancer"], dtype=float)
    return cel, normal, cancer


# ── The slab is a volume ─────────────────────────────────────────────

class TestTheSlabIsAVolume:
    def test_it_seeds_about_two_thousand_cells_inside_the_domain(self):
        cel, normal, cancer = _slab_cells()
        assert len(normal) == ts.SLAB_SPEC["n_normal"]
        assert len(cancer) == ts.SLAB_SPEC["n_cancer"]
        total = len(normal) + len(cancer)
        assert 1800 <= total <= 2200, f"{total} cells is not ~2,000"

        pos = np.vstack([normal, cancer])
        assert (pos > 0.0).all() and (pos < DOMAIN).all(), (
            "a cell is outside the 200 x 200 x 100 um domain, where the "
            "spatial field cannot register it as a sink"
        )
        extents = pos.max(axis=0) - pos.min(axis=0)
        # A slab, not a cube and not a sheet: wider than it is thick.
        assert extents[0] > 2.0 * extents[2]
        assert extents[1] > 2.0 * extents[2]
        assert extents[2] > 40.0, "too thin to be a volume"

    def test_the_radius_from_centre_is_a_distribution_not_two_values(self):
        _, normal, cancer = _slab_cells()
        centre = np.asarray(ts.SLAB_SPEC["center"], dtype=float)
        r = np.linalg.norm(np.vstack([normal, cancer]) - centre, axis=1)

        # The ring gives 11 distinct values at 0.1 um resolution and an
        # interquartile range under 1 um. A volume cannot.
        distinct = len(np.unique(np.round(r, 1)))
        assert distinct > 500, f"only {distinct} distinct radii: still a shell"
        iqr = float(np.percentile(r, 75) - np.percentile(r, 25))
        assert iqr > 25.0, f"interquartile radius range {iqr:.1f} um is a shell"
        # Cells all the way in to the centre, which the ring's 100 um hole
        # has none of.
        assert r.min() < 15.0, f"the slab has a {r.min():.0f} um hole in it"

    def test_the_ring_really_is_the_degenerate_case_this_replaces(self):
        """Guards the comparison the rest of this module is written against."""
        cel = _module(CellularModule, {"n_normal_cells": 80,
                                       "n_cancer_cells": 10})
        normal = np.array([c.position for c in cel.cells.values()
                           if c.cell_type == "normal"], dtype=float)
        r = np.linalg.norm(normal - np.array([100.0, 100.0, 50.0]), axis=1)
        assert r.max() - r.min() < 1.5, "the Baseline preset is not a ring"


# ── The cells touch ──────────────────────────────────────────────────

class TestThePackingIsContact:
    def test_every_cell_is_in_contact_with_a_neighbour(self):
        _, normal, cancer = _slab_cells()
        nn = ts.nearest_neighbour_distances(np.vstack([normal, cancer]))

        # The mechanics module's own numbers: contact at 2 * 5.0 um,
        # adhesion out to contact + 2.0 um. A nearest neighbour beyond
        # the cutoff is a cell adhering to nothing. Measured over six
        # seeds, at most one cell in 2,040 lands past it -- a surface
        # cell whose few neighbours all jittered away from it -- so the
        # bound is on the population, not on the single worst cell.
        stragglers = int((nn > ts.ADHESION_CUTOFF_UM).sum())
        assert stragglers <= 0.005 * len(nn), (
            f"{stragglers} cells have no neighbour inside the "
            f"{ts.ADHESION_CUTOFF_UM} um adhesion cutoff"
        )
        assert nn.max() < 13.5, (
            f"a cell's nearest neighbour is {nn.max():.2f} um away, "
            f"adrift of the tissue"
        )
        assert float(np.percentile(nn, 99)) <= ts.ADHESION_CUTOFF_UM
        # ... and not so compressed that cells have merged. A confluent
        # tissue has no interstitial space, so centre distances below
        # 2R are expected; half of 2R is not.
        assert nn.min() > 0.6 * ts.CONTACT_DISTANCE_UM, (
            f"tightest pair at {nn.min():.2f} um"
        )
        median = float(np.median(nn))
        assert 8.0 <= median <= ts.ADHESION_CUTOFF_UM, (
            f"median nearest-neighbour distance {median:.2f} um"
        )

    def test_the_lattice_is_jittered_rather_than_crystalline(self):
        _, normal, cancer = _slab_cells()
        nn = ts.nearest_neighbour_distances(np.vstack([normal, cancer]))
        # An unjittered close-packed lattice puts every cell at exactly
        # the lattice constant, which renders as a crystal.
        assert float(nn.std()) > 0.3, "the packing is an unjittered lattice"
        assert len(np.unique(np.round(nn, 2))) > 100

    def test_the_unjittered_lattice_is_exactly_close_packed(self):
        """The thing the jitter perturbs has to be right to begin with."""
        sites, extents = ts.packed_sites(2040, spacing=11.0, layers=8)
        nn = ts.nearest_neighbour_distances(sites)
        assert nn.min() == pytest.approx(11.0, abs=1e-9)
        assert nn.max() == pytest.approx(11.0, abs=1e-9)
        assert extents[0] > extents[2] and extents[1] > extents[2]


# ── The nodule ───────────────────────────────────────────────────────

class TestTheTumourNodule:
    def test_it_is_a_solid_mass_with_stroma_all_round(self):
        cel, normal, cancer = _slab_cells()
        layout = cel.slab_layout
        assert layout is not None, "slab seeding recorded no layout"
        nodule = np.asarray(layout["nodule_center"], dtype=float)
        radius = float(layout["nodule_radius"])

        d_cancer = np.linalg.norm(cancer - nodule, axis=1)
        assert d_cancer.max() == pytest.approx(radius)

        # Solid: no stroma inside the mass. A ball carved out of a
        # close-packed lattice is faceted rather than spherical, so a
        # handful of stromal cells can sit between the facets and the
        # circumscribed radius -- but none may be inside the body of it.
        d_normal = np.linalg.norm(normal - nodule, axis=1)
        assert d_normal.min() > 0.9 * radius, (
            f"a normal cell sits {d_normal.min():.1f} um from the nodule "
            f"centre, inside a mass of radius {radius:.1f} um"
        )
        intruders = int((d_normal < radius).sum())
        assert intruders < 0.15 * len(cancer), (
            f"{intruders} stromal cells inside the nodule's circumscribed "
            f"radius: the mass is not solid"
        )
        # Stroma on every side: the nodule is strictly inside the slab.
        lo, hi = layout["bounds"]
        assert (nodule - radius > lo + 1.0).all()
        assert (nodule + radius < hi - 1.0).all()

    def test_its_radius_follows_from_the_packing_density(self):
        """The radius is an output of the cell count, not a free knob."""
        cel, _, cancer = _slab_cells()
        layout = cel.slab_layout
        spacing = layout["spacing"]
        density = np.sqrt(2.0) / spacing ** 3          # close packing
        expected = (3.0 * len(cancer) / (4.0 * np.pi * density)) ** (1.0 / 3.0)
        assert layout["nodule_radius"] == pytest.approx(expected, rel=0.15)

    def test_it_sits_off_centre(self):
        cel, _, _ = _slab_cells()
        layout = cel.slab_layout
        offset = np.linalg.norm(
            np.asarray(layout["nodule_center"]) - np.asarray(layout["center"]))
        assert offset > 1.5 * layout["nodule_radius"], (
            "the nodule is on top of the slab centre, so there is no "
            "tumour/stroma boundary to look at"
        )


# ── Perfusion ────────────────────────────────────────────────────────

class TestTheSlabIsPerfused:
    def _network(self, seed=12345):
        cel, normal, cancer = _slab_cells(seed=seed)
        vas = _module(VascularModule, {"layout": "slab"}, seed=seed)
        return cel, np.vstack([normal, cancer]), vas

    def test_no_cell_is_outside_a_krogh_cylinder(self):
        _, pos, vas = self._network()
        segments = [(c.start, c.end) for c in vas.capillaries.values()]
        assert len(segments) > 10
        report = ts.perfusion_report(pos, segments)

        assert report["n_beyond_50um"] == 0, (
            f"{report['n_beyond_50um']} cells are past the lethal corner "
            f"and receive nothing at all"
        )
        assert report["n_beyond_25um"] == 0, (
            f"{report['n_beyond_25um']} cells are outside the normoxic "
            f"limit"
        )
        # Margin, not a near miss: the shipped lattice measures ~19.7 um
        # worst case against a 25 um limit.
        assert report["max"] < 22.0, f"worst cell at {report['max']:.1f} um"
        assert report["median"] < 12.0, (
            f"median cell-to-vessel distance {report['median']:.1f} um"
        )

    def test_the_slab_does_not_asphyxiate_at_steady_state(self):
        """The decisive check, and it needs no time stepping.

        ``VascularModule.get_O2_at_position`` is the same Krogh steady
        state ``_exchange_with_tissue`` integrates towards -- its
        docstring says so -- so the steady-state field over the seeded
        positions IS the fate of the cells, without paying for 480 steps
        over 2,000 cells.
        """
        _, pos, vas = self._network()
        o2 = np.array([vas.get_O2_at_position(p) for p in pos])
        assert (o2 > O2_DEATH).all(), (
            f"{int((o2 <= O2_DEATH).sum())} cells starve at steady state"
        )
        assert (o2 > O2_HYPOXIC).all(), (
            f"{int((o2 <= O2_HYPOXIC).sum())} cells are hypoxic at steady "
            f"state; min {o2.min():.4f}"
        )
        assert o2.mean() > 0.12, f"slab mean O2 {o2.mean():.4f}"

    def test_a_short_run_kills_nobody(self):
        """Integration check, on a slab shrunk to keep the suite quick.

        The vascular exchange is O(cells x vessels) in pure Python, so
        the full 2,040-cell slab costs ~0.3 s per step. The geometry that
        decides survival is covered at full size by the steady-state test
        above; this one checks that the modules agree once they are
        actually stepped.
        """
        small = {"slab": {"n_normal": 400, "n_cancer": 40}}
        np.random.seed(7)
        with redirect_stdout(io.StringIO()):
            engine = SimulationEngine(SimulationConfig(dt=0.05, duration=2.0))
            engine.register_module("cellular", CellularModule,
                                   dict(small, seeding="slab"))
            engine.register_module("vascular", VascularModule,
                                   dict(small, layout="slab"))
            engine.initialize()
            cel = engine.modules["cellular"]
            vas = engine.modules["vascular"]
            vas.set_cellular_module(cel)
            for _ in range(40):
                engine.step()

        alive = [c for c in cel.cells.values() if c.alive]
        assert cel.total_deaths == 0, f"{cel.total_deaths} cells died"
        assert vas.hypoxic_regions == 0, (
            f"{vas.hypoxic_regions} hypoxic cells in a perfused slab"
        )
        o2 = np.array([c.oxygen for c in alive])
        assert o2.min() > O2_HYPOXIC and o2.mean() > 0.12


# ── The three modules agree about where the tissue is ────────────────

class TestTheModulesAgree:
    def test_vessels_and_infiltrate_land_inside_the_slab(self):
        cel, normal, cancer = _slab_cells()
        vas = _module(VascularModule, {"layout": "slab"})
        imm = _module(ImmuneModule, {"seeding": "slab", "n_t_cells": 30,
                                     "n_nk_cells": 15, "n_macrophages": 15})
        lo, hi = cel.slab_layout["bounds"]

        immune = np.array([ic.position for ic in imm.immune_cells.values()],
                          dtype=float)
        assert len(immune) == 60
        assert (immune >= lo).all() and (immune <= hi).all(), (
            "immune cells were seeded outside the tissue"
        )
        # Spread, not clumped: the infiltrate has to cover the stroma.
        span = immune.max(axis=0) - immune.min(axis=0)
        assert (span > 0.6 * (hi - lo)).all(), (
            f"the infiltrate covers only {np.round(span / (hi - lo), 2)} "
            f"of the slab"
        )
        # The historical window is 100 x 100 x 20 um about the centre;
        # a slab infiltrate must reach outside it.
        assert immune[:, 2].min() < 40.0 or immune[:, 2].max() > 60.0

        # Vessels span the slab's long axis and stay inside it crosswise.
        axis = ts.SLAB_SPEC["vessel_axis"]
        for cap in vas.capillaries.values():
            for a in (0, 1, 2):
                if a == axis:
                    continue
                assert lo[a] - 6.0 <= cap.start[a] <= hi[a] + 6.0
            assert cap.start[axis] <= lo[axis] and cap.end[axis] >= hi[axis]

    def test_the_shared_spec_is_the_only_source_of_geometry(self):
        """A typo in a slab override must fail loudly, not seed defaults."""
        with pytest.raises(ValueError, match="unknown slab parameter"):
            ts.slab_spec(spaceing=12.0)
        assert ts.slab_spec()["spacing"] == ts.SLAB_SPEC["spacing"]
        assert ts.slab_spec(spacing=9.0)["spacing"] == 9.0

    def test_the_preset_is_wired_to_all_three_modules(self):
        preset = SCENARIOS["TissueSlab"]
        assert preset["cellular"]["seeding"] == "slab"
        assert preset["immune"]["seeding"] == "slab"
        assert preset["vascular"]["layout"] == "slab"
        # The counts belong to SLAB_SPEC, not to the preset: a copy here
        # would drift from the vessel lattice.
        assert "n_normal_cells" not in preset["cellular"]


# ── The presets that already shipped are untouched ───────────────────

class TestNoRegressionInTheRingPresets:
    def test_the_other_presets_do_not_opt_in(self):
        for name, preset in SCENARIOS.items():
            if name == "TissueSlab":
                continue
            assert "seeding" not in preset.get("cellular", {}), name
            assert "seeding" not in preset.get("immune", {}), name
            assert "layout" not in preset.get("vascular", {}), name

    def test_the_four_calibrated_presets_are_unchanged(self):
        """Baseline and Hypoxia carry the Krogh calibration's numbers."""
        assert SCENARIOS["Baseline"]["cellular"] == {
            "n_normal_cells": 80, "n_cancer_cells": 10}
        assert SCENARIOS["Baseline"]["vascular"] == {"n_capillaries": 8}
        assert SCENARIOS["Hypoxia"]["cellular"] == {
            "n_normal_cells": 60, "n_cancer_cells": 20}
        assert SCENARIOS["Hypoxia"]["vascular"] == {
            "n_capillaries": 3, "exchange_rate": 0.4}
        assert SCENARIOS["Immunotherapy"]["immune"]["checkpoint_block"] == 1.0
        assert SCENARIOS["Aggressive Tumor"]["cellular"][
            "division_time_cancer"] == 8.0

    def test_default_seeding_is_still_the_ring(self):
        cel = _module(CellularModule, {})
        normal = [c for c in cel.cells.values() if c.cell_type == "normal"]
        cancer = [c for c in cel.cells.values() if c.cell_type == "cancer"]
        assert len(normal) == 80 and len(cancer) == 20
        assert cel.slab_layout is None
        r = np.linalg.norm(
            np.array([c.position for c in normal]) - np.array([100., 100., 50.]),
            axis=1)
        assert r.min() == pytest.approx(50.0, abs=0.01)
        assert r.max() < 51.0

    def test_the_ring_seeding_is_bit_for_bit_what_it_was(self):
        """Same positions, so the same np.random draws in the same order."""
        expected = []
        np.random.seed(999)
        for i in range(80):
            angle = 2 * np.pi * i / 80
            expected.append([100 + 50 * np.cos(angle),
                             100 + 50 * np.sin(angle),
                             50 + np.random.uniform(-10, 10)])
        for _ in range(10):
            expected.append([120 + np.random.uniform(-15, 15),
                             120 + np.random.uniform(-15, 15),
                             50 + np.random.uniform(-5, 5)])

        cel = _module(CellularModule, {"n_normal_cells": 80,
                                       "n_cancer_cells": 10}, seed=999)
        got = np.array([c.position for c in cel.cells.values()], dtype=float)
        np.testing.assert_allclose(got, np.array(expected), rtol=0, atol=1e-4)

    def test_default_vascular_is_still_the_radial_network(self):
        vas = _module(VascularModule, {})
        assert len(vas.capillaries) == 8
        centre = np.array([100.0, 100.0, 50.0])
        for cap in vas.capillaries.values():
            assert np.linalg.norm(cap.start[:2] - centre[:2]) == pytest.approx(
                20.0, abs=1e-4)
            assert np.linalg.norm(cap.end[:2] - centre[:2]) == pytest.approx(
                80.0, abs=1e-4)

    def test_default_immune_is_still_the_window_scatter(self):
        expected = []
        np.random.seed(321)
        for _ in range(12 + 8 + 6):
            expected.append([np.random.uniform(50, 150),
                             np.random.uniform(50, 150),
                             np.random.uniform(40, 60)])
        imm = _module(ImmuneModule, {"n_t_cells": 12, "n_nk_cells": 8,
                                     "n_macrophages": 6}, seed=321)
        got = np.array([ic.position for ic in imm.immune_cells.values()],
                       dtype=float)
        np.testing.assert_allclose(got, np.array(expected), rtol=0, atol=1e-4)
        assert [ic.cell_type for ic in imm.immune_cells.values()] == (
            ["T_cell"] * 12 + ["NK_cell"] * 8 + ["macrophage"] * 6)
