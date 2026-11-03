#!/usr/bin/env python3
"""
Real Tissue Section Geometry
============================

Seeding geometry that takes its cell positions from a MEASURED human
prostate section instead of generating them.

WHY THIS EXISTS
---------------
:mod:`cognisom.modules.tissue_slab` replaced the ring with a
close-packed lattice, which was the right move -- a confluent slab is
far closer to tissue than a necklace of beads. But a lattice is still a
lattice: every cell has the same number of neighbours at the same
distance, the density is uniform everywhere, and there is no empty
space. Real prostate is nothing like that. It is mostly NOT cells: in
the section measured here, cells cover 28-38% of the section area, and
15-18% of 50 um subcells contain no cell at all, because the tissue is
built of glands around lumens with stroma between them.

That void structure is not cosmetic. Contact-dependent killing, oxygen
diffusion and immune traversal all depend on where the gaps are, and a
lattice has none. This module is the alternative: the arrangement comes
from a real section, so the density heterogeneity, the local
neighbour counts and the lumens are measured rather than assumed.

WHAT IS REAL HERE AND WHAT IS NOT
---------------------------------
This distinction is the whole point of the module, so it is stated
before anything else:

* REAL: the in-plane (x, y) position of every cell, and its segmented
  footprint diameter. Measured, 10x Genomics Xenium Prime 5K FFPE Human
  Prostate, CC BY 4.0. See ``cognisom/data/sections/``.
* NOT MEASURED: z. An FFPE section is ~5 um thick and a prostate cell is
  ~10 um across, so the section is ONE cell layer and the data is a 2D
  projection with no z coordinate at all. Each cell is given a z drawn
  uniformly within the section thickness about the domain mid-plane.
  That keeps cells inside the measured slice without inventing layers.
  A real 3D arrangement would need serial sections; we have one.
* NOT REAL: which cell is which TYPE. Cell types are assigned exactly as
  every other layout assigns them -- from the population prior -- and
  the tumour nodule is carved geometrically, by the same nearest-N rule
  :func:`~cognisom.modules.tissue_slab.seed_slab` uses. The fixture
  deliberately carries no cell-type column: typing cells from the
  panel's own marker genes and then consuming the result as a finding
  would be a round trip over our own markers, which is the error class
  recorded against the cell2sentence path. So this layout gives a real
  arrangement of cell LOCATIONS, and makes no claim about the spatial
  arrangement of cell TYPES. Do not report one.

THE NUMBERS ARE MEASURED FROM THE SECTION, NOT CHOSEN
-----------------------------------------------------
* THE WINDOW is 600 x 600 um. That is not a round number picked for
  looks: it is the window size whose typical cell count matches the
  slab's 2,040, so the two layouts are comparable at the same
  population. Measured over all non-overlapping windows of the full
  11.5 x 6.4 mm section: a 200 um window holds a median of 180 cells,
  400 um holds 668, 600 um holds 1,428, and 700 um holds 1,948. The
  shipped fixture window holds 2,118.
* WHICH window was chosen by a stated rule, not by eye: of the 30
  windows within 10% of 2,040 cells, the one whose median cell area is
  closest to the section-wide median (55.16 vs 55.41 um^2). It is
  therefore a typical window of about the right size, not the densest
  and not the prettiest.
* SPACING is reported as the MEASURED median in-plane nearest-neighbour
  distance of the cells that were actually seeded. It is a measurement,
  not a lattice constant, and nothing should treat it as a knob. Note
  that it reads SMALLER than the cell diameter: two cells at different
  depths within a 5 um section project close together, so a 2D
  projected separation is not a 3D centre-to-centre distance.
* THE DOMAIN is 600 x 600 x 30 um, which is 60 x 60 x 3 voxels at the
  spatial module's default 10 um resolution. The in-plane extent is the
  window; the 30 um z gives the cell layer one voxel above and one
  below it so that a field has somewhere to diffuse. This is a
  DIFFERENT domain from the slab's 200 x 200 x 100 um, so the spatial
  grid must be configured with it -- see :func:`section_grid` -- or the
  fields will not cover the cells.

TWO SEEDERS, AND WHY THE DEFAULT IS NOT THE LITERAL ONE
-------------------------------------------------------
:func:`seed_monolayer` places the measured positions as they were
measured: one flat layer. It is the literal section, and it is the
right choice for comparing against imaging. It is NOT the right choice
for driving the engine, and the reason was measured rather than
assumed. Against the lattice it would replace, at the same 2,040 cells:

    layout                     median NN   neighbours <=12 um   isolated
    slab (close-packed)          9.17 um        7.8              0
    monolayer (real positions)   7.61 um        2.8            121

Those numbers are artefacts of SECTIONING, not facts about tissue. A
5 um section shows only the neighbours whose footprint also crosses
that plane, so a cell's out-of-plane contacts are invisible and the
count is truncated; and two cells at different depths within the slice
project close together, so the apparent separation is compressed below
one cell diameter. The projection distorts contact distance and contact
count in opposite directions. Seeding it directly would hand the engine
systematically under-connected cells, and 121 of 2,040 adhering to
nothing -- a state ``tissue_slab``'s own docstring calls the error
condition it exists to avoid.

:func:`seed_section` is therefore the default. It keeps what a 2D
section measures RELIABLY -- the mesoscale density field, i.e. where
tissue is dense and where it is lumen -- and supplies the local packing
from the close-packed lattice, which is independently right: the
measured packed cell diameter puts the pitch at 9.8-11.5 um and the
engine runs at 11.0.

THE NUMBER THAT MADE IT WORK, AND WHAT IT SAYS ABOUT THE SLAB
-------------------------------------------------------------
The measured 2D density cannot be read as a 3D density -- doing that
implies a 13% cell volume fraction, i.e. 87% of prostate non-cellular,
which no histology supports. It is a PER-LAYER areal density. Measured
as a fraction of one layer's close-packed capacity it is strikingly
stable across window size:

    240 um window:  323 cells /  550 capacity = 0.588
    300 um window:  511 cells /  859 capacity = 0.595
    600 um window: 2118 cells / 3435 capacity = 0.617

So real prostate fills about **60%** of the sites a close packing
offers, and that is the same ~0.6 volume fraction the pitch
consistency check needed independently. Stacking layers at the measured
per-bin fill gives correct packing AND real heterogeneity:

    density-seeded volume        9.50 um        5.0             25 (1.3%)

**That 5.0 against the slab's 7.8 is the finding, not a shortfall.** The
slab is fully close-packed, so it has no empty space at all, while real
prostate is ~60% filled -- the slab therefore over-connects every cell
by about 1.6x. Contact-dependent biology on the slab is running against
more neighbours than tissue has.

Slab and section results are NOT interchangeable, and every
density-downstream number has to be re-measured rather than carried
across.
"""

import csv
import math
import os
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from cognisom.modules.tissue_slab import (
    DEFAULT_JITTER, DEFAULT_SPACING_UM, DEFAULT_VESSEL_PITCH_UM,
    capillary_lattice, packed_sites, rng_from,
)

# ── The fixture ──────────────────────────────────────────────────────

#: Directory holding committed section fixtures.
SECTION_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "sections",
)

#: The default section: one 600 um window of real human prostate.
DEFAULT_SECTION = "prostate_xenium_600um.csv"

#: Attribution. The fixture is CC BY 4.0 and reuse must credit it.
SECTION_ATTRIBUTION = (
    "10x Genomics, Xenium Prime 5K FFPE Human Prostate (CC BY 4.0)"
)

#: In-plane extent of the shipped fixture window, um.
SECTION_WINDOW_UM = 600.0

#: In-plane extent the VOLUME seeder uses, um. Smaller than the fixture
#: window on purpose: a 240 um window at the measured fill supplies
#: ~2,040 cells over 7-8 close-packed layers, which is the slab's
#: population in a genuine volume rather than a monolayer. It also
#: lands on a 24 x 24 x 8 voxel grid at the spatial module's 10 um
#: resolution -- 4,032 voxels against the slab's 4,000, so the section
#: costs the solver essentially what the slab costs.
VOLUME_WINDOW_UM = 240.0

#: Bin size the measured density field is read at, um. Coarse enough
#: that a bin holds several cells (so its fill fraction means
#: something), fine enough to resolve a gland from its lumen.
DENSITY_BIN_UM = 24.0

#: MEASURED fraction of a close-packed layer's sites that real prostate
#: actually fills. Stable across window size -- 0.588 at 240 um, 0.595
#: at 300 um, 0.617 at 600 um -- and equal to the ~0.6 packed volume
#: fraction the cell-diameter/pitch consistency check needs
#: independently. It is reported here as the headline difference from
#: the slab, which fills 1.00 by construction and so over-connects
#: every cell by about 1.6x.
MEASURED_LAYER_FILL = 0.60

#: FFPE section thickness, um -- the standard Xenium FFPE sectioning
#: thickness, which is what bounds the z a cell may be given. z itself
#: is NOT measured; see the module docstring.
SECTION_THICKNESS_UM = 5.0

#: Spatial-grid resolution the domain is expressed in, um per voxel.
#: Matches ``SpatialModule``'s default so the section does not silently
#: change the solver's voxel size as well as its extent.
SECTION_RESOLUTION_UM = 10.0

#: Section domain, um, for the SHIPPED ~2,040-cell presets only. The
#: depth a section needs is a function of the population -- density is
#: measured and only depth is free -- so a larger run has a larger
#: domain and must take it from :func:`section_domain` or from the
#: ``domain``/``grid`` keys the layout returns. In-plane it is the
#: window plus ONE VOXEL of margin on each side: the jitter is applied after the lattice is centred, so a cell
#: can sit up to ``jitter * spacing`` (1.54 um at the defaults) beyond
#: the window edge, and a domain sized to the window exactly would put
#: those cells at a negative coordinate -- outside the grid, with the
#: origin at a corner. The 80 um z holds eight close-packed layers and
#: their margin. Every extent divides into whole voxels at the spatial
#: module's 10 um resolution.
DOMAIN_UM = (
    VOLUME_WINDOW_UM + 2.0 * SECTION_RESOLUTION_UM,
    VOLUME_WINDOW_UM + 2.0 * SECTION_RESOLUTION_UM,
    # Eight layers for the shipped population. Stated rather than
    # computed, because computing it would read the fixture at import
    # time; a test pins it equal to section_domain(2040) so the two
    # cannot drift.
    70.0,
)

#: Centre of that domain.
DOMAIN_CENTER_UM = (
    0.5 * DOMAIN_UM[0], 0.5 * DOMAIN_UM[1], 0.5 * DOMAIN_UM[2],
)

#: Domain for the MONOLAYER seeder, which is flat and in-plane large:
#: the full fixture window, with three voxels of z so a field has
#: somewhere to diffuse above and below the single cell layer.
MONOLAYER_DOMAIN_UM = (SECTION_WINDOW_UM, SECTION_WINDOW_UM, 30.0)

#: Centre of the monolayer domain.
MONOLAYER_CENTER_UM = (
    0.5 * MONOLAYER_DOMAIN_UM[0],
    0.5 * MONOLAYER_DOMAIN_UM[1],
    0.5 * MONOLAYER_DOMAIN_UM[2],
)

#: Which segmentation methods bound the cell MEMBRANE. ``interior``
#: bounds 18S RNA, which stops short of the membrane and reads ~23%
#: smaller; ``expansion`` is a 5 um geometric dilation of the nucleus
#: and is not a measurement at all. Anything reading ``d_eq_um`` as a
#: cell size must stratify on this.
MEMBRANE_SEGMENTATION = ("boundary",)

#: Where the tumour nodule sits relative to the domain centre, um, for
#: the volume seeder's 240 um window. Offset in-plane only: the z extent
#: is only ~70 um, so displacing the nodule in z would push it through
#: a face.
DEFAULT_NODULE_OFFSET = (45.0, -35.0, 0.0)

#: The same, for the monolayer's 600 um window.
MONOLAYER_NODULE_OFFSET = (90.0, -70.0, 0.0)

#: Canonical section spec. Mirrors ``tissue_slab.SLAB_SPEC`` so that the
#: cellular and vascular modules can read one source of truth for where
#: the tissue is instead of keeping copies that drift.
SECTION_SPEC: Dict[str, Any] = {
    # Population. The slab's 2,040, so the two layouts are comparable
    # at the same cell count and a difference between them is a
    # difference in ARRANGEMENT rather than in how much tissue there is.
    "n_normal": 1960,
    "n_cancer": 80,
    # Which section, and how much of it.
    "section": DEFAULT_SECTION,
    "window": VOLUME_WINDOW_UM,
    "bin_um": DENSITY_BIN_UM,
    # Packing. The pitch is the slab's, because the measured packed cell
    # diameter puts it in the right band and the section changes
    # OCCUPANCY, not pitch.
    "spacing": DEFAULT_SPACING_UM,
    "jitter": DEFAULT_JITTER,
    # None means "the centre of the domain this population needs", which
    # is the only safe default because the depth depends on the count.
    "center": None,
    "nodule_offset": DEFAULT_NODULE_OFFSET,
    # Vasculature. Pitch is a length, so the slab's calibrated 22 um
    # carries over unchanged to a larger in-plane window.
    "vessel_pitch": DEFAULT_VESSEL_PITCH_UM,
    "vessel_axis": 1,
    "vessel_jitter": 0.12,
    "vessel_tilt": 0.06,
    # Immune scatter is inset from the faces by this much, so a
    # patrolling cell starts inside the tissue rather than on its skin.
    "immune_inset": 4.0,
    # None means "draw from the global numpy stream", so ``np.random.seed``
    # still makes a whole run reproducible. An int pins the layout.
    "seed": None,
}


def section_spec(**overrides: Any) -> Dict[str, Any]:
    """``SECTION_SPEC`` with ``overrides`` applied.

    Unknown keys are rejected rather than ignored, so a typo in a
    preset's ``section`` sub-dict fails loudly instead of silently
    seeding the default geometry.
    """
    unknown = set(overrides) - set(SECTION_SPEC)
    if unknown:
        raise ValueError(
            "unknown section spec keys: %s (known: %s)"
            % (sorted(unknown), sorted(SECTION_SPEC))
        )
    spec = dict(SECTION_SPEC)
    spec.update(overrides)
    return spec


def section_grid(
    domain: Sequence[float] = DOMAIN_UM,
    resolution: float = SECTION_RESOLUTION_UM,
) -> Dict[str, Any]:
    """The ``SpatialModule`` config this domain needs.

    The slab fits inside the spatial module's default 20 x 20 x 10 voxel
    grid and a section does not -- the volume domain is 24 x 24 x 8 and
    the monolayer's is 60 x 60 x 3. Without this the fields would cover
    only the first 200 um of the window and every cell beyond it would
    sit outside the grid, which is a silent failure rather than a loud
    one, so the config is returned here rather than left to a caller to
    remember.

    The resolution is deliberately the spatial module's own default, so
    that choosing a section changes the solver's EXTENT and not its
    voxel size: a layout should not quietly re-grade the diffusion grid
    as well.
    """
    nx, ny, nz = (max(1, int(round(float(d) / resolution))) for d in domain)
    return {"grid_size": (nx, ny, nz), "resolution": float(resolution)}


def section_domain(
    n_cells: int,
    *,
    section: str = DEFAULT_SECTION,
    window: float = VOLUME_WINDOW_UM,
    bin_um: float = DENSITY_BIN_UM,
    spacing: float = DEFAULT_SPACING_UM,
    jitter: float = DEFAULT_JITTER,
    resolution: float = SECTION_RESOLUTION_UM,
) -> Tuple[float, float, float]:
    """Domain extents, um, that hold ``n_cells`` at the measured density.

    The depth a section layout needs is a function of the POPULATION,
    because density is measured and only the depth is free: twice the
    cells is twice the tissue, not denser tissue. So the domain cannot
    be a constant, and :data:`DOMAIN_UM` is only the value this returns
    for the shipped ~2,040-cell presets. Asking for 3,000 cells needs
    11 layers and overflows that.

    Every extent is rounded UP to a whole voxel at ``resolution``, and
    carries one jitter amplitude of margin on each face, so the box is
    one the spatial grid can express exactly and one the jittered cells
    stay inside.
    """
    margin = float(jitter) * float(spacing)
    xy = window + 2.0 * float(resolution)
    layers = section_layers(max(0, int(n_cells)), section=section,
                            window=window, bin_um=bin_um, spacing=spacing)
    depth = max(0.0, layers - 1) * spacing * math.sqrt(2.0 / 3.0)
    z = (depth + 2.0 * margin) if layers else 0.0
    z = max(float(resolution),
            math.ceil(z / float(resolution)) * float(resolution))
    return (float(xy), float(xy), float(z))


def layer_capacity(
    window: float, spacing: float = DEFAULT_SPACING_UM,
) -> float:
    """How many cells ONE close-packed layer of ``window`` holds.

    A triangular lattice of pitch ``spacing`` has a row pitch of
    ``spacing * sqrt(3)/2``, so one layer offers
    ``window^2 / (spacing * row_pitch)`` sites. This is the denominator
    of :data:`MEASURED_LAYER_FILL`, which is why it is a named function
    rather than an expression inlined at the one call site: the fill
    fraction is only meaningful against a stated capacity.
    """
    if spacing <= 0.0:
        raise ValueError("spacing must be > 0, got %r" % (spacing,))
    return window * window / (spacing * spacing * math.sqrt(3.0) / 2.0)


# ── Loading ──────────────────────────────────────────────────────────

def section_path(section: str = DEFAULT_SECTION) -> str:
    """Absolute path of a committed section fixture."""
    if os.path.isabs(section):
        return section
    return os.path.join(SECTION_DIR, section)


def load_section(section: str = DEFAULT_SECTION) -> Dict[str, np.ndarray]:
    """Read a section fixture.

    Returns ``x``, ``y`` (um, origin at the window corner), ``d_eq``
    (um, the segmented footprint's equivalent circular diameter) and
    ``seg`` (which stain bounded the cell). Comment lines carrying the
    provenance are skipped; they are part of the file precisely so that
    the licence and the measurement's limits travel with the data.
    """
    path = section_path(section)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "no section fixture at %s. Committed fixtures live in %s"
            % (path, SECTION_DIR)
        )
    xs, ys, ds, sg = [], [], [], []
    with open(path, "r", newline="") as handle:
        rows = (line for line in handle if not line.startswith("#"))
        for row in csv.DictReader(rows):
            xs.append(float(row["x_um"]))
            ys.append(float(row["y_um"]))
            ds.append(float(row["d_eq_um"]))
            sg.append(row["seg"])
    if not xs:
        raise ValueError("section fixture %s holds no cells" % path)
    return {
        "x": np.asarray(xs, dtype=np.float64),
        "y": np.asarray(ys, dtype=np.float64),
        "d_eq": np.asarray(ds, dtype=np.float64),
        "seg": np.asarray(sg, dtype=object),
    }


def membrane_diameters(section: str = DEFAULT_SECTION) -> np.ndarray:
    """Footprint diameters of the MEMBRANE-segmented cells only.

    The one measurement in the fixture that is a cell size. Pooling all
    segmentation methods biases it low by ~14%, because the 18S interior
    stain bounds RNA rather than the membrane -- hence this accessor
    rather than letting callers reach for ``d_eq`` directly.
    """
    data = load_section(section)
    mask = np.isin(data["seg"].astype(str), MEMBRANE_SEGMENTATION)
    return data["d_eq"][mask]


# ── Seeding ──────────────────────────────────────────────────────────

def _centred_subwindow(
    x: np.ndarray, y: np.ndarray, n_keep: int,
) -> np.ndarray:
    """Indices of ``n_keep`` cells in a CONTIGUOUS centred sub-window.

    Taking a random subset would preserve the density statistics and
    destroy the thing this layout exists for: a cell's neighbours. Half
    of every cell's contacts would vanish and the gaps left behind would
    be indistinguishable from real lumens. Shrinking the window instead
    keeps every retained cell's neighbourhood exactly as it was
    measured, at the cost of a smaller field of view -- which is the
    honest trade.

    Cells are ranked by Chebyshev (square) distance from the window
    centre, so the retained region is a centred square like the window
    itself rather than a disc inscribed in it.
    """
    cx = 0.5 * (x.min() + x.max())
    cy = 0.5 * (y.min() + y.max())
    radius = np.maximum(np.abs(x - cx), np.abs(y - cy))
    return np.sort(np.argsort(radius, kind="stable")[:n_keep])


def density_field(
    window: float = VOLUME_WINDOW_UM,
    bin_um: float = DENSITY_BIN_UM,
    section: str = DEFAULT_SECTION,
) -> Tuple[np.ndarray, int]:
    """Measured cell count per in-plane bin, over a ``window`` corner.

    Returns ``(counts, n_bins)`` with ``counts`` flattened row-major on
    an ``n_bins x n_bins`` grid. This is the one thing a 2D section
    measures that survives into 3D unharmed: how much tissue sits where,
    as opposed to which individual cell sits where.
    """
    data = load_section(section)
    x, y = data["x"], data["y"]
    inside = (x < window) & (y < window)
    if not inside.any():
        raise ValueError(
            "no cells in the first %.0f x %.0f um of %s"
            % (window, window, section)
        )
    n_bins = max(1, int(round(window / bin_um)))
    ix = np.clip((x[inside] / bin_um).astype(int), 0, n_bins - 1)
    iy = np.clip((y[inside] / bin_um).astype(int), 0, n_bins - 1)
    counts = np.zeros(n_bins * n_bins, dtype=np.float64)
    np.add.at(counts, ix * n_bins + iy, 1.0)
    return counts, n_bins


def measured_layer_fill(
    window: float = VOLUME_WINDOW_UM,
    spacing: float = DEFAULT_SPACING_UM,
    section: str = DEFAULT_SECTION,
) -> float:
    """Measured cells as a fraction of one layer's packing capacity.

    The provenance of :data:`MEASURED_LAYER_FILL`, computed rather than
    quoted so that a different window or fixture reports its own number
    instead of inheriting this one. Stable at 0.588-0.617 across 240,
    300 and 600 um windows of the shipped section.
    """
    counts, _ = density_field(window=window, bin_um=window, section=section)
    return float(counts.sum() / layer_capacity(window, spacing))


def seed_section(
    n_normal: int,
    n_cancer: int,
    *,
    section: str = DEFAULT_SECTION,
    window: float = VOLUME_WINDOW_UM,
    bin_um: float = DENSITY_BIN_UM,
    spacing: float = DEFAULT_SPACING_UM,
    jitter: float = DEFAULT_JITTER,
    center: Optional[Sequence[float]] = None,
    nodule_offset: Sequence[float] = DEFAULT_NODULE_OFFSET,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Lay out a tissue VOLUME at the section's measured density field.

    The default section layout, and the one to use for a simulation.
    Each close-packed layer is filled to the fill fraction measured in
    that in-plane bin, so:

    * where the section shows packed glandular epithelium, the lattice
      fills and cells have their full complement of neighbours;
    * where it shows lumen, the lattice stays EMPTY -- the void is
      measured, not jittered into existence;
    * the local packing is the lattice's, which the measured packed cell
      diameter independently puts at the right pitch.

    The measured density is a PER-LAYER areal density, not a volumetric
    one. Treating it as volumetric implies a 13% cell volume fraction,
    which no histology supports; treating it per layer gives ~0.60, the
    same figure the diameter/pitch check arrives at independently. That
    distinction is the whole reason this function exists in the shape it
    does, so it is not a detail to optimise away.

    ``layers`` is not an argument: it is derived from the population, so
    the volume is whatever depth the requested cells occupy at the
    measured density. Asking for more cells makes the tissue deeper, not
    denser -- density is measured and is not a knob.

    Returns the same contract as
    :func:`~cognisom.modules.tissue_slab.seed_slab`.
    """
    if n_normal < 0 or n_cancer < 0:
        raise ValueError("cell counts must be >= 0")
    rng = np.random.default_rng() if rng is None else rng
    n_total = n_normal + n_cancer

    # The domain follows the population, so unless a caller pins the
    # centre we take the centre of the box THIS population needs. A
    # fixed centre would put a deep layout's lowest layer at a negative
    # z without anything complaining.
    domain = section_domain(n_total, section=section, window=window,
                            bin_um=bin_um, spacing=spacing, jitter=jitter)
    if center is None:
        center = tuple(0.5 * d for d in domain)
    center = np.asarray(center, dtype=np.float64)
    nodule_center = center + np.asarray(nodule_offset, dtype=np.float64)

    if n_total == 0:
        empty = np.zeros((0, 3), dtype=np.float64)
        return {
            "normal": empty, "cancer": empty.copy(),
            "center": center, "extents": np.zeros(3),
            "bounds": (center.copy(), center.copy()),
            "nodule_center": nodule_center, "nodule_radius": 0.0,
            "spacing": float(spacing), "layers": 0, "n_sites": 0,
            "section": section, "attribution": SECTION_ATTRIBUTION,
            "layer_fill": 0.0, "window": float(window),
            "domain": domain, "grid": section_grid(domain),
        }

    counts, n_bins = density_field(window=window, bin_um=bin_um,
                                   section=section)
    # Per-bin fill fraction: measured cells over this bin's ONE-layer
    # capacity, capped at 1.0 because a bin cannot be more than packed.
    bin_capacity = layer_capacity(bin_um, spacing)
    fill = np.minimum(counts / bin_capacity, 1.0)
    if fill.sum() <= 0.0:
        raise ValueError("measured density field is empty for %s" % section)

    # Depth needed to place n_total at the measured fill, plus a layer
    # of slack so the trim below always has something to trim.
    per_layer = fill.sum() * bin_capacity
    layers = max(1, int(math.ceil(n_total / max(per_layer, 1.0))) + 1)

    sites, _ = packed_sites(
        int(layer_capacity(window, spacing) * layers) + layers,
        spacing=spacing, layers=layers,
    )
    sites = sites - sites.min(axis=0)
    sites = sites[(sites[:, 0] < window) & (sites[:, 1] < window)]
    if not len(sites):
        raise ValueError(
            "no lattice sites inside a %.0f um window at %.1f um spacing"
            % (window, spacing)
        )

    bx = np.clip((sites[:, 0] / bin_um).astype(int), 0, n_bins - 1)
    by = np.clip((sites[:, 1] / bin_um).astype(int), 0, n_bins - 1)
    site_bin = bx * n_bins + by
    # Layer index from the z coordinate, so the fill is applied layer by
    # layer rather than smeared through the depth -- smearing it is what
    # breaks the packing, because every layer then gets a thin scatter
    # instead of some layers being solid.
    z_levels = np.unique(np.round(sites[:, 2], 6))
    site_layer = np.searchsorted(z_levels, np.round(sites[:, 2], 6))

    keep: list = []
    for bin_idx in np.flatnonzero(fill):
        for layer in range(len(z_levels)):
            idx = np.flatnonzero((site_bin == bin_idx) & (site_layer == layer))
            if not len(idx):
                continue
            take = int(round(fill[bin_idx] * len(idx)))
            if take <= 0:
                continue
            keep.append(idx if take >= len(idx)
                        else rng.choice(idx, size=take, replace=False))
    if not keep:
        raise ValueError("density field selected no sites")
    chosen = np.sort(np.concatenate(keep))
    placed = sites[chosen]

    # Trim the surplus from the top layer, which keeps the tissue a
    # solid block with one ragged face rather than thinning every layer.
    if len(placed) > n_total:
        order = np.lexsort((placed[:, 1], placed[:, 0], -placed[:, 2]))
        placed = placed[np.sort(order[len(placed) - n_total:])]

    placed = placed + center - 0.5 * (placed.min(axis=0) + placed.max(axis=0))
    amplitude = float(jitter) * float(spacing)
    if amplitude > 0.0:
        placed = placed + rng.uniform(-amplitude, amplitude,
                                      size=placed.shape)

    if n_cancer > 0:
        d_nodule = np.linalg.norm(placed - nodule_center, axis=1)
        cancer_idx = np.argsort(d_nodule, kind="stable")[:n_cancer]
    else:
        cancer_idx = np.zeros(0, dtype=np.int64)
    stroma_mask = np.ones(len(placed), dtype=bool)
    stroma_mask[cancer_idx] = False

    normal = placed[np.flatnonzero(stroma_mask)]
    cancer = placed[np.sort(cancer_idx)]
    nodule_radius = (
        float(np.linalg.norm(cancer - nodule_center, axis=1).max())
        if len(cancer) else 0.0
    )
    occupied = np.vstack([normal, cancer]) if len(cancer) else normal
    lo, hi = occupied.min(axis=0), occupied.max(axis=0)

    return {
        "normal": normal,
        "cancer": cancer,
        "center": center,
        "extents": hi - lo,
        "bounds": (lo, hi),
        "nodule_center": nodule_center,
        "nodule_radius": nodule_radius,
        # A real lattice constant here, unlike the monolayer's, because
        # the local packing IS a lattice -- only its occupancy is
        # measured.
        "spacing": float(spacing),
        "layers": int(len(z_levels)),
        "n_sites": int(len(sites)),
        "section": section,
        "attribution": SECTION_ATTRIBUTION,
        # The measured occupancy this layout was built at, carried so a
        # run can report it instead of implying full packing.
        "layer_fill": float(counts.sum() / layer_capacity(window, spacing)),
        "window": float(window),
        # The domain this population needs and the spatial grid that
        # expresses it. Carried on the layout rather than left to a
        # constant, because both depend on the cell count -- a caller
        # reading DOMAIN_UM for a 3,000-cell run would size the grid too
        # shallow and silently drop the top layers out of every field.
        "domain": domain,
        "grid": section_grid(domain),
    }


def seed_monolayer(
    n_normal: int,
    n_cancer: int,
    *,
    section: str = DEFAULT_SECTION,
    thickness: float = SECTION_THICKNESS_UM,
    center: Sequence[float] = MONOLAYER_CENTER_UM,
    nodule_offset: Sequence[float] = MONOLAYER_NODULE_OFFSET,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Lay out the measured cell positions AS MEASURED: one flat layer.

    This is the literal section, and it is the right thing for
    comparing against imaging -- every cell is where it was observed.
    It is NOT the engine default, because sectioning truncates each
    cell's neighbourhood and 2D projection compresses the apparent
    distances; see the module docstring for the measured numbers and
    :func:`seed_section` for the layout that avoids both. Reach for this
    when you want the section, not when you want tissue.

    ``n_normal + n_cancer`` cells are taken from the fixture. If that is
    fewer than the fixture holds, a contiguous centred sub-window is
    used (see :func:`_centred_subwindow`). If it is MORE, this raises:
    there is no honest way to invent cells that were not measured, and
    padding the shortfall with lattice sites would quietly turn a
    measured arrangement back into a generated one.

    The nodule is carved exactly as the slab carves it -- the
    ``n_cancer`` cells nearest ``center + nodule_offset`` become the
    tumour and leave the stromal pool -- so the mass is solid and its
    radius is an output of the cell count and the local density rather
    than a free parameter. Those positions are real; the decision that
    the cells THERE are the tumour is not a measurement.

    Returns the same contract as
    :func:`~cognisom.modules.tissue_slab.seed_slab`, so the vascular,
    immune and spatial modules need no special case: ``normal`` and
    ``cancer`` ((N, 3) float arrays), ``center``, ``extents``,
    ``bounds``, ``nodule_center``, ``nodule_radius``, ``spacing``,
    ``layers`` and ``n_sites``. Two of those keys mean something
    different here and are documented at the point of return.
    """
    if n_normal < 0 or n_cancer < 0:
        raise ValueError("cell counts must be >= 0")
    rng = np.random.default_rng() if rng is None else rng
    center = np.asarray(center, dtype=np.float64)
    nodule_center = center + np.asarray(nodule_offset, dtype=np.float64)
    n_total = n_normal + n_cancer

    if n_total == 0:
        empty = np.zeros((0, 3), dtype=np.float64)
        return {
            "normal": empty, "cancer": empty.copy(),
            "center": center, "extents": np.zeros(3),
            "bounds": (center.copy(), center.copy()),
            "nodule_center": nodule_center, "nodule_radius": 0.0,
            "spacing": 0.0, "layers": 1, "n_sites": 0,
            "section": section, "attribution": SECTION_ATTRIBUTION,
        }

    data = load_section(section)
    x, y = data["x"], data["y"]
    available = len(x)
    if n_total > available:
        raise ValueError(
            "section %s holds %d cells but %d were requested (%d normal + "
            "%d cancer). Cells that were not measured cannot be invented: "
            "either lower the population, or extract a larger window from "
            "the source section."
            % (section, available, n_total, n_normal, n_cancer)
        )
    if n_total < available:
        keep = _centred_subwindow(x, y, n_total)
        x, y = x[keep], y[keep]

    # Re-centre the measured window on the domain centre in plane, and
    # give each cell a z inside the section thickness. z is UNMEASURED
    # -- see the module docstring -- so it is drawn, not derived.
    x = x - 0.5 * (x.min() + x.max()) + center[0]
    y = y - 0.5 * (y.min() + y.max()) + center[1]
    half_t = 0.5 * float(thickness)
    z = center[2] + rng.uniform(-half_t, half_t, size=len(x))
    sites = np.column_stack([x, y, z])

    if n_cancer > 0:
        d_nodule = np.linalg.norm(sites - nodule_center, axis=1)
        cancer_idx = np.argsort(d_nodule, kind="stable")[:n_cancer]
    else:
        cancer_idx = np.zeros(0, dtype=np.int64)
    stroma_mask = np.ones(len(sites), dtype=bool)
    stroma_mask[cancer_idx] = False

    normal = sites[np.flatnonzero(stroma_mask)]
    cancer = sites[np.sort(cancer_idx)]

    nodule_radius = (
        float(np.linalg.norm(cancer - nodule_center, axis=1).max())
        if len(cancer) else 0.0
    )
    occupied = np.vstack([normal, cancer]) if len(cancer) else normal
    lo, hi = occupied.min(axis=0), occupied.max(axis=0)

    return {
        "normal": normal,
        "cancer": cancer,
        "center": center,
        "extents": hi - lo,
        "bounds": (lo, hi),
        "nodule_center": nodule_center,
        "nodule_radius": nodule_radius,
        # MEASURED median in-plane nearest-neighbour distance of the
        # cells actually seeded -- not a lattice constant, and not a
        # knob. It reads smaller than a cell diameter because a 2D
        # projected separation is not a 3D centre-to-centre distance.
        "spacing": float(median_nn_distance(occupied)),
        # A section is one cell layer. This is not a free parameter.
        "layers": 1,
        "n_sites": int(len(sites)),
        "section": section,
        "attribution": SECTION_ATTRIBUTION,
    }


def section_layers(
    n_cells: int,
    *,
    section: str = DEFAULT_SECTION,
    window: float = VOLUME_WINDOW_UM,
    bin_um: float = DENSITY_BIN_UM,
    spacing: float = DEFAULT_SPACING_UM,
) -> int:
    """How many close-packed layers ``n_cells`` occupy at measured fill.

    Deterministic -- it depends on the measured density field and the
    population, never on the rng -- which is what lets
    :func:`section_bounds` state the box before anything is seeded.
    Kept as one function so :func:`seed_section` and
    :func:`section_bounds` cannot derive different depths.
    """
    if n_cells <= 0:
        return 0
    counts, _ = density_field(window=window, bin_um=bin_um, section=section)
    bin_capacity = layer_capacity(bin_um, spacing)
    per_layer = np.minimum(counts / bin_capacity, 1.0).sum() * bin_capacity
    return max(1, int(math.ceil(n_cells / max(per_layer, 1.0))) + 1)


def section_bounds(
    n_normal: int,
    n_cancer: int,
    *,
    section: str = DEFAULT_SECTION,
    window: float = VOLUME_WINDOW_UM,
    bin_um: float = DENSITY_BIN_UM,
    spacing: float = DEFAULT_SPACING_UM,
    jitter: float = DEFAULT_JITTER,
    center: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """The volume layout's (lo, hi) box, without seeding anything.

    The vascular and immune modules are initialised independently of the
    cellular one and all three must agree on where the tissue is, so the
    box has to be derivable without paying for a seed.

    This is the NOMINAL box -- the window by the depth the population
    occupies -- and so a superset of the cells actually placed, because
    which sites a lumen bin declines is an rng draw. A superset is the
    correct answer for the callers: a vessel lattice must cover the
    tissue's extent, and sizing it to the realised minimum and maximum
    would leave the outermost cells beyond the vasculature.
    """
    n_total = n_normal + n_cancer
    if center is None:
        center = tuple(
            0.5 * d for d in section_domain(
                n_total, section=section, window=window, bin_um=bin_um,
                spacing=spacing, jitter=jitter)
        )
    center = np.asarray(center, dtype=np.float64)
    if n_total == 0:
        return center.copy(), center.copy()

    layers = section_layers(n_total, section=section, window=window,
                            bin_um=bin_um, spacing=spacing)
    # The jitter is applied AFTER the lattice is centred, so a cell can
    # sit up to one amplitude beyond the window edge and the deepest
    # layer. Leaving it out made this box a near-superset that excluded
    # a handful of cells, which is the kind of off-by-a-margin that
    # surfaces much later as an unperfused rim.
    margin = float(jitter) * float(spacing)
    half_xy = 0.5 * float(window) + margin
    # Matches packed_sites' ABAB stacking: z pitch is spacing*sqrt(2/3).
    depth = max(0.0, (layers - 1)) * spacing * math.sqrt(2.0 / 3.0)
    half_z = 0.5 * depth + margin
    lo = np.array([center[0] - half_xy, center[1] - half_xy,
                   center[2] - half_z])
    hi = np.array([center[0] + half_xy, center[1] + half_xy,
                   center[2] + half_z])
    return lo, hi


def monolayer_bounds(
    n_normal: int,
    n_cancer: int,
    *,
    section: str = DEFAULT_SECTION,
    thickness: float = SECTION_THICKNESS_UM,
    center: Sequence[float] = MONOLAYER_CENTER_UM,
) -> Tuple[np.ndarray, np.ndarray]:
    """The monolayer layout's (lo, hi) box, without seeding anything.

    Deterministic because the in-plane extent is read from the fixture
    and z is bounded by the section thickness whatever is drawn.
    """
    n_total = n_normal + n_cancer
    center = np.asarray(center, dtype=np.float64)
    if n_total == 0:
        return center.copy(), center.copy()

    data = load_section(section)
    x, y = data["x"], data["y"]
    if n_total > len(x):
        raise ValueError(
            "section %s holds %d cells but %d were requested"
            % (section, len(x), n_total)
        )
    if n_total < len(x):
        keep = _centred_subwindow(x, y, n_total)
        x, y = x[keep], y[keep]

    half_x = 0.5 * (x.max() - x.min())
    half_y = 0.5 * (y.max() - y.min())
    half_t = 0.5 * float(thickness)
    lo = np.array([center[0] - half_x, center[1] - half_y, center[2] - half_t])
    hi = np.array([center[0] + half_x, center[1] + half_y, center[2] + half_t])
    return lo, hi


# ── Measurement ──────────────────────────────────────────────────────

def median_nn_distance(positions: np.ndarray) -> float:
    """Median nearest-neighbour distance of ``positions``, um.

    Uses the full 3D separation of the seeded cells, which for a section
    is dominated by the in-plane component because the z spread is only
    the section thickness.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if len(positions) < 2:
        return 0.0
    from scipy.spatial import cKDTree
    dist, _ = cKDTree(positions).query(positions, k=2)
    return float(np.median(dist[:, 1]))


def void_fraction(
    positions: np.ndarray, subcell_um: float = 50.0,
) -> float:
    """Fraction of ``subcell_um`` in-plane subcells holding no cell.

    The quantity a lattice cannot have: on a close-packed slab it is
    zero by construction, and in the shipped section it is 15-18%. It is
    reported so that a run can state how much of its domain is lumen
    and stroma rather than leaving "real arrangement" as an adjective.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if len(positions) == 0:
        return 0.0
    x, y = positions[:, 0], positions[:, 1]
    nx = max(1, int(math.ceil((x.max() - x.min()) / subcell_um)))
    ny = max(1, int(math.ceil((y.max() - y.min()) / subcell_um)))
    ix = np.clip(((x - x.min()) / subcell_um).astype(int), 0, nx - 1)
    iy = np.clip(((y - y.min()) / subcell_um).astype(int), 0, ny - 1)
    occupied = len(np.unique(ix * ny + iy))
    return 1.0 - occupied / float(nx * ny)


def section_report(layout: Dict[str, Any]) -> Dict[str, Any]:
    """What this layout actually is, for a run to print or record.

    Deliberately includes ``attribution``: the positions are CC BY 4.0
    and a run that reports its geometry should carry the credit with it.
    """
    occupied = (
        np.vstack([layout["normal"], layout["cancer"]])
        if len(layout["cancer"]) else layout["normal"]
    )
    lo, hi = layout["bounds"]
    monolayer = int(layout["layers"]) <= 1
    report = {
        "n_cells": int(len(occupied)),
        "n_normal": int(len(layout["normal"])),
        "n_cancer": int(len(layout["cancer"])),
        "extent_um": tuple(round(float(v), 1) for v in (hi - lo)),
        "median_nn_um": round(float(median_nn_distance(occupied)), 2),
        # Reported at 20 um, not 50: at 50 um every subcell of the
        # shipped window holds a cell, so a 50 um void fraction reads
        # 0.0 and says nothing. The voids are a ~10-20 um feature.
        "void_fraction_20um": round(float(void_fraction(occupied, 20.0)), 3),
        "nodule_radius_um": round(float(layout["nodule_radius"]), 1),
        "layers": int(layout["layers"]),
        # Saying what is measured and what is not, every time, so a
        # consumer of this dict cannot acquire a claim by omission.
        "density": "measured",
        "cell_types": "population prior, NOT measured in this section",
        "section": layout.get("section"),
        "attribution": layout.get("attribution", SECTION_ATTRIBUTION),
    }
    if monolayer:
        report["positions"] = "measured (as sectioned)"
        report["z"] = "unmeasured (drawn within the section thickness)"
        report["caveat"] = (
            "a section truncates each cell's neighbourhood and compresses "
            "projected separations; not comparable to a volume layout"
        )
    else:
        report["positions"] = "lattice, occupancy from the measured density"
        report["layer_fill"] = round(float(layout.get("layer_fill", 0.0)), 3)
        report["window_um"] = round(float(layout.get("window", 0.0)), 1)
    return report


__all__ = [
    "DEFAULT_SECTION", "DENSITY_BIN_UM", "DOMAIN_CENTER_UM", "DOMAIN_UM",
    "MEASURED_LAYER_FILL", "MONOLAYER_CENTER_UM", "MONOLAYER_DOMAIN_UM",
    "SECTION_ATTRIBUTION", "SECTION_DIR", "SECTION_SPEC",
    "SECTION_THICKNESS_UM", "SECTION_WINDOW_UM", "VOLUME_WINDOW_UM",
    "density_field", "layer_capacity", "section_domain", "load_section", "measured_layer_fill",
    "median_nn_distance", "membrane_diameters", "monolayer_bounds", "rng_from",
    "section_bounds", "section_grid", "section_layers", "section_path",
    "section_report", "section_spec", "seed_monolayer", "seed_section",
    "void_fraction",
]
