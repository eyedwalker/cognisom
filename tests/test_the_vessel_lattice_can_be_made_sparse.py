"""The engine can model a badly perfused tumour. No preset asks it to.

Why this file exists
--------------------
The methods page used to list "the simulated vasculature is denser than
a real tumour's" as a limitation whose remedy was a sparser vessel
lattice -- phrased as work not yet done. It is not work not yet done.
``vessel_pitch`` is already a key in ``section_spec``/``slab_spec``, and
``VascularModule`` passes its ``section``/``slab`` overrides straight
into them, so a run can ask for one today.

MEASURED 2026-11-13 on the section layout's own seeding (2,040 cells),
overriding ``vascular.section.vessel_pitch`` and nothing else. Five
global seeds per pitch, because the lattice jitters and the spread is
the point:

    pitch  vessels   worst cell->vessel, 5 seeds   spread
      22      35          14.8 - 15.8 um            1.0
      40      13          25.1 - 27.9 um            2.8
      60       5          45.6 - 47.1 um            1.6
      90       4          55.6 - 59.0 um            3.4
     120       3          72.1 - 79.9 um            7.8
     150       3          74.9 - 83.5 um            8.6
     180       2         113.2 - 142.7 um          29.6
     250       2         110.5 - 149.7 um          39.2
     400       2         104.6 - 165.2 um          60.6

At a 180 um pitch the worst cell lands in Primeau 2005's 90-140 um
band, where tumour cells saw no detectable doxorubicin. So the
capability is there, and the DEFAULT is what makes three mechanisms
inert.

THE TRAP, which is the real reason this file exists. ``vessel_jitter``
is a FRACTION of the pitch (0.12), so a sparse lattice jitters in
absolute terms by as much as the tissue is thick. At the 22 um default
the worst distance varies by 1 um across seeds; at 400 um it varies by
60 um. Past about a 180 um pitch the lattice is down to two vessels and
WHICH CELLS STARVE DEPENDS MORE ON THE SEED THAN ON THE PITCH. A sparse
preset therefore has to be reported across seeds, not quoted from one.

TWO CLAIMS MADE AND RETRACTED ON THE WAY HERE, kept because each would
have gone into a public page:

1. "Worst distance is not monotonic in pitch -- 180 um gives 134 um and
   400 um gives 89 um." FALSE. That compared two EngineRunner runs whose
   CELL LAYOUTS differed, and then two UNSEEDED lattices. Against one
   layout with the seed fixed it rises monotonically.
2. "It plateaus around 120-125 um past a 180 um pitch." FALSE for the
   same reason -- it keeps rising; what plateaus is the vessel COUNT,
   at two, while the variance keeps growing.

Both looked like measurements. Neither held a comparison fixed.

WHAT THIS FILE DELIBERATELY DOES NOT CLAIM. Being able to set poor
perfusion is not the same as the model being right there. The Krogh
calibration in ``_exchange_with_tissue`` was anchored at the 22 um
pitch, where every cell is inside a Krogh cylinder. A sparse lattice
runs it precisely where it was never checked. Nothing here says those
runs are trustworthy; it says the knob exists and does something, which
is the claim the methods page makes.
"""

from __future__ import annotations

import numpy as np
import pytest

from cognisom.modules.tissue_slab import DEFAULT_VESSEL_PITCH_UM
from cognisom.modules.vascular_module import VascularModule

#: Every lattice in this file is built after seeding the global stream.
#: ``rng_from(None)`` takes one draw from ``np.random``, which is how
#: this project fixes randomness, so an unseeded build is a DIFFERENT
#: lattice every call -- the thing that produced the two retracted
#: claims above.
SEED = 7


def _lattice(pitch=None, seed=SEED):
    """A section-layout vascular module, optionally at a given pitch."""
    np.random.seed(seed)
    section = {} if pitch is None else {"vessel_pitch": float(pitch)}
    module = VascularModule({"layout": "section", "section": section})
    module.initialize()
    caps = module.capillaries
    return list(caps.values() if isinstance(caps, dict) else caps)


@pytest.fixture(scope="module")
def cells():
    """The section layout's own cells, at a fixed seed.

    The CELLS and not the box: they occupy a subset of it, because a
    lumen bin declines its sites. Measuring the box instead was part of
    how the retracted claims above survived as long as they did.
    """
    from cognisom.modules.tissue_section import (
        rng_from, section_spec, seed_section,
    )

    spec = section_spec()
    np.random.seed(SEED)
    seeded = seed_section(
        spec["n_normal"], spec["n_cancer"], section=spec["section"],
        window=spec["window"], bin_um=spec["bin_um"],
        spacing=spec["spacing"], jitter=spec["jitter"],
        center=spec["center"], nodule_offset=spec["nodule_offset"],
        rng=rng_from(spec["seed"]))
    return np.vstack([np.asarray(seeded["normal"], float),
                      np.asarray(seeded["cancer"], float)])


def _worst_cell(caps, positions):
    """How far the worst-supplied cell sits from its nearest vessel."""
    return max(min(c.distance_to(p) for c in caps) for p in positions)


def test_the_default_pitch_is_the_calibrated_one():
    """The limitation is a DEFAULT, and this is the default."""
    from cognisom.modules.tissue_section import section_spec

    assert section_spec()["vessel_pitch"] == DEFAULT_VESSEL_PITCH_UM == 22.0


def test_the_lattice_is_reproducible_once_the_seed_is_fixed():
    """The property the rest of this file depends on.

    ``rng_from(None)`` draws from the global stream on purpose, so that
    a caller who seeded gets a reproducible lattice. Without that, every
    number in this file would be noise -- which is exactly how the two
    retracted claims in the docstring happened.
    """
    first = _lattice(180.0)
    second = _lattice(180.0)
    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert np.allclose(np.asarray(a.start, float),
                           np.asarray(b.start, float)), (
            "the vessel lattice is no longer reproducible under a fixed "
            "global seed, so no measurement of it can be trusted")


def test_the_override_is_honoured_at_all():
    """The whole claim in one assertion: asking for a sparser lattice
    gets a sparser lattice."""
    dense, sparse = _lattice(), _lattice(180.0)
    assert len(sparse) < len(dense), (
        f"a 180 um pitch laid {len(sparse)} vessels against {len(dense)} "
        "at the 22 um default, so the override is being ignored -- which "
        "is what the methods page says is NOT happening")


def test_a_sparse_lattice_reaches_primeaus_regime(cells):
    """The claim the methods page rests on, with its number beside it.

    The CLAIM is that a sparse lattice can put a cell as far from a
    vessel as Primeau 2005's 90-140 um band. The band is the assertion;
    133.3 um is what seed 7 measured at.
    """
    worst = _worst_cell(_lattice(180.0), cells)
    assert 90.0 <= worst <= 145.0, (
        f"a 180 um pitch leaves the worst cell {worst:.1f} um from a "
        "vessel at seed 7 (measured 133.3). The methods page says this "
        "reaches Primeau's 90-140 um band -- re-measure across seeds and "
        "update the page rather than widening this")


def test_the_default_lattice_leaves_nobody_far_from_a_vessel(cells):
    """The other half: why three mechanisms are inert as shipped."""
    worst = _worst_cell(_lattice(), cells)
    assert worst < 20.0, (
        f"the default lattice now leaves a cell {worst:.1f} um from a "
        "vessel, against 15.0 measured at seed 7. Drug penetration and "
        "the lactate gate may no longer be inert -- re-measure both "
        "before changing what the pages say")


def test_a_sparse_lattice_is_far_more_seed_sensitive_than_the_default(cells):
    """The trap, pinned: jitter is a FRACTION of the pitch.

    At the shipped pitch the geometry barely moves between seeds, so one
    run characterises it. At a sparse pitch it moves by tens of microns,
    so one run does not -- and a preset quoted from a single seed would
    be reporting that seed, not that pitch.
    """
    seeds = (1, 7, 13, 42, 99)

    def spread(pitch):
        w = [_worst_cell(_lattice(pitch, s), cells) for s in seeds]
        return max(w) - min(w)

    tight, loose = spread(None), spread(400.0)
    assert tight < 3.0, (
        f"the default lattice now varies by {tight:.1f} um across seeds, "
        "against 1.0 measured; the shipped geometry is supposed to be "
        "essentially seed-independent")
    assert loose > 20.0, (
        f"a 400 um pitch now varies by only {loose:.1f} um across seeds, "
        "against 60.6 measured. If the jitter stopped scaling with the "
        "pitch, say so on the methods page -- a sparse preset would no "
        "longer need reporting across seeds")
    assert loose > 5 * tight


@pytest.mark.parametrize("pitch", [22.0, 40.0, 60.0, 120.0, 180.0])
def test_every_pitch_in_the_published_table_still_builds(pitch):
    """The table on the methods page names these pitches. A pitch that
    raises is a table row that cannot be reproduced."""
    assert _lattice(pitch), f"a {pitch:g} um pitch laid no vessels at all"
