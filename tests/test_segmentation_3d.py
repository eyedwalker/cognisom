"""The segmenter has to survive 3-D input, and must not invent shape descriptors.

`cognisom/imaging` is built for z-stacks: the importers read OME-TIFF and CZI
volumes, and `MeshGenerator.labels_to_mesh` documents `(D, H, W)` input. The
segmenter in the middle could not run on one. `_compute_properties` read
`region.eccentricity` behind a `hasattr` guard, and that guard never fires:
a `RegionProperties` object HAS the attribute in 3-D -- *accessing* it is what
raises `NotImplementedError`. So every 3-D stack died inside the property loop,
and the whole path from a microscope to a simulation mesh was unreachable.

The second half matters as much as the first. The obvious repair is to fall
back to the dataclass defaults, 0.0 for eccentricity and 1.0 for solidity. Those
are not neutral values: 0.0 eccentricity reads as a perfect circle and 1.0
solidity as a perfectly convex object, so a reader plotting them would get a
flawlessly round, flawlessly solid population that was never measured. The
module already makes this argument for the missing-scikit-image path, where it
logs that the defaults "are NOT measurements of anything". NaN carries the same
meaning without needing the log line to be read.
"""
from __future__ import annotations

import numpy as np
import pytest

from cognisom.imaging.segmentation import CellSegmenter


def _two_blobs_3d(shape=(24, 64, 64)):
    """Two separated spheroids, the simplest 3-D case with more than one object."""
    vol = np.zeros(shape, dtype=np.uint16)
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    for cz, cy, cx, r in ((12, 20, 20, 8), (12, 44, 44, 9)):
        m = ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) <= r * r
        vol[m] = 4000
    return vol


def test_a_3d_stack_segments_at_all():
    """The regression: this raised NotImplementedError before reaching a result."""
    seg = CellSegmenter(method="watershed", min_size=50, max_size=10_000_000)

    result = seg.segment(_two_blobs_3d())

    assert result.labels.ndim == 3
    assert result.n_cells >= 1, "nothing segmented from two clearly separated blobs"


def test_three_dimensional_shape_descriptors_are_nan_not_flattering_defaults():
    """eccentricity is undefined in 3-D, so it must not come back as 0.0.

    0.0 is the value for a perfect circle. Returning it for every object would
    put a column of perfectly round cells into any downstream analysis.
    """
    seg = CellSegmenter(method="watershed", min_size=50, max_size=10_000_000)

    result = seg.segment(_two_blobs_3d())

    assert result.cell_properties, "no per-object properties returned"
    for p in result.cell_properties:
        assert np.isnan(p.eccentricity), (
            f"eccentricity came back as {p.eccentricity!r} for a 3-D object; "
            "0.0 would be indistinguishable from a measured perfect circle"
        )


def test_quantities_that_are_defined_in_3d_are_still_measured():
    """NaN is for what cannot be computed, not a blanket retreat.

    Area (voxel count) and centroid are perfectly well defined on a volume and
    must survive the same code path.
    """
    seg = CellSegmenter(method="watershed", min_size=50, max_size=10_000_000)

    result = seg.segment(_two_blobs_3d())

    for p in result.cell_properties:
        assert p.area > 0, "area should be the voxel count, not NaN"
        assert len(p.centroid) == 3, "a 3-D centroid should have three components"
        assert all(np.isfinite(c) for c in p.centroid)


def test_two_dimensional_input_still_reports_eccentricity():
    """The fix must not cost the 2-D behaviour, where eccentricity is real."""
    img = np.zeros((64, 64), dtype=np.uint16)
    yy, xx = np.ogrid[:64, :64]
    img[((yy - 20) ** 2 + (xx - 20) ** 2) <= 64] = 4000
    img[((yy - 45) ** 2 / 4.0 + (xx - 45) ** 2) <= 36] = 4000   # deliberately elongated

    seg = CellSegmenter(method="watershed", min_size=20, max_size=10_000_000)
    result = seg.segment(img)

    assert result.cell_properties
    assert all(np.isfinite(p.eccentricity) for p in result.cell_properties), (
        "eccentricity is defined in 2-D and must still be measured there"
    )
