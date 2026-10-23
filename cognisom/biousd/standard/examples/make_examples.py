"""Write the example files. Run from the package's parent so imports resolve:

    python -m cognisom.biousd.standard.examples.make_examples   (inside Cognisom)
    python -m biousd.examples.make_examples                     (extracted)

Each file is produced by the reference writer with a fixed seed, so they
can be regenerated and diffed. Sizes are kept under a megabyte each.
"""
from __future__ import annotations

import os
import sys

import numpy as np

try:
    from ..write import write_cells
    from ..validate import validate
except ImportError:                                # run as a plain script
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from standard.write import write_cells        # type: ignore
    from standard.validate import validate        # type: ignore

HERE = os.path.dirname(os.path.abspath(__file__))

#: A few identities with their ontology terms, resolved by exact label.
LINEAGES = (
    ("cancer", "NCIT:C2919", 14.0),        # prostatic adenocarcinoma cell
    ("luminal", "CL:0002340", 12.0),       # luminal cell of prostate epithelium
    ("basal", "CL:0002341", 10.0),         # basal cell of prostate epithelium
    ("fibroblast", "CL:0000057", 11.0),
    ("endothelial", "CL:0000115", 10.0),
    ("immune", "CL:0000625", 8.0),         # CD8-positive, alpha-beta T cell
    ("immune", "CL:0000623", 9.0),         # natural killer cell
    ("immune", "CL:0000235", 15.0),        # macrophage
)


def _cells(rng, n, extent, lesion_fraction=0.2):
    cells = []
    n_lesion = int(n * lesion_fraction)
    for i in range(n):
        if i < n_lesion:
            lineage, term, diameter = LINEAGES[0]
            position = rng.normal(0.0, extent * 0.15, 3)
        else:
            lineage, term, diameter = LINEAGES[1 + (i % (len(LINEAGES) - 1))]
            position = rng.uniform(-extent / 2, extent / 2, 3)
        cells.append(dict(
            position=tuple(float(v) for v in position), cell_id=i,
            cell_type=lineage, ontology_term=term,
            phase=("G1", "S", "G2", "M", "G0")[i % 5],
            age=float(rng.uniform(0.0, 24.0)), alive=True,
            volume=float(rng.uniform(0.85, 1.15)), scale_um=diameter))
    return cells


def main() -> int:
    rng = np.random.default_rng(2026)
    made = []

    # 1. Small enough to read in a text editor: the pitch.
    path = os.path.join(HERE, "minimal_200_cells.usda")
    if os.path.exists(path): os.remove(path)
    write_cells(path, _cells(rng, 200, 120.0))
    made.append(path)

    # 2. A tissue slab, binary.
    path = os.path.join(HERE, "tissue_slab_10k_cells.usdc")
    if os.path.exists(path): os.remove(path)
    write_cells(path, _cells(rng, 10_000, 500.0))
    made.append(path)

    # 3. Coloured by a per-cell field, with the ramp declared on the layer.
    cells = _cells(rng, 2_000, 240.0)
    ages = np.array([c["age"] for c in cells])
    lo, hi = float(ages.min()), float(ages.max())
    for c in cells:
        t = (c["age"] - lo) / (hi - lo)
        c["color"] = (0.93 - 0.90 * t, 0.95 - 0.76 * t, 0.98 - 0.56 * t)
    path = os.path.join(HERE, "coloured_by_age_2k_cells.usdc")
    if os.path.exists(path): os.remove(path)
    write_cells(path, cells, colour_by="bio:cell:age", colour_range=(lo, hi))
    made.append(path)

    worst = 0
    for path in made:
        report = validate(path)
        print(f"{os.path.basename(path):32s} {os.path.getsize(path) / 1024:7.0f} KB  "
              f"{report.describe().splitlines()[0].split(': ', 1)[1]}")
        worst = max(worst, 0 if report.ok else 1)
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
