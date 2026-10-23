# PhysiCell sample snapshot

`output00000001.xml` and `output00000001_cells.mat` are one MultiCellDS
(version 2) snapshot written by PhysiCell 1.14.1: 125 cells of two
types at t = 60 min, two cycle models (live, flow-cytometry-separated)
and one death model (apoptosis).

Fetched 2026-10-23 from the Galaxy `tools-iuc` repository's `pcdl` tool
test data:

- https://raw.githubusercontent.com/galaxyproject/tools-iuc/main/tools/pcdl/test-data/output00000001.xml
- https://raw.githubusercontent.com/galaxyproject/tools-iuc/main/tools/pcdl/test-data/output00000001_cells.mat

tools-iuc is MIT-licensed; PhysiCell is BSD-3. The cell-type names are
not in the output (they live in the PhysiCell settings file), so the
tests name them 0 = normal, 1 = cancer for the exercise only.

The microenvironment and mesh matrices the XML also names are not
included; `biousd.physicell` does not read them.
