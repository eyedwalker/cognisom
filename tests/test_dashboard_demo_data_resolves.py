"""The data files the dashboard loads by name must exist where it looks.

``app.py`` built the SEQC2 demo path as ``_pages_dir.parent / "genomics"``.
``_pages_dir`` is ``cognisom/dashboard/_pages``, so one ``.parent`` lands on
``cognisom/dashboard`` and the path pointed at
``cognisom/dashboard/genomics/seqc2_precomputed.json``, which does not exist.
The primary "Load Patient" button -- the real-data demo the product leads with
-- died with FileNotFoundError, and no test noticed because the path is only
built inside a Streamlit callback.

These tests read the paths out of the source and check them on disk, which is
the cheapest thing that would have caught it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
APP = REPO / "cognisom" / "dashboard" / "app.py"
PAGES_DIR = REPO / "cognisom" / "dashboard" / "_pages"

#: Data files the dashboard loads by relative path, and where they really are.
EXPECTED = {
    "seqc2_precomputed.json": REPO / "cognisom" / "genomics" / "seqc2_precomputed.json",
    "seqc2_demo.vcf": REPO / "cognisom" / "genomics" / "seqc2_demo.vcf",
}


@pytest.mark.parametrize("name,path", sorted(EXPECTED.items()))
def test_the_demo_file_exists(name: str, path: Path):
    assert path.exists(), f"{name} is missing from {path.parent}"
    assert path.stat().st_size > 0, f"{name} is empty"


def test_the_app_resolves_the_precomputed_profile():
    """Evaluate the path expression app.py actually uses."""
    src = APP.read_text()
    m = re.search(r'_json_path = \(?(_pages_dir[\w.]*)\s*/\s*"genomics"', src)
    assert m, ("app.py no longer builds the precomputed path from _pages_dir; "
               "update this test to follow it")

    expr = m.group(1)
    base = PAGES_DIR
    for _ in range(expr.count(".parent")):
        base = base.parent
    resolved = base / "genomics" / "seqc2_precomputed.json"

    assert resolved.exists(), (
        f"app.py resolves the SEQC2 profile to {resolved}, which does not "
        f"exist. The Load Patient button raises FileNotFoundError. The file "
        f"is at {EXPECTED['seqc2_precomputed.json']}")


def test_the_precomputed_profile_has_what_the_loader_reads():
    """app.py indexes these keys directly, so a missing one is a KeyError."""
    data = json.loads(EXPECTED["seqc2_precomputed.json"].read_text())

    for key in ("patient_id", "cancer_type", "tmb", "hla_alleles",
                "driver_genes_found", "neoantigens"):
        assert key in data, f"the loader reads data[{key!r}] and it is absent"

    assert data["tmb"] > 0, "a demo patient with zero TMB shows nothing useful"
    assert data["driver_genes_found"], "no driver genes in the demo profile"
    assert data["neoantigens"], "no neoantigens in the demo profile"


def test_every_neoantigen_has_the_fields_the_loader_indexes():
    """app.py reads n["peptide"] without a default; the rest use .get()."""
    data = json.loads(EXPECTED["seqc2_precomputed.json"].read_text())

    for i, n in enumerate(data["neoantigens"]):
        assert n.get("peptide"), f"neoantigen {i} has no peptide"
