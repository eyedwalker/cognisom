"""MHC-I machinery per prostate cell type, from the CELLxGENE census.

What this measures
------------------
Per cell type, the median per-cell-normalised transcript level of the
class I machinery -- B2M + HLA-A + HLA-B + HLA-C, counts per 10,000 --
WITHIN one dataset, so that depth and assay cancel. The result is a
RANKING of cell types, and that is all it is used for: RNA is not surface
protein (the gene-level correlation is 0.4-0.6), so no number here becomes
a surface level. :mod:`cognisom.core.antigen_presentation` keeps its 0-1
scale and its anchors; this module says which types sit above and below
which, with the evidence beside it.

Two traps this module exists to avoid, both met on 2026-10-23:

* ``get_anndata`` returns RAW counts. Prostate datasets in the census
  differ 100x in depth (a 10x v2 set at 1,274 median counts per cell
  reads zero B2M in half its cells), so a cross-dataset mean of raw
  counts ranks datasets, not cell types. Normalise by ``raw_sum`` and
  compare within a dataset.
* The tumour cells live in a different dataset from the normal ones, so
  tumour-vs-normal is inescapably cross-dataset. Two estimates are
  reported, which bracket it: the direct ratio between the two datasets
  (same assay, similar depth), and a ratio referenced to each dataset's
  own T cells, which are the internal standard.

The query is pinned to one census build so it is reproducible; the
output is written beside this module as evidence and the sourced table
in antigen_presentation cites it.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

CENSUS_VERSION = "2025-11-08"
GENES = ("B2M", "HLA-A", "HLA-B", "HLA-C", "TAP1", "KLK3", "AR")
MHC1_GENES = ("B2M", "HLA-A", "HLA-B", "HLA-C")
MIN_CELLS = 200
EVIDENCE_PATH = os.path.join(os.path.dirname(__file__), "evidence",
                             "census_prostate_mhc1.json")

#: The two datasets the ranking rests on: the largest normal/BPH set and the
#: only adenocarcinoma set. Both 10x 3'; median depth 5,269 and 4,808.
NORMAL_DATASET = "574e9f9e-f8b4-41ef-bf19-89a9964fd9c7"
TUMOUR_DATASET = "881e0e6b-a185-45f1-9b56-81606e91e7ff"


def measure() -> Dict[str, Any]:
    """Run the census query. Needs the network and ~1 minute."""
    import numpy as np
    import pandas as pd
    import cellxgene_census

    t0 = time.time()
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        ad = cellxgene_census.get_anndata(
            census, "homo_sapiens",
            obs_value_filter="tissue_general == 'prostate gland' and is_primary_data == True",
            var_value_filter=f"feature_name in {list(GENES)}",
            obs_column_names=["cell_type", "cell_type_ontology_term_id", "dataset_id",
                              "disease", "assay", "raw_sum"])
    X = ad.X.toarray() if hasattr(ad.X, "toarray") else np.asarray(ad.X)
    genes = list(ad.var["feature_name"])
    df = pd.DataFrame(X, columns=genes)
    for c in ("cell_type", "cell_type_ontology_term_id", "dataset_id", "disease", "raw_sum"):
        df[c] = ad.obs[c].values
    df["raw_sum"] = df["raw_sum"].astype(float)
    df = df[df.raw_sum > 0]
    for g in genes:
        df[g] = df[g] / df.raw_sum * 1e4
    df["mhc1"] = df[list(MHC1_GENES)].sum(axis=1)

    out: Dict[str, Any] = {
        "census_version": CENSUS_VERSION, "measured_on": time.strftime("%Y-%m-%d"),
        "unit": "median counts per 10,000 per cell, B2M+HLA-A+HLA-B+HLA-C, within one dataset",
        "n_cells": int(len(df)), "n_datasets": int(df.dataset_id.nunique()),
        "min_cells_per_type": MIN_CELLS, "tables": {},
    }
    for ds, dis in ((NORMAL_DATASET, "normal"), (NORMAL_DATASET, "benign prostatic hyperplasia"),
                    (TUMOUR_DATASET, "prostatic acinar adenocarcinoma")):
        sub = df[(df.dataset_id == ds) & (df.disease == dis)]
        g = sub.groupby(["cell_type", "cell_type_ontology_term_id"], observed=True)
        rows: List[Dict[str, Any]] = []
        for (ct, term), grp in g:
            if len(grp) < MIN_CELLS:
                continue
            rows.append({"cell_type": ct, "term": term, "n": int(len(grp)),
                         "mhc1": round(float(grp.mhc1.median()), 2),
                         "b2m": round(float(grp["B2M"].median()), 2),
                         "ar": round(float(grp["AR"].mean()), 3),
                         "klk3": round(float(grp["KLK3"].median()), 2)})
        rows.sort(key=lambda r: -r["mhc1"])
        out["tables"][f"{ds[:8]}:{dis}"] = {"dataset_id": ds, "disease": dis, "rows": rows}
    out["seconds"] = round(time.time() - t0, 1)
    out["tumour_vs_normal_luminal"] = _tumour_ratios(out)
    return out


def _lookup(table: Dict[str, Any], cell_type: str) -> float:
    for r in table["rows"]:
        if r["cell_type"] == cell_type:
            return float(r["mhc1"])
    return float("nan")


def _tumour_ratios(out: Dict[str, Any]) -> Dict[str, Any]:
    """Two bracketing estimates of tumour luminal MHC-I relative to normal."""
    t = out["tables"]
    normal = t[f"{NORMAL_DATASET[:8]}:normal"]
    bph = t[f"{NORMAL_DATASET[:8]}:benign prostatic hyperplasia"]
    tumour = t[f"{TUMOUR_DATASET[:8]}:prostatic acinar adenocarcinoma"]
    lum_n = _lookup(normal, "luminal cell of prostate epithelium")
    lum_t = _lookup(tumour, "luminal cell of prostate epithelium")
    direct = lum_t / lum_n if lum_n else float("nan")
    # T-cell-referenced: each dataset's own T cells as the internal standard.
    t_bph = _lookup(bph, "effector CD8-positive, alpha-beta T cell")
    lum_bph = _lookup(bph, "luminal cell of prostate epithelium")
    t_tum = _lookup(tumour, "cytotoxic T cell")
    referenced = (lum_t / t_tum) / (lum_bph / t_bph) if t_tum and t_bph and lum_bph else float("nan")
    return {"direct_cross_dataset": round(direct, 3),
            "t_cell_referenced": round(referenced, 3),
            "note": "direct compares two datasets of the same assay and similar depth; "
                    "referenced divides each luminal median by its own dataset's T-cell median"}


def write_evidence(path: str = EVIDENCE_PATH) -> Dict[str, Any]:
    out = measure()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def load_evidence(path: str = EVIDENCE_PATH) -> Dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)


def ranking(disease: str = "normal") -> List[Dict[str, Any]]:
    """Cell types by median MHC-I, highest first, from the stored evidence."""
    ev = load_evidence()
    key = f"{(TUMOUR_DATASET if 'adenocarcinoma' in disease else NORMAL_DATASET)[:8]}:{disease}"
    return list(ev["tables"][key]["rows"])


if __name__ == "__main__":          # pragma: no cover
    result = write_evidence()
    print(f"{result['n_cells']:,} cells, {result['n_datasets']} datasets, {result['seconds']} s -> {EVIDENCE_PATH}")
    print("tumour vs normal luminal:", result["tumour_vs_normal_luminal"])
