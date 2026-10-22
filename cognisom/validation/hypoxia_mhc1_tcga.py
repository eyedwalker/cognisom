"""Test the MHC-I mechanism against human tumours, not against ourselves.

Every validation in this repo so far has been internal. TMB r = 0.9998 is a
round trip; the first pre-specified outcome test came back null. This is the
first check of a modelled mechanism against data the model never saw.

THE PREDICTION. ``core.antigen_presentation`` encodes Estephan et al.,
EMBO J 2025, doi 10.1038/s44318-024-00319-7: hypoxia destroys MHC-I protein
through the unfolded protein response, and HLA mRNA *rises* while it happens
(their Fig 2E, two- to threefold in HT29). If that mechanism is right, then
across human tumours a hypoxia signature must NOT correlate negatively with
HLA transcript abundance. A clear negative would mean transcriptional
downregulation dominates in vivo and the model has the mechanism wrong.

It is a negative prediction, which is weaker than a positive one -- but it is
falsifiable, and it was free to run.

THE DATA. TCGA colorectal PanCancer Atlas via cBioPortal, RNA Seq V2 RSEM,
592 tumours with complete data. The same study, assay and statistic the
source paper used for its own TCGA analysis.

THREE CORRECTIONS MADE WHILE WRITING THIS, each of which would have produced
a confident wrong answer:

1. GAPDH was used as a negative control. It cannot be one -- it is a
   glycolytic enzyme and a canonical HIF target, and it duly came out at
   rho +0.41 with the hypoxia score.
2. Every gene, including beta-actin at +0.17, correlated positively with
   hypoxia on raw RSEM. Something sample-wide inflates everything, so each
   value is now a within-sample log ratio to ACTB.
3. The raw positives survived that, but hypoxia and IFN-gamma are themselves
   correlated (rho +0.33) and every gene here is IFN-gamma inducible. The
   partial correlation controlling for an IFN-gamma score is what the result
   actually rests on.

THE RESULT, with IFN-gamma partialled out: no MHC-I gene is negatively
correlated with hypoxia. HLA-A, HLA-B, HLA-C and TAP2 sit at zero; B2M,
TAP1 and PSMB8 remain positive. The prediction survives.

WHAT IT DOES NOT SHOW. mRNA is not protein, and the model's claim is about
surface protein -- this tests only that the transcript does not fall. Bulk
expression mixes tumour, stroma and immune cells. And a null is consistent
with the model rather than confirmation of it.

A BYPRODUCT WORTH KEEPING. The two source papers disagree about the
presentation machinery: Sethumadhavan (PMID 29155844) reports hypoxia
downregulating TAP1/2 and LMP7; Estephan finds them not induced by hypoxia
and responding to IFN-gamma instead. In human tumours they track IFN-gamma,
which is Estephan's reading.

Run directly; needs network access to cbioportal.org. Not a pytest, because
a unit test that depends on a public API is a flake.
"""

from __future__ import annotations

import json
from typing import Dict, List, Sequence, Tuple

import numpy as np

API = "https://www.cbioportal.org/api"
STUDY = "coadread_tcga_pan_can_atlas_2018"

#: Canonical HIF-1 targets, each chosen and nameable rather than lifted from
#: a half-remembered signature. CA9 and SLC2A1 are the two markers the source
#: paper stained its tumour sections with.
HYPOXIA: Tuple[str, ...] = (
    "CA9", "SLC2A1", "VEGFA", "LDHA", "PGK1", "ADM", "NDRG1", "P4HA1",
    "ALDOA", "ENO1", "BNIP3")
#: Interferon-gamma response: the confounder, not a control.
INTERFERON: Tuple[str, ...] = (
    "IFNG", "STAT1", "CXCL9", "CXCL10", "GBP1", "IRF1")
MHC1: Tuple[str, ...] = ("HLA-A", "HLA-B", "HLA-C", "B2M")
MACHINERY: Tuple[str, ...] = ("TAP1", "TAP2", "PSMB8", "PSMB9")
#: Denominator for the within-sample ratio. NOT a negative control: see the
#: module docstring on why GAPDH was removed from that role.
REFERENCE = "ACTB"


def fetch(genes: Sequence[str]) -> Dict[str, np.ndarray]:
    """Expression for ``genes`` across the study, aligned by sample."""
    import requests

    found = requests.post(
        f"{API}/genes/fetch", params={"geneIdType": "HUGO_GENE_SYMBOL"},
        json=list(genes), timeout=60).json()
    ids = {g["hugoGeneSymbol"]: g["entrezGeneId"] for g in found}
    rows = requests.post(
        f"{API}/molecular-profiles/{STUDY}_rna_seq_v2_mrna/molecular-data/fetch",
        params={"projection": "SUMMARY"},
        json={"entrezGeneIds": list(ids.values()),
              "sampleListId": f"{STUDY}_rna_seq_v2_mrna"}, timeout=240).json()

    back = {v: k for k, v in ids.items()}
    table: Dict[str, Dict[str, float]] = {}
    for row in rows:
        symbol, value = back.get(row["entrezGeneId"]), row.get("value")
        if symbol is not None and value is not None:
            table.setdefault(symbol, {})[row["sampleId"]] = float(value)
    shared = sorted(set.intersection(*(set(v) for v in table.values())))
    return {g: np.array([table[g][s] for s in shared]) for g in table}


def _log_ratio(expression: Dict[str, np.ndarray], gene: str) -> np.ndarray:
    """log2(gene / ACTB) per sample.

    Exactly one value in this table is negative -- one sample's BNIP3, an
    artefact of the PanCancer batch correction -- and is clipped to zero.
    Left alone it turns one sample's signature into NaN and silently NaNs
    every correlation downstream.
    """
    def safe_log(values: np.ndarray) -> np.ndarray:
        return np.log2(np.clip(np.asarray(values, float), 0.0, None) + 1.0)

    return safe_log(expression[gene]) - safe_log(expression[REFERENCE])


def _zscore(values: np.ndarray) -> np.ndarray:
    """Plain z-score. Does NOT take a logarithm: its input already is one."""
    values = np.asarray(values, float)
    spread = values.std()
    return (values - values.mean()) / (spread if spread else 1.0)


def signature(expression: Dict[str, np.ndarray],
              genes: Sequence[str]) -> np.ndarray:
    """Median z-scored log-ratio across ``genes``, per sample."""
    return np.median(
        np.vstack([_zscore(_log_ratio(expression, g)) for g in genes]), axis=0)


def partial_spearman(x: np.ndarray, y: np.ndarray,
                     control: np.ndarray) -> Tuple[float, float]:
    """Spearman of x against y with ``control`` regressed out of both ranks."""
    from scipy.stats import rankdata, spearmanr

    rx, ry, rc = (rankdata(v) for v in (x, y, control))
    residual = lambda a: a - np.polyval(np.polyfit(rc, a, 1), rc)
    out = spearmanr(residual(rx), residual(ry))
    return float(out.statistic), float(out.pvalue)


def run() -> Dict[str, object]:
    from scipy.stats import spearmanr

    genes = (tuple(HYPOXIA) + tuple(INTERFERON) + tuple(MHC1)
             + tuple(MACHINERY) + (REFERENCE,))
    expression = fetch(genes)
    n = len(expression[REFERENCE])

    hypoxia = signature(expression, HYPOXIA)
    interferon = signature(expression, INTERFERON)

    controls = {g: float(spearmanr(hypoxia, _log_ratio(expression, g)).statistic)
                for g in ("CA9", "SLC2A1", "VEGFA")}
    if min(controls.values()) < 0.2:
        raise RuntimeError(
            f"the hypoxia score does not track known HIF targets: {controls}. "
            "Nothing below is interpretable.")

    results = {}
    for gene in tuple(MHC1) + tuple(MACHINERY):
        values = _log_ratio(expression, gene)
        raw = spearmanr(hypoxia, values)
        rho, p = partial_spearman(hypoxia, values, interferon)
        results[gene] = {"raw_rho": float(raw.statistic),
                         "partial_rho": rho, "partial_p": p}

    refuted = [g for g, r in results.items()
               if g in MHC1 and r["partial_rho"] < -0.1 and r["partial_p"] < 0.05]
    return {
        "n_tumours": n,
        "hypoxia_vs_interferon": float(spearmanr(hypoxia, interferon).statistic),
        "positive_controls": controls,
        "results": results,
        "prediction_refuted": bool(refuted),
        "refuted_by": refuted,
    }


if __name__ == "__main__":
    out = run()
    print(f"  {out['n_tumours']} tumours")
    print(f"  hypoxia vs IFN-gamma: rho {out['hypoxia_vs_interferon']:+.3f}"
          "   (the confounder this controls for)")
    print(f"  positive controls: "
          + ", ".join(f"{k} {v:+.3f}" for k, v in out["positive_controls"].items()))
    print()
    print(f"  {'gene':<8} {'raw':>8} {'partial':>9} {'p':>10}")
    for gene, r in out["results"].items():
        print(f"  {gene:<8} {r['raw_rho']:>+8.3f} {r['partial_rho']:>+9.3f}"
              f" {r['partial_p']:>10.2e}")
    print()
    print("  PREDICTION REFUTED:" if out["prediction_refuted"]
          else "  prediction survives: no MHC-I transcript falls with hypoxia.")
