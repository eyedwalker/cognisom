"""Test the engine's CENTRAL drug mechanism against human tumours.

PRE-SPECIFIED 2026-11-14, before any data was fetched. The prediction,
the confounder, the positive control and the refutation rule below were
written first and are not to be edited after seeing a result; a new
analysis choice means a new file and a stated reason.

WHY THIS MECHANISM. The engine's drug arm rests on one coupling. In
``cellular_module.py`` (around line 1069)::

    if signalling is not None and signalling < AR_INTACT_ABOVE:   # 0.95
        division_time = division_time / max(signalling, 0.05)
    if cell.age >= division_time and self._division_permitted(cell_id):
        ...divide...

Androgen-receptor signalling does not gate division; it STRETCHES the
interval. Lower signalling means a longer division time means slower
proliferation. Everything the simulation says about androgen
deprivation runs through those three lines, and nothing outside the
model has ever been asked whether they are right.

THE PREDICTION, and it is deliberately a weak one. Across human
prostate tumours, an AR-activity signature must NOT be negatively
correlated with a proliferation signature. A clear negative would mean
that in real tumours androgen signalling goes with LESS proliferation,
which is the opposite of the sign the engine integrates, and the drug
arm would need rethinking.

THE DEAD BAND MAKES IT SUBTLER, and this is stated in advance so a null
cannot be claimed afterwards as a success. The coupling is one-sided:
above 0.95 signalling nothing changes. TCGA PRAD is treatment-naive
primary prostate cancer, where most tumours have intact AR signalling,
so most samples should sit ABOVE the dead band, where the engine
predicts a FLAT relationship. A weak or null correlation is therefore
what the engine predicts here -- it is consistent with the model, and
it is NOT evidence for it. Only a clear negative is informative.

THE CONFOUNDER, named before measuring: TUMOUR PURITY. Both signatures
are expressed by tumour epithelium, so a sample with more stroma dilutes
both at once and manufactures a spurious POSITIVE correlation. That is
the direction that would flatter the model, which is exactly why it is
controlled for rather than mentioned. A stromal signature is partialled
out of both, the same way IFN-gamma was partialled out of the
hypoxia/MHC-I test.

THE DATA. TCGA PRAD PanCancer Atlas via cBioPortal, RNA Seq V2 RSEM --
the same cohort, assay, within-sample ACTB log-ratio and Spearman
statistic as ``hypoxia_mhc1_tcga``, so no analysis choice is new here.

WHAT IT CANNOT SHOW, stated in advance. The engine's claim is causal and
WITHIN a cell: block this cell's receptor and this cell divides more
slowly. This is a cross-sectional correlation ACROSS tumours, which is
a different quantity. Tumours differ in grade, purity and clonal
composition all at once. A negative result would be a genuine problem
for the engine's sign; a non-negative result does not establish the
causal claim. The causal test needs paired biopsies before and after
androgen deprivation, which this cohort does not contain.

-- RESULT, 2026-11-14 ------------------------------------------------

493 tumours. Positive control passes: the AR-target signature tracks AR
itself at rho +0.504.

    raw      rho +0.356   p 3.8e-16
    partial  rho +0.406   p 5.7e-21    (stroma partialled out)

THE PREDICTION SURVIVES, and this one survives POSITIVELY rather than as
a surviving null: AR activity goes WITH proliferation in human prostate
tumours, which is the sign the engine integrates.

It is not one gene. Every proliferation gene is positive against the AR
signature (PCNA +0.570, CCNB1 +0.463, RRM2 +0.451, MKI67 +0.366,
CDK1 +0.326, TOP2A +0.325, TYMS +0.286, BIRC5 +0.227), every AR target
is positive against the proliferation signature (NKX3-1 +0.512,
PMEPA1 +0.410, TMPRSS2 +0.314, STEAP4 +0.297, KLK2 +0.290, FKBP5 +0.267,
KLK3 +0.173), and leaving any one AR gene out moves rho only between
+0.355 and +0.452.

THE PRE-SPECIFIED CONFOUNDER DIRECTION WAS WRONG, which is recorded
rather than quietly corrected. The docstring above predicts that stromal
dilution would manufacture a spurious POSITIVE correlation. Measured, it
runs the other way: AR activity vs stroma is -0.250 and proliferation vs
stroma is +0.121, so the confound was SUPPRESSING the relationship, and
controlling for it moved rho UP from +0.356 to +0.406. The reason is
that the stromal signature is acting partly as a GRADE proxy and not
purely as a purity proxy -- higher-grade tumours carry more reactive
stroma and divide faster, while AR activity is a feature of the
well-differentiated luminal phenotype. The control was worth having; the
stated reason for it was not the operative one.

AND A TENSION WITH THE DEAD BAND, which is the most useful thing here.
``AR_INTACT_ABOVE = 0.95`` means androgen signalling has NO effect on
division anywhere above 95% of eugonadal -- a flat zone covering most of
the intact range. Treatment-naive tumours should mostly sit inside it,
so the engine predicts a roughly FLAT relationship in this cohort. The
data show a clearly graded one. That is the first evidence from outside
the model bearing on a constant that was chosen to keep the Baseline run
bit-identical, which is a convenience and not a measurement.

The tension is QUALITATIVE and must not be read as a number. The AR
signature is a cross-tumour z-score; it does not map onto the engine's
0-1 signalling fraction, so nothing here says where 0.95 should move to,
only that a wide flat zone is hard to reconcile with a graded
association across the normal range. Narrowing it would move every
pinned division count in the suite, so it needs its own measurement
first -- ideally the paired pre/post-deprivation biopsies that would
test the causal claim anyway.

Run directly; needs network access to cbioportal.org. Not a pytest, for
the same reason the hypoxia test is not: a unit test that depends on a
public API is a flake.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from cognisom.validation.hypoxia_mhc1_tcga import (
    API, REFERENCE, _log_ratio, fetch, partial_spearman, signature,
)

#: Treatment-naive primary prostate adenocarcinoma.
STUDY = "prad_tcga_pan_can_atlas_2018"

#: Canonical androgen-receptor target genes. Each is a named, textbook
#: AR target rather than a signature lifted from a package: KLK3 is PSA,
#: the readout the engine itself emits. AR is NOT in this list -- it is
#: held out as the positive control below.
AR_ACTIVITY: Tuple[str, ...] = (
    "KLK3", "KLK2", "TMPRSS2", "NKX3-1", "FKBP5", "PMEPA1", "STEAP4",
)

#: Proliferation. Cell-cycle and replication genes, the standard
#: readouts of how fast a tumour is dividing.
PROLIFERATION: Tuple[str, ...] = (
    "MKI67", "TOP2A", "CCNB1", "BIRC5", "RRM2", "TYMS", "PCNA", "CDK1",
)

#: The confounder, not a control: a stromal/mesenchymal signature
#: standing in for 1 - purity. Partialled out of both signatures.
STROMA: Tuple[str, ...] = (
    "COL1A1", "COL1A2", "DCN", "LUM", "ACTA2", "THY1", "FAP",
)

#: Positive control for the AR signature: AR itself, held out of the
#: signature so the check is not circular. If the signature does not
#: track AR expression, nothing below is interpretable.
AR_GENE = "AR"

#: Refutation rule. Fixed before the data was fetched.
REFUTE_RHO = -0.1
REFUTE_P = 0.05


def run(study: str = STUDY) -> Dict[str, object]:
    from scipy.stats import spearmanr

    genes = (tuple(AR_ACTIVITY) + tuple(PROLIFERATION) + tuple(STROMA)
             + (AR_GENE, REFERENCE))
    expression = fetch(genes, study)
    missing = [g for g in genes if g not in expression]
    if missing:
        raise RuntimeError(f"cBioPortal returned no data for {missing}")
    n = len(expression[REFERENCE])

    ar = signature(expression, AR_ACTIVITY)
    prolif = signature(expression, PROLIFERATION)
    stroma = signature(expression, STROMA)

    # Positive control: does the AR-target signature track AR itself?
    control = float(spearmanr(ar, _log_ratio(expression, AR_GENE)).statistic)
    if control < 0.2:
        raise RuntimeError(
            f"the AR-activity signature does not track AR expression "
            f"(rho {control:+.3f}). Nothing below is interpretable.")

    raw = spearmanr(ar, prolif)
    rho, p = partial_spearman(ar, prolif, stroma)

    # How much of whatever is seen is purity: both signatures against
    # the stromal one. Reported so the reader can judge the control.
    purity = {
        "ar_vs_stroma": float(spearmanr(ar, stroma).statistic),
        "proliferation_vs_stroma": float(spearmanr(prolif, stroma).statistic),
    }

    refuted = bool(rho < REFUTE_RHO and p < REFUTE_P)
    return {
        "study": study,
        "n_tumours": n,
        "positive_control_ar": control,
        "purity_confound": purity,
        "raw_rho": float(raw.statistic),
        "raw_p": float(raw.pvalue),
        "partial_rho": rho,
        "partial_p": p,
        "prediction_refuted": refuted,
    }


def _per_gene(expression: Dict[str, np.ndarray], ar: np.ndarray,
              stroma: np.ndarray, genes: Sequence[str]):
    """Each proliferation gene on its own, so one gene cannot carry it."""
    rows = {}
    for gene in genes:
        values = _log_ratio(expression, gene)
        rho, p = partial_spearman(ar, values, stroma)
        rows[gene] = (rho, p)
    return rows


if __name__ == "__main__":
    out = run()
    print(f"\n  AR ACTIVITY vs PROLIFERATION — {out['study']}")
    print(f"  {out['n_tumours']} tumours")
    print(f"  positive control, AR signature vs AR itself: "
          f"rho {out['positive_control_ar']:+.3f}")
    print(f"  purity confound: AR vs stroma "
          f"{out['purity_confound']['ar_vs_stroma']:+.3f}, "
          f"proliferation vs stroma "
          f"{out['purity_confound']['proliferation_vs_stroma']:+.3f}")
    print()
    print(f"  raw      rho {out['raw_rho']:+.3f}  p {out['raw_p']:.2e}")
    print(f"  partial  rho {out['partial_rho']:+.3f}  "
          f"p {out['partial_p']:.2e}   (stroma partialled out)")
    print()
    if out["prediction_refuted"]:
        print("  PREDICTION REFUTED: AR activity goes with LESS "
              "proliferation in human prostate tumours, which is the "
              "opposite of the sign the engine integrates.")
    else:
        print("  prediction survives: AR activity is not negatively "
              "correlated with proliferation.")
        print("  NOTE: the engine predicts a FLAT relationship in "
              "treatment-naive tumours (most sit above the 0.95 dead "
              "band), so this is consistent with the model and is NOT "
              "evidence for it.")
