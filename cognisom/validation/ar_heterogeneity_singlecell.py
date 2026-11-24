"""How much does AR drive differ BETWEEN CELLS inside one tumour?

PRE-SPECIFIED 2026-11-24, before any expression value was read. The
null, both positive controls, the negative control and the stopping
rule below were written first.

WHY THIS MEASUREMENT EXISTS. Sweeping ``AR_INTACT_ABOVE`` 0.80-1.00
(2026-11-17) showed the dead band is not what flattens the engine's
AR/division coupling. What flattens it is that every cell in a
treatment arm carries the SAME AR signalling, to about one part in ten
thousand: the engine's variation is between arms (0.10 blocked / 0.50
castrate / 1.00 untreated / 3.05 agonist-driven) and essentially none
is between cells. The external evidence -- 493 TCGA prostate tumours at
rho +0.41, 24 paired pre/post-ADT biopsies at rho +0.49 -- is a GRADED
association, and the engine has nothing graded across cells to
correspond to it. That is limitation 2 on ``?page=methods``, and its
stated fix is cell-to-cell variation in AR drive, NOT a different
threshold.

A fix needs a magnitude, and the project's standing rule is that a
capability blocked on an unsourced number stays blocked rather than
getting an invented one (the epigenetic module and the unwritten
cell-state columns are both in that state on purpose). This sources it.

THE DATA. "Single-cell atlas of 24 hormone therapy naive localised
prostate cancers" (CELLxGENE dataset 881e0e6b, 68,322 cells).
**Hormone-therapy naive** is the point: it is the arm the engine's
untreated signalling corresponds to. Fetched at run time from the
public census S3 mirror and NOT vendored, the same way the cBioPortal
and GEO tests work.

Two things the CENSUS cannot give, which is why the source h5ad is used
instead of ``cellxgene_census.get_anndata``:
  - the census normalises ``obs`` to a fixed schema and drops the
    authors' ``malignant_anno_merged``. Its own ``disease`` column is
    per-SAMPLE -- all 24 donors carry one label -- so the census cannot
    tell a malignant luminal cell from a normal one, and a spread
    computed over "luminal cells in a cancer sample" would be a mixture
    of tumour and normal epithelium.
  - ``X`` in the source file is log-normalised, so counts are read from
    ``raw`` and normalised here, where the normalisation is known.

PER DONOR, ALWAYS. The engine needs the spread WITHIN one tumour.
Pooling donors would measure between-patient variation, which is a
different and larger quantity -- and it is the quantity the TCGA test
already covers.

THE PROBLEM THE DESIGN EXISTS TO SOLVE. A per-cell score's spread is
mostly sequencing depth. The authors' own ``hallmark_androgen`` column
correlates with ``nCount_RNA`` at median |rho| 0.44 across all nine
analysable donors, so its raw spread is an upper bound and nothing
more. Memory's standing trap for this data is exactly that.

THE NULL, which absorbs depth automatically. For each real gene set,
``N_NULL`` random sets of the same size, each gene drawn from the same
mean-expression ventile as the gene it replaces. A random set suffers
the same depth inflation and the same counting noise, so

    excess = var_across_cells(real) / median(var_across_cells(null))

isolates variance from a SHARED factor. excess ~= 1 means
indistinguishable from any gene set at that expression level.

CONTROLS, without which the number is unfalsifiable:
  - NEGATIVE: ``HOUSEKEEPING`` must sit near 1. If housekeeping looks
    as variable as AR, the method is measuring depth.
  - POSITIVE: ``PROLIFERATION`` on malignant cells PLUS the epithelial
    cycling cells. On malignant cells alone it is FLAT (pooled excess
    0.58, p_null 1.000) -- which is not method blindness but the
    dataset's labelling: the authors put cycling cells in their own
    ``Cycling`` category and marked all 144 epithelial ones
    ``not_applicable``, so the compartment where proliferation varies
    was excluded. Adding them back takes the pooled excess to 3.52 at
    p_null 0.000, so the method does detect a real program.
  - SECOND POSITIVE: ``INTERFERON``, chosen because the authors' own
    cluster names (Club_IFN, CAFs_IFN, SMCs_IFN) say IFN programs vary
    in this tissue -- an expectation taken from their labelling, not
    from inspecting these genes' variance.

STOPPING RULE. Report AR variation as sourced only if AR's excess
exceeds ``AR_EXCESS_MIN``, housekeeping stays under
``HOUSEKEEPING_EXCESS_MAX``, and at least one positive control exceeds
``POSITIVE_CONTROL_MIN``. Otherwise the answer is "not sourceable from
this data" rather than an invented coefficient of variation.

WHAT THE RESULT MAY AND MAY NOT BE USED FOR. This measures the spread
of the AR programme's TRANSCRIPTIONAL READOUT. The engine's
``signalling_fraction`` is receptor occupancy, and target-gene
variation is signalling variation PLUS downstream transcriptional
noise. So the sigma here is an UPPER BOUND on the spread of the
engine's signalling fraction, not a point estimate, and anything wired
from it must say so.

MEASURED, 2026-11-24, 3,147 malignant cells across 9 donors:
AR excess 4.41 (9/9 donors p_null < 0.05), interferon 2.82 (6/9),
housekeeping 1.16, proliferation-with-cycling 3.52 pooled.
AR sigma (natural log) median 0.340, range 0.175-0.556.
AR CV (linear CP10K) median 0.456, range 0.315-1.004.
"""

from __future__ import annotations

import os
import tempfile
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# The signatures are IMPORTED, not retyped. A gene set that drifted
# between this file and the cross-sectional test would make the
# measurement and the thing it is meant to inform incomparable.
from cognisom.validation.ar_proliferation_tcga import (
    AR_ACTIVITY, PROLIFERATION)

#: CELLxGENE dataset and the census release its S3 mirror belongs to.
DATASET_ID = "881e0e6b-a185-45f1-9b56-81606e91e7ff"
CENSUS_VERSION = "2025-11-08"
DATASET_TITLE = ("Single-cell atlas of 24 hormone therapy naive "
                 "localised prostate cancers")
SOURCE_URL = (
    "https://cellxgene-census-public-us-west-2.s3.us-west-2.amazonaws.com"
    f"/cell-census/{CENSUS_VERSION}/h5ads/{DATASET_ID}.h5ad")

#: Negative control: expressed everywhere, no shared biological axis.
HOUSEKEEPING: Tuple[str, ...] = (
    "ACTB", "GAPDH", "RPL13A", "TUBB", "PGK1", "PPIA", "VCP")
#: Second positive control -- see the module docstring for why these.
INTERFERON: Tuple[str, ...] = (
    "ISG15", "IFI6", "IFIT1", "IFIT3", "MX1", "OAS1", "STAT1")

#: Random matched sets per real set. 200 gives a 0.5% resolution on
#: p_null, which is enough to separate 4.4 from 1.0.
N_NULL = 200
#: A donor needs this many malignant cells for a variance to mean
#: anything. Nine of the 24 donors clear it.
MIN_CELLS_PER_DONOR = 100
#: Expression ventiles used to match a null gene to the gene it
#: replaces. Matching on expression is what makes the null absorb depth.
N_EXPRESSION_BINS = 20
#: Fixed so the null is reproducible run to run.
SEED = 20261124

# ── the pre-specified decision thresholds ──────────────────────────
#: AR must clear this to count as real variation rather than noise.
AR_EXCESS_MIN = 2.0
#: The negative control must stay under this, or the method is
#: measuring depth and nothing else.
HOUSEKEEPING_EXCESS_MAX = 1.5
#: At least one positive control must clear this, or a null AR result
#: would be uninterpretable.
POSITIVE_CONTROL_MIN = 2.0


@dataclass(frozen=True)
class SetResult:
    """One gene set, in one donor's cells."""
    name: str
    n_genes: int
    n_cells: int
    variance: float
    null_variance: float
    excess: float
    p_null: float
    missing: Tuple[str, ...] = ()

    @property
    def biological_variance(self) -> float:
        """Variance above the counting-noise floor, floored at zero."""
        return max(self.variance - self.null_variance, 0.0)


@dataclass(frozen=True)
class DonorResult:
    donor: str
    n_cells: int
    sets: Dict[str, SetResult]
    #: Spread of AR drive as a lognormal sigma: an UPPER BOUND on the
    #: engine's signalling-fraction spread (see the module docstring).
    ar_sigma: float = 0.0
    #: The same thing as a coefficient of variation on a linear scale.
    ar_cv: float = 0.0


@dataclass
class Report:
    donors: List[DonorResult] = field(default_factory=list)
    positive_control_pooled: Optional[float] = None
    positive_control_pooled_p: Optional[float] = None

    # ── the pre-specified verdict ──────────────────────────────────
    def _median(self, name: str, attr: str = "excess") -> float:
        vals = sorted(getattr(d.sets[name], attr) for d in self.donors
                      if name in d.sets)
        if not vals:
            return float("nan")
        mid = len(vals) // 2
        return (vals[mid] if len(vals) % 2
                else (vals[mid - 1] + vals[mid]) / 2.0)

    @property
    def ar_excess(self) -> float:
        return self._median("AR")

    @property
    def housekeeping_excess(self) -> float:
        return self._median("housekeeping")

    @property
    def ar_sigma_median(self) -> float:
        vals = sorted(d.ar_sigma for d in self.donors)
        if not vals:
            return float("nan")
        mid = len(vals) // 2
        return (vals[mid] if len(vals) % 2
                else (vals[mid - 1] + vals[mid]) / 2.0)

    @property
    def ar_cv_median(self) -> float:
        vals = sorted(d.ar_cv for d in self.donors)
        if not vals:
            return float("nan")
        mid = len(vals) // 2
        return (vals[mid] if len(vals) % 2
                else (vals[mid - 1] + vals[mid]) / 2.0)

    @property
    def best_positive_control(self) -> float:
        candidates = [v for v in (self.positive_control_pooled,
                                  self._median("interferon"))
                      if v is not None and v == v]
        return max(candidates) if candidates else float("nan")

    @property
    def sourced(self) -> bool:
        """Did the measurement meet every pre-specified condition?"""
        return bool(
            self.ar_excess >= AR_EXCESS_MIN
            and self.housekeeping_excess <= HOUSEKEEPING_EXCESS_MAX
            and self.best_positive_control >= POSITIVE_CONTROL_MIN)

    def verdict(self) -> str:
        if not self.donors:
            return "no donors analysed"
        if self.sourced:
            return (
                f"SOURCED: AR drive varies between cancer cells within a "
                f"tumour at excess variance {self.ar_excess:.2f} "
                f"(negative control {self.housekeeping_excess:.2f}, "
                f"positive control {self.best_positive_control:.2f}); "
                f"sigma {self.ar_sigma_median:.3f} on a log scale, CV "
                f"{self.ar_cv_median:.3f} -- an UPPER BOUND on the "
                f"engine's signalling-fraction spread")
        return (
            f"NOT SOURCED: AR excess {self.ar_excess:.2f} "
            f"(needs >= {AR_EXCESS_MIN}), housekeeping "
            f"{self.housekeeping_excess:.2f} (needs <= "
            f"{HOUSEKEEPING_EXCESS_MAX}), best positive control "
            f"{self.best_positive_control:.2f} (needs >= "
            f"{POSITIVE_CONTROL_MIN})")


def cache_dir() -> str:
    """Where the downloaded atlas lives. Not inside the repository.

    700 MB of third-party single-cell data is not vendored, and a
    committed dataset would need a NOTICE entry. ``COGNISOM_SC_CACHE``
    overrides, so a run can point at an already-downloaded copy.
    """
    path = os.environ.get(
        "COGNISOM_SC_CACHE",
        os.path.join(tempfile.gettempdir(), f"cognisom-sc-{DATASET_ID[:8]}"))
    os.makedirs(path, exist_ok=True)
    return path


def local_path() -> str:
    return os.path.join(cache_dir(), f"{DATASET_ID}.h5ad")


def is_cached() -> bool:
    path = local_path()
    # A truncated download is worse than none: it fails deep inside
    # h5py with an unrelated-looking error.
    return os.path.exists(path) and os.path.getsize(path) > 500_000_000


def fetch(progress: bool = False) -> str:
    """Download the atlas if it is not already cached."""
    path = local_path()
    if is_cached():
        return path
    tmp = path + ".part"
    with urllib.request.urlopen(SOURCE_URL) as src, open(tmp, "wb") as dst:
        total = int(src.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = src.read(1 << 22)
            if not chunk:
                break
            dst.write(chunk)
            done += len(chunk)
            if progress and total:
                print(f"\r  {done / 1e6:.0f}/{total / 1e6:.0f} MB", end="")
    if progress:
        print()
    os.replace(tmp, path)
    return path


def _normalise(counts):
    """CP10K and log1p, from raw counts, computed here.

    ``X`` in the source file is already log-normalised; doing this on
    top of it would normalise twice.
    """
    import numpy as np
    from scipy import sparse

    counts = (counts.tocsr() if sparse.issparse(counts)
              else sparse.csr_matrix(counts))
    depth = np.asarray(counts.sum(axis=1)).ravel()
    linear = counts.multiply((1e4 / np.maximum(depth, 1))[:, None]).tocsr()
    logged = linear.copy()
    logged.data = np.log1p(logged.data)
    return linear, logged, depth


def run(progress: bool = False) -> Report:
    """Fetch, measure, and return the pre-specified verdict."""
    import anndata as ad
    import numpy as np
    import pandas as pd
    from scipy import sparse  # noqa: F401  (needed by _normalise)

    rng = np.random.default_rng(SEED)
    path = fetch(progress=progress)
    adata = ad.read_h5ad(path, backed="r")
    obs = adata.obs

    if "malignant_anno_merged" not in obs.columns:
        raise RuntimeError(
            "the atlas has no malignant_anno_merged column; the census "
            "copy drops it, so this must be the source h5ad")

    is_mal = (obs["malignant_anno_merged"] == "malignant").to_numpy()
    is_cycling_epi = (
        (obs["celltype_major_v2"] == "Cycling").to_numpy()
        & obs["celltype_minor_v2"].astype(str).str.contains(
            "Epi|Lum|Basal|Club|Tumor|Hillock", case=False,
            na=False).to_numpy())
    sub = adata[is_mal | is_cycling_epi, :].to_memory()

    if sub.raw is None:
        raise RuntimeError("no raw matrix: cannot normalise from counts")
    linear, logged, depth = _normalise(sub.raw.X)
    var = sub.raw.var if sub.raw.var is not None else sub.var
    symbols = var["gene_name"].astype(str).to_numpy()

    index: Dict[str, int] = {}
    for i, sym in enumerate(symbols):
        index.setdefault(sym, i)

    mean_linear = np.asarray(linear.mean(axis=0)).ravel()
    expressed = np.where(mean_linear > 0)[0]
    bins = pd.qcut(mean_linear[expressed], N_EXPRESSION_BINS,
                   labels=False, duplicates="drop")
    by_bin: Dict[int, List[int]] = {}
    bin_of: Dict[int, int] = {}
    for gene_i, b in zip(expressed, bins):
        by_bin.setdefault(int(b), []).append(int(gene_i))
        bin_of[int(gene_i)] = int(b)

    def score(matrix, gene_idx: Sequence[int], rows) -> "np.ndarray":
        return np.asarray(matrix[rows][:, list(gene_idx)].mean(axis=1)).ravel()

    def measure(name: str, genes: Sequence[str], rows,
                matrix) -> Optional[SetResult]:
        found = [index[g] for g in genes if g in index]
        missing = tuple(g for g in genes if g not in index)
        if len(found) < 3:
            return None
        real = float(np.var(score(matrix, found, rows)))
        nulls: List[float] = []
        for _ in range(N_NULL):
            pick = [int(rng.choice(by_bin[bin_of[i]])) for i in found
                    if bin_of.get(i) in by_bin]
            if len(pick) >= 3:
                nulls.append(float(np.var(score(matrix, pick, rows))))
        if not nulls:
            return None
        null_med = float(np.median(nulls))
        return SetResult(
            name=name, n_genes=len(found), n_cells=len(rows),
            variance=real, null_variance=null_med,
            excess=(real / null_med) if null_med > 0 else float("nan"),
            p_null=float(np.mean(np.asarray(nulls) >= real)),
            missing=missing)

    mal_local = (sub.obs["malignant_anno_merged"] == "malignant").to_numpy()
    donor_of = sub.obs["donor_id"].astype(str).to_numpy()
    counts_per = {}
    for d in donor_of[mal_local]:
        counts_per[d] = counts_per.get(d, 0) + 1
    donors = sorted((d for d, n in counts_per.items()
                     if n >= MIN_CELLS_PER_DONOR),
                    key=lambda x: int(str(x).lstrip("P")))

    report = Report()
    ar_idx = [index[g] for g in AR_ACTIVITY if g in index]
    for d in donors:
        rows = np.where(mal_local & (donor_of == d))[0]
        sets: Dict[str, SetResult] = {}
        for name, genes in (("AR", AR_ACTIVITY),
                            ("proliferation", PROLIFERATION),
                            ("housekeeping", HOUSEKEEPING),
                            ("interferon", INTERFERON)):
            res = measure(name, genes, rows, logged)
            if res is not None:
                sets[name] = res
        sigma = cv = 0.0
        if "AR" in sets:
            sigma = float(np.sqrt(sets["AR"].biological_variance))
            lin = measure("AR_linear", AR_ACTIVITY, rows, linear)
            mu = float(np.mean(score(linear, ar_idx, rows))) if ar_idx else 0.0
            if lin is not None and mu > 0:
                cv = float(np.sqrt(lin.biological_variance) / mu)
        report.donors.append(DonorResult(
            donor=str(d), n_cells=len(rows), sets=sets,
            ar_sigma=sigma, ar_cv=cv))

    # Positive control: proliferation needs the cycling cells it was
    # labelled out of. Pooled, because most donors have only a handful.
    in_donors = np.isin(donor_of, donors)
    both = np.where(in_donors)[0]
    pooled = measure("proliferation", PROLIFERATION, both, logged)
    if pooled is not None:
        report.positive_control_pooled = pooled.excess
        report.positive_control_pooled_p = pooled.p_null
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    report = run(progress=True)
    print(f"{DATASET_TITLE}\n{len(report.donors)} donors\n")
    header = (f"{'donor':<7}{'n':>6}{'AR':>8}{'IFN':>8}{'house':>8}"
              f"{'prolif':>8}{'sigma':>8}{'CV':>8}")
    print(header)
    for d in report.donors:
        def ex(name):
            return d.sets[name].excess if name in d.sets else float("nan")
        print(f"{d.donor:<7}{d.n_cells:>6}{ex('AR'):>8.2f}"
              f"{ex('interferon'):>8.2f}{ex('housekeeping'):>8.2f}"
              f"{ex('proliferation'):>8.2f}{d.ar_sigma:>8.3f}{d.ar_cv:>8.3f}")
    if report.positive_control_pooled is not None:
        print(f"\npositive control (proliferation + cycling, pooled): "
              f"{report.positive_control_pooled:.2f} "
              f"at p_null {report.positive_control_pooled_p:.3f}")
    print(f"\n{report.verdict()}")
    return 0 if report.sourced else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
