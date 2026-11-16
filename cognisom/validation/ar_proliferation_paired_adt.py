"""The engine's drug mechanism, tested WITHIN patients across treatment.

PRE-SPECIFIED 2026-11-14, before any expression value was read. The
prediction, the confounder, the positive control and the refutation
rule below were written first.

WHY THIS AND NOT THE CROSS-SECTIONAL TEST. ``ar_proliferation_tcga``
found AR activity and proliferation correlated at rho +0.41 across 493
treatment-naive TCGA prostate tumours, which is the sign the engine
integrates. Its own docstring says what it cannot show: the engine's
claim is causal and within a cell -- block this receptor and this cell
divides more slowly -- and a correlation across tumours is a different
quantity. This is the nearest available thing to the causal claim:
the SAME patients, before and after the drug class the engine models.

THE DATA. GEO GSE111177 (Sowalsky/Spratt et al., Cancers 2018):
whole-transcriptome RNA-seq of **20 patient-matched pre-ADT biopsies
and 20 post-ADT prostatectomy specimens**, 48 libraries in all, plus a
sample key giving patient, timepoint and specimen type. Downloaded at
run time and not vendored, the same way the cBioPortal tests work.

THE SIGNATURES ARE IMPORTED, NOT RETYPED, from
``ar_proliferation_tcga``. The whole point is the same signatures under
a different design, so a gene set that drifted between the two files
would make them incomparable. Normalisation is also the same -- a
within-sample log2 ratio to ACTB -- which here does double duty: it is
how the other tests handle sample-wide inflation, and it is also what
makes raw read counts usable without a library-size model.

THE PREDICTION. Across the 20 patients, the change in AR activity must
NOT be negatively correlated with the change in proliferation. The
engine says androgen deprivation lengthens the division interval, so a
patient whose AR activity fell further should show a larger fall in
proliferation. A clear negative would mean the two move oppositely
within patients under the actual drug, and the engine's sign would be
wrong where it matters most.

THE CONFOUNDER, named before measuring, and it is structural: **the
pre sample is a needle biopsy (Bx) and the post sample is a radical
prostatectomy specimen (RP)**. Specimen type is confounded with
timepoint in this cohort. A needle core and a whole-gland section
differ in epithelial and stromal content whatever the treatment, so the
MEAN change in any signature here is uninterpretable and is reported
only as context, never as a result.

    CORRECTION, after reading the sample key: the sentence above said
    PERFECTLY confounded, and it is not quite. Of the 24 pairs, 20 are
    Bx -> RP, **three are TURP -> TURP** and one is Bx -> TURP. So three
    patients had no specimen change at all. Three is far too few to
    estimate a correlation in, so the confound is NOT eliminated by
    them -- but "perfectly confounded" was wrong and is corrected here
    rather than left standing. What they do show is reported with the
    result.

What survives the confound is the ACROSS-PATIENT CORRELATION OF THE
CHANGES. A sampling shift common to all 20 patients moves every delta
by roughly the same amount and cancels from a correlation between two
deltas. That correlation is the pre-specified test; the paired means
are not.

THE POSITIVE CONTROL. Mean change in AR activity must be NEGATIVE: the
cohort received androgen deprivation, and if its AR target genes did
not fall then the dataset cannot test anything about AR. Reported with
a sign check, not assumed.

A TRAP FOUND WHILE BUILDING THIS, recorded because it would have
produced a confident wrong answer. The counts matrix carries only
versioned Ensembl gene IDs, no symbols, so the symbols above have to be
resolved. Ensembl's ``xrefs/symbol`` endpoint returns ALIAS matches:
"KLK3" comes back as two genes and "AR" as four, and the first is not
the right one in either case -- taking it would have measured KLKB1
(plasma kallikrein, chr4) instead of prostate-specific antigen, and
amphiregulin instead of the androgen receptor. ``lookup/symbol``
returns the canonical gene, and this module additionally REFUSES any
gene whose returned ``display_name`` is not the symbol asked for.

-- RESULT, 2026-11-14 ------------------------------------------------

**24 patient-matched pairs**, not the 20 the paper's abstract leads
with: the sample key carries 48 libraries over 24 patients, each with
exactly one Pre and one Post and no duplicates. Counted, not assumed.

POSITIVE CONTROL PASSES, strongly. Mean change in AR activity
**-1.073** (Wilcoxon p = 1.05e-05): androgen deprivation did suppress
AR target genes in this cohort, so it can test an AR mechanism. A
detail worth keeping: AR *itself* went slightly UP (+0.120) while its
targets fell, which is the receptor upregulation ADT is known to
provoke, and is a small independent sign that the signature is
measuring pathway OUTPUT rather than receptor abundance.

THE PRE-SPECIFIED TEST: **Spearman rho +0.490, p = 0.015, n = 24**.
Patients whose AR activity fell further showed a larger fall in
proliferation. The prediction survives, within patients, under the drug
class the engine models.

It is not one patient: leave-one-patient-out gives rho +0.421 to
+0.564, **24 of 24 positive**, every one still significant at p < 0.05.

THE MEAN CHANGE IN PROLIFERATION IS NULL (+0.048, Wilcoxon p = 0.833)
and that is exactly what the confound predicts rather than a problem
for the result. A specimen shift common to all patients moves every
delta by about the same amount: it destroys the mean and cancels from a
correlation between two deltas. The pre-specification said the means
would be uninterpretable and the correlation would be the quantity, and
this is what that looks like.

HOW MUCH OF THE COHORT CARRIES THE SHIFT: 20 pairs Bx -> RP, 3 pairs
TURP -> TURP, 1 pair Bx -> TURP. Within the 21 specimen-CHANGED pairs
the correlation holds at rho +0.447, p = 0.042. The 3 pairs with no
specimen change cannot carry a correlation, and in them AR activity
*rose* (+0.154) rather than fell -- an observation at n = 3, reported
because it is the only confound-free window in the cohort and it does
not support the result, not a finding.

SO THE CONFOUND IS NOT RULED OUT. It is argued to cancel, and the null
mean is consistent with that argument, but this cohort cannot
demonstrate it. A design with the same specimen type at both
timepoints would.

WHAT IT STILL CANNOT SHOW. Bulk tissue, so every signature is a mixture
over epithelium, stroma and infiltrate, and ADT changes that mixture as
the tumour involutes -- some of any change is composition rather than
per-cell transcription. Twenty patients is a small cohort for a
correlation. And a non-negative result is consistent with the engine
rather than proof of it; only a clear negative is decisive.

Run directly; needs network access to ftp.ncbi.nlm.nih.gov and
rest.ensembl.org. Not a pytest, for the same reason the others are not.
"""

from __future__ import annotations

import gzip
import os
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from typing import Dict, List, Tuple

import numpy as np

from cognisom.validation.ar_proliferation_tcga import (
    AR_ACTIVITY, AR_GENE, PROLIFERATION, REFUTE_P, REFUTE_RHO,
)
from cognisom.validation.hypoxia_mhc1_tcga import (
    REFERENCE, _log_ratio, partial_spearman, signature,
)

SERIES = "GSE111177"
BASE = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE111nnn/"
        f"{SERIES}/suppl/")
COUNTS_FILE = f"{SERIES}_Movreadcounts.txt.gz"
KEY_FILE = f"{SERIES}_Sample-key.xlsx"

ENSEMBL = "https://rest.ensembl.org/lookup/symbol/homo_sapiens/"

#: Every symbol this analysis needs, resolved once per run.
GENES: Tuple[str, ...] = (
    tuple(AR_ACTIVITY) + tuple(PROLIFERATION) + (AR_GENE, REFERENCE))

_XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


# ── getting the data ─────────────────────────────────────────────────

def cache_dir() -> str:
    """Where the downloaded series lives. Not inside the repository.

    The expression data is not vendored: it is third-party and large,
    and the other external tests fetch live too. ``COGNISOM_GEO_CACHE``
    overrides, so a run can point at an already-downloaded copy.
    """
    path = os.environ.get(
        "COGNISOM_GEO_CACHE",
        os.path.join(tempfile.gettempdir(), f"cognisom-geo-{SERIES}"))
    os.makedirs(path, exist_ok=True)
    return path


def _download(name: str) -> str:
    target = os.path.join(cache_dir(), name)
    if not os.path.exists(target) or os.path.getsize(target) == 0:
        # A generic client name: nothing identifying the user is sent.
        req = urllib.request.Request(
            BASE + name, headers={"User-Agent": "cognisom-validation"})
        with urllib.request.urlopen(req, timeout=300) as r, \
                open(target, "wb") as out:
            out.write(r.read())
    return target


# ── the sample key ───────────────────────────────────────────────────

def _xlsx_rows(path: str) -> List[List[str]]:
    """Rows of the first worksheet, standard library only.

    openpyxl is not installed and is not worth adding to this
    environment to read a 10 KB key.
    """
    with zipfile.ZipFile(path) as z:
        shared: List[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall(f"{_XLSX_NS}si"):
                shared.append("".join(t.text or ""
                                      for t in si.iter(f"{_XLSX_NS}t")))
        sheet = sorted(n for n in z.namelist()
                       if n.startswith("xl/worksheets/sheet"))[0]
        root = ET.fromstring(z.read(sheet))
        rows = []
        for row in root.iter(f"{_XLSX_NS}row"):
            out = []
            for c in row.findall(f"{_XLSX_NS}c"):
                v = c.find(f"{_XLSX_NS}v")
                txt = "" if v is None else (v.text or "")
                if c.get("t") == "s" and txt.isdigit():
                    txt = shared[int(txt)]
                out.append(txt)
            rows.append(out)
        return rows


def sample_key(path: str) -> List[Dict[str, str]]:
    rows = _xlsx_rows(path)
    header = [h.strip() for h in rows[0]]
    out = []
    for r in rows[1:]:
        if not any(r):
            continue
        rec = dict(zip(header, [x.strip() for x in r]))
        if rec.get("Patient") and rec.get("Pre.Post"):
            out.append(rec)
    return out


def _normalise(name: str) -> str:
    """Counts columns use underscores and carry a run suffix; the key
    uses hyphens. ``R031_9`` <-> ``R031-9``,
    ``MSA11033-PoolA1-27646640`` <-> ``MSA11033-PoolA1``."""
    return name.strip().replace("_", "-")


# ── gene identifiers ─────────────────────────────────────────────────

def resolve_genes(symbols=GENES) -> Dict[str, str]:
    """Symbol -> unversioned Ensembl gene id, refusing any mismatch.

    ``lookup/symbol`` and not ``xrefs/symbol``: the latter returns
    aliases, where "AR" matches amphiregulin and "KLK3" matches KLKB1.
    The display_name check is what makes a silent substitution
    impossible rather than merely unlikely.
    """
    import requests

    out = {}
    for symbol in symbols:
        r = requests.get(ENSEMBL + symbol,
                         params={"content-type": "application/json"},
                         timeout=60)
        r.raise_for_status()
        d = r.json()
        name, gid = d.get("display_name"), d.get("id")
        if name != symbol or not str(gid).startswith("ENSG"):
            raise RuntimeError(
                f"Ensembl returned {name!r} ({gid}) for {symbol!r}. "
                "Refusing to measure a gene that is not the one asked "
                "for -- this is the alias trap in the module docstring.")
        out[symbol] = gid
    return out


# ── reading the matrix ───────────────────────────────────────────────

def load_expression(counts_path: str, wanted: Dict[str, str],
                    columns: List[str]) -> Dict[str, np.ndarray]:
    """Counts for the wanted genes, aligned to ``columns``."""
    by_id = {gid: symbol for symbol, gid in wanted.items()}
    found: Dict[str, np.ndarray] = {}
    with gzip.open(counts_path, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        index = {}
        for col in columns:
            if col not in header:
                raise RuntimeError(f"column {col!r} not in the matrix")
            index[col] = header.index(col)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            gid = parts[0].split(".")[0]
            symbol = by_id.get(gid)
            if symbol is None:
                continue
            found[symbol] = np.array(
                [float(parts[index[c]]) for c in columns], float)
            if len(found) == len(wanted):
                break
    missing = sorted(set(wanted) - set(found))
    if missing:
        raise RuntimeError(f"no counts row for {missing}")
    return found


# ── the test ─────────────────────────────────────────────────────────

def run() -> Dict[str, object]:
    from scipy.stats import spearmanr, wilcoxon

    counts_path = _download(COUNTS_FILE)
    key = sample_key(_download(KEY_FILE))

    # Pair patients that have exactly one Pre and one Post.
    with gzip.open(counts_path, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
    lookup = {_normalise(h): h for h in header}

    def column_for(seqid: str):
        want = _normalise(seqid)
        for norm, original in lookup.items():
            if norm == want or norm.startswith(want + "-"):
                return original
        return None

    by_patient: Dict[str, Dict[str, str]] = {}
    unmatched = []
    for rec in key:
        col = column_for(rec["SeqID"])
        if col is None:
            unmatched.append(rec["SeqID"])
            continue
        by_patient.setdefault(rec["Patient"], {})[rec["Pre.Post"]] = col

    pairs = {p: d for p, d in by_patient.items()
             if "Pre" in d and "Post" in d}
    if len(pairs) < 10:
        raise RuntimeError(
            f"only {len(pairs)} complete pairs; unmatched seq ids "
            f"{unmatched}")

    patients = sorted(pairs)
    pre_cols = [pairs[p]["Pre"] for p in patients]
    post_cols = [pairs[p]["Post"] for p in patients]

    wanted = resolve_genes()
    # One matrix over all 2N libraries, so the z-scores inside
    # `signature` are taken over the same population for both arms.
    columns = pre_cols + post_cols
    expression = load_expression(counts_path, wanted, columns)

    ar_all = signature(expression, AR_ACTIVITY)
    prolif_all = signature(expression, PROLIFERATION)
    n = len(patients)
    d_ar = ar_all[n:] - ar_all[:n]
    d_prolif = prolif_all[n:] - prolif_all[:n]

    # Positive control: did androgen deprivation suppress AR targets?
    ar_drop = float(np.mean(d_ar))
    ar_drop_p = float(wilcoxon(d_ar).pvalue) if n >= 6 else float("nan")

    rho, p = (lambda s: (float(s.statistic), float(s.pvalue)))(
        spearmanr(d_ar, d_prolif))

    # Robustness, computed here so the printed result carries it and
    # nobody has to take a single rho on trust. With n in the twenties
    # one influential patient can make or break a correlation.
    loo = []
    for i in range(n):
        mask = np.ones(n, bool)
        mask[i] = False
        loo.append(float(spearmanr(d_ar[mask], d_prolif[mask]).statistic))

    # Specimen type per pair, to say how much of the cohort actually
    # carries the Bx -> RP shift.
    types: Dict[str, Dict[str, str]] = {}
    for rec in key:
        types.setdefault(rec["Patient"], {})[rec["Pre.Post"]] = rec["Type"]
    combos: Dict[str, int] = {}
    concordant = []
    for i, pid in enumerate(patients):
        pre_t = types.get(pid, {}).get("Pre", "?")
        post_t = types.get(pid, {}).get("Post", "?")
        combos[f"{pre_t}->{post_t}"] = combos.get(f"{pre_t}->{post_t}", 0) + 1
        concordant.append(pre_t == post_t)
    same = np.array(concordant, bool)
    discordant_rho = discordant_p = float("nan")
    if (~same).sum() >= 6:
        s = spearmanr(d_ar[~same], d_prolif[~same])
        discordant_rho, discordant_p = float(s.statistic), float(s.pvalue)

    refuted = bool(rho < REFUTE_RHO and p < REFUTE_P)
    return {
        "series": SERIES,
        "n_pairs": n,
        "patients": patients,
        "leave_one_out_rho": loo,
        "specimen_combos": combos,
        "n_type_concordant": int(same.sum()),
        "concordant_mean_delta_ar": (
            float(d_ar[same].mean()) if same.any() else float("nan")),
        "discordant_rho": discordant_rho,
        "discordant_p": discordant_p,
        "mean_delta_ar": ar_drop,
        "delta_ar_wilcoxon_p": ar_drop_p,
        "mean_delta_proliferation": float(np.mean(d_prolif)),
        "delta_proliferation_wilcoxon_p": (
            float(wilcoxon(d_prolif).pvalue) if n >= 6 else float("nan")),
        "rho": rho,
        "p": p,
        "prediction_refuted": refuted,
        "ar_gene_delta": float(np.mean(
            _log_ratio(expression, AR_GENE)[n:]
            - _log_ratio(expression, AR_GENE)[:n])),
        "deltas": {"ar": d_ar.tolist(), "proliferation": d_prolif.tolist()},
    }


if __name__ == "__main__":
    out = run()
    print(f"\n  PAIRED PRE/POST ANDROGEN DEPRIVATION — {out['series']}")
    print(f"  {out['n_pairs']} patient-matched pairs "
          "(pre-ADT biopsy vs post-ADT prostatectomy)")
    print()
    print("  positive control — did ADT suppress AR targets?")
    print(f"    mean change in AR activity   {out['mean_delta_ar']:+.3f}"
          f"   (Wilcoxon p {out['delta_ar_wilcoxon_p']:.3g})")
    print(f"    mean change in AR itself     {out['ar_gene_delta']:+.3f}")
    if out["mean_delta_ar"] >= 0:
        print("    *** AR ACTIVITY DID NOT FALL. The cohort cannot test "
              "an AR mechanism; nothing below is interpretable. ***")
    print()
    print("  context only — specimen type is confounded with timepoint, "
          "so a MEAN change is not a result:")
    print(f"    mean change in proliferation {out['mean_delta_proliferation']:+.3f}"
          f"   (Wilcoxon p {out['delta_proliferation_wilcoxon_p']:.3g})")
    print()
    print("  THE PRE-SPECIFIED TEST — across patients, does the change "
          "in AR activity")
    print("  track the change in proliferation?")
    print(f"    Spearman rho {out['rho']:+.3f}   p {out['p']:.3g}   "
          f"n = {out['n_pairs']}")
    loo = out["leave_one_out_rho"]
    print(f"    leave-one-patient-out: rho {min(loo):+.3f} .. {max(loo):+.3f}"
          f"   ({sum(1 for v in loo if v > 0)}/{len(loo)} positive)")
    print()
    print("  how much of the cohort carries the specimen shift:")
    for combo, count in sorted(out["specimen_combos"].items(),
                               key=lambda kv: -kv[1]):
        print(f"    {combo:<14} {count} pairs")
    print(f"    {out['n_type_concordant']} pairs had NO specimen change "
          f"(mean change in AR activity there "
          f"{out['concordant_mean_delta_ar']:+.3f}) — too few to "
          "estimate a correlation in, so the confound is not ruled out")
    if out["discordant_rho"] == out["discordant_rho"]:   # not NaN
        print(f"    within the specimen-CHANGED pairs: rho "
              f"{out['discordant_rho']:+.3f}  p {out['discordant_p']:.3g}")
    print()
    if out["prediction_refuted"]:
        print("  PREDICTION REFUTED: within patients, under the drug class "
              "the engine models, AR activity and proliferation move "
              "OPPOSITELY. The engine's sign is wrong where it matters.")
    else:
        print("  prediction survives: the change in AR activity is not "
              "negatively correlated with the change in proliferation.")
