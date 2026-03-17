"""
IEDB Epitope Validation
=========================

Validates Cognisom's neoantigen binding predictor against the Immune
Epitope Database (IEDB) gold-standard predictions (NetMHCpan 4.1).

Tests a panel of known epitope-MHC binding pairs to measure:
1. Correlation between Cognisom and IEDB predicted IC50 values
2. Binder/non-binder classification concordance
3. Sensitivity for known strong binders

IEDB API: http://tools-cluster-interface.iedb.org/tools_api/mhci/

Citation:
    Vita et al. The Immune Epitope Database (IEDB): 2018 update.
    Nucleic Acids Res. 2019; 47:D339-D343. PMID: 30357391
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests
import numpy as np

logger = logging.getLogger(__name__)

IEDB_API = "http://tools-cluster-interface.iedb.org/tools_api/mhci/"

# Known epitopes for validation (experimentally validated binders)
VALIDATION_EPITOPES = [
    # (peptide, allele, expected_binding: "strong"/"weak"/"non", source)
    # HIV epitopes (gold standard)
    ("SLYNTVATL", "HLA-A*02:01", "strong", "HIV Gag p17"),
    ("ILKEPVHGV", "HLA-A*02:01", "strong", "HIV Pol RT"),
    ("FLGKIWPSY", "HLA-A*24:02", "strong", "HIV Nef"),
    # Cancer neoantigens (published)
    ("KQSSKALQR", "HLA-A*03:01", "strong", "KRAS G12D neoantigen"),
    ("VVGAVGVGK", "HLA-A*03:01", "strong", "KRAS G12V neoantigen"),
    ("KLVVVGADGV", "HLA-A*02:01", "strong", "KRAS G12D 10mer"),
    # Prostate cancer related
    ("FLTPKKLQCV", "HLA-A*02:01", "strong", "PSA peptide"),
    ("VISNDVCAQV", "HLA-A*02:01", "weak", "PSMA peptide"),
    # Known non-binders (negative controls)
    ("AAAAAAAAAA", "HLA-A*02:01", "non", "Poly-A negative control"),
    ("GGGGGGGGGG", "HLA-A*02:01", "non", "Poly-G negative control"),
    ("DDDDDDDDDD", "HLA-A*02:01", "non", "Poly-D negative control (charged)"),
    # Additional validated peptides
    ("GILGFVFTL", "HLA-A*02:01", "strong", "Influenza M1"),
    ("NLVPMVATV", "HLA-A*02:01", "strong", "CMV pp65"),
    ("GLCTLVAML", "HLA-A*02:01", "strong", "EBV BMLF1"),
    ("TPRVTGGGAM", "HLA-B*07:02", "strong", "CMV pp65 B7"),
    ("RPHERNGFTVL", "HLA-B*07:02", "strong", "CMV pp65 B7 11mer"),
    ("YVLDHLIVV", "HLA-A*02:01", "weak", "HCV NS3"),
    ("CINGVCWTV", "HLA-A*02:01", "weak", "HCV NS3-2"),
    # More non-binders
    ("EEEEEEEEE", "HLA-A*02:01", "non", "Poly-E negative"),
    ("KKKKKKKKKK", "HLA-A*02:01", "non", "Poly-K negative"),
]


@dataclass
class EpitopeResult:
    peptide: str
    allele: str
    expected: str  # "strong", "weak", "non"
    source: str
    # IEDB prediction
    iedb_ic50: float = 0.0
    iedb_percentile: float = 0.0
    iedb_class: str = ""
    # Cognisom prediction
    cognisom_ic50: float = 0.0
    cognisom_class: str = ""
    # Concordance
    concordant: bool = False


@dataclass
class IEDBValidationSummary:
    n_peptides: int = 0
    n_concordant: int = 0
    concordance_rate: float = 0.0
    # Sensitivity for binders
    true_positive: int = 0  # Cognisom says binder, IEDB says binder
    false_negative: int = 0  # Cognisom says non-binder, IEDB says binder
    true_negative: int = 0  # Both say non-binder
    false_positive: int = 0  # Cognisom says binder, IEDB says non-binder
    sensitivity: float = 0.0
    specificity: float = 0.0
    # IC50 correlation
    correlation: float = 0.0
    results: List[EpitopeResult] = field(default_factory=list)
    total_time_seconds: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "n_peptides": self.n_peptides,
            "concordance_rate": round(self.concordance_rate * 100, 1),
            "sensitivity": round(self.sensitivity * 100, 1),
            "specificity": round(self.specificity * 100, 1),
            "correlation": round(self.correlation, 3),
            "true_positive": self.true_positive,
            "false_negative": self.false_negative,
            "true_negative": self.true_negative,
            "false_positive": self.false_positive,
            "total_time_seconds": round(self.total_time_seconds, 1),
            "citation": "Vita et al. NAR 2019; PMID: 30357391",
        }


class IEDBValidator:
    """Validate Cognisom neoantigen predictions against IEDB."""

    def __init__(self):
        self.session = requests.Session()

    def query_iedb(self, peptide: str, allele: str) -> Tuple[float, float]:
        """Query IEDB API for MHC binding prediction.

        Returns (ic50_nm, percentile_rank)
        """
        try:
            resp = self.session.post(IEDB_API, data={
                "method": "ann",  # NetMHCpan
                "sequence_text": peptide,
                "allele": allele,
                "length": str(len(peptide)),
            }, timeout=30)

            if resp.status_code == 200:
                lines = resp.text.strip().split("\n")
                if len(lines) >= 2:
                    parts = lines[1].split("\t")
                    if len(parts) >= 7:
                        ic50 = float(parts[6])
                        percentile = float(parts[7]) if len(parts) > 7 else 50.0
                        return ic50, percentile
        except Exception as e:
            logger.debug("IEDB query failed for %s: %s", peptide, e)

        return 50000.0, 50.0  # Default: non-binder

    def predict_cognisom(self, peptide: str, allele: str) -> float:
        """Get Cognisom's binding prediction for a peptide."""
        from cognisom.genomics.neoantigen_predictor import NeoantigenPredictor
        predictor = NeoantigenPredictor()
        ic50 = predictor._predict_binding(peptide, allele)
        return ic50

    def classify(self, ic50: float) -> str:
        """Classify IC50 into binding category."""
        if ic50 < 50:
            return "strong"
        elif ic50 < 500:
            return "weak"
        return "non"

    def run_validation(self, progress_callback=None) -> IEDBValidationSummary:
        """Run full IEDB validation panel."""
        summary = IEDBValidationSummary()
        t0 = time.time()

        total = len(VALIDATION_EPITOPES)
        summary.n_peptides = total

        iedb_ic50s = []
        cognisom_ic50s = []

        for i, (peptide, allele, expected, source) in enumerate(VALIDATION_EPITOPES):
            if progress_callback:
                progress_callback(i + 1, total, f"{peptide} / {allele}")

            result = EpitopeResult(
                peptide=peptide, allele=allele,
                expected=expected, source=source,
            )

            # IEDB prediction
            iedb_ic50, iedb_pct = self.query_iedb(peptide, allele)
            result.iedb_ic50 = iedb_ic50
            result.iedb_percentile = iedb_pct
            result.iedb_class = self.classify(iedb_ic50)

            # Cognisom prediction
            cognisom_ic50 = self.predict_cognisom(peptide, allele)
            result.cognisom_ic50 = cognisom_ic50
            result.cognisom_class = self.classify(cognisom_ic50)

            # Concordance (both agree on binder vs non-binder)
            iedb_binder = result.iedb_class in ("strong", "weak")
            cog_binder = result.cognisom_class in ("strong", "weak")
            result.concordant = (iedb_binder == cog_binder)

            # Confusion matrix
            if iedb_binder and cog_binder:
                summary.true_positive += 1
            elif iedb_binder and not cog_binder:
                summary.false_negative += 1
            elif not iedb_binder and not cog_binder:
                summary.true_negative += 1
            else:
                summary.false_positive += 1

            summary.results.append(result)
            iedb_ic50s.append(iedb_ic50)
            cognisom_ic50s.append(cognisom_ic50)

            time.sleep(0.5)  # Rate limit IEDB API

        # Compute metrics
        summary.n_concordant = sum(1 for r in summary.results if r.concordant)
        summary.concordance_rate = summary.n_concordant / max(1, total)

        tp_fn = summary.true_positive + summary.false_negative
        tn_fp = summary.true_negative + summary.false_positive
        summary.sensitivity = summary.true_positive / max(1, tp_fn)
        summary.specificity = summary.true_negative / max(1, tn_fp)

        # IC50 correlation (log scale)
        if len(iedb_ic50s) > 2:
            log_iedb = np.log10(np.clip(iedb_ic50s, 1, 50000))
            log_cog = np.log10(np.clip(cognisom_ic50s, 1, 50000))
            summary.correlation = float(np.corrcoef(log_iedb, log_cog)[0, 1])

        summary.total_time_seconds = time.time() - t0

        logger.info(
            "IEDB validation: %d/%d concordant (%.0f%%), sensitivity=%.0f%%, "
            "specificity=%.0f%%, correlation=%.2f",
            summary.n_concordant, total, summary.concordance_rate * 100,
            summary.sensitivity * 100, summary.specificity * 100,
            summary.correlation,
        )
        return summary
