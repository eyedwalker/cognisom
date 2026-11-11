"""The patients the public demo is allowed to run, and what is wrong with each.

Why this module exists
----------------------
Loading a demo callset was open-coded in every caller:
``scripts/record_demo_run.py`` built its own profile, the dashboard pages
read a precomputed JSON, and the tests each named the VCF path again. That
is how the HLA key mismatch survived -- the demo asked for ``HCC1395``
while the measured typing was registered under ``SEQC2``, and because no
single place owned "who the demo patient is", nothing could notice.

So the cohort is declared once, here, with its disclosures attached. A
caller picks a key; it cannot pick a different subset of the facts.

The disclosure rule
-------------------
Every entry carries ``caveats``: things true of THIS patient that weaken
a result, which no amount of reading the VCF would reveal. The profile
builder already derives what it can see -- unmeasured HLA, absent matched
normal, variants that cannot cross -- and those are NOT repeated here.
What belongs here is provenance the callset does not contain: which
tissue the biopsy came from, what the patient had already been treated
with, which genome build the coordinates are on.

The biopsy-site disclosure is the one that matters most and is easiest to
forget. ``SC_9184_T`` is a **liver metastasis**, and the engine lays its
cells out on a measured *prostate* section. That mismatch is real, it is
not fixable by choosing a different layout, and a page that showed this
run without saying so would be overclaiming.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

#: Directory holding the committed demo callsets.
GENOMICS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DemoPatient:
    """One callset the demo may run, with its disclosures."""

    key: str
    #: The id the profile is built under. This is the id the HLA registry
    #: is consulted with, so it must be a name that registry knows -- see
    #: ``hla_typer.MEASURED_TYPING_ALIASES`` for what a mismatch cost.
    patient_id: str
    vcf: str
    cancer_type: str
    label: str
    #: One line for a page heading: what this patient is.
    summary: str
    #: Provenance that weakens the result and is not in the callset.
    caveats: List[str] = field(default_factory=list)
    #: Attribution required by the source data's licence, or None.
    attribution: Optional[str] = None

    @property
    def vcf_path(self) -> Path:
        return GENOMICS_DIR / self.vcf

    def load(self, builder: Optional[Any] = None):
        """Build the profile, with this entry's disclosures attached.

        The caveats ride on the returned profile as
        ``provenance_caveats``, which ``sim_request.describe_subject``
        surfaces beside the ones it derives itself. A caller that builds
        a profile without going through here gets a profile with no
        disclosures, which is why the tests assert the demo scripts call
        this function.
        """
        if builder is None:
            from cognisom.genomics.patient_profile import PatientProfileBuilder
            builder = PatientProfileBuilder(cancer_type=self.cancer_type)
        profile = builder.from_vcf_file(str(self.vcf_path),
                                        patient_id=self.patient_id)
        # A plain dataclass field, so it survives asdict/from_dict and is
        # not a monkeypatched attribute that the next refactor drops.
        profile.provenance_caveats = list(self.caveats)
        return profile


#: ``SC_9184_T`` -- the prostate patient. Chosen over the alternatives for
#: one reason the simulator can act on: AR p.F877L, the substitution that
#: switches enzalutamide and apalutamide from antagonists to agonists
#: (Joseph 2013 PMID 23779130; Balbas 2013 PMID 23580326), in a man whose
#: record says ABI_ENZA_EXPOSURE_STATUS = Exposed. The engine models that
#: switch, so genome and treatment history both bear on a mechanism that
#: is actually implemented rather than merely described.
#:
#: Rejected alternatives, recorded so the choice can be re-litigated:
#: LNCaP carries AR T878A and 2,684 variants against a TCGA PRAD median
#: of 0.90 mut/Mb -- a cell line's mutational load, not a patient's. TCGA
#: PRAD itself has HLA typing only under dbGaP controlled access.
SU2C_PROSTATE = DemoPatient(
    key="su2c_prostate",
    patient_id="SC_9184_T",
    vcf="su2c_prostate_demo.vcf",
    cancer_type="prostate",
    label="SU2C/PCF prostate (SC_9184_T)",
    summary=("Metastatic castration-resistant prostate adenocarcinoma, "
             "Gleason 9, PSA 70, carrying AR F877L after abiraterone/"
             "enzalutamide exposure"),
    caveats=[
        # The one a reader is most likely to assume away.
        "The biopsy is a LIVER METASTASIS, not prostate tissue. The run "
        "lays these cells out on a measured prostate section, so the "
        "genome is this patient's and the tissue architecture is not. "
        "Choosing a different layout would not fix it -- no measured "
        "liver-metastasis geometry is in this repository.",
        "Coordinates are GRCh37 and protein changes come from the "
        "source MAF's own HGVSp_Short, not re-annotated here. Three "
        "coding variants are skipped because the reference proteome "
        "disagrees with the source at that residue, which is the "
        "transcript difference surfacing rather than being hidden.",
        "The somatic calls are the consortium's: the source publishes "
        "Mutation_Status against a matched normal but publishes no "
        "normal read counts, so the VCF's normal genotype asserts their "
        "finding rather than carrying evidence of our own.",
        "This patient had already received abiraterone/enzalutamide and "
        "a taxane (cabazitaxel). Any treatment the simulator applies is "
        "therefore being added to a history the model does not "
        "represent, not given to an untreated man.",
        "TMB is reported against a default 30 Mb exome because the "
        "study does not publish its capture footprint. The study's own "
        "nonsynonymous density is 1.8; agreement with ours is not "
        "independent confirmation if both divide by a similar number.",
    ],
    attribution=(
        "Abida et al., PNAS 2019 (PMID 31061129), SU2C/PCF Dream Team, "
        "via cBioPortal study prad_su2c_2019. Open Data Commons Open "
        "Database License (ODbL) 1.0."),
)

#: ``HCC1395`` -- the breast cell line the demo used to run on, kept
#: because it is the only callset here with a MEASURED HLA typing
#: (OptiType on the matched normal's reads) and so the only one on which
#: the NK licensing model engages at all.
SEQC2_BREAST = DemoPatient(
    key="seqc2_breast",
    patient_id="HCC1395",
    vcf="seqc2_demo.vep.vcf",
    cancer_type="breast",
    label="SEQC2 breast cell line (HCC1395)",
    summary=("Triple-negative breast cancer cell line with a matched "
             "normal and an OptiType HLA typing from its own reads"),
    caveats=[
        "HCC1395 is a BREAST cancer cell line, while the tissue "
        "geometry, the AR pathway and most of the drug library are "
        "prostate. This was the demo's subject until a real prostate "
        "callset was available.",
        "It is a cell line, not a patient: no clinical course, no "
        "treatment history, and a mutational load unlike a tumour "
        "biopsy's.",
    ],
)

#: How much of SC_9184_T's tumour carries AR F877L -- what the cohort's
#: own files allow, and no more. Computed by scripts/su2c_ar_clonality.py
#: (``data/`` is gitignored, so it runs once locally), 2026-11-11.
#:
#: The call was read at an allele fraction of 0.233, which looks like "a
#: 23% subclone" and is not: this sample's AR is AMPLIFIED (discrete call
#: 2; segment log2 1.934, about 5.3 copies per tumour cell at the purity
#: below), so the allele fraction counts gene copies, not cells. With
#: purity p, C copies per tumour cell, a fraction f of tumour cells
#: carrying it on m of their copies:
#:
#:     VAF = p * f * m / (p * C + 1 - p)
#:
#: Only the product f * m follows, 1.37 mutant copies per tumour cell.
#: So f is somewhere between "every cell, on one or two of its copies"
#: and "a quarter of the cells, on all of their copies". The data cannot
#: say which, and a run that starts the variant at one value is choosing,
#: not reading.
#:
#: Two further limits. The AR segment has 3 markers, so the copy number
#: is the weakest input. And the biopsy was taken AFTER abiraterone/
#: enzalutamide exposure: whatever the fraction is, it is the fraction
#: after selection, and says nothing about how rare the variant was
#: before treatment. A starting fraction below this band is a scenario.
SU2C_AR_F877L_CLONALITY: Dict[str, object] = {
    "label": "AR_F877L",
    "vaf": 0.233,                       # 190 of 817 reads
    "vaf_ci95": (0.204, 0.262),
    # Agreeing estimates from independent mutations: three clonal SNVs on
    # the single X at normal copy number (VAF = purity) and 26 clonal
    # heterozygous autosomal SNVs (VAF = purity / 2).
    "purity": 0.65,
    "purity_from_autosomes": 0.63,
    "ar_copies_per_tumour_cell": 5.34,
    "mutant_copies_per_tumour_cell": 1.37,
    # mutant_copies / ar_copies, and the same across the VAF interval.
    "carrier_fraction_min": 0.26,
    "carrier_fraction_min_ci95": (0.22, 0.29),
    "carrier_fraction_max": 1.0,
    "measured_after_treatment": True,
}

DEMO_PATIENTS: Dict[str, DemoPatient] = {
    SU2C_PROSTATE.key: SU2C_PROSTATE,
    SEQC2_BREAST.key: SEQC2_BREAST,
}

#: What the public demo runs unless a caller says otherwise. Prostate,
#: because every other part of the platform is.
DEFAULT_DEMO_PATIENT = SU2C_PROSTATE.key


def demo_patient(key: str) -> DemoPatient:
    """Look up a demo patient, failing loudly on an unknown key."""
    try:
        return DEMO_PATIENTS[key]
    except KeyError:
        raise KeyError(
            f"unknown demo patient {key!r}; known keys are "
            f"{sorted(DEMO_PATIENTS)}") from None


def load_demo_patient(key: str = DEFAULT_DEMO_PATIENT,
                      builder: Optional[Any] = None):
    """The profile for a demo patient, with its disclosures attached."""
    return demo_patient(key).load(builder=builder)
