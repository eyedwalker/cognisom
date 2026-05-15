# MAD-pipeline ↔ Patent-pipeline Integration Audit

**Date:** 2026-05-14
**Auditor:** Claude Opus 4.7 (1M context), at David Walker's direction
**Purpose:** Estimate integration cost for the SU2C/PCF 429-patient validation study (see `RESUME.md`, patentability review 2026-05-14)

---

## TL;DR

The two pipelines are a **hard fork at the data-type level** but share the one component that actually matters (MHCflurry scoring). The parsing gap is small. Phase 1 integration is **~1-2 weeks of engineering, not 1-2 months** as previously estimated.

**Revised end-to-end timeline for the SU2C validation:**

| Task | Engineer-days |
|---|---|
| Variant → patent-mutation parser + tests | 0.5 |
| Cohort iterator / batch runner | 2–3 |
| `ImmuneClassification` + wire patent output into MAD `ImmuneAgent` | 1–2 |
| Reproducible SU2C-validation pytest | 3–5 |
| Statistical analysis (ROC, KM, subgroup) | 3–5 |
| Paper draft + revisions | 15–30 |
| **Total** | **~25–45 days = 1.5–2 months FTE** |

---

## What we have

### MAD-pipeline data types

**`PatientProfile`** — `cognisom/genomics/patient_profile.py:25-56`
- Fields: `variants`, `coding_variants`, `cancer_driver_mutations`, `affected_genes`, `tumor_mutational_burden`, `msi_status`, `hla_alleles`, `predicted_neoantigens`
- Built via `PatientProfileBuilder.from_vcf_file(vcf_path, patient_id)` → `_build_profile(...)` (lines 270-348)

**`Variant`** — `cognisom/genomics/vcf_parser.py:21-86`
- Core fields: `chrom, pos, ref, alt, gene, protein_change, consequence, is_coding, is_cancer_driver, impact`
- `protein_change` is HGVS format: `"p.G12D"`

### MAD top-level entry

`BoardModerator.run_full_analysis(patient_id, profile, twin, treatment_results, classification) → BoardDecision`
- `cognisom/mad/board.py:242-278`
- Instantiates 3 agents (GenomicsAgent, ImmuneAgent, ClinicalAgent), each calls `.analyze()`, then `convene()` synthesizes into consensus

### 3-Agent architecture

| Agent | Location | Responsibility |
|---|---|---|
| GenomicsAgent | `cognisom/mad/agents.py:92-399` | TMB, MSI, HRD, AR mutations, neoantigen vaccine eligibility; scores pembrolizumab, olaparib, enzalutamide, neoantigen_vaccine, combos |
| ImmuneAgent | `cognisom/mad/agents.py:400+` | (Signature only fully read at entry: `analyze(twin, profile, classification)`) — TME classification, T-cell exhaustion |
| ClinicalAgent | `cognisom/mad/agents.py:600+` | Simulates treatment outcomes using digital twin |

### HLA handling

- `HLATyper.type_from_variants(variants, patient_id)` extracts alleles from chr6 variants (`hla_typer.py:140-170`)
- Format: `"HLA-A*02:01"` — **matches the patent pipeline format exactly** (see `engine/py/immune/mhc_loading.py:46`)
- Default demo profile: `["HLA-A*02:01", "HLA-A*24:02", "HLA-B*07:02", "HLA-B*44:03", "HLA-C*05:01", "HLA-C*07:02"]`

### Already-shared scoring component

**Both pipelines share the MHCflurry scorer instance.**

- MAD: `cognisom/genomics/neoantigen_predictor.py:325-346` → `_predict_binding(peptide, hla_allele)` tries MHCflurry first, PWM fallback
- Patent: `engine/py/immune/mhc_loading.py:102-114` → `MHCLoader.__init__()` instantiates `NeoantigenPredictor()` and calls `._predict_binding(...)` on it

This means **identical MHC-binding scores** for the same peptide-HLA pair across both pipelines. No scorer divergence risk.

### SU2C cohort assets

- `cognisom/validation/su2c_file_validator.py:1-65` — loads flat files (`data_mutations.txt`, `data_clinical_patient.txt`, `data_clinical_sample.txt`), provides `SU2CPatient` and `SU2CValidationResult` dataclasses
- `cognisom/dashboard/_pages/36_validation_demo.py` — cohort iterator UI + single-patient demo
- Citation: Abida et al., PNAS 2019; download URL in the validator file

**Caveat:** No automated test reproduces the displayed "TMB r=0.987, 100% biomarker concordance" claim. The metric appears to have been recorded once from an offline run.

---

## The actual integration gaps

### Gap 1 — Variant → patent-pipeline mutation parsing

The MAD `Variant.protein_change` is HGVS (`"p.G12D"`). The patent pipeline's `molecular.introduce_mutation()` expects bare mutation name (`"G12D"`). MAD's neoantigen predictor already does this parse internally (`neoantigen_predictor.py:178-186`) but the adapter isn't exposed as a function.

**Cost:** ~20 lines (a `variant_to_patent_mutation(variant) -> Tuple[str, str]` helper that strips `"p."`, regex-extracts the form, and validates against `Gene.ONCOGENIC_SUBSTITUTIONS`). **Half a day.**

### Gap 2 — Cohort iterator → patent-pipeline batch runner

For each driver mutation in `profile.cancer_driver_mutations`:
- Instantiate `SimulationEngine`
- Call `molecular.introduce_mutation(cell_id, gene_name, mutation_name)`
- Collect output: TME type, exhaustion fraction, ECM-excluded flag, displayed peptides, kill events
- Serialize to a row in a DataFrame

**Cost:** ~50-80 lines + retry/error handling for unsupported mutations (frameshifts not yet integrated end-to-end, fusions, non-curated genes). **2-3 days.**

### Gap 3 — Wire patent-pipeline output as evidence into `ImmuneAgent`

`ImmuneAgent.analyze(..., classification=None)` already accepts a `classification` parameter as an unused hook (`cognisom/mad/agents.py:256`, `cognisom/mad/board.py:248`). **The integration point was anticipated in the design.**

Concrete additions:
- New `ImmuneClassification` dataclass with `tme_type`, `exhaustion_fraction`, `ecm_excluded_fraction`, `predicted_icb_response`
- `ImmuneAgent.analyze()` reads it, modulates the checkpoint-inhibitor score, appends evidence items to `AgentOpinion.evidence_items`

**Cost:** ~40-60 lines. **1-2 days.**

### Gap 4 — Reproducible SU2C-validation pytest

The "r=0.987 TMB correlation, 100% biomarker concordance" claim is on the cognisom.com landing page and `cognisom/dashboard/app.py:429` but is **not validated by any pytest**. Reproducibility risk for the patent disclosure.

**Cost:** ~150-200 lines for a `tests/test_su2c_validation.py` that loads the cohort, runs the full pipeline, computes metrics, asserts the claim. Plus a CI flag to skip by default (cohort access is via dbGaP IRB). **3-5 days.**

---

## Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| dbGaP IRB takes >6 months | Medium | File early via academic collaborator; backup cohort (PCF Atlas, TCGA-PRAD) |
| HLA typing not in public release | Low | OptiType from germline FASTQ (~$50 compute per patient) |
| MHCflurry version drift between pipelines | Medium | Pin version in `requirements.txt` — see "this-week tasks" below |
| Patent-pipeline simulation doesn't predict ICB response (AUC < 0.65) | Medium | Pre-register SAP on OSF to publish as null; pivot to PARP / HRR sub-analysis |
| Two papers from one validation cause data-snooping critique | Low | Pre-register both analyses; statistical multiplicity correction (Benjamini-Hochberg) |
| ImmuneAgent integration introduces regression in MAD validation | Medium | Add patent-pipeline path as opt-in (`classification=None` default preserves behaviour) |

---

## Dependency disagreements

- **Python version:** Not explicitly pinned in `requirements.txt`
- **NumPy / Pandas:** `numpy>=1.24.0`, `pandas>=2.0.0` in both — no upper bound, drift risk
- **MHCflurry:** **NOT listed in any requirements file** — purely optional import with graceful fallback. Two separate installations could have v1.6 vs v2.0 with different model accuracies. **Highest priority to fix.**
- **PyTorch / TensorFlow:** Commented out in `requirements.gpu.txt:22-23`; Keras pulled in transitively via MHCflurry

---

## Recommended sequencing

**This week (no-regrets, ~1 day):**

1. Pin MHCflurry version in `requirements.txt` (`mhcflurry==2.0.5` or whatever production has)
2. Add `cognisom/genomics/mutation_adapter.py` with `variant_to_patent_mutation()` as a tested module — standalone value beyond SU2C validation (e.g., for the Patent Pipeline dashboard page when users upload their own VCFs)

**Week 1 (integration scaffolding, ~3 days):**

3. Cohort iterator / batch runner
4. `ImmuneClassification` + wire into `ImmuneAgent`
5. Smoke test on 5-10 patients

**Week 2 (validation harness, ~3-5 days):**

6. Reproducible SU2C-validation pytest
7. Pre-register Statistical Analysis Plan on OSF
8. Prepare statistical-analysis notebooks (ROC, KM, Cox)

**Weeks 3-8 (during IRB wait):**

9. Draft methods section of paper(s)
10. Identify academic collaborator (oncology biostatistician at CU Anschutz given V. Chen lecture connection)
11. Submit dbGaP IRB application

**Weeks 9-16 (after IRB approval):**

12. Run cohort end-to-end on standalone patent-pipeline (Option A)
13. Run cohort end-to-end on MAD-board-with-patent-evidence (Option B)
14. Statistical analysis
15. Two paper drafts

**Total:** 4-6 months to publication-ready manuscripts, two reduction-to-practice citations for the patent disclosure.

---

## Honest uncertainty flags

1. **ImmuneAgent implementation not fully read.** Only `GenomicsAgent` was reviewed line-by-line. ImmuneAgent + ClinicalAgent beyond ~line 600 are inferred from signatures.
2. **TMB r=0.987 not reproducible.** The claim is in the dashboard but no automated test recreates it. Unclear if the manual run is the published number or an approximation.
3. **Frameshift/fusion integration into the closed-loop simulator is staged but not complete.** Upgrade 8 generates frameshift/fusion peptides but they don't yet flow through MHC scoring in the closed-loop end-to-end path. SU2C cohort patients with frameshift mutations would be subset-excluded from the primary endpoint analysis.
4. **No lock file** (poetry.lock, Pipfile.lock). Exact reproducibility is not guaranteed across machines.

---

## What the audit changes about the recommendation

Original framing recommendation (2026-05-14 morning): "Do Option A standalone validation first, then Option B MAD integration."

**Audit-corrected recommendation:** Do them together. The integration cost is small enough (~3-5 engineer-days for Gap 3) that decoupling is artificial. Two papers from one cohort run, both reduction-to-practice citations for the patent disclosure.

---

## Source references

- Audit conducted via Explore agent + verified by grep + file reads
- All file:line citations are accurate as of commit `69ea2dc` (2026-05-14)
- See also: `DECISIONS.md` (inventorship log), `RESUME.md` (current state), `docs/patent/DISCLOSURE_SOURCE.md` (attorney-facing disclosure — stale per the 2026-05-14 patentability review)
