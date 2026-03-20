"""
Page 39: Pipeline Guide — DNA to Decision Deep Dive
=====================================================

Comprehensive documentation of the complete Cognisom pipeline from
raw sequencing reads to immunotherapy recommendation.

Covers:
  - Parabricks GPU pipeline (alignment → variant calling)
  - Matched tumor-normal somatic workflow
  - HLA typing and neoantigen prediction
  - MAD Agent multi-agent deliberation
  - Evidence sources and FDA compliance
"""

import streamlit as st
from cognisom.dashboard.page_config import safe_set_page_config

safe_set_page_config(page_title="Pipeline Guide", page_icon="📘", layout="wide")

from cognisom.auth.middleware import streamlit_page_gate
user = streamlit_page_gate("39_pipeline_guide")

st.title("📘 Pipeline Guide: DNA to Decision")
st.caption(
    "Complete documentation of the Cognisom precision oncology pipeline. "
    "From raw DNA sequencing to FDA-compliant immunotherapy recommendation."
)

# ─────────────────────────────────────────────────────────────────────────
# OVERVIEW
# ─────────────────────────────────────────────────────────────────────────

st.markdown("""
## The Complete Pipeline

The Cognisom pipeline transforms raw DNA sequencing data into a traceable
immunotherapy recommendation through **8 stages across 4 layers**.
Each stage has a specific purpose, and skipping any stage reduces accuracy.

---
""")

# ─────────────────────────────────────────────────────────────────────────
# LAYER 1: SEQUENCING & ALIGNMENT (GPU)
# ─────────────────────────────────────────────────────────────────────────

st.markdown("## Layer 1: The Personal Baseline (Normal DNA)")
st.info(
    "**Purpose:** Establish the patient's inherited genetic background — "
    "the 'locks' on their immune cells (HLA alleles) and their germline "
    "variants that are NOT cancer."
)

with st.expander("Step 1: Sample Unmapped Reads (FASTQ Input)", expanded=True):
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### What are FASTQ files?

        FASTQ files contain the **raw output of a DNA sequencer**. Each file holds
        millions of short DNA sequences ("reads") — typically 150 base pairs each —
        along with quality scores for every base.

        **For a 30x whole-genome sequencing run:**
        - ~1 billion read pairs (2 files: R1 and R2)
        - ~90-100 GB total file size (gzipped)
        - Covers the entire 3.2 billion base pair human genome ~30 times

        **Why paired-end (R1 + R2)?**
        The sequencer reads each DNA fragment from both ends. This creates two
        files that, when aligned together, give much higher confidence about
        where each fragment belongs in the genome. A single read might map to
        multiple locations, but a pair with the correct insert size maps uniquely.

        **Input format:** `s3://cognisom-genomics/fastq/{patient_id}/R1.fastq.gz`
        """)
    with col2:
        st.markdown("""
        #### Key Specs
        | Parameter | Value |
        |-----------|-------|
        | Format | FASTQ (gzipped) |
        | Read length | 150 bp |
        | Coverage | 30x (germline) |
        | Size | ~90 GB per sample |
        | Encoding | Phred+33 |
        | Platform | Illumina NovaSeq |
        """)

with st.expander("Step 2: BWA-MEM Alignment to GRCh38"):
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### What happens here?

        **BWA-MEM** (Burrows-Wheeler Aligner) takes each 150 bp read and finds
        where it belongs in the **GRCh38 human reference genome** (3.2 billion
        base pairs). This is like taking a billion puzzle pieces and placing
        each one in the correct position on a massive jigsaw puzzle.

        **Why GPU acceleration matters:**
        - CPU (24 cores): ~8 hours for 30x WGS
        - GPU (Parabricks on L40S): ~25 minutes — **19x faster**

        Parabricks reimplements BWA-MEM to run on NVIDIA GPUs, processing
        thousands of reads simultaneously instead of one-at-a-time.

        **The reference genome (GRCh38/hg38):**
        The "map" that reads are aligned against. GRCh38 is the latest
        human reference assembly (Genome Reference Consortium, 2013, patched
        through p14). It represents a "consensus" human genome — NOT any
        individual patient. Differences from this reference are "variants."
        """)
    with col2:
        st.markdown("""
        #### Performance
        | Platform | Time | Speedup |
        |----------|------|---------|
        | CPU 24-core | ~8 hrs | 1x |
        | L40S (48GB) | ~25 min | **19x** |
        | A100 (80GB) | ~15 min | **32x** |
        | HealthOmics | ~25 min | Managed |

        #### Output
        Aligned BAM file (~60-80 GB)
        with reads mapped to chromosomal
        positions.
        """)

with st.expander("Step 3: Co-ordinate Sorting"):
    st.markdown("""
    ### Why sort?

    After alignment, reads are in the order they came off the sequencer (random).
    **Co-ordinate sorting** reorders them by genomic position (chr1:1, chr1:2, ...,
    chrX:end). This is required for all downstream tools — you can't call variants
    efficiently without knowing which reads overlap the same position.

    **On GPU:** This is part of Parabricks `fq2bam` — sorting happens in GPU memory,
    ~100x faster than `samtools sort` on CPU.
    """)

with st.expander("Step 4: Mark Duplicates"):
    st.markdown("""
    ### What are duplicates and why remove them?

    During library preparation (before sequencing), DNA fragments are amplified
    by PCR. This creates **duplicate reads** — multiple copies of the same original
    fragment that look like independent observations but aren't.

    If you don't mark duplicates, you'll **overcount** the evidence for a variant.
    A real heterozygous variant should show ~50% of reads with the alternate allele.
    With duplicates, it might show 80% — making you falsely confident.

    **Parabricks** identifies duplicates by comparing read positions and
    orientations. Duplicates are flagged (not removed) in the BAM file.

    **Impact:** Typically 10-30% of reads are duplicates in a standard library.
    """)

with st.expander("Step 5: Base Quality Score Recalibration (BQSR)"):
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Why recalibrate quality scores?

        The sequencer assigns a quality score to each base (Q30 = 99.9% accuracy,
        Q20 = 99% accuracy). But these scores have **systematic biases** — certain
        sequence contexts, positions in the read, or machine cycles produce
        consistently over- or under-estimated quality scores.

        **BQSR** (Base Quality Score Recalibration) corrects these biases by:
        1. Comparing observed mismatches to known variant sites (dbSNP, known indels)
        2. Building a statistical model of quality score errors
        3. Recalibrating every base's quality score

        **Why it matters for variant calling:** DeepVariant and other callers use
        quality scores to weight evidence. If Q30 bases are actually Q20, you'll
        call false-positive variants. BQSR ensures the quality scores are truthful.

        **Known Sites input:** A VCF of known polymorphisms (dbSNP + known indels)
        so BQSR doesn't treat real variants as errors.
        """)
    with col2:
        st.markdown("""
        #### Inputs
        - Aligned BAM (from Step 2-4)
        - Known Sites VCF (optional)
          - `dbSNP_151.vcf.gz`
          - `known_indels.vcf.gz`

        #### Outputs
        - Recalibrated BAM
        - BQSR Report (quality metrics)

        #### Impact
        Reduces false-positive
        variant calls by ~5-10% in
        difficult genomic regions.
        """)

with st.expander("Step 6: DeepVariant (Germline Variant Calling)"):
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### The core science: finding variants

        **DeepVariant** is a deep learning-based variant caller developed by Google.
        It treats variant calling as an **image classification problem**:

        1. For each candidate position, it creates a "pileup image" showing all
           reads aligned to that position (reference bases, alternate bases,
           quality scores, strand orientation)
        2. A **convolutional neural network (CNN)** classifies each position as:
           - **Homozygous reference** (0/0) — no variant
           - **Heterozygous** (0/1) — one copy of the variant
           - **Homozygous alternate** (1/1) — two copies of the variant
        3. Output: A VCF file with all variant calls and confidence scores

        **Why DeepVariant over GATK HaplotypeCaller?**
        - Higher accuracy (F1 >99.7% for SNPs on GIAB benchmark)
        - Better indel calling (F1 >99.2%)
        - GPU-accelerated via Parabricks (~20 min vs ~6 hours CPU)

        **For a typical 30x WGS:** Calls ~4.5 million variants (SNPs + indels).
        Most are common population variants. Only a tiny fraction are
        disease-relevant.

        **Parabricks GPU acceleration:** DeepVariant's CNN inference runs
        natively on NVIDIA GPUs, processing thousands of pileup images
        in parallel.
        """)
    with col2:
        st.markdown("""
        #### Output: VCF File
        ```
        #CHROM POS  REF ALT QUAL FILTER
        chr1   1234 A   G   45   PASS
        chr7   5678 CT  C   38   PASS
        ```

        #### Accuracy (GIAB v4.2.1)
        | Type | Precision | Recall | F1 |
        |------|-----------|--------|-----|
        | SNP | 99.96% | 99.94% | 99.95% |
        | Indel | 99.60% | 99.41% | 99.51% |

        #### Typical Output
        - ~4.5M total variants
        - ~3.5M SNPs
        - ~1M indels
        - ~50-100 in cancer genes
        """)

st.divider()

# ─────────────────────────────────────────────────────────────────────────
# LAYER 2: SOMATIC CALLING (MATCHED TUMOR-NORMAL)
# ─────────────────────────────────────────────────────────────────────────

st.markdown("## Layer 2: The Malignant Shift (Tumor DNA)")
st.warning(
    "**Purpose:** Find mutations that exist ONLY in the tumor — the cancer's "
    "'keys' that might unlock an immune response. This requires comparing "
    "tumor DNA against the patient's normal DNA."
)

with st.expander("Step 7: Mutect2 Somatic Variant Calling (Matched Tumor-Normal)"):
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### The gold standard: tumor vs. normal

        **Mutect2** (GATK, GPU-accelerated by Parabricks) is the industry-standard
        somatic variant caller. It works fundamentally differently from germline
        callers:

        **The logic:**
        For each genomic position, Mutect2 compares the reads from the tumor
        sample against the reads from the matched normal sample:

        - **Variant in TUMOR only** → **Somatic mutation** (cancer-caused)
        - **Variant in BOTH** → **Germline variant** (inherited, not cancer)
        - **Variant in NORMAL only** → Likely artifact (very rare in reality)

        **Why matched normal is essential:**
        Without matched normal, you can't distinguish somatic from germline.
        Population databases (gnomAD) help filter common germline variants,
        but rare germline variants (present in <0.1% of the population) look
        identical to somatic mutations. This causes **~30% more false positives**
        in tumor-only analysis.

        **What Mutect2 also detects:**
        - **Variant Allele Frequency (VAF):** The fraction of reads showing
          the variant. Low VAF (5-10%) suggests a subclonal mutation present
          in only a fraction of tumor cells.
        - **Tumor purity:** Adjusts for normal cell contamination in the biopsy.
        - **Strand bias:** Filters artifacts that appear on only one DNA strand.

        **Somatic pipeline on Parabricks:**
        1. Align tumor FASTQs → tumor BAM (~25 min GPU)
        2. Align normal FASTQs → normal BAM (~25 min GPU, parallel with step 1)
        3. Mutect2: tumor BAM + normal BAM → somatic VCF (~45 min GPU)
        **Total: ~95 min on GPU** vs ~40 hours on CPU.
        """)
    with col2:
        st.markdown("""
        #### Somatic vs Germline
        | Feature | Germline | Somatic |
        |---------|----------|---------|
        | Input | 1 sample | 2 samples |
        | Time | ~45 min | ~95 min |
        | Output | All variants | Cancer-only |
        | False+ rate | Very low | Higher |
        | VAF | 50%/100% | 5-100% |
        | Cost (HO) | ~$14 | ~$27 |

        #### Why This Matters
        A patient with 4.5M germline
        variants might have only
        **50-500 somatic mutations**.

        Finding those 50 in 4.5M
        is the whole point of
        matched tumor-normal.
        """)

st.divider()

# ─────────────────────────────────────────────────────────────────────────
# LAYER 3: BINDING LOGIC (IMMUNE MATCHING)
# ─────────────────────────────────────────────────────────────────────────

st.markdown("## Layer 3: The Binding Logic (Lock and Key)")
st.success(
    "**Purpose:** Determine if the cancer's mutations produce peptides that "
    "fit into the patient's immune system 'locks' (HLA alleles). If the "
    "key fits the lock, the immune system CAN see the cancer."
)

with st.expander("Step 8: HLA Typing (The Immune Locks)"):
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### What are HLA alleles?

        **HLA (Human Leukocyte Antigen)** proteins sit on the surface of every
        cell in your body. They act as "display cases" that present short peptide
        fragments (8-14 amino acids) to T-cells. If a T-cell recognizes a
        displayed peptide as "foreign" (e.g., from a virus or cancer mutation),
        it triggers an immune attack.

        **The critical insight:** Every person has a unique set of 6 HLA-I alleles
        (2 each of HLA-A, HLA-B, HLA-C). Each allele has a different binding
        groove that only fits certain peptide shapes. This means:
        - **Same mutation, different patients** → different immune visibility
        - **Patient A** with HLA-A*02:01 might present a KRAS G12D peptide
        - **Patient B** with HLA-A*24:02 might NOT present that same peptide

        **How OptiType works:**
        1. Extract reads from the HLA region (chromosome 6, 29-34 Mb)
        2. Align to a database of ~7,000 known HLA allele sequences
        3. Integer Linear Programming solver finds the optimal 6-allele combination
        4. Output: 4-digit resolution typing (e.g., HLA-A*02:01)

        **Accuracy:** >97% concordance with serological typing.

        **Current Cognisom status:** Population-frequency assignment (placeholder).
        OptiType integration built, pending installation on GPU server.
        """)
    with col2:
        st.markdown("""
        #### Example HLA Profile
        | Locus | Allele 1 | Allele 2 |
        |-------|----------|----------|
        | HLA-A | A*02:01 | A*03:01 |
        | HLA-B | B*07:02 | B*44:02 |
        | HLA-C | C*05:01 | C*07:02 |

        #### Why It Matters
        HLA-A*02:01 (29% of people)
        presents different peptides
        than HLA-A*24:02 (10%).

        **Wrong HLA = wrong
        neoantigen predictions =
        wrong vaccine targets.**

        #### MHCflurry Coverage
        **14,847 HLA alleles**
        supported for binding
        prediction.
        """)

with st.expander("Step 9: Neoantigen Prediction (The Cancer Keys)"):
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Finding the keys that fit the locks

        A **neoantigen** is a mutant peptide from the tumor that:
        1. Binds to the patient's HLA allele (predicted IC50 < 500 nM)
        2. Is different enough from the wildtype peptide to be "foreign"
        3. Can be processed and presented on the cell surface

        **The prediction pipeline:**
        1. Take each somatic mutation → apply to the protein sequence
        2. Generate all overlapping 8-11 amino acid peptides around the mutation
        3. Score each peptide against each of the patient's 6 HLA alleles
        4. Rank by binding affinity (lower IC50 = stronger binding)

        **MHCflurry 2.0 (integrated):**
        - Neural network trained on >400,000 experimental binding measurements
        - Predicts: IC50 (nM), percentile rank, and presentation score
        - **14,847 alleles supported** — covers virtually all human HLA types
        - AUC >0.95 for most alleles

        **Key thresholds:**
        - **Strong binder:** IC50 < 50 nM (very likely presented)
        - **Weak binder:** IC50 < 500 nM (possibly presented)
        - **Non-binder:** IC50 > 500 nM (unlikely to trigger immune response)

        **Agretopicity:** Ratio of wildtype/mutant binding affinity.
        If >1.0, the mutant peptide binds BETTER than wildtype — the immune
        system is more likely to recognize it as "new."

        **Vaccine candidate selection:**
        Top 20 neoantigens with strong/weak binding + agretopicity >=1.0
        are selected as potential mRNA vaccine targets.
        """)
    with col2:
        st.markdown("""
        #### Example Neoantigen
        ```
        Peptide:   YMSLITMAI
        Gene:      FOXA1
        HLA:       HLA-A*02:01
        Affinity:  12.8 nM (strong)
        Agretop:   3.2 (selective)
        Vaccine:   YES
        ```

        #### Binding Strength
        | Category | IC50 | % of peptides |
        |----------|------|--------------|
        | Strong | <50 nM | ~1-2% |
        | Weak | <500 nM | ~5-10% |
        | Non-binder | >500 nM | ~90% |

        #### Clinical Context
        Moderna's mRNA-4157 vaccine
        targets 34 patient-specific
        neoantigens per dose
        (KEYNOTE-942 trial).
        """)

st.divider()

# ─────────────────────────────────────────────────────────────────────────
# LAYER 4: EVIDENCE & DECISION (MAD AGENT)
# ─────────────────────────────────────────────────────────────────────────

st.markdown("## Layer 4: External Evidence & Decision")
st.markdown(
    "**Purpose:** Match the patient's genomic findings against regulatory "
    "databases, clinical trials, and treatment evidence to produce a "
    "traceable, FDA-compliant recommendation."
)

with st.expander("Step 10: OncoKB Variant Actionability"):
    st.markdown("""
    ### Is this mutation clinically actionable?

    **OncoKB** (Memorial Sloan Kettering) is the premier precision oncology
    knowledge base. It annotates every known cancer variant with:

    - **Oncogenicity:** Is this a real cancer driver or a passenger?
    - **Evidence level:** How strong is the drug association?
      - Level 1: FDA-approved drug for this biomarker
      - Level 2: Standard care (NCCN guidelines)
      - Level 3A: Clinical evidence from trials
      - Level 3B: Emerging evidence
      - Level 4: Preclinical evidence
      - Level R1: Resistance biomarker (standard care)
    - **Drug associations:** Which drugs target this variant?

    **Example:** BRCA2 frameshift → Level 1 → Olaparib (PROfound trial)

    **700+ genes, 5,000+ variants** annotated with evidence levels.

    **Cognisom integration:** OncoKB fallback knowledge base (14 prostate
    cancer genes) + live API when token is configured.
    """)

with st.expander("Step 11: MAD Board (Multi-Agent Deliberation)"):
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Three AI agents deliberate like a tumor board

        The **MAD Board** (Molecular AI Decision) convenes three specialist
        agents that independently analyze the patient:

        **Genomics Agent:** Focuses on driver mutations and biomarkers.
        "This patient has BRCA2 loss → PARP inhibitors have Level 1 evidence."

        **Immune Agent:** Focuses on the tumor microenvironment.
        "This patient has a hot immune microenvironment with 45% exhausted
        T-cells → PD-1 blockade can reactivate them."

        **Clinical Agent:** Focuses on treatment simulation results.
        "Olaparib shows 75% tumor reduction with 180-day PFS and only 5% irAE risk."

        **The Board Moderator** then:
        1. Collects all three opinions
        2. Computes an agreement matrix (who agrees on what)
        3. Determines consensus level (unanimous / majority / split)
        4. Synthesizes a unified rationale with evidence citations
        5. Flags dissenting views and warnings

        **Every recommendation is traceable** — each evidence item links
        to a clinical trial (NCT number), guideline (NCCN), or FDA label.
        """)
    with col2:
        st.markdown("""
        #### Consensus Levels
        | Level | Meaning |
        |-------|---------|
        | Unanimous | All 3 agents agree |
        | Majority | 2 of 3 agree |
        | Split | No agreement |

        #### Evidence Sources
        - OncoKB (700+ genes)
        - FDA Biomarker Table
        - KEYNOTE trials
        - PROfound (PARP)
        - CheckMate-650
        - NCCN Guidelines
        - ClinicalTrials.gov

        #### Output
        - Recommended treatment
        - Alternative options
        - Matching clinical trials
        - Evidence chain (traceable)
        - Dissenting views
        - Audit record (immutable)
        """)

with st.expander("Step 12: Clinical Trial Matching"):
    st.markdown("""
    ### Finding recruiting trials for this patient

    The pipeline queries **ClinicalTrials.gov** (V2 API) to find actively
    recruiting trials that match the patient's biomarkers:

    - **BRCA mutation** → PARP inhibitor trials
    - **TMB-high** → Checkpoint inhibitor trials
    - **MSI-H** → Immunotherapy trials
    - **Neoantigen-rich** → Vaccine trials

    **Example output:**
    - NCT06952803: Phase 3 saruparib in HRD prostate cancer
    - NCT03452774: AI-based precision oncology trial (SYNERGY-AI)

    No authentication required. Results are live from the FDA's registry.
    """)

st.divider()

# ─────────────────────────────────────────────────────────────────────────
# COST & TIME SUMMARY
# ─────────────────────────────────────────────────────────────────────────

st.markdown("## Cost & Time Summary")

import pandas as pd

cost_df = pd.DataFrame({
    "Step": [
        "BWA-MEM Alignment (per sample)",
        "DeepVariant Germline Calling",
        "Mutect2 Somatic (tumor vs normal)",
        "TOTAL: Germline Pipeline",
        "TOTAL: Matched Somatic Pipeline",
        "HLA Typing (OptiType)",
        "Neoantigen Prediction (MHCflurry)",
        "MAD Board + Clinical Report",
    ],
    "GPU/CPU": ["GPU", "GPU", "GPU", "GPU", "GPU", "CPU", "CPU", "CPU"],
    "L40S Time": ["~25 min", "~20 min", "~45 min", "~45 min", "~95 min", "~5 min", "~2 min", "<1 sec"],
    "HealthOmics Time": ["~25 min", "~20 min", "~45 min", "~45 min", "~2 hrs", "N/A", "N/A", "N/A"],
    "HealthOmics Cost": ["~$8.84", "~$5", "~$9", "~$14", "~$27", "—", "—", "—"],
    "L40S Cost": ["~$0.77", "~$0.61", "~$1.38", "~$1.38", "~$2.90", "~$0.15", "~$0.06", "~$0.00"],
})

st.dataframe(cost_df.set_index("Step"), use_container_width=True)

st.caption(
    "L40S costs based on g6e.2xlarge ($1.836/hr). "
    "HealthOmics costs from AWS Ready2Run pricing. "
    "CPU steps run on apps-server (always-on, no additional cost)."
)

# ─────────────────────────────────────────────────────────────────────────
# FDA COMPLIANCE
# ─────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown("## FDA 7-Step Credibility Framework Alignment")

steps_data = {
    "Step": [
        "1. Question of Interest",
        "2. Context of Use",
        "3. Risk Assessment",
        "4. Credibility Plan",
        "5. Execute Plan",
        "6. Document Results",
        "7. Adequacy Assessment",
    ],
    "Cognisom Implementation": [
        "Which immunotherapy is most likely to produce a response in this mCRPC patient?",
        "Non-Device CDS: evidence for clinician's independent review",
        "Medium risk (clinician-in-loop) × Medium consequence (therapy selection)",
        "Retrospective validation on SU2C (429 pts) + TCGA (494 pts) + GIAB benchmark",
        "429-patient study complete: 100% biomarker concordance, TMB r=0.987",
        "Model cards, audit trail, provenance tracking, 42 unit tests",
        "Pending formal review — evidence supports defined Context of Use",
    ],
    "Status": ["Defined", "Defined", "Defined", "Defined", "Executed", "Executed", "Pending Review"],
}

st.dataframe(pd.DataFrame(steps_data).set_index("Step"), use_container_width=True)

st.divider()
st.caption("FOR RESEARCH USE ONLY — Not for clinical decision-making without independent physician review.")
