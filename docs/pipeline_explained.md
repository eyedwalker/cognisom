# Cognisom Pipeline — How Each Tool Works

## The Full Pipeline

```
Patient DNA Sample
    |
    v
[1. Sequencing] → FASTQ files (raw reads)
    |
    v
[2. Alignment] → BAM file (reads mapped to genome)
    |
    v
[3. Variant Calling] → VCF file (mutations found)
    |
    v
[4. Annotation] → Which genes are affected, protein changes
    |
    v
[5. HLA Typing] → Patient's immune receptor type
    |
    v
[6. Neoantigen Prediction] → Which mutations the immune system can see
    |
    v
[7. MAD Agent Board] → Treatment recommendation
    |
    v
[8. 3D Visualization] → See the drug target
```

---

## 1. DNA Sequencing (External Lab)

**What it does:** Reads the patient's DNA letter by letter.

**How it works:**
- A DNA sample (blood for germline, biopsy for tumor) is fragmented into
  millions of short pieces (~150 base pairs each)
- Each fragment is read by a sequencing machine (Illumina NovaSeq, etc.)
- The machine produces two files per sample:
  - R1.fastq.gz — the "forward" read of each fragment
  - R2.fastq.gz — the "reverse" read of each fragment
- Each read includes the DNA sequence (A, C, G, T letters) plus a quality
  score for each base (how confident the machine is)

**Output:** FASTQ files. A 30x whole genome = ~27 GB compressed, ~1 billion read pairs.

**Analogy:** Imagine shredding 1,000 copies of the same book, then reading
each shred. You need many copies (30x coverage) to reconstruct the original.

---

## 2. Alignment — NVIDIA Parabricks fq2bam

**What it does:** Figures out where each DNA fragment came from in the genome.

**How it works:**
- Takes the ~1 billion short reads from the FASTQ files
- Compares each 150-letter read against the 3.2-billion-letter human
  reference genome (GRCh38)
- Uses BWA-MEM algorithm (GPU-accelerated by Parabricks) to find the best
  matching position for each read
- Accounts for small differences (mutations, insertions, deletions)
- Sorts all reads by their genome position
- Marks duplicate reads (PCR artifacts from library preparation)
- Recalibrates base quality scores (BQSR) using known variant sites

**GPU acceleration:** BWA-MEM on CPU takes ~6 hours. Parabricks on L40S GPU
does it in ~25 minutes — the string matching is massively parallelizable.

**Output:** BAM file (Binary Alignment Map). Contains every read with its
genome position, quality, and alignment details. ~35-80 GB per sample.

**Analogy:** You have 1 billion book shreds. Alignment is figuring out which
page and which line each shred came from by matching the text against the
complete original book.

---

## 3a. Germline Variant Calling — DeepVariant

**What it does:** Finds inherited DNA variants (SNPs + indels) in a single sample.

**How it works:**
- Looks at each position in the genome
- At each position, counts how many reads show the reference base vs.
  an alternative base
- Uses a Convolutional Neural Network (CNN) — same type of AI used for
  image recognition — to classify each position as:
  - Homozygous reference (0/0): both copies match reference
  - Heterozygous variant (0/1): one copy differs
  - Homozygous variant (1/1): both copies differ
- The CNN was trained on millions of validated variants from GIAB
  (Genome in a Bottle) gold standard datasets
- It "sees" the read pileup as an image: rows are reads, columns are
  positions, colors encode base identity and quality

**Why CNN for variant calling?** Traditional tools use statistical models
(Bayesian). DeepVariant's CNN can learn complex patterns that statistical
models miss — like strand bias, mapping artifacts, and systematic errors.

**Output:** VCF file with ~3.5-4 million variants per human genome.

**Analogy:** Looking at 30 photocopies of the same page, if 15 copies show
"A" and 15 show "G" at one position, that's a heterozygous variant —
you inherited different versions from each parent.

---

## 3b. Somatic Variant Calling — Mutect2 (mutectcaller)

**What it does:** Finds cancer-specific mutations by comparing tumor DNA
against the patient's own normal DNA.

**How it works:**
- Takes two BAM files: tumor (from biopsy) and normal (from blood)
- At each genome position, compares the read patterns:
  - Present in TUMOR but NOT in NORMAL → **somatic mutation** (cancer-specific)
  - Present in BOTH → **germline variant** (inherited, not cancer-related)
  - Present only in NORMAL → sequencing artifact or clonal hematopoiesis
- Uses a Bayesian somatic model that accounts for:
  - Tumor purity (what % of cells in the biopsy are actually cancer)
  - Tumor ploidy (cancer cells often have extra/missing chromosomes)
  - Allele frequency (what fraction of reads show the mutation)
  - Strand bias and base quality artifacts
- Low allele frequency variants (5-10%) are the hardest to detect — these
  may represent subclonal mutations present in only some cancer cells

**Why matched normal matters:** Without the normal sample, you can't tell
if a variant is a cancer mutation or something the patient was born with.
Tumor-only calling has ~30% more false positives.

**Output:** Somatic VCF — typically 1,000-50,000 true somatic mutations
(after filtering), depending on cancer type and mutational burden.

**Analogy:** You have two versions of a document — the original (normal)
and one that's been edited (tumor). By comparing them side-by-side, you
can find exactly which changes were made (somatic mutations).

---

## 4. Variant Annotation — Gene Mapping + OncoKB

**What it does:** Translates DNA-level changes into biological meaning.

**How it works (Position → Gene):**
- Each variant has a chromosome position (e.g., chr17:7675088)
- A gene coordinate database maps positions to genes
  (e.g., chr17:7668402-7687550 = TP53)
- If the variant falls in a protein-coding exon, we determine:
  - Which codon is affected (3-letter DNA → 1 amino acid)
  - What the reference amino acid is vs. the mutant amino acid
  - Example: chr17:7675088 C>T → TP53 codon 248 → Arginine→Tryptophan → p.R248W

**How it works (Clinical significance — OncoKB):**
- OncoKB (Memorial Sloan Kettering) maintains a database of 700+ cancer genes
- Each variant is annotated with:
  - Evidence Level 1: FDA-approved therapy for this mutation
  - Evidence Level 2: Standard-of-care (NCCN guidelines)
  - Evidence Level 3: Clinical trial evidence
  - Evidence Level 4: Preclinical evidence
- Example: BRCA2 frameshift → Level 1 → Olaparib (PARP inhibitor)

**Output:** Annotated variants with gene name, protein change, clinical
significance, and matching therapies.

**Analogy:** Finding a typo in a blueprint is step 3. Step 4 is figuring
out what the typo actually changes — does it break a wall (loss of function)
or add an extra door (gain of function)?

---

## 5. HLA Typing — OptiType / Population Frequency

**What it does:** Determines the patient's HLA (Human Leukocyte Antigen) type —
the "lock" that determines which peptides their immune system can present.

**How it works:**
- HLA genes are the most polymorphic in the human genome (~25,000 known alleles)
- Each person has 6 HLA class I alleles: 2x HLA-A, 2x HLA-B, 2x HLA-C
- These alleles determine which peptide fragments (8-11 amino acids) can be
  displayed on cell surfaces for T-cell recognition

**OptiType (from germline sequencing):**
- Extracts reads mapping to the HLA region on chromosome 6
- Aligns them against a database of all known HLA allele sequences
- Uses integer linear programming to find the best 6-allele combination
  that explains all observed reads
- Accuracy: >97% for 4-digit resolution (e.g., HLA-A*02:01)

**Current fallback (population frequency):**
- When sequencing data isn't available, assigns alleles based on
  population frequency (e.g., HLA-A*02:01 is present in ~29% of people)
- Less accurate but enables the pipeline to run on pre-called VCFs

**Why it matters for immunotherapy:**
- Two patients with the SAME tumor mutation may have DIFFERENT immune
  responses because their HLA types present different peptides
- A neoantigen vaccine must be designed for each patient's specific HLA type
- Example: KRAS G12D produces a peptide that binds HLA-A*11:01 strongly
  but barely binds HLA-A*02:01

**Output:** 6 HLA alleles (e.g., HLA-A*02:01, HLA-A*03:01, HLA-B*07:02,
HLA-B*08:01, HLA-C*07:01, HLA-C*07:02)

---

## 6. Neoantigen Prediction — MHCflurry 2.0

**What it does:** Predicts which tumor mutations produce peptides that the
patient's immune system can recognize and attack.

**How it works:**

**Step 1: Peptide Generation**
- For each somatic missense mutation, extract the mutant protein sequence
- Generate all possible peptides of length 8-11 that contain the mutation
- Example: TP53 R248W → generate peptides like HMTEVVR**W**C, TEVVR**W**HCP, etc.
- Also generate the corresponding wild-type peptides for comparison

**Step 2: Binding Prediction (MHCflurry)**
- For each peptide × each HLA allele, predict binding affinity
- MHCflurry is a deep neural network trained on >800,000 experimentally
  validated peptide-MHC binding measurements (mass spectrometry data)
- Input: peptide amino acid sequence + HLA allele name
- Output: IC50 value in nanomolar (nM)
  - < 50 nM = strong binder (very likely to be presented)
  - < 500 nM = weak binder (may be presented)
  - > 500 nM = non-binder (unlikely to be presented)
- The neural network learns:
  - Anchor residue preferences (positions 2 and 9 are critical for binding)
  - Allele-specific binding motifs (each HLA allele prefers different amino acids)
  - Peptide length preferences (most alleles prefer 9-mers)

**Step 3: Immunogenicity Scoring**
- Agretopicity: ratio of mutant/wild-type binding affinity
  - >1 means the mutation creates a BETTER binder than wild-type
  - This is important because the immune system was tolerized to wild-type
  - A high agretopicity means the mutation creates a truly "foreign" peptide
- Foreignness: how different the peptide is from any human self-peptide

**Step 4: Vaccine Candidate Selection**
- Rank by: binding affinity × agretopicity × expression level × clonality
- Select top 10-20 candidates for vaccine inclusion
- Prefer clonal mutations (present in all cancer cells, not just subclones)

**Output:** Ranked list of neoantigen vaccine candidates with binding
affinity, HLA allele, and immunogenicity scores.

**Analogy:** The immune system is like a lock-and-key system. HLA molecules
are the locks (different for each patient). Neoantigens are the keys.
MHCflurry predicts which mutant peptide "keys" will fit which HLA "locks."

---

## 7. MAD Agent Board — Multi-Agent Decision Support

**What it does:** Three specialist AI agents independently analyze the patient
and reach a consensus treatment recommendation.

**How it works:**

**Genomics Agent:**
- Analyzes the variant profile: which genes are mutated, TMB, MSI, HRD status
- Consults OncoKB for actionable mutations (Level 1-4 evidence)
- Ranks treatments based on genomic biomarkers
- Example: BRCA2 mutation → recommends Olaparib (PARP inhibitor)

**Immune Agent:**
- Analyzes the immune landscape: T-cell exhaustion, macrophage polarization
- Evaluates neoantigen quality and HLA coverage
- Assesses whether the tumor is "hot" (immune-infiltrated) or "cold"
- Ranks treatments based on immunotherapy eligibility
- Example: TMB-high + strong neoantigens → recommends Pembrolizumab

**Clinical Agent:**
- Simulates treatment response for all 9 regimens
- Uses published clinical trial data (KEYNOTE, CheckMate, PROfound)
- Predicts RECIST response, progression-free survival, irAE risk
- Ranks treatments based on expected clinical outcomes
- Example: Simulates 50% tumor reduction with olaparib, PFS 7.4 months

**Board Moderator:**
- Collects opinions from all 3 agents
- Computes agreement matrix (which agents agree on which treatments)
- Determines consensus: unanimous, majority, or split
- Synthesizes a unified rationale with evidence chain
- Includes dissenting views when agents disagree
- Every recommendation is traceable to specific biomarkers, trials, and evidence

**FDA Compliance:**
- Designed as Non-Device Clinical Decision Support (CDS)
- The clinician independently reviews the evidence — the AI doesn't decide
- Every recommendation includes the "why" — not a black box
- Audit trail records every decision with timestamps and data hashes

**Output:** Treatment recommendation with consensus level, confidence score,
evidence chain (linked to specific trials), and dissenting views.

---

## 8. 3D Target Visualization — Peptide-MHC Complex

**What it does:** Shows the actual molecular structure of the drug target —
the neoantigen peptide sitting in the HLA binding groove.

**How it works:**

**Template Fetch (RCSB PDB):**
- For common HLA alleles (A*02:01, B*07:02, etc.), crystal structures
  already exist in the Protein Data Bank
- We fetch the template structure which shows:
  - MHC heavy chain (alpha chain) — forms the binding groove
  - Beta-2-microglobulin — structural support
  - A peptide already bound in the groove
- 13 common HLA alleles are mapped to known PDB structures

**AlphaFold2-Multimer Fallback:**
- For rare HLA alleles without crystal structures
- Predicts the 3D structure of the MHC + peptide complex from sequence
- Uses NVIDIA's AlphaFold2-Multimer NIM endpoint
- Takes ~5 minutes per prediction

**3Dmol.js Rendering:**
- Browser-based WebGL molecular viewer (no GPU needed)
- MHC heavy chain: semi-transparent surface showing the binding groove
- Beta-2-microglobulin: transparent ribbon (structural context)
- Peptide: ball-and-stick model with element coloring (prominent)
- Mutation site: highlighted with red sphere + label
- Interactive: drag to rotate, scroll to zoom

**What it shows the clinician:**
- The physical shape of the peptide in the MHC groove
- Where the mutation is relative to the T-cell-facing surface
- Whether the mutant residue is exposed (accessible to T-cell receptor)
  or buried (hidden inside the groove)
- This helps assess whether the neoantigen is truly immunogenic

**Output:** Interactive 3D molecular visualization in the browser.

---

## End-to-End Timing

| Step | Tool | Time | Cost |
|------|------|------|------|
| Sequencing | External lab | Days-weeks | ~$300-1000 |
| Alignment (tumor) | Parabricks fq2bam | ~25 min | $1.40 |
| Alignment (normal) | Parabricks fq2bam | ~25 min | $1.40 |
| Somatic calling | Parabricks mutectcaller | ~55 min | $3.10 |
| Annotation | OncoKB + gene coords | <1 sec | $0 |
| HLA typing | OptiType / population | ~5 min | $0 |
| Neoantigen prediction | MHCflurry 2.0 | ~2-5 min | $0 |
| MAD Board | 3 agents + moderator | <1 sec | $0 |
| 3D visualization | 3Dmol.js + RCSB PDB | ~3 sec | $0 |
| **Total compute** | | **~2 hours** | **~$6** |

---

## Key Evidence Sources

| Source | What It Provides | Access |
|--------|-----------------|--------|
| **OncoKB** (MSK) | Variant actionability, FDA evidence levels | Free academic API |
| **MHCflurry 2.0** | Peptide-MHC binding prediction | pip install (open source) |
| **ClinicalTrials.gov** | Active recruiting trials matching biomarkers | Free API |
| **RCSB PDB** | Crystal structures of HLA-peptide complexes | Free download |
| **GIAB/NIST** | Gold standard variant truth sets for validation | Free |
| **SU2C/PCF** | 429 real mCRPC patients with outcomes | cBioPortal |

---

## Validated Results

| Metric | Value |
|--------|-------|
| TMB correlation (SU2C 429 patients) | r = 0.987 |
| Biomarker concordance | 100% |
| MHCflurry binding accuracy | >90% (vs IEDB experimental data) |
| Germline VCF (NA12878) | 3.7M variants, matches GIAB truth set |
| Somatic VCF (SEQC2 HCC1395) | 639K calls from matched tumor-normal |
| Pipeline cost per patient | ~$6 on L40S GPU |
