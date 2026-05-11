# Cognisom — Pre-Filing Implementation Spec

**Purpose:** Three engineering upgrades that convert Cognisom from "patentable-but-narrow" to "patentable-and-strong" before filing a non-provisional application. Each upgrade is self-contained and independently shippable. Sequencing is recommended in §6.

**Audience:** Engineering implementer (you or a contractor) with Python skills, basic biology literacy, and willingness to read the existing codebase.

**Filing strategy this spec serves:** File a **provisional application** now to lock the priority date on Inventions A and B at their current narrow scope. Use the next 12 weeks (the 12-month provisional window gives more, but 12 weeks is the realistic execution sprint) to implement the three upgrades. File a **non-provisional with strengthened claims** at the end of week 14, well inside the 12-month deadline.

---

## 0. Cross-Cutting Prerequisites (1–2 days, do these first)

### 0.1 Fix the Tm calculation bug

**File:** `engine/py/molecular/nucleic_acids.py:90–97`

**Current bug:** The Wallace rule `Tm = 4(G+C) + 2(A+T)` is correct for oligos ≤14 bases but produces nonsense for longer sequences. Running the self-test on the 136-base KRAS fragment reports Tm = 386°C (water boils at 100°C).

**Fix:** Branch on sequence length:

```python
def _calculate_tm(self) -> float:
    n = self.length
    g_plus_c = self.sequence.upper().count('G') + self.sequence.upper().count('C')
    a_plus_t = n - g_plus_c
    if n <= 14:
        # Wallace rule — short oligos only
        return 4.0 * g_plus_c + 2.0 * a_plus_t
    # Marmur-Doty for long sequences (rough; nearest-neighbor for production)
    gc_fraction = g_plus_c / n
    return 64.9 + 41.0 * gc_fraction - 500.0 / n
```

**Why this matters for the patent:** A USPTO §112 examiner who runs the disclosed code and sees Tm=386°C will reject for non-enablement / lack of utility. Cheap fix, must ship before filing.

**Test:** Add `tests/test_tm_calculation.py` asserting 50 < Tm < 100 for typical mammalian gene fragments.

### 0.2 Repo hygiene — separate third-party code

**Problem:** `/Users/davidwalker/CascadeProjects/cognisom/PhysiCell/` is a top-level directory containing a full unmodified PhysiCell clone (BSD-3 licensed). This creates two risks: (a) the impression that cognisom is a PhysiCell derivative, weakening novelty arguments; (b) accidental inclusion of PhysiCell code in claim-supporting sections of the spec.

**Action:**
1. Move `PhysiCell/` to `vendor/PhysiCell/` and add a `vendor/README.md` documenting that this is third-party reference code, not used in cognisom's runtime.
2. Confirm via `grep -rn "from PhysiCell\|import PhysiCell"` that no cognisom Python code imports from it. (Spot-check earlier: zero imports.)
3. Add `vendor/` to `.gitignore` for the actual ZIP if license attribution allows, OR keep it in-repo with explicit `LICENSE-PHYSICELL.txt` attribution.

**Effort:** 1 hour.

### 0.3 Tag the pre-upgrade snapshot

Before any upgrade work begins:

```
git tag patent-snapshot-pre-upgrades-$(date +%Y-%m-%d)
git push --tags
```

The provisional application will reference this tag for priority-date enablement evidence.

---

## 1. Upgrade 1 — Reference-Genome + Per-Cell Delta Memory Architecture

### 1.1 What problem it solves

**Current:** `modules/molecular_module.py:130–138` allocates a fresh `Gene` object — including a full sequence string copy — for every gene of every cell.

```python
def add_cell(self, cell_id: int):
    self.cell_genes[cell_id] = {}
    for name, gene in self.genes.items():
        cell_gene = Gene(gene.name, gene.dna.sequence, gene.gene_type)  # full copy
        ...
```

**Memory cost:** 100,000 simulated cells × 10 genes × ~3,000 bases/gene × 1 byte/base = **3 GB of redundant sequence storage** before any tissue-level scaling. With realistic genomes (~3 × 10⁹ bp) and 10⁶-cell tissues the design does not scale.

**Patent implication:** This is exactly the data-structure improvement the patent strategy needs — a specific *technological improvement to computer functioning* that defeats §101 Alice-trap rejection. We must implement it.

### 1.2 Target architecture

Two new files plus refactors:

**New file: `engine/py/molecular/reference_genome.py`**

```python
class ReferenceGenome:
    """Canonical genome shared by all cells. Immutable after construction."""
    def __init__(self):
        self._sequences: Dict[str, bytes] = {}  # gene_name -> immutable bytes
        self._gene_metadata: Dict[str, GeneMetadata] = {}

    def add_gene(self, name: str, sequence: str, gene_type: str,
                 is_oncogene: bool = False,
                 is_tumor_suppressor: bool = False,
                 domain_annotations: Optional[List[Domain]] = None) -> None: ...

    def get_reference_base(self, gene_name: str, position: int) -> bytes: ...
    def get_reference_sequence(self, gene_name: str) -> bytes: ...  # for read-only callers
    def gene_names(self) -> Iterable[str]: ...
```

`bytes` is chosen over `str` because Python interns short bytes objects and bytes comparison is faster; we never need to modify reference sequences, so immutability is desirable.

**New file: `engine/py/molecular/sequence_view.py`**

```python
@dataclass(frozen=True)
class SubstitutionDelta:
    gene_name: str
    position: int      # 0-indexed base position
    new_base: bytes    # 1-byte
    mutation_id: str   # for tracking provenance

class CellGenomeView:
    """
    Per-cell view of the genome: reference + this cell's deltas.

    Never materializes a full per-cell sequence in memory. Bases are
    served on demand by base_at(); contiguous regions are produced
    lazily by iter_codons(). This is the key data structure for the
    memory-architecture patent claim.
    """
    def __init__(self, reference: ReferenceGenome, deltas: List[SubstitutionDelta] = None):
        self._ref = reference
        # Index deltas for O(1) base lookup
        self._delta_index: Dict[Tuple[str, int], bytes] = {
            (d.gene_name, d.position): d.new_base for d in (deltas or [])
        }
        self._deltas_log: List[SubstitutionDelta] = list(deltas or [])

    def base_at(self, gene_name: str, position: int) -> bytes:
        key = (gene_name, position)
        if key in self._delta_index:
            return self._delta_index[key]
        return self._ref.get_reference_base(gene_name, position)

    def iter_codons(self, gene_name: str, start: int = 0) -> Iterator[bytes]:
        """Yield successive 3-byte codons applying deltas on the fly."""
        seq_len = len(self._ref.get_reference_sequence(gene_name))
        for i in range(start, seq_len - 2, 3):
            b0 = self.base_at(gene_name, i)
            b1 = self.base_at(gene_name, i+1)
            b2 = self.base_at(gene_name, i+2)
            yield b0 + b1 + b2

    def materialize(self, gene_name: str) -> bytes:
        """ESCAPE HATCH for callers that absolutely need the full sequence
        (e.g., feeding to a protein language model). Allocates on demand,
        does not cache. Use sparingly."""
        ...

    def add_substitution(self, gene_name: str, position: int, new_base: bytes,
                         mutation_id: str) -> SubstitutionDelta: ...

    def fork(self) -> 'CellGenomeView':
        """Daughter-cell creation. Shares the reference, copies the delta list
        (which is small)."""
        return CellGenomeView(self._ref, list(self._deltas_log))
```

**Memory math after upgrade:** Reference = ~30 KB (10 genes × 3000 bases). Per cell = 8 bytes for the view object + ~50 bytes per delta. 100,000 cells × avg 3 deltas/cell × 50 bytes = **15 MB total** vs the current 3 GB. **200× reduction.**

### 1.3 Refactor surface

**File: `engine/py/molecular/nucleic_acids.py`**

The current `Gene` class owns a `DNA` instance with a mutable sequence string. After refactor:
- `Gene` becomes metadata-only: name, type, regulatory annotations, NOT sequence.
- `DNA` is removed (or kept as a legacy shim with `materialize()` returning the bytes).
- `Gene.transcribe(view: CellGenomeView, ...)` reads sequence through the view rather than from `self.dna.sequence`.
- `Gene.introduce_oncogenic_mutation(view, mutation_name)` adds deltas to the view, not the gene.

**File: `modules/molecular_module.py`**

```python
def __init__(self, config):
    ...
    self.reference = ReferenceGenome()
    self.cell_views: Dict[int, CellGenomeView] = {}

def add_cell(self, cell_id: int):
    self.cell_views[cell_id] = CellGenomeView(self.reference)

def on_cell_divided(self, data):
    parent_id, daughter_id = data['cell_id'], data['daughter_id']
    self.cell_views[daughter_id] = self.cell_views[parent_id].fork()
```

### 1.4 Tests

`tests/test_reference_genome_architecture.py`:

1. **Correctness regression:** Run `examples/molecular/cancer_transmission_demo.py` against the old code (saved as a frozen oracle) and verify the new architecture produces *bit-identical* event traces for a fixed random seed. This proves the upgrade is a pure performance/memory refactor with no behavioral change.
2. **Memory benchmark:** Create 10,000 cells, measure `psutil` RSS before and after. Assert RSS delta < 50 MB (vs ~300 MB pre-upgrade).
3. **Performance benchmark:** Time 100 simulation steps on a 1,000-cell tissue. Assert no regression > 10% vs pre-upgrade timing.
4. **Daughter inheritance:** Mutate gene X in parent cell, divide, verify daughter view returns the mutated base at the position.

### 1.5 Patent claim this upgrade enables

> A method for memory-efficient agent-based simulation of cellular populations carrying genomic sequence state, comprising: maintaining a canonical reference genome in a single shared memory region accessible to all cell objects; representing each cell's deviation from said reference as a sparse list of substitution records, each substitution record identifying a gene, a position, and a substituted base; serving base-level sequence queries from each cell object via a view object that consults the substitution list before falling back to the canonical reference; and producing daughter cell objects on simulated cell division by copying the parent's substitution list while sharing the reference, whereby per-cell memory usage scales with mutation count rather than genome size.

**This is the §101 anchor.** It is a specific technical improvement to computer functioning (memory efficiency in agent-based simulation) — not abstract math.

### 1.6 Effort & risks

**Effort:** 2–3 weeks. Bulk of the work is the regression test (oracle generation) and shaking out callers that assume direct sequence access.

**Risks:**
- Performance regression from per-base method-call overhead. Mitigation: profile; if hot, JIT-compile `base_at()` with Numba or move the hot path to a small Cython/C extension. The first iteration uses pure Python.
- Hidden callers that mutate `gene.dna.sequence` directly. Mitigation: grep-and-fix; mark the old `DNA` class deprecated; raise on direct attribute access.

---

## 2. Upgrade 2 — Closed-Loop Neoantigen → Tissue Coupling

### 2.1 What problem it solves

Today the loop is broken at two points:

1. **Cell mutation does not feed peptide presentation.** `cognisom/genomics/neoantigen_predictor.py:116–235` runs standalone given a mutation; nothing inside the tissue simulation calls it as part of the per-step update.
2. **Immune recognition is a threshold heuristic, not TCR-pMHC matching.** `gpu/spatial_ops.py:200–240` decides "T cell kills cancer cell" based on a scalar MHC-I threshold. There is no peptide identity, no TCR repertoire, no affinity matching.

**Patent implication:** This is the strongest claim space — concrete medical output (neoantigen profile per simulated tumor; predicted immunotherapy response) defeats §101 abstract-idea rejection most cleanly.

### 2.2 Target architecture

Five new files. Three of them are simplified for v1 to keep scope manageable; each has a documented upgrade path.

**New file: `engine/py/molecular/peptidome.py`**

```python
class PeptidomeGenerator:
    """Translate mutated mRNA -> protein -> peptide pool.

    v1 (this spec): sliding window of 8-11mers spanning the mutation,
    weighted by simplified proteasomal cleavage propensity at C-terminal
    anchor residues (Leu, Ile, Phe, Tyr, Trp, Val score 1.0; others 0.3).

    v2 (deferred): integrate NetChop or a learned cleavage model.
    """
    def generate(self, view: CellGenomeView, gene_name: str,
                 mutation_position: int,
                 lengths: Tuple[int, ...] = (8, 9, 10, 11)) -> List[Peptide]: ...
```

`Peptide` is a dataclass with fields `(sequence: bytes, gene: str, mutation_id: str, cleavage_weight: float)`.

**New file: `engine/py/immune/mhc_loading.py`**

```python
class MHCLoadingPipeline:
    """Score each peptide against the patient's HLA alleles and
    select the top-K for surface presentation.

    v1: PWM scorer (existing neoantigen_predictor).
    v2 (deferred): MHCflurry-2.0 binding affinity model.
    """
    def __init__(self, hla_alleles: List[str], predictor: NeoantigenPredictor,
                 top_k: int = 50, affinity_threshold_nM: float = 500.0):
        ...

    def present(self, peptides: List[Peptide]) -> List[PresentedPeptide]:
        """Returns peptides predicted to bind any of the patient's HLA-I alleles
        with affinity < threshold. Output is the cell's surface presentation."""
        ...
```

**New file: `engine/py/immune/tcr_repertoire.py`**

```python
class TCRRepertoire:
    """Model the patient's T-cell receptor repertoire.

    v1: a stochastic affinity model. Each T cell has a 16-dim feature
    vector sampled from N(0, I) at creation. Each presented peptide is
    embedded to a 16-dim feature vector via a fixed hash of its sequence.
    pMHC-TCR affinity = sigmoid(t_vec @ p_vec / sqrt(16)). This is the
    minimum viable abstraction.

    v2 (deferred): TCRdist3 or learned encoder (e.g., ESM-2 + projection
    head).
    """
    def __init__(self, n_tcrs: int, seed: int): ...
    def affinity(self, tcr_id: int, peptide: PresentedPeptide) -> float: ...
```

**New file: `engine/py/immune/tcell_kill.py`**

```python
class TCellKillModel:
    """Kill-probability per T-cell encounter.

    p_kill(dt) = 1 - exp( - max_kill_rate * max_affinity * mhc1_surface_density * dt )

    where max_affinity = max over all (TCR i scanning, peptide j on cell)
    of repertoire.affinity(i, j).
    """
    def kill_probability(self, t_cell, target_cell, presented_peptides,
                         repertoire: TCRRepertoire, dt: float) -> float: ...
```

**Refactor: `modules/cellular_module.py`** — extend each cell's state to include:
- `presented_peptides: List[PresentedPeptide]` (recomputed periodically, not every step — say every 10 steps for performance)
- `mhc1_expression` already exists; it is now the *aggregate surface density*, while `presented_peptides` carries identity.

**Refactor: `modules/immune_module.py`** — replace the `mhc1_expression > threshold` heuristic with:

```python
for t_cell in self.t_cells:
    for cell in neighbors(t_cell):
        if not cell.presented_peptides:
            continue
        p_kill = self.kill_model.kill_probability(
            t_cell, cell, cell.presented_peptides,
            self.tcr_repertoire, dt)
        if np.random.random() < p_kill:
            self.emit_event(EventTypes.CELL_KILLED_BY_TCELL, {...})
```

### 2.3 Tests

`tests/test_neoantigen_closed_loop.py`:

1. **No-mutation baseline:** Tumor with only synonymous mutations should show ~0 neoantigens presented and near-zero T-cell kill rate.
2. **Strong-neoantigen scenario:** Tumor with KRAS G12D in a patient HLA-A\*02:01 background should present ≥1 high-affinity peptide and show measurably elevated T-cell kill rate.
3. **Immune-evasion scenario:** Same KRAS G12D tumor but with MHC-I expression knocked to 0.05 should show suppressed T-cell kill rate (and elevated NK-cell kill rate if NK logic is included in v2).
4. **End-to-end propagation:** Mutate a cell in a fresh simulation; advance N steps; verify the event log contains `MUTATION_OCCURRED → PEPTIDE_GENERATED → PEPTIDE_PRESENTED → CELL_KILLED_BY_TCELL` in that order, with timestamps showing causal sequence.

### 2.4 Patent claim this upgrade enables

> A computer-implemented method for predicting cellular immune response to a tumor in silico, comprising: advancing in simulated time a population of cell objects each carrying a sequence-derived genomic state; on a mutation event, generating from said cell object's mutated sequence a peptide pool by sliding-window decomposition of the translated protein around the mutation site; scoring each peptide against one or more patient HLA-I alleles using a binding-affinity scorer to produce a presented-peptide set; advancing T-cell objects through the tissue and computing a kill probability per cell-cell encounter as a function of pMHC-TCR affinity between the T-cell's receptor and the presented-peptide set; and emitting a per-tumor predicted response trajectory.

Dependent claims: NK-cell missing-self detection; PD-L1/PD-1 checkpoint modulation; exosome-mediated MHC suppression (linking back to Invention B).

This is the **strongest §101 / §103 posture** of the three upgrades. The output (predicted neoantigen profile and predicted T-cell kill rate) is a concrete medical/biological prediction — not abstract math. And no published competitor closes this loop in a tissue simulator.

### 2.5 Effort & risks

**Effort:** 6–8 weeks for working v1; 3 months for benchmark-quality.

**Risks:**
- TCR repertoire is a real research-grade problem. The v1 stochastic abstraction is honest but simple; reviewers will notice it. Mitigation: document it as a v1 model and gate the patent claim on "a TCR-pMHC affinity scorer" generically without committing to one model.
- Performance: re-running the peptidome predictor every step would be too slow. Solution: recompute only on mutation events and decay presented peptides over a half-life timescale.
- Validation requires patient cohort data with known immunotherapy response. Defer this to post-filing; the patent claims need only enablement, not validation against clinical truth.

---

## 3. Upgrade 3 — Zero-Shot Mutation-Effect Biophysics

### 3.1 What problem it solves

`engine/py/molecular/nucleic_acids.py:402–416`:

```python
oncogenic_mutations = {
    'KRAS': {'G12D': (35, 'G', 'A'), 'G12V': (35, 'G', 'T'), 'G13D': (38, 'G', 'A')},
    'BRAF': {'V600E': (1799, 'T', 'A')},
    'TP53': {'R175H': (524, 'G', 'A'), 'R248W': (742, 'C', 'T')},
}
```

This is a hardcoded 5-entry dictionary. Mutations outside this table cannot be introduced via the convenience API, and there is no biophysics-based way to predict the consequence of an arbitrary novel mutation.

**Patent implication:** The competitive differentiator over PhysiCell + COSMIC-lookup approaches is *zero-shot* prediction — given any (gene, position, base) triple, compute a numerical phenotype modifier from the sequence alone, without a curated table. This is the §103 obviousness defense: it is not obvious to combine PhysiCell + COSMIC because Cognisom uses neither.

### 3.2 Target architecture — staged

**Stage A (rule-based codon classifier) — 2 weeks, ship first.**

**New file: `engine/py/molecular/mutation_effect.py`**

```python
class MutationEffectClassifier:
    """Rule-based per-codon impact scoring.

    For each substitution: classify as synonymous / missense / nonsense /
    frameshift / splice-site. For missense, score via BLOSUM62 substitution
    matrix. Output a numerical impact_score in [0, 1] where 0 = no effect
    and 1 = complete loss of function.
    """

    def classify(self, view: CellGenomeView, gene_name: str, position: int,
                 new_base: bytes) -> MutationEffect: ...
```

`MutationEffect` dataclass: `(category, impact_score, aa_change, blosum_score, in_critical_domain)`.

**Stage B (domain-aware) — 2 weeks.**

Annotate each gene in `ReferenceGenome` with a list of functional domains (from UniProt JSON or a local annotations file). Mutations inside critical domains (kinase, DNA-binding, transactivation) get an extra impact multiplier of 2–5×.

**Stage C (ML-based protein stability) — 4–6 weeks, the differentiator.**

**New file: `engine/py/molecular/protein_stability.py`**

```python
class ProteinStabilityPredictor:
    """ESM-2-based zero-shot stability differential.

    Loads a pretrained ESM-2-150M (or 650M on GPU) protein language model.
    Given a wild-type protein sequence and a single-residue substitution,
    computes the pseudo-log-likelihood ratio (mutant / wildtype) at the
    mutation position. Negative values indicate destabilization.

    This is the §103 differentiator: PhysiCell + COSMIC-lookup cannot
    do this. Cognisom can, because it has per-cell sequences.
    """

    def __init__(self, model_name: str = "facebook/esm2_t30_150M_UR50D",
                 device: str = "auto"):
        ...

    def delta_ll(self, wildtype_protein: bytes, position: int,
                 mutant_aa: bytes) -> float: ...

    def stability_modifier(self, wildtype_protein: bytes, position: int,
                           mutant_aa: bytes) -> float:
        """Returns a multiplier in [0.05, 1.0] applied to protein half-life
        or function. -3 nats of log-likelihood -> ~0.05x function."""
        return float(np.clip(np.exp(self.delta_ll(...) / 3.0), 0.05, 1.0))
```

ESM-2-150M is 150M parameters and runs on CPU in ~2 seconds per mutation. ESM-2-650M (4.4 GB GPU memory) runs in ~0.1 second per mutation. For the simulation loop, we cache results per (gene, position, mutant_aa) tuple — the LRU cache will hit ~100% after warmup because mutations recur.

**Stage D (gain-of-function detection) — research-grade, deferred.**

Distinguishing loss-of-function from gain-of-function (KRAS G12D activates, TP53 R175H deactivates) requires either a curated activation classifier or a much more sophisticated model. Skip for v1; note as future work.

### 3.3 Refactor surface

**File: `engine/py/molecular/nucleic_acids.py`**

Remove the hardcoded oncogenic-mutation dictionary. Replace `Gene.introduce_oncogenic_mutation()` with:

```python
def introduce_mutation(self, view: CellGenomeView,
                       position: int, new_base: bytes,
                       effect_classifier: MutationEffectClassifier,
                       stability_predictor: Optional[ProteinStabilityPredictor] = None,
                      ) -> MutationOutcome:
    delta = view.add_substitution(self.name, position, new_base,
                                  mutation_id=str(uuid.uuid4())[:8])
    effect = effect_classifier.classify(view, self.name, position, new_base)
    stability_mod = (stability_predictor.stability_modifier(...)
                     if stability_predictor and effect.category == "missense"
                     else 1.0)
    return MutationOutcome(delta=delta, effect=effect,
                           stability_modifier=stability_mod)
```

Convenience method `introduce_named_mutation('KRAS_G12D')` is kept as a thin wrapper but uses the classifier underneath — i.e., the hardcoded table becomes a *test fixture* of known mutations, not the source of phenotypic truth.

**File: `engine/py/molecular/nucleic_acids.py` — Gene.transcribe()**

The transcription rate now consumes a per-cell modifier derived from mutation effects:

```python
def transcribe(self, view: CellGenomeView, dt: float) -> Optional[RNA]:
    rate_modifier = view.get_aggregate_rate_modifier_for(self.name)  # product of per-mutation impact scores
    effective_rate = self.transcription_rate * rate_modifier
    if np.random.random() < effective_rate:
        return RNA.from_view(view, self.name)
    return None
```

### 3.4 Tests

`tests/test_mutation_effect_zero_shot.py`:

1. **Synonymous mutations score ~0:** Sample 20 random synonymous substitutions across KRAS, TP53, BRAF; assert impact_score < 0.1 for all.
2. **Nonsense mutations score ~1:** Sample stop-codon-introducing substitutions; assert impact_score > 0.8.
3. **Known oncogenic mutations score high:** KRAS G12D, BRAF V600E, TP53 R175H should all score > 0.5.
4. **Novel-mutation regression:** Pick 10 *random* non-synonymous mutations not in any training data; verify the predictor returns a finite numerical score with no crashes.
5. **ESM-2 stability sanity:** TP53 R175H (well-known destabilizing) should yield stability_modifier < 0.5; a synonymous TP53 substitution should yield stability_modifier > 0.95.

### 3.5 Patent claim this upgrade enables

> A computer-implemented method for in-silico phenotype prediction of arbitrary nucleotide substitutions in an agent-based cellular simulation, comprising: identifying a single-nucleotide substitution by gene, position, and substituted base; classifying said substitution by codon-level consequence selected from synonymous, missense, nonsense, frameshift, and splice-site; for substitutions classified as missense, computing a stability differential by querying a pretrained protein language model with the wild-type protein sequence and the substituted residue; deriving a numerical phenotype modifier from said codon-level classification and said stability differential; and applying said phenotype modifier as a multiplicative factor on at least one of a simulated transcription rate, translation rate, protein half-life, or protein function score in the agent-based simulation.

Dependent claims: domain-aware impact multipliers; caching of stability predictions by mutation tuple; integration with the closed-loop neoantigen pipeline of Upgrade 2.

**Differentiation argument for the attorney:** PhysiCell uses lookup-table parameter scalings. COSMIC and OncoKB are *databases*, not predictive simulators. ESM-2 and similar models predict stability but are not integrated with cellular simulators. Cognisom is the integration — a specific computational improvement enabling phenotype prediction for mutations that have never been catalogued.

### 3.6 Effort & risks

**Effort:**
- Stage A + B: 3–5 weeks.
- Stage C (ML): 4–6 weeks additional, plus GPU setup (ESM-2-150M runs on CPU but slowly).

**Risks:**
- ESM-2 dependency adds 4 GB of model weights to the install footprint. Mitigation: make Stage C optional via config; v1 patent claims can be drafted to read on Stage A+B alone, with Stage C as a dependent claim.
- BLOSUM62 is textbook and unpatentable on its own. Mitigation: claim the *integration into agent-based simulation*, not the score itself.
- ESM-2 license: Meta's ESM-2 is permissively licensed (MIT-style); confirm with attorney before commercial deployment but no expected issue.

---

## 4. Cross-Upgrade Test Suite

Add `tests/test_patent_evidence.py` — a single test module that produces *reproducible enablement evidence* for the provisional and non-provisional filings. Each test:

1. Has a fixed random seed.
2. Writes its output to `tests/evidence/<test_name>.log`.
3. Asserts on specific event-trace patterns.

Tests to include:
- `test_sequence_grounded_simulation_runs` — minimal mutation → daughter inheritance scenario (Invention A).
- `test_exosome_horizontal_transfer` — the existing cancer transmission demo, frozen as a test (Invention B).
- `test_hybrid_solver_hysteresis_no_oscillation` — toggle a species count around 100 with hysteresis; verify no partition flips below 120 / above 80 (Invention C).
- `test_memory_architecture_scales` — 10k cells under 50 MB RSS (Upgrade 1).
- `test_closed_loop_kill_rate` — mutation → presented peptide → measured kill-rate increase (Upgrade 2).
- `test_zero_shot_synonymous_vs_nonsense` — score distribution sanity (Upgrade 3).

The output of this suite is what the attorney attaches as §112 enablement evidence.

---

## 5. Dependency Graph

```
Prerequisites (0.1, 0.2, 0.3)
    |
    +-> Upgrade 1 (memory) ----+
    |                          |
    +-> Upgrade 3 Stage A/B ---+--> Upgrade 2 (closed loop)
    |                          |
    +-> Upgrade 3 Stage C -----+
```

**Hard dependencies:**
- Upgrade 2 depends on Upgrade 1 (the closed-loop test needs the view object) and Upgrade 3 Stage A (kill-rate modulation needs mutation impact scores).
- Upgrade 3 Stage C is independent and can be developed in parallel.

**Soft dependencies:**
- All upgrades benefit from Upgrade 1's `CellGenomeView` for clean APIs, but each can be retrofitted.

---

## 6. Recommended Sprint Plan (12 Weeks)

| Week | Sprint | Deliverable |
|---|---|---|
| 0 | Prep | File **provisional** application on Inventions A + B at current scope. Tag `patent-snapshot-provisional-filed`. Ship prerequisites (Tm fix, repo hygiene). |
| 1–2 | Sprint 1 | Upgrade 3 Stage A + B (rule-based mutation effect). Patent-eligible standalone. |
| 3–5 | Sprint 2 | Upgrade 1 (memory architecture). Patent-eligible standalone. |
| 6–9 | Sprint 3 | Upgrade 2 (closed loop, simplified TCR/MHC). Strongest patent claim. |
| 10–12 | Sprint 4 | Upgrade 3 Stage C (ESM-2 integration). Differentiator. |
| 13 | Hardening | Run full patent-evidence test suite. Generate enablement logs. |
| 14 | Filing | File **non-provisional** consolidating Inventions A + B + C plus Upgrades 1, 2, 3 as a single application family. Claim priority back to provisional. |

**Critical timing notes:**
- The provisional application is the priority anchor. File it in week 0 *before* starting any of the upgrades. Cost: ~$1,500 in USPTO fees plus attorney time.
- All upgrade work between weeks 1 and 13 is protected by the provisional's priority date — competitors who file before you on similar ideas will lose if your provisional clearly disclosed the inventive concept.
- The provisional has a 12-month deadline for non-provisional conversion. 14 weeks is well inside that window; this schedule has 9 months of slack.

---

## 7. Estimated Resources

| Role | Weeks | Notes |
|---|---|---|
| Senior Python/scientific-computing engineer | 14 | Owns the implementation |
| ML engineer (familiar with HuggingFace, ESM-2) | 4 | For Upgrade 3 Stage C |
| Computational biologist (advisory) | 1–2 | Sanity-check the peptidome, TCR, MHC models in Upgrade 2 |
| Patent attorney | ongoing | Provisional drafting (week 0), claim refinement (weeks 13–14) |
| GPU time (Upgrade 3 Stage C only) | 1 week of L40S or A10 | ~$200 on AWS spot |

**Total cash outlay (excluding inventor's own time):** approximately $40k–$70k for contractor engineering + $15k–$25k patent attorney fees through non-provisional filing.

---

## 8. Honest Risks & Open Questions

1. **PhysiCell or BioDynaMo could announce sequence-aware features in 2026.** Risk: prior-art collision. Mitigation: file the provisional in week 0 to lock priority. Re-run prior-art search at week 13 before non-provisional filing.

2. **Computational biology peer review may push back on the simplified v1 models** (8–11mer sliding window peptidome; stochastic TCR repertoire). Risk: validation papers reject the simulator. Mitigation: these are v1 only; document upgrade paths to NetChop / TCRdist3 / MHCflurry; patent claims are drafted generically to read on any reasonable implementation.

3. **ESM-2 license review.** Risk: an enterprise legal review later objects to MIT-style licensed model weights in a commercial product. Mitigation: confirm with attorney during week 0; alternatively use ProtBERT or train a small in-house model.

4. **Inventorship.** If a contractor implements any of the three upgrades and the resulting code is referenced in patent claim language, that contractor may need to be named as a co-inventor. Have all contractors sign an assignment of invention rights *before* they start work — standard IP boilerplate, ~$200 attorney fee per contractor.

5. **The provisional priority date does not cover any of the upgrades.** Risk: if a competitor files between week 0 and week 14 on (say) the memory-architecture idea, they could win that specific claim. Mitigation: Upgrade 1 in particular is conceptually simple — consider filing a *second provisional* in week 3 covering Upgrade 1 specifically, then consolidating all into the week-14 non-provisional.

---

## 9. Definition of Done

The implementation is patent-ready when:

- All three upgrades pass their tests in `tests/test_patent_evidence.py`.
- The full demo suite runs end-to-end on a clean checkout in under 10 minutes.
- The repository at the filing tag contains no `NotImplementedError` references in patent-relevant code paths.
- The Tm bug is fixed and the self-tests print physically reasonable values.
- A 1-page reproducibility README in `docs/PATENT_EVIDENCE.md` tells an examiner how to reproduce each claim with `python -m tests.test_patent_evidence`.

When these are true, the attorney drafts the non-provisional and you file.
