# Papers I still need, and where each one is

Generated 2026-10-21 by querying PubMed, PMC and Unpaywall for every
remaining PMID. Save each as `docs/papers/<PMID>.pdf` -- that directory
is gitignored and read before any network route, so a paper dropped
there is used immediately and never fetched again.

**Two of these are free right now** and need no institutional access:
15209560 and 16840338 are both in PubMed Central. Open the PMC link,
print to PDF, done.

The rest are paywalled to an automated fetch. Unpaywall reports several
as bronze, hybrid or green, which means a copy is legally free to read
somewhere and the publisher still blocks a script -- so they need a
browser and, for most, an institutional login.

| # | PMID | Paper | Where | What it holds up |
|---|------|-------|-------|------------------|
| 5 | `10381624` | Purification and catalytic properties of human caspase family members<br>*Cell Death Differ* 1999 — paywalled | [PubMed](https://pubmed.ncbi.nlm.nih.gov/10381624/) · [doi](https://doi.org/10.1038/sj.cdd.4400497) | Values behind: kcat_per_s |
| 5 | `12451422` | Plasmalemmal concentration and affinity of mouse vascular endothelial <br>*Eur Biophys J* 2002 — paywalled | [PubMed](https://pubmed.ncbi.nlm.nih.gov/12451422/) · [doi](https://doi.org/10.1007/s00249-002-0248-9) | Values behind: kd_nm |
| 5 | `15771445` | Binding thermodynamics as a tool to investigate the mechanisms of drug<br>*J Med Chem* 2005 — repository copy exists | [PubMed](https://pubmed.ncbi.nlm.nih.gov/15771445/) · [doi](https://doi.org/10.1021/jm040842z) | Values behind: kd_nm |
| 5 | `16490769` | Screening for caspase-3 inhibitors: a new class of potent small-molecu<br>*J Biomol Screen* 2006 — paywalled | [PubMed](https://pubmed.ncbi.nlm.nih.gov/16490769/) · [doi](https://doi.org/10.1177/1087057105285048) | Values behind: km_um |
| 5 | `20178337` | In vitro modulation of cytochrome P450 reductase supported indoleamine<br>*Biochemistry* 2010 — paywalled | [PubMed](https://pubmed.ncbi.nlm.nih.gov/20178337/) · [doi](https://doi.org/10.1021/bi100022c) | Values behind: kcat_per_s, km_um |

## Obtained — 2026-10-21 (migration)

`25679760` — Liu et al., *Cell* 2015. The stored `0.234 µm/min` is
correct and is **the wrong regime for a tumour**. The same cells reach
**1.7** (A1) and **5.3 µm/min** (A2) once confined to 3 µm without focal
adhesions — **23×, with no genetic change**. The paper's own closing
argument is that the confinement a growing tumour imposes on itself may
be enough to trigger it.

It also supplies the model the simulation was missing: a persistent
random walk, `MSD(t) = 2 S² P (t − P(1 − e^(−t/P)))`, `D = S²P/2` — the
conversion from a speed the library *has* to the cell diffusion
coefficient the mechanics needs.

**10 papers read, 8 changed something.**


## Obtained — 2026-10-21 (all four adhesion papers)

`9829984` · `11404363` · `9780171` · `9422729` — the whole adhesion
cascade, read and applied (`seed_adhesion_bonds.py`).

**Every stored number was already correct.** All four had been sourced
against the right PMID by an agent that had not read the papers, and
each value survived. What did not survive was the context:

| | |
|---|---|
| **Species** | L-selectin's conditions said "human CD62L". The secondary table said "Mouse". The primary says **rat CD62L binding mouse GlyCAM-1** — a cross-species pair the authors flag themselves. E-selectin/ESL-1 is mouse throughout; both carry human UniProt accessions. |
| **Temperature** | None recorded, while E-selectin's koff runs 1.1 → 4.6 s⁻¹ from 5 to 37 °C. The library silently mixed 25 °C and 37 °C constants across one cascade. |
| **Which bond** | Every value sat under a bare `kd_nm`. P-selectin binds PSGL-1 200× more tightly than E-selectin binds ESL-1, and E-selectin binds PSGL-1 too. Now under ligand-named aliases. |
| **Which state** | LFA-1's 500 nM is the **Mg²⁺-induced high-affinity state**, measured deliberately without Ca²⁺. Resting LFA-1 on an unstimulated T cell is ~100 µM — **200× weaker**. |
| **Observed vs calculated** | The two paper families are mirror images. Selectins: association too fast to fit, so kon = koff/Kd. ICAM-1/LFA-1: kass fitted, and "Kd was calculated from the ratio kdiss/kass". The library had both graded the same way. |

**9 papers read, 7 changed something.**


## Obtained — 2026-10-21

Three arrived today and all three are read, applied and committed
(`847a4e6`). Two of the three changed a stored value.

| PMID | What it settled |
|---|---|
| `2272970` | **6 cell radii.** Monocyte was 1.85x too large (6.3x in volume). The neutrophil already cited this PMID at 4.2 um — the Coulter column — written before the paper was read; it is 4.167 by the flow column the authors lead with. 19 entities deliberately left placeholder: the paper measured *unactivated* leukocytes. |
| `15209560` | **caspase-8 kcat 0.88 → 0.65.** The 0.88 was a median across three different molecular species, three of them in 0.7 M sodium citrate. Km 55.2 uM added, stored as `km_ietd_um` because it is a tetrapeptide constant. |
| `16840338` | **Confirmed** `replication_cycle_hours` = 6.0, and gained its 95% CI plus the two neighbouring quantities a reader could take by mistake. |

Running total: **5 papers read, 4 changed something.**


## Why these and not others

Every paper here is cited by a value this library already stores and
grades MEASURED. The grade asserts somebody read the measurement; for
these thirteen, nobody has. That is the gap, and it is why the TIER a
retrieval reached is recorded alongside the citation.

Three papers have come in this way so far and the record is not
reassuring: PMID 6775712 confirmed four stored values and revealed they
are overestimates the Appendix quantifies; PMID 15130947 corrected two
of three microvillus numbers; PMID 35676539 confirmed a kcat exactly.
Two of three changed something.

