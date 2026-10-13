"""
NK receptor balance: KIR-HLA licensing and the missing-self model
================================================================

The NK arm of the simulation used to be one bit::

    return cancer_cell.mhc1_expression < 0.4

measured as a hard step: identical killing at MHC-I 0.00, 0.20 and 0.39,
and no killing at all at 0.40, 0.41, 0.80 or 1.00. There was no receptor,
no activating/inhibitory balance, no KIR, and nothing patient-specific.
This module replaces that bit with the standard two-signal model.

The biology
-----------
NK cytotoxicity is decided by the balance of activating and inhibitory
receptor signals at the synapse (Lanier, Annu Rev Immunol 2005). Two
pieces matter here.

*Missing self.* Inhibitory killer-cell immunoglobulin-like receptors
(KIR) read HLA class I on the target. A target that loses surface HLA-I
loses that inhibition and becomes NK-sensitive -- the missing-self
hypothesis (Karre, Ljunggren, Piontek & Kiessling, Nature 1986,
319:675-678). MHC-I loss is one of the commonest routes of tumour escape
from CD8 T cells, so the same lesion that blinds the T-cell arm is what
exposes the target to NK cells. That complementarity is why this arm is
worth modelling at all.

*Licensing / education.* An NK clone becomes functionally competent only
if an inhibitory receptor it expresses engages a cognate self-MHC-I
ligand during development. Clones whose inhibitory receptors have no
self ligand stay hyporesponsive -- "unlicensed" -- but, having no
engaged inhibitory receptor, they are also not MHC-inhibited (Kim et al.,
Nature 2005, 436:709-713, licensing in mice; Anfossi et al., Immunity
2006, 25:331-342, the human KIR-HLA equivalent; reviewed in Elliott &
Yokoyama, Trends Immunol 2011). The consequence is the one this module
exists to express: how much of a *particular patient's* NK compartment
is licensed depends on which KIR ligand groups that patient's own HLA
class I carries.

KIR ligand groups
-----------------
Inhibitory KIR do not read alleles, they read two epitope families on
the alpha-1 helix:

* HLA-C residue 80. Asn80 forms the C1 epitope, read by KIR2DL2 and
  KIR2DL3; Lys80 forms the C2 epitope, read by KIR2DL1 (Colonna et al.,
  Science 1993; Winter & Long, J Immunol 1997, 158:4026-4028 -- a single
  residue discriminates the two groups; structural basis in Boyington
  2000 and Fan 2001).
* HLA-B/-A residues 77-83, the Bw4 epitope, read by KIR3DL1 (Cella
  et al., J Exp Med 1994; Gumperz et al., J Exp Med 1995). Bw4 splits
  by residue 80 into Bw4-80I and Bw4-80T, and KIR3DL1 binds 80I more
  strongly (Carr, Pando & Parham, J Immunol 2005). Arg83 is the Bw4
  discriminator against Bw6's Gly83; the Glu76-Arg83 salt bridge is what
  holds the Bw4 alpha-1 helix in the KIR3DL1-permissive conformation.

``HLA_LIGAND_TABLE`` below carries, for 43 class-I alleles, the mature
protein's residues 76-83 **read out of the IPD-IMGT/HLA sequence record
itself** together with that record's accession and its own
``matching.kir_ligand`` assignment (IPD-IMGT/HLA release 3.65.0,
retrieved 2026-10-11 via https://www.ebi.ac.uk/cgi-bin/ipd/api/allele).
The group is then *derived* from the motif by ``ligand_group_from_motif``
and ``test_nk_receptor_balance.py`` asserts that the derivation
reproduces IPD's own assignment for every B and C entry. Nothing here is
a hand-copied serotype table.

Alleles absent from the table raise ``UnknownAlleleError``. They are not
defaulted, because a guessed ligand group would silently decide a
patient's NK licensing.

What is patient-specific and what is not
----------------------------------------
HALF of this model is a real measurement and half is a population prior,
and the two are reported separately rather than blended:

* The HLA side is measured. ``hla_typer.MEASURED_HLA_TYPINGS["SEQC2"]``
  is A*29:02/A*29:02, B*45:01/B*45:01, C*06:02/C*06:02, typed with
  OptiType 1.3.5 from SRR7890874 (HCC1395BL matched normal, WXS) on
  1198 reads. Run through this module that patient is homozygous C2,
  Bw6/Bw6, and A-locus Bw4-negative: KIR2DL1 is licensed, KIR2DL2,
  KIR2DL3 and KIR3DL1 are all unlicensed. That is a non-trivial,
  checkable prediction about a real sample.
* The KIR side is a prior, not a genotype. Nothing in this platform can
  type KIR -- OptiType types HLA -- so gene content defaults to carrier
  frequencies from Bontadini et al., J Transl Med 2006, 4:44 (217
  unrelated Italian Caucasians, PCR-SSP): KIR2DL1 95%, KIR2DL2 53%,
  KIR2DL3 88%, KIR3DL1 96%; the framework genes KIR3DL3, KIR3DP1,
  KIR2DL4 and KIR3DL2 were present in all 217. ``KIRGeneContent`` carries
  ``source`` and ``is_patient_specific=False`` so this never passes for a
  typing, and a real genotype can be substituted.

Declared parameters that are NOT measurements
---------------------------------------------
``KIR_AVIDITY``, ``UNLICENSED_RESPONSIVENESS`` and the activating-signal
shape are parameterisations. The literature cited above establishes their
*ordering* and *direction*, which is what the test module asserts; it does
not give per-encounter numbers, so the magnitudes are exposed constants
rather than claims. ``DEFAULT_MISSING_SELF_THRESHOLD`` is the legacy 0.4
literal, lifted here from the middle of ``immune_module._recognize`` so
it can at least be seen and overridden; it has no source and the receptor
model supersedes it whenever patient HLA is on record.

Single-receptor approximation
-----------------------------
``licensing_profile`` represents the NK compartment as a partition into
single-inhibitory-KIR subsets whose sizes are proportional to gene
carriage, and each simulated NK agent draws one such identity at
creation and keeps it (``draw_kir_subset``, ``nk_clone_decision``).
Real repertoires co-express receptors and vary in per-allele expression
level (Yawata et al., Blood 2008; Beziat et al., Blood 2013), and
co-expression makes individual clones *more* inhibited than this
approximation says. The claim here is the licensing architecture and the
missing-self gradient, not the subset arithmetic.

Not modelled
------------
NKG2A/HLA-E education (the -21M/-21T HLA-B leader dimorphism of Horowitz
et al., Sci Immunol 2016, 1:eaag1672) is out of scope; the leader residue
is recorded in the table for whoever picks that up. Activating KIR
(KIR2DS1/KIR3DS1), ADCC via CD16, and the cytokine-licensing route are
all absent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Mapping, Optional, Sequence, Tuple


#: IPD-IMGT/HLA release the ligand table was read from, and the date.
IPD_IMGT_HLA_RELEASE = "3.65.0"
IPD_IMGT_HLA_RETRIEVED = "2026-10-11"


class LigandGroup(str, Enum):
    """A KIR ligand epitope family on HLA class I."""

    C1 = "C1"
    C2 = "C2"
    BW4_80I = "Bw4-80I"
    BW4_80T = "Bw4-80T"
    BW6 = "Bw6"
    #: Carries no KIR ligand epitope. Used for HLA-A allotypes without
    #: the Bw4 motif: the Bw6 epitope is a B-locus serological
    #: specificity and IPD-IMGT/HLA assigns no kir_ligand to HLA-A.
    NO_KIR_LIGAND = "none"


class KIR(str, Enum):
    """The inhibitory KIR modelled here."""

    KIR2DL1 = "KIR2DL1"
    KIR2DL2 = "KIR2DL2"
    KIR2DL3 = "KIR2DL3"
    KIR3DL1 = "KIR3DL1"


class UnknownAlleleError(ValueError):
    """An allele whose KIR ligand group is not on record.

    Raised rather than defaulted: a guessed ligand group decides which
    of a patient's NK subsets are licensed, so guessing here would
    manufacture a patient-specific result out of nothing.
    """


@dataclass(frozen=True)
class AlleleRecord:
    """One class-I allele as read from its IPD-IMGT/HLA sequence record.

    residues_76_83
        Residues 76-83 of the MATURE protein (the record's protein
        sequence with its annotated signal peptide removed). Positions
        80 and 83 are what the ligand-group rule reads.
    ipd_accession
        The IPD-IMGT/HLA accession the motif was read from.
    ipd_kir_ligand
        That record's own ``matching.kir_ligand`` string, verbatim
        ("C1", "C2", "Bw4 - 80I", "Bw4 - 80T", "Bw6"), or None where
        IPD assigns no group (every HLA-A record). Kept so the derived
        group can be checked against the database's own call.
    b_leader
        The record's ``matching.b_leader`` (-21 M/T dimorphism). Recorded
        only; the NKG2A/HLA-E axis it governs is not modelled.
    """

    allele: str
    residues_76_83: str
    ipd_accession: str
    ipd_kir_ligand: Optional[str] = None
    b_leader: Optional[str] = None


#: 43 class-I alleles: every allele this platform can produce -- the
#: measured SEQC2 typing, ``hla_typer.COMMON_HLA_ALLELES``,
#: ``hla_typer.SYNTHETIC_HLA_PROFILES``, ``cellular_module``'s default
#: panel -- plus the Bw4-positive HLA-A and HLA-B allotypes needed to
#: exercise the rule. Generated from the IPD-IMGT/HLA API responses, not
#: transcribed by hand.
HLA_LIGAND_TABLE: Dict[str, AlleleRecord] = {
    'A*01:01': AlleleRecord('A*01:01', 'ANLGTLRG', 'HLA00001', None, None),
    'A*02:01': AlleleRecord('A*02:01', 'VDLGTLRG', 'HLA00005', None, None),
    'A*03:01': AlleleRecord('A*03:01', 'VDLGTLRG', 'HLA00037', None, None),
    'A*11:01': AlleleRecord('A*11:01', 'VDLGTLRG', 'HLA00043', None, None),
    'A*23:01': AlleleRecord('A*23:01', 'ENLRIALR', 'HLA00048', None, None),
    'A*24:02': AlleleRecord('A*24:02', 'ENLRIALR', 'HLA00050', None, None),
    'A*25:01': AlleleRecord('A*25:01', 'ESLRIALR', 'HLA00071', None, None),
    'A*26:01': AlleleRecord('A*26:01', 'ANLGTLRG', 'HLA00073', None, None),
    'A*29:02': AlleleRecord('A*29:02', 'ANLGTLRG', 'HLA00086', None, None),
    'A*31:01': AlleleRecord('A*31:01', 'VDLGTLRG', 'HLA00097', None, None),
    'A*32:01': AlleleRecord('A*32:01', 'ESLRIALR', 'HLA00101', None, None),
    'A*68:01': AlleleRecord('A*68:01', 'VDLGTLRG', 'HLA00115', None, None),
    'B*07:02': AlleleRecord('B*07:02', 'ESLRNLRG', 'HLA00132', 'Bw6', 'M'),
    'B*08:01': AlleleRecord('B*08:01', 'ESLRNLRG', 'HLA00146', 'Bw6', 'M'),
    'B*13:02': AlleleRecord('B*13:02', 'ENLRTALR', 'HLA00153', 'Bw4 - 80T', 'T'),
    'B*15:01': AlleleRecord('B*15:01', 'ESLRNLRG', 'HLA00162', 'Bw6', 'T'),
    'B*18:01': AlleleRecord('B*18:01', 'ESLRNLRG', 'HLA00213', 'Bw6', 'T'),
    'B*27:05': AlleleRecord('B*27:05', 'EDLRTLLR', 'HLA00225', 'Bw4 - 80T', 'T'),
    'B*35:01': AlleleRecord('B*35:01', 'ESLRNLRG', 'HLA00237', 'Bw6', 'T'),
    'B*40:01': AlleleRecord('B*40:01', 'ESLRNLRG', 'HLA00291', 'Bw6', 'T'),
    'B*44:02': AlleleRecord('B*44:02', 'ENLRTALR', 'HLA00318', 'Bw4 - 80T', 'T'),
    'B*44:03': AlleleRecord('B*44:03', 'ENLRTALR', 'HLA00319', 'Bw4 - 80T', 'T'),
    'B*45:01': AlleleRecord('B*45:01', 'ESLRNLRG', 'HLA00329', 'Bw6', 'T'),
    'B*49:01': AlleleRecord('B*49:01', 'ENLRIALR', 'HLA00340', 'Bw4 - 80I', 'T'),
    'B*50:01': AlleleRecord('B*50:01', 'ESLRNLRG', 'HLA00341', 'Bw6', 'T'),
    'B*51:01': AlleleRecord('B*51:01', 'ENLRIALR', 'HLA00344', 'Bw4 - 80I', 'T'),
    'B*53:01': AlleleRecord('B*53:01', 'ENLRIALR', 'HLA00364', 'Bw4 - 80I', 'T'),
    'B*57:01': AlleleRecord('B*57:01', 'ENLRIALR', 'HLA00381', 'Bw4 - 80I', 'T'),
    'B*58:01': AlleleRecord('B*58:01', 'ENLRIALR', 'HLA00386', 'Bw4 - 80I', 'T'),
    'C*01:02': AlleleRecord('C*01:02', 'VSLRNLRG', 'HLA00401', 'C1', None),
    'C*02:02': AlleleRecord('C*02:02', 'VNLRKLRG', 'HLA00404', 'C2', None),
    'C*03:03': AlleleRecord('C*03:03', 'VSLRNLRG', 'HLA00411', 'C1', None),
    'C*03:04': AlleleRecord('C*03:04', 'VSLRNLRG', 'HLA00413', 'C1', None),
    'C*04:01': AlleleRecord('C*04:01', 'VNLRKLRG', 'HLA00420', 'C2', None),
    'C*05:01': AlleleRecord('C*05:01', 'VNLRKLRG', 'HLA00427', 'C2', None),
    'C*06:02': AlleleRecord('C*06:02', 'VNLRKLRG', 'HLA00430', 'C2', None),
    'C*07:01': AlleleRecord('C*07:01', 'VSLRNLRG', 'HLA00433', 'C1', None),
    'C*07:02': AlleleRecord('C*07:02', 'VSLRNLRG', 'HLA00434', 'C1', None),
    'C*08:02': AlleleRecord('C*08:02', 'VSLRNLRG', 'HLA00446', 'C1', None),
    'C*12:03': AlleleRecord('C*12:03', 'VSLRNLRG', 'HLA00455', 'C1', None),
    'C*14:02': AlleleRecord('C*14:02', 'VSLRNLRG', 'HLA00462', 'C1', None),
    'C*15:02': AlleleRecord('C*15:02', 'VNLRKLRG', 'HLA00467', 'C2', None),
    'C*16:02': AlleleRecord('C*16:02', 'VNLRKLRG', 'HLA00476', 'C2', None),
}


# ---------------------------------------------------------------------------
# Declared parameters. Ordering is cited; magnitudes are not measurements.
# ---------------------------------------------------------------------------

#: Relative inhibitory avidity of each KIR for each ligand group.
#:
#: The ORDERING is what the literature supports and what the tests
#: assert: KIR2DL1-C2 is the strongest of the KIR2D/HLA-C pairings;
#: KIR2DL2 is a stronger C1 receptor than KIR2DL3 (Moesta et al.,
#: J Immunol 2008, 180:3969-3979); KIR2DL2/2DL3 cross-react weakly with
#: some C2 allotypes while KIR2DL1 does not cross-react onto C1
#: (Moesta 2008; Hilton & Parham, Front Immunol 2012, 3:336, which also
#: reports the cross-reactive C2 allotypes); KIR3DL1 binds Bw4-80I more
#: strongly than Bw4-80T (Carr, Pando & Parham, J Immunol 2005,
#: 175:5222-5229). The MAGNITUDES are a declared parameterisation --
#: per-encounter avidities are not available -- and are exposed so a
#: reviewer can refit them. Bw6 appears in no row: it is not a KIR
#: ligand.
KIR_AVIDITY: Dict[KIR, Dict[LigandGroup, float]] = {
    KIR.KIR2DL1: {LigandGroup.C2: 1.00},
    KIR.KIR2DL2: {LigandGroup.C1: 0.75, LigandGroup.C2: 0.25},
    KIR.KIR2DL3: {LigandGroup.C1: 0.50},
    KIR.KIR3DL1: {LigandGroup.BW4_80I: 1.00, LigandGroup.BW4_80T: 0.50},
}

#: The ligand group(s) whose presence in the HOST licenses each receptor.
#: Distinct from KIR_AVIDITY: cross-reactivity lets KIR2DL2 be inhibited
#: by C2 on a target, but only its cognate C1 educates it.
KIR_COGNATE_LIGANDS: Dict[KIR, Tuple[LigandGroup, ...]] = {
    KIR.KIR2DL1: (LigandGroup.C2,),
    KIR.KIR2DL2: (LigandGroup.C1,),
    KIR.KIR2DL3: (LigandGroup.C1,),
    KIR.KIR3DL1: (LigandGroup.BW4_80I, LigandGroup.BW4_80T),
}

#: Effector competence retained by an UNLICENSED NK subset, relative to
#: a licensed one. Licensing establishes the direction and the ordering
#: (Kim 2005; Anfossi 2006; unlicensed cells are hyporesponsive but not
#: anergic, and recover function on transfer into an MHC-sufficient
#: host). No defensible single quantitative figure was found, so this is
#: a declared parameter: the tests assert licensed > unlicensed > 0, not
#: this value.
UNLICENSED_RESPONSIVENESS: float = 0.3

#: Activating-signal shape. ``baseline`` is the constitutive activating
#: ligand load on a transformed cell (DNAM-1/CD155, NCR ligands);
#: ``max_gain`` is the further induction attributable to genomic stress,
#: and ``half_max_mutations`` the mutation count at half of it. The
#: mechanism is real -- NKG2D ligands MICA/MICB/ULBP are induced by the
#: ATM/ATR DNA-damage response (Gasser, Orsulic, Brown & Raulet, Nature
#: 2005, 436:1186-1190) -- but no calibration maps a mutation count to a
#: surface ligand density, so the SHAPE (monotone, saturating) is the
#: claim and all three numbers are exposed parameters. Per-cell mutation
#: count is used because it is a quantity the simulation actually
#: computes; a DNA-damage state variable on CellState would be the right
#: driver and does not exist yet.
DEFAULT_ACTIVATING_BASELINE: float = 0.5
DEFAULT_ACTIVATING_MAX_GAIN: float = 0.5
DEFAULT_ACTIVATING_HALF_MAX_MUTATIONS: float = 5.0

# ── KNOWN LIMITATION: the two scales are not formally coupled ────────────
#
# Inhibition and activation are summed against each other, but nothing
# constrains their relative magnitudes, and at these values a high mutation
# load can overcome a COGNATE inhibitory engagement at full self-MHC. A
# licensed clone should never engage a target it is licensed against -- that
# is what licensing is for.
#
# MEASURED, on a fully licensed compartment (HLA-A*02:01, B*57:01, C*01:02,
# C*06:02 -- every KIR in the model has a cognate ligand) meeting a HEALTHY
# autologous cell at mhc1 = 1.0:
#
#     n_mutations      0      1      2     50
#     kill probability 0.0000 0.0177 0.0303 0.1225
#
# Two obvious repairs were measured and both FAIL, which is why this is a
# documented limitation rather than a constant tweak:
#
#   * Doubling KIR_AVIDITY so the weakest cognate pair meets the activating
#     ceiling fixes self-tolerance (p = 0.0000 at every load) but compresses
#     the graded response: kill probability across mhc1 in
#     (1.0, 0.75, 0.5, 0.25, 0.05) collapses from five distinct values to
#     four, because the low-MHC end saturates. The graded response IS the
#     model.
#   * Halving the activating ceiling instead is worse -- four distinct values
#     and two MHC levels pinned at zero.
#
# The real fix is structural: the activating and inhibitory terms need a
# shared, dimensioned scale (or a competitive rather than additive
# combination), not a rescaled constant. Until then the behaviour above is
# pinned by test_a_licensed_clone_tolerates_self_imperfectly in
# tests/test_nk_receptor_balance.py, so it is visible and cannot silently
# worsen. Do not "fix" this by moving one constant without re-running that
# module.

#: Deadband on the integrated signal before degranulation. 0.0 is the
#: pure balance model -- kill when activation exceeds inhibition -- and
#: is the default so that no unsourced number enters the decision.
DEFAULT_ACTIVATION_THRESHOLD: float = 0.0

#: The legacy missing-self step, lifted out of ``_recognize``. Used only
#: when no patient HLA is on record. It has no source; the receptor
#: model supersedes it.
DEFAULT_MISSING_SELF_THRESHOLD: float = 0.4

#: KIR gene carriage, Bontadini et al., J Transl Med 2006, 4:44, Table 1
#: (217 unrelated Italian Caucasians, gene-level PCR-SSP). A POPULATION
#: PRIOR, not anyone's genotype.
POPULATION_PRIOR_KIR_PRESENCE: Dict[KIR, float] = {
    KIR.KIR2DL1: 0.95,
    KIR.KIR2DL2: 0.53,
    KIR.KIR2DL3: 0.88,
    KIR.KIR3DL1: 0.96,
}

POPULATION_PRIOR_KIR_SOURCE = (
    "population_prior:Bontadini2006 (J Transl Med 4:44, n=217 Italian "
    "Caucasians, PCR-SSP gene-level carrier frequencies)"
)


# ---------------------------------------------------------------------------
# Ligand-group assignment
# ---------------------------------------------------------------------------

def normalize_allele(allele: str) -> str:
    """Reduce an allele name to locus + two fields, no ``HLA-`` prefix.

    ``"HLA-C*06:02:01:01"``, ``"C*06:02"`` and ``"hla-c*06:02"`` all
    become ``"C*06:02"``. Two-field resolution is the level at which KIR
    ligand groups are assigned; the residue-80/83 motif is invariant
    within a two-field group for every allele in the table.
    """
    if allele is None:
        raise UnknownAlleleError("allele is None")
    name = str(allele).strip().upper().replace("HLA-", "", 1)
    if "*" not in name:
        raise UnknownAlleleError(f"not an HLA allele name: {allele!r}")
    locus, _, fields = name.partition("*")
    parts = [p for p in fields.split(":") if p]
    if not parts:
        raise UnknownAlleleError(f"allele has no fields: {allele!r}")
    return f"{locus}*{':'.join(parts[:2])}"


def ligand_group_from_motif(locus: str, residues_76_83: str) -> LigandGroup:
    """Derive the KIR ligand group from residues 76-83 of the mature protein.

    HLA-C is decided by residue 80: Asn -> C1, Lys -> C2 (Winter & Long,
    J Immunol 1997). HLA-B and HLA-A are decided by residue 83: Arg ->
    Bw4, split by residue 80 into Bw4-80I / Bw4-80T (Cella 1994;
    Gumperz 1995; Carr 2005); Gly83 -> Bw6 at the B locus, and no KIR
    ligand at the A locus, where Bw6 is not an assigned specificity.

    Raises UnknownAlleleError on any residue combination the rule does
    not cover, rather than picking a group.
    """
    motif = (residues_76_83 or "").strip().upper()
    if len(motif) != 8:
        raise UnknownAlleleError(
            f"residues 76-83 must be 8 residues, got {motif!r}"
        )
    res80 = motif[4]
    res83 = motif[7]
    loc = (locus or "").strip().upper().replace("HLA-", "", 1).rstrip("*")

    if loc == "C":
        if res80 == "N":
            return LigandGroup.C1
        if res80 == "K":
            return LigandGroup.C2
        raise UnknownAlleleError(
            f"HLA-C residue 80 is {res80!r}; neither Asn (C1) nor Lys (C2)"
        )

    if loc in ("A", "B"):
        if res83 == "R":
            if res80 == "I":
                return LigandGroup.BW4_80I
            if res80 == "T":
                return LigandGroup.BW4_80T
            raise UnknownAlleleError(
                f"Bw4 (Arg83) with residue 80 {res80!r}; neither Ile nor Thr"
            )
        if res83 == "G":
            return LigandGroup.BW6 if loc == "B" else LigandGroup.NO_KIR_LIGAND
        raise UnknownAlleleError(
            f"HLA-{loc} residue 83 is {res83!r}; neither Arg (Bw4) nor Gly (Bw6)"
        )

    raise UnknownAlleleError(f"locus {loc!r} carries no KIR ligand group")


def ligand_group(allele: str) -> LigandGroup:
    """KIR ligand group of one class-I allele, derived from its motif.

    Raises UnknownAlleleError for an allele absent from
    ``HLA_LIGAND_TABLE``.
    """
    key = normalize_allele(allele)
    record = HLA_LIGAND_TABLE.get(key)
    if record is None:
        raise UnknownAlleleError(
            f"{key} is not in HLA_LIGAND_TABLE (IPD-IMGT/HLA "
            f"{IPD_IMGT_HLA_RELEASE}); its KIR ligand group is not on "
            f"record and is not guessed. Add the allele from its "
            f"IPD-IMGT/HLA sequence record."
        )
    return ligand_group_from_motif(key.partition("*")[0], record.residues_76_83)


@dataclass(frozen=True)
class HLALigandProfile:
    """The KIR ligand groups one patient's class-I genotype carries."""

    alleles: Tuple[str, ...]
    per_allele: Tuple[Tuple[str, LigandGroup], ...]
    groups: Tuple[LigandGroup, ...]
    is_patient_specific: bool
    source: str

    @property
    def has_c1(self) -> bool:
        return LigandGroup.C1 in self.groups

    @property
    def has_c2(self) -> bool:
        return LigandGroup.C2 in self.groups

    @property
    def has_bw4(self) -> bool:
        return (LigandGroup.BW4_80I in self.groups
                or LigandGroup.BW4_80T in self.groups)


def hla_ligand_profile(
    alleles: Sequence[str],
    *,
    is_patient_specific: bool = False,
    source: str = "unspecified",
) -> HLALigandProfile:
    """Resolve a class-I genotype into its KIR ligand groups.

    Every allele must be on record; one unknown allele raises rather
    than being dropped, because a dropped allele silently removes a
    ligand group and so silently unlicenses an NK subset.
    """
    if not alleles:
        raise UnknownAlleleError("no HLA alleles given")
    per_allele = []
    for a in alleles:
        key = normalize_allele(a)
        per_allele.append((key, ligand_group(key)))
    groups = sorted(
        {g for _, g in per_allele
         if g not in (LigandGroup.BW6, LigandGroup.NO_KIR_LIGAND)},
        key=lambda g: g.value,
    )
    return HLALigandProfile(
        alleles=tuple(k for k, _ in per_allele),
        per_allele=tuple(per_allele),
        groups=tuple(groups),
        is_patient_specific=bool(is_patient_specific),
        source=source,
    )


# ---------------------------------------------------------------------------
# KIR gene content and licensing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KIRGeneContent:
    """Inhibitory KIR gene content, as presence weights in [0, 1].

    A real genotype uses 1.0 / 0.0 and ``is_patient_specific=True``.
    The default is carrier frequencies from a population, which is a
    prior over gene content and is flagged as such.
    """

    presence: Tuple[Tuple[KIR, float], ...]
    source: str
    is_patient_specific: bool

    def weight(self, receptor: KIR) -> float:
        for k, v in self.presence:
            if k is receptor:
                return v
        return 0.0


def population_prior_kir() -> KIRGeneContent:
    """Carrier-frequency prior over inhibitory KIR gene content."""
    return KIRGeneContent(
        presence=tuple(
            (k, POPULATION_PRIOR_KIR_PRESENCE[k])
            for k in (KIR.KIR2DL1, KIR.KIR2DL2, KIR.KIR2DL3, KIR.KIR3DL1)
        ),
        source=POPULATION_PRIOR_KIR_SOURCE,
        is_patient_specific=False,
    )


def genotyped_kir(present: Sequence[KIR]) -> KIRGeneContent:
    """A real KIR genotype: the listed genes present, the rest absent."""
    have = {KIR(k) for k in present}
    return KIRGeneContent(
        presence=tuple(
            (k, 1.0 if k in have else 0.0)
            for k in (KIR.KIR2DL1, KIR.KIR2DL2, KIR.KIR2DL3, KIR.KIR3DL1)
        ),
        source="genotyped",
        is_patient_specific=True,
    )


@dataclass(frozen=True)
class KIRSubset:
    """One single-inhibitory-KIR subset of the NK compartment."""

    receptor: KIR
    weight: float               # fraction of the compartment
    licensed: bool
    competence: float           # effector competence multiplier
    inhibitory_avidity: float   # strongest avidity against self groups


@dataclass(frozen=True)
class NKLicensingProfile:
    """A patient's NK compartment: who is licensed, and by what."""

    hla: HLALigandProfile
    kir: KIRGeneContent
    subsets: Tuple[KIRSubset, ...]
    licensed_fraction: float
    unlicensed_responsiveness: float

    @property
    def is_patient_specific(self) -> bool:
        """True only when the HLA side is a real typing.

        The KIR side is reported separately via ``kir.is_patient_specific``
        and is a population prior by default.
        """
        return self.hla.is_patient_specific

    def subset_for(self, receptor) -> Optional[KIRSubset]:
        """The compartment subset expressing ``receptor``, or None."""
        want = KIR(receptor)
        for s in self.subsets:
            if s.receptor is want:
                return s
        return None

    def as_dict(self) -> Dict[str, object]:
        """Flat, JSON-safe summary for events and dashboards."""
        return {
            "alleles": list(self.hla.alleles),
            "ligand_groups": [g.value for g in self.hla.groups],
            "has_c1": self.hla.has_c1,
            "has_c2": self.hla.has_c2,
            "has_bw4": self.hla.has_bw4,
            "hla_source": self.hla.source,
            "hla_is_patient_specific": self.hla.is_patient_specific,
            "kir_source": self.kir.source,
            "kir_is_patient_specific": self.kir.is_patient_specific,
            "licensed_fraction": self.licensed_fraction,
            "subsets": [
                {
                    "receptor": s.receptor.value,
                    "weight": s.weight,
                    "licensed": s.licensed,
                    "competence": s.competence,
                    "inhibitory_avidity": s.inhibitory_avidity,
                }
                for s in self.subsets
            ],
            "ipd_release": IPD_IMGT_HLA_RELEASE,
        }


def licensing_profile(
    hla: HLALigandProfile,
    kir: Optional[KIRGeneContent] = None,
    *,
    unlicensed_responsiveness: float = UNLICENSED_RESPONSIVENESS,
) -> NKLicensingProfile:
    """Resolve licensing and self-inhibition for one patient.

    An NK subset expressing inhibitory receptor ``k`` is licensed iff the
    patient carries a cognate ligand group for ``k`` (Kim 2005; Anfossi
    2006). Its inhibitory avidity against an autologous target is the
    strongest ``KIR_AVIDITY`` entry among the groups that patient's own
    HLA carries -- an autologous tumour displays the patient's own
    allotypes, so the same genotype fixes both education and engagement.
    """
    content = kir if kir is not None else population_prior_kir()
    comp = _clamp01(unlicensed_responsiveness)
    raw = [(k, content.weight(k)) for k, _ in content.presence]
    total = sum(w for _, w in raw)
    if total <= 0.0:
        raise ValueError(
            "KIR gene content has zero total weight: no inhibitory "
            "receptor is present, so no NK compartment can be resolved"
        )
    subsets = []
    licensed_weight = 0.0
    for receptor, w in raw:
        if w <= 0.0:
            continue
        share = w / total
        licensed = any(g in hla.groups for g in KIR_COGNATE_LIGANDS[receptor])
        avidity = max(
            (KIR_AVIDITY[receptor].get(g, 0.0) for g in hla.groups),
            default=0.0,
        )
        if licensed:
            licensed_weight += share
        subsets.append(KIRSubset(
            receptor=receptor,
            weight=share,
            licensed=licensed,
            competence=1.0 if licensed else comp,
            inhibitory_avidity=float(avidity),
        ))
    return NKLicensingProfile(
        hla=hla,
        kir=content,
        subsets=tuple(subsets),
        licensed_fraction=float(licensed_weight),
        unlicensed_responsiveness=comp,
    )


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------

def activating_signal(
    n_mutations: int = 0,
    *,
    baseline: float = DEFAULT_ACTIVATING_BASELINE,
    max_gain: float = DEFAULT_ACTIVATING_MAX_GAIN,
    half_max_mutations: float = DEFAULT_ACTIVATING_HALF_MAX_MUTATIONS,
) -> float:
    """Activating receptor load on a target, in [0, 1].

    ``baseline + max_gain * m / (m + k)``: monotone in mutation count and
    saturating. See DEFAULT_ACTIVATING_* for why the numbers are
    parameters and not measurements.
    """
    m = max(0.0, float(n_mutations))
    k = max(1e-9, float(half_max_mutations))
    return _clamp01(float(baseline) + float(max_gain) * (m / (m + k)))


@dataclass(frozen=True)
class NKDecision:
    """One NK-vs-target encounter, decomposed."""

    recognized: bool
    kill_probability: float
    firing_fraction: float      # weight of the compartment above threshold
    activating: float
    mhc1_expression: float
    per_subset: Tuple[Tuple[str, float, float], ...]  # receptor, signal, p


def subset_signal(
    subset: KIRSubset,
    activating: float,
    mhc1_expression: float,
) -> float:
    """Integrated signal at one NK clone's synapse.

        activating * competence - inhibitory_avidity * mhc1

    Competence is the licensing term (Kim 2005; Anfossi 2006) and
    inhibitory_avidity * mhc1 is the missing-self term (Karre 1986). An
    unlicensed clone has a reduced first term; a clone with no cognate
    ligand on the target has NO second term, which is why unlicensed
    cells are weak but MHC-independent.
    """
    return (float(activating) * subset.competence
            - subset.inhibitory_avidity * _clamp01(mhc1_expression))


def draw_kir_subset(
    profile: NKLicensingProfile,
    rng,
) -> KIRSubset:
    """Draw one clone's inhibitory-KIR identity from the compartment.

    A real NK clone expresses a fixed receptor set for life, so the
    right place for the repertoire distribution is cell creation, not
    the per-encounter roll. ``rng`` is any object with ``.random()``
    (numpy Generator, random.Random); the draw is a cumulative walk so
    the result is identical across both.
    """
    if not profile.subsets:
        raise ValueError("licensing profile has no subsets to draw from")
    u = float(rng.random())
    acc = 0.0
    for s in profile.subsets:
        acc += s.weight
        if u < acc:
            return s
    return profile.subsets[-1]


def nk_clone_decision(
    subset: KIRSubset,
    *,
    mhc1_expression: float,
    n_mutations: int = 0,
    max_kill_probability: float,
    activation_threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
    activating_baseline: float = DEFAULT_ACTIVATING_BASELINE,
    activating_max_gain: float = DEFAULT_ACTIVATING_MAX_GAIN,
    activating_half_max_mutations: float = DEFAULT_ACTIVATING_HALF_MAX_MUTATIONS,
) -> NKDecision:
    """One NK CLONE against one target -- the per-cell decision.

    This is what the simulation uses: each NK agent carries a fixed
    inhibitory KIR identity drawn at creation, so both engagement and
    killing are deterministic functions of that clone's receptors and
    the target's state. ``nk_decision`` is the same arithmetic averaged
    over the whole compartment, which is the population-level readout.

    Engaging on the integrated signal (rather than killing only after
    an MHC-independent engagement) is what makes missing self visible:
    a licensed clone facing a target with intact MHC-I does not engage
    at all, which is self-tolerance, not merely weak killing.
    """
    a = activating_signal(
        n_mutations,
        baseline=activating_baseline,
        max_gain=activating_max_gain,
        half_max_mutations=activating_half_max_mutations,
    )
    mhc = _clamp01(mhc1_expression)
    thr = _clamp01(activation_threshold)
    span = max(1e-9, 1.0 - thr)
    signal = subset_signal(subset, a, mhc)
    if signal > thr:
        p = _clamp01(max_kill_probability) * _clamp01((signal - thr) / span)
        fires = True
    else:
        p = 0.0
        fires = False
    return NKDecision(
        recognized=fires,
        kill_probability=float(p),
        firing_fraction=1.0 if fires else 0.0,
        activating=a,
        mhc1_expression=mhc,
        per_subset=((subset.receptor.value, float(signal), float(p)),),
    )


def nk_decision(
    profile: NKLicensingProfile,
    *,
    mhc1_expression: float,
    n_mutations: int = 0,
    max_kill_probability: float,
    activation_threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
    activating_baseline: float = DEFAULT_ACTIVATING_BASELINE,
    activating_max_gain: float = DEFAULT_ACTIVATING_MAX_GAIN,
    activating_half_max_mutations: float = DEFAULT_ACTIVATING_HALF_MAX_MUTATIONS,
) -> NKDecision:
    """Integrate activating and inhibitory signals over the compartment.

    Per subset::

        signal = activating * competence - inhibitory_avidity * mhc1
        fires  = signal > activation_threshold
        p      = max_kill * (signal - threshold) / (1 - threshold)

    and the returned probability is the compartment-weighted mean, i.e.
    the expected per-encounter kill probability for an NK cell drawn at
    random from this patient's repertoire. Two properties follow and are
    asserted by the tests: licensed subsets lose inhibition as the target
    loses MHC-I (missing self), and unlicensed subsets are
    MHC-independent -- weaker, but not inhibited.

    This is the POPULATION readout. The simulation itself runs
    ``nk_clone_decision`` per NK agent, because a clone's receptors are
    fixed; averaging over the compartment inside a per-encounter roll
    would make every NK cell the average NK cell and would hide the
    fact that an intact-MHC target is invisible to the licensed subset.
    """
    mhc = _clamp01(mhc1_expression)
    a = activating_signal(
        n_mutations,
        baseline=activating_baseline,
        max_gain=activating_max_gain,
        half_max_mutations=activating_half_max_mutations,
    )
    thr = _clamp01(activation_threshold)
    span = max(1e-9, 1.0 - thr)
    max_kill = _clamp01(max_kill_probability)

    p_total = 0.0
    firing = 0.0
    rows = []
    for s in profile.subsets:
        signal = subset_signal(s, a, mhc)
        if signal > thr:
            p = max_kill * _clamp01((signal - thr) / span)
            firing += s.weight
            p_total += s.weight * p
        else:
            p = 0.0
        rows.append((s.receptor.value, float(signal), float(p)))

    return NKDecision(
        recognized=firing > 0.0,
        kill_probability=float(p_total),
        firing_fraction=float(firing),
        activating=a,
        mhc1_expression=mhc,
        per_subset=tuple(rows),
    )


def missing_self_recognized(
    mhc1_expression: float,
    threshold: float = DEFAULT_MISSING_SELF_THRESHOLD,
) -> bool:
    """The legacy step rule, kept for runs with no patient HLA on record.

    This is the ``mhc1_expression < 0.4`` literal that used to sit inside
    ``ImmuneModule._recognize``, lifted so it is visible and
    configurable. It is a step function with no source and no receptor
    behind it; ``nk_decision`` supersedes it whenever an HLA typing is
    available.
    """
    return float(mhc1_expression) < float(threshold)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _clamp01(x: float) -> float:
    if x is None:
        return 0.0
    v = float(x)
    if math.isnan(v):
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v
