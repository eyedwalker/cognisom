"""Turn a patient's genome into simulation inputs.

The gap this closes
-------------------
Until now every simulated cancer cell carried ``KRAS_G12D``, presented
on a generic three-allele HLA panel, whoever the patient was.
``EngineRunner`` had no way to receive a patient. The genomics pipeline
meanwhile ran end to end on a real genome -- HCC1395 goes from 1,495
variants to two somatic coding drivers, ARID1A p.N218T and NOTCH1
p.L2065Q, typed against that patient's measured HLA -- and stopped short
of the simulation. Personalisation is the platform's central claim, so
this was the most consequential instance of the built-but-never-
connected pattern found so far.

What crosses over
-----------------
* Every somatic coding MISSENSE mutation, not only the drivers.
  Passenger mutations make neoantigens too, and the T cells do not know
  which mutations drove the cancer.
* The patient's HLA class-I alleles, for peptide presentation on the
  cancer cells and for the NK cells' licensing.
* Whether that typing was measured from the patient's reads. The NK
  licensing model refuses an unmeasured typing on purpose, and this
  module does not launder one into looking measured.

What does not, and why
----------------------
Each mutation is checked against the reference protein before it is
handed over, and every one that fails is REPORTED with its reason rather
than dropped. The engine's own presentation step skips a mismatch
silently, which is how a mutation can vanish without anyone knowing:

* nonsense, frameshift and synonymous changes make no canonical
  missense neoantigen (the engine has no path for them yet);
* a protein change whose wild-type residue does not match the reference
  at that position. The synthetic prostate demo's ``AR p.T877A`` is the
  standing example: T877A is the historical numbering, and in the MANE
  Select protein the threonine is at 878, so the reference has F at 877.
  It is reported, not renumbered -- other code keys on "T877A", and a
  silent renumbering would be its own kind of fabrication.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from cognisom.genomics.hla_typer import READ_DERIVED_HLA_METHODS  # noqa: E402

#: Three-letter amino-acid codes, as VEP's HGVSp writes them.
_AA3 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Ter": "*", "Sec": "U",
}
_AA = r"(?:[A-Z][a-z]{2}|[A-Z*])"
_CHANGE = re.compile(rf"^(?:p\.)?\(?({_AA})(\d+)({_AA})\)?$")


def _one_letter(token: str) -> str:
    return _AA3.get(token, token) if len(token) == 3 else token


def parse_missense(protein_change: Optional[str]) -> Optional[Tuple[str, int, str]]:
    """``"p.N218T"`` or ``"p.Asn218Thr"`` -> ``("N", 218, "T")``.

    Returns None for anything that is not a single-residue missense
    substitution: nonsense (``*``), frameshift, in-frame indel,
    synonymous (same residue both sides) or unparseable.
    """
    if not protein_change:
        return None
    match = _CHANGE.match(protein_change.strip())
    if not match:
        return None
    wild, position, mutant = (_one_letter(match.group(1)), int(match.group(2)),
                              _one_letter(match.group(3)))
    if "*" in (wild, mutant) or wild == mutant or position < 1:
        return None
    return wild, position, mutant


@dataclass
class EngineMutation:
    """One missense mutation the engine can present."""
    gene: str
    aa_change: str          # one-letter, e.g. "N218T"
    label: str              # e.g. "ARID1A_N218T", what a cell carries
    oncogenic: bool


@dataclass
class PatientEngineInputs:
    """What a patient profile contributes to a simulation."""
    patient_id: str
    hla_alleles: List[str]
    hla_measured: bool
    hla_method: str
    mutations: List[EngineMutation] = field(default_factory=list)
    #: (gene, protein change, reason) for every coding variant left out.
    skipped: List[Tuple[str, str, str]] = field(default_factory=list)

    def module_configs(self) -> Dict[str, Dict[str, Any]]:
        """Config fragments for the cellular and immune modules."""
        cellular: Dict[str, Any] = {
            "patient_mutations": [asdict(m) for m in self.mutations],
        }
        immune: Dict[str, Any] = {}
        # A cell transformed mid-run joins this patient's tumour; it must
        # not acquire the historical KRAS G12D the patient does not have.
        molecular: Dict[str, Any] = {"default_transform_driver": None}
        if self.hla_alleles:
            # Presentation scores each distinct allele once; a homozygous
            # typing such as HCC1395's lists every allele twice. NK
            # licensing gets the diploid list, where copy number can matter.
            cellular["hla_alleles"] = list(dict.fromkeys(self.hla_alleles))
            immune["patient_hla"] = list(self.hla_alleles)
            immune["patient_hla_measured"] = bool(self.hla_measured)
        return {"cellular": cellular, "immune": immune, "molecular": molecular}

    def summary(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "hla_alleles": list(self.hla_alleles),
            "hla_measured": self.hla_measured,
            "hla_method": self.hla_method,
            "mutations": [m.label for m in self.mutations],
            "skipped": [f"{g} {c}: {r}" for g, c, r in self.skipped],
        }


def _reference_proteins():
    from cognisom.genomics.gene_protein_mapper import BUILTIN_PROTEINS
    return BUILTIN_PROTEINS


def engine_inputs(profile: Any) -> PatientEngineInputs:
    """Map a ``PatientProfile`` onto engine inputs, reporting every skip.

    Reads ``coding_variants`` (the somatic coding variants the profile
    builder kept), ``cancer_driver_mutations`` to flag drivers, and the
    HLA fields. Duplicates of the same gene and change collapse to one.
    """
    proteins = _reference_proteins()
    drivers = {(getattr(v, "gene", None), getattr(v, "protein_change", None))
               for v in (getattr(profile, "cancer_driver_mutations", None) or [])}

    method = str(getattr(profile, "hla_typing_method", "unknown"))
    out = PatientEngineInputs(
        patient_id=str(getattr(profile, "patient_id", "anonymous")),
        hla_alleles=list(getattr(profile, "hla_alleles", None) or []),
        # The NK licensing model's own standard: a typing determined from
        # the patient's reads. hla_is_patient_specific alone is not
        # enough: it also covers the invented demo profiles, which are
        # specific to a synthetic patient but were never measured.
        hla_measured=method in READ_DERIVED_HLA_METHODS,
        hla_method=method,
    )

    seen = set()
    for variant in _iter_variants(getattr(profile, "coding_variants", None)):
        gene = getattr(variant, "gene", None)
        change = getattr(variant, "protein_change", None) or ""
        if not gene:
            out.skipped.append(("?", change, "no gene symbol"))
            continue
        parsed = parse_missense(change)
        if parsed is None:
            consequence = getattr(variant, "consequence", None) or "not missense"
            out.skipped.append((gene, change, f"{consequence}: no canonical "
                                              "missense neoantigen"))
            continue
        wild, position, mutant = parsed
        info = proteins.get(gene.upper())
        if info is None:
            out.skipped.append((gene, change, "gene not in the reference proteome"))
            continue
        sequence = info.sequence
        if position > len(sequence):
            out.skipped.append((gene, change, f"position {position} is past the "
                                              f"protein's {len(sequence)} residues"))
            continue
        if sequence[position - 1] != wild:
            out.skipped.append((
                gene, change,
                f"reference has {sequence[position - 1]} at {position}, not "
                f"{wild} -- likely a different numbering or transcript"))
            continue
        aa_change = f"{wild}{position}{mutant}"
        key = (gene.upper(), aa_change)
        if key in seen:
            continue
        seen.add(key)
        out.mutations.append(EngineMutation(
            gene=gene.upper(),
            aa_change=aa_change,
            label=f"{gene.upper()}_{aa_change}",
            oncogenic=(gene, change) in drivers,
        ))
    return out


def _iter_variants(variants: Optional[Iterable[Any]]):
    return list(variants or [])
