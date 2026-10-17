"""Resolve library entities to real chemical structures, and check them.

The library shipped with the schema slots for structure and almost nothing in
them: 37 ``smiles`` keys with one value, 34 ``uniprot_id`` keys with none, 25
``drugbank_id`` keys with none. The one filled SMILES was wrong -- it parsed to
C28H24F6N4O2 (InChIKey KPEMKSCTFNDIQU) under the name Enzalutamide, which is
C21H16F4N4O2S (WXCXUHSOUPDCQV). Different formula, 40 atoms against 32, five
rings against three, and no sulfur where enzalutamide carries a thiohydantoin.

So this module does two things, and the second matters more than the first:

  * :func:`resolve_small_molecule` fetches a structure from PubChem through the
    client the repo already had in ``external_sources``.
  * :func:`validate_structure` checks a structure against its own recorded
    formula and weight. A SMILES string is unreadable to a person, so a wrong
    one is invisible until something recomputes from it. That is exactly how
    the Enzalutamide record survived.

BIOLOGICS ARE NOT SMALL MOLECULES
---------------------------------
A quarter of the library's "drugs" are monoclonal antibodies or recombinant
proteins -- pembrolizumab, nivolumab, rituximab, aldesleukin, interferon
alfa-2b. A 150 kDa antibody has no useful SMILES, and PubChem name search does
not fail cleanly on them: "Interferon alfa-2b" matches CID 71306834, a 746 Da
compound containing Cl3I2Na that is not interferon at all. Name lookup is
confident and wrong, so :data:`BIOLOGIC_PATTERNS` refuses these before a query
is made rather than trusting the result.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger(__name__)

#: Names that denote a protein drug. Matched before any lookup, because a name
#: query against a small-molecule database answers for the wrong molecule
#: instead of returning nothing.
BIOLOGIC_PATTERNS = (
    r"mab$", r"mab\b", r"cept$", r"interferon", r"interleukin", r"aldesleukin",
    r"\bIL-\d", r"peg", r"PSMA", r"^Lutetium", r"toxin", r"^Anti-",
)

#: Relative tolerance when comparing a recomputed molecular weight with a
#: recorded one. PubChem rounds to one decimal, so exact equality is wrong.
MW_RTOL = 0.01


def is_biologic(name: str) -> bool:
    """Whether a name denotes a protein drug rather than a small molecule."""
    return any(re.search(p, name, re.IGNORECASE) for p in BIOLOGIC_PATTERNS)


@dataclass
class StructureCheck:
    """The outcome of checking one structure against its own record."""

    ok: bool
    smiles_parses: bool = False
    formula: str = ""
    molecular_weight: float = 0.0
    inchikey: str = ""
    problems: list = field(default_factory=list)

    def __bool__(self) -> bool:                 # pragma: no cover - sugar
        return self.ok


def validate_structure(
    smiles: str,
    expected_formula: Optional[str] = None,
    expected_mw: Optional[float] = None,
) -> StructureCheck:
    """Recompute a structure from its SMILES and compare it with the record.

    This is the check that catches a structure field holding the wrong
    molecule. Nobody reads a SMILES string and notices; a formula recomputed
    from it and compared against the recorded formula is noticed immediately.

    ``expected_mw`` of 0.0 is treated as absent, because 0.0 is what the
    library stores where a weight was never filled in -- 430 of its 598 numeric
    properties are exactly zero -- and comparing against it would fail every
    real molecule.
    """
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Descriptors, rdMolDescriptors
        RDLogger.DisableLog("rdApp.*")
    except ImportError:  # pragma: no cover - rdkit is declared in requirements
        return StructureCheck(ok=False, problems=["rdkit not installed"])

    if not smiles:
        return StructureCheck(ok=False, problems=["no SMILES"])

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return StructureCheck(ok=False, problems=["SMILES does not parse"])

    formula = rdMolDescriptors.CalcMolFormula(mol)
    mw = float(Descriptors.MolWt(mol))
    try:
        inchikey = Chem.MolToInchiKey(mol)
    except Exception:                            # pragma: no cover
        inchikey = ""

    problems = []
    if expected_formula and expected_formula != formula:
        problems.append(
            f"formula {formula} from SMILES, record says {expected_formula}")
    if expected_mw:                              # 0.0 means "never filled in"
        if abs(mw - expected_mw) > MW_RTOL * max(expected_mw, 1.0):
            problems.append(
                f"MW {mw:.2f} from SMILES, record says {expected_mw}")

    return StructureCheck(
        ok=not problems, smiles_parses=True, formula=formula,
        molecular_weight=mw, inchikey=inchikey, problems=problems,
    )


def resolve_small_molecule(name: str, client: Any = None) -> Optional[Dict]:
    """Look one small molecule up in PubChem and verify what comes back.

    Returns ``None`` for a biologic, for a miss, and for a hit that fails its
    own cross-check -- a result that cannot be verified is not an answer.
    """
    if is_biologic(name):
        log.info("%s is a biologic; small-molecule lookup refused", name)
        return None

    if client is None:
        from cognisom.library.external_sources import PubChemClient
        client = PubChemClient()

    try:
        hit = client.get_compound(name)
    except Exception as exc:
        log.warning("PubChem lookup failed for %s: %s", name, exc)
        return None
    if hit is None:
        return None

    smiles = getattr(hit, "smiles", "") or ""
    formula = getattr(hit, "molecular_formula", "") or ""
    mw = float(getattr(hit, "molecular_weight", 0) or 0)

    check = validate_structure(smiles, expected_formula=formula, expected_mw=mw)
    if not check.ok:
        log.warning("%s: PubChem structure failed its own check: %s",
                    name, "; ".join(check.problems))
        return None

    return {
        "smiles": smiles,
        "molecular_formula": formula,
        "molecular_weight": mw,
        "inchikey": check.inchikey,
        "pubchem_cid": getattr(hit, "cid", None),
        "structure_source": "pubchem",
        "structure_retrieved": None,   # caller stamps the date it ran
    }


def conformer_prim_count(smiles: str, with_hydrogens: bool = True) -> Tuple[int, int]:
    """Atoms and bonds in an embedded 3-D conformer: the prim budget.

    A ball-and-stick rendering is one prim per atom plus one per bond, so this
    is what a single entity costs when it stops being a sphere.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return (0, 0)
    if with_hydrogens:
        mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE) != 0:
        return (mol.GetNumAtoms(), mol.GetNumBonds())
    AllChem.MMFFOptimizeMolecule(mol)
    return (mol.GetNumAtoms(), mol.GetNumBonds())
