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

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


#: Fixed seed for conformer embedding. The same SMILES must give the same
#: coordinates on every machine, or a geometry test pins nothing and two
#: exports of one library differ for no reason.
CONFORMER_SEED = 0xC0FFEE


def embed_conformer(smiles: str, with_hydrogens: bool = True):
    """A molecule with 3-D coordinates, or as far as we got.

    Returns ``(mol, embedded)``. ``mol`` is None only if the SMILES does not
    parse. ``embedded`` is False when the distance-geometry step could not
    place the atoms, in which case the topology is still there and the
    coordinates are not -- callers that need geometry must check it, because
    an unembedded molecule reports every atom at the origin rather than
    raising.

    ETKDG for the initial placement, then MMFF94 minimisation. Both are
    rdkit defaults; the seed is ours.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, False
    if with_hydrogens:
        mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=CONFORMER_SEED) != 0:
        # Random starting coordinates rescue the few cages and macrocycles
        # that the default series of starts cannot place.
        if AllChem.EmbedMolecule(mol, randomSeed=CONFORMER_SEED,
                                 useRandomCoords=True) != 0:
            return mol, False
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    except Exception as exc:                        # no MMFF params for it
        log.debug("MMFF minimisation skipped for %s: %s", smiles, exc)
    return mol, True


def conformer_prim_count(smiles: str, with_hydrogens: bool = True) -> Tuple[int, int]:
    """Atoms and bonds in an embedded 3-D conformer: the prim budget.

    One prim per atom plus one per bond is the floor. The exporter in
    ``cognisom.visualization.molecular_usd`` draws each bond as two
    half-length cylinders so that each half carries its own atom's colour,
    which is what makes the elements readable, so what it actually emits is
    ``atoms + 2 * bonds``. This stays the floor because it is the figure to
    compare against the one sphere a molecule used to get.
    """
    mol, _ = embed_conformer(smiles, with_hydrogens)
    if mol is None:
        return (0, 0)
    return (mol.GetNumAtoms(), mol.GetNumBonds())


# ── backfill ─────────────────────────────────────────────────────────

#: Entity types a small-molecule structure can belong to.
STRUCTURED_TYPES = ("drug", "metabolite")


def _disagrees_with_record(entity, found: Dict) -> str:
    """Why a fetched structure contradicts the entity's own record, or "".

    Compared against what the LIBRARY already says, which is a different
    question from ``validate_structure``'s -- that one checks a
    structure against the formula PubChem returned with it, and both can
    be internally consistent while describing different species.
    """
    recorded_formula = (getattr(entity, "molecular_formula", "") or "").strip()
    fetched_formula = (found.get("molecular_formula") or "").strip()
    if recorded_formula and fetched_formula and \
            recorded_formula != fetched_formula:
        return (f"formula {fetched_formula} from PubChem, record says "
                f"{recorded_formula}")
    try:
        recorded_mw = float(getattr(entity, "molecular_weight", 0) or 0)
        fetched_mw = float(found.get("molecular_weight") or 0)
    except (TypeError, ValueError):
        return ""
    if recorded_mw and fetched_mw and abs(recorded_mw - fetched_mw) > 1.0:
        return (f"MW {fetched_mw:.2f} from PubChem, record says "
                f"{recorded_mw:.2f}")
    return ""


def backfill_structures(store, *, dry_run: bool = False,
                        limit: Optional[int] = None,
                        client: Any = None) -> Dict[str, Any]:
    """Fill every empty ``smiles`` this library has a structure for.

    This did not exist, and that is why it matters. The SMILES in the
    shipped database were written by a one-off pass that left no code
    behind, the database is gitignored, and so the structures lived on
    one machine and nowhere else. Deleting the file lost them, and
    nothing in the repository could put them back -- the library would
    have rebuilt itself clean, silently, with 0 molecules where it had
    had 24, and the molecular USD export would have raised "no library
    entities carry a SMILES" with no clue why.

    A sourcing pass that cannot be re-run is not a source. This one is
    re-runnable, refuses biologics before it queries (a name search for
    "Interferon alfa-2b" confidently returns a 746 Da compound
    containing Cl3I2Na), and cross-checks every structure against its
    own recorded formula and weight before storing it.

    Returns a summary: filled, skipped-as-biologic, not-found,
    failed-verification, and already-present.
    """
    summary: Dict[str, Any] = {
        "filled": [], "biologic": [], "not_found": [],
        "failed_check": [], "conflicts": [], "already": 0,
    }
    entities, _ = store.search(limit=10000)
    todo = [e for e in entities
            if e.entity_type.value in STRUCTURED_TYPES]
    if limit is not None:
        todo = todo[:limit]

    for entity in todo:
        if (getattr(entity, "smiles", "") or "").strip():
            summary["already"] += 1
            continue
        name = entity.name
        if is_biologic(name):
            summary["biologic"].append(name)
            continue
        found = resolve_small_molecule(name, client=client)
        if found is not None:
            conflict = _disagrees_with_record(entity, found)
            if conflict:
                # The structure verified against PubChem's own formula
                # and weight and still contradicts what this library
                # records for the same name. Storing it would silently
                # change which SPECIES the entity means.
                #
                # It is not a lookup failure. For lactate, pyruvate and
                # citrate PubChem's name search returns the conjugate
                # BASE -- C3H5O3- for lactate -- while the library
                # records the neutral acid, C3H6O3. Both are right about
                # their own molecule and they are different molecules;
                # at physiological pH the anion is the one a simulation
                # means, which makes this a decision about the record
                # rather than a bad fetch.
                summary["conflicts"].append(f"{name}: {conflict}")
                continue
        if found is None:
            # Separating these two matters: a miss is a gap in PubChem's
            # naming, a failed check is a structure that disagreed with
            # its own formula, and they need different work.
            summary["not_found"].append(name)
            continue
        if dry_run:
            summary["filled"].append(name)
            continue
        entity.smiles = found["smiles"]
        extra = getattr(entity, "_extra_properties", None)
        if isinstance(extra, dict):
            extra.setdefault("pubchem_cid", found.get("pubchem_cid"))
            extra.setdefault("inchikey", found.get("inchikey"))
            extra.setdefault("structure_source", "pubchem")
        store.update_entity(entity, changed_by="structures.backfill")
        summary["filled"].append(name)
    return summary


#: Where the backfilled structures are kept so that every library has them.
#:
#: backfill_structures made the pass RE-RUNNABLE, which was the first half of
#: the problem; this file is the second. The pass needs PubChem over the
#: network, and CI has neither the network nor a reason to depend on it, so a
#: fresh library built from the seeds alone carried exactly ONE structure --
#: the single SMILES literal in seed_data -- while this machine's database
#: carried twenty. tests/test_molecular_usd.py asserts twenty, so it passed
#: here and failed everywhere else, for two nights on main before anyone
#: looked. The same shape as curated_parameters.json beside it: a sourcing
#: pass runs once, its result is committed, and every library applies it.
STRUCTURES_PATH = Path(__file__).resolve().parent / "structures.json"

#: The properties a resolved structure writes. Kept narrow on purpose: these
#: are what resolve_small_molecule returns and what validate_structure can
#: check. Descriptors PubChem also offers (logP, TPSA, H-bond counts) are not
#: carried, because nothing recomputes them and an unverifiable number in an
#: evidence file is the thing this library exists to avoid.
STRUCTURE_FIELDS = ("smiles", "molecular_formula", "molecular_weight",
                    "inchikey", "inchi_key", "pubchem_cid", "structure_source")


def _structure_of(entity) -> Dict[str, Any]:
    """The structure an entity carries, as a flat dict, or {}.

    The formula, weight and InChIKey are RECOMPUTED from the SMILES rather
    than copied from the record, because the record mostly does not have
    them -- the library stores 0.0 where a weight was never filled in -- and
    because computing them here is what gives apply_structures something to
    check against. A SMILES that no longer produces this formula is a
    different molecule, and nobody reads a SMILES string and notices.
    """
    smiles = (getattr(entity, "smiles", "") or "").strip()
    if not smiles:
        return {}
    props = getattr(entity, "properties", None) or {}
    out: Dict[str, Any] = {"smiles": smiles}
    check = validate_structure(smiles)
    if check.smiles_parses:
        out["molecular_formula"] = check.formula
        out["molecular_weight"] = round(check.molecular_weight, 4)
        if check.inchikey:
            out["inchikey"] = check.inchikey
    for field_name in ("pubchem_cid", "structure_source"):
        value = props.get(field_name)
        if value not in (None, ""):
            out[field_name] = value
    return out


def export_structures(store, path: Path = STRUCTURES_PATH) -> int:
    """Write every structure this library holds to ``path``. Returns the count."""
    entities, _ = store.search(limit=10000)
    found = {}
    for entity in sorted(entities, key=lambda e: e.name):
        if entity.entity_type.value not in STRUCTURED_TYPES:
            continue
        structure = _structure_of(entity)
        if structure:
            found[entity.name] = structure
    payload = {
        "about": ("Chemical structures resolved from PubChem by "
                  "cognisom.library.structures.backfill_structures and verified "
                  "against each entity's own recorded formula and weight. "
                  "Committed because the pass needs the network and the library "
                  "database is local state; apply_structures puts them into any "
                  "fresh library. Regenerate with: python -m "
                  "cognisom.library.structures export"),
        "source": "PubChem (https://pubchem.ncbi.nlm.nih.gov/)",
        "entities": found,
    }
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    return len(found)


def apply_structures(store, path: Path = STRUCTURES_PATH,
                     *, overwrite: bool = False) -> Dict[str, List[str]]:
    """Put the committed structures into ``store``.

    Only fills an EMPTY structure unless ``overwrite``; a library that has
    resolved its own is left alone. Every SMILES is re-checked against the
    formula and weight recorded beside it before it is stored, so a damaged
    or edited file cannot quietly change which molecule an entity means --
    the same guard backfill_structures applies to PubChem's own answers.
    """
    report: Dict[str, List[str]] = {"written": [], "already": [],
                                    "missing": [], "refused": []}
    if not path.exists():
        return report
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:                                # noqa: BLE001
        log.warning("structures file unreadable (%s): %s", path, exc)
        report["refused"].append(f"{path}: {exc}")
        return report

    for name, structure in sorted((payload.get("entities") or {}).items()):
        smiles = (structure.get("smiles") or "").strip()
        if not smiles:
            continue
        entity = store.find_entity_by_name(name)
        if entity is None:
            report["missing"].append(name)
            continue
        if (getattr(entity, "smiles", "") or "").strip() and not overwrite:
            report["already"].append(name)
            continue
        check = validate_structure(
            smiles,
            expected_formula=structure.get("molecular_formula", ""),
            expected_mw=float(structure.get("molecular_weight") or 0) or None)
        if not check.ok:
            report["refused"].append(f"{name}: {'; '.join(check.problems)}")
            continue
        entity.smiles = smiles
        extra = getattr(entity, "_extra_properties", None)
        if isinstance(extra, dict):
            for field_name in STRUCTURE_FIELDS:
                value = structure.get(field_name)
                if field_name != "smiles" and value not in (None, ""):
                    extra.setdefault(field_name, value)
        store.update_entity(entity, changed_by="structures.apply")
        report["written"].append(name)
    if report["written"] or report["refused"]:
        log.info("structures applied: %d written, %d refused, %d already present",
                 len(report["written"]), len(report["refused"]),
                 len(report["already"]))
    return report


def main(argv=None) -> int:                                 # pragma: no cover
    """``python -m cognisom.library.structures`` -- fill or move the structures."""
    import argparse
    import contextlib
    import io

    parser = argparse.ArgumentParser(description=backfill_structures.__doc__)
    parser.add_argument("command", nargs="?", default="backfill",
                        choices=("backfill", "export", "apply"),
                        help="backfill: resolve from PubChem (needs the "
                             "network). export: write what this library holds "
                             "to structures.json. apply: put that file into "
                             "this library.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    from .store import EntityStore
    with contextlib.redirect_stdout(io.StringIO()):
        store = EntityStore()

    if args.command == "export":
        count = export_structures(store)
        print(f"wrote {count} structures to {STRUCTURES_PATH}")
        return 0
    if args.command == "apply":
        report = apply_structures(store)
        print(f"{len(report['written'])} written, "
              f"{len(report['already'])} already present, "
              f"{len(report['missing'])} entities absent, "
              f"{len(report['refused'])} refused")
        for line in report["refused"]:
            print(f"    {line}")
        return 0

    out = backfill_structures(store, dry_run=args.dry_run, limit=args.limit)
    print(f"already had a structure : {out['already']}")
    print(f"filled                  : {len(out['filled'])}")
    print(f"refused as a biologic   : {len(out['biologic'])}")
    print(f"no PubChem match        : {len(out['not_found'])}")
    for name in out["not_found"]:
        print(f"    {name}")
    print(f"conflicts with record   : {len(out['conflicts'])}")
    for line in out["conflicts"]:
        print(f"    {line}")
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
