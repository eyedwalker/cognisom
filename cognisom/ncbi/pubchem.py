"""
PubChem PUG REST Integration
==============================

Search and retrieve compound data from PubChem.
Base URL: https://pubchem.ncbi.nlm.nih.gov/rest/pug
Rate limit: 5 requests/second.
"""

import json
import logging
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

_last_pubchem_time = 0.0


def _pubchem_rate_limit():
    global _last_pubchem_time
    elapsed = time.time() - _last_pubchem_time
    if elapsed < 0.2:  # 5 req/sec
        time.sleep(0.2 - elapsed)
    _last_pubchem_time = time.time()


def _pubchem_get(path: str, timeout: int = 30) -> str:
    _pubchem_rate_limit()
    url = f"{PUBCHEM_BASE}/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "cognisom/2.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode()


def _pubchem_get_json(path: str, timeout: int = 30) -> Any:
    return json.loads(_pubchem_get(path, timeout))


@dataclass
class CompoundInfo:
    """PubChem compound summary."""
    cid: int
    name: str
    molecular_formula: str
    molecular_weight: float
    canonical_smiles: str
    isomeric_smiles: str
    inchi_key: str
    xlogp: Optional[float] = None
    hbond_donors: int = 0
    hbond_acceptors: int = 0
    tpsa: float = 0.0
    rotatable_bonds: int = 0
    complexity: float = 0.0
    heavy_atom_count: int = 0
    charge: int = 0
    # Bioactivity
    bioassay_count: int = 0
    active_assay_count: int = 0


def search_compound(query: str, search_type: str = "name") -> Optional[CompoundInfo]:
    """Search PubChem for a compound by name or SMILES.

    Args:
        query: Compound name ("aspirin") or SMILES string.
        search_type: "name" or "smiles".

    Returns:
        CompoundInfo or None.
    """
    try:
        namespace = search_type
        encoded = urllib.parse.quote(query, safe="")
        path = f"compound/{namespace}/{encoded}/property/MolecularFormula,MolecularWeight,CanonicalSMILES,IsomericSMILES,InChIKey,XLogP,HBondDonorCount,HBondAcceptorCount,TPSA,RotatableBondCount,Complexity,HeavyAtomCount,Charge/JSON"
        data = _pubchem_get_json(path)

        props = data.get("PropertyTable", {}).get("Properties", [])
        if not props:
            return None

        p = props[0]
        cid = p.get("CID", 0)

        info = CompoundInfo(
            cid=cid,
            name=query if search_type == "name" else "",
            molecular_formula=p.get("MolecularFormula", ""),
            molecular_weight=p.get("MolecularWeight", 0.0),
            canonical_smiles=p.get("CanonicalSMILES", ""),
            isomeric_smiles=p.get("IsomericSMILES", ""),
            inchi_key=p.get("InChIKey", ""),
            xlogp=p.get("XLogP"),
            hbond_donors=p.get("HBondDonorCount", 0),
            hbond_acceptors=p.get("HBondAcceptorCount", 0),
            tpsa=p.get("TPSA", 0.0),
            rotatable_bonds=p.get("RotatableBondCount", 0),
            complexity=p.get("Complexity", 0.0),
            heavy_atom_count=p.get("HeavyAtomCount", 0),
            charge=p.get("Charge", 0),
        )

        # Get compound name if searched by SMILES
        if search_type == "smiles" and cid:
            try:
                name_data = _pubchem_get_json(f"compound/cid/{cid}/property/IUPACName/JSON")
                name_props = name_data.get("PropertyTable", {}).get("Properties", [])
                if name_props:
                    info.name = name_props[0].get("IUPACName", "")
            except Exception:
                pass

        return info

    except Exception as e:
        log.error(f"PubChem search failed for '{query}': {e}")
        return None


def get_compound_properties(cid: int) -> Optional[CompoundInfo]:
    """Get compound properties by PubChem CID."""
    return search_compound(str(cid), search_type="cid")


def similarity_search(smiles: str, threshold: float = 0.8,
                      max_results: int = 10) -> List[Dict[str, Any]]:
    """Find similar compounds by SMILES using PubChem 2D similarity.

    Args:
        smiles: Query SMILES string.
        threshold: Tanimoto similarity threshold (0-1).
        max_results: Max compounds to return.

    Returns:
        List of dicts with CID, SMILES, similarity info.
    """
    try:
        encoded = urllib.parse.quote(smiles, safe="")
        # PubChem similarity search (2D fingerprint)
        # Uses a two-step async process for large searches
        path = (
            f"compound/fastsimilarity_2d/smiles/{encoded}"
            f"/property/MolecularFormula,MolecularWeight,CanonicalSMILES,XLogP,TPSA"
            f"/JSON?Threshold={int(threshold * 100)}&MaxRecords={max_results}"
        )
        data = _pubchem_get_json(path, timeout=60)
        props = data.get("PropertyTable", {}).get("Properties", [])

        results = []
        for p in props[:max_results]:
            results.append({
                "cid": p.get("CID", 0),
                "smiles": p.get("CanonicalSMILES", ""),
                "formula": p.get("MolecularFormula", ""),
                "mw": p.get("MolecularWeight", 0.0),
                "xlogp": p.get("XLogP"),
                "tpsa": p.get("TPSA", 0.0),
                "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{p.get('CID', 0)}",
            })

        return results

    except Exception as e:
        log.error(f"PubChem similarity search failed: {e}")
        return []
