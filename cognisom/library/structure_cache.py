"""Fetch, cache and parse 3-D structures for library entities.

Two sources, and the difference between them is the whole point:

  * **AlphaFold DB** gives exactly one model per UniProt accession, covering the
    full canonical sequence. There is no selection problem and no coverage
    problem -- and it is a prediction, carrying per-residue confidence (pLDDT)
    that varies enormously within a single protein.
  * **RCSB PDB** gives experimental structures, but dozens to hundreds per
    accession, each a different construct. Measured on this library: EGFR has
    393 entries, TP53 323, AR 95. Best coverage sampled was 29% for AR (the
    ligand-binding domain), 50% for TP53, 60% for PD-1. The first hit is worse
    still -- TP53's is a 30-residue tetramerisation peptide.

WHY pLDDT IS NOT A FOOTNOTE
---------------------------
"Full coverage" from AlphaFold means full *extent*, not full *confidence*.
Measured across 16 of this library's proteins, the fraction of residues above
pLDDT 70 runs from 38% to 90%, median 76%. One protein is an outlier and it is
the one this project cares most about:

    Androgen Receptor  920 res   median pLDDT 40.0   38% confident   58% below 50
    TLR4               839 res   median pLDDT 95.5   90% confident    5% below 50

AR's N-terminal domain is intrinsically disordered, so its predicted
coordinates there are not wrong so much as meaningless -- a shape where the
protein has no single shape. Rendering that solid, next to a confident domain,
shows a viewer a structure the model does not claim. Confidence travels with
the coordinates here so a renderer can act on it.

The mmCIF reader below is deliberately minimal rather than a dependency:
``_atom_site`` is a plain ``loop_`` table, and parsing it needs forty lines.
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction"
RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_ENTRY = "https://data.rcsb.org/rest/v1/core/polymer_entity"

#: pLDDT bands as AlphaFold itself defines them.
PLDDT_CONFIDENT = 70.0
PLDDT_VERY_LOW = 50.0

#: Where cached structure files live. Kept out of the repo: these are large,
#: re-fetchable, and versioned upstream.
DEFAULT_CACHE = os.path.expanduser("~/.cache/cognisom/structures")


@dataclass
class Structure:
    """One parsed structure, with the provenance needed to read it honestly."""

    accession: str
    source: str                     # "alphafold" | "pdb"
    identifier: str                 # AF model id, or PDB entry id
    ca_xyz: np.ndarray              # (n_res, 3) alpha-carbon coordinates, angstrom
    ca_plddt: Optional[np.ndarray]  # (n_res,) per-residue confidence, AlphaFold only
    n_atoms: int
    n_residues: int
    sequence_length: Optional[int] = None   # canonical length, for coverage
    version: str = ""
    retrieved: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def coverage(self) -> Optional[float]:
        """Fraction of the canonical sequence this structure contains."""
        if not self.sequence_length:
            return None
        return self.n_residues / self.sequence_length

    @property
    def confident_fraction(self) -> Optional[float]:
        """Fraction of residues AlphaFold calls confident (pLDDT > 70)."""
        if self.ca_plddt is None or not len(self.ca_plddt):
            return None
        return float((self.ca_plddt > PLDDT_CONFIDENT).mean())

    @property
    def is_prediction(self) -> bool:
        return self.source == "alphafold"


def parse_mmcif_atoms(text: str) -> Tuple[List[str], List[List[str]]]:
    """Columns and rows of the ``_atom_site`` loop. No dependency needed."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue
        j = i + 1
        headers: List[str] = []
        while j < len(lines) and lines[j].strip().startswith("_"):
            headers.append(lines[j].strip())
            j += 1
        if headers and headers[0].startswith("_atom_site."):
            cols = [h.split(".", 1)[1] for h in headers]
            rows = []
            while j < len(lines) and lines[j].strip() and not lines[j].startswith("#"):
                rows.append(lines[j].split())
                j += 1
            return cols, rows
        i = j
    return [], []


class StructureFetchError(RuntimeError):
    """A fetch failed for reasons that are not "this structure does not exist".

    Separate from a clean miss on purpose. Conflating the two writes a
    permanent gap into the library on the strength of one bad response.
    """


def _fetch_with_retry(session, accession, attempts: int = 4):
    """Return ``(text, version, absent)``.

    ``absent`` is True only when the API answered cleanly that there is no
    model. Every other failure returns ``absent=False`` so the caller can tell
    a real gap from a bad minute on the network.
    """
    import time as _time

    delay = 1.0
    for attempt in range(attempts):
        try:
            meta = session.get(f"{ALPHAFOLD_API}/{accession}", timeout=30)
            if meta.status_code == 404:
                return None, "", True
            if meta.status_code == 200:
                payload = meta.json()
                if not payload:
                    return None, "", True          # answered, and the answer is none
                entry = payload[0]
                url = entry.get("cifUrl")
                if not url:
                    return None, "", True
                resp = session.get(url, timeout=120)
                if resp.status_code == 200:
                    return resp.text, str(entry.get("latestVersion", "")), False
        except Exception as exc:                   # network-level, always retryable
            log.debug("AlphaFold attempt %d for %s: %s", attempt + 1, accession, exc)
        if attempt < attempts - 1:
            _time.sleep(delay)
            delay *= 2
    return None, "", False


def _cache_path(cache_dir: str, accession: str) -> str:
    return os.path.join(cache_dir, f"AF-{accession}.cif.gz")


def fetch_alphafold(
    accession: str,
    cache_dir: str = DEFAULT_CACHE,
    session: object = None,
    sequence_length: Optional[int] = None,
) -> Optional[Structure]:
    """The AlphaFold model for one accession, from cache or the network.

    Cached gzipped because these are ~800 KB of text each and 137 of them is
    most of a gigabyte uncompressed.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(cache_dir, accession)

    text = None
    version = ""
    if os.path.exists(path):
        with gzip.open(path, "rt") as fh:
            text = fh.read()
    else:
        if session is None:
            import requests
            session = requests.Session()
            session.headers["User-Agent"] = "cognisom-library/1.0"
        # A transient failure must not be recorded as "no model". Fetching 133
        # accessions in a loop without backoff reported 30 as missing, among
        # them BCL-2, Fibronectin and cytochrome c -- all of which had been
        # fetched successfully minutes earlier. A gap that is really a network
        # hiccup is the most durable kind of wrong data: nothing ever retries
        # it, and the entity is simply blank forever.
        text, version, absent = _fetch_with_retry(session, accession)
        if text is None:
            if absent:
                log.info("AlphaFold has no model for %s", accession)
            else:
                raise StructureFetchError(
                    f"AlphaFold fetch for {accession} failed after retries; "
                    f"this is a transient failure, not a missing model")
            return None
        with gzip.open(path, "wt") as fh:
            fh.write(text)

    cols, rows = parse_mmcif_atoms(text)
    if not cols:
        return None
    ix = {c: n for n, c in enumerate(cols)}
    ca = [r for r in rows if r[ix["label_atom_id"]] == "CA"]
    if not ca:
        return None

    xyz = np.array([[float(r[ix["Cartn_x"]]), float(r[ix["Cartn_y"]]),
                     float(r[ix["Cartn_z"]])] for r in ca], dtype=float)
    plddt = np.array([float(r[ix["B_iso_or_equiv"]]) for r in ca], dtype=float)

    return Structure(
        accession=accession, source="alphafold",
        identifier=f"AF-{accession}-F1", ca_xyz=xyz, ca_plddt=plddt,
        n_atoms=len(rows), n_residues=len(ca),
        sequence_length=sequence_length, version=version,
    )


def confidence_bands(structure: Structure) -> Dict[str, int]:
    """Residue counts per AlphaFold confidence band.

    A renderer that treats all residues alike shows a disordered tail as solidly
    as a folded domain, which is the single most misleading thing it can do with
    a predicted model.
    """
    if structure.ca_plddt is None:
        return {}
    p = structure.ca_plddt
    return {
        "very_high": int((p >= 90).sum()),
        "confident": int(((p >= 70) & (p < 90)).sum()),
        "low": int(((p >= 50) & (p < 70)).sum()),
        "very_low": int((p < 50).sum()),
    }


#: Mean PAE below which AlphaFold is taken to assert the relative placement of
#: two confident segments. 5 A is the conventional reading of "same rigid body".
#:
#: 5 A is the conventional reading of "same rigid body", and measured here it
#: is too strict: TLR4's leucine-rich-repeat ectodomain splits at 7.44 A
#: although it is one continuous solenoid. 8 A joins it and still separates
#: the ectodomain from the TIR domain across the membrane, which sits at
#: 25-28 A.
#:
#: The choice barely matters for the cases that matter, which is the real
#: reassurance. Measured at both 5 and 8 A:
#:
#:     AR     3 groups either way -- placement never asserted
#:     TP53   2 groups either way -- DBD halves together, tetramerisation apart
#:     TLR4   5 groups at 5 A, 3 at 8 A -- only this one moves
#:
#: Treat a split as "the model does not assert this placement", never as
#: "these are separate domains". The first is what PAE measures; the second is
#: a biological claim it cannot make.
RIGID_PAE_MAX = 8.0


def confident_segments(
    plddt, threshold: float = PLDDT_CONFIDENT, min_length: int = 20
) -> List[Tuple[int, int]]:
    """Runs of confidently predicted residues, as 1-based inclusive spans.

    Short dips below the threshold split a run, so a continuous domain can
    arrive as several segments. That is why :func:`rigid_groups` exists -- the
    segments are a starting partition, not a domain assignment.
    """
    out: List[Tuple[int, int]] = []
    start = None
    for i, p in enumerate(plddt):
        if p >= threshold and start is None:
            start = i
        elif p < threshold and start is not None:
            if i - start >= min_length:
                out.append((start + 1, i))
            start = None
    if start is not None and len(plddt) - start >= min_length:
        out.append((start + 1, len(plddt)))
    return out


def fetch_pae(accession: str, session: object = None, cache_dir: str = DEFAULT_CACHE):
    """The predicted aligned error matrix, (n_res, n_res), or None.

    pLDDT says how sure the model is of a residue's own position. It says
    nothing about whether two confident regions are correctly placed relative
    to each other, and that is a separate and often worse uncertainty.
    Measured on this library:

        AR    three confident segments, pairwise PAE 26.5-28.9 A
        TP53  DNA-binding halves 3.45 A, but DBD to tetramerisation 20.1 A

    So AR's domains are each confidently folded and their arrangement is not
    claimed at all. Drawing them in one coordinate frame asserts a quaternary
    structure the model declines to assert -- the same class of overclaim as
    rendering a disordered tail solid, one level up.
    """
    import json as _json
    import os as _os

    _os.makedirs(cache_dir, exist_ok=True)
    path = _os.path.join(cache_dir, f"PAE-{accession}.json.gz")
    if _os.path.exists(path):
        with gzip.open(path, "rt") as fh:
            return np.array(_json.load(fh), dtype=float)

    if session is None:
        import requests
        session = requests.Session()
        session.headers["User-Agent"] = "cognisom-library/1.0"
    text, _version, absent = None, "", False
    try:
        meta = session.get(f"{ALPHAFOLD_API}/{accession}", timeout=30)
        if meta.status_code != 200 or not meta.json():
            return None
        url = meta.json()[0].get("paeDocUrl")
        if not url:
            return None
        resp = session.get(url, timeout=180)
        if resp.status_code != 200:
            raise StructureFetchError(
                f"PAE fetch for {accession} returned {resp.status_code}")
        payload = resp.json()
        block = payload[0] if isinstance(payload, list) else payload
        matrix = block.get("predicted_aligned_error")
        if matrix is None:
            return None
    except StructureFetchError:
        raise
    except Exception as exc:
        raise StructureFetchError(f"PAE fetch for {accession} failed: {exc}")

    with gzip.open(path, "wt") as fh:
        _json.dump(matrix, fh)
    return np.array(matrix, dtype=float)


def rigid_groups(
    segments: List[Tuple[int, int]],
    pae: "np.ndarray",
    max_pae: float = RIGID_PAE_MAX,
) -> List[List[Tuple[int, int]]]:
    """Group segments whose relative placement the model does assert.

    Segments land in one group when their mean inter-segment PAE is at or
    below ``max_pae``, transitively. Separate groups must not be drawn in one
    shared frame as though their arrangement were meaningful.
    """
    n = len(segments)
    if n == 0:
        return []
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a0, a1 = segments[i]
            b0, b1 = segments[j]
            if float(pae[a0 - 1:a1, b0 - 1:b1].mean()) <= max_pae:
                parent[find(i)] = find(j)

    buckets: Dict[int, List[Tuple[int, int]]] = {}
    for i, seg in enumerate(segments):
        buckets.setdefault(find(i), []).append(seg)
    return [sorted(v) for _, v in sorted(buckets.items())]


def pdb_entry_count(accession: str, session: object = None) -> Optional[int]:
    """How many experimental entries exist for an accession.

    Useful as a fact to record even when an experimental structure is not the
    one attached: "95 entries exist, none covering more than 29%" is a more
    honest field than silence.
    """
    if session is None:
        import requests
        session = requests.Session()
        session.headers["User-Agent"] = "cognisom-library/1.0"
    query = {
        "query": {"type": "group", "logical_operator": "and", "nodes": [
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": ("rcsb_polymer_entity_container_identifiers"
                              ".reference_sequence_identifiers.database_accession"),
                "operator": "exact_match", "value": accession}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": ("rcsb_polymer_entity_container_identifiers"
                              ".reference_sequence_identifiers.database_name"),
                "operator": "exact_match", "value": "UniProt"}}]},
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": 1},
                            "results_verbosity": "compact"},
    }
    try:
        resp = session.post(RCSB_SEARCH, json=query, timeout=30)
        if resp.status_code != 200:
            return None
        return resp.json().get("total_count")
    except Exception as exc:
        log.warning("RCSB count failed for %s: %s", accession, exc)
        return None
