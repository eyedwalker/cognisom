"""
NCBI Sequence Fetching
======================

Fetch DNA/protein sequences by gene name, accession, or Gene ID.
"""

import logging
import re
import urllib.parse
from dataclasses import dataclass
from typing import List, Optional

from .client import NCBIClient

log = logging.getLogger(__name__)


@dataclass
class SequenceResult:
    """A fetched sequence with metadata."""
    accession: str
    description: str
    sequence: str
    length: int
    db: str  # "nucleotide" or "protein"


def fetch_sequence(query: str, db: str = "nucleotide",
                   rettype: str = "fasta") -> Optional[SequenceResult]:
    """Fetch a single sequence by accession or gene name.

    Args:
        query: Accession number (NM_007294.4), Gene ID, or gene name (BRCA1).
        db: Database — "nucleotide" or "protein".
        rettype: "fasta" (default) or "gb" for GenBank format.

    Returns:
        SequenceResult or None if not found.
    """
    client = NCBIClient()

    # If it looks like an accession (has letters + numbers, possibly with dots/underscores)
    if re.match(r'^[A-Z]{1,3}[_]?\d+(\.\d+)?$', query.strip(), re.IGNORECASE):
        ids = [query.strip()]
    else:
        # Search by gene name — get the RefSeq mRNA for nucleotide
        if db == "nucleotide":
            term = f"{query}[Gene Name] AND Homo sapiens[Organism] AND RefSeq[Filter] AND mRNA[Filter]"
        else:
            term = f"{query}[Gene Name] AND Homo sapiens[Organism] AND RefSeq[Filter]"
        result = client.esearch(db=db, term=term, retmax=1)
        ids = result.get("idlist", [])
        if not ids:
            # Broader search without RefSeq filter
            term = f"{query}[Gene Name] AND Homo sapiens[Organism]"
            result = client.esearch(db=db, term=term, retmax=1)
            ids = result.get("idlist", [])

    if not ids:
        return None

    raw = client.efetch(db=db, ids=ids, rettype=rettype, retmode="text")
    if not raw.strip():
        return None

    return _parse_fasta(raw.strip(), db) if rettype == "fasta" else _parse_raw(raw.strip(), db, ids[0])


def search_gene_sequences(gene: str, db: str = "nucleotide",
                          max_results: int = 5) -> List[SequenceResult]:
    """Search for multiple sequences by gene name.

    Returns a list of SequenceResult with truncated sequences (first 200 nt).
    """
    client = NCBIClient()
    term = f"{gene}[Gene Name] AND Homo sapiens[Organism]"
    result = client.esearch(db=db, term=term, retmax=max_results)
    ids = result.get("idlist", [])
    if not ids:
        return []

    # Get summaries for descriptions
    summaries = client.esummary(db=db, ids=ids)

    results = []
    for uid in ids:
        doc = summaries.get(uid, {})
        raw = client.efetch(db=db, ids=[uid], rettype="fasta", retmode="text")
        parsed = _parse_fasta(raw.strip(), db) if raw.strip() else None
        if parsed:
            parsed.description = doc.get("title", parsed.description)
            results.append(parsed)

    return results


def _parse_fasta(text: str, db: str) -> Optional[SequenceResult]:
    """Parse FASTA text into SequenceResult."""
    lines = text.split("\n")
    if not lines or not lines[0].startswith(">"):
        return None

    header = lines[0][1:].strip()
    # Extract accession from header (first word)
    accession = header.split()[0] if header else ""
    description = header

    sequence = "".join(line.strip() for line in lines[1:] if not line.startswith(">"))

    return SequenceResult(
        accession=accession,
        description=description,
        sequence=sequence,
        length=len(sequence),
        db=db,
    )


def _parse_raw(text: str, db: str, uid: str) -> SequenceResult:
    """Wrap raw text (GenBank, etc.) as a SequenceResult."""
    return SequenceResult(
        accession=uid,
        description=f"Raw {db} record",
        sequence=text,
        length=len(text),
        db=db,
    )
