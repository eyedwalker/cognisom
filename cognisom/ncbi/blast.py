"""
NCBI BLAST API Integration
===========================

Submit BLAST searches, poll for completion, and retrieve results.
Uses the BLAST URL API (https://blast.ncbi.nlm.nih.gov/Blast.cgi).
"""

import logging
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

BLAST_URL = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"


@dataclass
class BlastHit:
    """A single BLAST alignment hit."""
    accession: str
    title: str
    score: float
    evalue: float
    identity_pct: float
    query_cover: float
    length: int


@dataclass
class BlastResult:
    """Complete BLAST result."""
    rid: str
    program: str
    database: str
    query_length: int
    hits: List[BlastHit] = field(default_factory=list)
    status: str = "READY"


def submit_blast(sequence: str, program: str = "blastn",
                 database: str = "core_nt", expect: float = 10.0,
                 hitlist_size: int = 50) -> str:
    """Submit a BLAST search. Returns the Request ID (RID).

    Args:
        sequence: DNA or protein sequence (plain or FASTA).
        program: blastn, blastp, blastx, tblastn, tblastx.
        database: core_nt, nr, swissprot, refseq_protein, pdb, etc.
        expect: E-value threshold.
        hitlist_size: Max number of hits.

    Returns:
        RID string for polling.
    """
    params = {
        "CMD": "Put",
        "PROGRAM": program,
        "DATABASE": database,
        "QUERY": sequence,
        "EXPECT": str(expect),
        "HITLIST_SIZE": str(hitlist_size),
        "FORMAT_TYPE": "JSON2",
    }
    data = urllib.parse.urlencode(params).encode()
    req = urllib.request.Request(BLAST_URL, data=data,
                                headers={"User-Agent": "cognisom/2.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        text = resp.read().decode()

    # Parse RID from response
    rid_match = re.search(r"RID = (\S+)", text)
    if not rid_match:
        raise RuntimeError(f"Could not parse BLAST RID from response: {text[:200]}")

    rid = rid_match.group(1)
    log.info(f"BLAST submitted: RID={rid}, program={program}, db={database}")

    # Parse estimated time
    rtoe_match = re.search(r"RTOE = (\d+)", text)
    if rtoe_match:
        log.info(f"Estimated time: {rtoe_match.group(1)}s")

    return rid


def check_blast_status(rid: str) -> str:
    """Check BLAST job status. Returns 'WAITING', 'READY', or 'UNKNOWN'."""
    url = f"{BLAST_URL}?CMD=Get&RID={rid}&FORMAT_OBJECT=SearchInfo"
    req = urllib.request.Request(url, headers={"User-Agent": "cognisom/2.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        text = resp.read().decode()

    if "Status=WAITING" in text:
        return "WAITING"
    if "Status=READY" in text:
        return "READY"
    return "UNKNOWN"


def get_blast_results(rid: str, max_hits: int = 50) -> BlastResult:
    """Retrieve BLAST results as parsed BlastResult.

    Falls back to tabular text if JSON parsing fails.
    """
    # Try tabular format (most reliable parsing)
    url = (
        f"{BLAST_URL}?CMD=Get&RID={rid}"
        f"&FORMAT_TYPE=Tabular&HITLIST_SIZE={max_hits}"
        f"&ALIGNMENT_VIEW=Tabular"
        f"&DESCRIPTIONS={max_hits}&ALIGNMENTS={max_hits}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "cognisom/2.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        text = resp.read().decode()

    hits = _parse_tabular(text)

    return BlastResult(
        rid=rid,
        program="",
        database="",
        query_length=0,
        hits=hits,
        status="READY",
    )


def wait_for_blast(rid: str, timeout: int = 300, poll_interval: int = 15,
                   callback=None) -> str:
    """Poll until BLAST job completes or timeout.

    Args:
        rid: Request ID from submit_blast.
        timeout: Max seconds to wait.
        poll_interval: Seconds between polls (min 10 per NCBI guidelines).
        callback: Optional callable(status, elapsed) for progress updates.

    Returns:
        Final status string.
    """
    poll_interval = max(poll_interval, 10)  # NCBI requires >= 10s
    start = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            return "TIMEOUT"

        status = check_blast_status(rid)
        if callback:
            callback(status, elapsed)

        if status != "WAITING":
            return status

        time.sleep(poll_interval)


def _parse_tabular(text: str) -> List[BlastHit]:
    """Parse BLAST tabular output into BlastHit objects."""
    hits = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("<"):
            continue
        parts = line.split("\t")
        if len(parts) < 12:
            continue
        try:
            hits.append(BlastHit(
                accession=parts[1],
                title=parts[1],  # tabular doesn't include full title
                identity_pct=float(parts[2]),
                length=int(parts[3]),
                score=float(parts[11]),
                evalue=float(parts[10]),
                query_cover=0.0,
            ))
        except (ValueError, IndexError):
            continue
    return hits
