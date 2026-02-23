"""
NCBI E-utilities Base Client
=============================

Shared HTTP helpers for all NCBI API calls.
Appends api_key, tool, and email to every request.
"""

import json
import logging
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL_NAME = "cognisom"
TOOL_EMAIL = "support@cognisom.com"

# Simple rate limiter: track last request time
_last_request_time = 0.0


def _get_api_key() -> str:
    return os.environ.get("NCBI_API_KEY", "")


def _rate_limit():
    """Enforce rate limit: 10 req/sec with key, 3 without."""
    global _last_request_time
    key = _get_api_key()
    min_interval = 0.1 if key else 0.34
    elapsed = time.time() - _last_request_time
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_request_time = time.time()


def _build_params(**kwargs) -> str:
    """Build URL params with api_key, tool, email appended."""
    params = {k: v for k, v in kwargs.items() if v is not None}
    params["tool"] = TOOL_NAME
    params["email"] = TOOL_EMAIL
    key = _get_api_key()
    if key:
        params["api_key"] = key
    return urllib.parse.urlencode(params)


def http_get(url: str, timeout: int = 30) -> str:
    """GET request with rate limiting and standard headers."""
    _rate_limit()
    req = urllib.request.Request(url, headers={"User-Agent": f"{TOOL_NAME}/2.0"})
    log.debug(f"GET {url[:120]}...")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode()


def http_get_json(url: str, timeout: int = 30) -> Any:
    """GET request returning parsed JSON."""
    return json.loads(http_get(url, timeout=timeout))


class NCBIClient:
    """Unified client for NCBI E-utilities."""

    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            os.environ["NCBI_API_KEY"] = api_key

    def esearch(self, db: str, term: str, retmax: int = 20,
                sort: str = "", usehistory: str = "") -> Dict:
        """Search a database and return UIDs."""
        params = _build_params(
            db=db, term=term, retmax=retmax, retmode="json",
            sort=sort or None, usehistory=usehistory or None,
        )
        url = f"{EUTILS_BASE}/esearch.fcgi?{params}"
        data = http_get_json(url)
        return data.get("esearchresult", {})

    def efetch(self, db: str, ids: List[str], rettype: str = "",
               retmode: str = "text", seq_start: int = 0,
               seq_stop: int = 0) -> str:
        """Fetch full records. Returns raw text (FASTA, GenBank, XML, etc.)."""
        params = _build_params(
            db=db, id=",".join(ids),
            rettype=rettype or None, retmode=retmode,
            seq_start=seq_start or None, seq_stop=seq_stop or None,
        )
        url = f"{EUTILS_BASE}/efetch.fcgi?{params}"
        return http_get(url)

    def esummary(self, db: str, ids: List[str]) -> Dict:
        """Get document summaries for UIDs."""
        params = _build_params(db=db, id=",".join(ids), retmode="json")
        url = f"{EUTILS_BASE}/esummary.fcgi?{params}"
        data = http_get_json(url)
        return data.get("result", {})

    def elink(self, dbfrom: str, db: str, ids: List[str]) -> Dict:
        """Find related records across databases."""
        params = _build_params(dbfrom=dbfrom, db=db, id=",".join(ids), retmode="json")
        url = f"{EUTILS_BASE}/elink.fcgi?{params}"
        return http_get_json(url)

    def efetch_xml(self, db: str, ids: List[str]) -> ET.Element:
        """Fetch records as parsed XML ElementTree."""
        raw = self.efetch(db, ids, retmode="xml")
        return ET.fromstring(raw)
