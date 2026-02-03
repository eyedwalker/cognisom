"""
Base NIM Client
===============

Shared HTTP client for all NVIDIA NIM API calls.
Handles authentication, retries, and error handling.
"""

import os
import json
import logging
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

# NVIDIA API base URLs
HEALTH_API_BASE = "https://health.api.nvidia.com/v1/biology"
NVCF_ASSETS_URL = "https://api.nvcf.nvidia.com/v2/nvcf/assets"


class NIMClient:
    """Base client for NVIDIA NIM API calls."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "NVIDIA_API_KEY not found. Set it in .env or pass api_key= argument."
            )
        self._base_url = base_url or HEALTH_API_BASE
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    def _post(self, endpoint: str, payload: Dict, timeout: int = 120) -> Dict:
        """Make a POST request to a NIM endpoint. Returns parsed JSON."""
        url = f"{self._base_url}/{endpoint}"
        logger.debug(f"POST {url}")
        r = self._session.post(url, json=payload, timeout=timeout)
        if r.status_code != 200:
            logger.error(f"NIM error {r.status_code}: {r.text[:500]}")
            r.raise_for_status()
        return r.json()

    def _post_raw(self, endpoint: str, payload: Dict, timeout: int = 120) -> str:
        """Make a POST request and return raw response text.

        Useful for NIMs that return non-JSON formats (PDB, mmCIF, a3m).
        """
        url = f"{self._base_url}/{endpoint}"
        logger.debug(f"POST (raw) {url}")
        r = self._session.post(url, json=payload, timeout=timeout)
        if r.status_code != 200:
            logger.error(f"NIM error {r.status_code}: {r.text[:500]}")
            r.raise_for_status()
        return r.text

    def upload_asset(self, data: str, content_type: str = "text/plain",
                     description: str = "asset") -> str:
        """Upload a file to NVCF assets and return the asset ID.

        Required for NIMs like DiffDock that need file uploads.
        """
        # Request upload URL
        r = self._session.post(
            NVCF_ASSETS_URL,
            json={"contentType": content_type, "description": description},
            timeout=30,
        )
        r.raise_for_status()
        info = r.json()
        asset_id = info["assetId"]
        upload_url = info["uploadUrl"]

        # Upload the data
        requests.put(
            upload_url,
            data=data,
            headers={
                "Content-Type": content_type,
                "x-amz-meta-nvcf-asset-description": description,
            },
            timeout=60,
        )
        logger.debug(f"Uploaded asset {asset_id}")
        return asset_id
