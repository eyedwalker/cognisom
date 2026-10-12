"""
Enzyme Kinetics Database Clients
==================================

Query SABIO-RK and BRENDA for experimentally measured kinetic constants
(Km, kcat, Ki, Kd) to replace estimated parameters in entity library
with real published values.

Each parameter gets source attribution:
    physics_params["_source"] = "SABIO-RK"
    physics_params["_pmid"] = "12482937"
    physics_params["_conditions"] = "37°C, pH 7.0, Homo sapiens"

SABIO-RK: Free REST API, no registration
BRENDA: SOAP API, requires academic registration
BioNumbers: Curated cellular quantities
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests

log = logging.getLogger(__name__)

SABIO_RK_BASE = "http://sabiork.h-its.org/sabioRestWebServices"


# ── Data Classes ──────────────────────────────────────────────────────

@dataclass
class KineticParameter:
    """A single kinetic measurement from a database."""
    parameter_type: str  # "Km", "kcat", "Ki", "Kd", "Vmax"
    value: float
    unit: str  # "uM", "s-1", "nM"
    substrate: str = ""
    enzyme: str = ""
    organism: str = ""
    tissue: str = ""
    ph: float = 0.0
    temperature_c: float = 0.0
    pmid: str = ""
    source_db: str = ""  # "SABIO-RK", "BRENDA", "BioNumbers"
    entry_id: str = ""
    conditions: str = ""


@dataclass
class EnzymeKinetics:
    """Complete kinetic profile for an enzyme/reaction."""
    enzyme_name: str
    ec_number: str = ""
    parameters: List[KineticParameter] = field(default_factory=list)
    reactions: List[str] = field(default_factory=list)

    def get_km(self, substrate: str = "") -> Optional[float]:
        """Get Km value, optionally filtered by substrate."""
        for p in self.parameters:
            if p.parameter_type == "Km":
                if not substrate or substrate.lower() in p.substrate.lower():
                    return p.value
        return None

    def get_kcat(self) -> Optional[float]:
        for p in self.parameters:
            if p.parameter_type == "kcat":
                return p.value
        return None

    def get_ki(self, inhibitor: str = "") -> Optional[float]:
        for p in self.parameters:
            if p.parameter_type == "Ki":
                if not inhibitor or inhibitor.lower() in p.substrate.lower():
                    return p.value
        return None

    def to_physics_params(self) -> Dict:
        """Convert to entity physics_params format with source attribution."""
        params = {}
        for p in self.parameters:
            key = f"{p.parameter_type.lower()}_{p.substrate.replace(' ', '_').lower()}" if p.substrate else p.parameter_type.lower()
            # Normalize units
            value = p.value
            if p.unit == "mM" and "km" in p.parameter_type.lower():
                key = f"km_{p.substrate.replace(' ', '_').lower()}_um"
                value = p.value * 1000  # mM → uM
            elif p.unit == "uM":
                key = f"km_{p.substrate.replace(' ', '_').lower()}_um"
            elif p.unit == "s-1" or p.unit == "1/s":
                key = f"kcat_per_s"
            elif p.unit == "nM":
                key = f"kd_{p.substrate.replace(' ', '_').lower()}_nm"

            params[key] = round(value, 4)
            params[f"_{key}_source"] = p.source_db
            if p.pmid:
                params[f"_{key}_pmid"] = p.pmid
            if p.organism:
                params[f"_{key}_organism"] = p.organism

        return params


# ── SABIO-RK Client ──────────────────────────────────────────────────

class KineticsSourceUnavailable(RuntimeError):
    """The kinetics source answered, but not with an API payload.

    SABIO-RK's REST paths currently return HTTP 200 with a 1182-byte
    HTML application shell for every query, including ones that
    certainly have records. The old code read that as an empty result
    set, so :func:`enrich_all_enzymes` reported
    ``{"enriched": 0, "total_params": 0, "errors": []}`` over the whole
    library -- a retired endpoint presented as "this enzyme has no
    kinetic data", for every enzyme at once.

    A source that cannot be reached is not a source that returned
    nothing. Raising keeps those two outcomes distinguishable, which is
    the only way a caller can tell "look somewhere else" from "there is
    nothing to find".
    """


#: Content types that can carry a SABIO-RK payload. A 200 response in
#: anything else is the web UI, not the API.
_API_CONTENT_TYPES = (
    "application/json", "application/xml", "text/xml",
    "application/sbml", "text/plain",
)


def _require_api_response(resp, url: str):
    """Raise unless ``resp`` is plausibly an API payload."""
    ctype = (resp.headers.get("content-type") or "").split(";")[0].strip()
    if resp.status_code != 200:
        raise KineticsSourceUnavailable(
            f"SABIO-RK {url} answered HTTP {resp.status_code}"
        )
    if ctype not in _API_CONTENT_TYPES:
        raise KineticsSourceUnavailable(
            f"SABIO-RK {url} answered HTTP 200 with content-type "
            f"{ctype!r} ({len(resp.content)} bytes) rather than an API "
            "response; the REST endpoint appears to have been retired"
        )
    return resp


class SABIORKClient:
    """Query SABIO-RK for reaction kinetics.

    SABIO-RK (sabiork.h-its.org) contains 100,000+ kinetic records
    for biochemical reactions with full parameter sets.

    Free REST API, no registration required.

    Example:
        client = SABIORKClient()
        kinetics = client.search_by_enzyme("Androgen receptor")
        km = kinetics.get_km("DHT")
    """

    def __init__(self):
        self.base_url = SABIO_RK_BASE
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/json"

    def search_by_enzyme(self, enzyme_name: str,
                         organism: str = "Homo sapiens",
                         limit: int = 20) -> EnzymeKinetics:
        """Search SABIO-RK for kinetic data by enzyme name."""
        result = EnzymeKinetics(enzyme_name=enzyme_name)

        try:
            # Search for entries
            query = f"EnzymeName:{enzyme_name}"
            if organism:
                query += f" AND Organism:{organism}"

            url = f"{self.base_url}/searchKineticLaws/sbml"
            params = {"q": query, "format": "json"}

            resp = self.session.get(url, params=params, timeout=15)
            _require_api_response(resp, url)

            url2 = f"{self.base_url}/kineticLaws"
            params2 = {"q": query, "format": "json", "limit": limit}
            resp2 = self.session.get(url2, params=params2, timeout=15)
            _require_api_response(resp2, url2)

            try:
                entries = resp2.json()
            except ValueError as e:
                raise KineticsSourceUnavailable(
                    f"SABIO-RK {url2} returned undecodable JSON: {e}"
                ) from e

            for entry in entries[:limit]:
                self._parse_entry(entry, result)

        except KineticsSourceUnavailable:
            # Deliberately not swallowed: an unreachable source must not
            # look like an enzyme with no published kinetics.
            raise
        except requests.RequestException as e:
            raise KineticsSourceUnavailable(
                f"SABIO-RK request failed for {enzyme_name}: {e}"
            ) from e

        return result

    def search_by_reaction(self, substrate: str, product: str = "",
                           organism: str = "Homo sapiens") -> EnzymeKinetics:
        """Search SABIO-RK by reaction substrate/product."""
        result = EnzymeKinetics(enzyme_name=f"{substrate} reaction")

        try:
            query = f"Substrate:{substrate}"
            if product:
                query += f" AND Product:{product}"
            if organism:
                query += f" AND Organism:{organism}"

            url = f"{self.base_url}/kineticLaws"
            params = {"q": query, "format": "json", "limit": 10}
            resp = self.session.get(url, params=params, timeout=15)
            _require_api_response(resp, url)

            try:
                entries = resp.json()
            except ValueError as e:
                raise KineticsSourceUnavailable(
                    f"SABIO-RK {url} returned undecodable JSON: {e}"
                ) from e
            for entry in entries:
                self._parse_entry(entry, result)

        except KineticsSourceUnavailable:
            raise
        except requests.RequestException as e:
            raise KineticsSourceUnavailable(
                f"SABIO-RK reaction request failed: {e}"
            ) from e

        return result

    def get_entry_by_id(self, entry_id: int) -> Optional[KineticParameter]:
        """Get a specific SABIO-RK entry by ID."""
        try:
            url = f"{self.base_url}/kineticLaws/{entry_id}"
            resp = self.session.get(url, params={"format": "json"}, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                # Parse kinetic parameters from entry
                return self._extract_parameter(data)
        except Exception as e:
            log.debug("SABIO-RK entry %d error: %s", entry_id, e)
        return None

    def _parse_entry(self, entry: dict, result: EnzymeKinetics):
        """Extract kinetic parameters from a SABIO-RK entry."""
        if not isinstance(entry, dict):
            return

        params = entry.get("parameters", entry.get("kineticParameters", []))
        if isinstance(params, list):
            for p in params:
                param = self._extract_parameter(p)
                if param:
                    result.parameters.append(param)

    def _extract_parameter(self, data: dict) -> Optional[KineticParameter]:
        """Extract a single kinetic parameter from entry data."""
        if not isinstance(data, dict):
            return None

        ptype = data.get("type", data.get("parameterType", ""))
        value = data.get("value", data.get("startValue", 0))
        unit = data.get("unit", "")

        if not ptype or not value:
            return None

        # Normalize parameter type
        ptype_map = {
            "Km": "Km", "km": "Km", "KM": "Km",
            "kcat": "kcat", "Kcat": "kcat",
            "Ki": "Ki", "ki": "Ki",
            "Kd": "Kd", "kd": "Kd",
            "Vmax": "Vmax", "vmax": "Vmax",
        }
        ptype = ptype_map.get(ptype, ptype)

        return KineticParameter(
            parameter_type=ptype,
            value=float(value),
            unit=unit,
            substrate=data.get("substrate", data.get("compound", "")),
            enzyme=data.get("enzyme", ""),
            organism=data.get("organism", ""),
            ph=float(data.get("ph", 0)),
            temperature_c=float(data.get("temperature", 0)),
            pmid=str(data.get("pubmedId", data.get("pmid", ""))),
            source_db="SABIO-RK",
            entry_id=str(data.get("entryId", data.get("id", ""))),
        )


# ── Entity Enrichment with Kinetics ─────────────────────────────────

def enrich_entity_kinetics(store, entity_name: str,
                            entity_type: str = None) -> Dict:
    """Enrich a single entity with kinetic data from SABIO-RK.

    Args:
        store: EntityStore
        entity_name: Name of entity to enrich
        entity_type: Optional entity type filter

    Returns:
        Dict with enrichment results
    """
    entity = store.find_entity_by_name(entity_name, entity_type)
    if not entity:
        return {"error": f"Entity '{entity_name}' not found"}

    client = SABIORKClient()
    kinetics = client.search_by_enzyme(entity_name)

    if not kinetics.parameters:
        # Try searching by gene name as enzyme
        kinetics = client.search_by_enzyme(entity_name.split("(")[0].strip())

    if not kinetics.parameters:
        return {"entity": entity_name, "parameters_found": 0}

    # Convert to physics_params format
    new_params = kinetics.to_physics_params()

    # Merge with existing (don't overwrite user-set values)
    existing = entity.physics_params if hasattr(entity, "physics_params") else {}
    for key, value in new_params.items():
        if key not in existing or key.startswith("_"):
            existing[key] = value

    entity.physics_params = existing

    try:
        store.update_entity(entity, changed_by="sabio_rk_enrichment")
    except Exception as e:
        return {"error": str(e)}

    return {
        "entity": entity_name,
        "parameters_found": len(kinetics.parameters),
        "parameters_added": len(new_params),
        "source": "SABIO-RK",
    }


def enrich_all_enzymes(store, progress_callback=None) -> Dict:
    """Enrich all protein/gene entities with kinetic data.

    ``source_available`` is False when the kinetics source could not be
    reached at all. Without it, a caller cannot tell a library whose
    enzymes have no published kinetics from a library that was never
    asked -- both used to come back as ``enriched: 0`` with an empty
    error list.
    """
    results = {
        "enriched": 0, "total_params": 0, "errors": [],
        "source_available": True,
    }

    for etype in ["gene", "protein"]:
        entities, total = store.search(entity_type=etype, limit=100)
        for i, entity in enumerate(entities):
            if progress_callback:
                progress_callback(f"[{i+1}/{total}] {entity.name}")

            try:
                result = enrich_entity_kinetics(store, entity.name, etype)
                if result.get("parameters_found", 0) > 0:
                    results["enriched"] += 1
                    results["total_params"] += result["parameters_found"]
            except KineticsSourceUnavailable as e:
                # One unreachable source means every remaining lookup
                # would fail the same way. Stop and say so instead of
                # grinding through the library to report a clean zero.
                results["source_available"] = False
                results["errors"].append(f"kinetics source unavailable: {e}")
                return results
            except Exception as e:
                results["errors"].append(f"{entity.name}: {e}")

            time.sleep(0.5)  # Rate limiting

    return results
