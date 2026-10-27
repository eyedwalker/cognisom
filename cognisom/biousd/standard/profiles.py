"""UsdProfiles capabilities and claims for Bio-USD.

OpenUSD 26.08 added UsdProfiles: a registry of named *capabilities*
(reverse-domain tokens in a DAG under the one root ``usd``, declared in a
plugin's ``plugInfo.json`` under ``Info.Profiles.Capabilities``) and one
applied schema, ``UsdProfilesClaimsAPI``, with no properties, that stores
what a stage *uses* and which *profiles* it declares itself compatible
with in the prim's ``customData["profilesInfo"]``. A consumer then asks
``ClaimsAPI.IsCompatibleWith(profile)`` instead of opening the file to
find out. Upstream ships four capabilities and no profile, and nothing
checks a claim; it is infrastructure, and this module is Bio-USD's use
of it.

What was measured before writing this (OpenUSD 26.8 and 26.11, scripts in
the project's scratch area):

- The capability block in the SAME ``plugInfo.json`` as the schema plugin
  registers; two plugins with one ``Name`` are dropped first-wins, so
  there is one plugin file and this module renders its Profiles block.
- ``BioCellAPI`` declares ``impliesCapabilities = ["biousd.cells"]`` in
  its schema ``customData.extraPlugInfo``; usdGenSchema copies it into
  the plugin's ``Types`` entry and ``ProfileRegistry.
  GetSchemaImpliedCapabilities`` returns it.
- The claims are plain metadata. A writer on usd-core 25.11, which has
  no ``pxr.UsdProfiles`` at all, authors the identical file with
  ``AddAppliedSchema("ClaimsAPI")`` and ``SetCustomDataByKey``; a 26.11
  reader with the plugin answers ``ValidPath``, and without the plugin
  ``NoPath`` (it does not know the capabilities), which is the honest
  answer. The exception list must be a *typed* empty ``Vt.TokenArray()``:
  a bare ``[]`` cannot be typed and raises on the first run.

Claims go on the default prim, where upstream's examples put them, and
the writer records ``biousd.cells`` as ``hard`` (a consumer without it
cannot present the cells correctly) and ``biousd.ontology`` as
``enhancement`` (a consumer without it still draws the scene).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .vocabulary import (
    API_SCHEMA, CAPABILITY_BIOUSD, CAPABILITY_CELLS, CAPABILITY_ONTOLOGY,
    CAPABILITY_ROOT, CLAIMS_SINCE, METERS_PER_UNIT, PROFILE_V1, REQUIRED_PRIMVARS,
    VERSION)

#: The schema identifier of ``UsdProfilesClaimsAPI`` and the customData
#: key it reads. Neither needs the UsdProfiles plugin to author.
CLAIMS_API = "ClaimsAPI"
PROFILES_INFO = "profilesInfo"

#: Upstream's degradation classes for a capability usage.
USAGE_HARD, USAGE_SOFT, USAGE_ENHANCEMENT = "hard", "soft", "enhancement"
USAGES: Tuple[str, ...] = (USAGE_HARD, USAGE_SOFT, USAGE_ENHANCEMENT)

#: The profile a Bio-USD stage declares itself compatible with.
PROFILE = PROFILE_V1

#: The capability graph, as the ``Info.Profiles.Capabilities`` block of
#: the plugin. Each entry is one node; ``predecessors`` are the edges
#: towards the root, and a profile is a node with ``isProfile``.
CAPABILITIES: Dict[str, Dict[str, Any]] = {
    CAPABILITY_BIOUSD: {
        "name": "Bio-USD",
        "docstring": ("Bio-USD: a population of biological cells carried as a "
                      "UsdGeom.PointInstancer whose varying bio:cell:* primvars are "
                      f"per-instance biology (Bio-USD {VERSION.rsplit('.', 1)[0]}.x)."),
        "predecessors": [CAPABILITY_ROOT],
        "style": "biousd",
        "subgraph": "BioUSD",
    },
    CAPABILITY_CELLS: {
        "name": "Cell population",
        "docstring": ("A PointInstancer carries "
                      + " and ".join(f"{n} ({'int[]' if n.endswith('id') else 'token[]'})"
                                     for n in REQUIRED_PRIMVARS)
                      + " as varying primvars, one value per instance; positions in "
                      f"micrometres (metersPerUnit {METERS_PER_UNIT:g}); scales are cell "
                      "diameters applied to unit-diameter prototypes. Implied by the "
                      f"{API_SCHEMA} applied schema."),
        "predecessors": [CAPABILITY_BIOUSD],
        "style": "biousd",
        "subgraph": "BioUSD",
    },
    CAPABILITY_ONTOLOGY: {
        "name": "Cell Ontology terms",
        "docstring": ("bio:cell:ontologyTerm carries CL:/NCIT: CURIEs that a consumer "
                      "resolves against the Cell Ontology / NCI Thesaurus to join cells "
                      "to external data."),
        "predecessors": [CAPABILITY_CELLS],
        "style": "biousd",
        "subgraph": "BioUSD",
    },
    PROFILE: {
        "name": "Bio-USD profile v1",
        "docstring": ("What a consumer must support to present a Bio-USD stage "
                      "meaningfully: the cell-population carrier and ontology-term "
                      "resolution."),
        "predecessors": [CAPABILITY_CELLS, CAPABILITY_ONTOLOGY],
        "style": "profile",
        "subgraph": "BioUSD",
        "isProfile": True,
    },
}

#: What the writer claims: cells are load-bearing, terms an enhancement.
CAPABILITY_USAGES: Dict[str, str] = {
    CAPABILITY_CELLS: USAGE_HARD,
    CAPABILITY_ONTOLOGY: USAGE_ENHANCEMENT,
}


def render_plug_info_profiles() -> Dict[str, Any]:
    """The ``Info.Profiles`` block of ``plugin/plugInfo.json``.

    ``CapabilityGroups`` and ``CapabilityStyles`` only affect the graph
    renderings UsdProfiles can produce (Markdown, Mermaid, DOT).
    """
    return {
        "Capabilities": {name: dict(entry) for name, entry in CAPABILITIES.items()},
        "CapabilityGroups": {"BioUSD": {"label": "Bio-USD capabilities", "order": 10}},
        "CapabilityStyles": {
            "biousd": "fill:#e8f5e8,stroke:#2e7d32",
            "profile": "fill:#fff3cd,stroke:#856404",
        },
    }


def default_claims(*, ontology: bool = True) -> Dict[str, Any]:
    """The ``profilesInfo`` dictionary a Bio-USD writer records.

    ``ontology=False`` leaves ``biousd.ontology`` out, for a stage in
    which no cell asserts a term. Needs ``pxr`` for the typed empty
    exception list.
    """
    from pxr import Vt

    usages = dict(CAPABILITY_USAGES)
    if not ontology:
        usages.pop(CAPABILITY_ONTOLOGY)
    return {
        "capabilityUsages": usages,
        # The value per profile is the list of EXCEPTED capabilities;
        # empty means none, and it must be typed (a bare [] raises
        # "cannot infer type from empty vector/list").
        "profileCompatibility": {PROFILE: Vt.TokenArray()},
    }


def write_claims(stage, prim=None, *, ontology: bool = True) -> Optional[str]:
    """Author the claims on ``prim`` (default: the stage's default prim).

    Works on any OpenUSD version: ``AddAppliedSchema`` does not require
    the schema to be known, and the claims are customData. Returns the
    path of the prim written, or ``None`` when the stage has no default
    prim and none was given (then there is no conventional place for
    the claims, and a consumer would not look elsewhere).
    """
    if prim is None:
        prim = stage.GetDefaultPrim()
    if not prim or not prim.IsValid():
        return None
    prim.AddAppliedSchema(CLAIMS_API)
    prim.SetCustomDataByKey(PROFILES_INFO, default_claims(ontology=ontology))
    return str(prim.GetPath())


@dataclass
class ClaimsInfo:
    """What a stage claims, as read; empty when it claims nothing."""

    prim: Optional[str] = None
    #: ``ClaimsAPI`` is in the prim's authored apiSchemas.
    applied: bool = False
    #: ``profilesInfo`` is present (possibly empty) in the prim's customData.
    present: bool = False
    usages: Dict[str, str] = field(default_factory=dict)
    #: profile -> excepted capabilities
    profiles: Dict[str, List[str]] = field(default_factory=dict)


def read_claims(stage) -> ClaimsInfo:
    """The claims on the default prim, read as metadata (no UsdProfiles needed).

    ``GetAppliedSchemas()`` drops a schema the registry does not know,
    which is every schema on a 25.x wheel here, so the applied list is
    read from ``GetPrimTypeInfo()``, which keeps what was authored.
    """
    prim = stage.GetDefaultPrim()
    if not prim or not prim.IsValid():
        return ClaimsInfo()
    applied = [str(s) for s in prim.GetPrimTypeInfo().GetAppliedAPISchemas()]
    info = ClaimsInfo(prim=str(prim.GetPath()), applied=CLAIMS_API in applied)
    data = prim.GetCustomDataByKey(PROFILES_INFO)
    if data is None:
        return info
    info.present = True
    data = dict(data)
    info.usages = {str(k): str(v) for k, v in dict(data.get("capabilityUsages") or {}).items()}
    info.profiles = {str(k): [str(x) for x in (v or [])]
                     for k, v in dict(data.get("profileCompatibility") or {}).items()}
    return info


def _written_to_at_least(version: Any, floor: Tuple[int, int]) -> bool:
    try:
        parts = [int(p) for p in str(version).split(".")[:2]]
        return tuple(parts + [0] * (2 - len(parts))) >= floor
    except ValueError:
        return False


def claims_issues(claims: ClaimsInfo, *, version: Any = VERSION,
                  populations: int = 0, ontology_asserted: bool = False
                  ) -> List[Tuple[str, str]]:
    """Warnings about the claims, as ``(code, message)`` pairs.

    ``version`` is the stage's ``bio:version``: a stage older than
    :data:`~.vocabulary.CLAIMS_SINCE` is not told it lacks claims, since
    the convention did not exist when it was written. ``populations``
    and ``ontology_asserted`` describe the stage's content, which the
    claims must agree with. Every finding is a warning: a claim is a
    statement about the file, and a wrong one does not make the cells
    wrong.
    """
    out: List[Tuple[str, str]] = []
    where = claims.prim or "(no default prim)"
    if not claims.applied and not claims.present:
        if _written_to_at_least(version, CLAIMS_SINCE):
            out.append(("profile.absent",
                        f"{where}: no UsdProfiles claims (apiSchemas {CLAIMS_API} with "
                        f"customData {PROFILES_INFO}); a {CLAIMS_SINCE[0]}.{CLAIMS_SINCE[1]}+ "
                        "writer records them"))
        return out
    if not claims.present:
        out.append(("profile.absent", f"{where}: {CLAIMS_API} is applied but no {PROFILES_INFO} "
                                       "customData is authored"))
        return out
    if not claims.applied:
        out.append(("profile.inconsistent", f"{where}: {PROFILES_INFO} is authored without "
                                             f"{CLAIMS_API} in apiSchemas; UsdProfiles reads "
                                             "it only through the applied schema"))
    known = set(CAPABILITIES)
    for cap, usage in claims.usages.items():
        if cap.startswith(CAPABILITY_BIOUSD) and cap not in known:
            out.append(("profile.unknown", f"{where}: capability {cap!r} is not one Bio-USD "
                                            f"{VERSION} declares ({sorted(known)})"))
        if usage not in USAGES:
            out.append(("profile.usage", f"{where}: {cap} usage {usage!r} is not one of {USAGES}"))
    for profile, excepted in claims.profiles.items():
        if profile.startswith(CAPABILITY_BIOUSD) and profile != PROFILE:
            out.append(("profile.unknown", f"{where}: profile {profile!r} is not {PROFILE}"))
        for cap in excepted:
            if cap.startswith(CAPABILITY_BIOUSD) and cap not in known:
                out.append(("profile.unknown", f"{where}: excepted capability {cap!r} is not "
                                                f"one Bio-USD {VERSION} declares"))
    if populations and CAPABILITY_CELLS not in claims.usages:
        out.append(("profile.inconsistent", f"{where}: the stage holds {populations} cell "
                                             f"population(s) but claims no {CAPABILITY_CELLS} usage"))
    elif CAPABILITY_CELLS in claims.usages and claims.usages[CAPABILITY_CELLS] != USAGE_HARD:
        out.append(("profile.inconsistent", f"{where}: {CAPABILITY_CELLS} is claimed "
                                             f"{claims.usages[CAPABILITY_CELLS]!r}; a consumer "
                                             "without it cannot present the cells, so it is "
                                             f"{USAGE_HARD!r}"))
    if ontology_asserted and CAPABILITY_ONTOLOGY not in claims.usages:
        out.append(("profile.inconsistent", f"{where}: cells assert ontology terms but the "
                                             f"claims name no {CAPABILITY_ONTOLOGY} usage"))
    if PROFILE not in claims.profiles:
        out.append(("profile.inconsistent", f"{where}: profileCompatibility does not declare "
                                             f"{PROFILE}"))
    return out


def check_claims(stage) -> List[Tuple[str, str]]:
    """The claim warnings for an open stage, exactly as the validator reports them.

    Returns ``(code, message)`` pairs; an empty list means the claims are
    present and consistent with the stage. Reads the stage's populations
    for the consistency part, through :mod:`.read` (imported here, not at
    module level, because :mod:`.read` stores :class:`ClaimsInfo` and so
    imports this module).
    """
    from .read import read_stage_info

    info = read_stage_info(stage)
    return claims_issues(
        info.claims or ClaimsInfo(), version=info.layer_data.get("bio:version"),
        populations=len(info.populations),
        ontology_asserted=any(any(str(v) for v in t.primvars.get("bio:cell:ontologyTerm", []))
                              for t in info.populations))
