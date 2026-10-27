"""UsdProfiles claims: written by hand on any wheel, read by 26.08+ with the plugin.

Measured on usd-core 25.11 (no `pxr.UsdProfiles` at all) and 26.11: the
claims are prim metadata, `AddAppliedSchema("ClaimsAPI")` plus a
`profilesInfo` customData dictionary, so a 25.11 writer authors the
identical file a 26.11 `ClaimsAPI.Apply` would, and a 26.11 reader with
the plugin registered answers `IsCompatibleWith(biousd.profile_v1) ->
ValidPath`. The exception list must be a typed empty `Vt.TokenArray()`;
a bare `[]` cannot be typed and raised on the first run.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("pxr")

from cognisom.biousd import standard as biousd                          # noqa: E402
from cognisom.biousd.standard import profiles                           # noqa: E402
from cognisom.biousd.standard.profiles import (                         # noqa: E402
    CAPABILITIES, CAPABILITY_CELLS, CAPABILITY_ONTOLOGY, CLAIMS_API, PROFILE,
    ClaimsInfo, check_claims, claims_issues, read_claims, write_claims)

PKG = pathlib.Path(biousd.__file__).resolve().parent
REPO = PKG.parents[2]
EXAMPLE = PKG / "examples" / "minimal_200_cells.usda"


def _cells(n=20, terms=True):
    rng = np.random.default_rng(1)
    return [dict(position=tuple(map(float, p)), cell_id=i, cell_type="cancer",
                 ontology_term="NCIT:C2919" if terms else "", scale_um=10.0)
            for i, p in enumerate(rng.uniform(-30, 30, (n, 3)))]


def _usd_version():
    from pxr import Usd
    return tuple(Usd.GetVersion())


# ── the plugin's capability graph ───────────────────────────────────


def test_the_plugin_carries_the_capability_graph_under_the_usd_root():
    info = json.loads((PKG / "plugin" / "plugInfo.json").read_text())
    block = info["Plugins"][0]["Info"]["Profiles"]
    assert block == profiles.render_plug_info_profiles()
    caps = block["Capabilities"]
    assert set(caps) == set(CAPABILITIES) == {"biousd", "biousd.cells", "biousd.ontology",
                                              "biousd.profile_v1"}
    assert caps["biousd"]["predecessors"] == ["usd"]
    assert caps[PROFILE]["isProfile"] is True
    assert set(caps[PROFILE]["predecessors"]) == {CAPABILITY_CELLS, CAPABILITY_ONTOLOGY}
    # every edge leads back to the root
    for name, entry in caps.items():
        for pred in entry["predecessors"]:
            assert pred == "usd" or pred in caps, (name, pred)


# ── writing claims by hand ──────────────────────────────────────────


def test_write_claims_authors_exactly_the_hand_written_structure(tmp_path):
    from pxr import Usd, Vt

    path = str(tmp_path / "c.usda")
    report = biousd.write_cells(path, _cells())
    assert report.claims_prim == "/World"
    stage = Usd.Stage.Open(path)
    world = stage.GetDefaultPrim()
    assert CLAIMS_API in [str(s) for s in world.GetPrimTypeInfo().GetAppliedAPISchemas()]
    data = world.GetCustomDataByKey(profiles.PROFILES_INFO)
    assert dict(data["capabilityUsages"]) == {CAPABILITY_CELLS: "hard",
                                              CAPABILITY_ONTOLOGY: "enhancement"}
    excepted = data["profileCompatibility"][PROFILE]
    assert isinstance(excepted, Vt.TokenArray) and len(excepted) == 0
    text = pathlib.Path(path).read_text()
    assert 'prepend apiSchemas = ["ClaimsAPI"]' in text
    assert f'token[] "{PROFILE}" = []' in text

    claims = read_claims(stage)
    assert claims == ClaimsInfo(prim="/World", applied=True, present=True,
                                usages={CAPABILITY_CELLS: "hard", CAPABILITY_ONTOLOGY: "enhancement"},
                                profiles={PROFILE: []})
    assert check_claims(stage) == []


def test_write_claims_needs_a_default_prim_or_one_named():
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    assert write_claims(stage) is None
    prim = UsdGeom.Xform.Define(stage, "/Scene").GetPrim()
    assert write_claims(stage, prim) == "/Scene"
    assert read_claims(stage).present is False            # not the default prim
    stage.SetDefaultPrim(prim)
    assert read_claims(stage).applied and read_claims(stage).present


def test_the_ontology_usage_follows_the_terms(tmp_path):
    from pxr import Usd

    path = str(tmp_path / "noterms.usda")
    biousd.write_cells(path, _cells(terms=False))
    claims = read_claims(Usd.Stage.Open(path))
    assert CAPABILITY_ONTOLOGY not in claims.usages and claims.usages[CAPABILITY_CELLS] == "hard"
    report = biousd.validate(path)
    assert report.ok
    assert not [c for c in report.codes if c.startswith("profile.")], report.describe()


def test_the_examples_claims_are_consistent():
    from pxr import Usd

    for name in ("minimal_200_cells.usda", "tissue_slab_10k_cells.usdc"):
        assert check_claims(Usd.Stage.Open(str(PKG / "examples" / name))) == []


# ── the consistency check ───────────────────────────────────────────


def _codes(claims, **kw):
    kw.setdefault("version", "0.2.0")
    kw.setdefault("populations", 1)
    kw.setdefault("ontology_asserted", True)
    return [code for code, _ in claims_issues(claims, **kw)]


def test_claims_issues_names_each_inconsistency():
    good = ClaimsInfo(prim="/World", applied=True, present=True,
                      usages={CAPABILITY_CELLS: "hard", CAPABILITY_ONTOLOGY: "enhancement"},
                      profiles={PROFILE: []})
    assert _codes(good) == []
    # absent: a warning for a 0.2 stage, nothing for an older one
    assert _codes(ClaimsInfo()) == ["profile.absent"]
    assert _codes(ClaimsInfo(), version="0.1.2") == []
    assert _codes(ClaimsInfo(), version=None) == []
    applied_only = ClaimsInfo(prim="/World", applied=True, present=False)
    assert _codes(applied_only) == ["profile.absent"]
    # present but the schema not applied: UsdProfiles would not read it
    unapplied = ClaimsInfo(prim="/World", applied=False, present=True, usages=good.usages,
                           profiles=good.profiles)
    assert _codes(unapplied) == ["profile.inconsistent"]
    # names we do not declare, and a usage strength that does not exist
    odd = ClaimsInfo(prim="/World", applied=True, present=True,
                     usages={CAPABILITY_CELLS: "hard", "biousd.flux": "soft",
                             CAPABILITY_ONTOLOGY: "mandatory"},
                     profiles={PROFILE: [], "biousd.profile_v9": ["biousd.cells"]})
    assert sorted(_codes(odd)) == ["profile.unknown", "profile.unknown", "profile.usage"]
    # content the claims do not cover
    soft = ClaimsInfo(prim="/World", applied=True, present=True,
                      usages={CAPABILITY_CELLS: "soft"}, profiles={})
    assert _codes(soft) == ["profile.inconsistent"] * 3     # soft cells, no ontology, no profile
    none_claimed = ClaimsInfo(prim="/World", applied=True, present=True, usages={},
                              profiles={PROFILE: []})
    assert _codes(none_claimed, ontology_asserted=False) == ["profile.inconsistent"]
    assert _codes(none_claimed, populations=0, ontology_asserted=False) == []
    # a foreign capability is not ours to judge
    foreign = ClaimsInfo(prim="/World", applied=True, present=True,
                         usages={**good.usages, "usd.core.colorspace": "soft"},
                         profiles=good.profiles)
    assert _codes(foreign) == []


def test_the_validator_reports_the_claim_warnings_on_the_default_prim(tmp_path):
    from pxr import Usd

    path = str(tmp_path / "soft.usda")
    biousd.write_cells(path, _cells())
    stage = Usd.Stage.Open(path)
    world = stage.GetDefaultPrim()
    world.SetCustomDataByKey(profiles.PROFILES_INFO,
                             {"capabilityUsages": {CAPABILITY_CELLS: "soft"},
                              "profileCompatibility": {}})
    stage.GetRootLayer().Save()
    report = biousd.validate(path)
    assert report.ok
    assert report.codes == ["profile.inconsistent"] * 3
    assert all(i.site == "/World" for i in report.warnings)
    assert check_claims(Usd.Stage.Open(path)) == [(i.code, i.message) for i in report.warnings]


# ── a 26.08+ reader ─────────────────────────────────────────────────


def _reader_in_subprocess(path: str, *, plugin_route: str) -> dict:
    """Open ``path`` on this interpreter's pxr with UsdProfiles, the plugin
    registered by importing the package or by PXR_PLUGINPATH_NAME alone."""
    env = dict(os.environ)
    env.pop("PXR_PLUGINPATH_NAME", None)
    if plugin_route == "import":
        env["BIOUSD_NO_PLUGIN"] = ""
    else:
        env["BIOUSD_NO_PLUGIN"] = "1"
        env["PXR_PLUGINPATH_NAME"] = biousd.plugin_path()
    code = f"""
import json
try:
    from pxr import UsdProfiles
except ImportError:
    print(json.dumps(dict(skip="no pxr.UsdProfiles")))
    raise SystemExit(0)
import cognisom.biousd.standard as b
from pxr import Usd
PR = UsdProfiles.ProfileRegistry
stage = Usd.Stage.Open({path!r})
api = UsdProfiles.ClaimsAPI(stage.GetDefaultPrim())
status, results = api.IsCompatibleWith({PROFILE!r})
print(json.dumps(dict(
    profiles=sorted(PR.GetAllProfiles()), status=str(status),
    results=sorted((r.capability, str(r.status)) for r in results),
    usages=dict(api.GetCapabilityUsages()), compatible=list(api.GetCompatibleProfiles()),
    implied=sorted(PR.GetSchemaImpliedCapabilities(Usd.SchemaRegistry.GetTypeFromName("BioCellAPI"))),
    covers=str(PR.CoversCapabilities({PROFILE!r}, list(api.GetCapabilityUsages()))[0]),
    populated=dict(api.PopulateCapabilityUsages()))))
"""
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(REPO), env=env)
    assert run.returncode == 0, run.stdout + run.stderr
    return json.loads(run.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("plugin_route", ["import", "env"])
def test_a_26_reader_with_the_plugin_accepts_the_written_claims(tmp_path, plugin_route):
    """Skips below 26.08, where pxr.UsdProfiles does not exist."""
    out = _reader_in_subprocess(str(EXAMPLE), plugin_route=plugin_route)
    if "skip" in out:
        pytest.skip(f"{out['skip']} on usd-core {'.'.join(map(str, _usd_version()))}; "
                    "UsdProfiles shipped in OpenUSD 26.08")
    assert out["profiles"] == [PROFILE]
    assert out["status"].endswith("ValidPath"), out
    assert out["results"] == [[CAPABILITY_CELLS, "ValidPath"], [CAPABILITY_ONTOLOGY, "ValidPath"]]
    assert out["usages"] == {CAPABILITY_CELLS: "hard", CAPABILITY_ONTOLOGY: "enhancement"}
    assert out["compatible"] == [PROFILE]
    assert out["implied"] == [CAPABILITY_CELLS]
    assert out["covers"].endswith("ValidPath")
    # PopulateCapabilityUsages discovers biousd.cells from the applied
    # BioCellAPI and keeps what the writer said
    assert out["populated"] == out["usages"]
