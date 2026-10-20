"""Every library entity builds, and the geometry it builds is sound.

The defects in these shapes were all found by rendering them and
looking: a fibroblast nucleus 37% outside its own cell, an endothelial
nucleus thicker than the cell it sits in, flagella that walked past the
grid edge and came back as a fragment floating beside the bacterium, a
plasma cell's ER as a solid ball hiding the nucleus, NaN quaternions
that made instances vanish. Every one of them passed the tests that
existed at the time.

Looking does not scale and does not repeat. These tests are the part of
looking that can be written down: for all 56 entities at once, at their
own library scales, the invariants that each of those bugs violated.

They are NOT a substitute for looking. Nothing here can tell that a
macrophage reads as a jack rather than a cell, or that an eosinophil is
not obviously granulated -- those need eyes. What they can do is stop a
fixed bug coming back silently, which is what happened to most of them
the first time.
"""
from __future__ import annotations

import collections
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("skimage")

from cognisom.visualization import cell_detail as cd                # noqa: E402
from cognisom.visualization.cell_profiles import (                   # noqa: E402
    library_detail, markers_of, profile_for)
from cognisom.visualization.morphology import (                      # noqa: E402
    build_masks, build_parts)
from cognisom.visualization.morphology_usd import CELLULAR_TYPES     # noqa: E402

#: Coarse, because these are structural invariants rather than fidelity
#: checks, and 56 entities have to fit in a test run.
GRID = 32


@pytest.fixture(scope="module")
def entities():
    with contextlib.redirect_stdout(io.StringIO()):
        from cognisom.library.store import EntityStore
        store = EntityStore()
    rows = [tuple(r) for r in store._conn.execute(
        "SELECT entity_type, name, data_json FROM entities")]
    out = []
    for etype, name, blob in sorted(rows):
        if etype not in CELLULAR_TYPES or etype == "tissue_type":
            continue
        data = json.loads(blob or "{}")
        if not data.get("mesh_type") or data.get("scale_um") is None:
            continue
        out.append({"name": name, "type": etype,
                    "mesh_type": data["mesh_type"],
                    "scale_um": float(data["scale_um"]),
                    "raw": blob, "entity": store.find_entity_by_name(name)})
    assert len(out) >= 50, f"only {len(out)} entities found"
    return out


@pytest.fixture(scope="module")
def built(entities):
    """Every entity's geometry, built once."""
    out = {}
    for e in entities:
        parts, fell_back = build_parts(e["mesh_type"], scale_um=e["scale_um"],
                                       seed=0, grid=GRID)
        masks, _ = build_masks(e["mesh_type"], seed=0, grid=GRID)
        out[e["name"]] = (parts, masks, fell_back, e)
    return out


def _boundary_edges(triangles):
    counts = collections.Counter()
    for tri in triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            counts[(min(a, b), max(a, b))] += 1
    return sum(1 for v in counts.values() if v == 1)


# ── it builds at all ─────────────────────────────────────────────────

def test_every_entity_produces_geometry(built):
    empty = [n for n, (parts, _m, _f, _e) in built.items()
             if not parts or "body" not in parts]
    assert empty == [], empty
    for name, (parts, _m, _f, _e) in built.items():
        for part, (v, f) in parts.items():
            assert len(v) > 0 and len(f) > 0, f"{name}/{part} is empty"


def test_no_geometry_is_degenerate(built):
    """A NaN renders as nothing, so the prim silently disappears rather
    than looking wrong -- which is how a NaN quaternion survived in the
    instancing code."""
    for name, (parts, _m, _f, _e) in built.items():
        for part, (v, f) in parts.items():
            assert np.isfinite(v).all(), f"{name}/{part} has non-finite vertices"
            assert f.max() < len(v), f"{name}/{part} indexes past its vertices"
            assert f.min() >= 0
            extent = v.max(axis=0) - v.min(axis=0)
            assert (extent > 0).all(), f"{name}/{part} is flat in some axis"


def test_a_fallback_is_declared_not_silent(built):
    """An unknown mesh_type becomes a sphere. A sphere wearing the label
    'neutrophil' while nothing says so is the failure the flag exists to
    prevent.

    The positive case has to be asserted too. An earlier version of this
    test only checked that anything which fell back had mesh_type
    "mesh" -- and no entity in this fixture does, so it asserted nothing
    at all and passed happily when the flag was hard-wired to False.
    """
    # The flag fires when it should.
    _parts, fell_back = build_parts("no_such_mesh_type_exists",
                                    scale_um=10.0, seed=0, grid=16)
    assert fell_back is True, "an unknown mesh_type did not report a fallback"

    # And not when it should not: every entity here has a real builder.
    silent = {name: e["mesh_type"]
              for name, (_p, _m, fb, e) in built.items() if fb}
    assert silent == {}, silent


# ── the bugs that were found by looking ──────────────────────────────

def test_the_body_is_a_closed_surface(built):
    """A hole in the membrane shows as a cell you can see inside, and it
    means the shape reached the edge of the grid it was built on."""
    open_bodies = {}
    for name, (parts, _m, _f, _e) in built.items():
        v, f = parts["body"]
        boundary = _boundary_edges(f)
        if boundary > len(f) * 0.02:
            open_bodies[name] = boundary
    assert open_bodies == {}, open_bodies


def test_nothing_is_clipped_by_the_grid(built):
    """Flagella built as a random walk reached 1.29 on a grid ending at
    1.0 and rendered as a fragment floating beside the cell."""
    clipped = {}
    for name, (_p, masks, _f, _e) in built.items():
        for part, mask in masks.items():
            for axis in range(3):
                if mask.take(0, axis=axis).any() or \
                        mask.take(mask.shape[axis] - 1, axis=axis).any():
                    clipped.setdefault(name, []).append(f"{part}:{'xyz'[axis]}")
    assert clipped == {}, clipped


def test_no_part_escapes_the_body(built):
    """A fibroblast nucleus was 37% outside its own cell and an
    endothelial nucleus was thicker than the cell it sits in. Both
    rendered as a nucleus bursting through the membrane."""
    escaping = {}
    for name, (parts, masks, _f, e) in built.items():
        body_mask = masks.get("body")
        body_mesh = parts.get("body")
        if body_mask is None or body_mesh is None:
            continue
        resolution, offset = cd.mask_frame(body_mask, body_mesh[0],
                                           e["scale_um"])
        for part, (v, _f2) in parts.items():
            # flagella and spores are external structures by definition
            if part in ("body", "flagella", "spore"):
                continue
            idx = np.rint((v - offset) / resolution).astype(int)
            inside = ((idx >= 0) & (idx < np.array(body_mask.shape))).all(axis=1)
            hit = np.zeros(len(idx), bool)
            good = idx[inside]
            hit[inside] = body_mask[good[:, 0], good[:, 1], good[:, 2]]
            out = float((~hit).mean())
            if out > 0.05:
                escaping[f"{name}/{part}"] = round(out, 3)
    assert escaping == {}, escaping


def test_the_geometry_is_at_the_declared_scale(built):
    """scale_um is the width of the whole shape. A bug in the builder
    once doubled it, so a '20 um' cell measured 35 um across."""
    wrong = {}
    for name, (parts, _m, _f, e) in built.items():
        allv = np.vstack([v for v, _f2 in parts.values()])
        width = float(np.ptp(allv, axis=0).max())
        want = e["scale_um"]
        if not 0.45 < width / want < 1.35:
            wrong[name] = (round(width, 2), want)
    assert wrong == {}, wrong


# ── the detail each entity actually carries ──────────────────────────

def test_every_entity_builds_its_detail(entities):
    """The profile has to survive each entity's real markers and scale --
    a marker with no recorded compartment, or a cell whose cytoplasm is
    too thin to hold an organelle, must not raise."""
    for e in entities:
        profile = profile_for(e["name"], e["mesh_type"],
                              markers_of(e["raw"]), scale_um=e["scale_um"],
                              library=library_detail(e["entity"]))
        assert profile.rationale
        assert profile.microvilli_count >= 0
        for count in profile.membrane_stud_groups.values():
            assert count >= 0


def test_lymphocytes_get_microvilli_and_phagocytes_get_ruffles(entities):
    """Two surface grammars, from one citation (PMID 15130947), and the
    only thing that lets a viewer tell a lymphocyte from a monocyte at a
    glance. Mixing them throws that away."""
    lymph = phago = 0
    for e in entities:
        profile = profile_for(e["name"], e["mesh_type"],
                              markers_of(e["raw"]), scale_um=e["scale_um"],
                              library=library_detail(e["entity"]))
        if e["mesh_type"] in ("smooth_sphere", "granular_sphere"):
            assert profile.microvilli_count > 0, e["name"]
            assert profile.ruffle_amplitude_um == 0, e["name"]
            lymph += 1
        elif e["mesh_type"] in ("ruffled_pseudopod", "irregular_ruffled"):
            assert profile.microvilli_count == 0, e["name"]
            assert profile.ruffle_amplitude_um > 0, e["name"]
            phago += 1
    assert lymph >= 15 and phago >= 2, (lymph, phago)


def test_the_microvillus_count_follows_each_cells_own_size(entities):
    """A measured density times this cell's area. The fixed count it
    replaced was wrong at every diameter but one."""
    seen = {}
    for e in entities:
        profile = profile_for(e["name"], e["mesh_type"],
                              markers_of(e["raw"]), scale_um=e["scale_um"],
                              library=library_detail(e["entity"]))
        if profile.microvilli_count:
            seen[e["scale_um"]] = profile.microvilli_count
    assert len(seen) > 1, "every lymphocyte is the same size; nothing proven"
    sizes = sorted(seen)
    assert seen[sizes[0]] < seen[sizes[-1]]
    # and the count is the density times the area, not a stored number
    for diameter, count in seen.items():
        assert count == pytest.approx(cd.microvillus_count(diameter), rel=0.02)


def test_instanced_detail_lands_on_or_in_the_cell(entities):
    """Microvilli outward from the membrane, organelles inside it. A
    microvillus pointing inward is invisible rather than wrong-looking,
    which is the worse failure."""
    # Sample the cells that actually carry microvilli. Taking the first
    # twelve entities by name took mostly bacteria and epithelium, so the
    # test checked two cells and reported success.
    candidates = [e for e in entities
                  if e["mesh_type"] in ("smooth_sphere", "granular_sphere")]
    assert len(candidates) >= 10, len(candidates)

    checked = 0
    for e in candidates[:8]:
        parts, _fb = build_parts(e["mesh_type"], scale_um=e["scale_um"],
                                 seed=0, grid=GRID)
        masks, _ = build_masks(e["mesh_type"], seed=0, grid=GRID)
        profile = profile_for(e["name"], e["mesh_type"],
                              markers_of(e["raw"]), scale_um=e["scale_um"],
                              library=library_detail(e["entity"]))
        if not profile.microvilli_count:
            continue
        bv, bf = parts["body"]
        inst = cd.projections(bv, bf, count=min(profile.microvilli_count, 150),
                              length_um=profile.microvillus_length_um,
                              radius_um=profile.microvillus_radius_um, seed=1)
        assert len(inst) > 0
        assert np.isfinite(inst.orientations).all(), e["name"]
        centre = bv.mean(axis=0)
        axes = _instance_axes(inst.orientations)
        radial = inst.positions - centre
        radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-9)
        assert (np.einsum("ij,ij->i", axes, radial) > 0).all(), e["name"]
        checked += 1
    assert checked >= 5


def _instance_axes(quaternions: np.ndarray) -> np.ndarray:
    w, x, y, z = (np.asarray(quaternions)[:, i] for i in range(4))
    return np.stack([2 * (x * z + w * y),
                     2 * (y * z - w * x),
                     1 - 2 * (x * x + y * y)], axis=1)
