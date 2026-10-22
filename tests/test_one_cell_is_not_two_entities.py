"""Some library entities describe the same cell twice, and that is known.

Two kinds of duplication exist, and only one of them is handled.

**Same name, two types.** "NK cell" is registered as both a ``cell_type``
and an ``immune_cell``. The codebase knows: ``tissue_config`` resolves it
as the pair ``("NK cell", "immune_cell")``, ``parameter_resolver`` calls it
"the only such collision in the shipped library", and ``morphology_usd``
carries a prim path rather than recomputing one because recomputing "sent
the second one's transform to the first one's prim". That is a real bug
this duplication already caused once.

**Different names, indistinguishable geometry.** Three pairs share a
mesh_type, a scale_um and a colour while one name contains the other, so
nothing that resolves by name catches them. Only one of the three is
actually the same cell twice:

    Regulatory T cell / Regulatory T cell (Treg)      same cell
    Macrophage / Macrophage M1                        M1 is a polarisation
    Endothelial cell / Postcapillary venule e.c.      a distinct subtype,
                                                      and where diapedesis
                                                      happens

The other two are different cells the library cannot draw differently --
the same defect as the nineteen entities that all resolve to a smooth
sphere. Pinning all three here keeps the distinction visible: the first is
a duplicate to collapse by ontology term, the other two are shapes to
earn.

These tests do NOT delete anything. Deleting would be destructive, one
relationship points at the group, and the collision is load-bearing in at
least three modules. What they do is stop the set growing silently: a third
collision, or a second same-cell-two-names pair, fails here rather than
surfacing later as a transform written to the wrong prim.
"""

from __future__ import annotations

import collections
import contextlib
import io
import json

import pytest

from cognisom.visualization.morphology_usd import CELLULAR_TYPES

#: The one name registered under two entity types, and the type that wins.
#: ``tissue_config`` resolves nk_cell to the immune_cell row.
KNOWN_NAME_COLLISIONS = {"NK cell"}

#: Pairs that one name contains the other AND whose geometry is identical.
#: Only the Treg pair is one cell twice; the other two are distinct cells
#: the library draws identically. A fourth has to be declared deliberately.
KNOWN_INDISTINGUISHABLE_PAIRS = {
    ("Regulatory T cell", "Regulatory T cell (Treg)"),
    ("Macrophage", "Macrophage M1"),
    ("Endothelial cell", "Postcapillary venule endothelial cell"),
}


@pytest.fixture(scope="module")
def rows():
    with contextlib.redirect_stdout(io.StringIO()):
        from cognisom.library.store import EntityStore
        store = EntityStore()
    out = []
    for etype, name, blob in store._conn.execute(
            "SELECT entity_type, name, data_json FROM entities"):
        if etype not in CELLULAR_TYPES or etype == "tissue_type":
            continue
        data = json.loads(blob or "{}")
        if not data.get("mesh_type") or data.get("scale_um") is None:
            continue
        out.append((etype, name, data))
    assert len(out) >= 50, len(out)
    return out


def test_the_set_of_name_collisions_has_not_grown(rows):
    """One name under two entity types is a trap; two would be a pattern."""
    by_name = collections.defaultdict(set)
    for etype, name, _ in rows:
        by_name[name].add(etype)
    found = {n for n, types in by_name.items() if len(types) > 1}
    assert found == KNOWN_NAME_COLLISIONS, (
        f"name collisions changed: {sorted(found)}. Resolving a library "
        "entity by name alone returns an arbitrary one of these, which is "
        "how a transform was once written to the wrong prim. Either give "
        "the new one a distinct name or add it here deliberately.")


def test_no_new_pair_is_indistinguishable_by_geometry(rows):
    """Identical geometry plus a containing name is the signature."""
    names = [n for _t, n, _d in rows]
    geom = {n: (d.get("mesh_type"), d.get("scale_um"), tuple(d.get("color_rgb") or ()))
            for _t, n, d in rows}
    found = set()
    for a in names:
        for b in names:
            if a >= b:
                continue
            short, long_ = (a, b) if len(a) < len(b) else (b, a)
            if short.lower() in long_.lower() and geom[a] == geom[b]:
                found.add((short, long_))
    assert found == KNOWN_INDISTINGUISHABLE_PAIRS, (
        f"indistinguishable pairs changed: {sorted(found)}. These do not "
        "collide by name, so nothing catches them. Decide which kind each "
        "new one is: the same cell twice, to collapse by ontology term, or "
        "two cells the library cannot yet draw apart.")


def test_the_collision_resolves_to_the_immune_cell_row(rows):
    """Which one wins is a decision, so it is written down."""
    from cognisom.core.tissue_config import ARCHETYPE_ENTITY

    name, etype = ARCHETYPE_ENTITY["nk_cell"]
    assert name in KNOWN_NAME_COLLISIONS
    assert etype == "immune_cell", (
        "nk_cell resolves to the cell_type row; the immune_cell row is the "
        "one carrying the immune markers")
    assert any(t == etype and n == name for t, n, _ in rows)
