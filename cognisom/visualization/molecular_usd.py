"""Library entities as real molecular geometry in OpenUSD.

The entity library carries 24 validated small-molecule structures -- every
one of them a SMILES string whose recomputed formula matches its recorded
one. In the USD track every one of them was a single sphere:
``prototype_library`` gives glucose ``geometry_type="sphere"`` and oxygen
the same, so a C6H12O6 ring and an O=O diatomic rendered identically, and
a drug with five rings and a thiohydantoin rendered as a ball.

This module turns those structures into geometry: one sphere per atom at
its real conformer coordinate, two half-length cylinders per bond so each
half carries its own atom's colour, per-element materials, and the
provenance of every number in ``customData`` so a stage can be audited
without the library beside it.

WHAT IS AND IS NOT MEASURED HERE
--------------------------------
The *topology* is measured: it comes from the library's SMILES, which was
checked against formula and molecular weight (see
``cognisom.library.structures``, written after a SMILES recorded as
Enzalutamide turned out to be a different compound with 40 atoms instead
of 32).

The *coordinates* are computed, not measured. ETKDG distance geometry
followed by MMFF94 minimisation gives one plausible low-energy conformer,
seeded so it is reproducible. It is not a crystal structure, and for a
flexible molecule it is not the bound pose either -- docetaxel has nine
rotatable bonds and its conformation in a tubulin site is an experimental
question this does not answer. Anything that depends on the actual
geometry of a bound ligand needs a structure, not this.

The *radii* are rdkit's periodic-table van der Waals values, scaled by a
rendering convention rather than a physical one: a ball-and-stick ball is
deliberately smaller than the atom, or the bonds would be invisible
inside it. :data:`BALL_SCALE` is that convention, not a measurement.

Units: the stage is authored in angstroms with ``metersPerUnit = 1e-10``.
The cell-scale exporters in this package use ``1e-6`` (micrometres), which
is four orders of magnitude away, so a molecule referenced into a cell
scene must be scaled by the reference and not simply laid beside it.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

#: Angstroms, so a bond length reads as ~1.5 rather than ~1.5e-10.
METERS_PER_UNIT = 1e-10

#: Ball radius as a fraction of the van der Waals radius. A rendering
#: convention: at 1.0 this is a space-filling model and the bonds vanish
#: inside the atoms. 0.25 is the usual ball-and-stick proportion.
BALL_SCALE = 0.25

#: Stick radius in angstroms. Thinner than every ball at BALL_SCALE, so a
#: bond reads as a bond and not as a neck between two spheres.
STICK_RADIUS = 0.15

#: Jmol's element colours, the palette most chemistry viewers use, so a
#: structure here looks like the same structure in PyMOL or usdview. Carbon
#: is the convention's grey rather than black: black reads as a hole
#: against the dark viewport most USD tools open with.
#: Source: Jmol/JSmol ``colors.js``, the CPK-derived default scheme.
CPK_COLORS: Dict[str, Tuple[float, float, float]] = {
    "H": (1.000, 1.000, 1.000),
    "C": (0.565, 0.565, 0.565),
    "N": (0.188, 0.314, 0.973),
    "O": (1.000, 0.051, 0.051),
    "F": (0.565, 0.878, 0.314),
    "Na": (0.671, 0.361, 0.949),
    "Mg": (0.541, 1.000, 0.000),
    "P": (1.000, 0.502, 0.000),
    "S": (1.000, 1.000, 0.188),
    "Cl": (0.122, 0.941, 0.122),
    "K": (0.561, 0.251, 0.831),
    "Ca": (0.239, 1.000, 0.000),
    "Fe": (0.878, 0.400, 0.200),
    "Zn": (0.490, 0.502, 0.690),
    "Br": (0.651, 0.161, 0.161),
    "I": (0.580, 0.000, 0.580),
}

#: Anything outside CPK_COLORS. Jmol's own fallback for an unknown element.
UNKNOWN_COLOR = (1.000, 0.078, 0.576)


@dataclass
class MoleculeGeometry:
    """What one molecule cost and what it is made of."""

    name: str
    smiles: str
    formula: str
    atoms: int
    bonds: int
    prims: int
    """Spheres plus cylinders actually authored: ``atoms + 2 * bonds``."""
    radius_a: float
    """Half the longest extent, in angstroms -- what it needs for layout."""
    elements: Dict[str, int]
    prim_path: str
    """Where it landed. Carried rather than recomputed from the name: two
    entities can share a prim name, and recomputing sends the second
    one's transform to the first one's prim."""
    embedded: bool
    """False when distance geometry could not place the atoms. The prim is
    still written, with its topology, and every atom sits at the origin;
    the stage records this so a viewer is not left guessing why a molecule
    looks like a single blob."""


def _element_radius(symbol: str) -> float:
    """Van der Waals radius in angstroms, from rdkit's periodic table.

    Not a number of ours: ``GetRvdw`` reads the table rdkit ships, so a
    radius here is the same radius every other rdkit tool uses.
    """
    from rdkit.Chem import GetPeriodicTable
    table = GetPeriodicTable()
    try:
        return float(table.GetRvdw(symbol))
    except Exception:
        return 1.7                       # carbon's, the commonest fallback


def _color_for(symbol: str) -> Tuple[float, float, float]:
    return CPK_COLORS.get(symbol, UNKNOWN_COLOR)


def _quat_z_to(direction) -> Tuple[float, Tuple[float, float, float]]:
    """Quaternion rotating +Z onto ``direction``.

    USD's implicit cylinder lies along an axis attribute -- Z here -- so a
    bond is drawn by rotating that axis onto the vector between its atoms.
    The two degenerate cases have to be handled explicitly: parallel needs
    no rotation, and antiparallel has no unique rotation axis, so any
    perpendicular one will do and the cross product supplies none.
    """
    import numpy as np

    v = np.asarray(direction, dtype=float)
    length = float(np.linalg.norm(v))
    if length == 0.0:
        return 1.0, (0.0, 0.0, 0.0)
    v = v / length
    z = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(z, v), -1.0, 1.0))
    if dot > 1.0 - 1e-9:
        return 1.0, (0.0, 0.0, 0.0)
    if dot < -1.0 + 1e-9:
        return 0.0, (1.0, 0.0, 0.0)      # 180 deg about X
    axis = np.cross(z, v)
    axis = axis / float(np.linalg.norm(axis))
    angle = math.acos(dot)
    s = math.sin(angle / 2.0)
    return math.cos(angle / 2.0), (float(axis[0] * s),
                                   float(axis[1] * s),
                                   float(axis[2] * s))


def _material_for(stage, scope_path: str, symbol: str, cache: Dict[str, object]):
    """One UsdPreviewSurface per element, shared by every atom of it.

    A material per atom would be hundreds of shader prims saying the same
    thing, and a viewer would have to resolve each separately.
    """
    from pxr import Sdf, UsdShade

    if symbol in cache:
        return cache[symbol]

    safe = symbol if symbol.isalnum() else "X"
    path = f"{scope_path}/{safe}"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    r, g, b = _color_for(symbol)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((r, g, b))
    # Slightly glossy, which is what makes a sphere read as a sphere under a
    # single light; fully rough spheres flatten into discs.
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface")
    cache[symbol] = material
    return material


def _bind(prim, material) -> None:
    from pxr import UsdShade
    UsdShade.MaterialBindingAPI(prim).Bind(material)


def _safe_name(name: str) -> str:
    """A USD prim name: alphanumerics and underscore, not leading a digit."""
    out = "".join(c if c.isalnum() else "_" for c in name)
    return out if out and not out[0].isdigit() else f"_{out}"


def author_molecule(
    stage,
    parent_path: str,
    name: str,
    smiles: str,
    *,
    formula: str = "",
    molecular_weight: Optional[float] = None,
    with_hydrogens: bool = True,
    material_scope: str = "/Materials",
) -> MoleculeGeometry:
    """Write one molecule under ``parent_path`` and report what it cost."""
    from pxr import Gf, Sdf, UsdGeom

    from cognisom.library.structures import CONFORMER_SEED, embed_conformer

    mol, embedded = embed_conformer(smiles, with_hydrogens)
    if mol is None:
        raise ValueError(f"{name}: SMILES does not parse: {smiles!r}")

    prim_path = f"{parent_path}/{_safe_name(name)}"
    root = UsdGeom.Xform.Define(stage, prim_path)
    conf = mol.GetConformer() if embedded else None

    def pos(i):
        if conf is None:
            return (0.0, 0.0, 0.0)
        p = conf.GetAtomPosition(i)
        return (float(p.x), float(p.y), float(p.z))

    materials: Dict[str, object] = {}
    elements: Dict[str, int] = {}
    extent_max = 0.0

    atoms_scope = UsdGeom.Scope.Define(stage, f"{prim_path}/Atoms")
    for i, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        elements[symbol] = elements.get(symbol, 0) + 1
        x, y, z = pos(i)
        extent_max = max(extent_max, abs(x), abs(y), abs(z))
        radius = _element_radius(symbol) * BALL_SCALE

        sphere = UsdGeom.Sphere.Define(
            stage, f"{prim_path}/Atoms/{symbol}_{i}")
        sphere.CreateRadiusAttr(radius)
        sphere.CreateExtentAttr([(-radius, -radius, -radius),
                                 (radius, radius, radius)])
        UsdGeom.Xformable(sphere.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(x, y, z))
        _bind(sphere.GetPrim(),
              _material_for(stage, material_scope, symbol, materials))

        prim = sphere.GetPrim()
        prim.SetCustomDataByKey("element", symbol)
        prim.SetCustomDataByKey("atomIndex", i)
        prim.SetCustomDataByKey("formalCharge", atom.GetFormalCharge())
        prim.SetCustomDataByKey("aromatic", bool(atom.GetIsAromatic()))

    bonds_scope = UsdGeom.Scope.Define(stage, f"{prim_path}/Bonds")
    for b, bond in enumerate(mol.GetBonds()):
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        pi, pj = pos(i), pos(j)
        mid = tuple((a + c) / 2.0 for a, c in zip(pi, pj))
        vec = tuple(c - a for a, c in zip(pi, pj))
        length = math.sqrt(sum(c * c for c in vec))
        w, (qx, qy, qz) = _quat_z_to(vec)

        # Two half-length cylinders, each from an atom to the midpoint, so
        # each half takes that atom's colour. A single grey stick loses
        # which element is which end, which is most of what a structure is
        # for.
        for half, (start, end, atom_idx) in enumerate((
                (pi, mid, i), (mid, pj, j))):
            symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
            centre = tuple((a + c) / 2.0 for a, c in zip(start, end))
            height = max(length / 2.0, 1e-6)

            cyl = UsdGeom.Cylinder.Define(
                stage, f"{prim_path}/Bonds/bond_{b}_{half}")
            cyl.CreateRadiusAttr(STICK_RADIUS)
            cyl.CreateHeightAttr(height)
            cyl.CreateAxisAttr(UsdGeom.Tokens.z)
            cyl.CreateExtentAttr([
                (-STICK_RADIUS, -STICK_RADIUS, -height / 2.0),
                (STICK_RADIUS, STICK_RADIUS, height / 2.0)])
            xf = UsdGeom.Xformable(cyl.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(*centre))
            xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(qx, qy, qz)))
            _bind(cyl.GetPrim(),
                  _material_for(stage, material_scope, symbol, materials))

            prim = cyl.GetPrim()
            prim.SetCustomDataByKey("bondIndex", b)
            prim.SetCustomDataByKey("bondOrder", str(bond.GetBondType()))
            prim.SetCustomDataByKey("aromatic", bool(bond.GetIsAromatic()))

    n_atoms, n_bonds = mol.GetNumAtoms(), mol.GetNumBonds()
    geom = MoleculeGeometry(
        name=name, smiles=smiles, formula=formula,
        atoms=n_atoms, bonds=n_bonds, prims=n_atoms + 2 * n_bonds,
        radius_a=extent_max, elements=elements, prim_path=prim_path,
        embedded=embedded,
    )

    prim = root.GetPrim()
    prim.SetCustomDataByKey("smiles", smiles)
    prim.SetCustomDataByKey("formula", formula)
    if molecular_weight is not None:
        prim.SetCustomDataByKey("molecularWeight", float(molecular_weight))
    prim.SetCustomDataByKey("atoms", n_atoms)
    prim.SetCustomDataByKey("bonds", n_bonds)
    prim.SetCustomDataByKey("prims", geom.prims)
    prim.SetCustomDataByKey("elements", dict(sorted(elements.items())))
    # Provenance: the topology is the library's and checked; the
    # coordinates are computed here and are one conformer, not a structure.
    prim.SetCustomDataByKey("topologySource", "cognisom entity library (SMILES, formula-checked)")
    prim.SetCustomDataByKey("coordinateSource", "computed: rdkit ETKDG + MMFF94")
    prim.SetCustomDataByKey("coordinateSeed", int(CONFORMER_SEED))
    prim.SetCustomDataByKey("coordinatesAreExperimental", False)
    prim.SetCustomDataByKey("conformerEmbedded", embedded)
    prim.SetCustomDataByKey("ballScale", BALL_SCALE)
    prim.SetCustomDataByKey("stickRadiusAngstrom", STICK_RADIUS)
    prim.SetCustomDataByKey("radiiSource", "rdkit periodic table, van der Waals")
    prim.SetCustomDataByKey("colorScheme", "Jmol/CPK")
    if not embedded:
        prim.SetCustomDataByKey(
            "warning",
            "distance geometry could not place these atoms; every atom is "
            "at the origin and only the topology is meaningful")
    return geom


def library_molecules(store=None) -> List[Tuple[str, str, str, Optional[float]]]:
    """Every library entity carrying a SMILES, as (name, smiles, formula, mw).

    Sorted by name so an export is stable between runs.
    """
    import json

    if store is None:
        import contextlib
        import io

        from cognisom.library.store import EntityStore
        with contextlib.redirect_stdout(io.StringIO()):
            store = EntityStore()

    rows = store._conn.execute("SELECT name, data_json FROM entities").fetchall()
    out = []
    for name, blob in rows:
        props = (json.loads(blob or "{}").get("properties") or {})
        smiles = props.get("smiles")
        if not smiles:
            continue
        mw = props.get("molecular_weight")
        out.append((name, smiles, props.get("molecular_formula") or "",
                    float(mw) if mw is not None else None))
    return sorted(out)


def export_library(
    path: str,
    *,
    store=None,
    names: Optional[Sequence[str]] = None,
    with_hydrogens: bool = True,
    gap_a: float = 4.0,
) -> List[MoleculeGeometry]:
    """Write every structured library entity to one USD stage.

    Molecules are laid out on a grid in XY, each cell wide enough for the
    molecule in it, so the whole set is legible at once without prims from
    one overlapping the next. ``gap_a`` is the clear space between them in
    angstroms.
    """
    from pxr import Usd, UsdGeom

    entries = library_molecules(store)
    if names is not None:
        wanted = {n.lower() for n in names}
        entries = [e for e in entries if e[0].lower() in wanted]
    if not entries:
        raise ValueError("no library entities carry a SMILES to export")

    stage = Usd.Stage.CreateNew(path) if not Usd.Stage.IsSupportedFile(path) \
        else Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, METERS_PER_UNIT)

    UsdGeom.Scope.Define(stage, "/Materials")
    root = UsdGeom.Xform.Define(stage, "/Molecules")
    stage.SetDefaultPrim(root.GetPrim())

    geoms: List[MoleculeGeometry] = []
    for name, smiles, formula, mw in entries:
        geoms.append(author_molecule(
            stage, "/Molecules", name, smiles,
            formula=formula, molecular_weight=mw,
            with_hydrogens=with_hydrogens))

    _lay_out(stage, geoms, gap_a)

    total = sum(g.prims for g in geoms)
    root.GetPrim().SetCustomDataByKey("moleculeCount", len(geoms))
    root.GetPrim().SetCustomDataByKey("primCount", total)
    root.GetPrim().SetCustomDataByKey(
        "note",
        "Topology from the entity library and formula-checked; coordinates "
        "computed (ETKDG + MMFF94), one conformer each, not experimental "
        "structures. Authored in angstroms.")
    stage.GetRootLayer().Save()
    log.info("wrote %d molecules, %d prims to %s", len(geoms), total, path)
    return geoms


def _lay_out(stage, geoms: Sequence[MoleculeGeometry], gap_a: float) -> None:
    """Place each molecule on a grid sized by the largest of them.

    A uniform cell keeps the grid readable as a grid; sizing it to the
    largest molecule is what stops docetaxel from reaching into its
    neighbour's cell.
    """
    from pxr import Gf, UsdGeom

    if not geoms:
        return
    cell = 2.0 * max(g.radius_a for g in geoms) + gap_a
    columns = max(1, int(math.ceil(math.sqrt(len(geoms)))))
    for index, geom in enumerate(geoms):
        row, col = divmod(index, columns)
        prim = stage.GetPrimAtPath(geom.prim_path)
        if not prim:
            continue
        # Rows run -Y so the grid reads top-left to bottom-right in a
        # Z-up viewport looking down.
        UsdGeom.Xformable(prim).AddTranslateOp().Set(
            Gf.Vec3d(col * cell, -row * cell, 0.0))
    stage.GetPrimAtPath("/Molecules").SetCustomDataByKey("gridCellAngstrom", cell)
    stage.GetPrimAtPath("/Molecules").SetCustomDataByKey("gridColumns", columns)
