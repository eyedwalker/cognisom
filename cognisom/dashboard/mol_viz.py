"""
Molecular Visualization Utilities
==================================

Renders 2D molecule diagrams (RDKit) and 3D protein structures (py3Dmol)
for the Cognisom dashboard.
"""

import io
import base64
from typing import List, Optional, Dict, Tuple

import numpy as np


# ── 2D Molecule Rendering ────────────────────────────────────────────

def smiles_to_image(smiles: str, size: Tuple[int, int] = (400, 300),
                    highlight_scaffold: Optional[str] = None) -> Optional[bytes]:
    """Render a SMILES string to a PNG image.

    Args:
        smiles: SMILES string.
        size: (width, height) in pixels.
        highlight_scaffold: Optional SMILES of substructure to highlight.

    Returns:
        PNG image bytes, or None if parsing fails.
    """
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    AllChem.Compute2DCoords(mol)

    highlight_atoms = []
    if highlight_scaffold:
        scaffold = Chem.MolFromSmiles(highlight_scaffold)
        if scaffold:
            match = mol.GetSubstructMatch(scaffold)
            highlight_atoms = list(match)

    drawer = Draw.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.addAtomIndices = False
    opts.bondLineWidth = 2.0

    if highlight_atoms:
        from rdkit.Chem import rdMolDraw2D
        colors = {a: (0.3, 0.7, 1.0, 0.4) for a in highlight_atoms}
        drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms,
                            highlightAtomColors=colors)
    else:
        drawer.DrawMolecule(mol)

    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def smiles_to_svg(smiles: str, size: Tuple[int, int] = (400, 300)) -> Optional[str]:
    """Render SMILES to SVG string."""
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    AllChem.Compute2DCoords(mol)
    drawer = Draw.MolDraw2DSVG(size[0], size[1])
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2.0
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def molecule_grid(smiles_list: List[str], labels: Optional[List[str]] = None,
                  mols_per_row: int = 4, sub_img_size: Tuple[int, int] = (350, 280)) -> Optional[bytes]:
    """Render a grid of molecules as a single PNG image.

    Args:
        smiles_list: List of SMILES strings.
        labels: Optional labels for each molecule.
        mols_per_row: Number of molecules per row.
        sub_img_size: Size of each sub-image.

    Returns:
        PNG image bytes.
    """
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem

    mols = []
    valid_labels = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            AllChem.Compute2DCoords(mol)
            mols.append(mol)
            if labels:
                valid_labels.append(labels[i] if i < len(labels) else "")
            else:
                valid_labels.append(f"Mol {i+1}")

    if not mols:
        return None

    img = Draw.MolsToGridImage(
        mols, molsPerRow=mols_per_row, subImgSize=sub_img_size,
        legends=valid_labels,
    )

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def compute_properties(smiles: str) -> Optional[Dict]:
    """Compute molecular properties from SMILES.

    Returns dict with MW, LogP, HBD, HBA, TPSA, RotBonds, QED, etc.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return {
        "MW": round(Descriptors.MolWt(mol), 1),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "TPSA": round(Descriptors.TPSA(mol), 1),
        "RotBonds": Descriptors.NumRotatableBonds(mol),
        "Rings": Descriptors.RingCount(mol),
        "Atoms": mol.GetNumHeavyAtoms(),
        "QED": round(QED.qed(mol), 3),
        "Formula": rdMolDescriptors.CalcMolFormula(mol),
    }


def lipinski_check(props: Dict) -> Dict:
    """Check Lipinski's Rule of Five."""
    violations = 0
    checks = {
        "MW <= 500": props["MW"] <= 500,
        "LogP <= 5": props["LogP"] <= 5,
        "HBD <= 5": props["HBD"] <= 5,
        "HBA <= 10": props["HBA"] <= 10,
    }
    violations = sum(1 for v in checks.values() if not v)
    return {"checks": checks, "violations": violations, "pass": violations <= 1}


# ── 3D Protein Visualization ─────────────────────────────────────────

def protein_viewer_html(pdb_data: str, width: int = 800, height: int = 500,
                        style: str = "cartoon", color: str = "spectrum",
                        surface: bool = False, spin: bool = False,
                        highlight_residues: Optional[List[int]] = None) -> str:
    """Generate HTML for an interactive 3D protein viewer.

    Args:
        pdb_data: PDB format string.
        width: Viewer width in pixels.
        height: Viewer height in pixels.
        style: Visualization style (cartoon, stick, sphere, line, surface).
        color: Color scheme (spectrum, chain, ssType, element).
        surface: Whether to add transparent surface.
        highlight_residues: Residue numbers to highlight.

    Returns:
        HTML string with embedded py3Dmol viewer.
    """
    # Escape PDB data for JavaScript
    pdb_escaped = pdb_data.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")

    style_js = _build_style_js(style, color)

    highlight_js = ""
    if highlight_residues:
        resi_str = ",".join(str(r) for r in highlight_residues)
        highlight_js = f"""
        viewer.addStyle({{resi: [{resi_str}]}},
                        {{stick: {{radius: 0.3, color: 'red'}}}});
        """

    surface_js = ""
    if surface:
        surface_js = """
        viewer.addSurface(
            $3Dmol.SurfaceType.VDW,
            {opacity: 0.15, color: 'white'},
            {}, {}
        );
        """

    spin_js = "viewer.spin(true);" if spin else ""

    html = f"""
    <div id="mol-viewer" style="width:{width}px; height:{height}px; position:relative;
         border: 1px solid #333; border-radius: 8px; overflow: hidden;
         background: #1a1a2e;">
    </div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
    (function() {{
        var viewer = $3Dmol.createViewer("mol-viewer", {{
            backgroundColor: "#1a1a2e"
        }});
        var pdbData = `{pdb_escaped}`;
        viewer.addModel(pdbData, "pdb");
        {style_js}
        {highlight_js}
        {surface_js}
        viewer.zoomTo();
        {spin_js}
        viewer.render();
    }})();
    </script>
    """
    return html


def _build_style_js(style: str, color: str) -> str:
    """Build the py3Dmol style JavaScript."""
    if color == "spectrum":
        color_js = "spectrum"
    elif color == "chain":
        color_js = "chain"
    elif color == "ssType":
        color_js = "ssType"
    else:
        color_js = f"'{color}'"

    if style == "cartoon":
        return f'viewer.setStyle({{}}, {{cartoon: {{color: "{color_js}"}}}});'
    elif style == "stick":
        return f'viewer.setStyle({{}}, {{stick: {{color: "{color_js}"}}}});'
    elif style == "sphere":
        return f'viewer.setStyle({{}}, {{sphere: {{color: "{color_js}"}}}});'
    elif style == "line":
        return f'viewer.setStyle({{}}, {{line: {{color: "{color_js}"}}}});'
    elif style == "cartoon+stick":
        return (f'viewer.setStyle({{}}, {{cartoon: {{color: "{color_js}"}}}});'
                f'viewer.addStyle({{}}, {{stick: {{radius: 0.15}}}});')
    else:
        return f'viewer.setStyle({{}}, {{cartoon: {{color: "{color_js}"}}}});'


def dual_viewer_html(pdb_data: str, ligand_sdf: Optional[str] = None,
                     width: int = 800, height: int = 500) -> str:
    """Protein + ligand viewer for docking results."""
    pdb_escaped = pdb_data.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")

    ligand_js = ""
    if ligand_sdf:
        lig_escaped = ligand_sdf.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
        ligand_js = f"""
        viewer.addModel(`{lig_escaped}`, "sdf");
        viewer.setStyle({{model: 1}}, {{stick: {{colorscheme: "greenCarbon", radius: 0.25}}}});
        viewer.addSurface(
            $3Dmol.SurfaceType.VDW,
            {{opacity: 0.5, color: "lime"}},
            {{model: 1}}, {{}}
        );
        """

    html = f"""
    <div id="dock-viewer" style="width:{width}px; height:{height}px; position:relative;
         border: 1px solid #333; border-radius: 8px; overflow: hidden;
         background: #1a1a2e;">
    </div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
    (function() {{
        var viewer = $3Dmol.createViewer("dock-viewer", {{
            backgroundColor: "#1a1a2e"
        }});
        viewer.addModel(`{pdb_escaped}`, "pdb");
        viewer.setStyle({{model: 0}}, {{cartoon: {{color: "spectrum", opacity: 0.8}}}});
        viewer.addSurface(
            $3Dmol.SurfaceType.VDW,
            {{opacity: 0.1, color: "white"}},
            {{model: 0}}, {{}}
        );
        {ligand_js}
        viewer.zoomTo();
        viewer.render();
    }})();
    </script>
    """
    return html


def fetch_pdb(pdb_id: str) -> Optional[str]:
    """Fetch a PDB structure from RCSB.

    Args:
        pdb_id: 4-character PDB ID (e.g., '2AM9').

    Returns:
        PDB format string, or None if fetch fails.
    """
    import requests
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            return r.text
        return None
    except Exception:
        return None


def smiles_to_3d_sdf(smiles: str) -> Optional[str]:
    """Convert SMILES to 3D SDF for visualization."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if result != 0:
        # Fallback
        AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)

    writer = Chem.SDWriter(io.StringIO())
    sdf_io = io.StringIO()
    writer = Chem.SDWriter(sdf_io)
    writer.write(mol)
    writer.close()
    return sdf_io.getvalue()


def pdb_stats(pdb_data: str) -> Dict:
    """Extract basic stats from PDB data."""
    chains = set()
    residues = set()
    atoms = 0
    for line in pdb_data.splitlines():
        if line.startswith("ATOM"):
            atoms += 1
            chain = line[21]
            chains.add(chain)
            resi = line[22:26].strip()
            residues.add((chain, resi))
    return {
        "chains": len(chains),
        "residues": len(residues),
        "atoms": atoms,
        "chain_ids": sorted(chains),
    }
